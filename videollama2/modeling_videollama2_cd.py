from functools import partial
from typing import Optional, List, Tuple, Union
import os, cv2
import torch
import imageio
import numpy as np
from PIL import Image
from decord.ndarray import NDArray
import torch
from torch import nn
import einops
import copy

from videollama2.constants import NUM_FRAMES, DEFAULT_IMAGE_TOKEN, DEFAULT_VIDEO_TOKEN, MODAL_INDEX_MAP
from videollama2.mm_utils import process_image, get_model_name_from_path, tokenizer_multimodal_token, KeywordsStoppingCriteria

import transformers

from transformers import AutoTokenizer, AutoConfig, MistralModel
from transformers.generation.utils import GenerateOutput, GenerateNonBeamOutput, GenerateEncoderDecoderOutput, GenerateDecoderOnlyOutput, GenerationConfig, CausalLMOutputWithPast
from transformers.generation.logits_process import LogitsProcessorList
from transformers.generation.stopping_criteria import StoppingCriteriaList


from videollama2.model.videollama2_mistral import Videollama2MistralForCausalLM

from videollama2.mm_utils import expand2square, frame_sample, SafeVideoReader
from videollama2.constants import MAX_FRAMES, MODAL_INDEX_MAP, IGNORE_INDEX

def sample(
        self,
        input_ids: torch.LongTensor,
        logits_processor: LogitsProcessorList,
        stopping_criteria: StoppingCriteriaList,
        generation_config: GenerationConfig,
        synced_gpus: bool,
        streamer: Optional["BaseStreamer"],
        **model_kwargs,
    ) -> Union[GenerateNonBeamOutput, torch.LongTensor]:
        print("#################### tcd sample ####################")
        # init values
        pad_token_id = generation_config._pad_token_tensor
        output_attentions = generation_config.output_attentions
        output_hidden_states = generation_config.output_hidden_states
        output_scores = generation_config.output_scores
        output_logits = generation_config.output_logits
        return_dict_in_generate = generation_config.return_dict_in_generate
        max_length = generation_config.max_length
        has_eos_stopping_criteria = any(hasattr(criteria, "eos_token_id") for criteria in stopping_criteria)
        do_sample = generation_config.do_sample

        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        raw_logits = () if (return_dict_in_generate and output_logits) else None
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        cross_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
        if return_dict_in_generate and self.config.is_encoder_decoder:
            encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
            encoder_hidden_states = (
                model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
            )

        # keep track of which sequences are already finished
        batch_size, cur_len = input_ids.shape
        this_peer_finished = False
        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)

        # NOTE: ##### Contrastive Decoding #####
        use_cd = model_kwargs.get("input_ids_cd") != None or model_kwargs.get("inputs_embeds_cd") != None
        use_embed = model_kwargs.get("inputs_embeds_cd") != None
        if use_cd:
            model_kwargs_cd = copy.deepcopy(model_kwargs)

            for k in model_kwargs_cd.keys():
                # NOTE: input_ids_cd will not be assigned to input_ids
                if k + '_cd' in model_kwargs_cd.keys():
                    model_kwargs_cd[k] = model_kwargs_cd[k + '_cd']
                    
            input_ids_cd = model_kwargs_cd.get("input_ids_cd") if model_kwargs_cd.get("input_ids_cd") != None else copy.deepcopy(input_ids)
            model_kwargs_cd = self._get_initial_cache_position(input_ids_cd, model_kwargs_cd)

        model_kwargs = self._get_initial_cache_position(input_ids, model_kwargs)

        while self._has_unfinished_sequences(
            this_peer_finished, synced_gpus, device=input_ids.device, cur_len=cur_len, max_length=max_length
        ):
            # prepare model inputs
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

            # prepare variable output controls (note: some models won't accept all output controls)
            model_inputs.update({"output_attentions": output_attentions} if output_attentions else {})
            model_inputs.update({"output_hidden_states": output_hidden_states} if output_hidden_states else {})

            # forward pass to get next token
            outputs = self(**model_inputs, return_dict=True)

            if synced_gpus and this_peer_finished:
                continue  # don't waste resources running the code we don't need

            # Clone is needed to avoid keeping a hanging ref to outputs.logits which may be very large for first iteration
            # (the clone itself is always small)
            next_token_logits = outputs.logits.clone()[:, -1, :].float()

            # NOTE: Contrastive Decoding
            if use_cd:
                # prepare model cd inputs
                model_inputs_cd = self.prepare_inputs_for_generation(input_ids if use_embed else model_kwargs.get("input_ids_cd"), **model_kwargs_cd)

                # prepare variable output controls (note: some models won't accept all output controls)
                model_inputs_cd.update({"output_attentions": output_attentions} if output_attentions else {})
                model_inputs_cd.update({"output_hidden_states": output_hidden_states} if output_hidden_states else {})

                # forward pass to get next token
                outputs_cd = self(**model_inputs_cd, return_dict=True)

                next_token_logits_cd = outputs_cd.logits.clone()[:, -1, :].float()

                # pre-process logits from contrastive inputs
                cd_alpha = model_kwargs.get("cd_alpha")
                cd_beta = model_kwargs.get("cd_beta")
                print("cd_alpha: ", cd_alpha, "; cd_beta: ", cd_beta)
                
                # version 1  set cutoff for Adaptive Plausibility Constraints
                # probs = nn.functional.softmax(next_token_logits, dim=-1)
                # cutoff = cd_beta * probs.max(dim=-1, keepdim=True).values

                # version 2 set cutoff for Adaptive Plausibility Constraints
                cutoff = torch.log(torch.tensor(cd_beta)) + next_token_logits.max(dim=-1, keepdim=True).values
                
                diffs = (1 + cd_alpha) * next_token_logits - cd_alpha * next_token_logits_cd
                cd_logits = diffs.masked_fill(next_token_logits < cutoff, -float("inf"))

                # pre-process distribution
                next_token_scores = logits_processor(input_ids, cd_logits)

            else:
                # pre-process distribution
                next_token_scores = logits_processor(input_ids, next_token_logits)

            # Store scores, attentions and hidden_states when required
            if return_dict_in_generate:
                if output_scores:
                    scores += (next_token_scores,)
                if output_logits:
                    raw_logits += (next_token_logits,)
                if output_attentions:
                    decoder_attentions += (
                        (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                    )
                    if self.config.is_encoder_decoder:
                        cross_attentions += (outputs.cross_attentions,)

                if output_hidden_states:
                    decoder_hidden_states += (
                        (outputs.decoder_hidden_states,)
                        if self.config.is_encoder_decoder
                        else (outputs.hidden_states,)
                    )

            # token selection
            if do_sample:
                probs = nn.functional.softmax(next_token_scores, dim=-1)
                # TODO (joao): this OP throws "skipping cudagraphs due to ['incompatible ops']", find solution
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                next_tokens = torch.argmax(next_token_scores, dim=-1)

            # finished sentences should have their next token be a padding token
            if has_eos_stopping_criteria:
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

            # update generated ids, model inputs, and length for next step
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            if streamer is not None:
                streamer.put(next_tokens.cpu())
            model_kwargs = self._update_model_kwargs_for_generation(
                outputs,
                model_kwargs,
                is_encoder_decoder=self.config.is_encoder_decoder,
            )

            # NOTE: ##### Contrastive Decoding #####
            if use_cd:
                input_ids_cd = torch.cat([input_ids_cd, next_tokens[:, None]], dim=-1)
                model_kwargs_cd = self._update_model_kwargs_for_generation(
                    outputs_cd,
                    model_kwargs_cd,
                    is_encoder_decoder=self.config.is_encoder_decoder,
                )

                del outputs_cd

            unfinished_sequences = unfinished_sequences & ~stopping_criteria(input_ids, scores)
            this_peer_finished = unfinished_sequences.max() == 0
            cur_len += 1

            # This is needed to properly delete outputs.logits which may be very large for first iteration
            # Otherwise a reference to outputs is kept which keeps the logits alive in the next iteration
            del outputs

        if streamer is not None:
            streamer.end()

        if return_dict_in_generate:
            if self.config.is_encoder_decoder:
                return GenerateEncoderDecoderOutput(
                    sequences=input_ids,
                    scores=scores,
                    logits=raw_logits,
                    encoder_attentions=encoder_attentions,
                    encoder_hidden_states=encoder_hidden_states,
                    decoder_attentions=decoder_attentions,
                    cross_attentions=cross_attentions,
                    decoder_hidden_states=decoder_hidden_states,
                    past_key_values=model_kwargs.get("past_key_values"),
                )
            else:
                return GenerateDecoderOnlyOutput(
                    sequences=input_ids,
                    scores=scores,
                    logits=raw_logits,
                    attentions=decoder_attentions,
                    hidden_states=decoder_hidden_states,
                    past_key_values=model_kwargs.get("past_key_values"),
                )
        else:
            return input_ids

def evolve_tcd_sampling():
    transformers.generation.utils.GenerationMixin._sample = sample
    transformers.generation.utils.GenerationMixin.sample = sample


class Videollama2MistralForCausalLM_cd(Videollama2MistralForCausalLM):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,

        input_ids_cd: torch.LongTensor = None,
        attention_mask_cd: Optional[torch.Tensor] = None,
        position_ids_cd: Optional[torch.LongTensor] = None,
        past_key_values_cd: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds_cd: Optional[torch.FloatTensor] = None,
        labels_cd: Optional[torch.LongTensor] = None,
        images_cd: Optional[torch.FloatTensor] = None,
        cd_beta: Optional[torch.FloatTensor] = None,
        cd_alpha: Optional[torch.FloatTensor] = None,
        **kwargs
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            images=images,
            return_dict=return_dict,
            **kwargs
        )

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if images is not None:
            (
                input_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                _
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids=inputs,
                attention_mask=attention_mask,
                past_key_values=None,
                labels=None,
                images=images
            )
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)

        # NOTE: ##### Contrastive Decoding #####
        inputs_cd = kwargs.pop("inputs_cd", None)
        use_cd = inputs_cd is not None
        if use_cd:
            position_ids_cd = kwargs.pop("position_ids_cd", None)
            attention_mask_cd = kwargs.pop("attention_mask_cd", None)
            if "inputs_embeds_cd" in kwargs:
                raise NotImplementedError("`inputs_embeds_cd` is not supported")
            
            images_cd = kwargs.pop("images_cd", None)
            if images_cd is not None:
                (
                    input_ids_cd,
                    attention_mask_cd,
                    past_key_values_cd,
                    inputs_embeds_cd,
                    _
                ) = self.prepare_inputs_labels_for_multimodal(
                    input_ids=inputs_cd,
                    attention_mask=attention_mask_cd,
                    past_key_values=None,
                    labels=None,
                    images=images_cd
                )
            else:
                inputs_embeds_cd = self.get_model().embed_tokens(inputs_cd)

            return MistralModel.generate(
                self,
                position_ids=position_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                position_ids_cd=position_ids_cd,
                attention_mask_cd=attention_mask_cd,
                inputs_embeds_cd=inputs_embeds_cd,
                **kwargs
            )
        else:
            return MistralModel.generate(
                self,
                position_ids=position_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                **kwargs
            )
    
    @torch.no_grad()
    def generate_dino_heal(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        attn_weights: Optional[torch.Tensor] = None,
        dino_norm: bool = False,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if images is not None:
            (
                input_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                _
            ) = self.prepare_inputs_labels_for_multimodal_dino_heal(
                input_ids=inputs,
                attention_mask=attention_mask,
                past_key_values=None,
                labels=None,
                images=images,
                attn_weights=attn_weights,
                dino_norm=dino_norm,
            )
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)

        return MistralModel.generate(
            self,
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs
        )

    def prepare_inputs_labels_for_multimodal_dino_heal(
        self, input_ids, attention_mask, past_key_values, labels, images, attn_weights, dino_norm
    ):
        vision_tower = self.get_vision_tower()
        # NOTE: text-only situation
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            # if past_key_values is not None and vision_tower is not None and Xs is not None and input_ids.shape[1] == 1:
            #    attention_mask = torch.ones((attention_mask.shape[0], past_key_values[-1][-1].shape[-2] + 1), dtype=attention_mask.dtype, device=attention_mask.device)
            return input_ids, attention_mask, past_key_values, None, labels
        
        ## DINO_HEAL: add attn_weights into visual features
        mm_features = self.encode_images_or_videos_dino_heal(images, attn_weights, dino_norm)

        new_input_embeds = []
        new_labels = [] if labels is not None else None
        cur_mm_idx = 0
        # replace image/video/audio tokens with pre-computed embeddings
        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_multimodals = sum((cur_input_ids == mm_token_idx).sum() for mm_token_idx in MODAL_INDEX_MAP.values())
            # pure text input
            if num_multimodals == 0:
                half_len = cur_input_ids.shape[0] // 2
                cur_mm_features = mm_features[cur_mm_idx]
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids[:half_len])
                cur_input_embeds_2 = self.get_model().embed_tokens(cur_input_ids[half_len:])
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_mm_features[0:0], cur_input_embeds_2], dim=0)
                new_input_embeds.append(cur_input_embeds)
                if labels is not None:
                    new_labels.append(labels[batch_idx])
                cur_mm_idx += 1 
                continue

            cur_new_input_embeds = []
            if labels is not None:
                cur_labels = labels[batch_idx]
                cur_new_labels = []
                assert cur_labels.shape == cur_input_ids.shape

            mm_token_indices = torch.where(sum([cur_input_ids == mm_token_idx for mm_token_idx in MODAL_INDEX_MAP.values()]))[0]
            while mm_token_indices.numel() > 0:
                cur_mm_features = mm_features[cur_mm_idx]
                mm_token_start = mm_token_indices[0]

                cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids[:mm_token_start])) 
                cur_new_input_embeds.append(cur_mm_features)
                if labels is not None:
                    cur_new_labels.append(cur_labels[:mm_token_start])
                    cur_new_labels.append(torch.full((cur_mm_features.shape[0],), IGNORE_INDEX, device=labels.device, dtype=labels.dtype))
                    cur_labels = cur_labels[mm_token_start+1:]

                cur_mm_idx += 1
                cur_input_ids = cur_input_ids[mm_token_start+1:] 
                mm_token_indices = torch.where(sum([cur_input_ids == mm_token_idx for mm_token_idx in MODAL_INDEX_MAP.values()]))[0]

            if cur_input_ids.numel() > 0:
                cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids))
                if labels is not None:
                    cur_new_labels.append(cur_labels)
            cur_new_input_embeds = [x.to(device=self.device) for x in cur_new_input_embeds]
            # NOTE: one cur_new_input_embeds per each  
            cur_new_input_embeds = torch.cat(cur_new_input_embeds, dim=0)
            new_input_embeds.append(cur_new_input_embeds)
            if labels is not None:
                cur_new_labels = torch.cat(cur_new_labels, dim=0)
                new_labels.append(cur_new_labels)

        # padding
        if any(x.shape != new_input_embeds[0].shape for x in new_input_embeds):
            max_len = max(x.shape[0] for x in new_input_embeds)

            new_input_embeds_align = []
            for cur_new_embed in new_input_embeds:
                cur_new_embed = torch.cat((cur_new_embed, torch.zeros((max_len - cur_new_embed.shape[0], cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)), dim=0)
                new_input_embeds_align.append(cur_new_embed)
            new_input_embeds = torch.stack(new_input_embeds_align, dim=0)

            if labels is not None:
                new_labels_align = []
                _new_labels = new_labels
                for cur_new_label in new_labels:
                    cur_new_label = torch.cat((cur_new_label, torch.full((max_len - cur_new_label.shape[0],), IGNORE_INDEX, dtype=cur_new_label.dtype, device=cur_new_label.device)), dim=0)
                    new_labels_align.append(cur_new_label)
                new_labels = torch.stack(new_labels_align, dim=0)

            if attention_mask is not None:
                new_attention_mask = []
                for cur_attention_mask, cur_new_labels, cur_new_labels_align in zip(attention_mask, _new_labels, new_labels):
                    new_attn_mask_pad_left = torch.full((cur_new_labels.shape[0] - labels.shape[1],), True, dtype=attention_mask.dtype, device=attention_mask.device)
                    new_attn_mask_pad_right = torch.full((cur_new_labels_align.shape[0] - cur_new_labels.shape[0],), False, dtype=attention_mask.dtype, device=attention_mask.device)
                    cur_new_attention_mask = torch.cat((new_attn_mask_pad_left, cur_attention_mask, new_attn_mask_pad_right), dim=0)
                    new_attention_mask.append(cur_new_attention_mask)
                attention_mask = torch.stack(new_attention_mask, dim=0)
                assert attention_mask.shape == new_labels.shape
        else:
            new_input_embeds = torch.stack(new_input_embeds, dim=0)
            if labels is not None:
                new_labels  = torch.stack(new_labels, dim=0)

            if attention_mask is not None:
                new_attn_mask_pad_left = torch.full((attention_mask.shape[0], new_input_embeds.shape[1] - input_ids.shape[1]), True, dtype=attention_mask.dtype, device=attention_mask.device)
                attention_mask = torch.cat((new_attn_mask_pad_left, attention_mask), dim=1)
                assert attention_mask.shape == new_input_embeds.shape[:2]

        return None, attention_mask, past_key_values, new_input_embeds, new_labels

    
    def encode_images_or_videos_dino_heal(self, images, attn_weights, dino_norm):
        num_frames = self.config.num_frames if hasattr(self.config, 'num_frames') else NUM_FRAMES

        data_batch = []
        for i, (data, modal) in enumerate(images):
            if modal == 'image':
                data = data.expand(num_frames, -1, -1, -1)
            else:
                data = data
            data_batch.append(data)

        data_batch = torch.stack(data_batch, dim=0)

        assert len(data_batch.size()) == 5
        batch_size = data_batch.size(0)

        frames = einops.rearrange(data_batch, 'b t c h w -> (b t) c h w')
        frames_features = self.get_model().get_vision_tower()(frames)
        frames_features = einops.rearrange(frames_features, '(b t) n h -> b t n h', b = batch_size)

        ## DINO_HEAL: add attn_weights into visual features
        attn_weights = attn_weights[list(attn_weights.keys())[-1]].to(frames_features.device)
        if dino_norm:
            attn_weights_flatten = attn_weights.reshape(attn_weights.size(0), attn_weights.size(1), -1)
            attn_weights_norm = torch.sigmoid(attn_weights_flatten).unsqueeze(-1)
        else:
            attn_weights_norm = attn_weights
        frames_features = frames_features * attn_weights_norm

        return self.temporal_aggregator(frames_features)


def load_pretrained_model(model_path, device_map="auto", device="cuda", use_flash_attn=False, **kwargs):
    if 'token' in kwargs:
        token = kwargs['token']
    else:
        token = None
    
    kwargs = {"device_map": device_map, **kwargs}

    if device != "cuda":
        kwargs['device_map'] = {"": device}

    kwargs['torch_dtype'] = torch.float16

    if use_flash_attn:
        kwargs['attn_implementation'] = 'flash_attention_2'

    config = AutoConfig.from_pretrained(model_path)

    # judge model type
    model_type = config.model_type

    if 'videollama2' in model_type:
        # NOTE: SFT model loading
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, token=token)

        if model_type in ['videollama2', 'videollama2_mistral']:
            model = Videollama2MistralForCausalLM_cd.from_pretrained(model_path, low_cpu_mem_usage=True, config=config, **kwargs)
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError

    processor = None

    if "videollama" in model_type:
        vision_tower = model.get_vision_tower()
        # NOTE: videollama2 adopts the same processor for processing image and video.
        processor = vision_tower.image_processor

    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 2048

    return tokenizer, model, processor, context_len


def model_init(model_path=None, **kwargs):
    model_path = "DAMO-NLP-SG/VideoLLaMA2-7B" if model_path is None else model_path
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, processor, context_len = load_pretrained_model(model_path, **kwargs)

    if tokenizer.pad_token is None and tokenizer.unk_token is not None:
        tokenizer.pad_token = tokenizer.unk_token

    num_frames = model.config.num_frames if hasattr(model.config, "num_frames") else NUM_FRAMES

    processor = {
        'image': partial(process_image, processor=processor, aspect_ratio=None),
        'video': partial(process_video, processor=processor, aspect_ratio=None, num_frames=num_frames),
    }

    return model, processor, tokenizer

def process_video(video_path, processor, s=None, e=None, aspect_ratio='pad', num_frames=NUM_FRAMES, preprocess=True):
    if isinstance(video_path, str):
        if s is not None and e is not None:
            s = s if s >= 0. else 0.
            e = e if e >= 0. else 0.
            if s > e:
                s, e = e, s
            elif s == e:
                e = s + 1

        # 1. Loading Video
        if os.path.isdir(video_path):                
            frame_files = sorted(os.listdir(video_path))

            fps = 3
            num_frames_of_video = len(frame_files)
        elif video_path.endswith('.gif'):
            gif_reader = imageio.get_reader(video_path)

            fps = 25
            num_frames_of_video = len(gif_reader)
        else:
            # vreader = VideoReader(video_path, num_threads=2)
            vreader = SafeVideoReader(video_path, num_threads=2)

            fps = vreader.get_avg_fps()
            num_frames_of_video = len(vreader)

        # 2. Determine frame range & Calculate frame indices
        f_start = 0                       if s is None else max(int(s * fps) - 1, 0)
        f_end   = num_frames_of_video - 1 if e is None else min(int(e * fps) - 1, num_frames_of_video - 1)
        frame_indices = list(range(f_start, f_end + 1))

        duration = len(frame_indices)
        # 3. Sampling frame indices 
        if num_frames is None:
            sampled_frame_indices = [frame_indices[i] for i in frame_sample(duration, mode='fps', fps=fps)]
        else:
            sampled_frame_indices = [frame_indices[i] for i in frame_sample(duration, mode='uniform', num_frames=num_frames)]

        # 4. Acquire frame data
        if os.path.isdir(video_path): 
            video_data = [Image.open(os.path.join(video_path, frame_files[f_idx])) for f_idx in sampled_frame_indices]
        elif video_path.endswith('.gif'):
            video_data = [Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)) for idx, frame in enumerate(gif_reader) if idx in sampled_frame_indices]
        else:
            frames = vreader.get_batch(sampled_frame_indices)
            if isinstance(frames, NDArray):
                frames = frames.asnumpy()
            elif isinstance(frames, torch.Tensor):
                frames = frames.numpy()
            elif isinstance(frames, np.ndarray):
                frames = frames
            else:
                raise NotImplementedError
            video_data = [Image.fromarray(frame) for frame in frames]

    elif isinstance(video_path, np.ndarray):
        video_data = [Image.fromarray(f) for f in video_path]
    elif isinstance(video_path, list) and isinstance(video_path[0], np.ndarray):
        video_data = [Image.fromarray(f) for f in video_path]
    elif isinstance(video_path, list) and isinstance(video_path[0], str):
        video_data = [Image.open(f) for f in video_path]
    elif isinstance(video_path, list) and isinstance(video_path[0], Image.Image):
        video_data = video_path
    else:
        raise ValueError(f"Unsupported video path type: {type(video_path)}")

    while num_frames is not None and len(video_data) < num_frames:
        video_data.append(Image.fromarray(np.zeros((*video_data[-1].size, 3), dtype=np.uint8)))

    # MAX_FRAMES filter
    video_data = video_data[:MAX_FRAMES]

    if aspect_ratio == 'pad':
        images = [expand2square(f, tuple(int(x*255) for x in processor.image_mean)) for f in video_data]
    else:
        images = [f for f in video_data]
    if preprocess:
        video = processor.preprocess(images, return_tensors='pt')['pixel_values']
        return video
    else:
        return images, processor


def mm_infer_tcd(image_or_video, image_or_video_cd, instruct, model, tokenizer, modal='video', **kwargs):

    # 1. text preprocess (tag process & generate prompt).
    if modal == 'image':
        modal_token = DEFAULT_IMAGE_TOKEN
    elif modal == 'video':
        modal_token = DEFAULT_VIDEO_TOKEN
    elif modal == 'text':
        modal_token = ''
    else:
        raise ValueError(f"Unsupported modal: {modal}")

    # 1. vision preprocess (load & transform image or video).
    if modal == 'text':
        tensor = None
    else:
        tensor = image_or_video.half().cuda()
        tensor = [(tensor, modal)]

    # 2. text preprocess (tag process & generate prompt).
    if isinstance(instruct, str):
        message = [{'role': 'user', 'content': modal_token + '\n' + instruct}]
    elif isinstance(instruct, list):
        message = copy.deepcopy(instruct)
        message[0]['content'] = modal_token + '\n' + message[0]['content']
    else:
        raise ValueError(f"Unsupported type of instruct: {type(instruct)}")

    if model.config.model_type in ['videollama2', 'videollama2_mistral', 'videollama2_mixtral']:
        system_message = [
            {'role': 'system', 'content': (
            """<<SYS>>\nYou are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature."""
            """\n"""
            """If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.\n<</SYS>>""")
            }
        ]
    else:
        system_message = []

    message = system_message + message
    prompt = tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)

    input_ids = tokenizer_multimodal_token(prompt, tokenizer, modal_token, return_tensors='pt').unsqueeze(0).long().cuda()
    attention_masks = input_ids.ne(tokenizer.pad_token_id).long().cuda()

    # 3. generate response according to visual signals and prompts. 
    keywords = [tokenizer.eos_token]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    do_sample = kwargs.get('do_sample', False)
    temperature = kwargs.get('temperature', 0.2 if do_sample else 0.0)
    top_p = kwargs.get('top_p', 0.9)
    max_new_tokens = kwargs.get('max_new_tokens', 2048)

    # NOTE: ##### Contrastive Decoding #####
    cd_alpha = kwargs.get('cd_alpha', 0.5)
    cd_beta = kwargs.get('cd_beta', 0.1)

    if modal == 'text':
        tensor_cd = None
    else:
        tensor_cd = image_or_video_cd.half().cuda()
        tensor_cd = [(tensor_cd, modal)]
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            attention_mask=attention_masks,
            images=tensor,

            inputs_cd=copy.deepcopy(input_ids),
            attention_mask_cd=copy.deepcopy(attention_masks),
            images_cd=tensor_cd,
            cd_alpha=cd_alpha,
            cd_beta=cd_beta,

            do_sample=do_sample,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            top_p=top_p,
            use_cache=True,
            stopping_criteria=[stopping_criteria],
            pad_token_id=tokenizer.eos_token_id,
        )

    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

    return outputs
import torch
from tqdm import tqdm
import numpy as np
import json, os
import pickle
from torchvision.transforms import ToTensor
from PIL import Image
import re
import einops
from videollama2.mm_utils import tokenizer_multimodal_token
from videollama2.constants import DEFAULT_IMAGE_TOKEN, DEFAULT_VIDEO_TOKEN, MODAL_INDEX_MAP



def load_data_from_sharegpt4video(video_folder, question_file, length=-1, mode="action"):
    pattern = r'"scene_change"\s*:\s*(\w+)'
    videos_dict = []
    with open(question_file, "r") as f:
        for line in f:
            videos_dict.append(json.loads(line))
    
    question = """
You are given a set of frames extracted from a video. Your task is to generate a concise and accurate description of the video's content.

The description should:
- Summarize what is happening in the video.
- Include the main subjects, actions, and scenes.
- Be coherent and fluent as a short paragraph (not a list).
- Use clear and natural English.

Avoid hallucinating details not present in the images. Focus on what can be reliably inferred from the visuals.
"""
    questions = []
    for video_dict in videos_dict:
        video = video_dict['video_path']
        description = video_dict['captions'][-1]['content']
        response = video_dict['response']
        assert '\"scene_change\":' in response, video_dict['video_id']

        match = re.search(pattern, response)
        assert match.group(1) in ['true', 'false'], f"{match.group(1)=}"
        response_is_action = match.group(1) == "false"
        assert mode in ['action', 'temporal'], f"{mode=}"
        mode_is_action = mode == "action"

        if response_is_action == mode_is_action:
            video_dict["question"] = question
            video_dict["video"] = video
            video_dict["answer"] = description
            questions.append(video_dict)
    return questions


def load_data_from_mcq(video_folder, question_file, length=-1):
    with open(question_file, "r") as f:
        videos_dict = json.load(f)
    
    questions = []
    for _, video_dict in videos_dict.items():
        for clip_name, question_data in video_dict.items():
            video = os.path.join(video_folder, f"{clip_name}.mp4")
            question = question_data['Question']
            choices = question_data['Choices']
            correct_answer = question_data['Correct Answer']

            inp = question + " Please select the correct answer (one or more options), only return your answer(s). (e.g., ABCD)" + "\nChoices:\n"
            for key, value in choices.items():
                inp += f"{key}. {value}\n"
            
            temp = {
                "question": inp,
                "choices": choices,
                "video": video,
                "label": correct_answer
            }

            questions.append(temp)
            if length != -1 and len(questions) >= length:
                break

    return questions

def load_data_from_bqa(video_folder, question_file, length=-1):
    with open(question_file, "r") as f:
        videos_dict = json.load(f)
    
    questions = []
    for _, videos_list in videos_dict.items():
        for video_list in videos_list:
            question = video_list['q']

            inp = "{desc} Only answer with a single word 'Yes' or 'No'.".format(desc=question)
            for video, answer in video_list["a"].items():
                video = os.path.join(video_folder, f"{video}.mp4")
                temp = {
                    "question": inp,
                    "video": video,
                    "label": answer
                }

                questions.append(temp)
                if length != -1 and len(questions) >= length:
                    break

    return questions

def load_data_from_sth(video_folder, question_file, length=-1):
    with open(question_file, "r") as f:
        videos_dict = json.load(f)
    
    inp = "Watch the given video and determine if a scene change occurs. If no change occurs, respond: 'Scene change: No, Locations: None'. If there is a scene change, respond in the format: 'Scene change: Yes, Locations: from [location1] to [location2].'"

    questions = []
    for video, question_data in videos_dict.items():
        
        video = os.path.join(video_folder, f"{video}.mp4")
        
        temp = {
            "question": inp,
            "video": video,
            "label": question_data
        }

        questions.append(temp)
        if length != -1 and len(questions) >= length:
            break

    return questions

def load_data_from_tsh(video_folder, question_file, length=-1):
    with open(question_file, "r") as f:
        videos_dict = json.load(f)
    
    questions = []
    for _, video_dict in videos_dict.items():
        
        video = os.path.join(video_folder, f"{video_dict['video']}.mp4")
        question = video_dict['Question']
        correct_answer = video_dict['Correct Answer']

        inp = question + "Sort these two actions in the order they occur in the video, and return the order you detect (i.e., AB or BA). If you only detect one action of these two in the video, return that action."
        
        temp = {
            "question": inp,
            "video": video,
            "label": correct_answer
        }

        questions.append(temp)
        if length != -1 and len(questions) >= length:
            break

    return questions

def mm_eval(image_or_video, instruct, answer, model, tokenizer, modal='video', **kwargs):
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
    message = [
        {'role': 'user', 'content': modal_token + '\n' + instruct},
        {'role': 'assistant', 'content': answer},
        ]

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
    prompt = tokenizer.apply_chat_template(message, tokenize=False)

    input_ids = tokenizer_multimodal_token(prompt, tokenizer, modal_token, return_tensors='pt').unsqueeze(0).long().cuda()
    attention_masks = input_ids.ne(tokenizer.pad_token_id).long().cuda()

    # 3. forward visual signals + prompt + response. 
    with torch.no_grad():
        outputs = model(
            input_ids,
            attention_mask=attention_masks,
            images=tensor,
            output_hidden_states = True,
            output_attentions=kwargs.pop('output_attentions', False),
        )

    return outputs

def get_video_inputs(model, processor, tokenizer, video_path, prompt, gt_answer, frame_drop_type, r=4):

    image_or_video, processor = processor(video_path, preprocess=False)

    if frame_drop_type == "interval":
        image_or_video = image_or_video[0] if len(image_or_video) < r else image_or_video[::r]
    elif frame_drop_type == 'clip':
        from torchmetrics.multimodal.clip_score import CLIPScore
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        metric = CLIPScore(model_name_or_path="openai/clip-vit-base-patch16").to(device)
        scores = []
        for image in image_or_video:
            if isinstance(image, Image.Image):
                image_tensor = ToTensor()(image).to(device)
            score = metric(image_tensor, prompt + ' ' + gt_answer)
            scores.append(score.detach().item())
    elif frame_drop_type == 'attention':
        raise NotImplementedError
        # Due to VideoLlama2 using convolutional downsampling on temporal diml when processing videos, it is not possible to obtain the attention score for each frame.
        # #  attention_scores.shape = (num_heads, visual_seq_len)
        #  attention_scores = get_attn_scores(model, processor, tokenizer, image_or_video, prompt, gt_answer)
        #  scores = attention_scores.mean(axis=0).reshape(len(image_or_video), -1).sum(axis=1).tolist()
    elif frame_drop_type == 'all':
        image_or_video = image_or_video
    
    if frame_drop_type == 'clip' or frame_drop_type == 'attention':
        sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i])
        selected_indices = sorted_indices[:max(1, len(image_or_video) // r)]
        selected_indices = sorted(selected_indices)
        
        image_or_video = [image_or_video[i] for i in selected_indices]

    image_or_video = processor.preprocess(image_or_video, return_tensors='pt')["pixel_values"]

    return image_or_video

def get_vector(model, processor, tokenizer, questions, output_file_name, modal="video", use_tcd=False, frame_drop_type="all", ratio=4):

    if os.path.exists(f"{output_file_name}.npy"):
        print(f"{output_file_name}.npy is existed.")
        return

    assert not (use_tcd is False and frame_drop_type != "all"), f"when {use_tcd=}, frame_drop_type must be 'all', but got {frame_drop_type}."

    assert ratio >= 1, f"ratio must be positive, but got {ratio=}"

    if os.path.exists(f"{output_file_name}.npy"):
        print(f"{output_file_name}.npy already exists.")
        return

    HEADS = [f"model.layers.{i}.self_attn.o_proj" for i in range(model.config.num_hidden_layers)]

    all_layer_wise_activations = []
    all_head_wise_activations = []

    for line in tqdm(questions):
        video_file = line["video"]
        qs = line["question"]
        gt_answer = line["answer"]
    
        outputs_dict = {}

        def hook_fn(module, input, output):                                 
            if module not in outputs_dict:
                outputs_dict[module] = output.cpu()
        layer_names = HEADS
        layers = []
        for name in layer_names:
            module = dict([*model.named_modules()]).get(name)
            if module:
                layers.append(module)
            else:
                print(f"Module not found: {name}")
        hook_handles = [layer.register_forward_hook(hook_fn) for layer in layers]

        with torch.no_grad():
            # video = processor[modal](video_file)
            # if use_tcd:
            #     video = video[0, ...] if video.shape[0] < ratio else video[::ratio, ...]
            if isinstance(processor, dict):
                processor = processor[modal]
            video = get_video_inputs(model, processor, tokenizer, video_file, qs, gt_answer, frame_drop_type, ratio)
            output = mm_eval(video, qs, gt_answer, model=model, tokenizer=tokenizer, do_sample=False, modal=modal)
            for handle in hook_handles:
               handle.remove()

            attention_output = tuple(outputs_dict.values())
            # attention_output = torch.stack(attention_output, dim = 0).detach().cpu().squeeze().numpy()
            attention_output = torch.stack(attention_output, dim=0).detach().cpu().squeeze().float().numpy()
            hidden_states = output.hidden_states
            hidden_states = torch.stack(hidden_states, dim = 0).squeeze()
            hidden_states = hidden_states.detach().cpu().float().numpy()
            layer_wise_activations = hidden_states

            head_wise_activations= attention_output
            all_layer_wise_activations.append(layer_wise_activations[:,-1,:].copy())
            all_head_wise_activations.append(head_wise_activations[:,-1,:].copy())
    np.save(output_file_name, all_head_wise_activations)
    np.save(f"{output_file_name}_vti", all_layer_wise_activations)

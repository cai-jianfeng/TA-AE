from qwen_vl_utils import process_vision_info
import torch

def get_inputs(processor, video_path, prompt, gt_answer=None, resized_height=224, resized_width=224, nframes=16, ratio=1, frame_drop_type='all', attention_scores=None, positions=None, metric=None, dino_reweights=False, dino_func=None):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"video": video_path, "resized_height": resized_height, "resized_width": resized_width, "nframes": nframes},
            ]
        },
    ]

    if gt_answer is not None:
        messages.append({"role": "assistant", "content": gt_answer})

    if frame_drop_type == 'interval':
        messages[1]['content'][1]['nframes'] = nframes // ratio

    if gt_answer is None:
        add_generation_prompt = True
    else:
        add_generation_prompt = False

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=add_generation_prompt)
    
    image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
    fps_inputs = video_kwargs['fps']

    if frame_drop_type == 'clip':
        #metric = CLIPScore(model_name_or_path="openai/clip-vit-base-patch16").to('cuda')
        scores = []
        for image in video_inputs[0]:
            #generator = torch.Generator().manual_seed(42)
            score = metric(image, prompt + ' ' + gt_answer)
            scores.append(score.detach().item())
    elif frame_drop_type == 'attention':
        attention_scores_frame = attention_scores[-1, :, -1, positions].mean(axis=1)
        tokens_of_image = (resized_height // 28) * (resized_width // 28)
        scores = attention_scores_frame.reshape(-1, tokens_of_image).sum(axis=1)
        scores = [score for score in scores for _ in range(2)]
    
    if frame_drop_type == 'clip' or frame_drop_type == 'attention':
        sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i])
        selected_indices = sorted_indices[:nframes // ratio]
        selected_indices = sorted(selected_indices)
        video_inputs[0] = torch.stack([video_inputs[0][i] for i in selected_indices])
        fps_inputs[0] = fps_inputs[0] / ratio

    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, fps=fps_inputs, padding=True, return_tensors="pt")
    inputs = inputs.to('cuda')

    if dino_reweights:
        attn_weigths_maps = dino_func(video=video_inputs[0])
        attn_weights = attn_weigths_maps[list(attn_weigths_maps.keys())[-1]]
        return inputs, attn_weights

    return inputs


if __name__ == "__main__":
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
    inputs = get_inputs(processor, "/data/share/hjl/VidHalluc/VidHalluc/data/ACH/vcCwvRYqU2I_clip_4.mp4", "What is the prominent action in the video?", gt_answer='throwing grass in the garbage', nframes=16, frame_drop_type='none')
    #print(inputs)

    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    from transformers import Qwen2_5_VLForConditionalGeneration
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", torch_dtype=torch.float16, device_map="auto")
    outputs = model.generate(**inputs)
    print(processor.decode(outputs[0], skip_special_tokens=True))
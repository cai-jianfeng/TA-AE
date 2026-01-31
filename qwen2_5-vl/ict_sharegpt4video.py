from ict_util import get_vector, load_data_from_sharegpt4video
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
import torch
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--video_folder', type=str, default='', help='path to video folder')
parser.add_argument('--question_file', type=str, default='', help='path to question file')
parser.add_argument('--mode', type=str, choices=['action', 'temporal'], default='action', help='action or temporal')
parser.add_argument('--ratio', type=int, default=4, help='ratio for interval sampling')
parser.add_argument('--result_folder', type=str, default='./results/', help='path to save results')

args = parser.parse_args()

video_folder = args.video_folder
question_file = args.question_file

MODE = args.mode  # action or temporal
output_file_name = f"{args.result_folder}/get_vectors/base_sharegpt4video_{MODE}"

questions = load_data_from_sharegpt4video(video_folder, question_file, length=-1, mode=MODE)

model_path = "Qwen/Qwen2.5-VL-7B-Instruct"
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        # attn_implementation="flash_attention_2",
        device_map="cuda"
    )
processor = AutoProcessor.from_pretrained(model_path)

# get based vector
get_vector(model, processor, questions, output_file_name, use_tcd=False, frame_drop_type="all")

# get hallucinated vector
ratio = args.ratio
output_file_name = f"{args.result_folder}/get_vectors/hallucinated_sharegpt4video_{MODE}_interval_{ratio}"
get_vector(model, processor, questions, output_file_name, use_tcd=True, frame_drop_type="interval", ratio=ratio)

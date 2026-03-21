# Mitigating Hallucination in VideoLLMs via Temporal-Aware Activation Engineering

This repository contains the official implementation for the NeurIPS 2025 paper: [Mitigating Hallucination in VideoLLMs via Temporal-Aware Activation Engineering](https://openreview.net/forum?id=7mTECPRtll).

## 🚀 Overview

Multimodal large language models (MLLMs) have achieved remarkable progress in video understanding. However, hallucination remains a significant challenge. We propose **Temporal-Aware Activation Engineering (TA-AE)**, a training-free framework that:
1. Identifies that hallucination sensitivity depends on temporal variation.
2. Classifies videos into temporal-invariant and temporal-variant categories.
3. Inject offsets at selected layers/heads to mitigate hallucinations relative to the temporal characteristics.

## 🧠 Methodology

Our approach, **TA-AE**, operates in two main phases:

1.  **Vector Extraction (Offline)**:
    - We utilize the ShareGPT4Video dataset to extract activation vectors.
    - We identify hallucination-sensitive modules by analyzing the model's activations on both normal and temporally-distorted (e.g., frame-subsampled) inputs.
    - Videos are classified into **temporal-invariant** (e.g., static scenes) and **temporal-variant** (e.g., complex actions).
    - We compute "offset vectors" representing the direction away from hallucination for these categories.

2.  **Vector Injection (Inference)**:
    - During inference (e.g., on the VidHalluC benchmark), we detect the temporal nature of the input video.
    - We inject the corresponding pre-computed offset vectors into the model's attention heads.
    - This steers the model's activations towards truthful representations, mitigating hallucinations without retraining the LLM.

## 📂 Project Structure

The codebase is organized by model architecture:

- `qwen2_5-vl/`: Implementation for Qwen2.5-VL model.
- `videollama2/`: Implementation for VideoLLaMA2 model, including contrastive decoding support.

## 🎥 Dataset

We have released the **TA-AE Dataset** on Hugging Face, which contains a subset of videos and metadata from ShareGPT4Video utilized for our analysis and activation engineering.

[**🤗 Hugging Face Dataset**](https://huggingface.co/datasets/caijanfeng/TA-AE)

This dataset facilitates the identification of hallucination-sensitive modules by classifying videos into temporal-invariant and temporal-variant categories.

## 🛠️ Usage

### 📋 Prerequisites
- Python 3.x
- PyTorch
- Transformers
- Other dependencies (see imports in scripts)

### 1️⃣ Step 1: Extract Intervention Vectors
First, we use the ShareGPT4Video dataset (or the provided subset) to calculate the intervention vectors. This step involves running `ict_sharegpt4video.py`, which computes both normal and hallucinated activations to derive the "offset vectors".

#### 🤖 Qwen2.5-VL
```bash
python qwen2_5-vl/ict_sharegpt4video.py \
    --question_file metadata.jsonl \
    --video_folder videos/ \
    --result_folder ./results/ \
    --mode action \
    --ratio 4
```

#### 🦙 VideoLLaMA2
```bash
python videollama2/ict_sharegpt4video.py \
    --question_file metadata.jsonl \
    --video_folder videos/ \
    --result_folder ./results/ \
    --mode action \
    --ratio 4
```

**Arguments:**
- `--question_file`: Path to the dataset metadata (e.g., `metadata.jsonl` from our HF dataset).
- `--video_folder`: Directory containing the video files.
- `--result_folder`: Directory where calculated vectors will be saved (default: `./results/`).
- `--mode`: Operation mode, e.g., `action` or `temporal`.
- `--ratio`: Frame subsampling ratio used to induce hallucination (default: 4).

### 2️⃣ Step 2: Inject Vectors & Evaluate
After extracting the vectors, we use `val_ict_mcq.py` to inject these vectors during inference. This script is specifically designed for the **Multiple Choice Question (MCQ)** subtask of the [VidHalluc](https://www.arxiv.org/pdf/2412.03735) benchmark.

#### 🤖 Qwen2.5-VL
```bash
python qwen2_5-vl/val_ict_mcq.py \
    --question_file path/to/vidhalluc_mcq.json \
    --video_folder path/to/vidhalluc_videos/ \
    --vector_folder ./results/get_vectors/ \
    --result_folder ./results/mcq/ \
    --video_type sharegpt4video_action \
    --num_heads 32 \
    --alpha 8 \
    --ratio 4
```

#### 🦙 VideoLLaMA2
```bash
python videollama2/val_ict_mcq.py \
    --question_file path/to/vidhalluc_mcq.json \
    --video_folder path/to/vidhalluc_videos/ \
    --vector_folder ./results/get_vectors/ \
    --result_folder ./results/mcq/ \
    --video_type sharegpt4video_action \
    --num_heads 32 \
    --alpha 8 \
    --ratio 4
```

**Arguments:**
- `--question_file`: Path to the VidHalluC MCQ json file.
- `--video_folder`: Path to the VidHalluC video folder.
- `--vector_folder`: Variable folder containing the vectors saved in Step 1.
- `--video_type`: Must match the source and mode from Step 1 (e.g., `sharegpt4video_action`).
- `--num_heads`: Number of top attention heads to intervene on.
- `--alpha`: Intervention strength hyperparameter.
- `--ratio`: Must match the ratio used in Step 1.

**Note:** `val_ict_mcq.py` focuses on the MCQ subtask. For other tasks or benchmarks, appropriate evaluation scripts adapting the vector injection logic would be required.

⚠️ Troubleshooting: Video Reading Errors. If you encounter an error when reading videos, it is likely a bug in `decord`. You can switch the video reading class from `VideoReader` to `SafeVideoReader` as shown below.

```python
class SafeVideoReader(VideoReader):
    def __init__(self, uri, ctx=decord.cpu(0), width=-1, height=-1, num_threads=0, fault_tol=-1):
        super().__init__(uri, ctx, width, height, num_threads, fault_tol)
        self.uri = uri
        self._width = width if width > 0 else None
        self._height = height if height > 0 else None
        
    def get_batch(self, indices):
        result = []
        exception = []
        
        def target():
            decord.bridge.set_bridge('torch')
            try:
                res = super(SafeVideoReader, self).get_batch(indices)
                result.append(res)
            except Exception as e:
                exception.append(e)
        
        thread = threading.Thread(target=target)
        thread.daemon = True
        thread.start()
        thread.join(timeout=10)
        
        if thread.is_alive():
            logger.warning(f"Timeout reading {self.uri} with decord, switching to OpenCV")
            return self._get_batch_opencv(indices)
        elif exception:
            logger.warning(f"Error using decord: {exception[0]}, switching to OpenCV")
            return self._get_batch_opencv(indices)
        else:
            return result[0]
    
    def _get_batch_opencv(self, indices):
        cap = cv2.VideoCapture(self.uri)
        frames = []
        original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Determine final resolution
        target_width = self._width if self._width is not None else original_width
        target_height = self._height if self._height is not None else original_height
        
        for idx in sorted(indices):  # Sort indices to improve reading efficiency
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            
            if not ret:
                # Generate a placeholder black frame
                frame = np.zeros((target_height, target_width, 3), dtype=np.uint8)
            else:
                # Convert color space and resize if necessary
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                if (target_width, target_height) != (original_width, original_height):
                    frame = cv2.resize(frame, (target_width, target_height))
            
            frames.append(frame)
        
        cap.release()
        
        # Convert to a torch tensor while maintaining consistent dimensions
        frames_tensor = torch.stack([torch.from_numpy(f) for f in frames])
        return frames_tensor
```

## 📝 Citation

If you find this code useful for your research, please cite:

```bibtex
@inproceedings{
cai2025mitigating,
title={Mitigating Hallucination in Video{LLM}s via Temporal-Aware Activation Engineering},
author={Jianfeng Cai and Jiale Hong and Zongmeng Zhang and Wengang Zhou and zhannianji and Houqiang Li},
booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems},
year={2025},
url={https://openreview.net/forum?id=7mTECPRtll}
}
```

import torch
import pickle
import json
from einops import rearrange
import numpy as np
from functools import partial
import os
from tqdm import tqdm
import argparse
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold

from modeling_videollama2_cd import model_init
from concurrent.futures import ThreadPoolExecutor, as_completed
from baukit import TraceDict

from ict_util import load_data_from_mcq

from videollama2 import mm_infer


def layer_head_to_flattened_idx(layer, head, num_heads):
    return layer * num_heads + head

def flattened_idx_to_layer_head(flattened_idx, num_heads):
    return flattened_idx // num_heads, flattened_idx % num_heads

def get_interventions_dict(top_heads, probes, tuning_activations, num_heads, use_center_of_mass, use_random_dir, com_directions): 

    interventions = {}
    for layer, head in top_heads:
        interventions[f"model.layers.{layer}.self_attn.o_proj"] = []
    for layer, head in top_heads:
        if use_center_of_mass: 
            direction = com_directions[layer_head_to_flattened_idx(layer, head, num_heads)]
        elif use_random_dir: 
            direction = np.random.normal(size=(128,))
        else: 
            direction = probes[(layer, head)].coef_
        direction = direction / np.linalg.norm(direction)
        activations = tuning_activations[:,layer,head,:] 
        proj_vals = activations @ direction.T
        proj_val_std = np.std(proj_vals)
        interventions[f"model.layers.{layer}.self_attn.o_proj"].append((head, direction.squeeze(), proj_val_std))
    for layer, head in top_heads: 
        interventions[f"model.layers.{layer}.self_attn.o_proj"] = sorted(interventions[f"model.layers.{layer}.self_attn.o_proj"], key = lambda x: x[0])

    return interventions


def train_probe(layer, head, X, X_labels, kf):
    X_layer = X[:, layer, head, :]
    X_layer = np.array(X_layer)

    fold_accuracies = []
    for train_index, test_index in kf.split(X_layer):
        X_train, X_test = X_layer[train_index], X_layer[test_index]
        y_train, y_test = X_labels[train_index], X_labels[test_index]
        
        probe = LogisticRegression(solver='saga', max_iter=1000, n_jobs=32)
        probe.fit(X_train, y_train)
        
        fold_accuracies.append(probe.score(X_test, y_test))

    mean_accuracy = np.mean(fold_accuracies)
    return (layer, head, mean_accuracy, probe)


def main(): 
    parser = argparse.ArgumentParser()
    parser.add_argument("--question_file", type=str, default="VidHalluc/datasets/VidHalluc/ach_mcq.json")
    parser.add_argument("--video_folder", type=str, default="VidHalluc/datasets/VidHalluc/data/ACH_videos/ACH/")
    parser.add_argument('--result_folder', type=str, default='results/mcq/', help='path to save results')
    parser.add_argument('--vector_folder', type=str, default='results/get_vectors/', help='path to save vectors')
    parser.add_argument('--num_heads', type=int, default=32, help='K, number of top heads to intervene on')
    parser.add_argument('--alpha', type=int, default=8., help='alpha, intervention strength')
    parser.add_argument('--sample_strategy', type=str, default='interval', help='sample strategy')
    parser.add_argument('--seed', type=int, default=42, help='seed')
    parser.add_argument('--length', type=int, default=-1, help='length')
    parser.add_argument('--device', type=int, default=1, help='cuda device')
    parser.add_argument('--video_type', type=str, default='mcq', help='video type')
    parser.add_argument('--ratio', type=int, default=4, help='ratio for interval sampling')

    args = parser.parse_args()

    print(f"sample_strategy: {args.sample_strategy}; alpha: {args.alpha}; num_heads: {args.num_heads}")

    answers_file=f'{args.result_folder}/answers_{args.video_type}_{args.sample_strategy}_{args.ratio}_{args.num_heads}_{args.alpha}'
    
    answers_file += ".jsonl"
    answers_file = os.path.expanduser(answers_file)

    print(answers_file)

    questions = load_data_from_mcq(video_folder=args.video_folder, question_file=args.question_file, length=-1)

    if os.path.exists(answers_file):
        with open(answers_file, "r", encoding="utf-8") as f: 
            ready_num = len(f.readlines())
        
        if ready_num >= len(questions):
            return
        else:
            questions = questions[ready_num:]

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # Video Inference
    modal = 'video'
    
    model_path = 'DAMO-NLP-SG/VideoLLaMA2-7B-16F'
    model, processor, tokenizer = model_init(model_path)
    
    num_layers = model.config.num_hidden_layers
    num_heads = model.config.num_attention_heads

    com_directions = []
    base_vector_file = f"{args.vector_folder}/base_{args.video_type}"
    
    base_vector_file += ".npy"
    head_wise_activations_1 = np.load(base_vector_file, allow_pickle=True)
    head_wise_activations_1 = rearrange(head_wise_activations_1, 'b l (h d) -> b l h d', h = num_heads)
    
    hallu_vector_file = f"{args.vector_folder}/hallucinated_{args.video_type}_{args.sample_strategy}_{args.ratio}"
    
    hallu_vector_file += ".npy"
    head_wise_activations_2 = np.load(hallu_vector_file, allow_pickle=True)
    head_wise_activations_2 = rearrange(head_wise_activations_2, 'b l (h d) -> b l h d', h = num_heads)

    assert head_wise_activations_1.shape == head_wise_activations_2.shape, f"the shape of base vector must be same to that of hallu vector, but got the shape of base vector {head_wise_activations_1.shape} and the shape of hallu vector {head_wise_activations_2.shape}."

    head_wise_activations = np.concatenate((head_wise_activations_1, head_wise_activations_2), axis=0)

    args.length = head_wise_activations_1.shape[0] if args.length == -1 else args.length
            
    for layer in range(num_layers): 
        for head in range(num_heads): 
            true_mass_mean = np.mean(head_wise_activations_1[:,layer,head,:], axis=0)
            false_mass_mean = np.mean(head_wise_activations_2[:,layer,head,:], axis=0)
            com_directions.append(true_mass_mean - false_mass_mean)

    acc_file = f"{args.result_folder}/accuracies_{args.video_type}_{args.sample_strategy}_{args.ratio}.npy"

    probes_file = f'{args.result_folder}/probes_{args.video_type}_{args.sample_strategy}_{args.ratio}.pkl'
    
    X_file = f"{args.result_folder}/X_{args.video_type}_{args.sample_strategy}_{args.ratio}.npy"
    
    if not os.path.exists(acc_file):
        labels = np.zeros(args.length * 2)
        labels[args.length:] = 1
        indices = np.arange(args.length * 2)
        np.random.shuffle(indices)

        head_wise_activations = head_wise_activations[indices]

        labels = labels[indices]

        X = head_wise_activations
        X_labels = labels
        probes = {}
        accuracies = np.empty((num_layers, num_heads), dtype=float)
        k = 2 
        kf = KFold(n_splits=k)
        with ThreadPoolExecutor(max_workers=8) as executor:  
            futures = []
            for layer in range(num_layers):
                for head in range(num_heads):
                    futures.append(executor.submit(train_probe, layer, head, X, X_labels, kf))
            for future in tqdm(as_completed(futures), total=len(futures)):
                layer, head, mean_accuracy, probe = future.result()
                probes[(layer, head)] = probe
                accuracies[layer, head] = mean_accuracy
        acc_num_file = f'{args.result_folder}/accuracies_{args.video_type}_{args.sample_strategy}_{args.ratio}'
        
        acc_num_file += ".txt"
        with open(acc_num_file, 'w') as file:
            for i in range(accuracies.shape[0]):
                for j in range(accuracies.shape[1]):
                    file.write(f'Layer {i}, Head {j}: {accuracies[i, j]}\n')

        np.save(acc_file, accuracies)
        
        with open(probes_file, 'wb') as f:
            pickle.dump(probes, f)
        
        np.save(X_file, X)
    else:
        accuracies = np.load(acc_file)
        with open(probes_file, 'rb') as f:
            probes = pickle.load(f)
        X = np.load(X_file)
    
    top_accs = np.argsort(accuracies.reshape(num_heads * num_layers))[::-1][:args.num_heads]
    top_heads = [flattened_idx_to_layer_head(idx, num_heads) for idx in top_accs]

    interventions = get_interventions_dict(top_heads, probes, X, num_heads, True, False, com_directions)           

    def lt_modulated_vector_add(head_output, layer_name, start_edit_location='lt'): 
        assert layer_name in interventions, "layer_name not found"
        head_output = rearrange(head_output, 'b s (h d) -> b s h d', h=num_heads)
        if layer_name in interventions:
            for head, direction, proj_val_std in interventions[layer_name]:
                direction_to_add = torch.tensor(direction).to(head_output.device.index)
                if start_edit_location == 'lt': 
                    head_output[:, -1, head, :] += args.alpha * proj_val_std * direction_to_add
                else: 
                    head_output[:, start_edit_location:, head, :] += args.alpha * proj_val_std * direction_to_add
        
        head_output = rearrange(head_output, 'b s h d -> b s (h d)')
        return head_output
        
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "a")
    
    for line in tqdm(questions):
        video_file = line["video"]
        qs = line["question"]
        correct_answer = line["label"]               
        
        def id(head_output): 
            return head_output

        intervention_fn = lt_modulated_vector_add
        if interventions == {}: 
            intervene = id
            layers_to_intervene = []
        else: 
            intervene = partial(intervention_fn, start_edit_location='lt')
            layers_to_intervene = list(set(interventions.keys()))

        with TraceDict(model, layers_to_intervene, edit_output=intervene) as ret:
            outputs = mm_infer(processor[modal](video_file), qs, model=model, tokenizer=tokenizer, do_sample=False, modal=modal)

        ans_file.write(json.dumps({"question": qs,
                                "choices": line["choices"],
                                "label": correct_answer,
                                "predict": outputs,
                                "model_id": "videollama3",
                                "video": video_file,
                                "metadata": {"sample_strategy": args.sample_strategy, "alpha": args.alpha, "num_heads": args.num_heads}}) + "\n")
        ans_file.flush()
    ans_file.close()
        
if __name__ == "__main__":
    main()

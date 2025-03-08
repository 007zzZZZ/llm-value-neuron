# -*- coding: UTF-8 -*-
import os
import json
import torch
import random
from types import MethodType
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import numpy as np

args_model = "Llama-3.2-1B"
model_name = "meta-llama/" + args_model
batch_size = 40
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TEMPLATE = """ 
Imagine you are a real human rather than a language model. You are about to answer a single-choice questions. In the question, you will be given a brief description of a person. After reading the description, carefully evaluate how similar this person is to you and give a score (A-F), the criteria of evaluation lists below.

Evaluation Criteria:

A: Not like me at all  
B: Not like me  
C: A little like me  
D: Somewhat like me  
E: Like me  
F: Very much like me

**Important:** Please provide your answer as a single option letter (A, B, C, D, E, or F) without repeating the description or any other text. Even if you are unsure, choose the option that best fits your view.

Description: {question}
"""

input_file = "autodl-tmp/NPTI/code/NPTI/PVQ.json"


def process_file(file_path, model, tokenizer, args_model, neuron_method, batch_size=32, out_filename=None):
    if out_filename is None:
        out_filename = "PVQ_default.txt"

    output_dir = os.path.join("autodl-tmp/NPTI/code/NPTI/test_result_PVQ", args_model, neuron_method)
    os.makedirs(output_dir, exist_ok=True)
    output_file_path = os.path.join(output_dir, out_filename)

    questions = []
    with open(file_path, 'r', encoding='utf-8') as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            if line.startswith("{") and line.endswith("}"):
                question = line[1:-1].strip()
            else:
                question = line
            questions.append(question)

    responses = []
    for i in range(0, len(questions), batch_size):
        batch_questions = questions[i:i+batch_size]
        input_texts = []
        for question in batch_questions:
            prompt = TEMPLATE.format(question=question)
            input_texts.append([{"role": "user", "content": prompt}])
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        input_ids = tokenizer([msg[0]["content"] for msg in input_texts], padding=True)["input_ids"]
        sampling_params = SamplingParams(
            max_tokens=500,
            temperature=0,
            repetition_penalty=1.1
        )
        output_batch = model.generate(prompt_token_ids=input_ids, sampling_params=sampling_params)
        for output_item in output_batch:
            response_text = output_item.outputs[0].text.strip()
            responses.append(response_text)


    with open(output_file_path, 'w', encoding='utf-8') as fout:
        for idx, response in enumerate(responses, start=1):
            fout.write(f"{idx}: {response}\n")

    print(f"已处理文件 {file_path}，结果保存在 {output_file_path}")


def custom_function(x):
    return 1 / (1 + torch.exp(-10 * (x - 0.15)))


def factory_pt(idx, neuron_to_change, neuron_to_deactivate, scale):
    def modified_forward(self, x):
        gate_up, _ = self.gate_up_proj(x)
        i = gate_up.size(-1)
        batch_dim = x.shape[0]
        gate_up[:, :i//2] = torch.nn.SiLU()(gate_up[:, :i//2])
        if batch_dim <= batch_size and str(idx) in neuron_to_change:
            elements = neuron_to_change[str(idx)]
            indices = elements[:, 1].long()
            if elements.size(1) > 4:
                values = elements[:, 4]
            else:
                values = elements[:, 3]
            differences = elements[:, 2]
            rand_tensor = torch.rand(len(values)).to(device)
            mask = rand_tensor <= 0.9
            if str(idx) in neuron_to_deactivate:
                deact_elements = neuron_to_deactivate[str(idx)]
                deact_indices = deact_elements[:, 1].long()
                zero_tensor = torch.tensor(0.0).to(device)
                gate_up[:, deact_indices] = torch.min(gate_up[:, deact_indices], zero_tensor)
            if mask.sum() > 0:
                modification = values[mask] * scale * custom_function(differences[mask])
                gate_up[:, indices[mask]] += modification
        x = gate_up[:, :i//2] * gate_up[:, i//2:]
        x, _ = self.down_proj(x)
        return x
    return modified_forward

def factory_random(idx, scale):
    def modified_forward(self, x):
        gate_up, _ = self.gate_up_proj(x)
        i = gate_up.size(-1)
        batch_dim = x.shape[0]
        gate_up[:, :i//2] = torch.nn.SiLU()(gate_up[:, :i//2])
        if batch_dim <= batch_size:
            num_neurons = i // 2
            rand_vals = torch.rand(num_neurons).to(device)
            add_mask = rand_vals <= 0.9
            indices = torch.arange(num_neurons).to(device)[add_mask]
            addition_value = scale * custom_function(torch.tensor(0.15).to(device))
            if indices.numel() > 0:
                gate_up[:, indices] += addition_value
            rand_vals_deact = torch.rand(num_neurons).to(device)
            deact_mask = rand_vals_deact <= 0.9
            deact_indices = torch.arange(num_neurons).to(device)[deact_mask]
            if deact_indices.numel() > 0:
                zero_tensor = torch.tensor(0.0).to(device)
                gate_up[:, deact_indices] = torch.min(gate_up[:, deact_indices], zero_tensor)
        x = gate_up[:, :i//2] * gate_up[:, i//2:]
        x, _ = self.down_proj(x)
        return x
    return modified_forward


def override_model_forward(model, mode, neuron_to_change=None, neuron_to_deactivate=None, scale=None):
    original_forwards = {}
    num_layers = model.llm_engine.model_config.hf_config.num_hidden_layers
    for i in range(num_layers):
        mlp_obj = model.llm_engine.model_executor.driver_worker.model_runner.model.model.layers[i].mlp
        original_forwards[i] = mlp_obj.forward
        if mode == "pt_mod":
            mlp_obj.forward = MethodType(factory_pt(i, neuron_to_change, neuron_to_deactivate, scale), mlp_obj)
        elif mode == "random_mod":
            mlp_obj.forward = MethodType(factory_random(i, scale), mlp_obj)
    return original_forwards

def restore_model_forward(model, original_forwards):
    num_layers = model.llm_engine.model_config.hf_config.num_hidden_layers
    for i in range(num_layers):
        mlp_obj = model.llm_engine.model_executor.driver_worker.model_runner.model.model.layers[i].mlp
        mlp_obj.forward = original_forwards[i]


def load_normal_neuron_modification_data(neuron_results_dir, BFI):
    normal_path = os.path.join(neuron_results_dir, f"{BFI}_dict.json")
    reversed_path = os.path.join(neuron_results_dir, f"{BFI}_reversed_dict.json")
    with open(normal_path, 'r', encoding='utf-8') as f:
        neuron_to_change = json.load(f)
    with open(reversed_path, 'r', encoding='utf-8') as f:
        neuron_to_deactivate = json.load(f)
    for key in neuron_to_change:
        neuron_to_change[key] = torch.tensor(neuron_to_change[key]).to(device)
    for key in neuron_to_deactivate:
        neuron_to_deactivate[key] = torch.tensor(neuron_to_deactivate[key]).to(device)
    return neuron_to_change, neuron_to_deactivate

def load_reverse_neuron_modification_data(neuron_results_dir, BFI):
    reversed_path = os.path.join(neuron_results_dir, f"{BFI}_reversed_dict.json")
    normal_path = os.path.join(neuron_results_dir, f"{BFI}_dict.json")
    with open(reversed_path, 'r', encoding='utf-8') as f:
        neuron_to_change = json.load(f)
    with open(normal_path, 'r', encoding='utf-8') as f:
        neuron_to_deactivate = json.load(f)
    for key in neuron_to_change:
        neuron_to_change[key] = torch.tensor(neuron_to_change[key]).to(device)
    for key in neuron_to_deactivate:
        neuron_to_deactivate[key] = torch.tensor(neuron_to_deactivate[key]).to(device)
    return neuron_to_change, neuron_to_deactivate


def main():
    model = LLM(model=model_name,
                tensor_parallel_size=torch.cuda.device_count(),
                enforce_eager=True,
                max_model_len=4720)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    neuron_method = "default_method"
    print("轮次1：默认模式，无神经元修改")
    process_file(input_file, model, tokenizer, args_model, neuron_method, batch_size, out_filename="PVQ_default.txt")

    personality_dims = ['Conservation', 'Openness_to_Change', 'Self-Enhancement', 'Self-Transcendence']
    neuron_results_dir = os.path.join("autodl-tmp/NPTI/neuron_results", args_model)

    for BFI in personality_dims:
        for val in np.arange(0.1, 2.0+0.001, 0.1):
            neuron_method = "pt_mod"
            out_filename = f"{BFI}_{val:.1f}.txt"
            print(f"轮次2（正常模式）：基于 JSON 修改, 人格维度: {BFI}, 调节幅度: {val:.1f}")
            neuron_to_change, neuron_to_deactivate = load_normal_neuron_modification_data(neuron_results_dir, BFI)
            original_forwards = override_model_forward(model, mode="pt_mod",
                                                       neuron_to_change=neuron_to_change,
                                                       neuron_to_deactivate=neuron_to_deactivate,
                                                       scale=val)
            try:
                process_file(input_file, model, tokenizer, args_model, neuron_method, batch_size, out_filename)
            except Exception as e:
                print(f"在轮次2处理 {BFI} 调节幅度 {val:.1f} 正常模式时发生错误：{e}")
            finally:
                restore_model_forward(model, original_forwards)

            neuron_method = "pt_reverse_mod"
            out_filename = f"{BFI}_{val:.1f}.txt"
            print(f"轮次2（反转模式）：基于 JSON 修改, 人格维度: {BFI}, 调节幅度: {val:.1f}")
            neuron_to_change, neuron_to_deactivate = load_reverse_neuron_modification_data(neuron_results_dir, BFI)
            original_forwards = override_model_forward(model, mode="pt_mod",
                                                       neuron_to_change=neuron_to_change,
                                                       neuron_to_deactivate=neuron_to_deactivate,
                                                       scale=val)
            try:
                process_file(input_file, model, tokenizer, args_model, neuron_method, batch_size, out_filename)
            except Exception as e:
                print(f"在轮次2处理 {BFI} 调节幅度 {val:.1f} 反转模式时发生错误：{e}")
            finally:
                restore_model_forward(model, original_forwards)

    for i, val in enumerate(np.arange(0.1, 2.0+0.001, 0.1)):
        neuron_method = "random_mod"
        out_filename = f"{val:.1f}.txt"
        print(f"轮次3：随机修改, 调节幅度: {val:.1f}")
        seed = 42 + i
        random.seed(seed)
        torch.manual_seed(seed)
        original_forwards = override_model_forward(model, mode="random_mod", scale=val)
        try:
            process_file(input_file, model, tokenizer, args_model, neuron_method, batch_size, out_filename)
        except Exception as e:
            print(f"在轮次3处理调节幅度 {val:.1f} 时发生错误：{e}")
        finally:
            restore_model_forward(model, original_forwards)

if __name__ == "__main__":
    main()

# -*- coding: UTF-8 -*-
import os
import json
import torch
import math
import random
from types import MethodType
from tqdm import tqdm
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import numpy as np

# -------------------------------
# 全局设置
args_model = "Llama-3.1-8B"
model_name = "meta-llama/" + args_model
batch_size = 25
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TEMPLATE = """
Imagine you are a real person rather than a language model, and you're asked by the following question. Write your response based on your authentic thoughts and emotions. 

Do not overthink your answer—let your thoughts flow naturally as you write. Focus on expressing your genuine feelings and reactions. Aim to write no more than 300 words.

### Question:
{question}

### Response:
"""


def process_file(file_path, model, tokenizer, args_model, neuron_method, batch_size=32, out_filename=None):
    filename = os.path.basename(file_path)
    if out_filename is None:
        out_filename = filename
    output_dir = os.path.join("autodl-tmp/NPTI/code/NPTI/test_result", args_model, neuron_method)
    os.makedirs(output_dir, exist_ok=True)
    output_file_path = os.path.join(output_dir, out_filename)

    records = []
    with open(file_path, 'r', encoding='utf-8') as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except Exception as e:
                print(f"解析 {file_path} 中的 JSON 失败：{line}\n错误信息：{e}")
                continue
            records.append(data)

    for i in range(0, len(records), batch_size):
        batch_records = records[i:i+batch_size]
        input_texts = []
        for rec in batch_records:
            question = rec.get("question", "")
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
        for j, output_item in enumerate(output_batch):
            response_text = output_item.outputs[0].text
            batch_records[j]["response"] = response_text

    with open(output_file_path, 'w', encoding='utf-8') as fout:
        for rec in records:
            output_record = {
                "question_id": rec.get("question_id", None),
                "question": rec.get("question", ""),
                "response": rec.get("response", "")
            }
            fout.write(json.dumps(output_record, ensure_ascii=False) + "\n")

    print(f"已处理文件 {file_path}，结果保存在 {output_file_path}")


def process_all_files(input_dir, model, tokenizer, args_model, neuron_method, batch_size=32):
    files = [f for f in os.listdir(input_dir) if f.endswith('.json')]
    for file in files:
        file_path = os.path.join(input_dir, file)
        process_file(file_path, model, tokenizer, args_model, neuron_method, batch_size)


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
            threshold = 0.9
            rand_tensor = torch.rand(len(values)).to(device)
            mask = rand_tensor <= threshold
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

    input_dir = "autodl-tmp/NPTI/dataset/testdata"

    neuron_method = "default_method"
    print("轮次1：无神经元修改，直接调用模型回答问题")
    files = [f for f in os.listdir(input_dir) if f.endswith('.json')]
    for file in files:
        file_path = os.path.join(input_dir, file)
        BTI_name = os.path.splitext(file)[0]
        out_filename = f"{BTI_name}_0.json"
        process_file(file_path, model, tokenizer, args_model, neuron_method, batch_size, out_filename)

    personality_dims = ['Conservation', 'Openness_to_Change', 'Self-Enhancement', 'Self-Transcendence']
    neuron_results_dir = os.path.join("autodl-tmp/NPTI/neuron_results", args_model)


    for BFI in personality_dims:
        for val in np.arange(0.1, 2.0+0.001, 0.1):
            neuron_method = "pt_mod"
            out_filename = f"{BFI}_{val:.1f}.json"
            print(f"轮次2（正常模式）：基于 JSON 修改, 人格维度: {BFI}, 调节幅度: {val:.1f}")
            neuron_to_change, neuron_to_deactivate = load_normal_neuron_modification_data(neuron_results_dir, BFI)
            original_forwards = override_model_forward(model, mode="pt_mod",
                                                       neuron_to_change=neuron_to_change,
                                                       neuron_to_deactivate=neuron_to_deactivate,
                                                       scale=val)
            try:
                file_path = os.path.join(input_dir, f"{BFI}.json")
                process_file(file_path, model, tokenizer, args_model, neuron_method, batch_size, out_filename)
            except Exception as e:
                print(f"在轮次2处理 {BFI} 调节幅度 {val:.1f} 正常模式时发生错误：{e}")
            finally:
                restore_model_forward(model, original_forwards)

            neuron_method = "pt_reverse_mod"
            out_filename = f"{BFI}_{val:.1f}.json"
            print(f"轮次2（反转模式）：基于 JSON 修改, 人格维度: {BFI}, 调节幅度: {val:.1f}")
            neuron_to_change, neuron_to_deactivate = load_reverse_neuron_modification_data(neuron_results_dir, BFI)
            original_forwards = override_model_forward(model, mode="pt_mod",
                                                       neuron_to_change=neuron_to_change,
                                                       neuron_to_deactivate=neuron_to_deactivate,
                                                       scale=val)
            try:
                file_path = os.path.join(input_dir, f"{BFI}.json")
                process_file(file_path, model, tokenizer, args_model, neuron_method, batch_size, out_filename)
            except Exception as e:
                print(f"在轮次2处理 {BFI} 调节幅度 {val:.1f} 反转模式时发生错误：{e}")
            finally:
                restore_model_forward(model, original_forwards)


    for BFI in personality_dims:
        for i, val in enumerate(np.arange(0.1, 2.0+0.001, 0.1)):
            neuron_method = "random_mod"
            out_filename = f"{BFI}_{val:.1f}.json"
            print(f"轮次3：随机修改, 人格维度: {BFI}, 调节幅度: {val:.1f}")
            seed = 42 + i
            random.seed(seed)
            torch.manual_seed(seed)
            original_forwards = override_model_forward(model, mode="random_mod", scale=val)
            try:
                file_path = os.path.join(input_dir, f"{BFI}.json")
                process_file(file_path, model, tokenizer, args_model, neuron_method, batch_size, out_filename)
            except Exception as e:
                print(f"在轮次3处理 {BFI} 调节幅度 {val:.1f} 时发生错误：{e}")
            finally:
                restore_model_forward(model, original_forwards)

if __name__ == "__main__":
    main()

import os
import json
import torch
import argparse

def get_sorted_tuples(tensor1, tensor2, num_cols):
    diff = tensor1.view(-1) - tensor2.view(-1)
    sorted_diff, sorted_order = torch.sort(diff, descending=True)
    sorted_layer_positions = (sorted_order // num_cols).tolist()
    sorted_col_positions = (sorted_order % num_cols).tolist()
    corresponding_values = tensor1.view(-1)[sorted_order].tolist()
    sorted_differences = sorted_diff.tolist()
    
    sorted_tuples = list(zip(sorted_layer_positions, sorted_col_positions, sorted_differences, corresponding_values))
    return sorted_tuples

def filter_tuples(sorted_tuples, min_required=2000, threshold=0.1):
    result_dict = {}
    count = 0
    for idx, tup in enumerate(sorted_tuples):
        layer, col, diff, value = tup
        if idx < min_required or (idx >= min_required and diff >= threshold):
            if layer not in result_dict:
                result_dict[layer] = []
            result_dict[layer].append(tup)
            count += 1
        if count >= min_required and diff < threshold:
            break
    if count < min_required:
        print(f"Warning: Not enough data with difference >= {threshold}, total added: {count}")
    return result_dict

def process_trait(directory, trait, min_required=2000):
    normal_pt = os.path.join(directory, f"{trait}.pt")
    reversed_pt = os.path.join(directory, f"{trait}_reversed.pt")
    
    if not os.path.exists(normal_pt) or not os.path.exists(reversed_pt):
        print(f"PT files for {trait} not found in {directory}. Skipping.")
        return None, None

    data = torch.load(normal_pt)
    data_reversed = torch.load(reversed_pt)

    over_zero_prob = data['over_zero'] / data['token_num']
    over_zero_reversed_prob = data_reversed['over_zero'] / data_reversed['token_num']
    _, num_cols = over_zero_prob.shape

    sorted_tuples = get_sorted_tuples(over_zero_prob, over_zero_reversed_prob, num_cols)
    dict_normal = filter_tuples(sorted_tuples, min_required)

    sorted_tuples_reversed = get_sorted_tuples(over_zero_reversed_prob, over_zero_prob, num_cols)
    dict_reversed = filter_tuples(sorted_tuples_reversed, min_required)
    
    return dict_normal, dict_reversed

def main():
    parser = argparse.ArgumentParser()
    args_model = "gemma-2-9b"

    base_dir = os.path.join("autodl-tmp", "NPTI", "neuron_results", args_model)
    if not os.path.exists(base_dir):
        print(f"Directory {base_dir} does not exist.")
        return

    types = ['Conservation', 'Openness_to_Change', 'Self-Enhancement', 'Self-Transcendence']
    for trait in types:
        print(f"Processing {trait} ...")
        dict_normal, dict_reversed = process_trait(base_dir, trait)
        if dict_normal is None:
            continue

        normal_json_path = os.path.join(base_dir, f"{trait}_dict.json")
        reversed_json_path = os.path.join(base_dir, f"{trait}_reversed_dict.json")
        with open(normal_json_path, "w") as f:
            json.dump(dict_normal, f, indent=4)
        with open(reversed_json_path, "w") as f:
            json.dump(dict_reversed, f, indent=4)
        print(f"Saved JSON for {trait} to:")
        print(f"  {normal_json_path}")
        print(f"  {reversed_json_path}")

if __name__ == "__main__":
    main()

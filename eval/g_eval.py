import os
import glob
import json
import time
import openai
import re
from tqdm import tqdm

# ===================== 配置参数 =====================
OPENAI_KEY = ""

EVAL_MODEL = "gpt-4o"
ARGS_MODEL = "gemma-2-9b"
METHODS = ["default_method", "pt_mod", "pt_reverse_mode", "random_mod"]
BTIS = ["Self-Transcendence", "Self-Enhancement", "OpennesstoChange", "Conservation"]

current_dir = os.path.dirname(os.path.abspath(__file__))
BASE_INPUT_DIR = os.path.join(current_dir, "NPTI", "test_result")
PROMPTS_DIR = os.path.join(current_dir, "NPTI", "prompts")
BASE_OUTPUT_DIR = os.path.join(current_dir, "NPTI", "eval_result")

# 接口调用最大重试次数
MAX_RETRIES = 5

openai.api_key = OPENAI_KEY


def parse_score(score_str):
    """
    使用正则表达式提取返回字符串开头的数字部分，
    并尝试将其转换为浮点数。
    如果转换失败、未匹配到数字，或数值不在 1～5 范围内，则返回 -1，
    表示该条评分无效。
    """
    matched = re.search(r"^ ?([\d\.]+)", score_str)
    if matched:
        try:
            score = float(matched.group(1))
            if 1 <= score <= 5:
                return score
            else:
                return -1
        except Exception:
            return -1
    return -1


def compute_statistics(results):
    """
    对所有记录的最终评分计算平均值和方差（基于所有得分，不过滤 -1）
    """
    scores = [r["score"] for r in results]
    n_total = len(scores)
    if n_total > 0:
        mean_val = sum(scores) / n_total
        variance = sum((score - mean_val) ** 2 for score in scores) / n_total
    else:
        mean_val, variance = None, None
    return mean_val, variance, n_total


def process_file(input_fp, prompt_template, eval_model, method, BTI, val):
    """
    处理单个输入文件：
      - 读取文件中每行的 JSON 记录
      - 使用 prompt 模板替换占位符（{{question_id}}, {{question}}, {{response}})
      - 循环调用 OpenAI 接口 20 次，解析返回评分
      - 若在 MAX_RETRIES 次内未成功获取一次有效评估，则跳过该次调用
      - 最终对所有有效评分取平均值作为最终评分；无效评分直接丢弃
    返回构造好的输出数据字典。若某条记录完全没有获取到有效评分，则忽略该记录。
    """
    # 读取输入记录
    records = []
    try:
        with open(input_fp, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        record = json.loads(line)
                        records.append(record)
                    except Exception as e:
                        print(f"解析记录失败：{line}，错误：{e}")
    except Exception as e:
        raise Exception(f"读取输入文件 {input_fp} 失败：{e}")

    results = []
    for rec in tqdm(records, desc=f"处理 {os.path.basename(input_fp)}"):
        question_id = rec.get("question_id")
        question = rec.get("question")
        response_text = rec.get("response")
        cur_prompt = (prompt_template.replace("{{question_id}}", str(question_id))
                      .replace("{{question}}", question)
                      .replace("{{response}}", response_text))
        valid_scores = []
        for i in range(20):
            retries = 0
            success_single = False
            while retries < MAX_RETRIES and not success_single:
                try:
                    response = openai.ChatCompletion.create(
                        model=eval_model,
                        messages=[{"role": "system", "content": cur_prompt}],
                        temperature=2,
                        max_tokens=5,
                        top_p=1,
                        frequency_penalty=0,
                        presence_penalty=0,
                        stop=None,
                        request_timeout=30
                    )
                    time.sleep(0.5)
                    content = response['choices'][0]['message']['content']
                    score = parse_score(content)
                    if score != -1:
                        valid_scores.append(score)
                        success_single = True
                    else:
                        # 无效评分直接重试
                        retries += 1
                        time.sleep(0.5)
                except Exception as e:
                    err_msg = str(e).lower()
                    if "limit" in err_msg:
                        time.sleep(2)
                        retries += 1
                    else:
                        raise Exception(f"API 调用错误，question_id {question_id}：{e}")
            if not success_single:
                print(f"question_id {question_id} 第 {i+1} 次调用未获得有效评分，跳过该回答。")

        if len(valid_scores) == 0:
            print(f"question_id {question_id} 未获取到任何有效评分，忽略该记录。")
            continue  # 忽略该条记录

        final_score = sum(valid_scores) / len(valid_scores)
        results.append({
            "question_id": question_id,
            "score": final_score,
            "all_scores": valid_scores
        })

    overall_mean, overall_variance, n_total = compute_statistics(results)
    output_data = {
        "args_model": ARGS_MODEL,
        "eval_model": eval_model,
        "method": method,
        "BTI": BTI,
        "val": val,
        "results": results,
        "statistics": {
            "mean": overall_mean,
            "variance": overall_variance,
            "n_total": n_total
        }
    }
    return output_data


def main():
    # 遍历当前 ARGS_MODEL 下各个方法文件夹
    for method in METHODS:
        input_dir = os.path.join(BASE_INPUT_DIR, ARGS_MODEL, method)
        if not os.path.isdir(input_dir):
            print(f"目录 {input_dir} 不存在，跳过方法 {method}")
            continue

        # 列举该目录下所有 JSON 文件
        json_files = glob.glob(os.path.join(input_dir, "*.json"))
        if not json_files:
            print(f"目录 {input_dir} 下没有 JSON 文件")
            continue

        # 对应输出目录为 BASE_OUTPUT_DIR/ARGS_MODEL/method
        output_dir = os.path.join(BASE_OUTPUT_DIR, ARGS_MODEL, method)
        os.makedirs(output_dir, exist_ok=True)

        for input_fp in json_files:
            # 文件命名格式：
            # default_method 下的文件为 {BTI}_0.json
            # 其他方法下的文件为 {BTI}_{val}.json
            base_name = os.path.basename(input_fp)
            name_without_ext = base_name[:-5]
            parts = name_without_ext.split("_")
            if len(parts) != 2:
                print(f"文件名 {base_name} 格式不符合预期，跳过")
                continue

            BTI_from_file, val = parts[0], parts[1]

            # 检查 BTI 是否在预设列表中
            if BTI_from_file not in BTIS:
                print(f"文件名 {base_name} 的 BTI {BTI_from_file} 不在预设列表中，跳过")
                continue

            # 读取对应的 prompt 模板（保证 BTI 与模板文件名一致）
            prompt_template_path = os.path.join(PROMPTS_DIR, f"{BTI_from_file}.txt")
            try:
                with open(prompt_template_path, 'r', encoding='utf-8') as f:
                    prompt_template = f.read()
            except Exception as e:
                print(f"读取 prompt 文件 {prompt_template_path} 失败：{e}")
                continue

            # 处理文件；如遇接口调用异常则直接退出程序
            try:
                output_data = process_file(input_fp, prompt_template, EVAL_MODEL, method, BTI_from_file, val)
            except Exception as e:
                print(f"处理文件 {input_fp} 失败：{e}")
                exit(1)

            # 输出文件命名与原输入文件名一致，覆盖写模式
            output_fp = os.path.join(output_dir, base_name)
            try:
                with open(output_fp, 'w', encoding='utf-8') as f:
                    json.dump(output_data, f, indent=4, ensure_ascii=False)
                print(f"处理完成：文件目录模型={ARGS_MODEL}，评估模型={EVAL_MODEL}，方法={method}，文件={base_name} 写入 {output_fp}")
            except Exception as e:
                print(f"写入文件 {output_fp} 失败：{e}")


if __name__ == "__main__":
    main()

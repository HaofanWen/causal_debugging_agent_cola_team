# 文件：repair_engine.py

import os
import json
import argparse

from ollama_models import repair_planner, verify_model

def generate_code_patch(analysis: str, code: str, max_tokens: int = 512) -> str:
    prompt = (
        "Below is a causal-chain analysis of why a code snippet failed:\n\n"
        f"{analysis}\n\n"
        "Original code:\n"
        f"{code}\n\n"
        "Please provide a complete Python patch that fixes the root cause, "
        "and include a short explanation of your changes.\n\n"
        "```python\n# patched code here\n```\n\nExplanation:\n- …"
    )
    resp = repair_planner.client.completion(
        model="ollama/deepseek-coder-v2:16b",
        provider="ollama",
        api_base="http://localhost:11434",
        api_key="ollama",
        messages=[{"role":"user","content":prompt}],
        max_tokens=max_tokens,
        stream=False
    )
    return resp.choices[0].message.content

def generate_text_fix(analysis: str, question: str, max_tokens: int = 256) -> str:
    prompt = (
        "Below is a question and a causal-chain analysis of a previous reasoning attempt. "
        "Please provide only the corrected final answer in English:\n\n"
        "Question:\n"
        f"{question}\n\n"
        "Causal-Chain Analysis:\n"
        f"{analysis}\n\n"
        "Final Answer:"
    )
    resp = verify_model.client.completion(
        model="ollama/llama3.1:8b",
        provider="ollama",
        api_base="http://localhost:11434",
        api_key="ollama",
        messages=[{"role":"user","content":prompt}],
        max_tokens=max_tokens,
        stream=False
    )
    return resp.choices[0].message.content.strip()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Batch repair: generate answer_<task_id>.txt for all GAIA tasks"
    )
    parser.add_argument(
        "--analysis-jsonl", required=True,
        help="Path to causal_outputs.jsonl"
    )
    parser.add_argument(
        "--metadata-jsonl", required=True,
        help="Path to GAIA validation metadata.jsonl"
    )
    parser.add_argument(
        "--code-dir", default="",
        help="Directory where code files (<task_id>.py) are stored"
    )
    parser.add_argument(
        "--output-dir", default="answers",
        help="Directory to write answer_<task_id>.txt files"
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 1) 读取 metadata.jsonl，建立 task_id → metadata 映射
    meta = {}
    with open(args.metadata_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            tid = rec["task_id"]
            meta[tid] = {
                "question": rec.get("Question", "").strip(),
                "file_name": rec.get("file_name", "").strip()
            }

    # 2) 遍历 causal_outputs.jsonl
    with open(args.analysis_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            task_id = rec["task_id"]
            analysis = rec["causal_analysis"]
            info = meta.get(task_id, {})
            question = info.get("question", "")
            file_name = info.get("file_name", "")
            is_code = file_name.lower().endswith(".py")

            # 3) 根据是否有代码文件来选择生成逻辑
            if is_code and args.code_dir:
                code_path = os.path.join(args.code_dir, file_name)
                if os.path.isfile(code_path):
                    code = open(code_path, encoding="utf-8").read()
                    answer_text = generate_code_patch(analysis, code)
                else:
                    print(f"[WARN] Code not found for {task_id}, generating text answer instead.")
                    answer_text = generate_text_fix(analysis, question)
            else:
                answer_text = generate_text_fix(analysis, question)

            # 4) 写入统一的 answer_<task_id>.txt
            out_path = os.path.join(args.output_dir, f"answer_{task_id}.txt")
            with open(out_path, "w", encoding="utf-8") as outf:
                outf.write(answer_text)
            print(f"[OK] Wrote {out_path}")

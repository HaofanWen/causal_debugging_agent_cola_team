# 文件：evaluate_predictions.py

import os
import sys
import re

from datasets import load_dataset
import evaluate

def extract_core_answer(text: str) -> str:
    t = text.replace("\n", " ").strip()
    m = re.search(
        r"(?:the final answer is|final answer[:：]?|answer[:：]?)[\s]*([^\.\n\r]+)",
        t, flags=re.IGNORECASE
    )
    core = m.group(1) if m else t
    return core.strip().strip('"\'')

def normalize(text: str) -> str:
    core = extract_core_answer(text)
    core = core.lower().strip()
    core = re.sub(r"[，。,.!?；;:\-–—\"\'”“‘’（）()]", "", core)
    return core

def main():
    # 1. 加载 GAIA 验证集
    ds = load_dataset(
        "gaia-benchmark/GAIA",
        "2023_all",
        split="validation",
        trust_remote_code=True
    )

    # 2. 加载 SQuAD 风格指标
    squad = evaluate.load("squad")

    sq_preds = []
    sq_refs  = []

    for tid, gold in zip(ds["task_id"], ds["Final answer"]):
        txt = f"./output/answer_{tid}.txt"
        py  = f"./output/patched_{tid}.py"
        if os.path.exists(txt):
            path = txt
        elif os.path.exists(py):
            path = py
        else:
            sys.stderr.write(f"Warning: missing prediction for {tid}\n")
            continue

        raw = open(path, "r", encoding="utf-8", errors="ignore").read().strip()
        p = normalize(raw)
        g = normalize(gold)

        sq_preds.append({"id": tid, "prediction_text": p})
        sq_refs.append({
            "id": tid,
            "answers": {"text": [g], "answer_start": [0]}
        })

    if not sq_preds:
        raise RuntimeError("No predictions found in ./output")

    # 3. 调用 compute
    results = squad.compute(predictions=sq_preds, references=sq_refs)

    # 4. 自动识别 exact / exact_match 字段
    exact_key = "exact"
    if exact_key not in results:
        exact_key = "exact_match" if "exact_match" in results else None

    f1_key = "f1"
    if f1_key not in results:
        # 有的版本可能叫 'f1_score'
        f1_key = "f1_score" if "f1_score" in results else None

    if exact_key is None and f1_key is None:
        print("⚠️ squad.compute 返回字段：", results.keys())
        raise KeyError("Can't find exact or f1 in the results.")

    if exact_key:
        print(f"GAIA Validation Exact Match: {results[exact_key]:.4f}")
    if f1_key:
        print(f"GAIA Validation Token-level F1: {results[f1_key]:.4f}")

if __name__ == "__main__":
    main()

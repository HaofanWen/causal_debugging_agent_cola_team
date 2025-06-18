import os
import json
import gradio as gr
from evaluate_predictions import extract_core_answer

# 文件路径定义
ANSWER_DIR = "./output"
ANALYSIS_FILE = "./causal_outputs.jsonl"

# 预读取所有分析
causal_map = {}
if os.path.exists(ANALYSIS_FILE):
    with open(ANALYSIS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            causal_map[record["task_id"]] = record["causal_analysis"]

def load_answer(task_id):
    path = os.path.join(ANSWER_DIR, f"answer_{task_id}.txt")
    if not os.path.exists(path):
        return "", "❌ 找不到答案文件", "❌ 找不到因果分析"
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read().strip()
    cleaned = extract_core_answer(raw)
    analysis = causal_map.get(task_id, "⚠️ 无对应分析记录")
    return cleaned, raw, analysis

def export_submission():
    answers = []
    for fname in os.listdir(ANSWER_DIR):
        if fname.startswith("answer_") and fname.endswith(".txt"):
            task_id = fname[len("answer_"):-len(".txt")]
            with open(os.path.join(ANSWER_DIR, fname), "r", encoding="utf-8") as f:
                raw = f.read().strip()
            cleaned = extract_core_answer(raw)
            answers.append({"task_id": task_id, "submitted_answer": cleaned})

    submission = {
        "username": "HaofanWen",
        "agent_code": "https://huggingface.co/spaces/HaofanWen/causal_debugging_agent/tree/main",
        "answers": answers
    }

    with open("gaia_submission.json", "w", encoding="utf-8") as f:
        json.dump(submission, f, indent=2, ensure_ascii=False)

    return f"✅ 已导出 {len(answers)} 个答案到 gaia_submission.json"

with gr.Blocks() as demo:
    gr.Markdown("## 🎯 GAIA Agent 调试界面\n输入 task_id 查看答案和因果链分析")

    with gr.Row():
        task_input = gr.Text(label="输入 Task ID（例如 abc123）")
        submit_btn = gr.Button("查找")

    with gr.Row():
        final_answer = gr.Text(label="🌟 提取出的最终答案")
        raw_answer = gr.Text(label="📝 原始答案文本")

    analysis_output = gr.Text(label="📘 因果链分析内容", lines=10)

    submit_btn.click(fn=load_answer, inputs=[task_input],
                     outputs=[final_answer, raw_answer, analysis_output])

    gr.Markdown("---")
    with gr.Row():
        export_button = gr.Button("📥 生成提交 JSON（gaia_submission.json）")
        export_status = gr.Textbox(label="导出状态")

    export_button.click(fn=export_submission, outputs=[export_status])

demo.launch()

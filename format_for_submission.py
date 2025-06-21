import os
import json

script_dir = os.path.abspath(os.path.dirname(__file__))
output_dir = os.path.join(script_dir, 'output_test_0621')
output_jsonl_path = os.path.join(output_dir, 'output.jsonl')

if not os.path.exists(output_dir):
    raise FileNotFoundError(f"❌ output file not exists: {output_dir}")

# Collect all answer_*.txt files
submission_data = []
for filename in os.listdir(output_dir):
    if filename.startswith("answer_") and filename.endswith(".txt"):
        task_id = filename[len("answer_"):-len(".txt")]
        filepath = os.path.join(output_dir, filename)

        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            model_answer = f.read().strip()

        submission_data.append({
            "task_id": task_id,
            "model_answer": model_answer
        })

# write down output.jsonl
with open(output_jsonl_path, "w", encoding="utf-8") as f:
    for item in submission_data:
        json.dump(item, f, ensure_ascii=False)
        f.write("\n")

print(f"✅ Successfully wrote {len(submission_data)} results to {output_jsonl_path}.")

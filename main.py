import os
import json
from langchain_core.messages import HumanMessage
from causal_analyzer import build_graph as build_default_agent
import repair_engine
from repair_engine import generate_code_patch

INPUT_PATH = "bug_data/debug_dataset.jsonl"
# INPUT_PATH = "student_data/questions.json"

OUTPUT_PATH = "output.json"


def load_questions(file_path):
    """Load questions (and metadata) from JSON or JSONL, preserving all fields."""
    entries = []
    if file_path.endswith(".json"):
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            for entry in data:
                q = entry.get("question") or entry.get("Question")
                if q:
                    entries.append(entry)
    elif file_path.endswith(".jsonl"):
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                entry = json.loads(line)
                q = entry.get("question") or entry.get("Question")
                if q:
                    entries.append(entry)
    return entries


def run_agent_on_questions(entries, default_agent, repair_agent):
    """Invoke either the default agent or the repair agent based on presence of bug/error/fix fields."""
    results = []
    for item in entries:
        try:
            # Extract question text
            question_text = item.get("question") or item.get("Question")
            # Determine if this needs repair workflow
            if any(field in item for field in ("bug", "error", "fix")):
                # Combine code and question for the repair agent
                code_snippet = item.get("code", "")
                combined_prompt = f"{question_text}\n\nCode:\n{code_snippet}"
                messages = [HumanMessage(content=combined_prompt)]
                output = repair_agent.invoke({"messages": messages})
            else:
                # Default workflow
                messages = [HumanMessage(content=question_text)]
                output = default_agent.invoke({"messages": messages})

            # Extract and clean the final answer
            answer = output["messages"][-1].content
            submitted = answer[14:] if answer.startswith("FINAL ANSWER: ") else answer
            results.append({
                "task_id": item.get("task_id", ""),
                "submitted_answer": submitted
            })
        except Exception as e:
            results.append({
                "task_id": item.get("task_id", ""),
                "submitted_answer": f"AGENT ERROR: {e}"
            })
    return results


def save_results(results, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    print("Building agents.")
    default_agent = build_default_agent()
    repair_agent  = repair_engine.build_graph()

    print(f"Loading questions from {INPUT_PATH}.")
    entries = load_questions(INPUT_PATH)
    print(f"Running agent on {len(entries)} questions.")

    # Load a few-shot library of repair examples
    examples = []
    with open("bug_data/repair_example.jsonl", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            if record.get("type") == "code_repair":
                examples.append(record)

    # with open("repair_example.jsonl", encoding="utf-8") as f:
    #     examples = [json.loads(line) for line in f if line.get("type") == "code_repair"]

    results = []
    for item in entries:
        task_id = item.get("task_id", "")
        q_type  = item.get("type", "")
        question = item.get("Question") or item.get("question")
        # decide branch
        if q_type == "code_repair":
            # code-repair path
            code   = item.get("code", "")
            patch  = generate_code_patch(
                question=question,
                code=code,
                examples=examples
            )
            submitted = patch
        else:
            # default (causal) path
            msgs = [HumanMessage(content=question)]
            out  = default_agent.invoke({"messages": msgs})
            text = out["messages"][-1].content
            submitted = text.removeprefix("FINAL ANSWER: ").strip()

        results.append({
            "task_id": task_id,
            "submitted_answer": submitted
        })

    print(f"Saving answers to {OUTPUT_PATH}.")
    save_results(results, OUTPUT_PATH)

    print("All done.")


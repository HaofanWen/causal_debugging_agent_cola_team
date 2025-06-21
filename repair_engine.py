import os
import json
import argparse

from ollama_models import repair_planner, verify_model

# k = 5, 2 level 1 questions, 2 level 2 questions, 1 level 3 question.
examples = [
    {
      "Question": "How many at bats did the Yankee with the most walks in the 1977 regular season have that same season?",
      "Answer": "519"
    },
    {
      "Question": "What was the actual enrollment count of the clinical trial on H. pylori in acne vulgaris patients from Jan-May 2018 as listed on the NIH website?",
      "Answer": "90"
    },
    {
      "Question": "When was a picture of St. Thomas Aquinas first added to the Wikipedia page on the Principle of double effect? Answer using the format DD/MM/YYYY.",
      "Answer": "19/02/2009"
    },
    {
        "Question": "According to the USGS, in what year was the American Alligator first found west of Texas (not including Texas)?",
        "Answer": "1954"
    },
    {
        "Question": "Which of the fruits shown in the 2008 painting \"Embroidery from Uzbekistan\" were served as part of the October 1949 breakfast menu for the ocean liner that was later used as a floating prop for the film \"The Last Voyage\"? Give the items as a comma-separated list, ordering them in clockwise order based on their arrangement in the painting starting from the 12 o'clock position. Use the plural form of each fruit.",
        "Answer": "pears, bananas"
    },
]

def build_few_shot_prompt(examples, question, analysis=None):
    prompt = (
        "You are answering GAIA questions. "
        "**ONLY output the final answer exactly as shown—no explanations, no steps.**\n\n"
    )
    for ex in examples:
        prompt += f"Q: {ex['Question']}\nA: {ex['Answer']}\n\n"
    prompt += f"Now answer this question:\nQ: {question}\n"
    if analysis:
        prompt += f"\nCausal analysis:\n{analysis}\n"
    prompt += "A:"
    return prompt

def generate_code_patch(analysis: str, code: str, max_tokens: int = 2048) -> str:
    prompt = (
        "Below is a causal-chain analysis of why a code snippet failed:\n\n"
        f"{analysis}\n\n"
        "Original code (language to be detected):\n"
        "```<auto-detect>\n"
        f"{code}\n"
        "```\n\n"
        "Please do the following:\n"
        "1. Automatically detect the programming language of the original code.\n"
        "2. Provide a complete patched version in the same language,\n"
        "   enclosed in triple backticks tagged with that language (e.g. ```java, ```python, etc.).\n"
        "3. Include a brief explanation of your changes below the code.\n\n"
        "```<same-language-as-above>\n"
        "# patched code here\n"
        "```\n\n"
        "Explanation:\n"
        "- ...\n"
    )
    resp = repair_planner.client.completion(
        model="ollama/deepseek-coder-v2:16b",
        provider="ollama",
        api_base="http://localhost:11434",
        api_key="ollama",
        messages=[{"role":"user","content":prompt}],
        max_tokens=max_tokens,
        temperature=0.0,  # Turn off random sampling → Greedy decoding
        top_p=1.0,  # Guaranteed not to truncate the probability space
        seed=0,  # Fixed random seed
        stream=False
    )
    return resp.choices[0].message.content

def generate_text_fix(analysis: str, question: str, max_tokens: int = 2048) -> str:
    prompt = (
        "Below is a user’s question and a causal-chain analysis of a previous reasoning attempt.\n\n"
        "Question:\n"
        f"{question}\n\n"
        "Causal-Chain Analysis:\n"
        f"{analysis}\n\n"
        "Please do ONLY the following:\n"
        "1. Provide the fully corrected final answer in English.\n"
        "2. Do NOT include any reasoning steps, analysis, or extra commentary.\n"
        "3. If multiple points are needed, list them as short numbered items.\n\n"
        "Examples of desired behavior:\n"
        "Standard Answer: 3\n"
        "Incorrect Prediction: The final answer is 3.\n\n"
        "Standard Answer: 50\n"
        "Incorrect Prediction: This man is 50 years old.\n\n"
        "Final Answer:"
    )
    resp = verify_model.client.completion(
        model="ollama/llama3.1:8b",
        provider="ollama",
        api_base="http://localhost:11434",
        api_key="ollama",
        messages=[{"role":"user","content":prompt}],
        max_tokens=max_tokens,
        temperature=0.0,  # Turn off random sampling → Greedy decoding
        top_p=1.0,  # Guaranteed not to truncate the probability space
        seed=0,  # Fixed random seed
        stream=False
    )
    # resp = verify_model.client.completion(
    #     model="ollama/nous-hermes2-mixtral:latest",
    #     provider="ollama",
    #     api_base="http://localhost:11434",
    #     api_key="ollama",
    #     messages=[{"role": "user", "content": prompt}],
    #     max_tokens=max_tokens,
    #     stream=False
    # )
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

    # 1) Read metadata.jsonl, create task_id → metadata mapping
    meta = {}
    with open(args.metadata_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            tid = rec["task_id"]
            meta[tid] = {
                "question": rec.get("Question", "").strip(),
                "file_name": rec.get("file_name", "").strip()
            }

    # 2) ergodic causal_outputs.jsonl
    with open(args.analysis_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            task_id = rec["task_id"]
            analysis = rec["causal_analysis"]
            info = meta.get(task_id, {})
            question = info.get("question", "")
            file_name = info.get("file_name", "")
            is_code = file_name.lower().endswith(".py")

            # 3) Generation logic is selected based on the availability of code files
            if is_code and args.code_dir:
                code_path = os.path.join(args.code_dir, file_name)
                if os.path.isfile(code_path):
                    code = open(code_path, encoding="utf-8").read()
                    answer_text = generate_code_patch(analysis, code)
                else:
                    print(f"[WARN] Code not found for {task_id}, generating text answer instead.")
                    answer_text = generate_text_fix(analysis, question)
            else:
                # for non-code task, few-shot prompt
                prompt = build_few_shot_prompt(examples, question, analysis)
                resp = verify_model.client.completion(
                    model="ollama/llama3.1:8b",
                    provider="ollama",
                    api_base="http://localhost:11434",
                    api_key="ollama",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=50,
                    temperature=0.0,  # Turn off random sampling → Greedy decoding
                    top_p=1.0,  # Guaranteed not to truncate the probability space
                    seed=0,  # Fixed random seed
                    stream=False
                )
                # resp = verify_model.client.completion(
                #     model="ollama/nous-hermes2-mixtral:latest",
                #     provider="ollama",
                #     api_base="http://localhost:11434",
                #     api_key="ollama",
                #     messages=[{"role": "user", "content": prompt}],
                #     max_tokens=50,
                #     stream=False
                # )
                answer_text = resp.choices[0].message.content.strip()

            # 4) Write to unified answer_<task_id>.txt
            out_path = os.path.join(args.output_dir, f"answer_{task_id}.txt")
            with open(out_path, "w", encoding="utf-8") as outf:
                outf.write(answer_text)
            print(f"[OK] Wrote {out_path}")

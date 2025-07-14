import json
from smolagents import LiteLLMModel
from langchain_core.messages import HumanMessage, AIMessage

repair_planner = LiteLLMModel(
    llm_provider="ollama",
    model_id="ollama/deepseek-coder-v2:16b",
    api_base="http://localhost:11434",
    api_key="ollama"
)

# Load a handful of few-shot examples for code repair
# load few-shot examples for code repair
examples = []
with open("bug_data/repair_example.jsonl", encoding="utf-8") as f:
    for line in f:
        record = json.loads(line)
        if record.get("type") == "code_repair":
            examples.append(record)

def build_repair_prompt(question: str, code: str, examples: list[dict]) -> str:
    """
    Construct a few-shot prompt for the repair planner:
    - Show 2–3 examples of Question+code → Final answer
    - Then present the real Question and code
    - Instruct the model to output full patched code plus a brief explanation.
    """
    # Take the first 2 examples
    few = examples[:3]
    parts = [
        "You are a multi-language code repair assistant (Java, Python, C++, etc.).",
        "For each “Fix the following buggy XXX function.” prompt and its code snippet,",
        "output:\n  1) A complete, compilable patched version in the same language,\n",
        "  2) A brief explanation of your changes.",
        ""
    ]
    # Insert few-shot examples
    for ex in few:
        parts.append("### Example")
        parts.append(f"Q: {ex['Question']}")
        parts.append("```<auto-detect>")
        parts.append(ex["code"])
        parts.append("```")
        parts.append("Patched code:")
        parts.append("```java")  # or the language tag of that example
        parts.append(ex["Final answer"])
        parts.append("```")
        parts.append("")
    # Now the real task
    parts.append("### Now please repair this:")
    parts.append(f"Q: {question}")
    parts.append("```<auto-detect>")
    parts.append(code)
    parts.append("```")
    parts.append("Patched code:")
    parts.append("```<same-language-as-above>")
    parts.append("# your patched code here")
    parts.append("```")
    parts.append("Explanation:")
    parts.append("- …")
    return "\n".join(parts)

def generate_code_patch(question: str, code: str, examples: list[dict]) -> str:
    """
    Build the prompt and call the repair planner.
    """
    prompt = build_repair_prompt(question, code, examples)
    resp = repair_planner.client.completion(
        model="ollama/deepseek-coder-v2:16b",
        provider="ollama",
        api_base="http://localhost:11434",
        api_key="ollama",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=2048,
        temperature=0.0,
        top_p=1.0,
        seed=0,
        stream=False,
    )
    # Return the full content (patched code + explanation)
    return resp.choices[0].message.content

def build_graph():
    """
    Returns a RepairAgent whose .invoke() expects:
       {"messages":[HumanMessage(content=prompt_string)]}
    and returns:
       {"messages":[AIMessage(content=patched_code_and_explanation)]}
    """
    class RepairAgent:
        def __init__(self, max_tokens: int = 2048):
            self.max_tokens = max_tokens

        def invoke(self, inputs: dict) -> dict:
            # 1) extract the raw prompt
            msgs = inputs.get("messages", [])
            if not msgs or not isinstance(msgs[0], HumanMessage):
                raise ValueError("RepairAgent.invoke requires a HumanMessage in inputs['messages']")
            prompt = msgs[0].content

            # 2) here, we assume `prompt` already contains the Question + code + few-shot examples
            #    so we just send it to the planner
            resp = repair_planner.client.completion(
                model="ollama/deepseek-coder-v2:16b",
                provider="ollama",
                api_base="http://localhost:11434",
                api_key="ollama",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=self.max_tokens,
                temperature=0.0,
                top_p=1.0,
                seed=0,
                stream=False,
            )
            patched = resp.choices[0].message.content

            # 3) wrap in an AIMessage so it matches the agent API
            return {"messages": [AIMessage(content=patched)]}

    return RepairAgent()
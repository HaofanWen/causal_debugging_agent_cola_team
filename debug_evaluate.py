import json
import re
import sys
from pathlib import Path
from sacrebleu import corpus_bleu
import javalang

# Paths
GOLD_PATH = Path("bug_data/debug_dataset.jsonl")
PRED_PATH = Path("debug_output.json")
OUT_PATH  = Path("evaluate_debug.json")

def extract_code(text: str) -> str:
    m = re.search(r"```(?:[^\n]*)\n([\s\S]*?)```", text)
    if m:
        return m.group(1).strip()
    return text.strip()

def normalize_whitespace(s: str) -> str:
    return re.sub(r"\s+", "", s)

def load_gold(path: Path) -> dict:
    gold = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            tid  = rec.get("task_id")
            code = rec.get("Final answer", "").strip()
            if tid:
                gold[tid] = code
    return gold

def load_preds(path: Path) -> dict:
    preds = {}
    with open(path, encoding="utf-8") as f:
        data = json.load(f)  # single JSON list
        for rec in data:
            tid = rec.get("task_id")
            ans = rec.get("submitted_answer", "").strip()
            if tid:
                preds[tid] = ans
    return preds

def extract_subtrees(code: str) -> set:
    """
    Wrap the snippet in a dummy class, parse into AST, and collect (node_type, child_types) tuples.
    """
    # 1) Wrap in a minimal class
    wrapped = f"public class Dummy {{\n{code}\n}}"
    try:
        tree = javalang.parse.parse(wrapped)
    except Exception:
        return set()  # parse error → empty

    subs = set()
    def visit(node):
        from javalang.tree import Node
        if not isinstance(node, Node):
            return
        # collect this node type and its direct child node types
        child_types = tuple(
            type(child).__name__
            for _, child in node.filter(lambda x: isinstance(x, Node))
        )
        subs.add((type(node).__name__, child_types))
        # recurse into children
        for child in node.children:
            if isinstance(child, Node):
                visit(child)
            elif isinstance(child, list):
                for c in child:
                    if isinstance(c, Node):
                        visit(c)

    visit(tree)
    return subs

def ast_score(pred_code: str, ref_code: str) -> float:
    """
    AST score = |subtrees(pred) ∩ subtrees(ref)| / |subtrees(ref)|
    """
    pred_subs = extract_subtrees(pred_code)
    ref_subs  = extract_subtrees(ref_code)
    if not ref_subs:
        return 0.0
    overlap = pred_subs & ref_subs
    return len(overlap) / len(ref_subs)

def main():
    if not GOLD_PATH.exists():
        print(f"Error: gold file not found at {GOLD_PATH}", file=sys.stderr)
        sys.exit(1)
    if not PRED_PATH.exists():
        print(f"Error: predictions file not found at {PRED_PATH}", file=sys.stderr)
        sys.exit(1)

    gold_map = load_gold(GOLD_PATH)
    pred_map = load_preds(PRED_PATH)

    total = len(gold_map)
    if total == 0:
        print("No gold records to evaluate.", file=sys.stderr)
        sys.exit(1)

    per_task_em  = {}
    per_task_ast = {}
    hyps = []
    refs = []
    em_count = 0
    ast_sum = 0.0

    for tid, gold_code in gold_map.items():
        pred_text = pred_map.get(tid, "")
        pred_code = extract_code(pred_text)

        # Exact Match
        em = int(normalize_whitespace(gold_code) == normalize_whitespace(pred_code))
        per_task_em[tid] = em
        em_count += em

        # AST Score
        a_score = ast_score(pred_code, gold_code)
        per_task_ast[tid] = round(a_score, 4)
        ast_sum += a_score

        # for BLEU
        hyps.append(pred_code)
        refs.append(gold_code)

    em_score  = em_count / total
    bleu_score = corpus_bleu(hyps, [refs]).score
    avg_ast   = ast_sum / total

    result = {
        "exact_match": round(em_score, 4),
        "bleu":        round(bleu_score, 2),
        "ast":         round(avg_ast, 4),
        "per_task_em":  per_task_em,
        "per_task_ast": per_task_ast
    }

    with open(OUT_PATH, "w", encoding="utf-8") as outf:
        json.dump(result, outf, indent=2, ensure_ascii=False)

    print(f"Evaluated {total} examples")
    print(f"Exact Match = {em_score:.4f}")
    print(f"BLEU        = {bleu_score:.2f}")
    print(f"AST Score   = {avg_ast:.4f}")
    print(f"Results saved to {OUT_PATH}")

if __name__ == "__main__":
    main()

# 文件：ollama_models.py

from smolagents import LiteLLMModel

# ─────── 主因果分析模型 ───────
causal_analyzer = LiteLLMModel(
    llm_provider="ollama",
    model_id="ollama/nous-hermes2-mixtral:latest",
    api_base="http://localhost:11434",
    api_key="ollama"
)

# ─────── 修复补丁生成模型 ───────
repair_planner = LiteLLMModel(
    llm_provider="ollama",
    model_id="ollama/deepseek-coder-v2:16b",
    api_base="http://localhost:11434",
    api_key="ollama"
)

# ─────── 验证与备用建议模型 ───────
verify_model = LiteLLMModel(
    llm_provider="ollama",
    model_id="ollama/llama3.1:8b",
    api_base="http://localhost:11434",
    api_key="ollama"
)

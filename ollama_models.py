from smolagents import LiteLLMModel

# ─────── principal causal analysis model ───────
causal_analyzer = LiteLLMModel(
    llm_provider="ollama",
    model_id="ollama/nous-hermes2-mixtral:latest",
    api_base="http://localhost:11434",
    api_key="ollama"
)

# ─────── fixing patch generation models ───────
repair_planner = LiteLLMModel(
    llm_provider="ollama",
    model_id="ollama/deepseek-coder-v2:16b",
    api_base="http://localhost:11434",
    api_key="ollama"
)

# ─────── validation and alternate recommendation modelling ───────
verify_model = LiteLLMModel(
    llm_provider="ollama",
    model_id="ollama/llama3.1:8b",
    api_base="http://localhost:11434",
    api_key="ollama"
)

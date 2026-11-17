
from contextlib import nullcontext

import pandas as pd
import llm_app.rag_chain as rag_chain


class FakeMLModel:

    def predict(self, customer_df):
        assert isinstance(customer_df, pd.DataFrame)
        return [1]

    def predict_proba(self, customer_df):
        return [[0.30, 0.70]]


def test_llm_pipeline_response_contract(monkeypatch):

    expected_response = """
Prediction Interpretation:
The customer is likely to subscribe.

Recommendation 1:
Contact the customer through a suitable channel.

Recommendation 2:
Share relevant term deposit information.
"""

    # Mock the trained ML model
    monkeypatch.setattr(
        rag_chain,
        "load_ml_model",
        lambda: FakeMLModel()
    )

    # Mock Qwen loading
    monkeypatch.setattr(
        rag_chain,
        "load_llm",
        lambda: ("fake_tokenizer", "fake_model")
    )

    # Mock LLM recommendation generation
    monkeypatch.setattr(
        rag_chain,
        "generate_recommendation",
        lambda *args: {
            "response": expected_response,
            "latency": 0.5,
            "prompt_tokens": 100,
            "generated_tokens": 50
        }
    )

    # Prevent actual MLflow tracking operations
    monkeypatch.setattr(
        rag_chain.mlflow,
        "start_run",
        lambda **kwargs: nullcontext()
    )

    monkeypatch.setattr(
        rag_chain.mlflow,
        "log_params",
        lambda *args, **kwargs: None
    )

    monkeypatch.setattr(
        rag_chain.mlflow,
        "log_metrics",
        lambda *args, **kwargs: None
    )

    monkeypatch.setattr(
        rag_chain.mlflow,
        "log_text",
        lambda *args, **kwargs: None
    )

    customer = {
        "age": 42,
        "job": "management",
        "marital": "married",
        "education": "tertiary",
        "default": "no",
        "balance": 1500,
        "housing": "no",
        "loan": "no",
        "contact": "cellular",
        "day": 15,
        "month": "may",
        "campaign": 1,
        "pdays": -1,
        "previous": 0,
        "poutcome": "unknown"
    }

    result = rag_chain.run_llmops(customer)

    assert isinstance(result, dict)
    assert "response" in result

    response = result["response"]

    assert isinstance(response, str)
    assert len(response.strip()) > 0

    assert "Prediction Interpretation:" in response
    assert "Recommendation 1:" in response
    assert "Recommendation 2:" in response

from llm_app.rag_chain import run_llmops


def test_llm_recommendation_generation():

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

    result = run_llmops(customer)

    assert isinstance(result, dict)
    assert "response" in result

    response = result["response"]

    assert isinstance(response, str)
    assert len(response.strip()) > 0

    assert "Prediction Interpretation:" in response
    assert "Recommendation 1:" in response
    assert "Recommendation 2:" in response

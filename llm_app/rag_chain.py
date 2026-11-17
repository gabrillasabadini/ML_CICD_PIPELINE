
import json
import time
from pathlib import Path

import joblib
import mlflow
import pandas as pd
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM
)


# ==================================================
# PROJECT CONFIGURATION
# ==================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    PROJECT_ROOT / "models" / "bank_marketing_model.joblib"
)

PROMPT_PATH = (
    PROJECT_ROOT / "llm_app" / "prompts" / "system_prompt.txt"
)

MLFLOW_DB = PROJECT_ROOT / "mlflow.db"

LLM_MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

PROMPT_VERSION = "v1"


# ==================================================
# MLFLOW CONFIGURATION
# ==================================================

mlflow.set_tracking_uri(
    f"sqlite:///{MLFLOW_DB.as_posix()}"
)

mlflow.set_experiment(
    "Bank_Marketing_LLMOps"
)


# ==================================================
# LOAD TRAINED ML MODEL
# ==================================================

def load_ml_model():

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            "Trained ML model not found. "
            "Run python ml_pipeline/train.py first."
        )

    print("\nLoading trained ML model...")

    ml_model = joblib.load(MODEL_PATH)

    print("ML model loaded successfully!")

    return ml_model


# ==================================================
# LOAD OPEN-SOURCE LLM
# ==================================================

def load_llm():

    print("\nLoading Qwen LLM...")

    tokenizer = AutoTokenizer.from_pretrained(
        LLM_MODEL_NAME
    )

    model = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL_NAME,
        dtype=torch.float32
    )

    model.eval()

    print("Qwen LLM loaded successfully!")

    return tokenizer, model


# ==================================================
# LOAD SYSTEM PROMPT
# ==================================================

def load_system_prompt():

    if not PROMPT_PATH.exists():

        raise FileNotFoundError(
            "System prompt file not found."
        )

    with open(
        PROMPT_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        return file.read()


# ==================================================
# GENERATE LLM RECOMMENDATIONS
# ==================================================

def generate_recommendation(
    customer_data,
    prediction,
    probability,
    tokenizer,
    model
):

    system_prompt = load_system_prompt()

    prediction_label = (
        "Likely to subscribe"
        if prediction == 1
        else "Unlikely to subscribe"
    )

    customer_context = {

        "customer_information": customer_data,

        "ml_prediction": prediction_label,

        "subscription_probability": round(
            probability,
            4
        )
    }

    user_prompt = (
        "Analyze the following customer context "
        "and provide marketing recommendations.\n\n"
        + json.dumps(
            customer_context,
            indent=4
        )
    )

    messages = [

        {
            "role": "system",
            "content": system_prompt
        },

        {
            "role": "user",
            "content": user_prompt
        }

    ]


    # ----------------------------------------------
    # TOKENIZATION
    # ----------------------------------------------

    input_ids = tokenizer.apply_chat_template(

        messages,

        tokenize=True,

        add_generation_prompt=True,

        return_tensors="pt"

    )

    # Handle BatchEncoding or tensor output
    if hasattr(input_ids, "input_ids"):

        input_ids = input_ids.input_ids

    elif isinstance(input_ids, dict):

        input_ids = input_ids["input_ids"]


    # Validate tokenizer output
    if not isinstance(input_ids, torch.Tensor):

        raise TypeError(
            "Tokenizer did not return a PyTorch tensor."
        )

    prompt_tokens = input_ids.shape[1]

    attention_mask = torch.ones_like(input_ids)


    # ----------------------------------------------
    # LLM RESPONSE GENERATION
    # ----------------------------------------------

    start_time = time.time()

    with torch.no_grad():

        output = model.generate(

            input_ids=input_ids,

            attention_mask=attention_mask,

            max_new_tokens=160,

            do_sample=False,

            pad_token_id=tokenizer.eos_token_id

        )

    latency = time.time() - start_time

    generated_tokens = (
        output.shape[1] - prompt_tokens
    )


    # ----------------------------------------------
    # DECODE RESPONSE
    # ----------------------------------------------

    response = tokenizer.decode(

        output[0][prompt_tokens:],

        skip_special_tokens=True

    ).strip()


    return {

        "response": response,

        "latency": latency,

        "prompt_tokens": prompt_tokens,

        "generated_tokens": generated_tokens

    }


# ==================================================
# END-TO-END LLMOPS PIPELINE
# ==================================================

def run_llmops(customer_data):

    print("\nStarting LLMOps pipeline...")


    # ----------------------------------------------
    # LOAD ML MODEL
    # ----------------------------------------------

    ml_model = load_ml_model()


    # ----------------------------------------------
    # LOAD LLM
    # ----------------------------------------------

    tokenizer, llm = load_llm()


    # ----------------------------------------------
    # STAGE 1: TRADITIONAL ML PREDICTION
    # ----------------------------------------------

    print("\nGenerating ML prediction...")

    # Convert customer dictionary into DataFrame
    customer_df = pd.DataFrame(
        [customer_data]
    )

    # Generate prediction
    prediction = int(
        ml_model.predict(customer_df)[0]
    )

    # Generate subscription probability
    probability = float(
        ml_model.predict_proba(customer_df)[0][1]
    )


    print("\nML PREDICTION RESULTS")
    print("---------------------")

    print(
        "Prediction:",
        "Likely to subscribe"
        if prediction == 1
        else "Unlikely to subscribe"
    )

    print(
        "Subscription probability:",
        round(probability * 100, 2),
        "%"
    )


    # ----------------------------------------------
    # STAGE 2: LLM RECOMMENDATION
    # ----------------------------------------------

    print("\nGenerating LLM recommendations...")

    result = generate_recommendation(

        customer_data,

        prediction,

        probability,

        tokenizer,

        llm

    )


    print("\nLLM RECOMMENDATIONS")
    print("-------------------")

    print(result["response"])


    # ----------------------------------------------
    # STAGE 3: MLFLOW LOGGING
    # ----------------------------------------------

    with mlflow.start_run(
        run_name="Qwen_Customer_Recommendation"
    ):

        # Log parameters
        mlflow.log_params({

            "llm_model": LLM_MODEL_NAME,

            "prompt_version": PROMPT_VERSION,

            "ml_model": "LogisticRegression",

            "task": "Customer_Marketing_Recommendation"

        })


        # Log metrics
        mlflow.log_metrics({

            "ml_prediction": prediction,

            "subscription_probability": probability,

            "latency_seconds": result["latency"],

            "prompt_tokens": result["prompt_tokens"],

            "generated_tokens": result["generated_tokens"]

        })


        # Log generated recommendation
        mlflow.log_text(

            result["response"],

            "llm_recommendation.txt"

        )


        # Log system prompt
        mlflow.log_text(

            load_system_prompt(),

            "system_prompt.txt"

        )


        # Log ML prediction context
        mlflow.log_text(

            json.dumps(

                {
                    "prediction": prediction,

                    "probability": probability,

                    "customer_data": customer_data
                },

                indent=4

            ),

            "ml_prediction_context.json"

        )


    print("\nMLflow LLMOps logging completed!")

    print("\nLLMOps pipeline execution completed!")

    return result


# ==================================================
# SAMPLE CUSTOMER
# ==================================================

if __name__ == "__main__":

    sample_customer = {

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

    run_llmops(sample_customer)

# ML_CICD_PIPELINE
Classic ML (MLflow Projects + Model Validation) and LLM/RAG (Prompt Regression + Quality/Safety Gates)

# End-to-End ML and LLMOps CI/CD Pipeline

## Project Overview

This project demonstrates an end-to-end Machine Learning (ML) and Large Language Model Operations (LLMOps) pipeline using the Bank Marketing dataset.

The project combines a traditional Machine Learning model with an open-source Large Language Model (LLM) to generate actionable customer marketing recommendations.

It also demonstrates automated testing and Continuous Integration (CI) using GitHub Actions.

## Project Objectives

- Build and train a traditional Machine Learning classification model.
- Track ML experiments, parameters, metrics, and artifacts using MLflow.
- Integrate an open-source LLM (Qwen 2.5 1.5B Instruct) with ML predictions.
- Generate customer-specific marketing recommendations using an LLM.
- Track LLMOps experiments, prompt information, response latency, and generated outputs.
- Implement automated ML and LLM pipeline tests using Pytest.
- Configure GitHub Actions to execute automated tests whenever code is pushed or a pull request is created.

## Technology Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-learn |
| Model Serialization | Joblib |
| LLM | Qwen 2.5 1.5B Instruct |
| LLM Framework | Hugging Face Transformers |
| Experiment Tracking | MLflow |
| Testing | Pytest |
| CI/CD | GitHub Actions |
| Version Control | Git and GitHub |

## Dataset

**Bank Marketing Dataset**

Source: UCI Machine Learning Repository  
https://archive.ics.uci.edu/dataset/222/bank+marketing

The dataset contains information about customers contacted during direct marketing campaigns conducted by a Portuguese banking institution.

The objective is to predict whether a customer will subscribe to a term deposit.

- Dataset: `bank-full.csv`
- Total records: 45,211
- Original columns: 17
- Target variable: `y`
- Classification type: Binary classification

The `duration` feature is excluded to prevent data leakage because call duration is only known after the marketing call.

The target variable is mapped as:
- `yes` → 1
- `no` → 0

Place the downloaded dataset in `data/raw/` before running the preprocessing pipeline.

## Project Architecture

The project consists of three major components:

### 1. Traditional Machine Learning Pipeline

The ML pipeline performs:

1. Data loading and validation.
2. Data preprocessing and feature preparation.
3. Training and testing data splitting.
4. Feature scaling and categorical encoding.
5. Logistic Regression model training.
6. Model evaluation using classification metrics.
7. Model serialization using Joblib.
8. Experiment tracking using MLflow.

Evaluation metrics include:
- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC

The trained model and evaluation metrics are saved in the `models/` directory.

### 2. LLMOps Pipeline

The LLMOps component integrates the trained ML model with the Qwen language model.

Workflow:

1. Load the trained ML model.
2. Load the Qwen LLM and tokenizer.
3. Receive customer information.
4. Generate a prediction using the ML model.
5. Calculate the customer's subscription probability.
6. Pass the customer information and ML prediction to the LLM.
7. Generate two practical marketing recommendations.
8. Track the execution and generated response using MLflow.

The LLM uses a system prompt stored in `llm_app/prompts/system_prompt.txt`.

The ML model provides the prediction, while the LLM interprets the prediction alongside the customer information to generate recommendations.

### 3. Continuous Integration Using GitHub Actions

GitHub Actions automates testing whenever code is pushed to the `main` branch or a pull request is created against `main`.

Two independent workflows are configured:

**ML Pipeline CI – `ci_ml.yml`**

- Checks out the repository.
- Sets up Python.
- Installs dependencies.
- Executes ML model and data validation tests.

**LLMOps Regression CI – `ci_llm.yml`**

- Checks out the repository.
- Sets up Python.
- Installs dependencies.
- Executes the mocked LLM pipeline regression test.

The lightweight LLM CI test validates the pipeline response contract without downloading or running the actual Qwen model.

The actual LLM integration test can be executed separately in the local environment.

## Project Structure

```text
ML_CICD_PIPELINE/
│
├── .github/
│   └── workflows/
│       ├── ci_ml.yml
│       └── ci_llm.yml
│
├── ml_pipeline/
│   ├── MLproject
│   ├── conda.yaml
│   ├── preprocess.py
│   ├── train.py
│   └── validate.py
│
├── llm_app/
│   ├── prompts/
│   │   └── system_prompt.txt
│   └── rag_chain.py
│
├── models/
│   ├── bank_marketing_model.joblib
│   └── metrics.json
│
├── tests/
│   ├── test_data_and_model.py
│   ├── test_llm_regression.py
│   └── test_llm_ci.py
│
├── data/
│   └── raw/
│
├── requirements.txt
├── .gitignore
└── README.md
```

## Installation and Setup

### Step 1: Clone the Repository

```bash
git clone https://github.com/gabrillasabadini/ML_CICD_PIPELINE.git
cd ML_CICD_PIPELINE
```

### Step 2: Create a Virtual Environment

```bash
python -m venv myenv
```

Activate the environment on Windows:

```powershell
myenv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Download the Dataset

Download the Bank Marketing dataset from the UCI repository.

Place `bank-full.csv` inside:

```text
data/raw/bank-full.csv
```

## Running the ML Pipeline

Execute the following commands from the project root.

### Step 1: Data Preprocessing

```bash
python ml_pipeline/preprocess.py
```

### Step 2: Model Training

```bash
python ml_pipeline/train.py
```

This trains the classification model, logs the experiment in MLflow, and saves the trained model and metrics.

### Step 3: Model Validation

```bash
python ml_pipeline/validate.py
```

The validation script checks the configured model performance thresholds.

## MLflow Experiment Tracking

MLflow is used to track the experiments and their associated metrics and artifacts.

Start the MLflow UI from the project root:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

Open the following URL in your browser:

http://127.0.0.1:5000

The project uses separate experiment names:

- `Bank_Marketing_ML_CI_CD`
- `Bank_Marketing_LLMOps`

## Running Automated Tests

Run the tests from the project root.

### ML Pipeline Tests

```bash
python -m pytest tests/test_data_and_model.py -v
```

These tests validate:
- Trained model availability.
- Metrics file availability.
- Model prediction and probability generation.
- Required evaluation metrics.

### LLMOps Regression Test

```bash
python -m pytest tests/test_llm_regression.py -v
```

This test executes the actual LLM integration and validates the generated response.

The first execution may download the Qwen model weights and requires sufficient memory and processing resources.

### Lightweight LLM CI Test

```bash
python -m pytest tests/test_llm_ci.py -v
```

This test uses mocked model loading and response generation to validate the LLMOps orchestration and expected response structure without running actual LLM inference.

### Run All Tests

```bash
python -m pytest tests/ -v
```

## GitHub Actions Execution

The GitHub Actions workflows are located in:

```text
.github/workflows/
```

Both workflows are configured to trigger on:

- Push to `main`.
- Pull requests targeting `main`.

To view execution results:

1. Open the GitHub repository.
2. Navigate to the **Actions** tab.
3. Select `ML Pipeline CI`.
4. Select `LLMOps Regression CI`.
5. Review the job execution logs and test results.

A successful workflow indicates that the configured automated tests have passed.

## CI/CD Workflow

The overall project workflow is:

```text
Developer
    |
    v
Local Development
    |
    v
Git Commit and Push
    |
    v
GitHub Repository
    |
    v
GitHub Actions Trigger
    |
    +----------------------+
    |                      |
    v                      v
ML Pipeline CI        LLMOps CI
    |                      |
    v                      v
ML Tests             LLM Regression Test
    |                      |
    +----------+-----------+
               |
               v
       Test Results
               |
               v
       CI Validation
```

**Continuous Integration (CI):** Automated testing and validation are performed on GitHub-hosted runners whenever changes are pushed or pull requests are created.

**Continuous Delivery/Deployment (CD):** Deployment is a planned next stage. The current repository contains the CI workflows; an automated deployment workflow has not yet been configured.

## Future Enhancements

- Configure a Continuous Deployment workflow.
- Deploy the application using Streamlit or another hosting platform.
- Add model performance monitoring.
- Introduce model versioning and approval workflows.
- Add automated failure scenarios to demonstrate CI failure detection and recovery.
- Improve dependency version pinning and reproducibility.



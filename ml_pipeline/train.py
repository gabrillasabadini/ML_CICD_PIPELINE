
import json
import joblib
import mlflow
import mlflow.sklearn

from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

from preprocess import load_and_preprocess_data


# --------------------------------------------------
# PROJECT DIRECTORIES
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODELS_DIR = PROJECT_ROOT / "models"

MLFLOW_DB = PROJECT_ROOT / "mlflow.db"


# --------------------------------------------------
# TRAINING FUNCTION
# --------------------------------------------------

def train_model():

    print("\nStarting Bank Marketing ML pipeline...")

    # Load and preprocess data
    (
        X_train,
        X_test,
        y_train,
        y_test,
        numerical_features,
        categorical_features
    ) = load_and_preprocess_data()

    # Create models directory
    MODELS_DIR.mkdir(parents=True, exist_ok=True)


    # --------------------------------------------------
    # MLFLOW CONFIGURATION
    # --------------------------------------------------

    mlflow.set_tracking_uri(
        f"sqlite:///{MLFLOW_DB.as_posix()}"
    )

    mlflow.set_experiment(
        "Bank_Marketing_ML_CI_CD"
    )


    with mlflow.start_run(
        run_name="Logistic_Regression"
    ):

        print("\nBuilding preprocessing pipeline...")

        # Numerical preprocessing
        numerical_transformer = StandardScaler()

        # Categorical preprocessing
        categorical_transformer = OneHotEncoder(
            handle_unknown="ignore"
        )

        # Combine preprocessing
        preprocessor = ColumnTransformer(
            transformers=[
                (
                    "num",
                    numerical_transformer,
                    numerical_features
                ),
                (
                    "cat",
                    categorical_transformer,
                    categorical_features
                )
            ]
        )

        # Complete ML pipeline
        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=2000,
                        class_weight="balanced",
                        random_state=42
                    )
                )
            ]
        )


        # --------------------------------------------------
        # MODEL TRAINING
        # --------------------------------------------------

        print("\nTraining Logistic Regression model...")

        pipeline.fit(X_train, y_train)

        print("Model training completed!")


        # --------------------------------------------------
        # MODEL PREDICTION
        # --------------------------------------------------

        predictions = pipeline.predict(X_test)

        probabilities = pipeline.predict_proba(X_test)[:, 1]


        # --------------------------------------------------
        # MODEL EVALUATION
        # --------------------------------------------------

        metrics = {
            "accuracy": float(
                accuracy_score(y_test, predictions)
            ),

            "precision": float(
                precision_score(
                    y_test,
                    predictions,
                    zero_division=0
                )
            ),

            "recall": float(
                recall_score(
                    y_test,
                    predictions,
                    zero_division=0
                )
            ),

            "f1_score": float(
                f1_score(
                    y_test,
                    predictions,
                    zero_division=0
                )
            ),

            "roc_auc": float(
                roc_auc_score(
                    y_test,
                    probabilities
                )
            )
        }


        print("\nMODEL EVALUATION RESULTS")
        print("------------------------")

        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")


        # --------------------------------------------------
        # SAVE TRAINED MODEL
        # --------------------------------------------------

        model_path = (
            MODELS_DIR / "bank_marketing_model.joblib"
        )

        joblib.dump(
            pipeline,
            model_path
        )

        print("\nTrained model saved successfully!")
        print("Model path:", model_path)


        # --------------------------------------------------
        # MLFLOW PARAMETER LOGGING
        # --------------------------------------------------

        mlflow.log_params({

            "model": "LogisticRegression",

            "max_iter": 2000,

            "class_weight": "balanced",

            "random_state": 42,

            "test_size": 0.20,

            "excluded_feature": "duration",

            "numerical_features": len(numerical_features),

            "categorical_features": len(categorical_features)

        })


        # --------------------------------------------------
        # MLFLOW METRIC LOGGING
        # --------------------------------------------------

        mlflow.log_metrics(metrics)


        # --------------------------------------------------
        # MLFLOW MODEL LOGGING
        # --------------------------------------------------

        mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path="bank_marketing_model"
        )


        # --------------------------------------------------
        # SAVE METRICS AS JSON
        # --------------------------------------------------

        metrics_path = MODELS_DIR / "metrics.json"

        with open(
            metrics_path,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                metrics,
                file,
                indent=4
            )


        # Log metrics file
        mlflow.log_artifact(
            str(metrics_path)
        )


        # --------------------------------------------------
        # COMPLETION
        # --------------------------------------------------

        print("\nML PIPELINE EXECUTION COMPLETED!")
        print("--------------------------------")

        print("Trained model:", model_path)

        print("Metrics file:", metrics_path)

        print("MLflow experiment: Bank_Marketing_ML_CI_CD")

        print("\nAll artifacts saved successfully!")


# --------------------------------------------------
# MAIN EXECUTION
# --------------------------------------------------

if __name__ == "__main__":

    train_model()

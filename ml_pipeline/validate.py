
import json
import sys

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

METRICS_FILE = PROJECT_ROOT / "models" / "metrics.json"


# Model quality thresholds
THRESHOLDS = {
    "recall": 0.50,
    "f1_score": 0.30
}


def validate_model():

    print("\nStarting model quality validation...")

    # Check whether metrics exist
    if not METRICS_FILE.exists():

        print("ERROR: Model metrics file not found.")
        print("Please execute train.py first.")

        return False

    # Load model metrics
    with open(METRICS_FILE, "r") as file:
        metrics = json.load(file)

    validation_passed = True

    print("\nMODEL QUALITY GATE")
    print("------------------")

    # Validate each metric
    for metric, threshold in THRESHOLDS.items():

        actual_value = metrics.get(metric)

        if actual_value is None:

            print(f"FAIL: {metric} is missing.")
            validation_passed = False
            continue

        if actual_value >= threshold:

            print(
                f"PASS: {metric} = {actual_value:.4f} "
                f"(minimum = {threshold})"
            )

        else:

            print(
                f"FAIL: {metric} = {actual_value:.4f} "
                f"(minimum = {threshold})"
            )

            validation_passed = False

    print("\nVALIDATION RESULT")
    print("-----------------")

    if validation_passed:

        print("MODEL VALIDATION PASSED!")
        print("Model meets the required quality thresholds.")

        return True

    else:

        print("MODEL VALIDATION FAILED!")
        print("Model does not meet the required quality thresholds.")

        return False


if __name__ == "__main__":

    success = validate_model()

    if not success:
        sys.exit(1)

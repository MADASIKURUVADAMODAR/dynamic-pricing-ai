from pathlib import Path
import argparse
import joblib
import sklearn


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect serialized pricing model metadata")
    parser.add_argument(
        "--model-path",
        default=str(Path(__file__).resolve().parent / "pricing_model.pkl"),
        help="Path to pricing_model.pkl",
    )
    args = parser.parse_args()

    model_path = Path(args.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    artifact = joblib.load(model_path)
    metadata = artifact.get("metadata", {})

    print("Model file:", model_path)
    print("Runtime sklearn:", sklearn.__version__)
    print("Trained sklearn:", metadata.get("sklearn_version", "unknown"))
    print("Trained python:", metadata.get("python_version", "unknown"))
    print("Model class:", metadata.get("model_class", type(artifact.get("model")).__name__))
    print("Features:", len(artifact.get("features", [])))


if __name__ == "__main__":
    main()

"""
SageMaker Training Script for RF-DETR (COCO format).

Expects the SageMaker "train" channel to contain a dataset in RF-DETR layout
(as produced by split/coco_split.py):

  <channel>/
    train/
      _annotations.coco.json
      image1.jpg
      ...
    valid/
      _annotations.coco.json
      ...
    test/
      _annotations.coco.json
      ...

If the channel instead uses the legacy layout (train/images/ + annotations/annotations.json),
the script will convert it to the above layout before training.
"""

import argparse
import json
import os
import shutil
from datetime import datetime
from pathlib import Path

import mlflow


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    # Core training hyperparameters
    parser.add_argument("--model-size", type=str, default="base", choices=["base", "medium", "large"])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum-steps", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)

    # Optional: continue training / initialization
    parser.add_argument("--resume", type=str, default=None, help="Path to a checkpoint.pth to resume from.")
    parser.add_argument(
        "--pretrain-weights",
        type=str,
        default=None,
        help="Path to a .pth weights file used to initialize training.",
    )

    # SageMaker directories/channels
    parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN"))
    parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
    parser.add_argument(
        "--output-data-dir",
        type=str,
        default=os.environ.get("SM_OUTPUT_DATA_DIR", "/opt/ml/output/data"),
    )

    # MLflow tracking (results, metrics, artifacts)
    parser.add_argument(
        "--mlflow-tracking-uri",
        type=str,
        default="databricks",
        help="MLflow server URI for results/metrics/artifacts",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="/Users/lokeshwaran@opendatafabric.com/acnedoubledetection",
        help="MLflow experiment name/path",
    )

    return parser.parse_args()


def _safe_symlink_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        os.symlink(src, dst)
    except Exception:
        shutil.copy2(src, dst)


def _is_rfdetr_layout(input_root: Path) -> bool:
    """True if input already has train/valid/test with _annotations.coco.json in each."""
    for split in ("train", "valid"):
        ann = input_root / split / "_annotations.coco.json"
        if not ann.exists():
            return False
    return True


def prepare_rfdetr_dataset(input_root: Path, work_root: Path) -> Path:
    """
    Convert the repo's COCO split layout (train/val/test + images/ + annotations/)
    into RF-DETR's expected layout (train/valid/test with images beside _annotations.coco.json).
    Uses symlinks when possible to avoid copying large image sets.
    """
    if not input_root.exists():
        raise FileNotFoundError(f"Train channel path does not exist: {input_root}")

    work_root.mkdir(parents=True, exist_ok=True)
    dataset_dir = work_root / "rfdetr_dataset"
    if dataset_dir.exists():
        shutil.rmtree(dataset_dir)
    dataset_dir.mkdir(parents=True, exist_ok=True)

    split_map = {
        "train": ("train", "train"),
        "val": ("val", "valid"),
        "valid": ("valid", "valid"),
        
    }

    # Prefer val -> valid, but if valid exists already, use it.
    chosen_splits: dict[str, str] = {"train": "train"}
    if (input_root / "valid").exists():
        chosen_splits["valid"] = "valid"
    else:
        chosen_splits["valid"] = "val"

    for out_split, in_split in chosen_splits.items():
        in_split_dir = input_root / in_split
        in_images_dir = in_split_dir / "images"
        in_ann_path = in_split_dir / "annotations" / "annotations.json"

        if not in_split_dir.exists():
            raise FileNotFoundError(f"Missing split directory: {in_split_dir}")
        if not in_images_dir.exists():
            raise FileNotFoundError(f"Missing images directory: {in_images_dir}")
        if not in_ann_path.exists():
            raise FileNotFoundError(f"Missing annotations file: {in_ann_path}")

        out_split_dir = dataset_dir / out_split
        out_split_dir.mkdir(parents=True, exist_ok=True)

        # Copy annotations to RF-DETR expected filename
        out_ann_path = out_split_dir / "_annotations.coco.json"
        shutil.copy2(in_ann_path, out_ann_path)

        # Link/copy images into split root (RF-DETR expects them beside the JSON)
        for img_path in in_images_dir.iterdir():
            if not img_path.is_file():
                continue
            _safe_symlink_or_copy(img_path, out_split_dir / img_path.name)

    return dataset_dir


def _collect_metrics_from_output(output_dir: Path) -> dict:
    """Try to extract metrics from RF-DETR output_dir (e.g. CSV or log files)."""
    metrics: dict = {}
    # RF-DETR may write results CSV or similar; adapt keys if your version differs
    for csv_name in ("results.csv", "training_log.csv", "metrics.csv"):
        csv_path = output_dir / csv_name
        if csv_path.exists():
            try:
                import csv as csv_module
                with csv_path.open("r") as f:
                    reader = csv_module.DictReader(f)
                    rows = list(reader)
                    if rows:
                        last = rows[-1]
                        for k, v in last.items():
                            try:
                                metrics[k.strip()] = float(v)
                            except (ValueError, TypeError):
                                pass
                break
            except Exception:
                pass
    return metrics


def train_rfdetr(args: argparse.Namespace) -> None:
    print("=" * 70)
    print("RF-DETR Training on SageMaker")
    print("=" * 70)

    train_channel = Path(args.train) if args.train else None
    if not train_channel:
        raise ValueError("SM_CHANNEL_TRAIN (or --train) is not set.")

    model_dir = Path(args.model_dir)
    output_dir = Path(args.output_data_dir) / "rfdetr"
    model_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nTrain channel: {train_channel}")
    print(f"Output dir  : {output_dir}")
    print(f"Model dir   : {model_dir}")

    # Use channel as dataset_dir if already in RF-DETR layout; otherwise convert
    if _is_rfdetr_layout(train_channel):
        dataset_dir = train_channel
        print("\nDataset already in RF-DETR layout (train/valid/test + _annotations.coco.json).")
    else:
        dataset_dir = prepare_rfdetr_dataset(train_channel, output_dir)
        print(f"\nPrepared RF-DETR dataset_dir: {dataset_dir}")
    print("\nDataset summary:")
    for split in ["train", "valid"]:
        split_dir = dataset_dir / split
        ann = split_dir / "_annotations.coco.json"
        n_imgs = sum(1 for p in split_dir.iterdir() if p.is_file() and p.name != "_annotations.coco.json")
        n_ann = 0
        try:
            with ann.open("r") as f:
                coco = json.load(f)
            n_ann = len(coco.get("annotations", []))
        except Exception:
            pass
        print(f"  - {split:5s}: {n_imgs} images, {n_ann} annotations")

    # MLflow setup
    if args.mlflow_tracking_uri:
        mlflow.set_tracking_uri(args.mlflow_tracking_uri)
        mlflow.set_experiment(args.experiment_name)
        print(f"\nMLflow tracking: {args.mlflow_tracking_uri}")
        print(f"Experiment: {args.experiment_name}")

    params = {
        "model_size": args.model_size,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "grad_accum_steps": args.grad_accum_steps,
        "lr": args.lr,
        "dataset_dir": str(dataset_dir),
    }
    if args.resume:
        params["resume"] = args.resume
    if args.pretrain_weights:
        params["pretrain_weights"] = args.pretrain_weights

    run_name = f"rfdetr-{args.model_size}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(params)

        # Import here so the script can still print useful errors if dependency import fails
        from rfdetr import RFDETRSegSmall, RFDETRSegMedium, RFDETRSegLarge  # type: ignore

        model = {
            "base": RFDETRSegSmall,
            "medium": RFDETRSegMedium,
            "large": RFDETRSegLarge,
        }[args.model_size]()

        train_kwargs = dict(
            dataset_dir=str(dataset_dir),
            epochs=int(args.epochs),
            batch_size=int(args.batch_size),
            grad_accum_steps=int(args.grad_accum_steps),
            lr=float(args.lr),
            output_dir=str(output_dir),
        )
        if args.resume:
            train_kwargs["resume"] = args.resume
        if args.pretrain_weights:
            train_kwargs["pretrain_weights"] = args.pretrain_weights

        print("\nTraining parameters:")
        for k, v in train_kwargs.items():
            print(f"  {k}: {v}")

        print("\nStarting training...")
        model.train(**train_kwargs)

        # Collect metrics from output_dir if RF-DETR wrote any
        metrics = _collect_metrics_from_output(output_dir)
        if metrics:
            mlflow.log_metrics(metrics)
            print("\n" + "=" * 70)
            print("Training results (from output):")
            print("=" * 70)
            for k, v in metrics.items():
                if isinstance(v, float):
                    print(f"  {k}: {v:.4f}")
                else:
                    print(f"  {k}: {v}")
            print("=" * 70)

        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        training_params = {
            **params,
            "timestamp": timestamp,
            "run_id": run_name,
        }
        metrics_with_params = {
            "metrics": metrics,
            "parameters": training_params,
            "timestamp": timestamp,
        }
        metrics_file = output_dir / "metrics.json"
        metrics_file.write_text(json.dumps(metrics_with_params, indent=2))
        mlflow.log_artifact(str(metrics_file), artifact_path="results")

        # Log any result files (plots, CSV) from output_dir
        for ext in ("*.png", "*.csv", "*.json"):
            for f in output_dir.glob(ext):
                if f.name != "metrics.json":
                    mlflow.log_artifact(str(f), artifact_path="results")

        # RF-DETR convention: best checkpoint saved as checkpoint_best_total.pth
        best_ckpt = output_dir / "checkpoint_best_total.pth"
        last_ckpt = output_dir / "checkpoint.pth"

        if best_ckpt.exists():
            shutil.copy2(best_ckpt, model_dir / "checkpoint_best_total.pth")
            shutil.copy2(best_ckpt, model_dir / "model.pth")
            mlflow.log_artifact(str(best_ckpt), artifact_path="model")
            print(f"\n✓ Saved best checkpoint to: {model_dir / 'checkpoint_best_total.pth'}")
        elif last_ckpt.exists():
            shutil.copy2(last_ckpt, model_dir / "checkpoint.pth")
            shutil.copy2(last_ckpt, model_dir / "model.pth")
            mlflow.log_artifact(str(last_ckpt), artifact_path="model")
            print(f"\n✓ Saved last checkpoint to: {model_dir / 'checkpoint.pth'}")
        else:
            print("\n[WARNING] No checkpoint files found in output_dir.")

    if args.mlflow_tracking_uri:
        print(f"\nResults (metrics, artifacts, model): {args.mlflow_tracking_uri}")
    else:
        print("\nResults (metrics, artifacts, model): logged to MLflow (local ./mlruns)")
    print("\n✓ Training completed.")


if __name__ == "__main__":
    train_rfdetr(parse_args())

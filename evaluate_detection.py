from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean
from typing import Any


COUNT_FIELDS = ("person", "car", "bicycle", "motorcycle", "bus", "truck")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate count-level detection performance against manual labels.")
    parser.add_argument("--predictions", default="outputs/detections_per_second.jsonl")
    parser.add_argument("--ground-truth", default="evaluation/manual_counts.csv")
    parser.add_argument("--make-template", action="store_true", help="Create a CSV template from prediction keys.")
    parser.add_argument("--limit", type=int, default=20, help="Rows to include when making a template.")
    return parser.parse_args()


def read_predictions(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def key_for(record: dict[str, Any]) -> str:
    if record.get("video_second") not in (None, ""):
        return f"second:{record['video_second']}"
    return f"frame:{record.get('frame_index')}"


def counts_for(record: dict[str, Any]) -> dict[str, int]:
    counts = record.get("counts", record)
    return {name: int(float(counts.get(name, 0) or 0)) for name in COUNT_FIELDS}


def make_template(predictions: list[dict[str, Any]], output_path: Path, limit: int) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["key", "video_second", "frame_index", *COUNT_FIELDS, "notes"]
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for record in predictions[:limit]:
            writer.writerow(
                {
                    "key": key_for(record),
                    "video_second": record.get("video_second", ""),
                    "frame_index": record.get("frame_index", ""),
                    **{name: "" for name in COUNT_FIELDS},
                    "notes": "fill manual ground-truth counts",
                }
            )


def read_ground_truth(path: Path) -> dict[str, dict[str, int]]:
    truth: dict[str, dict[str, int]] = {}
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = row.get("key") or (f"second:{row['video_second']}" if row.get("video_second") else f"frame:{row.get('frame_index')}")
            if not key:
                continue
            truth[key] = {name: int(float(row.get(name, 0) or 0)) for name in COUNT_FIELDS}
    return truth


def evaluate(predictions: list[dict[str, Any]], truth: dict[str, dict[str, int]]) -> dict[str, Any]:
    matched = [(record, truth[key_for(record)]) for record in predictions if key_for(record) in truth]
    if not matched:
        raise RuntimeError("No prediction rows matched the ground-truth CSV keys.")

    per_class: dict[str, dict[str, Any]] = {}
    for name in COUNT_FIELDS:
        abs_errors: list[int] = []
        exact_matches = 0
        total_true = 0
        total_pred = 0
        total_overlap = 0
        for record, gt_counts in matched:
            pred = counts_for(record)[name]
            true = gt_counts[name]
            abs_errors.append(abs(pred - true))
            exact_matches += int(pred == true)
            total_true += true
            total_pred += pred
            total_overlap += min(pred, true)
        per_class[name] = {
            "mae": round(mean(abs_errors), 3),
            "exact_count_rate": round(exact_matches / len(matched), 3),
            "count_precision_proxy": round(total_overlap / total_pred, 3) if total_pred else None,
            "count_recall_proxy": round(total_overlap / total_true, 3) if total_true else None,
            "total_pred": total_pred,
            "total_true": total_true,
        }

    all_exact = 0
    for record, gt_counts in matched:
        pred_counts = counts_for(record)
        all_exact += int(all(pred_counts[name] == gt_counts[name] for name in COUNT_FIELDS))

    return {
        "matched_rows": len(matched),
        "all_class_exact_rate": round(all_exact / len(matched), 3),
        "per_class": per_class,
    }


def main() -> None:
    args = parse_args()
    predictions = read_predictions(Path(args.predictions))
    gt_path = Path(args.ground_truth)

    if args.make_template:
        make_template(predictions, gt_path, args.limit)
        print(f"Wrote manual labeling template: {gt_path}")
        return

    metrics = evaluate(predictions, read_ground_truth(gt_path))
    output_path = gt_path.parent / "detection_metrics.json"
    output_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"Saved metrics: {output_path}")


if __name__ == "__main__":
    main()

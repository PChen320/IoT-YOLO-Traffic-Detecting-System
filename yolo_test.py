from __future__ import annotations

import argparse
import json
import os
import time
import urllib.request
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

YOLO_CONFIG_DIR = Path.cwd() / "Ultralytics"
YOLO_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("YOLO_CONFIG_DIR", str(YOLO_CONFIG_DIR))

import cv2
from ultralytics import YOLO


SAMPLE_VIDEO_URL = "https://raw.githubusercontent.com/DeGirum/PySDKExamples/main/images/Traffic2.mp4"
CORE_CLASSES = ("person", "car", "bicycle")
OPTIONAL_CLASSES = ("motorcycle", "bus", "truck")
DEFAULT_CLASSES = CORE_CLASSES + OPTIONAL_CLASSES
SIGNAL_STATES = {"red", "green"}
COCO_CLASS_IDS = {
    "person": 0,
    "bicycle": 1,
    "car": 2,
    "motorcycle": 3,
    "bus": 5,
    "truck": 7,
}


@dataclass(frozen=True)
class FrameMeta:
    intersection_id: str
    timestamp: float
    frame_index: int
    video_timestamp_sec: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run YOLOv8 traffic video detection.")
    parser.add_argument("--source", default="data/traffic2.mp4", help="Input video path.")
    parser.add_argument("--sample-url", default=SAMPLE_VIDEO_URL, help="URL used when --source is missing.")
    parser.add_argument("--weights", default="yolov8n.pt", help="YOLO weights path/name.")
    parser.add_argument("--intersection-id", default="A01", help="Intersection identifier.")
    parser.add_argument("--output-dir", default="outputs", help="Directory for video and JSON outputs.")
    parser.add_argument("--conf", type=float, default=0.25, help="YOLO confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.7, help="YOLO IoU threshold.")
    parser.add_argument("--device", default="cpu", help="Inference device, e.g. cpu, 0, cuda:0.")
    parser.add_argument("--max-frames", type=int, default=300, help="0 means process the whole video.")
    parser.add_argument("--frame-stride", type=int, default=1, help="Run inference every N frames.")
    parser.add_argument("--download-sample", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-video", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-screenshots", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--save-frame-json", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--include-detections", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--crosswalk-roi",
        default="0.30,0.55,0.78,0.95",
        help="Normalized crosswalk ROI as x1,y1,x2,y2.",
    )
    parser.add_argument("--signal-mode", choices=("simulated", "manual"), default="simulated")
    parser.add_argument("--signal-state", choices=("red", "green"), default="red", help="Used when --signal-mode manual.")
    parser.add_argument("--red-seconds", type=float, default=15.0)
    parser.add_argument("--green-seconds", type=float, default=15.0)
    parser.add_argument("--signal-offset", type=float, default=0.0)
    return parser.parse_args()


def empty_counts(classes: Iterable[str] = DEFAULT_CLASSES) -> dict[str, int]:
    return {name: 0 for name in classes}


def counts_from_yolo_result(result: Any, classes: Iterable[str] = DEFAULT_CLASSES) -> dict[str, int]:
    counts = empty_counts(classes)
    boxes = getattr(result, "boxes", None)
    if boxes is None or getattr(boxes, "cls", None) is None:
        return counts

    names = getattr(result, "names", {}) or {}
    class_ids = boxes.cls.detach().cpu().numpy().astype(int).tolist()
    for class_id in class_ids:
        class_name = names.get(class_id, str(class_id))
        if class_name in counts:
            counts[class_name] += 1
    return counts


def detections_from_yolo_result(result: Any, classes: Iterable[str] = DEFAULT_CLASSES) -> list[dict[str, Any]]:
    allowed = set(classes)
    boxes = getattr(result, "boxes", None)
    if boxes is None or getattr(boxes, "cls", None) is None:
        return []

    names = getattr(result, "names", {}) or {}
    class_ids = boxes.cls.detach().cpu().numpy().astype(int).tolist()
    confidences = boxes.conf.detach().cpu().numpy().tolist()
    xyxy_boxes = boxes.xyxy.detach().cpu().numpy().tolist()

    detections: list[dict[str, Any]] = []
    for class_id, confidence, xyxy in zip(class_ids, confidences, xyxy_boxes):
        class_name = names.get(class_id, str(class_id))
        if class_name not in allowed:
            continue
        x1, y1, x2, y2 = [round(float(v), 2) for v in xyxy]
        detections.append(
            {
                "class": class_name,
                "confidence": round(float(confidence), 4),
                "bbox_xyxy": [x1, y1, x2, y2],
                "center_xy": [round((x1 + x2) / 2, 2), round((y1 + y2) / 2, 2)],
            }
        )
    return detections


def build_frame_payload(
    result: Any,
    meta: FrameMeta,
    classes: Iterable[str] = DEFAULT_CLASSES,
    include_detections: bool = False,
) -> dict[str, Any]:
    payload = {
        "intersection_id": meta.intersection_id,
        "timestamp": round(meta.timestamp, 3),
        "frame_index": meta.frame_index,
        "video_timestamp_sec": round(meta.video_timestamp_sec, 3),
        "counts": counts_from_yolo_result(result, classes),
    }
    if include_detections:
        payload["detections"] = detections_from_yolo_result(result, classes)
    return payload


def aggregate_by_second(records: Iterable[dict[str, Any]], classes: Iterable[str] = DEFAULT_CLASSES) -> list[dict[str, Any]]:
    class_names = tuple(classes)
    buckets: dict[int, dict[str, Any]] = {}
    for record in records:
        second = int(float(record.get("video_timestamp_sec", 0.0)))
        counts = record.get("counts", {})
        bucket = buckets.setdefault(
            second,
            {
                "intersection_id": record.get("intersection_id", "A01"),
                "timestamp": record.get("timestamp"),
                "video_second": second,
                "frame_start": record.get("frame_index"),
                "frame_end": record.get("frame_index"),
                "aggregation": "max_count_per_second",
                "counts": empty_counts(class_names),
            },
        )
        bucket["frame_end"] = record.get("frame_index")
        for name in class_names:
            bucket["counts"][name] = max(int(bucket["counts"].get(name, 0)), int(counts.get(name, 0)))
    return [buckets[key] for key in sorted(buckets)]


def summarize_records(records: Iterable[dict[str, Any]]) -> dict[str, int]:
    summary: Counter[str] = Counter()
    for record in records:
        for name, count in record.get("counts", {}).items():
            summary[name] = max(summary[name], int(count))
    return dict(summary)


def parse_roi(value: str) -> tuple[float, float, float, float]:
    parts = [float(part.strip()) for part in value.split(",")]
    if len(parts) != 4:
        raise ValueError("ROI must be x1,y1,x2,y2")
    x1, y1, x2, y2 = parts
    if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
        raise ValueError("ROI values must be normalized floats between 0 and 1")
    return x1, y1, x2, y2


def roi_to_pixels(roi: tuple[float, float, float, float], width: int, height: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = roi
    return int(x1 * width), int(y1 * height), int(x2 * width), int(y2 * height)


def signal_state(
    mode: str,
    manual_state: str,
    video_timestamp_sec: float,
    red_seconds: float,
    green_seconds: float,
    offset_seconds: float,
) -> str:
    if mode == "manual":
        if manual_state not in SIGNAL_STATES:
            raise ValueError(f"Unsupported signal state: {manual_state}")
        return manual_state
    red_seconds = max(0.1, red_seconds)
    green_seconds = max(0.1, green_seconds)
    cycle_position = (video_timestamp_sec + offset_seconds) % (red_seconds + green_seconds)
    return "red" if cycle_position < red_seconds else "green"


def point_in_roi(point_xy: list[float], roi: tuple[float, float, float, float], width: int, height: int) -> bool:
    px1, py1, px2, py2 = roi_to_pixels(roi, width, height)
    x, y = point_xy
    return px1 <= x <= px2 and py1 <= y <= py2


def evaluate_crosswalk_events(
    payload: dict[str, Any],
    roi: tuple[float, float, float, float],
    frame_width: int,
    frame_height: int,
    current_signal_state: str,
) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for detection in payload.get("detections", []):
        if detection.get("class") != "person":
            continue
        if not point_in_roi(detection.get("center_xy", [0, 0]), roi, frame_width, frame_height):
            continue
        is_alert = current_signal_state == "red"
        events.append(
            {
                "event_type": "pedestrian_in_crosswalk_during_red" if is_alert else "pedestrian_in_crosswalk",
                "severity": "high" if is_alert else "info",
                "alert": is_alert,
                "intersection_id": payload.get("intersection_id", "A01"),
                "timestamp": payload.get("timestamp"),
                "frame_index": payload.get("frame_index"),
                "video_timestamp_sec": payload.get("video_timestamp_sec"),
                "signal_state": current_signal_state,
                "roi": [round(v, 4) for v in roi],
                "object": detection,
            }
        )
    return events


def download_file(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    print(f"Downloading sample video: {url}")
    urllib.request.urlretrieve(url, tmp_path)
    tmp_path.replace(path)


def ensure_video(source: Path, should_download: bool, sample_url: str) -> None:
    if source.exists():
        return
    if not should_download:
        raise FileNotFoundError(f"Video not found: {source}")
    download_file(sample_url, source)


def open_writer(output_path: Path, fps: float, width: int, height: int) -> cv2.VideoWriter:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to create output video: {output_path}")
    return writer


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def draw_counts(frame: Any, payload: dict[str, Any]) -> Any:
    counts = payload["counts"]
    text = " | ".join(f"{name}:{count}" for name, count in counts.items() if count)
    if not text:
        text = "no target classes"
    cv2.rectangle(frame, (8, 8), (8 + min(760, 18 * len(text)), 42), (0, 0, 0), -1)
    cv2.putText(frame, text, (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    return frame


def draw_safety_overlay(
    frame: Any,
    roi: tuple[float, float, float, float],
    signal: str,
    alert_count: int,
) -> Any:
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = roi_to_pixels(roi, width, height)
    color = (0, 0, 255) if signal == "red" else (0, 180, 0)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    label = f"crosswalk ROI | signal:{signal}"
    if alert_count:
        label += f" | alerts:{alert_count}"
    cv2.rectangle(frame, (x1, max(0, y1 - 28)), (min(width - 1, x1 + 360), y1), color, -1)
    cv2.putText(frame, label, (x1 + 8, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2)
    return frame


def main() -> None:
    args = parse_args()
    source = Path(args.source)
    output_dir = Path(args.output_dir)
    ensure_video(source, args.download_sample, args.sample_url)
    crosswalk_roi = parse_roi(args.crosswalk_roi)

    model = YOLO(args.weights)
    target_class_ids = [COCO_CLASS_IDS[name] for name in DEFAULT_CLASSES]

    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {source}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_fps = max(1.0, fps / max(1, args.frame_stride))

    video_path = output_dir / "detected_traffic.mp4"
    writer = open_writer(video_path, output_fps, width, height) if args.save_video else None
    frame_records: list[dict[str, Any]] = []
    alert_records: list[dict[str, Any]] = []
    run_start = time.time()
    processed = 0
    screenshots = 0

    try:
        frame_index = 0
        last_annotated = None
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if args.max_frames and frame_index >= args.max_frames:
                break
            if frame_index % max(1, args.frame_stride) != 0:
                frame_index += 1
                continue

            video_timestamp_sec = frame_index / fps
            result = model.predict(
                frame,
                conf=args.conf,
                iou=args.iou,
                classes=target_class_ids,
                device=args.device,
                verbose=False,
            )[0]
            payload = build_frame_payload(
                result,
                FrameMeta(
                    intersection_id=args.intersection_id,
                    timestamp=run_start + video_timestamp_sec,
                    frame_index=frame_index,
                    video_timestamp_sec=video_timestamp_sec,
                ),
                DEFAULT_CLASSES,
                include_detections=args.include_detections,
            )
            current_signal = signal_state(
                args.signal_mode,
                args.signal_state,
                video_timestamp_sec,
                args.red_seconds,
                args.green_seconds,
                args.signal_offset,
            )
            payload["signal_state"] = current_signal

            events = evaluate_crosswalk_events(payload, crosswalk_roi, width, height, current_signal)
            if events:
                payload["safety_events"] = events
                alert_records.extend(events)
            frame_records.append(payload)

            annotated = result.plot()
            annotated = draw_counts(annotated, payload)
            annotated = draw_safety_overlay(annotated, crosswalk_roi, current_signal, len(events))
            last_annotated = annotated
            if writer is not None:
                writer.write(annotated)

            has_detection = any(payload["counts"].values())
            if args.save_screenshots and has_detection and screenshots < 5:
                cv2.imwrite(str(output_dir / f"screenshot_{screenshots + 1:02d}.jpg"), annotated)
                screenshots += 1

            processed += 1
            if processed % 30 == 0:
                print(f"Processed {processed} frames")
            frame_index += 1

        if args.save_screenshots and screenshots == 0 and last_annotated is not None:
            cv2.imwrite(str(output_dir / "screenshot_01.jpg"), last_annotated)
    finally:
        cap.release()
        if writer is not None:
            writer.release()

    per_second_records = aggregate_by_second(frame_records, DEFAULT_CLASSES)
    if args.save_frame_json:
        write_jsonl(output_dir / "detections.jsonl", frame_records)
    write_jsonl(output_dir / "detections_per_second.jsonl", per_second_records)
    write_jsonl(output_dir / "alerts.jsonl", alert_records)

    summary = summarize_records(frame_records)
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "source": str(source),
                "sample_url": args.sample_url if args.download_sample else None,
                "weights": args.weights,
                "device": args.device,
                "processed_frames": processed,
                "output_video": str(video_path) if args.save_video else None,
                "frame_jsonl": str(output_dir / "detections.jsonl") if args.save_frame_json else None,
                "per_second_jsonl": str(output_dir / "detections_per_second.jsonl"),
                "alerts_jsonl": str(output_dir / "alerts.jsonl"),
                "crosswalk_roi": [round(v, 4) for v in crosswalk_roi],
                "signal_mode": args.signal_mode,
                "red_seconds": args.red_seconds,
                "green_seconds": args.green_seconds,
                "alerts": len(alert_records),
                "max_counts_seen": summary,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(f"Done. Processed frames: {processed}")
    print(f"Output video: {video_path if args.save_video else 'disabled'}")
    if args.save_frame_json:
        print(f"Frame JSONL: {output_dir / 'detections.jsonl'}")
    print(f"Per-second JSONL: {output_dir / 'detections_per_second.jsonl'}")
    print(f"Alerts JSONL: {output_dir / 'alerts.jsonl'}")
    print(f"Max counts seen: {summary}")
    print(f"Safety alerts/events: {len(alert_records)}")


if __name__ == "__main__":
    main()

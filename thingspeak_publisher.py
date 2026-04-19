from __future__ import annotations

import argparse
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


CHANNEL_ID = "3348307"
WRITE_API_KEY = "PASTE_YOUR_WRITE_API_KEY_HERE"
UPDATE_URL = "https://api.thingspeak.com/update.json"
INPUT_JSONL = Path("outputs/detections_per_second.jsonl")
ALERTS_JSONL = Path("outputs/alerts.jsonl")
INTERVAL_SECONDS = 15


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Publish YOLO traffic counts to ThingSpeak.")
    parser.add_argument("--input", default=str(INPUT_JSONL))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def count(record: dict[str, Any], name: str) -> int:
    return int(record.get("counts", {}).get(name, 0) or 0)


def alert_counts_by_second() -> dict[int, int]:
    totals: dict[int, int] = {}
    for alert in read_jsonl(ALERTS_JSONL):
        second = alert.get("video_second", alert.get("video_timestamp_sec", 0))
        second = int(float(second or 0))
        totals[second] = totals.get(second, 0) + 1
    return totals


def payload_for(record: dict[str, Any], alerts: dict[int, int]) -> dict[str, Any]:
    car = count(record, "car")
    truck = count(record, "truck")
    bus = count(record, "bus")
    motorcycle = count(record, "motorcycle")
    second = int(float(record.get("video_second", 0) or 0))

    return {
        "api_key": WRITE_API_KEY,
        "field1": car + truck + bus + motorcycle,
        "field2": count(record, "person"),
        "field3": count(record, "bicycle"),
        "field4": car,
        "field5": truck,
        "field6": alerts.get(second, 0),
        "status": f"intersection={record.get('intersection_id', 'A01')} second={second}",
    }


def post(payload: dict[str, Any]) -> int:
    data = urllib.parse.urlencode(payload).encode("utf-8")
    request = urllib.request.Request(
        UPDATE_URL,
        data=data,
        method="POST",
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    with urllib.request.urlopen(request, timeout=20) as response:
        result = json.loads(response.read().decode("utf-8"))
    return int(result.get("entry_id", 0) or 0)


def public_channel_url() -> str:
    return f"https://thingspeak.com/channels/{CHANNEL_ID}"


def main() -> None:
    args = parse_args()
    if WRITE_API_KEY == "PASTE_YOUR_WRITE_API_KEY_HERE" and not args.dry_run:
        raise RuntimeError("Set WRITE_API_KEY in thingspeak_publisher.py before uploading to ThingSpeak.")

    records = read_jsonl(Path(args.input))
    if args.limit:
        records = records[: args.limit]
    if not records:
        raise RuntimeError(f"No records found: {args.input}")

    alerts = alert_counts_by_second()
    print(f"ThingSpeak channel: {public_channel_url()}")
    print("Fields: vehicle, pedestrian, bicycle, car, truck, alert")

    index = 0
    while True:
        for record in records:
            index += 1
            payload = payload_for(record, alerts)
            visible = {k: v for k, v in payload.items() if k != "api_key"}
            if args.dry_run:
                print(f"[dry-run {index}] {visible}")
            else:
                entry_id = post(payload)
                if not entry_id:
                    raise RuntimeError("ThingSpeak rejected the update. Wait 15 seconds and try again.")
                print(f"[{index}] entry_id={entry_id} {visible}")
            if not args.loop and index >= len(records):
                return
            if not args.dry_run:
                time.sleep(INTERVAL_SECONDS)
        if not args.loop:
            return


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import paho.mqtt.client as mqtt


INTERSECTION_ID = "A01"
INPUT_JSONL = Path("outputs/detections_per_second.jsonl")
ALERTS_JSONL = Path("outputs/alerts.jsonl")
CONTROL_LOG = Path("outputs/control_messages.jsonl")

MQTT_BROKER = "broker.hivemq.com"
MQTT_PORT = 1883
CONTROL_TOPIC = "cityflow/team7/a01/control/test"

VEHICLE_GREEN_THRESHOLD = 8
INTERVAL_SECONDS = 15


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create RED/GREEN/WARNING control messages for Wokwi nodes.")
    parser.add_argument("--input", default=str(INPUT_JSONL))
    parser.add_argument("--alerts", default=str(ALERTS_JSONL))
    parser.add_argument("--broker", default=MQTT_BROKER)
    parser.add_argument("--port", type=int, default=MQTT_PORT)
    parser.add_argument("--topic", default=CONTROL_TOPIC)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--json", action="store_true", help="Publish full JSON instead of plain RED/GREEN/WARNING text.")
    parser.add_argument("--publish", action="store_true", help="Publish to MQTT. Without this flag, only dry-run output is shown.")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def count(record: dict[str, Any], name: str) -> int:
    return int(record.get("counts", {}).get(name, 0) or 0)


def alert_counts_by_second(alerts_path: Path) -> dict[int, int]:
    totals: dict[int, int] = {}
    for alert in read_jsonl(alerts_path):
        second = alert.get("video_second", alert.get("video_timestamp_sec", 0))
        second = int(float(second or 0))
        totals[second] = totals.get(second, 0) + 1
    return totals


def build_control_message(record: dict[str, Any], alerts: dict[int, int]) -> dict[str, Any]:
    second = int(float(record.get("video_second", 0) or 0))
    car = count(record, "car")
    truck = count(record, "truck")
    bus = count(record, "bus")
    motorcycle = count(record, "motorcycle")
    pedestrian = count(record, "person")
    bicycle = count(record, "bicycle")
    vehicle_count = car + truck + bus + motorcycle
    alert_count = alerts.get(second, 0)

    if alert_count > 0:
        state = "WARNING"
        reason = "safety_alert"
    elif pedestrian > 0 or bicycle > 0:
        state = "WARNING"
        reason = "vulnerable_road_user_detected"
    elif vehicle_count >= VEHICLE_GREEN_THRESHOLD:
        state = "GREEN"
        reason = "vehicle_demand"
    else:
        state = "RED"
        reason = "low_vehicle_demand"

    return {
        "intersection_id": record.get("intersection_id", INTERSECTION_ID),
        "state": state,
        "reason": reason,
        "timestamp": round(time.time(), 3),
        "video_second": second,
        "vehicle_count": vehicle_count,
        "pedestrian_count": pedestrian,
        "bicycle_count": bicycle,
        "alert_count": alert_count,
        "counts": record.get("counts", {}),
    }


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def make_mqtt_client() -> mqtt.Client:
    return mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=f"cityflow-control-{int(time.time())}")


def main() -> None:
    args = parse_args()
    records = read_jsonl(Path(args.input))
    if args.limit:
        records = records[: args.limit]
    if not records:
        raise RuntimeError(f"No records found: {args.input}")

    alerts = alert_counts_by_second(Path(args.alerts))
    client = None
    if args.publish:
        client = make_mqtt_client()
        client.connect(args.broker, args.port, keepalive=60)
        client.loop_start()

    index = 0
    try:
        while True:
            for record in records:
                index += 1
                message = build_control_message(record, alerts)
                payload = json.dumps(message, ensure_ascii=False) if args.json else message["state"]

                if client is not None:
                    append_jsonl(CONTROL_LOG, message)
                    client.publish(args.topic, payload).wait_for_publish()
                    print(f"[{index}] published {args.topic}: {payload}")
                else:
                    print(f"[dry-run {index}] topic={args.topic} payload={payload}")

                if not args.loop and index >= len(records):
                    return
                if client is not None or args.loop:
                    time.sleep(INTERVAL_SECONDS)
            if not args.loop:
                return
    finally:
        if client is not None:
            client.loop_stop()
            client.disconnect()


if __name__ == "__main__":
    main()

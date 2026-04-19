from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import time
from pathlib import Path

import psutil


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure CPU/RAM usage for a command or an existing process.")
    parser.add_argument("--pid", type=int, default=0, help="Existing process id to monitor.")
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument("--output", default="evaluation/resource_usage.csv")
    parser.add_argument("command", nargs=argparse.REMAINDER, help="Command to launch and monitor.")
    return parser.parse_args()


def proc_tree(process: psutil.Process) -> list[psutil.Process]:
    try:
        return [process, *process.children(recursive=True)]
    except psutil.Error:
        return [process]


def sample(process: psutil.Process) -> dict[str, float]:
    cpu = 0.0
    rss = 0
    for proc in proc_tree(process):
        try:
            cpu += proc.cpu_percent(interval=None)
            rss += proc.memory_info().rss
        except psutil.Error:
            continue
    return {"cpu_percent": round(cpu, 2), "rss_mb": round(rss / (1024 * 1024), 2)}


def main() -> None:
    args = parse_args()
    if not args.pid and not args.command:
        raise SystemExit("Provide --pid or a command after --, for example: python monitor_resources.py -- python yolo_test.py --max-frames 120")

    launched = None
    process = psutil.Process(args.pid) if args.pid else None
    if process is None:
        command = args.command[1:] if args.command and args.command[0] == "--" else args.command
        launched = subprocess.Popen(command)
        process = psutil.Process(launched.pid)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["timestamp", "pid", "cpu_percent", "rss_mb"])
        writer.writeheader()
        for proc in proc_tree(process):
            try:
                proc.cpu_percent(interval=None)
            except psutil.Error:
                pass
        while True:
            time.sleep(args.interval)
            values = sample(process)
            writer.writerow({"timestamp": round(time.time(), 3), "pid": process.pid, **values})
            f.flush()
            print(values)
            if launched is not None and launched.poll() is not None:
                break
            if launched is None and not process.is_running():
                break

    if launched is not None:
        sys.exit(launched.returncode)


if __name__ == "__main__":
    main()

# YOLOv8 Traffic Detection + ThingSpeak Demo

Main pipeline:

```text
traffic video -> YOLOv8 -> JSON counts -> ThingSpeak dashboard
```

The project is intentionally kept small. The main files are:

```text
yolo_test.py              Run YOLO, generate counts, detect simple safety alerts
thingspeak_publisher.py   Upload counts to ThingSpeak every 15 seconds
evaluate_detection.py     Optional count-level detection evaluation
monitor_resources.py      Optional CPU/RAM usage sampling
```

## 1. Install

```powershell
python -m pip install -r requirements.txt
```

## 2. Run YOLO

```powershell
python yolo_test.py --max-frames 120 --no-download-sample
```

Outputs:

```text
outputs/detected_traffic.mp4
outputs/detections_per_second.jsonl
outputs/alerts.jsonl
outputs/summary.json
```

By default, the script does not save screenshots or full per-frame JSON. If needed:

```powershell
python yolo_test.py --save-screenshots --save-frame-json
```

## 3. Upload to ThingSpeak

Channel:

```text
https://thingspeak.com/channels/3348307
```

Upload one record:

```powershell
python thingspeak_publisher.py --limit 1
```

Before uploading, set your ThingSpeak Write API Key in `thingspeak_publisher.py`:

```python
WRITE_API_KEY = "PASTE_YOUR_WRITE_API_KEY_HERE"
```

Loop for demo:

```powershell
python thingspeak_publisher.py --loop
```

ThingSpeak field mapping:

```text
Field 1: vehicle_count = car + truck + bus + motorcycle
Field 2: pedestrian_count = person
Field 3: bicycle_count = bicycle
Field 4: car_count = car
Field 5: truck_count = truck
Field 6: alert_count
```

## 4. Safety Logic

The project does not detect real traffic lights from the video yet. It uses a simple simulated signal.

Default:

```text
red 15 seconds -> green 15 seconds -> repeat
```

The decision is inside `yolo_test.py`:

```text
cycle_position = (video_timestamp_sec + signal_offset) % (red_seconds + green_seconds)

if cycle_position < red_seconds:
    signal = red
else:
    signal = green
```

You can force red for demo:

```powershell
python yolo_test.py --signal-mode manual --signal-state red
```

Alert condition:

```text
YOLO detects person
person bbox center is inside the crosswalk ROI
signal_state is red
```

Default crosswalk ROI:

```text
0.30,0.55,0.78,0.95
```

When the condition is true, an event is written to:

```text
outputs/alerts.jsonl
```

ThingSpeak Field 6 uploads that alert count. The default demo video has no detected pedestrians, so alert_count is currently 0.

## 5. Optional Evaluation

Create manual labeling template:

```powershell
python evaluate_detection.py --make-template --limit 20
```

After filling `evaluation/manual_counts.csv`, compute metrics:

```powershell
python evaluate_detection.py --predictions outputs/detections_per_second.jsonl --ground-truth evaluation/manual_counts.csv
```

Sample resource usage:

```powershell
python monitor_resources.py --interval 0.5 --output evaluation/resource_usage.csv -- python yolo_test.py --max-frames 120 --no-download-sample
```

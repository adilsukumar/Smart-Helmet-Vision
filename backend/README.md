# Python camera pipeline

This folder contains the rule engine behind the prototype. It associates riders with a motorcycle track, keeps detections stable over several frames and emits one review event for each confirmed rule violation.

Quick check (no model download required):

```bash
python -m scripts.simulate_traffic
python -m unittest discover -s tests -v
```

Real video inference requires a custom Ultralytics model containing suitable motorcycle and helmet/no-helmet classes:

```bash
python -m pip install -r requirements.txt
python -m traffic_system.app --source path/to/video.mp4 --model path/to/model.pt --display
```

Edit `configs/traffic_demo.json` for the camera's stop-line coordinates, region of interest and class names.

Important: events are review candidates, not automatic fines. Real deployment needs data collected from the target road, day/night validation and an agreed evidence-retention policy.

# Smart Helmet Vision — Internship Prototype

I built a working prototype for detecting three motorcycle traffic-rule cases: no helmet, triple riding and crossing the stop line during a red signal. The aim is to turn a camera feed into a small set of reviewable events instead of treating every detection in every frame as a separate violation.

## What I completed

The submission has an interactive web demo and a Python camera pipeline. The web demo can be hosted on Vercel and lets a reviewer run five controlled cases: combined violation, no helmet, triple riding, red-light crossing and a compliant ride. The Python side contains the detection adapter, motorcycle/rider association, stop-line geometry, temporal confirmation, duplicate suppression, evidence output, dataset conversion and training entry points. I also wrote six automated tests; all six pass.

## How it works right now

1. A custom object detector identifies the motorcycle and the riders' helmet states.
2. Tracking keeps the same motorcycle ID across consecutive frames.
3. Spatial association connects nearby riders to that motorcycle.
4. The rule engine checks several frames before confirming no-helmet or triple-riding cases. For signal jumping, it checks whether a tracked motorcycle crosses the configured stop line while the signal is red.
5. A confirmed case is recorded once with its frame, rule, track ID and confidence for human review.

The Vercel page is a transparent browser simulation of this logic. The Python code is the part intended for real video or Raspberry Pi use. I have not claimed final real-road accuracy because that needs the approved dataset, trained weights and testing on the actual camera position.

## What I can do next

Given access to the dataset and Raspberry Pi hardware, I can train a small model, validate it separately for day/night and crowded scenes, export it to a Pi-friendly format such as NCNN, measure FPS/latency, tune the stop-line region for the target junction and add a simple review dashboard. The system should create candidate events for a person to verify; it should not issue automatic challans or perform identity recognition.

## Current result

Browser demo: production build passed; combined test produced all 3 expected events with no browser errors. Python prototype: 6/6 automated tests passed and the deterministic simulation emitted one NO_HELMET, one TRIPLE_RIDING and one RED_LIGHT_CROSSING event.

## Research used

- AI City Challenge 2024, Track 5 dataset/evaluation: https://www.aicitychallenge.org/2024-data-and-evaluation/
- Raspberry Pi AI Camera documentation: https://www.raspberrypi.com/documentation/accessories/ai-camera.html
- Ultralytics Raspberry Pi/NCNN deployment guide: https://docs.ultralytics.com/guides/raspberry-pi
- Motor Vehicles Act, 1988 (Sections 128 and 129): https://www.indiacode.nic.in/handle/123456789/13700?locale=en

Source code and deployment instructions are included in the project README.

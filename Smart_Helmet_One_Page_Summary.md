# Smart Helmet Vision — one-page summary

I made this as a practical starting point for the internship topic: helmet detection, triple riding and signal jumping using a Raspberry Pi, camera and machine learning. It is not only a proposal now. The repository has a real traffic-video run, a browser demo, working Python code and tests.

For the real demo I used the full 12.5-second Pexels clip “Busy Indian Street with Traffic and Motorbikes”. A normal YOLO11 model tracks motorcycles and people, while a separate YOLO11 helmet model checks for “With Helmet” and “Without Helmet”. The code then associates nearby riders with a motorcycle track and waits for the same condition across several frames before creating a candidate event. The processed video is on the website, so the result can be reviewed without installing Python.

This run produced four no-helmet candidates and two triple-riding candidates. I am deliberately calling them candidates because the scene becomes crowded near the right edge and rider-to-bike association is not perfect. Signal jumping was not evaluated in this clip: there is no suitable traffic signal and calibrated stop line in view. The interactive browser sandbox still demonstrates that rule, and the Python code supports a configurable signal region and stop line for the correct junction footage.

What is working now: full video input, motorcycle/person tracking, separate helmet inference, rider-to-bike association, multi-frame confirmation, duplicate-event control, evidence images, annotated-video export, five browser rule scenarios and six automated Python tests. The Vercel build also keeps the heavy ML packages out of the web deployment.

The next hardware step is to connect a fixed Raspberry Pi camera, collect footage from its actual angle, label/train with the chosen dataset, export a smaller model to NCNN and measure FPS, latency and temperature on the Pi. The stop line and signal region would be calibrated once for that camera. Results should remain review suggestions, not automatic fines or identity recognition.

Research and assets I used:

- AI City Challenge 2024 Track 5: https://www.aicitychallenge.org/2024-data-and-evaluation/
- Raspberry Pi AI Camera: https://www.raspberrypi.com/documentation/accessories/ai-camera.html
- Ultralytics tracking: https://docs.ultralytics.com/modes/track/
- Ultralytics Raspberry Pi / NCNN guide: https://docs.ultralytics.com/guides/raspberry-pi/
- MIT-licensed helmet model used for this demo: https://huggingface.co/nnsohamnn/helmet-detection-yolo11
- Pexels source video by Aamir Somewhere: https://www.pexels.com/video/busy-indian-street-with-traffic-and-motorbikes-34394424/
- Motor Vehicles Act, 1988: https://www.indiacode.nic.in/handle/123456789/13700?locale=en

Code: https://github.com/adilsukumar/Smart-Helmet-Vision

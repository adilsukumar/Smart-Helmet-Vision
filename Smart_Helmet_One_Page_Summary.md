# Smart Helmet Vision — one-page summary

I made this as a practical starting point for the internship topic: helmet detection, triple riding and signal jumping using a Raspberry Pi, camera and machine learning. It is not only a proposal now. The repository has an uploadable video detector, a browser rule demo, working Python code and tests.

The website accepts a normal video from the user’s device. A YOLO11 model finds motorcycles and people in the current frame, while a lightweight YOLOv8n helmet model checks for “With Helmet” and “Without Helmet”. The detections are drawn on a canvas above the original video, so they are produced for whichever file is selected instead of being baked into a prepared result. The selected video stays inside the browser and is not uploaded.

To try it, I first wait for the note above the video to turn green and say “Models ready”. The page shows whether it is loading model 1 or model 2. After that I can play the built-in road sample or choose any browser-supported video from my device. The controls are kept unavailable until the models are ready so the loading step is clear. I also replaced the earlier 80 MB helmet export with an approximately 12 MB version because the larger one could stall in the browser.

The live page reports motorcycles, riders, no-helmet detections and triple-riding candidates for the frames it analyses. I am deliberately calling them candidates because crowded scenes and poor camera angles can create incorrect rider-to-bike associations. Signal jumping is not guessed for arbitrary uploads: it needs one fixed camera with a calibrated stop line and signal area. The rule sandbox demonstrates that logic, and the Python code supports the actual configuration.

For accuracy, the browser no longer gives a helmet verdict for every small rider in the distance. It first checks whether a motorcycle and rider are large enough in the frame, crops around the nearest usable group, and then runs the helmet model on that closer view. A helmet result needs higher confidence and must repeat across two checks; otherwise it stays uncertain. I also reduced the model input from 416 to 320 pixels to make each browser check faster.

What is working now: selecting local videos in the deployed page, browser-side motorcycle/person and helmet inference, rider-to-bike association, an unmarked sample video, full Python video input, tracking, multi-frame confirmation, duplicate-event control, evidence output, five browser rule scenarios and six automated Python tests. The first browser model load is large and the frame rate depends on the device, so this is still a prototype rather than an enforcement-ready product.

The next hardware step is to connect a fixed Raspberry Pi camera, collect footage from its actual angle, label/train with the chosen dataset, export a smaller model to NCNN and measure FPS, latency and temperature on the Pi. The stop line and signal region would be calibrated once for that camera. Results should remain review suggestions, not automatic fines or identity recognition.

Research and assets I used:

- AI City Challenge 2024 Track 5: https://www.aicitychallenge.org/2024-data-and-evaluation/
- Raspberry Pi AI Camera: https://www.raspberrypi.com/documentation/accessories/ai-camera.html
- Ultralytics tracking: https://docs.ultralytics.com/modes/track/
- Ultralytics Raspberry Pi / NCNN guide: https://docs.ultralytics.com/guides/raspberry-pi/
- MIT-licensed lightweight helmet model used for this demo: https://huggingface.co/iam-tsr/yolov8n-helmet-detection
- Pexels source video by Aamir Somewhere: https://www.pexels.com/video/busy-indian-street-with-traffic-and-motorbikes-34394424/
- Motor Vehicles Act, 1988: https://www.indiacode.nic.in/handle/123456789/13700?locale=en

Code: https://github.com/adilsukumar/Smart-Helmet-Vision

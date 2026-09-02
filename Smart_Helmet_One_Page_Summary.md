# Smart Helmet Vision — One-Page Summary

## What I researched

I researched how a camera-based traffic system can detect motorcycles, riders without helmets, triple riding and signal jumping. My main focus was how this could later run with a Raspberry Pi and a fixed road camera. I looked into suitable datasets, object-detection models, rider-to-motorcycle matching, multi-frame tracking and Raspberry Pi deployment. I also checked the relevant Indian traffic rules, especially Sections 128 and 129 of the Motor Vehicles Act.

## What I have completed

I built a working software prototype and deployed it as a website. A user can play the sample road video or choose a video from their own device. The browser loads two small detection models: one finds motorcycles and people, and the other checks helmet use. The video is analysed inside the browser and is not uploaded to a server.

The page ignores motorcycles that are too far away for a reasonable helmet decision. For closer riders, the visual label normally has to repeat before it is shown. The numbers beside the video are running totals for that video. A very strong no-helmet result is added once and tracked instead of being counted again in every frame. Old boxes are cleared so they do not get stuck over later frames.

I also wrote a Python version for the complete system logic. It supports video or camera input, tracking, rider-to-bike association, multi-frame confirmation, duplicate-event control and saving evidence. A separate rule demo on the website explains helmet, triple-riding and red-light cases. Six automated tests currently check the main rule logic.

## How it works

The camera provides video frames. The first model finds motorcycles and people. The code then checks which people are positioned like riders on a motorcycle. Only a close rider group is sent to the helmet model. Results are compared across frames before a case is confirmed. Triple riding is based on the number of riders associated with one motorcycle. Signal jumping needs a fixed camera, a marked stop line and the current traffic-light state; it cannot be judged reliably from any random uploaded video.

## What I can do next

The next step is to connect a Raspberry Pi camera and collect footage from the actual installation angle. I can label this footage, train or fine-tune a smaller model, export it to a Raspberry Pi-friendly format such as NCNN, and measure speed, delay and temperature on the device. I can also add a proper tracker, configure the stop line and signal area, save review evidence and create a small dashboard for confirmed events.

This is still a prototype. Crowded scenes, poor lighting, blocked riders and low camera angles can cause mistakes, so the output should be reviewed by a person and should not be used for automatic fines or identification.

## Research links

- AI City Challenge 2024, Track 5: https://www.aicitychallenge.org/2024-data-and-evaluation/
- Raspberry Pi AI Camera: https://www.raspberrypi.com/documentation/accessories/ai-camera.html
- Ultralytics tracking: https://docs.ultralytics.com/modes/track/
- Raspberry Pi and NCNN guide: https://docs.ultralytics.com/guides/raspberry-pi/
- Helmet model used in the browser demo: https://huggingface.co/iam-tsr/yolov8n-helmet-detection
- Motor Vehicles Act, 1988: https://www.indiacode.nic.in/handle/123456789/13700?locale=en

Code: https://github.com/adilsukumar/Smart-Helmet-Vision

Live demo: https://smart-helmet-vision.vercel.app/

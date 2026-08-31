<p align="center">
  <img src="assets/helmet-vision-banner.svg" alt="Smart Helmet Vision animated project banner" width="100%" />
</p>

# Smart Helmet Vision

This started as an internship research task. I didn't want to submit only a document or a notebook, so I made a small working demo too.

The project watches a motorcycle across a few frames and checks three things: helmet use, triple riding and crossing the stop line while the signal is red. The website is the easy-to-open version for a reviewer. The Python folder is where the actual camera and rule-engine work lives.

<p align="center">
  <img src="assets/project-icons.svg" width="360" alt="Helmet, riders and signal icons" />
</p>

<table>
  <tr>
    <td align="center" width="33%"><b>Helmet check</b><br><sub>Waits for the result to stay consistent</sub></td>
    <td align="center" width="33%"><b>Triple riding</b><br><sub>Associates riders with one bike track</sub></td>
    <td align="center" width="33%"><b>Signal crossing</b><br><sub>Checks line crossing only during red</sub></td>
  </tr>
</table>

## What is here

The browser demo has five cases you can run: all violations together, no helmet, triple riding, red-light crossing and a normal ride. It is a simulation of the decision logic, which makes it possible to host on Vercel without pretending a Raspberry Pi model is somehow running in the browser.

The `backend` folder has the Python code for video input, tracking, rider-to-bike matching, stop-line crossing, event confirmation and saving evidence. It also has a small deterministic simulation and six unit tests.

At the moment:

- the website builds and runs;
- the combined demo gives all three expected events;
- all six Python tests pass;
- training on the full dataset and Raspberry Pi speed testing are still left to do.

## Running the website

You need a recent Node.js version and pnpm.

```bash
pnpm install
pnpm dev
```

For the same production build Vercel will use:

```bash
pnpm build
pnpm preview
```

## Checking the Python logic

```bash
cd backend
python helmet_system.py --demo
python -m unittest -v test_helmet_system.py
```

The controlled simulation should print one event for each rule:

```text
NO_HELMET
TRIPLE_RIDING
RED_LIGHT_CROSSING
```

## Trying a real video

Install the Python packages first:

```bash
cd backend
python -m pip install -r ml-requirements.txt
```

Then give the program a video/camera source and custom model weights:

```bash
python helmet_system.py \
  --source path/to/video.mp4 \
  --model path/to/custom_helmet_model.pt \
  --display
```

A normal object-detection model is not enough for helmet compliance. The model needs classes that match the project configuration, including helmet/no-helmet and motorcycle-related objects. Camera regions and the stop line can be changed in `backend/traffic_demo.json`.

## Training direction

I used AI City Challenge 2024 Track 5 as the main dataset reference. The conversion script prepares the annotations for a YOLO-style training setup:

```bash
cd backend
python training.py convert \
  --frames-root path/to/frames \
  --annotations path/to/groundtruth.txt \
  --out data/aicity_yolo

python training.py train \
  --data data/aicity_yolo/dataset.yaml \
  --model yolo11n.pt \
  --epochs 50 \
  --export ncnn
```

Training should be done on a laptop GPU or cloud notebook. The smaller exported model can then be measured on the Raspberry Pi for FPS, latency and heat.

## Project layout

```text
src/App.tsx                browser demo and page entry
src/styles.css             page styling and motion
backend/helmet_system.py   camera, rules and controlled demo
backend/training.py        dataset conversion and training
backend/test_helmet_system.py
assets/                    two SVG artwork files
```

## Deploying it

Push the repository to GitHub, open Vercel, choose **New Project**, and import this repo. Vercel should detect Vite automatically. The build command is `pnpm build` and the output directory is `dist`.

## References I used

- [AI City Challenge 2024 — Track 5](https://www.aicitychallenge.org/2024-data-and-evaluation/)
- [Raspberry Pi AI Camera documentation](https://www.raspberrypi.com/documentation/accessories/ai-camera.html)
- [Ultralytics Raspberry Pi and NCNN guide](https://docs.ultralytics.com/guides/raspberry-pi)
- [Motor Vehicles Act, 1988 — India Code](https://www.indiacode.nic.in/handle/123456789/13700?locale=en)

## One honest limitation

The rule engine is tested, but I am not calling this a finished traffic-enforcement system. Real accuracy numbers need trained weights and test footage from the actual road/camera angle. Any event should go to a person for review; it should not become an automatic challan or an identity-recognition system.

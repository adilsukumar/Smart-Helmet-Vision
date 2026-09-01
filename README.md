<p align="center">
  <img src="assets/helmet-vision-banner.svg" alt="Smart Helmet Vision animated project banner" width="100%" />
</p>

# Smart Helmet Vision

This started as an internship research task. I didn't want to submit only a document or a notebook, so I made a small working demo too.

The project watches a motorcycle across several frames and checks three things: helmet use, triple riding and crossing the stop line while the signal is red. The website now includes a processed real traffic clip, not only the animated rule demo. The Python folder is where the camera, model and rule-engine work lives.

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

The first thing on the browser demo is a real 12.5-second road video processed with two models. YOLO11 handles motorcycle/person tracking, and a separate helmet model predicts `With Helmet` or `Without Helmet`. The page also keeps five small simulated cases so each rule is easy to test on demand.

The `backend` folder has the Python code for video input, tracking, rider-to-bike matching, stop-line crossing, event confirmation and saving evidence. It also has a small deterministic simulation and six unit tests.

At the moment:

- the full real clip is included as a browser-friendly annotated video;
- the website builds and runs;
- the combined demo gives all three expected events;
- all six Python tests pass;
- this traffic run produced 4 no-helmet and 2 triple-riding candidates;
- Raspberry Pi speed testing and training on footage from the final camera angle are still left to do.

Those numbers are model suggestions, not verified ground truth. The crowded right edge makes some rider-to-bike associations uncertain. I also left signal jumping unevaluated for the real clip because it has no usable traffic signal and calibrated stop line. I would rather show that limitation than draw a made-up line and call it a detection.

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

Then give the program a video/camera source, a general tracking model and the helmet model:

```bash
python helmet_system.py \
  --source path/to/video.mp4 \
  --model path/to/yolo11n.pt \
  --helmet-model path/to/helmet-model.pt \
  --signal unknown \
  --save-video annotated.mp4 \
  --display
```

A normal object-detection model is not enough for helmet compliance, which is why the demo uses a second model. For a fixed junction, camera regions and the stop line can be changed in `backend/traffic_demo.json`. Use `--signal auto` only after setting the signal region for that camera.

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
public/real-traffic-demo.mp4   processed full demo clip
assets/                    two SVG artwork files
```

## Deploying it

Push the repository to GitHub, open Vercel, choose **New Project**, and import this repo. Vercel should detect Vite automatically. The build command is `pnpm build` and the output directory is `dist`.

## References I used

- [AI City Challenge 2024 — Track 5](https://www.aicitychallenge.org/2024-data-and-evaluation/)
- [Raspberry Pi AI Camera documentation](https://www.raspberrypi.com/documentation/accessories/ai-camera.html)
- [Ultralytics Raspberry Pi and NCNN guide](https://docs.ultralytics.com/guides/raspberry-pi)
- [Ultralytics tracking mode](https://docs.ultralytics.com/modes/track/)
- [MIT-licensed helmet model used for the real demo](https://huggingface.co/nnsohamnn/helmet-detection-yolo11)
- [Pexels source video by Aamir Somewhere](https://www.pexels.com/video/busy-indian-street-with-traffic-and-motorbikes-34394424/)
- [Motor Vehicles Act, 1988 — India Code](https://www.indiacode.nic.in/handle/123456789/13700?locale=en)

## One honest limitation

The rule engine is tested, but I am not calling this a finished traffic-enforcement system. Real accuracy numbers need trained weights and test footage from the actual road/camera angle. Any event should go to a person for review; it should not become an automatic challan or an identity-recognition system.

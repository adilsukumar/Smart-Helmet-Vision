import { StrictMode, useEffect, useMemo, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import "./styles.css";

type ScenarioId = "combined" | "helmet" | "triple" | "signal" | "safe";
type Rule = "NO_HELMET" | "TRIPLE_RIDING" | "RED_LIGHT_CROSSING";

type Scenario = {
  id: ScenarioId;
  name: string;
  note: string;
  riders: number;
  missingHelmet: boolean;
  redSignal: boolean;
  expected: Rule[];
};

type EventItem = {
  rule: Rule;
  frame: number;
  confidence: number;
  detail: string;
};

const scenarios: Scenario[] = [
  {
    id: "combined",
    name: "Combined violation",
    note: "Three riders, one without a helmet, crossing during red.",
    riders: 3,
    missingHelmet: true,
    redSignal: true,
    expected: ["NO_HELMET", "TRIPLE_RIDING", "RED_LIGHT_CROSSING"],
  },
  {
    id: "helmet",
    name: "No helmet",
    note: "A persistent no-helmet detection becomes one review event.",
    riders: 2,
    missingHelmet: true,
    redSignal: false,
    expected: ["NO_HELMET"],
  },
  {
    id: "triple",
    name: "Triple riding",
    note: "Three associated riders stay visible for several frames.",
    riders: 3,
    missingHelmet: false,
    redSignal: false,
    expected: ["TRIPLE_RIDING"],
  },
  {
    id: "signal",
    name: "Red-light crossing",
    note: "The tracked bike crosses the configured line while red is active.",
    riders: 2,
    missingHelmet: false,
    redSignal: true,
    expected: ["RED_LIGHT_CROSSING"],
  },
  {
    id: "safe",
    name: "Compliant ride",
    note: "Two helmeted riders cross on green. No event should be generated.",
    riders: 2,
    missingHelmet: false,
    redSignal: false,
    expected: [],
  },
];

const ruleCopy: Record<Rule, { short: string; detail: string; threshold: number; confidence: number }> = {
  NO_HELMET: {
    short: "No helmet",
    detail: "Passenger helmet state remained negative across the confirmation window.",
    threshold: 34,
    confidence: 0.89,
  },
  TRIPLE_RIDING: {
    short: "Triple riding",
    detail: "Three riders remained associated with motorcycle track #07.",
    threshold: 47,
    confidence: 0.91,
  },
  RED_LIGHT_CROSSING: {
    short: "Red-light crossing",
    detail: "Track #07 crossed the stop line while the observed state was red.",
    threshold: 76,
    confidence: 0.95,
  },
};

type VideoBox = {
  x: number;
  y: number;
  width: number;
  height: number;
  score: number;
  classId: number;
  label: string;
};

type FrameResult = {
  bikes: number;
  riders: number;
  noHelmet: number;
  triple: number;
  took: number;
};

type FrameCrop = { x: number; y: number; width: number; height: number };
type HelmetMemory = { x: number; y: number; label: string; streak: number };
type ModelName = "traffic" | "helmet";
type ModelOutput = { data: Float32Array; dims: number[] };
type BrowserModels = { run: (model: ModelName, data: Float32Array) => Promise<ModelOutput> };

const inputSize = 320;
const detectionInterval = 500;
let browserModels: Promise<BrowserModels> | null = null;

function loadBrowserModels(onProgress?: (message: string) => void) {
  if (!browserModels) {
    browserModels = new Promise((resolve, reject) => {
      const worker = new Worker(new URL("./detector.worker.ts", import.meta.url), { type: "module" });
      const waiting = new Map<number, { resolve: (output: ModelOutput) => void; reject: (error: Error) => void }>();
      let requestId = 0;

      worker.onmessage = (event) => {
        const message = event.data;
        if (message.type === "progress") {
          onProgress?.(message.message);
          return;
        }
        if (message.type === "ready") {
          resolve({
            run: (model, data) => new Promise((finish, fail) => {
              const id = ++requestId;
              waiting.set(id, { resolve: finish, reject: fail });
              worker.postMessage({ type: "run", id, model, data }, [data.buffer]);
            }),
          });
          return;
        }
        if (message.type === "result") {
          const request = waiting.get(message.id);
          if (!request) return;
          waiting.delete(message.id);
          request.resolve({ data: message.data, dims: message.dims });
          return;
        }
        if (message.type === "error") {
          const error = new Error(message.message);
          if (!message.id) reject(error);
          const request = waiting.get(message.id);
          if (request) {
            waiting.delete(message.id);
            request.reject(error);
          }
        }
      };
      worker.onerror = () => reject(new Error("Detection worker could not start"));
      worker.postMessage({ type: "load" });
    });
  }
  return browserModels;
}

function overlap(a: VideoBox, b: VideoBox) {
  const x = Math.max(0, Math.min(a.x + a.width, b.x + b.width) - Math.max(a.x, b.x));
  const y = Math.max(0, Math.min(a.y + a.height, b.y + b.height) - Math.max(a.y, b.y));
  const intersection = x * y;
  return intersection / (a.width * a.height + b.width * b.height - intersection || 1);
}

function nms(boxes: VideoBox[], threshold = 0.45) {
  const kept: VideoBox[] = [];
  for (const box of [...boxes].sort((a, b) => b.score - a.score)) {
    if (kept.length >= 80) break;
    if (!kept.some((item) => item.classId === box.classId && overlap(item, box) > threshold)) kept.push(box);
  }
  return kept;
}

function readOutput(
  tensor: ModelOutput,
  classes: Array<{ id: number; label: string }>,
  scale: number,
  padX: number,
  padY: number,
  videoWidth: number,
  videoHeight: number,
  minimum: number,
  offsetX = 0,
  offsetY = 0,
) {
  const channels = tensor.dims[1];
  const anchors = tensor.dims[2];
  const values = tensor.data;
  const boxes: VideoBox[] = [];
  for (let index = 0; index < anchors; index += 1) {
    let selected = classes[0];
    let score = 0;
    for (const item of classes) {
      const value = values[(4 + item.id) * anchors + index];
      if (value > score) {
        score = value;
        selected = item;
      }
    }
    if (score < minimum) continue;
    const centerX = values[index];
    const centerY = values[anchors + index];
    const width = values[anchors * 2 + index];
    const height = values[anchors * 3 + index];
    const x1 = offsetX + Math.max(0, (centerX - width / 2 - padX) / scale);
    const y1 = offsetY + Math.max(0, (centerY - height / 2 - padY) / scale);
    const x2 = offsetX + Math.min(videoWidth, (centerX + width / 2 - padX) / scale);
    const y2 = offsetY + Math.min(videoHeight, (centerY + height / 2 - padY) / scale);
    if (x2 > x1 && y2 > y1) boxes.push({ x: x1, y: y1, width: x2 - x1, height: y2 - y1, score, classId: selected.id, label: selected.label });
  }
  return nms(boxes);
}

function prepareFrame(video: HTMLVideoElement, scratch: HTMLCanvasElement, crop?: FrameCrop) {
  scratch.width = inputSize;
  scratch.height = inputSize;
  const context = scratch.getContext("2d", { willReadFrequently: true })!;
  const sourceX = crop?.x ?? 0;
  const sourceY = crop?.y ?? 0;
  const sourceWidth = crop?.width ?? video.videoWidth;
  const sourceHeight = crop?.height ?? video.videoHeight;
  const scale = Math.min(inputSize / sourceWidth, inputSize / sourceHeight);
  const width = sourceWidth * scale;
  const height = sourceHeight * scale;
  const padX = (inputSize - width) / 2;
  const padY = (inputSize - height) / 2;
  context.fillStyle = "rgb(114, 114, 114)";
  context.fillRect(0, 0, inputSize, inputSize);
  context.drawImage(video, sourceX, sourceY, sourceWidth, sourceHeight, padX, padY, width, height);
  const pixels = context.getImageData(0, 0, inputSize, inputSize).data;
  const input = new Float32Array(3 * inputSize * inputSize);
  const plane = inputSize * inputSize;
  for (let i = 0; i < plane; i += 1) {
    input[i] = pixels[i * 4] / 255;
    input[plane + i] = pixels[i * 4 + 1] / 255;
    input[plane * 2 + i] = pixels[i * 4 + 2] / 255;
  }
  return { input, scale, padX, padY, sourceX, sourceY, sourceWidth, sourceHeight };
}

function pointInside(box: VideoBox, x: number, y: number, upperOnly = false) {
  const bottom = upperOnly ? box.y + box.height * 0.5 : box.y + box.height;
  return x >= box.x && x <= box.x + box.width && y >= box.y && y <= bottom;
}

function riderRegion(bike: VideoBox): VideoBox {
  return {
    ...bike,
    x: bike.x - bike.width * 0.45,
    y: bike.y - bike.height * 1.8,
    width: bike.width * 1.9,
    height: bike.height * 3.1,
  };
}

function cropAround(boxes: VideoBox[], frameWidth: number, frameHeight: number): FrameCrop {
  const left = Math.min(...boxes.map((box) => box.x));
  const top = Math.min(...boxes.map((box) => box.y));
  const right = Math.max(...boxes.map((box) => box.x + box.width));
  const bottom = Math.max(...boxes.map((box) => box.y + box.height));
  const width = right - left;
  const height = bottom - top;
  const x = Math.max(0, left - width * 0.3);
  const y = Math.max(0, top - height * 0.28);
  return {
    x,
    y,
    width: Math.min(frameWidth - x, width * 1.6),
    height: Math.min(frameHeight - y, height * 1.48),
  };
}

function drawTag(context: CanvasRenderingContext2D, box: VideoBox, text: string, color: string) {
  context.strokeStyle = color;
  context.lineWidth = Math.max(2, context.canvas.width / 600);
  context.strokeRect(box.x, box.y, box.width, box.height);
  context.font = `${Math.max(15, context.canvas.width / 70)}px ui-monospace, monospace`;
  const textWidth = context.measureText(text).width + 12;
  const top = Math.max(0, box.y - 25);
  context.fillStyle = color;
  context.fillRect(box.x, top, textWidth, 25);
  context.fillStyle = "#fff";
  context.fillText(text, box.x + 6, top + 18);
}

function LiveVideoDetector() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const overlayRef = useRef<HTMLCanvasElement>(null);
  const scratchRef = useRef(document.createElement("canvas"));
  const modelsRef = useRef<BrowserModels | null>(null);
  const objectUrl = useRef<string | null>(null);
  const running = useRef(false);
  const lastRun = useRef(0);
  const animation = useRef(0);
  const clearTimer = useRef<number | null>(null);
  const helmetHistory = useRef<HelmetMemory[]>([]);
  const [modelState, setModelState] = useState("Please wait — the models are still loading");
  const [modelsReady, setModelsReady] = useState(false);
  const [fileName, setFileName] = useState("Built-in road sample");
  const [result, setResult] = useState<FrameResult>({ bikes: 0, riders: 0, noHelmet: 0, triple: 0, took: 0 });

  const analyse = async () => {
    const video = videoRef.current;
    const overlay = overlayRef.current;
    const models = modelsRef.current;
    if (!video || !overlay || !models || video.readyState < 2 || running.current || !video.videoWidth) return;
    running.current = true;
    const started = performance.now();
    const frameTime = video.currentTime;
    overlay.width = video.videoWidth;
    overlay.height = video.videoHeight;
    const context = overlay.getContext("2d")!;
    context.clearRect(0, 0, overlay.width, overlay.height);
    if (clearTimer.current) window.clearTimeout(clearTimer.current);
    try {
      const frame = prepareFrame(video, scratchRef.current);
      const trafficTensor = await models.run("traffic", frame.input);
      const common = [frame.scale, frame.padX, frame.padY, frame.sourceWidth, frame.sourceHeight] as const;
      const traffic = readOutput(trafficTensor, [{ id: 0, label: "person" }, { id: 3, label: "motorcycle" }], ...common, 0.28);
      const people = traffic.filter((box) => box.label === "person");
      const bikes = traffic.filter((box) => box.label === "motorcycle");
      const closeBikes = bikes
        .filter((bike) => bike.score >= 0.32 && bike.width >= video.videoWidth * 0.075 && bike.height >= video.videoHeight * 0.16)
        .sort((a, b) => b.width * b.height - a.width * a.height);
      const targetBike = closeBikes[0];
      const targetRegion = targetBike ? riderRegion(targetBike) : null;
      const checkedPeople = targetRegion
        ? people.filter((person) => person.height >= video.videoHeight * 0.18 && pointInside(targetRegion, person.x + person.width / 2, person.y + person.height))
        : [];

      let helmets: VideoBox[] = [];
      if (targetBike && checkedPeople.length) {
        const crop = cropAround([targetBike, ...checkedPeople], video.videoWidth, video.videoHeight);
        const helmetFrame = prepareFrame(video, scratchRef.current, crop);
        const helmetTensor = await models.run("helmet", helmetFrame.input);
        helmets = readOutput(
          helmetTensor,
          [{ id: 0, label: "helmet" }, { id: 1, label: "no helmet" }],
          helmetFrame.scale,
          helmetFrame.padX,
          helmetFrame.padY,
          helmetFrame.sourceWidth,
          helmetFrame.sourceHeight,
          0.35,
          helmetFrame.sourceX,
          helmetFrame.sourceY,
        );
      }

      let noHelmet = 0;
      let drewLabel = false;
      const nextHistory: HelmetMemory[] = [];
      for (const person of checkedPeople) {
        const mark = helmets
          .filter((helmet) => pointInside(person, helmet.x + helmet.width / 2, helmet.y + helmet.height / 2, true))
          .sort((a, b) => b.score - a.score)[0];
        const minimum = mark?.label === "no helmet" ? 0.58 : 0.55;
        const reliable = mark && mark.score >= minimum ? mark : null;
        if (!reliable) continue;
        const centerX = person.x + person.width / 2;
        const centerY = person.y + person.height / 2;
        const previous = helmetHistory.current.find((item) => item.label === reliable.label && Math.hypot(item.x - centerX, item.y - centerY) < Math.max(person.width, person.height) * 0.75);
        const memory = { x: centerX, y: centerY, label: reliable.label, streak: (previous?.streak ?? 0) + 1 };
        nextHistory.push(memory);
        const confirmed = memory.streak >= 2;
        if (!confirmed) continue;
        const color = reliable.label === "no helmet" ? "#ef514b" : "#4dcc85";
        if (confirmed && reliable.label === "no helmet") noHelmet += 1;
        const frameIsCurrent = video.paused || Math.abs(video.currentTime - frameTime) < 0.75;
        if (frameIsCurrent) {
          drawTag(context, person, `${reliable.label} ${reliable.score.toFixed(2)}`, color);
          drewLabel = true;
        }
      }
      helmetHistory.current = nextHistory;

      let triple = 0;
      for (const bike of closeBikes) {
        const region = riderRegion(bike);
        const riderCount = people.filter((person) => person.height >= video.videoHeight * 0.18 && pointInside(region, person.x + person.width / 2, person.y + person.height)).length;
        if (riderCount >= 3) {
          triple += 1;
        }
      }
      if (drewLabel && !video.paused) {
        clearTimer.current = window.setTimeout(() => context.clearRect(0, 0, overlay.width, overlay.height), 450);
      }
      setResult({ bikes: targetBike && checkedPeople.length ? 1 : 0, riders: checkedPeople.length, noHelmet, triple, took: Math.round(performance.now() - started) });
      if (!checkedPeople.length) setModelState(video.paused ? "Ready" : "Watching video…");
      else setModelState(video.paused ? "Close riders checked" : "Checking close riders while video plays");
    } catch (error) {
      setModelState(error instanceof Error ? `Detection error: ${error.message}` : "Could not analyse this frame");
    } finally {
      running.current = false;
      lastRun.current = performance.now();
    }
  };

  useEffect(() => {
    let active = true;
    loadBrowserModels((message) => {
      if (active) setModelState(message);
    })
      .then((models) => {
        if (!active) return;
        modelsRef.current = models;
        setModelsReady(true);
        setModelState("Models ready — now play or choose a video");
        void analyse();
      })
      .catch((error) => {
        setModelsReady(false);
        setModelState(error instanceof Error ? `Model load failed: ${error.message}` : "Model load failed");
      });
    const tick = (time: number) => {
      const video = videoRef.current;
      if (video && !video.paused && time - lastRun.current > detectionInterval) {
        lastRun.current = time;
        void analyse();
      }
      animation.current = requestAnimationFrame(tick);
    };
    animation.current = requestAnimationFrame(tick);
    return () => {
      active = false;
      cancelAnimationFrame(animation.current);
      if (clearTimer.current) window.clearTimeout(clearTimer.current);
      if (objectUrl.current) URL.revokeObjectURL(objectUrl.current);
    };
  }, []);

  const chooseVideo = (file?: File) => {
    const video = videoRef.current;
    if (!file || !video || !modelsReady) return;
    if (objectUrl.current) URL.revokeObjectURL(objectUrl.current);
    objectUrl.current = URL.createObjectURL(file);
    video.src = objectUrl.current;
    video.load();
    const overlay = overlayRef.current;
    if (overlay) overlay.getContext("2d")?.clearRect(0, 0, overlay.width, overlay.height);
    void video.play().catch(() => undefined);
    setFileName(file.name);
    setResult({ bikes: 0, riders: 0, noHelmet: 0, triple: 0, took: 0 });
    helmetHistory.current = [];
    setModelState("Video selected — loading first frame");
  };

  const useSample = () => {
    const video = videoRef.current;
    if (!video || !modelsReady) return;
    if (objectUrl.current) URL.revokeObjectURL(objectUrl.current);
    objectUrl.current = null;
    video.src = "/sample-traffic.mp4";
    video.load();
    const overlay = overlayRef.current;
    if (overlay) overlay.getContext("2d")?.clearRect(0, 0, overlay.width, overlay.height);
    void video.play().catch(() => undefined);
    setFileName("Built-in road sample");
    setResult({ bikes: 0, riders: 0, noHelmet: 0, triple: 0, took: 0 });
    helmetHistory.current = [];
  };

  return (
    <div>
      <div className={`model-wait-note ${modelsReady ? "ready" : "loading"}`} role="status">
        <span aria-hidden="true" />
        {modelsReady
          ? "Models ready. You can play the sample or choose your own video now."
          : "Please wait until both models finish loading. The step being loaded is shown on the right."}
      </div>
      <div className="live-detector">
        <div className="live-video-column">
          <div className="live-video-stage">
            <video ref={videoRef} src="/sample-traffic.mp4" controls muted loop playsInline preload="metadata" onPlay={(event) => { if (!modelsReady) event.currentTarget.pause(); }} onLoadedData={() => void analyse()} onSeeked={() => void analyse()} />
            <canvas ref={overlayRef} />
          </div>
          <div className="video-picker">
            <label className={`upload-button ${modelsReady ? "" : "disabled"}`} aria-disabled={!modelsReady}>
              Choose your video
              <input disabled={!modelsReady} type="file" accept="video/mp4,video/webm,video/ogg,video/*" onChange={(event) => chooseVideo(event.target.files?.[0])} />
            </label>
            <button disabled={!modelsReady} type="button" onClick={useSample}>Use sample</button>
            <span title={fileName}>{fileName}</span>
          </div>
        </div>
        <aside className="live-readout" aria-live="polite">
          <span className="label">What it found</span>
          <h3>{modelState}</h3>
          <dl>
            <div><dt>{result.bikes}</dt><dd>motorcycles</dd></div>
            <div><dt>{result.riders}</dt><dd>close riders checked</dd></div>
            <div><dt>{result.noHelmet}</dt><dd>confirmed no-helmet cases</dd></div>
            <div><dt>{result.triple}</dt><dd>possible triple riding</dd></div>
          </dl>
          <p>{result.took ? `Last frame took ${(result.took / 1000).toFixed(1)} seconds on this device.` : "Nothing has been checked yet."}</p>
          <p className="run-caveat">Only close results that repeat with strong confidence are labelled. Your video stays on this device.</p>
        </aside>
      </div>
    </div>
  );
}

function App() {
  const [scenarioId, setScenarioId] = useState<ScenarioId>("combined");
  const [progress, setProgress] = useState(0);
  const [running, setRunning] = useState(false);
  const [events, setEvents] = useState<EventItem[]>([]);
  const emitted = useRef(new Set<Rule>());
  const scenario = useMemo(
    () => scenarios.find((item) => item.id === scenarioId) ?? scenarios[0],
    [scenarioId],
  );

  const reset = () => {
    setRunning(false);
    setProgress(0);
    setEvents([]);
    emitted.current = new Set();
  };

  useEffect(() => {
    reset();
  }, [scenarioId]);

  useEffect(() => {
    if (!running) return;
    const timer = window.setInterval(() => {
      setProgress((value) => {
        const next = Math.min(100, value + 1);
        if (next >= 100) setRunning(false);
        return next;
      });
    }, 55);
    return () => window.clearInterval(timer);
  }, [running]);

  useEffect(() => {
    scenario.expected.forEach((rule) => {
      const copy = ruleCopy[rule];
      if (progress >= copy.threshold && !emitted.current.has(rule)) {
        emitted.current.add(rule);
        setEvents((current) => [
          ...current,
          {
            rule,
            frame: Math.round(progress * 1.8),
            confidence: copy.confidence,
            detail: copy.detail,
          },
        ]);
      }
    });
  }, [progress, scenario]);

  const start = () => {
    if (progress >= 100) reset();
    window.setTimeout(() => setRunning(true), 0);
  };

  const bikeX = 20 + progress * 7.25;
  const crossed = bikeX + 115 >= 650;
  const status = running ? "Analysing frames" : progress >= 100 ? "Run complete" : "Ready";

  return (
    <div className="app-shell">
      <header className="site-header">
        <a className="brand" href="#top" aria-label="Smart Helmet Vision home">
          <span className="brand-mark" aria-hidden="true">
            <svg viewBox="0 0 32 32">
              <path d="M6 18c0-7 4-12 11-12 6 0 10 5 10 11v2h-9c-2 0-3 1-3 3v4H9v-5c-2 0-3-1-3-3Z" />
              <path d="M18 19h9" />
            </svg>
          </span>
          <span>Smart Helmet Vision</span>
        </a>
        <nav aria-label="Main navigation">
          <a href="#real-demo">Real video</a>
          <a href="#logic-demo">Rule demo</a>
          <a href="#system">How it works</a>
          <a href="#research">Research</a>
        </nav>
      </header>

      <main id="top">
        <section className="hero">
          <div className="eyebrow">Raspberry Pi + camera + computer vision</div>
          <h1>A small traffic-video detector built for the internship task.</h1>
          <p className="hero-copy">
            It looks for motorcycles, riders and helmet use in an ordinary video. Try the road sample first,
            then choose a file from your own device. The signal-jumping rule is shown separately below.
          </p>
          <div className="hero-actions">
            <a className="primary-link" href="#real-demo">Watch the real run</a>
            <a className="secondary-link" href="#system">See the pipeline</a>
          </div>
          <div className="truth-note">
            <strong>Current status:</strong> uploadable video inference and the rule engine are working. Raspberry Pi benchmarking still needs the actual hardware.
          </div>
        </section>

        <section className="real-demo-section" id="real-demo">
          <div className="section-heading">
            <div><span className="section-number">01</span><h2>Try it with a video</h2></div>
            <span className="real-badge"><i /> RUNS IN THIS TAB</span>
          </div>
          <LiveVideoDetector />
          <p className="video-credit">The built-in sample is “Busy Indian Street with Traffic and Motorbikes” by Aamir Somewhere, used under the Pexels license. You can replace it with your own video above.</p>
        </section>

        <section className="demo-section" id="logic-demo">
          <div className="section-heading">
            <div>
              <span className="section-number">02</span>
              <h2>Rule logic sandbox</h2>
            </div>
            <span className={`run-status ${running ? "live" : ""}`}><i />{status}</span>
          </div>

          <div className="scenario-bar">
            <label htmlFor="scenario">Choose a test case</label>
            <select id="scenario" value={scenarioId} onChange={(event) => setScenarioId(event.target.value as ScenarioId)}>
              {scenarios.map((item) => <option key={item.id} value={item.id}>{item.name}</option>)}
            </select>
            <p>{scenario.note}</p>
            <div className="control-row">
              <button className="run-button" type="button" onClick={running ? () => setRunning(false) : start}>
                {running ? "Pause" : progress > 0 && progress < 100 ? "Continue" : "Start run"}
              </button>
              <button className="reset-button" type="button" onClick={reset}>Reset</button>
            </div>
          </div>

          <div className="demo-grid">
            <div className="camera-panel">
              <div className="camera-meta">
                <span>CAM-01 / FIXED VIEW</span>
                <span>FRAME {Math.round(progress * 1.8).toString().padStart(3, "0")}</span>
              </div>
              <svg className="traffic-scene" viewBox="0 0 900 470" role="img" aria-label="Animated traffic camera simulation">
                <rect width="900" height="470" fill="#dfe5e8" />
                <rect y="0" width="900" height="150" fill="#edf0f1" />
                <rect x="36" y="54" width="156" height="96" fill="#c5ced2" />
                <rect x="208" y="28" width="104" height="122" fill="#b8c4c9" />
                <rect x="328" y="66" width="190" height="84" fill="#c9d1d4" />
                <rect x="534" y="39" width="130" height="111" fill="#bac5c8" />
                <rect y="150" width="900" height="42" fill="#aeb9b4" />
                <rect y="192" width="900" height="278" fill="#30373b" />
                <path d="M0 356 H900" stroke="#f2d45c" strokeWidth="5" strokeDasharray="34 28" />
                <path d="M650 202 V459" stroke="#f5f3e8" strokeWidth="12" strokeDasharray="18 12" />
                <text x="622" y="447" fill="#cfd5d7" fontSize="13" transform="rotate(-90 622 447)">STOP LINE</text>

                <g transform="translate(730 68)">
                  <rect x="20" y="0" width="46" height="110" rx="7" fill="#20272b" />
                  <circle cx="43" cy="24" r="13" fill={scenario.redSignal ? "#dc3c35" : "#6a2625"} />
                  <circle cx="43" cy="55" r="13" fill="#6c5d23" />
                  <circle cx="43" cy="86" r="13" fill={scenario.redSignal ? "#1f5138" : "#36b875"} />
                  <rect x="39" y="110" width="8" height="102" fill="#4e585c" />
                </g>

                <g transform={`translate(${bikeX} 0)`}>
                  <rect x="5" y="248" width="125" height="145" fill="none" stroke="#62c7ef" strokeWidth="2" strokeDasharray="7 4" />
                  <rect x="7" y="228" width="96" height="18" fill="#162d3b" />
                  <text x="12" y="241" fill="#ffffff" fontSize="12">motorbike #07 0.95</text>
                  <circle cx="30" cy="391" r="27" fill="#14191c" stroke="#8b969b" strokeWidth="7" />
                  <circle cx="112" cy="391" r="27" fill="#14191c" stroke="#8b969b" strokeWidth="7" />
                  <path d="M30 390 L59 345 L94 388 L112 390 M59 345 L106 341" fill="none" stroke="#d85a46" strokeWidth="9" strokeLinecap="round" />
                  <path d="M94 388 L74 314" stroke="#d85a46" strokeWidth="8" />

                  {Array.from({ length: scenario.riders }).map((_, index) => {
                    const riderX = 55 + index * 28;
                    const noHelmet = scenario.missingHelmet && index === scenario.riders - 1;
                    return (
                      <g key={index}>
                        <rect x={riderX - 18} y={240 - index * 3} width="40" height="101" fill="none" stroke={noHelmet ? "#ef5c55" : "#68d596"} strokeWidth="2" />
                        <circle cx={riderX} cy={263 - index * 3} r="17" fill={noHelmet ? "#bb8064" : "#f2c230"} stroke="#22292c" strokeWidth="3" />
                        <path d={`M${riderX} ${280 - index * 3} L${riderX + 4} ${326 - index * 3}`} stroke="#56707c" strokeWidth="17" strokeLinecap="round" />
                        <path d={`M${riderX + 2} ${321 - index * 3} L${riderX - 7} 361`} stroke="#22292c" strokeWidth="8" strokeLinecap="round" />
                        {noHelmet && <text x={riderX - 17} y={231 - index * 3} fill="#ffffff" fontSize="10">no helmet</text>}
                      </g>
                    );
                  })}
                </g>

                {crossed && scenario.redSignal && (
                  <g>
                    <rect x="24" y="24" width="310" height="48" fill="#a52f2b" />
                    <text x="42" y="55" fill="#fff" fontSize="20" fontWeight="700">REVIEW EVENT CREATED</text>
                  </g>
                )}
              </svg>
              <div className="timeline"><span style={{ width: `${progress}%` }} /></div>
            </div>

            <aside className="event-panel" aria-live="polite">
              <div className="event-panel-head">
                <div>
                  <small>TRACK #07</small>
                  <h3>Event log</h3>
                </div>
                <strong>{events.length}</strong>
              </div>
              {events.length === 0 ? (
                <div className="empty-events">
                  <span>Waiting for a confirmed condition.</span>
                  <p>The rule engine checks several frames before adding an event.</p>
                </div>
              ) : (
                <ol className="event-list">
                  {events.map((event) => (
                    <li key={event.rule}>
                      <div className="event-title"><span>{ruleCopy[event.rule].short}</span><b>{event.confidence.toFixed(2)}</b></div>
                      <p>{event.detail}</p>
                      <small>Frame {event.frame} · emitted once</small>
                    </li>
                  ))}
                </ol>
              )}
              {progress >= 100 && scenario.expected.length === 0 && (
                <div className="safe-result">No violation event generated.</div>
              )}
            </aside>
          </div>
        </section>

        <section className="system-section" id="system">
          <div className="section-heading">
            <div><span className="section-number">03</span><h2>How the complete system works</h2></div>
          </div>
          <div className="pipeline">
            {[
              ["01", "Detect", "A small custom model finds the motorcycle and each rider's helmet state."],
              ["02", "Track", "Persistent IDs follow the same motorcycle across consecutive frames."],
              ["03", "Associate", "Spatial checks decide which riders belong to which motorcycle."],
              ["04", "Confirm", "A temporal rule engine removes one-frame noise and duplicate events."],
              ["05", "Record", "The system stores the frame, rule, track ID and confidence for review."],
            ].map(([number, title, copy]) => (
              <article key={number}><span>{number}</span><h3>{title}</h3><p>{copy}</p></article>
            ))}
          </div>
          <div className="implementation-note">
            <div>
              <span className="label">Web demo</span>
              <p>Runs both models on a selected video and keeps a deterministic sandbox for reviewing each rule.</p>
            </div>
            <div>
              <span className="label">Python prototype</span>
              <p>Accepts a camera or video, an Ultralytics tracking model, and writes JSON/image evidence.</p>
            </div>
            <div>
              <span className="label">Raspberry Pi target</span>
              <p>Train off-device, export a small model to NCNN, then measure FPS and latency on the Pi.</p>
            </div>
          </div>
        </section>

        <section className="research-section" id="research">
          <div className="section-heading">
            <div><span className="section-number">04</span><h2>Research used</h2></div>
          </div>
          <p className="research-intro">I used official or primary sources for the dataset, Raspberry Pi deployment path and Indian traffic-rule context.</p>
          <div className="source-list">
            <a href="https://www.aicitychallenge.org/2024-data-and-evaluation/" target="_blank" rel="noreferrer"><span>Dataset</span><strong>AI City Challenge 2024 · Helmet-rule Track 5</strong><i>↗</i></a>
            <a href="https://www.raspberrypi.com/documentation/accessories/ai-camera.html" target="_blank" rel="noreferrer"><span>Hardware</span><strong>Raspberry Pi AI Camera documentation</strong><i>↗</i></a>
            <a href="https://docs.ultralytics.com/guides/raspberry-pi" target="_blank" rel="noreferrer"><span>Deployment</span><strong>Ultralytics Raspberry Pi and NCNN guide</strong><i>↗</i></a>
            <a href="https://huggingface.co/iam-tsr/yolov8n-helmet-detection" target="_blank" rel="noreferrer"><span>Demo model</span><strong>Lightweight YOLOv8n helmet model · MIT licensed</strong><i>↗</i></a>
            <a href="https://www.pexels.com/video/busy-indian-street-with-traffic-and-motorbikes-34394424/" target="_blank" rel="noreferrer"><span>Demo video</span><strong>Pexels traffic footage · Aamir Somewhere</strong><i>↗</i></a>
            <a href="https://www.indiacode.nic.in/handle/123456789/13700?locale=en" target="_blank" rel="noreferrer"><span>Rules</span><strong>Motor Vehicles Act · Sections 128 and 129</strong><i>↗</i></a>
          </div>
        </section>

        <section className="closing">
          <p><strong>Responsible-use boundary:</strong> this prototype creates candidate events for human review. It is not an automatic challan or identity system.</p>
        </section>
      </main>

      <footer><span>Smart Helmet Vision · Internship prototype</span><span>Rule engine tested with six automated cases</span></footer>
    </div>
  );
}

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <App />
  </StrictMode>,
);

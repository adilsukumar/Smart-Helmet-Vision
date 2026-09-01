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
          <h1>Motorcycle safety events, explained frame by frame.</h1>
          <p className="hero-copy">
            A working computer-vision prototype for helmet use, triple riding and red-light crossing.
            The page now includes a real traffic-video run as well as a small rule sandbox.
          </p>
          <div className="hero-actions">
            <a className="primary-link" href="#real-demo">Watch the real run</a>
            <a className="secondary-link" href="#system">See the pipeline</a>
          </div>
          <div className="truth-note">
            <strong>Current status:</strong> real footage processed with two models, rule engine tested, and browser demo working. Raspberry Pi benchmarking still needs the actual hardware.
          </div>
        </section>

        <section className="real-demo-section" id="real-demo">
          <div className="section-heading">
            <div><span className="section-number">01</span><h2>Real traffic run</h2></div>
            <span className="real-badge"><i /> PROCESSED FOOTAGE</span>
          </div>
          <div className="real-demo-grid">
            <div className="video-wrap">
              <video controls muted loop playsInline preload="metadata" poster="/real-traffic-poster.jpg">
                <source src="/real-traffic-demo.mp4" type="video/mp4" />
                Your browser could not play this video.
              </video>
            </div>
            <aside className="run-notes">
              <span className="label">What actually ran</span>
              <h3>12.5 seconds, two models, full clip</h3>
              <p>A YOLO11 model tracks motorcycles and people. A separate helmet model marks each associated rider as helmet, no helmet, or uncertain.</p>
              <dl>
                <div><dt>4</dt><dd>no-helmet candidates</dd></div>
                <div><dt>2</dt><dd>triple-riding candidates</dd></div>
                <div><dt>0</dt><dd>signal events evaluated</dd></div>
              </dl>
              <p className="run-caveat">These are candidate events, not ground truth. The crowded right edge produces uncertain rider-to-bike associations, so a person still needs to review them. Signal jumping is not evaluated because this clip has no usable traffic signal and stop line.</p>
            </aside>
          </div>
          <p className="video-credit">Footage: “Busy Indian Street with Traffic and Motorbikes” by Aamir Somewhere, used under the Pexels license.</p>
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
              <p>Shows the processed real clip and keeps a deterministic sandbox for reviewing each rule.</p>
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
            <a href="https://huggingface.co/nnsohamnn/helmet-detection-yolo11" target="_blank" rel="noreferrer"><span>Demo model</span><strong>Helmet Detection YOLO11 · MIT licensed weights</strong><i>↗</i></a>
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

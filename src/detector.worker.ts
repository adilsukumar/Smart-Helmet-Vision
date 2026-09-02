import * as ort from "onnxruntime-web/wasm";

let traffic: ort.InferenceSession | null = null;
let helmet: ort.InferenceSession | null = null;

self.onmessage = async (event) => {
  const message = event.data;
  try {
    if (message.type === "load") {
      ort.env.wasm.numThreads = 1;
      ort.env.wasm.proxy = false;
      self.postMessage({ type: "progress", message: "Loading traffic model — 1 of 2" });
      traffic = await ort.InferenceSession.create("/models/traffic.onnx", { executionProviders: ["wasm"] });
      self.postMessage({ type: "progress", message: "Traffic model ready — loading helmet model 2 of 2" });
      helmet = await ort.InferenceSession.create("/models/helmet.onnx", { executionProviders: ["wasm"] });
      self.postMessage({ type: "ready" });
      return;
    }

    if (message.type === "run") {
      const session = message.model === "traffic" ? traffic : helmet;
      if (!session) throw new Error("Models are not ready");
      const input = new ort.Tensor("float32", message.data, [1, 3, 320, 320]);
      const output = await session.run({ images: input });
      const tensor = output[session.outputNames[0]];
      const data = tensor.data as Float32Array;
      self.postMessage({ type: "result", id: message.id, data, dims: Array.from(tensor.dims) }, [data.buffer]);
    }
  } catch (error) {
    self.postMessage({ type: "error", id: message.id, message: error instanceof Error ? error.message : "Detection failed" });
  }
};

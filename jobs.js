const express = require("express");
const fs = require("fs");
const fsp = require("fs/promises");
const path = require("path");
const cors = require("cors");

const app = express();
const PORT = 8000;
const LOG_PATH = path.join(__dirname, "jobs_list.txt");

app.use(cors({
  origin: [
    "http://localhost:5173",
  ],
  credentials: true
}));

// --------------------------------------------------
// Helpers
// --------------------------------------------------

async function readState(filePath) {
  const state = {};
  const content = await fsp.readFile(filePath, "utf8");
  const lines = content.split("\n");

  for (let line of lines) {
    line = line.trim();
    if (!line) continue;
    // if (line.includes("VV3Q")) continue;

    // line = line.replace("NQKE", "-1").trim();
    const parts = line.split(",");
    const queue = parts[0];

    const metrics = parts.slice(1)
    state[queue] = metrics;
  }
  return state;
}

function diffStates(oldState, newState) {
  const deltas = [];

  for (const queue in newState) {
    const newMetrics = newState[queue];
    const oldMetrics = oldState[queue];

    if (!oldMetrics) {
      deltas.push({ queue, type: "new", metrics: newMetrics });
      continue;
    }

    const changes = [];
    newMetrics.forEach((n, i) => {
      if (oldMetrics[i] !== n) {
        changes.push({ index: i, old: oldMetrics[i], new: n });
      }
    });

    if (changes.length > 0) {
      deltas.push({ queue, type: "update", changes });
    }
  }

  return deltas;
}

// --------------------------------------------------
// SSE Clients Store
// --------------------------------------------------

const clients = new Set();

function broadcast(event, data) {
  const payload = `event: ${event}\ndata: ${JSON.stringify(data)}\n\n`;
  for (const res of clients) {
    res.write(payload);
  }
}

// --------------------------------------------------
// File Watcher (Production Pattern)
// --------------------------------------------------

let prevState = {};
let debounceTimer = null;

async function processFileChange() {
  try {
    const currState = await readState(LOG_PATH);
    const deltas = diffStates(prevState, currState);

    deltas.forEach(d => broadcast("delta", d));
    prevState = currState;

  } catch (err) {
    broadcast("error", { error: err.message });
  }
}

function startWatcher() {
  if (!fs.existsSync(LOG_PATH)) {
    console.error("File not found:", LOG_PATH);
    return;
  }

  // Initial snapshot
  readState(LOG_PATH).then(state => {
    prevState = state;
  });

  fs.watch(LOG_PATH, (eventType) => {
    if (eventType === "change") {
      // Debounce (important for multi-write bursts)
      if (debounceTimer) clearTimeout(debounceTimer);

      debounceTimer = setTimeout(() => {
        processFileChange();
      }, 100);
    }
  });

  console.log("File watcher started.");
}

// --------------------------------------------------
// SSE Endpoint
// --------------------------------------------------

app.get("/events", async (req, res) => {

  res.setHeader("Content-Type", "text/event-stream");
  res.setHeader("Cache-Control", "no-cache");
  res.setHeader("Connection", "keep-alive");

  res.flushHeaders?.();

  clients.add(res);

  // Send initial snapshot
  if (Object.keys(prevState).length > 0) {
    res.write(`event: snapshot\ndata: ${JSON.stringify(prevState)}\n\n`);
  }

  // Keep-alive ping
  const pingInterval = setInterval(() => {
    res.write(": ping\n\n");
  }, 15000);

  req.on("close", () => {
    clearInterval(pingInterval);
    clients.delete(res);
  });
});

// --------------------------------------------------

startWatcher();

app.listen(PORT, () => {
  console.log(`Production SSE server running at http://localhost:${PORT}`);
});

const express = require("express");
const fs = require("fs");
const fsp = require("fs/promises");
const path = require("path");
const cors = require("cors");

const app = express();
const PORT = 8000;

const FILES = {
  jobs: path.join(__dirname, "jobs_list.txt"),
  queue: path.join(__dirname, "queue_file.txt")
};

app.use(cors({
  origin: ["http://localhost:5173"],
  credentials: true
}));

// --------------------------------------------------
// Generic Diff
// --------------------------------------------------

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
// File Specific Readers (Logic Intact)
// --------------------------------------------------

async function readJobsState(filePath) {
  const state = {};
  const content = await fsp.readFile(filePath, "utf8");
  const lines = content.split("\n");

  for (let line of lines) {
    line = line.trim();
    if (!line) continue;

    const parts = line.split(",");
    const queue = parts[0];
    const metrics = parts.slice(1);

    state[queue] = metrics;
  }

  return state;
}

async function readQueueState(filePath) {
  const state = {};
  const content = await fsp.readFile(filePath, "utf8");
  const lines = content.split("\n");

  for (let line of lines) {
    line = line.trim();
    if (!line) continue;
    if (line.includes("VV3Q")) continue;

    line = line.replace("NQKE", "-1").trim();
    const parts = line.split(",");
    const queue = parts[0];

    const metrics = parts.slice(1).map(v => {
      const num = parseInt(v.trim(), 10);
      if (isNaN(num)) return 0;
      return (num >= 1 && num < 150) ? 0 : num;
    });

    state[queue] = metrics;
  }

  return state;
}

// --------------------------------------------------
// SSE Clients
// --------------------------------------------------

const clients = new Set();

function broadcast(event, data) {
  const payload = `event: ${event}\ndata: ${JSON.stringify(data)}\n\n`;
  for (const res of clients) {
    res.write(payload);
  }
}

// --------------------------------------------------
// Watcher Factory (Reusable)
// --------------------------------------------------

function createWatcher(name, filePath, readerFn) {
  let prevState = {};
  let debounceTimer = null;

  async function processFileChange() {
    try {
      const currState = await readerFn(filePath);
      const deltas = diffStates(prevState, currState);

      deltas.forEach(d =>
        broadcast("delta", { source: name, ...d })
      );

      prevState = currState;
    } catch (err) {
      broadcast("error", { source: name, error: err.message });
    }
  }

  function start() {
    if (!fs.existsSync(filePath)) {
      console.error(`File not found: ${filePath}`);
      return;
    }

    // Initial snapshot
    readerFn(filePath).then(state => {
      prevState = state;
    });

    fs.watch(filePath, (eventType) => {
      if (eventType === "change") {
        if (debounceTimer) clearTimeout(debounceTimer);

        debounceTimer = setTimeout(() => {
          processFileChange();
        }, 100);
      }
    });

    console.log(`${name} watcher started.`);
  }

  return {
    start,
    getState: () => prevState
  };
}

// --------------------------------------------------
// Create Both Watchers
// --------------------------------------------------

const jobsWatcher = createWatcher("jobs", FILES.jobs, readJobsState);
const queueWatcher = createWatcher("queue", FILES.queue, readQueueState);

jobsWatcher.start();
queueWatcher.start();

// --------------------------------------------------
// SSE Endpoint
// --------------------------------------------------

app.get("/events", (req, res) => {

  res.setHeader("Content-Type", "text/event-stream");
  res.setHeader("Cache-Control", "no-cache");
  res.setHeader("Connection", "keep-alive");

  res.flushHeaders?.();
  clients.add(res);

  // Send initial snapshot of BOTH files
  res.write(`event: snapshot\ndata: ${JSON.stringify({
    jobs: jobsWatcher.getState(),
    queue: queueWatcher.getState()
  })}\n\n`);

  const pingInterval = setInterval(() => {
    res.write(": ping\n\n");
  }, 15000);

  req.on("close", () => {
    clearInterval(pingInterval);
    clients.delete(res);
  });
});

// --------------------------------------------------

app.listen(PORT, () => {
  console.log(`SSE server running at http://localhost:${PORT}`);
});
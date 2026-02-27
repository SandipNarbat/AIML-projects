const express = require("express");
const fs = require("fs");
const fsp = require("fs/promises");
const path = require("path");
const cors = require("cors");

const app = express();
const PORT = 8000;

app.use(cors({
  origin: ["http://localhost:5173"],
  credentials: true
}));

// --------------------------------------------------
// Diff Engine
// --------------------------------------------------

function diffStates(oldState, newState) {
  const deltas = [];

  for (const key in newState) {
    const newMetrics = newState[key];
    const oldMetrics = oldState[key];

    if (!oldMetrics) {
      deltas.push({ key, type: "new", metrics: newMetrics });
      continue;
    }

    const changes = [];

    if (Array.isArray(newMetrics)) {
      newMetrics.forEach((n, i) => {
        if (oldMetrics[i] !== n) {
          changes.push({ index: i, old: oldMetrics[i], new: n });
        }
      });
    } else {
      if (JSON.stringify(oldMetrics) !== JSON.stringify(newMetrics)) {
        changes.push({ old: oldMetrics, new: newMetrics });
      }
    }

    if (changes.length > 0) {
      deltas.push({ key, type: "update", changes });
    }
  }

  return deltas;
}

// --------------------------------------------------
// Readers
// --------------------------------------------------

async function readJobsState(filePath) {
  const state = {};
  const content = await fsp.readFile(filePath, "utf8");

  content.split("\n").forEach(line => {
    line = line.trim();
    if (!line) return;

    const parts = line.split(",");
    state[parts[0]] = parts.slice(1);
  });

  return state;
}

async function readQueueState(filePath) {
  const state = {};
  const content = await fsp.readFile(filePath, "utf8");

  content.split("\n").forEach(line => {
    line = line.trim();
    if (!line) return;
    if (line.includes("VV3Q")) return;

    line = line.replace("NQKE", "-1");

    const parts = line.split(",");
    const key = parts[0];

    const metrics = parts.slice(1).map(v => {
      const num = parseInt(v.trim(), 10);
      if (isNaN(num)) return 0;
      return (num >= 1 && num < 150) ? 0 : num;
    });

    state[key] = metrics;
  });

  return state;
}
async function readPaymentsState(filePath) {
  const state = {};
  const content = await fsp.readFile(filePath, "utf8");

  content.split("\n").forEach(line => {
    line = line.trim();
    if (!line) return;

    const [id, amount, status] = line.split(",");

    state[id] = {
      amount: Number(amount),
      status: status?.trim()
    };
  });

  return state;
}

// --------------------------------------------------
// Watcher Factory (Now Per-Source Clients)
// --------------------------------------------------

function createWatcher(name, filePath, readerFn) {
  let prevState = {};
  let debounceTimer = null;
  const clients = new Set();

  function broadcast(event, data) {
    const payload = `event: ${event}\ndata: ${JSON.stringify(data)}\n\n`;
    for (const res of clients) {
      res.write(payload);
    }
  }

  async function processFileChange() {
    try {
      const currState = await readerFn(filePath);
      const deltas = diffStates(prevState, currState);

      deltas.forEach(delta => broadcast("delta", delta));

      prevState = currState;

    } catch (err) {
      broadcast("error", { error: err.message });
    }
  }

  function start() {
    if (!fs.existsSync(filePath)) {
      console.error(`File not found: ${filePath}`);
      return;
    }

    readerFn(filePath).then(state => {
      prevState = state;
    });

    fs.watch(filePath, (eventType) => {
      if (eventType === "change") {
        if (debounceTimer) clearTimeout(debounceTimer);
        debounceTimer = setTimeout(processFileChange, 100);
      }
    });

    console.log(`${name} watcher started`);
  }

  function addClient(res) {
    clients.add(res);
  }

  function removeClient(res) {
    clients.delete(res);
  }

  function getState() {
    return prevState;
  }

  return {
    start,
    addClient,
    removeClient,
    getState
  };
}

// --------------------------------------------------
// File Configuration
// --------------------------------------------------

const FILE_CONFIG = [
  { name: "jobs", path: "jobs_list.txt", reader: readJobsState },
  { name: "queue", path: "queue_file.txt", reader: readQueueState },
  { name: "payments", path: "payment.txt", reader: readPaymentsState },
];
const watchers = {};

FILE_CONFIG.forEach(config => {
  const fullPath = path.join(__dirname, config.path);
  const watcher = createWatcher(config.name, fullPath, config.reader);
  watcher.start();
  watchers[config.name] = watcher;
});

// --------------------------------------------------
// Dynamic Per-File SSE Route
// --------------------------------------------------

app.get("/events/:source", (req, res) => {

  const { source } = req.params;
  const watcher = watchers[source];

  if (!watcher) {
    return res.status(404).json({ error: "Invalid source" });
  }

  res.setHeader("Content-Type", "text/event-stream");
  res.setHeader("Cache-Control", "no-cache");
  res.setHeader("Connection", "keep-alive");

  res.flushHeaders?.();
  watcher.addClient(res);

  // Send initial snapshot for this file only
  res.write(`event: snapshot\ndata: ${JSON.stringify(watcher.getState())}\n\n`);

  const pingInterval = setInterval(() => {
    res.write(": ping\n\n");
  }, 15000);

  req.on("close", () => {
    clearInterval(pingInterval);
    watcher.removeClient(res);
  });
});

// --------------------------------------------------

app.listen(PORT, () => {
  console.log(`SSE server running at http://localhost:${PORT}`);
});
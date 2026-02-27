import { useEffect, useRef, useState } from "react";

const SOURCES = ["jobs", "queue", "payments"];
const BASE_URL = "http://localhost:8000/events";

export default function Dashboard() {
  const [data, setData] = useState({});
  const [status, setStatus] = useState({});
  const reconnectTimers = useRef({});

  const applyDelta = (source, delta) => {
    setData((prev) => {
      const updated = { ...prev };
      const sourceData = { ...(updated[source] || {}) };

      if (delta.type === "new") {
        sourceData[delta.key] = delta.metrics;
      }

      if (delta.type === "update") {
        if (Array.isArray(sourceData[delta.key])) {
          const arr = [...sourceData[delta.key]];
          delta.changes.forEach((change) => {
            arr[change.index] = change.new;
          });
          sourceData[delta.key] = arr;
        } else {
          sourceData[delta.key] = delta.changes[0].new;
        }
      }

      updated[source] = sourceData;
      return updated;
    });
  };

  const connectToSource = (source, retry = 0) => {
    setStatus((prev) => ({ ...prev, [source]: "connecting" }));

    const es = new EventSource(`${BASE_URL}/${source}`);

    es.onopen = () => {
      setStatus((prev) => ({ ...prev, [source]: "connected" }));
    };

    es.onerror = () => {
      es.close();
      setStatus((prev) => ({ ...prev, [source]: "disconnected" }));

      const timeout = Math.min(5000, 1000 * (retry + 1));
      reconnectTimers.current[source] = setTimeout(() => {
        connectToSource(source, retry + 1);
      }, timeout);
    };

    es.addEventListener("snapshot", (e) => {
      setData((prev) => ({
        ...prev,
        [source]: JSON.parse(e.data),
      }));
    });

    es.addEventListener("delta", (e) => {
      const delta = JSON.parse(e.data);
      applyDelta(source, delta);
    });

    return es;
  };

  useEffect(() => {
    const connections = {};

    SOURCES.forEach((source) => {
      connections[source] = connectToSource(source);
    });

    return () => {
      Object.values(connections).forEach((es) => es.close());
      Object.values(reconnectTimers.current).forEach(clearTimeout);
    };
  }, []);

  const renderTable = (source) => {
    const sourceData = data[source] || {};
    const keys = Object.keys(sourceData);

    if (!keys.length) {
      return (
        <div className="card">
          <div className="card-header">
            <h2>{source.toUpperCase()}</h2>
            <span className={`status ${status[source]}`}>
              {status[source] || "connecting"}
            </span>
          </div>
          <p className="empty">No Data</p>
        </div>
      );
    }

    const firstRow = sourceData[keys[0]];
    const headers = Array.isArray(firstRow)
      ? firstRow.map((_, i) => `Col ${i + 1}`)
      : Object.keys(firstRow);

    return (
      <div className="card">
        <div className="card-header">
          <h2>{source.toUpperCase()}</h2>
          <span className={`status ${status[source]}`}>
            {status[source] || "connecting"}
          </span>
        </div>

        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Key</th>
                {headers.map((h) => (
                  <th key={h}>{h}</th>
                ))}
              </tr>
            </thead>

            <tbody>
              {keys.map((key) => (
                <tr key={key}>
                  <td>{key}</td>
                  {Array.isArray(sourceData[key])
                    ? sourceData[key].map((val, i) => (
                        <AnimatedCell key={i} value={val} />
                      ))
                    : Object.values(sourceData[key]).map((val, i) => (
                        <AnimatedCell key={i} value={val} />
                      ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    );
  };

  return (
    <div className="dashboard">
      <div className="top-row">
        {renderTable("jobs")}
        {renderTable("queue")}
      </div>
      <div className="bottom-row">{renderTable("payments")}</div>
      <style>{styles}</style>
    </div>
  );
}

function AnimatedCell({ value }) {
  const [highlight, setHighlight] = useState(false);
  const prev = useRef(value);

  useEffect(() => {
    if (prev.current !== value) {
      setHighlight(true);
      const timer = setTimeout(() => setHighlight(false), 800);
      prev.current = value;
      return () => clearTimeout(timer);
    }
  }, [value]);

  return <td className={highlight ? "highlight" : ""}>{String(value)}</td>;
}

const styles = `
.dashboard {
  background: #0f172a;
  min-height: 100vh;
  width: 100%;
  padding: 20px;
  display: grid;
  gap: 20px;
  font-family: Inter, sans-serif;
  color: #e2e8f0;
}

.top-row {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 20px;
}

.bottom-row {
  min-width: 0;
}

.card {
  background: #1e293b;
  border-radius: 14px;
  padding: 16px;
  box-shadow: 0 10px 30px rgba(0,0,0,0.4);
  min-width: 0;
}

.card-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 15px;
  gap: 10px;
}

h2 {
  font-size: 18px;
  letter-spacing: 1px;
  margin: 0;
}

.status {
  padding: 4px 10px;
  border-radius: 12px;
  font-size: 12px;
  text-transform: uppercase;
  white-space: nowrap;
}

.status.connected {
  background: #065f46;
  color: #34d399;
}

.status.connecting {
  background: #78350f;
  color: #facc15;
}

.status.disconnected {
  background: #7f1d1d;
  color: #f87171;
}

.table-wrap {
  width: 100%;
  overflow-x: auto;
  -webkit-overflow-scrolling: touch;
}

table {
  width: max-content;
  min-width: 100%;
  border-collapse: collapse;
}

th,
td {
  padding: 8px 10px;
  text-align: left;
  font-size: 13px;
  white-space: nowrap;
}

th {
  background: #334155;
  border-radius: 6px;
}

tbody tr {
  border-bottom: 1px solid #334155;
}

tbody tr:hover {
  background: rgba(255,255,255,0.05);
}

.highlight {
  animation: flash 0.8s ease;
}

@keyframes flash {
  0% {
    background-color: #16a34a;
  }
  100% {
    background-color: transparent;
  }
}

.empty {
  opacity: 0.6;
  font-style: italic;
  margin: 0;
}

@media (max-width: 900px) {
  .top-row {
    grid-template-columns: 1fr;
  }
}

@media (max-width: 768px) {
  .dashboard {
    padding: 12px;
    gap: 12px;
  }

  .top-row {
    gap: 12px;
  }

  .card {
    padding: 12px;
  }

  th,
  td {
    padding: 6px 8px;
    font-size: 12px;
  }
}
`;
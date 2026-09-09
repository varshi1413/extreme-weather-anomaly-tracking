function ThreatPanel({ anomalies, selected, onSelect, onClose }) {
  function severityClass(severity) {
    if (severity === "severe") return "sev-severe";
    if (severity === "moderate") return "sev-moderate";
    return "sev-low";
  }

  function severityColor(severity) {
    if (severity === "severe") return "#FF3B5C";
    if (severity === "moderate") return "#FFC94A";
    return "#37D67A";
  }

  return (
    <aside className="panel threats-panel">
      <div className="panel-head">
        <h2>Tracked Anomalies</h2>

        <div className="panel-head-actions">
          <span className="panel-tag">
            {String(anomalies.length).padStart(2, "0")} TRACKED
          </span>

          {onClose && (
            <button
              className="panel-close-btn"
              onClick={onClose}
              aria-label="Close panel"
            >
              ✕
            </button>
          )}
        </div>
      </div>

      <div className="threats-list">
        {anomalies.map((anomaly) => {
          const probability =
            anomaly.probability ?? anomaly.efi ?? 0;

          const severity =
            anomaly.severity ?? anomaly.sev ?? "low";

          return (
            <div
              key={anomaly.id}
              className={
                "threat-card" +
                (anomaly.id === selected?.id ? " selected" : "")
              }
              onClick={() => onSelect(anomaly)}
            >
              <div className="threat-top">
                <span className="threat-name">
                  {anomaly.name}
                </span>

                <span
                  className={
                    "threat-sev " + severityClass(severity)
                  }
                >
                  {severity.toUpperCase()}
                </span>
              </div>

              <div className="threat-meta">
                <span>{anomaly.type}</span>

                <span>
                  {anomaly.durationDays
                    ? `${anomaly.durationDays} day track`
                    : anomaly.lead
                    ? `${anomaly.lead}h lead`
                    : "Tracked"}
                </span>
              </div>

              <div className="threat-efi">
                <div className="threat-efi-bar">
                  <div
                    className="threat-efi-fill"
                    style={{
                      width: `${Math.min(
                        100,
                        Math.max(0, probability * 100)
                      )}%`,
                      background: severityColor(severity),
                    }}
                  />
                </div>

                <span className="mono efi-number">
                  P {probability.toFixed(2)}
                </span>
              </div>
            </div>
          );
        })}
      </div>

      <div className="panel-divider"></div>

      <div className="panel-head">
        <h2>Model Probability</h2>
      </div>

      <div className="efi-legend">
        <div className="efi-bar"></div>

        <div className="efi-labels">
          <span>0.0</span>
          <span>0.5</span>
          <span>0.8</span>
          <span>0.95</span>
          <span>1.0</span>
        </div>

        <p className="efi-caption">
          Node-level anomaly probability produced by the
          spatio-temporal GNN.
        </p>
      </div>
    </aside>
  );
}

export default ThreatPanel;
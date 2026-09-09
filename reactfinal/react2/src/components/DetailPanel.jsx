import { useEffect, useRef } from "react";

function DetailPanel({ selected, onClose }) {
  const canvasRef = useRef(null);

  // ============================================
  // AMPLITUDE FIDELITY GRAPH
  // ============================================

  useEffect(() => {
    const canvas = canvasRef.current;

    if (!canvas) return;

    const parent = canvas.parentElement;
    const rect = parent.getBoundingClientRect();

    const dpr = window.devicePixelRatio || 1;

    canvas.width = rect.width * dpr;
    canvas.height = 90 * dpr;

    canvas.style.width = `${rect.width}px`;
    canvas.style.height = "90px";

    const ctx = canvas.getContext("2d");

    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    const W = rect.width;
    const H = 90;

    ctx.clearRect(0, 0, W, H);

    const n = 60;
    const peakScale = selected.efi;

    // ========================================
    // SMOOTH BASELINE
    // ========================================

    ctx.beginPath();

    for (let i = 0; i < n; i++) {
      const x = i / (n - 1);

      const bump =
        Math.exp(-Math.pow((x - 0.5) * 3.2, 2)) *
        peakScale *
        0.55;

      const y =
        H -
        (bump + 0.08) *
          H *
          0.85;

      if (i === 0) {
        ctx.moveTo(x * W, y);
      } else {
        ctx.lineTo(x * W, y);
      }
    }

    ctx.strokeStyle = "#4A5568";
    ctx.lineWidth = 1.6;
    ctx.stroke();

    // ========================================
    // DIFFUSION CURVE
    // ========================================

    ctx.beginPath();

    for (let i = 0; i < n; i++) {
      const x = i / (n - 1);

      const bump =
        Math.exp(-Math.pow((x - 0.5) * 3.6, 2)) *
        peakScale *
        0.95;

      const noise =
        Math.sin(
          x *
            selected.id.length *
            13 *
            6
        ) * 0.025;

      const y =
        H -
        (bump + noise + 0.05) *
          H *
          0.9;

      if (i === 0) {
        ctx.moveTo(x * W, y);
      } else {
        ctx.lineTo(x * W, y);
      }
    }

    ctx.strokeStyle = "#4FD1E8";
    ctx.lineWidth = 2;
    ctx.shadowColor = "rgba(79,209,232,0.5)";
    ctx.shadowBlur = 6;
    ctx.stroke();

    ctx.shadowBlur = 0;
  }, [selected]);

  // ============================================
  // DISPATCH ALERT
  // ============================================

  function dispatchAlert() {
    alert(
      `Pinpoint alert dispatched for ${selected.name}`
    );
  }

  return (
    <aside className="panel detail-panel">

      {/* ======================================
          PIPELINE TRACE
      ======================================= */}

      <div className="panel-head">
        <h2>Pipeline Trace</h2>

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

      <div className="pipeline-trace">

        {/* STEP 1 */}

        <div className="trace-step done">

          <div className="trace-icon">
            ①
          </div>

          <div className="trace-body">

            <span className="trace-title">
              Icosahedral Mesh Ingest
            </span>

            <span className="trace-meta mono">
              NEPS-G · 44 members · 12km
            </span>

          </div>

        </div>

        <div className="trace-connector"></div>

        {/* STEP 2 */}

        <div className="trace-step done">

          <div className="trace-icon">
            ②
          </div>

          <div className="trace-body">

            <span className="trace-title">
              GNN Trajectory Isolation
            </span>

            <span className="trace-meta mono">
              EFI σ = 3.8 vs ERA5
            </span>

          </div>

        </div>

        <div className="trace-connector active"></div>

        {/* STEP 3 */}

        <div className="trace-step active">

          <div className="trace-icon">
            ③
          </div>

          <div className="trace-body">

            <span className="trace-title">
              Diffusion Downscale
            </span>

            <span className="trace-meta mono">
              denoising step{" "}
              <span id="diffStep">
                640
              </span>
              /1000
            </span>

          </div>

        </div>

        <div className="trace-connector"></div>

        {/* STEP 4 */}

        <div className="trace-step">

          <div className="trace-icon">
            ④
          </div>

          <div className="trace-body">

            <span className="trace-title">
              Alert API Dispatch
            </span>

            <span className="trace-meta mono">
              5km radius, pending
            </span>

          </div>

        </div>

      </div>


      {/* ======================================
          ANOMALY READOUT
      ======================================= */}

      <div className="panel-divider"></div>

      <div className="panel-head">
        <h2>Anomaly Readout</h2>
      </div>

      <div className="readout-grid">

        {/* EFI */}

        <div className="readout-cell">

          <span className="readout-label">
            EFI Score
          </span>

          <span className="readout-value amber">
            {selected.efi.toFixed(2)}
          </span>

        </div>


        {/* WIND */}

        <div className="readout-cell">

          <span className="readout-label">
            Peak Wind
          </span>

          <span className="readout-value">
            {selected.wind} km/h
          </span>

        </div>


        {/* PRESSURE */}

        <div className="readout-cell">

          <span className="readout-label">
            Central Pressure
          </span>

          <span className="readout-value">
            {selected.pressure} hPa
          </span>

        </div>


        {/* CONFIDENCE */}

        <div className="readout-cell">

          <span className="readout-label">
            Confidence (ens.)
          </span>

          <span className="readout-value">
            {selected.confidence}%
          </span>

        </div>


        {/* LEAD TIME */}

        <div className="readout-cell">

          <span className="readout-label">
            Lead Time
          </span>

          <span className="readout-value">
            {selected.lead} h
          </span>

        </div>


        {/* IMPACT RADIUS */}

        <div className="readout-cell">

          <span className="readout-label">
            Impact Radius
          </span>

          <span className="readout-value">
            {selected.radius} km
          </span>

        </div>

      </div>


      {/* ======================================
          AMPLITUDE FIDELITY
      ======================================= */}

      <div className="panel-divider"></div>

      <div className="panel-head">
        <h2>Amplitude Fidelity</h2>
      </div>

      <div className="amp-compare">

        <canvas
          ref={canvasRef}
          height="90"
        ></canvas>

        <p className="amp-caption">
          Diffusion output (cyan) retains peak
          amplitude vs mean-optimized U-Net
          baseline (grey), which spectrally
          smooths the extreme.
        </p>

      </div>


      {/* ======================================
          DISPATCH
      ======================================= */}

      <div className="panel-divider"></div>

      <button
        className="alert-dispatch-btn"
        onClick={dispatchAlert}
      >

        <span>
          ⛛ Dispatch Pinpoint Alert
        </span>

        <span className="mono dispatch-sub">
          REST /v1/alerts · 5km radius
        </span>

      </button>

    </aside>
  );
}

export default DetailPanel;
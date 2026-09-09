function Timeline({
  forecastHour,
  setForecastHour,
}) {
  // ============================================
  // FORECAST DATE / TIME
  // ============================================

  function getForecastDate() {
    const baseDate = new Date(
      Date.UTC(
        2026,
        8,
        9,
        6,
        0,
        0
      )
    );

    const projected = new Date(
      baseDate.getTime() +
        forecastHour *
          3600 *
          1000
    );

    const day = String(
      projected.getUTCDate()
    ).padStart(2, "0");

    const month =
      projected.toLocaleString(
        "en-US",
        {
          month: "short",
          timeZone: "UTC",
        }
      );

    const hours = String(
      projected.getUTCHours()
    ).padStart(2, "0");

    const minutes = String(
      projected.getUTCMinutes()
    ).padStart(2, "0");

    return `${day} ${month}, ${hours}:${minutes} UTC`;
  }

  // ============================================
  // SLIDER
  // ============================================

  function handleChange(event) {
    setForecastHour(
      Number(event.target.value)
    );
  }

  return (
    <div className="timeline">

      <div className="timeline-head">
        <span>T+0h</span>

        <span
          className="mono"
          aria-live="polite"
        >
          FORECAST HOUR:{" "}
          <b>+{forecastHour}h</b>
          {" · "}
          {getForecastDate()}
        </span>

        <span>T+240h</span>
      </div>

      <input
        type="range"
        id="timelineSlider"
        min="0"
        max="240"
        step="6"
        value={forecastHour}
        onChange={handleChange}
        aria-label="Forecast hour"
        aria-valuetext={`Forecast hour plus ${forecastHour} hours`}
      />

      <div className="timeline-ticks">
        <span>Day 0</span>
        <span>Day 2</span>
        <span>Day 4</span>
        <span>Day 6</span>
        <span>Day 8</span>
        <span>Day 10</span>
      </div>

    </div>
  );
}

export default Timeline;

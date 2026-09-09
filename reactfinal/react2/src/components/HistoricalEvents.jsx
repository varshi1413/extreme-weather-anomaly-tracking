import { useEffect, useState } from "react";
import "./HistoricalEvents.css";

function HistoricalEvents({ onSelect }) {
  const [years, setYears] = useState([]);
  const [year, setYear] = useState("");
  const [events, setEvents] = useState([]);
  const [open, setOpen] = useState(false);

  useEffect(() => {
    fetch("/api/historical")
      .then((response) => response.json())
      .then((payload) => {
        setYears(payload.years || []);
        if (payload.years?.length) setYear(String(payload.years[0]));
      })
      .catch(() => {});
  }, []);

  useEffect(() => {
    if (!year) return;
    fetch(`/api/historical?year=${year}`)
      .then((response) => response.json())
      .then((payload) => setEvents(payload.events || []))
      .catch(() => setEvents([]));
  }, [year]);

  async function selectEvent(event) {
    const response = await fetch(`/api/historical?year=${year}`);
    const payload = await response.json();
    const fullEvent = payload.events?.find((item) => item.id === event.id);
    if (fullEvent) onSelect(fullEvent);
  }

  return (
    <section className={`historical-menu ${open ? "is-open" : ""}`}>
      <button className="historical-toggle" type="button" onClick={() => setOpen((value) => !value)}>
        <span className="historical-icon">↗</span>
        <span>Historical events</span>
        <span className="historical-chevron">{open ? "−" : "+"}</span>
      </button>
      {open && (
        <div className="historical-browser">
          <label htmlFor="historical-year">SEASON</label>
          <select id="historical-year" value={year} onChange={(event) => setYear(event.target.value)}>
            {years.map((item) => <option key={item} value={item}>{item}</option>)}
          </select>
          <div className="historical-events-list">
            {events.map((event) => (
              <button key={event.id} type="button" onClick={() => selectEvent(event)}>
                <span>{event.name}</span>
                <small>{event.maxWind ? `${Math.round(event.maxWind)} kt` : "TRACK"}</small>
              </button>
            ))}
            {!events.length && <p>No historical tracks for this season.</p>}
          </div>
        </div>
      )}
    </section>
  );
}

export default HistoricalEvents;
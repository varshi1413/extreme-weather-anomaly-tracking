import { useEffect, useState } from "react";

function TopBar({ weatherStatus, onLocationSelect, onWeatherData, onWeatherStatus }) {
  const [query, setQuery] = useState("");
  const [searching, setSearching] = useState(false);
  const [time, setTime] = useState(
    new Date().toISOString().slice(11, 19) + " UTC"
  );

  useEffect(() => {
    const updateClock = () => {
      const now = new Date();

      setTime(
        now.toISOString().slice(11, 19) + " UTC"
      );
    };

    updateClock();

    const timer = setInterval(updateClock, 1000);

    return () => clearInterval(timer);
  }, []);

  async function searchLocation(event) {
    event.preventDefault();
    if (!query.trim() || searching) return;
    setSearching(true);
    try {
      let response;
      let result;
      try {
        response = await fetch(`/api/geocode?q=${encodeURIComponent(query.trim())}`);
        result = await response.json();
      } catch {
        response = await fetch(`https://nominatim.openstreetmap.org/search?format=jsonv2&limit=1&q=${encodeURIComponent(query.trim())}`);
        const places = await response.json();
        if (!places.length) throw new Error("Location not found");
        result = { location: { name: places[0].display_name, lat: Number(places[0].lat), lon: Number(places[0].lon) } };
      }
      if (!response.ok || !result.location) throw new Error(result.error || "Location not found");
      onLocationSelect(result.location);
      onWeatherStatus("LOADING WEATHER");
      try {
        const weatherResponse = await fetch(`/api/weather?lat=${result.location.lat}&lon=${result.location.lon}`);
        const weather = await weatherResponse.json();
        if (weatherResponse.ok) {
          onWeatherData(weather);
          onWeatherStatus(`${weather.source.toUpperCase()} LIVE`);
        } else {
          onWeatherStatus("MAP LOCATION READY");
        }
      } catch {
        onWeatherStatus("MAP LOCATION READY");
      }
    } catch {
      onWeatherStatus("LOCATION NOT FOUND");
    } finally {
      setSearching(false);
    }
  }

  return (
    <header className="topbar">
      <form className="weather-search" onSubmit={searchLocation}>
        <span aria-hidden="true">⌕</span>
        <input aria-label="Search location" placeholder={searching ? "Searching..." : "Search location..."} value={query} onChange={(event) => setQuery(event.target.value)} />
        <button type="submit" title="Search location" aria-label="Search location">↵</button>
      </form>

      <div className="brand">

        <div className="brand-mark">
          <svg
            viewBox="0 0 32 32"
            width="26"
            height="26"
          >
            <circle
              cx="16"
              cy="16"
              r="14"
              fill="none"
              stroke="var(--cyan)"
              strokeWidth="1.4"
              opacity="0.5"
            />

            <circle
              cx="16"
              cy="16"
              r="9"
              fill="none"
              stroke="var(--cyan)"
              strokeWidth="1.4"
            />

            <circle
              cx="16"
              cy="16"
              r="2.4"
              fill="var(--amber)"
            />

            <path
              d="M16 2 L16 7 M16 25 L16 30 M2 16 L7 16 M25 16 L30 16"
              stroke="var(--cyan)"
              strokeWidth="1.2"
              opacity="0.6"
            />
          </svg>
        </div>

        <div className="brand-text">

          <span className="brand-name">
            VAYUNETRA <small>AI</small>
          </span>

          <span className="brand-sub">
            Spatio-Temporal Anomaly Tracking —
            NCMRWF / MoES
          </span>

        </div>

      </div>


      <div className="topbar-status">
        <button className="premium-chip" type="button">{weatherStatus}</button>
        <button className="menu-button" type="button" aria-label="Open menu">☰</button>
        <div className="status-chip clock">{time}</div>

      </div>

    </header>
  );
}

export default TopBar;
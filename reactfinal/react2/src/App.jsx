import "./App.css";
import "./components/Layout.css";
import { useEffect, useState } from "react";
import TopBar from "./components/TopBar";
import { ANOMALIES } from "./data/anomalies";
import ThreatPanel from "./components/ThreatPanel";
import WeatherMap from "./components/WeatherMap";
import LayerMenu from "./components/LayerMenu";
import Footer from "./components/Footer";
import HistoricalEvents from "./components/HistoricalEvents";

function App() {
  const [anomalies, setAnomalies] = useState(ANOMALIES);
  const [selected, setSelected] = useState(ANOMALIES[0]);
  const [resolution, setResolution] = useState(12);
  const [forecastHour, setForecastHour] = useState(72);
  const [activeLayer, setActiveLayer] = useState("heatmap");
  const [showThreats, setShowThreats] = useState(true);
  const [mapLocation, setMapLocation] = useState(null);
  const [weatherStatus, setWeatherStatus] = useState("MODEL ACTIVE");
  const [weatherData, setWeatherData] = useState(null);
  const [historicalEvent, setHistoricalEvent] = useState(null);

  useEffect(() => {
    let active = true;
    fetch("/api/anomalies")
      .then((response) => {
        if (!response.ok) throw new Error(`API responded ${response.status}`);
        return response.json();
      })
      .then((payload) => {
        if (!active || !Array.isArray(payload.anomalies) || payload.anomalies.length === 0) return;
        setAnomalies(payload.anomalies);
        setSelected(payload.anomalies[0]);
      })
      .catch(() => {});
    return () => { active = false; };
  }, []);

  return (
    <div className="app app-fullscreen">
      <div className="scanline-overlay"></div>

      {/* MAP — fills the entire viewport, everything else floats on top */}
      <div className="map-fullscreen-wrap">
        <WeatherMap
          selected={selected}
          resolution={resolution}
          setResolution={setResolution}
          forecastHour={forecastHour}
          setForecastHour={setForecastHour}
          activeLayer={activeLayer}
          mapLocation={mapLocation}
          weatherData={weatherData}
          historicalEvent={historicalEvent}
        />
      </div>

      {/* Floating slim topbar */}
      <div className="floating-topbar">
        <TopBar
          weatherStatus={weatherStatus}
          onLocationSelect={(location) => setMapLocation(location)}
          onWeatherData={setWeatherData}
          onWeatherStatus={setWeatherStatus}
        />
      </div>

      {/* Windy-style right rail: layer picker + resolution */}
      <LayerMenu
        activeLayer={activeLayer}
        setActiveLayer={setActiveLayer}
        resolution={resolution}
        setResolution={setResolution}
      />

      <div className="historical-rail">
        <HistoricalEvents onSelect={setHistoricalEvent} />
      </div>

      {/* Reopen chips for closed popups */}
      {!showThreats && (
        <button
          className="reopen-btn reopen-threats"
          onClick={() => setShowThreats(true)}
        >
          ☰ Anomalies
        </button>
      )}

      {showThreats && (
        <div className="floating-panel panel-threats">
          <ThreatPanel
            anomalies={anomalies}
            selected={selected}
            onSelect={setSelected}
            onClose={() => setShowThreats(false)}
          />
        </div>
      )}

      {/* Floating status strip */}
      <div className="floating-footer">
        <Footer />
      </div>
    </div>
  );
}

export default App;
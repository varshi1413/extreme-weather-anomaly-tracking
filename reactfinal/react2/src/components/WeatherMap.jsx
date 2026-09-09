import { useEffect, useRef } from "react";
import L from "leaflet";
import "leaflet/dist/leaflet.css";
import "leaflet.heat";
import Timeline from "./Timeline";
import "./WeatherMap.css";

const LAYER_CONFIG = {
  heatmap: {
    gradient: {
      0.0: "#092c55",
      0.22: "#075f91",
      0.42: "#00a9b7",
      0.58: "#48d7c5",
      0.70: "#f6d743",
      0.82: "#ff9138",
      0.92: "#ff4b45",
      1.0: "#ff174f",
    },
    legendTitle: "GNN ANOMALY PROBABILITY",
    legendCss:
      "linear-gradient(90deg,#092c55,#075f91,#00a9b7,#48d7c5,#f6d743,#ff9138,#ff4b45,#ff174f)",
    showHeat: true,
    showWind: false,
    showPressure: false,
  },
  coldmap: {
    gradient: {
      0.0: "#ffffff",
      0.2: "#dbeeff",
      0.4: "#a9d4ff",
      0.6: "#6fa8ff",
      0.8: "#3d6fe0",
      1.0: "#1b2f8c",
    },
    legendTitle: "COLD ANOMALY INDEX",
    legendCss:
      "linear-gradient(90deg,#ffffff,#dbeeff,#a9d4ff,#6fa8ff,#3d6fe0,#1b2f8c)",
    showHeat: true,
    showWind: false,
    showPressure: false,
  },
  wind: {
    gradient: {
      0.0: "#0d2b3d",
      0.4: "#1c6b8a",
      0.7: "#42e6f5",
      1.0: "#ffffff",
    },
    legendTitle: "WIND FIELD INTENSITY",
    legendCss: "linear-gradient(90deg,#0d2b3d,#1c6b8a,#42e6f5,#ffffff)",
    showHeat: false,
    showWind: true,
    showPressure: false,
  },
  cyclone: {
    gradient: {
      0.0: "#102d50",
      0.35: "#1aa6c4",
      0.68: "#ffc53d",
      1.0: "#f04444",
    },
    legendTitle: "CYCLONE TRACKER",
    legendCss: "linear-gradient(90deg,#102d50,#1aa6c4,#ffc53d,#f04444)",
    showHeat: false,
    showWind: true,
    showPressure: false,
  },
  pressure: {
    gradient: { 0.0: "#0d2b3d", 0.5: "#4fd1e8", 1.0: "#ffffff" },
    legendTitle: "PRESSURE ISOBARS",
    legendCss: "linear-gradient(90deg,#0d2b3d,#4fd1e8,#ffffff)",
    showHeat: false,
    showWind: false,
    showPressure: true,
  },
  satellite: {
    gradient: null,
    legendTitle: "SATELLITE VIEW",
    legendCss: "linear-gradient(90deg,#111,#333,#777,#ccc)",
    showHeat: false,
    showWind: false,
    showPressure: false,
  },
};

function WeatherMap({
  selected,
  resolution,
  setResolution,
  forecastHour,
  setForecastHour,
  activeLayer = "heatmap",
  mapLocation,
  weatherData,
  historicalEvent,
}) {
  const mapRef = useRef(null);
  const mapContainerRef = useRef(null);
  const tileLayerRef = useRef(null);
  const heatLayerRef = useRef(null);
  const trackLineRef = useRef(null);
  const waypointLayerRef = useRef(null);
  const bboxRef = useRef(null);
  const historicalLayerRef = useRef(null);
  const windLayerRef = useRef(null);
  const pressureLayerRef = useRef(null);

  const config = LAYER_CONFIG[activeLayer] || LAYER_CONFIG.heatmap;

  const severityColor = (severity) => {
    if (severity === "severe") return "#ff3b5c";
    if (severity === "moderate") return "#ffc94a";
    return "#37d67a";
  };

  // --------------------------------------------------
  // NOISE
  // --------------------------------------------------

  function hashNoise(x, y, seed) {
    const value =
      Math.sin(x * 127.1 + y * 311.7 + seed * 43.7) * 43758.5453;

    return value - Math.floor(value);
  }

  function smoothNoise(x, y, seed, scale) {
    const xi = Math.floor(x * scale);
    const yi = Math.floor(y * scale);

    const xf = x * scale - xi;
    const yf = y * scale - yi;

    const a = hashNoise(xi, yi, seed);
    const b = hashNoise(xi + 1, yi, seed);
    const c = hashNoise(xi, yi + 1, seed);
    const d = hashNoise(xi + 1, yi + 1, seed);

    const u = xf * xf * (3 - 2 * xf);
    const v = yf * yf * (3 - 2 * yf);

    return (
      a * (1 - u) * (1 - v) +
      b * u * (1 - v) +
      c * (1 - u) * v +
      d * u * v
    );
  }

  function fbm(x, y, seed, octaves) {
    let value = 0;
    let amplitude = 0.5;
    let frequency = 1;

    for (let i = 0; i < octaves; i++) {
      value += amplitude * smoothNoise(x, y, seed + i * 7, frequency);

      frequency *= 2.15;
      amplitude *= 0.55;
    }

    return value;
  }
  function getTrackPoints(anomaly) {
    if (!anomaly?.track?.length) return [];

    return anomaly.track
      .map((point) => {
        // New GNN tracker output:
        // { day, lat, lon, probability }
        if (point && typeof point === "object" && !Array.isArray(point)) {
          const lat = Number(point.lat);
          const lon = Number(point.lon);

          if (Number.isFinite(lat) && Number.isFinite(lon)) {
            return [lat, lon];
          }

          return null;
        }

        // Old frontend/demo format:
        // [lat, lon]
        if (Array.isArray(point) && point.length >= 2) {
          const lat = Number(point[0]);
          const lon = Number(point[1]);

          if (Number.isFinite(lat) && Number.isFinite(lon)) {
            return [lat, lon];
          }
        }

        return null;
      })
      .filter(Boolean);
  }
  // --------------------------------------------------
  // CYCLONE POSITION
  // --------------------------------------------------

  function interpolateCentroid(anomaly, hour) {
    const track = getTrackPoints(anomaly);

    if (!track.length) {
      return [Number(anomaly.lat), Number(anomaly.lon)];
    }

    if (track.length === 1) {
      return track[0];
    }

    const progress = Math.min(1, Math.max(0, hour / 168));

    const trackLength = track.length - 1;
    const position = progress * trackLength;

    const index0 = Math.floor(position);
    const index1 = Math.min(trackLength, index0 + 1);
    const fraction = position - index0;

    const lat =
      track[index0][0] +
      (track[index1][0] - track[index0][0]) * fraction;

    const lon =
      track[index0][1] +
      (track[index1][1] - track[index0][1]) * fraction;

    return [lat, lon];
  }
  // --------------------------------------------------
  // HEAT FIELD
  // --------------------------------------------------

  function buildHeatPoints(anomaly, centerLat, centerLon) {
    const points = [];

    const seed = anomaly.id
      .split("")
      .reduce((sum, character) => sum + character.charCodeAt(0), 0);

    const span = resolution === 12 ? 4.0 : 1.5;

    const step = resolution === 12 ? 0.11 : 0.045;

    for (let dy = -span; dy <= span; dy += step) {
      for (let dx = -span; dx <= span; dx += step) {
        const lat = centerLat + dy;
        const lon = centerLon + dx;

        const distance = Math.sqrt(dx * dx * 1.45 + dy * dy);

        const core =
          Math.exp(-Math.pow(distance / 1.3, 2)) * anomaly.efi;

        const turbulence =
          fbm(lon * 2.1, lat * 2.1, seed, resolution === 12 ? 3 : 5) *
          0.45;

        let value = Math.pow(core, 0.78) + turbulence * 0.23;

        value = Math.min(1, value);

        if (value > 0.035) {
          points.push([lat, lon, value]);
        }
      }
    }

    return points;
  }

  // --------------------------------------------------
  // MAP INITIALIZATION
  // --------------------------------------------------

  useEffect(() => {
    if (mapRef.current) return;
    if (!mapContainerRef.current) return;

    const map = L.map(mapContainerRef.current, {
      zoomControl: true,
      attributionControl: true,
      worldCopyJump: true,
    }).setView([selected.lat, selected.lon], 6);

    tileLayerRef.current = L.tileLayer(
      "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
      {
        attribution: "&copy; OpenStreetMap contributors",
        maxZoom: 19,
      }
    ).addTo(map);

    waypointLayerRef.current = L.layerGroup().addTo(map);

    map.on("mousemove", (event) => {
      const readout = document.getElementById("coordReadout");

      if (readout) {
        readout.innerHTML = `
          <span class="mono">
            LAT ${event.latlng.lat.toFixed(3)}°
            &nbsp;&nbsp;
            LON ${event.latlng.lng.toFixed(3)}°
          </span>
        `;
      }
    });

    mapRef.current = map;

    setTimeout(() => {
      map.invalidateSize();
    }, 200);

    return () => {
      map.remove();
      mapRef.current = null;
    };
  }, []);

  // --------------------------------------------------
  // REFRESH VISUALIZATION
  // --------------------------------------------------

  useEffect(() => {
    const map = mapRef.current;

    if (!map || !selected) return;

    // ------------------------------------------------
    // BASEMAP SWAP (satellite vs street)
    // ------------------------------------------------

    if (tileLayerRef.current) {
      const url =
        activeLayer === "satellite"
          ? "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
          : "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png";

      tileLayerRef.current.setUrl(url);
    }

    const [centerLat, centerLon] = interpolateCentroid(
      selected,
      forecastHour
    );

    // ------------------------------------------------
    // HEAT FIELD
    // ------------------------------------------------

    const points = buildHeatPoints(selected, centerLat, centerLon);

    if (heatLayerRef.current) {
      map.removeLayer(heatLayerRef.current);
      heatLayerRef.current = null;
    }

    if (config.showHeat) {
      heatLayerRef.current = L.heatLayer(points, {
        radius: resolution === 12 ? 30 : 20,

        blur: resolution === 12 ? 34 : 22,

        maxZoom: 12,

        max: 1,

        minOpacity: 0.22,

        gradient: config.gradient,
      }).addTo(map);
    }

    // ------------------------------------------------
    // FORECAST TRACK
    // ------------------------------------------------

    if (trackLineRef.current) {
      map.removeLayer(trackLineRef.current);
    }

    const track = getTrackPoints(selected);

    if (track.length > 1) {
      const trackLength = track.length - 1;

      const progress = Math.min(1, forecastHour / 168);

      const trackPosition = progress * trackLength;

      const trackIndex = Math.floor(trackPosition);

      const trackFraction = trackPosition - trackIndex;

      const nextIndex = Math.min(trackLength, trackIndex + 1);

      const currentLat =
        track[trackIndex][0] +
        (track[nextIndex][0] - track[trackIndex][0]) * trackFraction;

      const currentLon =
        track[trackIndex][1] +
        (track[nextIndex][1] - track[trackIndex][1]) * trackFraction;

      const visibleTrack = track.slice(
        0,
        Math.min(trackIndex + 1, track.length)
      );

      visibleTrack.push([currentLat, currentLon]);

      trackLineRef.current = L.polyline(visibleTrack, {
        color: "#55e5ff",
        weight: 3,
        opacity: 0.95,
        lineCap: "round",
        lineJoin: "round",
      }).addTo(map);

      // Forecast dashed line
      const futureTrack = track.slice(
        Math.min(trackIndex, track.length - 1)
      );

      futureTrack.unshift([currentLat, currentLon]);

      L.polyline(futureTrack, {
        color: "#ffc94a",
        weight: 2,
        opacity: 0.75,
        dashArray: "5,8",
        lineCap: "round",
      }).addTo(map);
    }

    // ------------------------------------------------
    // WAYPOINTS
    // ------------------------------------------------

    waypointLayerRef.current.clearLayers();

    if (track.length) {
      const trackPosition = (forecastHour / 168) * (track.length - 1);

      track.forEach((point, index) => {
        const passed = index <= trackPosition;

        L.circleMarker(point, {
          radius: passed ? 4 : 2.5,

          color: passed ? "#55e5ff" : "#55e5ff55",

          fillColor: passed ? "#55e5ff" : "#55e5ff44",

          fillOpacity: passed ? 0.9 : 0.25,

          weight: passed ? 1.2 : 1,

          opacity: passed ? 1 : 0.35,
        }).addTo(waypointLayerRef.current);
      });
    }

    // ------------------------------------------------
    // CYCLONE CIRCULATION
    // ------------------------------------------------

    if (windLayerRef.current) {
      map.removeLayer(windLayerRef.current);
    }

    windLayerRef.current = L.layerGroup().addTo(map);

    if (config.showWind) {
      const circulationColor = severityColor(selected.sev);

      const radii = [0.35, 0.65, 0.95, 1.3, 1.7, 2.15];

      radii.forEach((radius, index) => {
        L.circle([centerLat, centerLon], {
          radius: radius * 110000,

          color: index % 2 === 0 ? "#49dff5" : circulationColor,

          weight: index < 2 ? 1.4 : 0.8,

          opacity: index < 2 ? 0.48 : 0.25,

          fill: false,

          dashArray: index % 2 === 0 ? "3,8" : "8,12",

          interactive: false,
        }).addTo(windLayerRef.current);
      });

      // ----------------------------------------------
      // WIND ARCS
      // ----------------------------------------------

      for (let i = 0; i < 12; i++) {
        const angle = (i / 12) * Math.PI * 2;

        const radius = 0.75 + (i % 3) * 0.32;

        const startAngle = angle;

        const endAngle = angle + Math.PI * 0.72;

        const arcPoints = [];

        for (let j = 0; j <= 18; j++) {
          const t = j / 18;

          const a = startAngle + (endAngle - startAngle) * t;

          const r = radius + Math.sin(t * Math.PI) * 0.08;

          const lat = centerLat + Math.sin(a) * r;

          const lon = centerLon + Math.cos(a) * r;

          arcPoints.push([lat, lon]);
        }

        L.polyline(arcPoints, {
          color: i % 2 === 0 ? "#42e6f5" : "#ff713b",

          weight: 1.6,

          opacity: 0.48,

          dashArray: "2,7",

          lineCap: "round",

          interactive: false,
        }).addTo(windLayerRef.current);
      }
    }

    // ------------------------------------------------
    // PRESSURE / ISOBARS
    // ------------------------------------------------

    if (pressureLayerRef.current) {
      map.removeLayer(pressureLayerRef.current);
    }

    pressureLayerRef.current = L.layerGroup().addTo(map);

    if (config.showPressure) {
      const livePressure = weatherData?.parameters?.pressure;
      const pressureValues = Array.isArray(livePressure)
        ? livePressure.slice(0, 6).map((value) => Math.round(Number(value))).filter(Number.isFinite)
        : [1008, 1004, 1000, 996, 992, 988];

      pressureValues.forEach((pressure, index) => {
        const radius = 0.45 + index * 0.42;

        L.circle([centerLat, centerLon], {
          radius: radius * 110000,

          color: "#b8d9e4",

          weight: 0.7,

          opacity: 0.22,

          fill: false,

          dashArray: "2,9",

          interactive: false,
        })
          .bindTooltip(`${pressure} hPa`, {
            permanent: false,
            direction: "center",
            className: "pressure-label",
          })
          .addTo(pressureLayerRef.current);
      });
    }

    // ------------------------------------------------
    // GNN BOUNDING BOX
    // ------------------------------------------------

    const span =
      resolution === 12
        ? { lat: 1.8, lon: 2.1 }
        : { lat: 0.7, lon: 0.85 };

    const bounds = [
      [centerLat - span.lat / 2, centerLon - span.lon / 2],
      [centerLat + span.lat / 2, centerLon + span.lon / 2],
    ];

    if (bboxRef.current) {
      map.removeLayer(bboxRef.current);
    }

    bboxRef.current = L.rectangle(bounds, {
      color: resolution === 12 ? "#ff713b" : "#55e5ff",

      weight: resolution === 12 ? 2 : 1.2,

      fill: false,

      dashArray: resolution === 12 ? "7,5" : "3,6",

      opacity: 0.8,
    }).addTo(map);

    bboxRef.current.bindTooltip(
      resolution === 12
        ? "AI / GNN DETECTION BOUNDARY"
        : "5 KM DOWN-SCALED IMPACT FIELD",
      {
        permanent: true,
        direction: "top",
        className: "bbox-label",
        offset: [0, -5],
      }
    );

    if (historicalLayerRef.current) {
      map.removeLayer(historicalLayerRef.current);
      historicalLayerRef.current = null;
    }

    if (historicalEvent?.points?.length > 1) {
      historicalLayerRef.current = L.layerGroup().addTo(map);
      const points = historicalEvent.points;
      for (let index = 1; index < points.length; index += 1) {
        const first = points[index - 1];
        const second = points[index];
        const wind = Number(second.wind) || 0;
        const color = wind >= 100 ? "#e63232" : wind >= 64 ? "#ff8b2e" : wind >= 34 ? "#ffd34d" : "#53c7e8";
        L.polyline([[first.lat, first.lon], [second.lat, second.lon]], {
          color,
          weight: 4,
          opacity: 0.95,
          lineCap: "round",
          interactive: false,
        }).addTo(historicalLayerRef.current);
      }
      map.fitBounds(points.map((point) => [point.lat, point.lon]), { padding: [80, 80], maxZoom: 7 });
    }

    // ------------------------------------------------
    // BADGE
    // ------------------------------------------------

    const badge = document.getElementById("centroidBadge");

    if (badge) {
      badge.textContent = `${centerLat.toFixed(2)}°N, ${centerLon.toFixed(
        2
      )}°E`;
    }

    const forecastBadge = document.getElementById("forecastBadge");

    if (forecastBadge) {
      forecastBadge.textContent = `T+${forecastHour}H`;
    }
  }, [selected, resolution, forecastHour, activeLayer, weatherData, historicalEvent]);

  // --------------------------------------------------
  // FLY TO FORECAST POSITION
  // --------------------------------------------------

  useEffect(() => {
    const map = mapRef.current;

    if (!map || !selected) return;

    const [centerLat, centerLon] = interpolateCentroid(
      selected,
      forecastHour
    );

    map.flyTo([centerLat, centerLon], resolution === 12 ? 6 : 9, {
      duration: 0.7,
    });
  }, [selected, resolution, forecastHour]);

  useEffect(() => {
    const map = mapRef.current;
    if (!map || !mapLocation) return;
    map.flyTo([mapLocation.lat, mapLocation.lon], 7, { duration: 0.8 });
  }, [mapLocation]);

  return (
    <section className="panel map-panel">
      {/* HEADER */}

      <div className="panel-head map-head">
        <div>
          <h2>
            {selected.type} —{" "}
            <span className="mono">
              {selected.name.split(" ").slice(-1)[0].toUpperCase()}
              -CLASS
            </span>
          </h2>

          <div className="map-subtitle">
            AI SPATIO-TEMPORAL ANOMALY FIELD
          </div>
        </div>

        <div className="map-controls">
          <div className="seg-toggle">
            <button
              className={
                "seg-btn " + (resolution === 12 ? "active" : "")
              }
              onClick={() => setResolution(12)}
            >
              12 KM · GNN
            </button>

            <button
              className={
                "seg-btn " + (resolution === 5 ? "active" : "")
              }
              onClick={() => setResolution(5)}
            >
              5 KM · DIFFUSION
            </button>
          </div>

          <button className="icon-btn">☰ WIND</button>
        </div>
      </div>

      {/* MAP */}

      <div className="map-stage">
        <div id="leafletMap" ref={mapContainerRef} />

        {/* TOP LEFT */}

        <div
          className="map-badge severity-badge"
          style={{
            color: severityColor(selected.sev),

            borderColor: severityColor(selected.sev) + "77",

            background: severityColor(selected.sev) + "1c",
          }}
        >
          <span
            className="badge-dot"
            style={{
              background: severityColor(selected.sev),
            }}
          />

          {selected.sev.toUpperCase()}

          <span className="badge-divider">/</span>

          P {(selected.probability ?? selected.efi ?? 0).toFixed(2)}
        </div>

        {/* TOP RIGHT */}

        <div className="map-badge coord-badge" id="centroidBadge">
          {selected.lat.toFixed(2)}°N, {selected.lon.toFixed(2)}°E
        </div>

        {/* FORECAST */}

        <div className="forecast-badge" id="forecastBadge">
          T+{forecastHour}H
        </div>

        {/* COORDINATES */}

        <div className="coord-readout" id="coordReadout">
          <span className="mono">
            LAT —.——° &nbsp;&nbsp; LON —.——°
          </span>
        </div>

        {/* MAP LEGEND */}

        <div className="map-legend">
          <div className="legend-title">{config.legendTitle}</div>

          <div
            className="legend-gradient"
            style={{ background: config.legendCss }}
          ></div>

          <div className="legend-values">
            <span>0</span>
            <span>0.25</span>
            <span>0.5</span>
            <span>0.75</span>
            <span>1.0</span>
          </div>
        </div>
      </div>

      {/* TIMELINE */}

      <Timeline
        forecastHour={forecastHour}
        setForecastHour={setForecastHour}
      />
    </section>
  );
}

export default WeatherMap;
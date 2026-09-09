from __future__ import annotations
from src.extreme_weather.heatwave_inference import HeatwavePredictor
from src.extreme_weather.stage2_inference import Stage2DiffusionInference

import json
import os
import sys
import csv
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, quote, urlparse
from urllib.request import Request, urlopen

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent / "src"))

from extreme_weather import AnomalyTracker, SphericalAnomalyGNN
from extreme_weather.mesh import build_indian_icosahedral_mesh


FRONTEND_ANOMALIES = (
    ("amphan", "Cyclone AMPHAN-class", "Tropical Cyclone", "severe", 20.31, 87.94, "Bay of Bengal"),
    ("heatdome", "North India Heat Dome", "Heat Anomaly", "severe", 27.2, 75.8, "Rajasthan / NW India"),
    ("coldwave", "Indo-Gangetic Cold Wave", "Cold Wave", "moderate", 29.9, 78.1, "Uttar Pradesh"),
    ("flashflood", "Western Ghats Flash Flood Cell", "Convective / Rainfall", "severe", 15.4, 74.0, "Karnataka Coast"),
    ("hailstorm", "Deccan Plateau Hail Cell", "Severe Convection", "moderate", 18.6, 76.2, "Maharashtra"),
    ("coastalsurge", "Odisha Coastal Surge", "Storm Surge", "low", 19.8, 85.8, "Odisha Coast"),
)

WINDY_PARAMETERS = ["wind", "dewpoint", "rh", "pressure", "temp", "precip", "lclouds", "mclouds", "hclouds"]
WINDY_LEVELS = ["surface", "850h", "700h", "500h", "300h"]
IBTRACS_PATH = Path(r"C:\Users\Admin\Downloads\ibtracs.NI.list.v04r01.csv")
_historical_cache: dict[str, list[dict]] | None = None

PROJECT_ROOT = Path(__file__).resolve().parent

HEATWAVE_CHECKPOINT = (
    PROJECT_ROOT / "checkpoints" / "heatwave_schema_test.pt"
)

DIFFUSION_CHECKPOINT = (
    PROJECT_ROOT / "checkpoints" / "stage2_era5_256.pt"
)



def historical_events() -> dict[str, list[dict]]:
    global _historical_cache
    if _historical_cache is not None:
        return _historical_cache
    if not IBTRACS_PATH.exists():
        raise FileNotFoundError(f"historical dataset not found: {IBTRACS_PATH}")
    with IBTRACS_PATH.open(encoding="utf-8-sig", newline="") as source:
        reader = csv.reader(source)
        headers = next(reader)
        next(reader, None)
        indexes = {name: index for index, name in enumerate(headers)}
        grouped: dict[str, dict[str, dict]] = {}
        for row in reader:
            if len(row) <= max(indexes.values()) or row[indexes["BASIN"]] != "NI":
                continue
            sid = row[indexes["SID"]].strip()
            year = row[indexes["SEASON"]].strip()
            try:
                lat = float(row[indexes["LAT"]])
                lon = float(row[indexes["LON"]])
            except (ValueError, TypeError):
                continue
            wind_text = row[indexes["USA_WIND"]].strip() or row[indexes["WMO_WIND"]].strip()
            try:
                wind = float(wind_text)
            except (ValueError, TypeError):
                wind = 0.0
            event = grouped.setdefault(year, {}).setdefault(sid, {
                "id": sid,
                "year": int(year),
                "name": row[indexes["NAME"]].strip() or "UNNAMED",
                "basin": "North Indian Ocean",
                "points": [],
                "maxWind": 0,
            })
            event["points"].append({"lat": lat, "lon": lon, "wind": wind, "time": row[indexes["ISO_TIME"]].strip()})
            event["maxWind"] = max(event["maxWind"], wind)
    _historical_cache = {
        year: sorted(events.values(), key=lambda event: (-event["maxWind"], event["name"]))
        for year, events in grouped.items()
    }
    return _historical_cache


def historical_response(year: str | None = None) -> dict:
    grouped = historical_events()
    years = sorted((int(value) for value in grouped), reverse=True)
    if year is None:
        return {"years": years, "events": [{key: value for key, value in event.items() if key != "points"} for event in grouped[str(years[0])]]}
    if year not in grouped:
        raise ValueError("historical year not found")
    return {"year": int(year), "events": grouped[year]}


def run_tracker(forecast: np.ndarray, baseline: np.ndarray) -> dict:
    mesh = build_indian_icosahedral_mesh(subdivisions=3)
    if forecast.ndim != 3 or baseline.ndim != 3 or forecast.shape[1:] != baseline.shape[1:]:
        raise ValueError("forecast and baseline must both have shape [time, mesh_nodes, channels]")
    if forecast.shape[1] != mesh.num_nodes:
        raise ValueError(f"expected {mesh.num_nodes} mesh nodes")
    torch.manual_seed(7)
    model = SphericalAnomalyGNN(input_channels=forecast.shape[2], hidden_channels=32, layers=2)
    tracker = AnomalyTracker(mesh, model, threshold=-np.inf)
    scores, mask, box = tracker.track(forecast.astype(np.float32), baseline.astype(np.float32))
    return {
        "scores": scores.tolist(),
        "mask": mask.tolist(),
        "bbox": None if box is None else {
            "start": box.start,
            "end": box.end,
            "lat_min": box.lat_min,
            "lat_max": box.lat_max,
            "lon_min": box.lon_min,
            "lon_max": box.lon_max,
        },
    }


def demo_response() -> dict:
    mesh = build_indian_icosahedral_mesh(subdivisions=3)
    rng = np.random.default_rng(7)
    baseline = rng.normal(0, 1, (12, mesh.num_nodes, 2)).astype(np.float32)
    forecast = baseline[-1:] + rng.normal(0, 0.12, (12, mesh.num_nodes, 2)).astype(np.float32)
    forecast[:, :, 0] += np.exp(-((mesh.lat - 20) ** 2 + (mesh.lon - 86) ** 2) / 35) * 4
    result = run_tracker(forecast, baseline)
    signal = np.max(np.abs(forecast[-1]), axis=1)
    peak = float(np.clip(signal.max() / 4, 0, 1))
    anomalies = []
    for index, (item_id, name, kind, severity, lat, lon, region) in enumerate(FRONTEND_ANOMALIES):
        anomalies.append({
            "id": item_id,
            "name": name,
            "type": kind,
            "sev": severity,
            "efi": round(float(np.clip(0.62 + peak * 0.35 - index * 0.025, 0.4, 0.99)), 2),
            "wind": 184 - index * 22,
            "pressure": 932 + index * 14,
            "conf": round(72 + peak * 20 - index * 2),
            "lead": 48 + index * 12,
            "radius": 5,
            "lat": lat,
            "lon": lon,
            "region": region,
            "track": [[lat - 0.8, lon - 0.8], [lat - 0.3, lon - 0.3], [lat, lon], [lat + 0.5, lon + 0.4]],
        })
    return {"anomalies": anomalies, "model": {"stage": "stage-1", "nodes": mesh.num_nodes, "peak_signal": round(peak, 3)}, "tracker": result}


def fetch_json(url: str, method: str = "GET", payload: dict | None = None, headers: dict | None = None) -> dict:
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    request_headers = {"Accept": "application/json", "User-Agent": "vayunetra/0.1"}
    if headers:
        request_headers.update(headers)
    if body is not None:
        request_headers["Content-Type"] = "application/json"
    request = Request(url, data=body, headers=request_headers, method=method)
    try:
        with urlopen(request, timeout=20) as response:
            return json.load(response)
    except HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"upstream weather provider returned {error.code}: {detail[:500]}") from error


def geocode(query: str) -> dict:
    if not query.strip():
        raise ValueError("search query is required")
    payload = fetch_json(
        "https://nominatim.openstreetmap.org/search?format=jsonv2&limit=1&q=" + quote(query.strip())
    )
    if not payload:
        raise ValueError("location not found")
    result = payload[0]
    return {"name": result["display_name"], "lat": float(result["lat"]), "lon": float(result["lon"])}


def live_weather(latitude: float, longitude: float) -> dict:
    imd_url = os.environ.get("IMD_API_URL")
    windy_key = os.environ.get("WINDY_API_KEY")
    if imd_url:
        url = imd_url.format(lat=latitude, lon=longitude)
        payload = fetch_json(url)
        required = {"valid_time", "latitude", "longitude", "variables"}
        missing = required.difference(payload)
        if missing:
            raise ValueError(f"IMD payload is missing fields: {sorted(missing)}")
        return {"source": "imd", "valid_time": payload["valid_time"], "latitude": payload["latitude"], "longitude": payload["longitude"], "variables": payload["variables"]}
    if windy_key:
        payload = fetch_json(
            "https://api.windy.com/api/point-forecast/v2",
            method="POST",
            payload={"lat": latitude, "lon": longitude, "model": "gfs", "parameters": WINDY_PARAMETERS, "levels": WINDY_LEVELS, "key": windy_key},
        )
        return {"source": "windy", "valid_time": payload.get("ts"), "latitude": latitude, "longitude": longitude, "parameters": payload}
    raise RuntimeError("live weather is not configured; set IMD_API_URL or WINDY_API_KEY")


def model_status():
    return {
        "status": "ready",
        "stage1": {
            "model": "HeatwaveGNN",
            "architecture": "Spherical GNN + Temporal Transformer",
            "checkpoint": HEATWAVE_CHECKPOINT.name,
            "window_days": 5,
            "horizon_days": 5,
            "tracking": "graph connected-components + temporal association",
        },
        "stage2": {
            "model": "ConditionalDiffusionDownscaler",
            "checkpoint": DIFFUSION_CHECKPOINT.name,
            "variables": [
                "u10",
                "v10",
                "t2m",
                "d2m",
                "tp",
                "msl",
                "wind_speed",
            ],
            "scale_factor": 4,
        },
        "integration": {
            "stage1_to_stage2_bridge": True,
            "real_stage2_normalization": False,
            "normalization_status":
                "waiting for training climatology and std.nc",
        },
    }

class ApiHandler(BaseHTTPRequestHandler):
    def _send(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.end_headers()

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        query = parse_qs(parsed.query)
        if path == "/api/health":
            self._send(200, {"ok": True, "model": "SphericalAnomalyGNN", "weather": "imd" if os.environ.get("IMD_API_URL") else "windy" if os.environ.get("WINDY_API_KEY") else "unconfigured"})
        elif path == "/api/model-status":
            self._send(200, model_status())
        elif path == "/api/anomalies":
            self._send(200, demo_response())
        elif path == "/api/historical":
            try:
                self._send(200, historical_response(query.get("year", [None])[0]))
            except (FileNotFoundError, ValueError, KeyError) as error:
                self._send(404, {"error": str(error)})
        elif path == "/api/geocode":
            try:
                self._send(200, {"location": geocode(query.get("q", [""])[0])})
            except (ValueError, KeyError, HTTPError, URLError) as error:
                self._send(400, {"error": str(error)})
        elif path == "/api/weather":
            try:
                latitude = float(query["lat"][0])
                longitude = float(query["lon"][0])
                self._send(200, live_weather(latitude, longitude))
            except (KeyError, ValueError, RuntimeError, HTTPError, URLError) as error:
                self._send(503, {"error": str(error)})
        else:
            self._send(404, {"error": "Not found"})

    def do_POST(self) -> None:
        if urlparse(self.path).path != "/api/predict":
            self._send(404, {"error": "Not found"})
            return
        try:
            size = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(size))
            forecast = np.asarray(payload["forecast"], dtype=np.float32)
            baseline = np.asarray(payload["baseline"], dtype=np.float32)
            self._send(200, run_tracker(forecast, baseline))
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            self._send(400, {"error": str(error)})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8000"))
    print(f"Extreme Weather API listening on http://localhost:{port}")
    ThreadingHTTPServer(("0.0.0.0", port), ApiHandler).serve_forever()
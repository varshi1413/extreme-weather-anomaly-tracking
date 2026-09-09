import "./Layout.css";
const LAYERS = [
  { id: "heatmap", label: "Anomaly", icon: "◉", swatch: "layer-hot" },
  { id: "coldmap", label: "Cold anomaly", icon: "◌", swatch: "layer-cold" },
  { id: "cyclone", label: "Cyclone tracker", icon: "↻", swatch: "layer-cyclone" },
  { id: "wind", label: "Wind field", icon: "≋", swatch: "layer-wind" },
  { id: "pressure", label: "Pressure", icon: "∿", swatch: "layer-pressure" },
  { id: "satellite", label: "Satellite", icon: "◒", swatch: "layer-satellite" },
];

function LayerMenu({ activeLayer, setActiveLayer, resolution, setResolution }) {
  return (
    <nav className="layer-menu">
      {LAYERS.map((layer) => (
        <button
          key={layer.id}
          className={
            "layer-menu-btn" +
            (activeLayer === layer.id ? " active" : "")
          }
          onClick={() => setActiveLayer(layer.id)}
          title={layer.label}
        >
          <span className={`layer-menu-swatch ${layer.swatch}`}>{layer.icon}</span>
          <span className="layer-menu-label">{layer.label}</span>
        </button>
      ))}

      <div className="layer-menu-divider"></div>

      <div className="seg-toggle layer-menu-res">
        <button
          className={"seg-btn" + (resolution === 12 ? " active" : "")}
          onClick={() => setResolution(12)}
        >
          12KM
        </button>

        <button
          className={"seg-btn" + (resolution === 5 ? " active" : "")}
          onClick={() => setResolution(5)}
        >
          5KM
        </button>
      </div>
    </nav>
  );
}

export default LayerMenu;
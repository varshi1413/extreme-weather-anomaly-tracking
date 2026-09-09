import { useEffect, useState } from "react";

function Footer() {
  const [rate, setRate] = useState("2.4");

  useEffect(() => {
    const updateRate = () => {
      const newRate = (
        2.1 +
        Math.random() * 0.6
      ).toFixed(1);

      setRate(newRate);
    };

    updateRate();

    const timer = setInterval(
      updateRate,
      1400
    );

    return () => clearInterval(timer);
  }, []);

  return (
    <footer className="statusbar">

      <span>
        PyTorch/JAX · DGL Icosahedral Mesh ·
        HF Diffusers · Xarray/Dask · MetPy ·
        Cartopy · Leaflet
      </span>

      <span
        id="ingestRate"
        className="mono"
      >
        INGEST {rate} GB/s
      </span>

    </footer>
  );
}

export default Footer;
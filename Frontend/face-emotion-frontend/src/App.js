import React, { useRef, useState, useEffect } from "react";
import Webcam from "react-webcam";
import axios from "axios";

function App() {
  const webcamRef = useRef(null);
  const [result, setResult] = useState(null);

  // Function to capture and send frame to FastAPI
  const captureAndPredict = async () => {
    if (webcamRef.current) {
      const imageSrc = webcamRef.current.getScreenshot();

      try {
        const res = await axios.post("http://127.0.0.1:8000/predict/", {
          image: imageSrc,
        });
        setResult(res.data);
      } catch (err) {
        console.error("Prediction error:", err);
      }
    }
  };

  // Run prediction every 30 seconds
  useEffect(() => {
    const interval = setInterval(() => {
      captureAndPredict();
    }, 1000); // 30000ms = 30 sec

    return () => clearInterval(interval); // cleanup on unmount
  }, []);

  return (
    <div style={{ textAlign: "center", marginTop: "20px" }}>
      <h2>Live Face Emotion Recognition (Every 30 sec)</h2>
      <Webcam
        ref={webcamRef}
        screenshotFormat="image/jpeg"
        style={{ width: 480, height: 360, borderRadius: "10px" }}
      />

      {result && (
        <div style={{ marginTop: "20px" }}>
          <h3>Last Prediction:</h3>
          <p><strong>Emotion:</strong> {result.emotion}</p>
          <p><strong>Confidence:</strong> {(result.confidence * 100).toFixed(2)}%</p>
          <p style={{ fontSize: "12px", color: "gray" }}>
            (Updated every 30 seconds)
          </p>
        </div>
      )}
    </div>
  );
}

export default App;

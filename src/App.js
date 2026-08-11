import React, { useEffect, useRef, useState } from "react";
import "./App.css";

function App() {
  const [image, setImage] = useState(null);
  const [preview, setPreview] = useState(null);
  const [type, setType] = useState("oral");
  const [result, setResult] = useState("");
  const [confidence, setConfidence] = useState(0);
  const [isDragging, setIsDragging] = useState(false);
  const fileInputRef = useRef(null);

  useEffect(() => {
    return () => {
      if (preview) URL.revokeObjectURL(preview);
    };
  }, [preview]);

  const setFile = (file) => {
    if (!file) return;
    if (!file.type?.startsWith("image/")) {
      alert("Please drop/select an image file");
      return;
    }

    setImage(file);
    setPreview((prev) => {
      if (prev) URL.revokeObjectURL(prev);
      return URL.createObjectURL(file);
    });
    setResult("");
    setConfidence(0);
  };

  const handleInputChange = (e) => {
    const file = e.target.files?.[0];
    setFile(file);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);

    const file = e.dataTransfer?.files?.[0];
    setFile(file);
  };

  const submitForm = async (e) => {
    e.preventDefault();

    if (!image) {
      alert("Please select an image");
      return;
    }

    // reset so the bar animates up on each prediction
    setResult("");
    setConfidence(0);

    const formData = new FormData();
    formData.append("image", image);
    formData.append("type", type);

    try {
      const res = await fetch("http://127.0.0.1:5000/predict", {
        method: "POST",
        body: formData
      });

      const data = await res.json();
      if (!res.ok) throw new Error(data?.error || "Prediction failed");

      setResult(data.prediction);
      setConfidence(Math.round(data.confidence * 100));
    } catch (err) {
      alert(err?.message || "Prediction failed");
    }
  };

  return (
    <div className="container">
      <h2>AI-Based Multi-Cancer Detection</h2>

      <form onSubmit={submitForm}>
        <select value={type} onChange={(e) => setType(e.target.value)}>
          <option value="oral">Oral Cancer</option>
          <option value="eye">Eye Cancer</option>
          <option value="skin">Skin Cancer</option>
        </select>

        <div
          className={`dropzone ${isDragging ? "isDragging" : ""}`}
          onClick={() => fileInputRef.current?.click()}
          onDragEnter={(e) => {
            e.preventDefault();
            e.stopPropagation();
            setIsDragging(true);
          }}
          onDragOver={(e) => {
            e.preventDefault();
            e.stopPropagation();
            setIsDragging(true);
          }}
          onDragLeave={(e) => {
            e.preventDefault();
            e.stopPropagation();
            setIsDragging(false);
          }}
          onDrop={handleDrop}
          role="button"
          tabIndex={0}
          onKeyDown={(e) => {
            if (e.key === "Enter" || e.key === " ") {
              e.preventDefault();
              fileInputRef.current?.click();
            }
          }}
        >
          <p>{isDragging ? "Drop the image here" : "Drag & Drop image here"}</p>
          <p className="dropzoneSubtext">or click to browse</p>
          <input
            ref={fileInputRef}
            className="fileInput"
            type="file"
            accept="image/*"
            onChange={handleInputChange}
          />
        </div>

        {preview && (
          <img src={preview} alt="preview" className="preview" />
        )}

        <button type="submit">Predict</button>

        <button
          type="button"
          style={{ marginTop: "10px", background: "#16a34a" }}
          onClick={async () => {
            try {
              await fetch("http://127.0.0.1:5000/webcam");
              alert("Webcam started");
            } catch {
              alert("Failed to start webcam");
            }
          }}
        >
          Open Webcam Detection
        </button>
      </form>

      {result && (
        <div className="result">
          <h3>{result}</h3>
          <div className="confidenceGraph" aria-label="Confidence graph">
            <div
              className="confidenceGraphFill"
              style={{ width: `${confidence}%` }}
            />
          </div>
          <p>Confidence: {confidence}%</p>
        </div>
      )}
    </div>
  );
}

export default App;
import { useState } from "react";
import axios from "axios";
import "./App.css";

const API_URL = "http://127.0.0.1:5000";
function App() {
   const [mode, setMode] = useState("image");
   const [image, setImage] = useState(null);
   const [preview, setPreview] = useState("");
   const [video, setVideo] = useState(null);
   const [videoPreview, setVideoPreview] = useState("");
   const [result, setResult] = useState(null);
   const [loading, setLoading] = useState(false);
   const [error, setError] = useState("");
   const [history, setHistory] = useState([]);
   const [historyLoading, setHistoryLoading] = useState(false);
   const [historyError, setHistoryError] = useState("");
   const [showHistory, setShowHistory] = useState(false);
   const [gradcamResult, setGradcamResult] = useState(null);
   const [gradcamLoading, setGradcamLoading] = useState(false);
   function handleModeChange(newMode) {
     setMode(newMode);
     setResult(null);
     setError("");
   }
   function handleImageChange(event) {
     const file = event.target.files[0];
     if (!file) return;
     if (!file.type.startsWith("image/")) {
       setError("Please select a valid image.");
       return;
     }
     setImage(file);
     setPreview(URL.createObjectURL(file));
     setResult(null);
     setError("");
   }
   function handleVideoChange(event) {
     const file = event.target.files[0];
     if (!file) return;
     const allowedExtensions = /\.(mp4|avi|mov|mkv|webm)$/i;
     if (!allowedExtensions.test(file.name)) {
       setError("Please select an MP4, AVI, MOV, MKV, or WEBM video.");
       return;
     }
     setVideo(file);
     setVideoPreview(URL.createObjectURL(file));
     setResult(null);
     setError("");
   }
   async function handleImageDetection() {
     if (!image) {
       setError("Please select an image first.");
       return;
     }
     const formData = new FormData();
     formData.append("image", image);
     try {
       setLoading(true);
       setError("");
       setResult(null);
       const response = await axios.post(
         `${API_URL}/predict`,
         formData
       );
       setResult(response.data);
     } catch (err) {
       setError(
         err.response?.data?.error ||
           "Unable to connect to the detection server. Make sure Flask is running."
       );
     } finally {
       setLoading(false);
     }
   }
   
  async function handleGradcam() {
  if (!image) {
    setError("Please select an image first.");
    return;
  }

  const formData = new FormData();
  formData.append("image", image);

  try {
    setGradcamLoading(true);
    setError("");
    setGradcamResult(null);

    const response = await axios.post(
      `${API_URL}/gradcam`,
      formData,
      { timeout: 120000 }
    );

    setGradcamResult(response.data);
  } catch (err) {
    setError(
      err.response?.data?.error ||
      "Unable to generate the heatmap. Make sure Flask is running."
    );
  } finally {
    setGradcamLoading(false);
  }
}

   async function handleVideoDetection() {
     if (!video) {
       setError("Please select a video first.");
       return;
     }
     const formData = new FormData();
     formData.append("video", video);
     try {
       setLoading(true);
       setError("");
       setResult(null);
       const response = await axios.post(
         `${API_URL}/predict-video`,
         formData,
         {
           timeout: 300000,
         }
       );
       setResult(response.data);
     } catch (err) {
       setError(
         err.code === "ECONNABORTED"
           ? "Video analysis took too long. Try a shorter video."
           : err.response?.data?.error ||
               "Unable to analyze the video. Check that Flask is running."
       );
     } finally {
       setLoading(false);
     }
   }
   async function handleHistory() {
   setHistoryLoading(true);
   setHistoryError("");
   try {
     const response = await axios.get(`${API_URL}/history`);
     setHistory(response.data.history || []);
   } catch (err) {
     setHistoryError(
       err.response?.data?.error ||
         "Unable to load history. Make sure Flask is running."
     );
   } finally {
     setHistoryLoading(false);
   }
}
   function handleReset() {
     setImage(null);
     setPreview("");
     setVideo(null);
     setVideoPreview("");
     setResult(null);
     setError("");
   }
   const isVideoResult = result && "frame_details" in result;
   return (
     <div className="app">
       <header className="navbar">
         <div className="brand">
           <span className="brand-icon">D</span>
           <span>
             Deepfake<span className="brand-highlight">Detector</span>
           </span>
         </div>
         <div className="navbar-actions">
   <button
     type="button"
     className="history-nav-button"
     onClick={() => {
       setShowHistory(true);
       handleHistory();
     }}
     disabled={historyLoading}
   >
     {historyLoading ? "Loading..." : "📋 History"}
   </button>
   <span className="status">
     <span className="status-dot"></span>
     AI Detection System
   </span>
</div>
       </header>
       <main className="main-content">
         <section className="hero">
           <span className="eyebrow">AI-POWERED MEDIA ANALYSIS</span>
           <h1>
             Is it real, or is it <span>synthetic?</span>
           </h1>
           <p>
             Analyze images and videos with our experimental deepfake
             detection model.
           </p>
         </section>
         <section className="detector-card">
           <div className="card-heading">
             <div>
               <h2>Media Detection</h2>
               <p>Select what you want to analyze.</p>
             </div>
             <span className="model-tag">EfficientNet-B0</span>
           </div>
           <div className="mode-switch">
             <button
               type="button"
               className={mode === "image" ? "mode-button active" : "mode-button"}
               onClick={() => handleModeChange("image")}
             >
               🖼️ Image Detection
             </button>
             <button
               type="button"
               className={mode === "video" ? "mode-button active" : "mode-button"}
               onClick={() => handleModeChange("video")}
             >
               🎬 Video Detection
             </button>
           </div>
           {mode === "image" ? (
             <>
               <label className="upload-area">
                 {preview ? (
                   <img
                     src={preview}
                     alt="Selected image preview"
                     className="image-preview"
                   />
                 ) : (
                   <div className="upload-placeholder">
                     <div className="upload-icon">↑</div>
                     <h3>Upload your image</h3>
                     <p>Click here to browse your files</p>
                     <span>JPG, JPEG, PNG, WEBP</span>
                   </div>
                 )}
                 <input
                   type="file"
                   accept="image/*"
                   onChange={handleImageChange}
                   hidden
                 />
               </label>
               {image && (
                 <div className="selected-file">
                   <span>Selected: {image.name}</span>
                   <button type="button" onClick={handleReset}>
                     Remove
                   </button>
                 </div>
               )}
               <button
                 className="detect-button"
                 onClick={handleImageDetection}
                 disabled={!image || loading}
               >
                 {loading ? "Analyzing Image..." : "Detect Deepfake"}
               </button>
               
<button
  type="button"
  className="detect-button gradcam-button"
  onClick={handleGradcam}
  disabled={!image || gradcamLoading}
>
  {gradcamLoading
    ? "Generating Heatmap..."
    : "Generate Grad-CAM Heatmap"}
</button>

             </>
           ) : (
             <>
               <label className="upload-area video-upload-area">
                 {videoPreview ? (
                   <video
                     src={videoPreview}
                     controls
                     className="video-preview"
                   />
                 ) : (
                   <div className="upload-placeholder">
                     <div className="upload-icon">▶</div>
                     <h3>Upload your video</h3>
                     <p>Click here to browse your files</p>
                     <span>MP4, AVI, MOV, MKV, WEBM</span>
                   </div>
                 )}
                 <input
                   type="file"
                   accept=".mp4,.avi,.mov,.mkv,.webm,video/*"
                   onChange={handleVideoChange}
                   hidden
                 />
               </label>
               {video && (
                 <div className="selected-file">
                   <span>Selected: {video.name}</span>
                   <button type="button" onClick={handleReset}>
                     Remove
                   </button>
                 </div>
               )}
               <button
                 className="detect-button"
                 onClick={handleVideoDetection}
                 disabled={!video || loading}
               >
                 {loading ? "Analyzing Video..." : "Analyze Video"}
               </button>
             </>
           )}
           {loading && (
             <p className="loading-message">
               {mode === "video"
                 ? "Analyzing sampled video frames. This may take a little while..."
                 : "Please wait while the model analyzes your image."}
             </p>
           )}
           {error && (
             <div className="error-message" role="alert">
               {error}
             </div>
           )}
           {result?.duplicate_detected && result.duplicate_of && (
          <section className="duplicate-warning" aria-live="polite">
            <h3>⚠️ Duplicate File Detected</h3>
            <p>{result.duplicate_message}</p>
            <div className="duplicate-details">
              <p>
                <strong>Previously analyzed file:</strong>{" "}
                {result.duplicate_of.filename}
              </p>
              <p>
                <strong>Previous prediction:</strong>{" "}
                {result.duplicate_of.prediction}
              </p>
              <p>
                <strong>Previously analyzed on:</strong>{" "}
                {result.duplicate_of.created_at}
              </p>
              <p>
                <strong>Record ID:</strong>{" "}
                {result.duplicate_of.id}
              </p>
            </div>
            <p className="duplicate-note">
              SHA-256 detects byte-identical files. Edited, resized, or re-encoded
              files generally have different hashes.
            </p>
          </section>
        )}

        {result && !isVideoResult && (
             <section className="result-panel" aria-live="polite">
               <span className="result-label">MODEL PREDICTION</span>
               <h2
                 className={
                   result.prediction.toLowerCase() === "fake"
                     ? "fake-result"
                     : "real-result"
                 }
               >
                 {result.prediction === "Fake"
                   ? "Potentially Fake"
                   : "Potentially Real"}
               </h2>
               <div className="score-row">
                 <div className="score-heading">
                   <span>Fake score</span>
                   <strong>{result.class_scores.fake}%</strong>
                 </div>
                 <div className="score-track">
                   <div
                     className="score-fill fake-fill"
                     style={{ width: `${result.class_scores.fake}%` }}
                   />
                 </div>
               </div>
               <div className="score-row">
                 <div className="score-heading">
                   <span>Real score</span>
                   <strong>{result.class_scores.real}%</strong>
                 </div>
                 <div className="score-track">
                   <div
                     className="score-fill real-fill"
                     style={{ width: `${result.class_scores.real}%` }}
                   />
                 </div>
               </div>
               <p className="result-warning">⚠️ {result.warning}</p>
             </section>
           )}
           
{gradcamResult?.gradcam_image && (
  <section className="result-panel gradcam-panel">
    <span className="result-label">MODEL EXPLANATION</span>

    <h2>Grad-CAM Heatmap</h2>

    <img
      src={`data:image/png;base64,${gradcamResult.gradcam_image}`}
      alt="Grad-CAM heatmap showing image regions that influenced the model's prediction"
      className="gradcam-image"
    />

    <p className="result-warning">
      ⚠️ {gradcamResult.warning}
    </p>
  </section>
)}

           {isVideoResult && (
             <section className="result-panel" aria-live="polite">
               <span className="result-label">VIDEO ANALYSIS REPORT</span>
               <h2
                 className={
                   result.majority_vote_result === "Potentially Fake"
                     ? "fake-result"
                     : result.majority_vote_result === "Potentially Real"
                       ? "real-result"
                       : ""
                 }
               >
                 {result.majority_vote_result}
               </h2>
               <p>
                 Video duration: <strong>{result.video_duration_seconds} seconds</strong>
               </p>
               <p>
                 Frames analyzed: <strong>{result.total_frames_analyzed}</strong>
               </p>
               <div className="score-row">
                 <div className="score-heading">
                   <span>Frames classified as Fake</span>
                   <strong>
                     {result.fake_frames} ({result.fake_percentage}%)
                   </strong>
                 </div>
                 <div className="score-track">
                   <div
                     className="score-fill fake-fill"
                     style={{ width: `${result.fake_percentage}%` }}
                   />
                 </div>
               </div>
               <div className="score-row">
                 <div className="score-heading">
                   <span>Frames classified as Real</span>
                   <strong>
                     {result.real_frames} ({result.real_percentage}%)
                   </strong>
                 </div>
                 <div className="score-track">
                   <div
                     className="score-fill real-fill"
                     style={{ width: `${result.real_percentage}%` }}
                   />
                 </div>
               </div>
               <details className="frame-details">
                 <summary>View frame-by-frame results</summary>
                 <div className="frame-table-wrapper">
                   <table className="frame-table">
                     <thead>
                       <tr>
                         <th>Time (s)</th>
                         <th>Prediction</th>
                         <th>Fake score</th>
                         <th>Real score</th>
                       </tr>
                     </thead>
                     <tbody>
                       {result.frame_details.map((frame) => (
                         <tr key={frame.timestamp_seconds}>
                           <td>{frame.timestamp_seconds}</td>
                           <td>{frame.prediction}</td>
                           <td>{frame.fake_score}%</td>
                           <td>{frame.real_score}%</td>
                         </tr>
                       ))}
                     </tbody>
                   </table>
                 </div>
               </details>
               <p className="result-warning">⚠️ {result.warning}</p>
             </section>
           )}
         </section>
{showHistory && (
   <section className="history-card" id="history">
     <div className="history-heading">
       <div>
         <h2>Analysis History</h2>
         <p>Previously analyzed images and videos</p>
       </div>
       <button
         type="button"
         className="history-refresh-button"
         onClick={handleHistory}
         disabled={historyLoading}
       >
         {historyLoading ? "Refreshing..." : "↻ Refresh"}
       </button>
     </div>
     {historyError && (
       <p className="history-error" role="alert">
         {historyError}
       </p>
     )}
     {!historyLoading && !historyError && history.length === 0 && (
       <p>No analysis records found.</p>
     )}
     {history.length > 0 && (
       <div className="history-table-wrapper">
         <table className="history-table">
           <thead>
             <tr>
               <th>ID</th>
               <th>Filename</th>
               <th>Type</th>
               <th>Prediction</th>
               <th>SHA-256 Fingerprint</th>
               <th>Analyzed At</th>
             </tr>
           </thead>
           <tbody>
             {history.map((record) => (
               <tr key={record.id}>
                 <td>{record.id}</td>
                 <td className="history-filename">
                   {record.filename}
                 </td>
                 <td>{record.file_type}</td>
                 <td>{record.prediction}</td>
                 <td>
                   <code className="hash-value">
                     {record.sha256_hash}
                   </code>
                 </td>
                 <td>{record.created_at}</td>
               </tr>
             ))}
           </tbody>
         </table>
       </div>
     )}
     <p className="history-note">
       SHA-256 identifies the exact file bytes. A prediction is an
       experimental model result, not proof of authenticity.
     </p>
   </section>
)}
         <p className="disclaimer">
           This is a research prototype. Model predictions can be wrong and
           should not be treated as proof of authenticity or manipulation.
         </p>
       </main>
       <footer>
         Deepfake Detection &amp; Provenance System · Research Prototype
       </footer>
     </div>
   );
}
export default App;

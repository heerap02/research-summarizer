import React, { useState } from "react";

function FileUpload() {
  const [summary, setSummary] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [fileName, setFileName] = useState("");

  const handleFileChange = async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    const allowedTypes = ["application/pdf", "application/vnd.openxmlformats-officedocument.wordprocessingml.document", "text/plain"];
    if (!allowedTypes.includes(file.type)) {
      setError("❌ Unsupported file type. Please upload a PDF, DOCX, or TXT.");
      return;
    }

    const formData = new FormData();
    formData.append("file", file);
    setFileName(file.name);
    setLoading(true);
    setError("");
    setSummary(null);

    try {
      const response = await fetch("http://localhost:8000/summarize/", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) throw new Error("Upload failed");

      const data = await response.json();
      setSummary(data.final_summary);
    } catch {
      setError("Something went wrong. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="upload-card">
      <label className="upload-label">
        <input type="file" onChange={handleFileChange} accept=".pdf,.docx,.txt" />
        <span>📎 Choose File</span>
      </label>
      {fileName && <p className="file-name">📄 {fileName}</p>}
      {loading && <p className="status">⏳ Summarizing...</p>}
      {error && <p className="error">{error}</p>}
      {summary && (
       <div className="summary-box">
         <h2>📝 Summary</h2>
         {summary.split('\n\n').map((para, index) => (
         <p key={index}>{para}</p>
         ))}
       </div>
      )}

    </div>
  );
}

export default FileUpload;
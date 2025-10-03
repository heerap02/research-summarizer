import React from "react";
import FileUpload from "./FileUpload";
import "./App.css";

function App() {
  return (
    <div className="App">
      <header>
        <h1>📄Concise Research Summarizer</h1>
        <p className="subtitle">Upload a document to generate a concise summary</p>
      </header>
      <FileUpload />
    </div>
  );
}

export default App;

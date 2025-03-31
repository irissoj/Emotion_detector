import React, { useState } from "react";
import axios from "axios";

function App() {
  const [text, setText] = useState("");
  const [emotion, setEmotion] = useState("");

  const handleSubmit = async () => {
    const response = await axios.post("http://127.0.0.1:8000/api/predict/", { text });
    setEmotion(response.data.predicted_emotion);
  };

  return (
    <div className="container">
      <h2>Multi-Modal Emotion Detection</h2>
      <input type="text" value={text} onChange={(e) => setText(e.target.value)} />
      <button onClick={handleSubmit}>Predict</button>
      {emotion && <h3>Emotion: {emotion}</h3>}
    </div>
  );
}

export default App;
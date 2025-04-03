import React, { useState, useRef } from "react";
import axios from "axios";

function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [audioEmotion, setAudioEmotion] = useState(null);
  const [loading, setLoading] = useState(false);
  const fileInputRef = useRef(null);

  // Set base URL for Django backend
  const API_BASE = "http://localhost:8000";

  const handleAudioUpload = async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    const formData = new FormData();
    formData.append("audio", file);

    try {
      setLoading(true);
      const response = await axios.post(
        `${API_BASE}/api/analyze-audio/`,
        formData,
        {
          headers: {
            "Content-Type": "multipart/form-data",
          },
        }
      );

      setAudioEmotion(response.data.emotion);
      setMessages((prev) => [
        ...prev,
        {
          type: "system",
          content: `Voice emotion detected: ${response.data.emotion}`,
        },
      ]);
    } catch (error) {
      console.error("Audio analysis error:", error);
      setMessages((prev) => [
        ...prev,
        {
          type: "error",
          content: "Failed to analyze audio. Please try again.",
        },
      ]);
    } finally {
      setLoading(false);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!input.trim() || loading) return;

    const userMessage = { type: "user", content: input };
    setMessages((prev) => [...prev, userMessage]);
    setInput("");
    setLoading(true);

    try {
      const response = await axios.post(
        `${API_BASE}/api/chat/`,
        {
          text: input,
          audio_emotion: audioEmotion,
        },
        {
          headers: {
            "Content-Type": "application/json",
          },
        }
      );

      setMessages((prev) => [
        ...prev,
        {
          type: "bot",
          content: response.data.response,
          emotion: response.data.text_emotion,
        },
      ]);
    } catch (error) {
      console.error("Chat error:", error);
      setMessages((prev) => [
        ...prev,
        {
          type: "error",
          content: "Failed to get response. Please try again.",
        },
      ]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="chat-container">
      <div className="chat-header">
        <h1>Multimodal Emotion Chat</h1>
        <input
          type="file"
          accept="audio/*"
          ref={fileInputRef}
          onChange={handleAudioUpload}
          style={{ display: "none" }}
          disabled={loading}
        />
        <button onClick={() => fileInputRef.current.click()} disabled={loading}>
          {loading ? "Processing..." : "Upload Audio Emotion"}
        </button>
      </div>

      <div className="chat-messages">
        {messages.map((msg, index) => (
          <div key={index} className={`message ${msg.type}`}>
            <div className="message-content">
              {msg.content}
              {msg.emotion && (
                <span className="emotion-badge">{msg.emotion}</span>
              )}
            </div>
          </div>
        ))}
      </div>

      <form onSubmit={handleSubmit} className="chat-input">
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Type your message..."
          disabled={loading}
        />
        <button type="submit" disabled={loading}>
          {loading ? "Sending..." : "Send"}
        </button>
      </form>
    </div>
  );
}

export default App;

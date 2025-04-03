import React, { useState, useRef } from "react";
import axios from "axios";
import logo from "./logo.png"; // Import your logo file

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

  const formatMessage = (text) => {
    return text.split("**").map((part, index) => {
      if (index % 2 === 1) {
        return (
          <span key={index} className="emphasis-text">
            {part}
          </span>
        );
      }
      return part;
    });
  };

  // Then modify the message rendering part:
  <div className="message-content">
    {messages.map((msg, index) => (
      <div key={index} className={`message ${msg.type}`}>
        {formatMessage(msg.content)}
        {msg.emotion && <span className="emotion-badge">{msg.emotion}</span>}
      </div>
    ))}
  </div>;

  return (
    <div className="app-container">
      <div className="chat-container">
        <div className="chat-header">
          <div className="logo-container">
            <img src={logo} alt="Emotion AI Logo" className="logo" />
            <h1>Multimodal Emotion Chat</h1>
          </div>
          <input
            type="file"
            accept="audio/*"
            ref={fileInputRef}
            onChange={handleAudioUpload}
            style={{ display: "none" }}
            disabled={loading}
          />
          <button
            className="upload-btn"
            onClick={() => fileInputRef.current.click()}
            disabled={loading}
          >
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

      <style jsx>{`
        .message-content {
          line-height: 1.6;
          font-size: 1rem;
          color: #374151;
        }

        .emphasis-text {
          font-weight: 600;
          color: #1f2937;
        }

        .message-content strong {
          font-weight: 600;
          color: #111827;
        }

        .message-content em {
          font-style: italic;
          color: #4b5563;
        }

        .message-content ul {
          padding-left: 1.5rem;
          margin: 0.5rem 0;
          list-style-type: disc;
        }

        .message-content li {
          margin-bottom: 0.25rem;
        }
        .app-container {
          height: 100vh;
          background-color: #f5f5f5;
          display: flex;
          justify-content: center;
          align-items: center;
        }

        .chat-container {
          width: 800px;
          height: 90vh;
          display: flex;
          flex-direction: column;
          background-color: white;
          border-radius: 10px;
          box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
          overflow: hidden;
        }

        .chat-header {
          padding: 15px 20px;
          background-color: #ffffff;
          display: flex;
          justify-content: space-between;
          align-items: center;
          border-bottom: 1px solid #e0e0e0;
        }

        .logo-container {
          display: flex;
          align-items: center;
          gap: 15px;
        }

        .logo {
          height: 50px;
          width: auto;
        }

        .chat-header h1 {
          color: #333;
          font-size: 1.5rem;
          margin: 0;
          font-weight: 600;
        }

        .upload-btn {
          padding: 8px 16px;
          background-color: #6b7280;
          color: white;
          border: none;
          border-radius: 5px;
          cursor: pointer;
          font-size: 0.9rem;
          transition: background-color 0.2s;
        }

        .upload-btn:hover {
          background-color: #4b5563;
        }

        .upload-btn:disabled {
          background-color: #d1d5db;
          cursor: not-allowed;
        }

        .chat-messages {
          flex: 1;
          overflow-y: auto;
          padding: 20px;
          background-color: #f9f9f9;
        }

        .message {
          margin-bottom: 15px;
          display: flex;
        }

        .message.user {
          justify-content: flex-end;
        }

        .message.bot,
        .message.system,
        .message.error {
          justify-content: flex-start;
        }

        .message-content {
          max-width: 70%;
          padding: 12px 16px;
          border-radius: 8px;
          position: relative;
          background-color: #ffffff;
          color: #333;
          border: 1px solid #e0e0e0;
          box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05);
        }

        .user .message-content {
          background-color: #6b7280;
          color: white;
          border-color: #6b7280;
        }

        .system .message-content {
          background-color: #e5e7eb;
          border-color: #d1d5db;
        }

        .error .message-content {
          background-color: #fee2e2;
          border-color: #fca5a5;
          color: #dc2626;
        }

        .emotion-badge {
          position: absolute;
          top: -8px;
          right: -8px;
          background-color: #6b7280;
          color: white;
          padding: 3px 10px;
          border-radius: 12px;
          font-size: 0.75rem;
          font-weight: 500;
        }

        .chat-input {
          padding: 15px;
          background-color: white;
          display: flex;
          border-top: 1px solid #e0e0e0;
        }

        .chat-input input {
          flex: 1;
          padding: 10px 15px;
          border: 1px solid #e0e0e0;
          border-radius: 5px;
          margin-right: 10px;
          font-size: 1rem;
          outline: none;
          transition: border-color 0.2s;
        }

        .chat-input input:focus {
          border-color: #6b7280;
        }

        .chat-input button {
          padding: 10px 20px;
          background-color: #6b7280;
          border: none;
          border-radius: 5px;
          color: white;
          cursor: pointer;
          font-size: 1rem;
          transition: background-color 0.2s;
        }

        .chat-input button:hover {
          background-color: #4b5563;
        }

        .chat-input button:disabled {
          background-color: #d1d5db;
          cursor: not-allowed;
        }
      `}</style>
    </div>
  );
}

export default App;

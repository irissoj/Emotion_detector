import React, { useState, useRef } from "react";
import {
  AppBar,
  Toolbar,
  Avatar,
  Typography,
  Box,
  TextField,
  IconButton,
  Grid,
  Paper,
  Container,
} from "@mui/material";
import { Send, Mic, FileUpload } from "@mui/icons-material";
import ChatMessage from "./Components/ChatMessage";

function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [audioFile, setAudioFile] = useState(null);
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  const sendMessage = async () => {
    if (!input.trim() && !audioFile) return;

    const newMessages = [
      ...messages,
      { text: input, isBot: false, audio: audioFile },
    ];
    setMessages(newMessages);

    const formData = new FormData();
    formData.append("text", input);
    if (audioFile) {
      formData.append("audio", audioFile);
    }

    try {
      const response = await fetch("http://localhost:8000/api/process/", {
        method: "POST",
        body: formData,
      });

      const data = await response.json();
      setMessages([...newMessages, { text: data.response, isBot: true }]);
    } catch (error) {
      console.error("Error:", error);
    }

    setInput("");
    setAudioFile(null);
    scrollToBottom();
  };

  return (
    <Box sx={{ display: "flex", flexDirection: "column", height: "100vh" }}>
      <AppBar
        position="static"
        sx={{
          bgcolor: "white",
          color: "black",
          boxShadow: "none",
          borderBottom: "1px solid #e0e0e0",
        }}
      >
        <Toolbar sx={{ padding: "8px !important" }}>
          <Avatar
            src="/logo.png"
            sx={{
              width: 300, // Increased from 120
              height: 150, // Increased from 40
              borderRadius: 1,
              "& img": {
                objectFit: "contain",
                padding: "8px", // Add padding around the logo
              },
            }}
            variant="square"
          />
        </Toolbar>
      </AppBar>
      <Box sx={{ flexGrow: 1, overflow: "auto", bgcolor: "#f0f4f8", p: 2 }}>
        <Container maxWidth="lg">
          {messages.map((msg, i) => (
            <ChatMessage key={i} message={msg} />
          ))}
          <div ref={messagesEndRef} />
        </Container>
      </Box>
      <Box sx={{ p: 2, bgcolor: "white", borderTop: "1px solid #e0e0e0" }}>
        <Container maxWidth="lg">
          <Grid container spacing={1} alignItems="center">
            <Grid item>
              <IconButton component="label">
                <FileUpload />
                <input
                  type="file"
                  hidden
                  accept="audio/*"
                  onChange={(e) => setAudioFile(e.target.files[0])}
                />
              </IconButton>
            </Grid>
            <Grid item xs>
              <TextField
                fullWidth
                variant="outlined"
                placeholder="Type your message..."
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyPress={(e) => e.key === "Enter" && sendMessage()}
                sx={{
                  "& .MuiOutlinedInput-root": {
                    borderRadius: 4,
                    bgcolor: "white",
                  },
                }}
              />
            </Grid>
            <Grid item>
              <IconButton
                onClick={sendMessage}
                sx={{
                  bgcolor: "primary.main",
                  color: "white",
                  "&:hover": { bgcolor: "primary.dark" },
                }}
              >
                <Send />
              </IconButton>
            </Grid>
          </Grid>
        </Container>
      </Box>
    </Box>
  );
}

export default App;

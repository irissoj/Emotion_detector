import React from "react";
import { Box, Paper, Typography, Avatar } from "@mui/material";
import { Mic } from "@mui/icons-material";

const ChatMessage = ({ message }) => {
  return (
    <Box
      sx={{
        display: "flex",
        justifyContent: message.isBot ? "flex-start" : "flex-end",
        mb: 2,
      }}
    >
      <Paper
        sx={{
          p: 2,
          maxWidth: "70%",
          bgcolor: message.isBot ? "white" : "primary.light",
          color: message.isBot ? "text.primary" : "white",
          borderRadius: message.isBot
            ? "20px 20px 20px 4px"
            : "20px 20px 4px 20px",
        }}
      >
        {message.audio && (
          <Box sx={{ mb: 1 }}>
            <audio controls style={{ width: "100%" }}>
              <source src={URL.createObjectURL(message.audio)} />
            </audio>
          </Box>
        )}
        <Typography variant="body1">{message.text}</Typography>
      </Paper>
    </Box>
  );
};

export default ChatMessage;

import React, { useEffect, useState } from "react";
import axios from "axios";
import "../styles/chathistory.css";

function SecondPage() {
  const [chatHistory, setChatHistory] = useState([]);
  const [modelInfo, setModelInfo] = useState({
    checkpoint: "",
    system_prompt: "",
    messages: [],
  });

  useEffect(() => {
    axios.get("http://127.0.0.1:8080/model_info").then((res) => {
      if (res.data.messages == null) {
        setModelInfo({
          checkpoint: res.data.checkpoint,
          system_prompt: res.data.system_prompt,
          messages: [],
        });
      } else {
        setModelInfo(res.data);
      }
    });
  }, []);

  useEffect(() => {
    const ws = new WebSocket("ws://127.0.0.1:8080/ws");

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);

      if (data.type === "model_info") {
        setModelInfo({
          checkpoint: data.checkpoint,
          system_prompt: data.system_prompt,
          messages: data.messages,
        });
      } else {
        const { super_prompt, response } = data;
        setChatHistory((prevHistory) => [
          ...prevHistory,
          { sender: "user", text: super_prompt },
          { sender: "ai", text: response },
        ]);
      }
    };
    return () => ws.close();
  }, []);

  const userMessages = modelInfo.messages
    .filter((msg) => msg.role === "user")
    .map((msg) => `${msg.content}`)
    .join("\n");

  const assistantMessages = modelInfo.messages
    .filter((msg) => msg.role === "assistant")
    .map((msg) => `${msg.content}`)
    .join("\n");

  return (
    <div className="app-container">
      <div className="model-recap">
        <h3>Model Information</h3>
        <p>
          <strong>Model :</strong> {modelInfo.checkpoint}
        </p>
        <strong>System Prompt :</strong>
        <p>
          <textarea
            className="textarea system_prompt-textarea"
            value={modelInfo.system_prompt}
            readOnly
            placeholder="System Prompt Set will appear here..."
          ></textarea>
        </p>
        <div className="user-messages-container">
          <strong>User Messages:</strong>
          <textarea
            className="textarea user-messages-textarea"
            value={userMessages}
            readOnly
            placeholder="User Messages will appear here..."
          ></textarea>
        </div>
        <div className="assistant-messages-container">
          <strong>Assistant Messages:</strong>
          <textarea
            className="textarea assistant-messages-textarea"
            value={assistantMessages}
            readOnly
            placeholder="Assistant Messages will appear here..."
          ></textarea>
        </div>
      </div>
      <div className="chat-container">
        <h2>Chat History</h2>
        <div className="chat-window">
          {chatHistory.map((message, index) => (
            <div
              key={index}
              className={`chat-bubble ${message.sender === "user" ? "user-bubble" : "ai-bubble"}`}
            >
              {message.text}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

export default SecondPage;

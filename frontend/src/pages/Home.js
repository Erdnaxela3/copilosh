import React, { useState, useEffect } from "react";
import axios from "axios";
import "../styles/home.css";

function Home() {
  const [model, setModel] = useState("");
  const [response, setResponse] = useState("");
  const [responseAI, setResponseAI] = useState("");
  const [loading, setLoading] = useState(false);
  const [loadingModel, setLoadingModel] = useState(false);
  const [command, setCommand] = useState("");
  const [stderr, setStderr] = useState("");
  const [systemPrompt, setSystemPrompt] = useState("");
  const [systemPrompts, setSystemPrompts] = useState([]);
  const [prePromptUser, setPrePromptUser] = useState("");
  const [prePromptAssistant, setPrePromptAssistant] = useState("");
  const [prePrompts, setPrePrompts] = useState([]);
  const [selectedPrePrompt, setSelectedPrePrompt] = useState("");
  const [selectedSystemPrompt, setSelectedSystemPrompt] = useState("");
  const [availableModels, setAvailableModels] = useState([]);

  useEffect(() => {
    axios.get("http://127.0.0.1:8080/system_prompts").then((res) => {
      setSystemPrompts([
        { id: -1, type: "No system Prompt", prompt: "" },
        ...res.data,
      ]);
    });
  }, []);

  useEffect(() => {
    axios.get("http://127.0.0.1:8080/pre_prompts").then((res) => {
      setPrePrompts([
        {
          id: "None",
          type: "No Pre Prompt",
          user_prompt: "",
          assistant_response: "",
        },
        ...res.data,
      ]);
    });
  }, []);

  useEffect(() => {
    axios.get("http://127.0.0.1:8080/models").then((res) => {
      setAvailableModels(res.data);
    });
  }, []);

  const loadModel = async () => {
    setLoadingModel(true);
    var messages = [];
    if (prePromptUser != "" || prePromptAssistant != "") {
      messages = [
        {
          role: "user",
          content: prePromptUser,
        },
        {
          role: "assistant",
          content: prePromptAssistant,
        },
      ];
    }
    try {
      const res = await axios.post("http://127.0.0.1:8080/load_model", {
        model: model,
        messages: messages,
        system_prompt: systemPrompt,
      });
      setResponse(res.data.message || res.data.error);
    } catch (err) {
      setResponse("Error: " + err.message);
    } finally {
      setLoadingModel(false);
    }
  };

  const handleErrorSubmit = async () => {
    setLoading(true);
    try {
      const res = await axios.post("http://127.0.0.1:8080/error", {
        command,
        exit_code: 1,
        stderr,
      });
      setResponseAI(res.data.explanation || "No explanation provided.");
    } catch (err) {
      setResponseAI("Error: " + err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleSystemPromptsChange = (event) => {
    const selectedId = event.target.value;
    setSelectedSystemPrompt(selectedId);

    const selected = systemPrompts.find((p) => p.id === parseInt(selectedId));
    if (selected) {
      setSystemPrompt(selected.prompt);
    }
  };

  const handlePrePromptsChange = (event) => {
    const selectedId = event.target.value;
    setSelectedPrePrompt(selectedId);

    const selected = prePrompts.find((p) => p.id === parseInt(selectedId));
    if (selected) {
      setPrePromptUser(selected.user_prompt);
      setPrePromptAssistant(selected.assistant_response);
    }
  };

  return (
    <div className="app-container">
      <h1>Load Model</h1>
      <div className="model-settings">
        <div className="input-container">
          <label htmlFor="model-select">Select Model : </label>
          <select
            id="model-select"
            value={model}
            onChange={(e) => setModel(e.target.value)}
            disabled={loadingModel}
          >
            <option value="" disabled>
              {loadingModel ? "Loading models..." : "Select a model"}
            </option>
            {availableModels.map((availableModel, index) => (
              <option key={index} value={availableModel}>
                {availableModel}
              </option>
            ))}
          </select>
        </div>
        <h2>Choose System Prompt</h2>
        <select
          onChange={handleSystemPromptsChange}
          value={selectedSystemPrompt}
          disabled={loadingModel}
        >
          <option value="">Select a personality...</option>
          {systemPrompts.map((personality) => (
            <option key={personality.id} value={personality.id}>
              {personality.type}
            </option>
          ))}
        </select>
        <textarea
          className="error-textarea"
          value={systemPrompt}
          readOnly
          placeholder="System Prompt will appear here..."
        ></textarea>

        <div className="pre-prompt-container">
          <div className="pre-prompt-header-container">
            <h2>Choose Pre-Prompt</h2>
            <select
              onChange={handlePrePromptsChange}
              value={selectedPrePrompt}
              style={{ width: "100%" }}
              disabled={loadingModel}
            >
              <option value="">Select a pre-prompt...</option>
              {prePrompts.map((prePrompt) => (
                <option key={prePrompt.id} value={prePrompt.id}>
                  Pre-Prompt id : {prePrompt.id}
                </option>
              ))}
            </select>
          </div>

          <div className="pre-prompt-user-container">
            User Pre-Prompt :
            <textarea
              className="pre-prompt-textarea pre-prompt-user-textarea"
              value={prePromptUser}
              readOnly
              placeholder="Pre-Prompt User will appear here..."
            ></textarea>
          </div>

          <div className="pre-prompt-assistant-container">
            Assistant Pre-Prompt :
            <textarea
              className="pre-prompt-textarea  pre-prompt-assistant-textarea"
              value={prePromptAssistant}
              readOnly
              placeholder="Pre-Prompt Assistant will appear here..."
            ></textarea>
          </div>
        </div>
      </div>
      <button onClick={loadModel} disabled={loadingModel}>
        {loadingModel ? (
          <span className="loading-spinner">Loading...</span>
        ) : (
          "Load Model"
        )}
      </button>
      <p>{response}</p>
      <div className="error-inputs">
        <h2>Submit an Error</h2>
        <input
          type="text"
          value={command}
          onChange={(e) => setCommand(e.target.value)}
          placeholder="Enter Command"
        />
        <textarea
          className="error-textarea"
          value={stderr}
          onChange={(e) => setStderr(e.target.value)}
          placeholder="Enter Prompt User"
        ></textarea>
        <button onClick={handleErrorSubmit} disabled={loading}>
          {loading ? "Submitting..." : "Submit Error"}
        </button>
      </div>
      <div className="response">
        <h3>AI Response</h3>
        <p className="response-responseAI" style={{ whiteSpace: "pre-wrap" }}>
          {responseAI}
        </p>
      </div>
    </div>
  );
}

export default Home;


import { useState, useRef } from "react";
import {
  Bot,
  Send,
  Plus,
  FileText,
  Upload,
  MessageSquare,
  Trash2,
  Sparkles,
  BrainCircuit,
  LoaderCircle,
  Paperclip,
  Menu,
  X,
  BookOpen,
  AlertCircle,
  CheckCircle2,
  FileUp
} from "lucide-react";

import "./App.css";

const API = "http://127.0.0.1:8000";

function App() {
  const [mode, setMode] = useState("pdf");

  const [file, setFile] = useState(null);
  const [uploaded, setUploaded] = useState(false);
  const [uploading, setUploading] = useState(false);

  const [messages, setMessages] = useState([]);
  const [question, setQuestion] = useState("");

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const [history, setHistory] = useState([]);
  const [sidebarOpen, setSidebarOpen] = useState(true);

  const fileInput = useRef(null);
  const chatEnd = useRef(null);

  const scrollToBottom = () => {
    setTimeout(() => {
      chatEnd.current?.scrollIntoView({
        behavior: "smooth"
      });
    }, 100);
  };

  // NEW CHAT
  const newChat = () => {
    setMessages([]);
    setQuestion("");
    setError("");
  };

  // SELECT PDF
  const handleFileSelect = (selectedFile) => {
    if (!selectedFile) return;

    if (selectedFile.type !== "application/pdf") {
      setError("Please select a valid PDF file.");
      return;
    }

    setFile(selectedFile);
    setUploaded(false);
    setError("");
    setMessages([]);
  };

  // UPLOAD PDF
  const uploadPDF = async () => {
    if (!file) {
      setError("Please select a PDF first.");
      return;
    }

    setUploading(true);
    setError("");

    try {
      const formData = new FormData();

      formData.append("file", file);

      const response = await fetch(`${API}/upload`, {
        method: "POST",
        body: formData
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || "PDF upload failed.");
      }

      setUploaded(true);

      setMessages([
        {
          role: "assistant",
          content: `Your PDF has been uploaded successfully! I can now retrieve information from ${file.name}. Ask me anything about this document.`
        }
      ]);

      scrollToBottom();

    } catch (err) {
      setError(err.message || "Unable to connect to backend.");
    } finally {
      setUploading(false);
    }
  };

  // ASK QUESTION
  const askQuestion = async (e) => {
    e.preventDefault();

    if (!question.trim() || loading) return;

    if (mode === "pdf" && !uploaded) {
      setError("Upload a PDF before asking questions.");
      return;
    }

    const currentQuestion = question.trim();

    setMessages(prev => [
      ...prev,
      {
        role: "user",
        content: currentQuestion
      }
    ]);

    setQuestion("");
    setLoading(true);
    setError("");

    scrollToBottom();

    try {
      const endpoint = mode === "pdf" ? "/ask" : "/chat";

      const body =
        mode === "pdf"
          ? { question: currentQuestion }
          : { prompt: currentQuestion };

      const response = await fetch(`${API}${endpoint}`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify(body)
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(
          data.detail || "Unable to generate a response."
        );
      }

      // Support different response formats.
      const answer =
        data.answer ??
        data.response ??
        data.result ??
        data.message ??
        (data.passages
          ? data.passages.map(p =>
              typeof p === "string"
                ? p
                : p.text || JSON.stringify(p)
            ).join("\n\n")
          : null);

      if (!answer) {
        throw new Error(
          "Backend responded, but no answer was found in its response."
        );
      }

      const sources =
        data.sources ??
        data.passages ??
        data.results ??
        [];

      const sourceList = Array.isArray(sources)
        ? sources.map((source, index) => ({
            page:
              source.page ??
              source.page_number ??
              source.pageNumber ??
              null,
            text:
              typeof source === "string"
                ? source
                : source.text ?? source.content ?? ""
          }))
        : [];

      setMessages(prev => [
        ...prev,
        {
          role: "assistant",
          content: answer,
          sources: sourceList
        }
      ]);

      setHistory(prev => {
        const title = currentQuestion.slice(0, 35);

        return [
          {
            id: Date.now(),
            title
          },
          ...prev
        ];
      });

    } catch (err) {
      setMessages(prev => [
        ...prev,
        {
          role: "assistant",
          content: `Error: ${err.message}`,
          isError: true
        }
      ]);
    } finally {
      setLoading(false);
      scrollToBottom();
    }
  };

  // CLEAR DOCUMENT
  const removeFile = () => {
    setFile(null);
    setUploaded(false);
    setMessages([]);
    setError("");
    if (fileInput.current) {
      fileInput.current.value = "";
    }
  };

  return (
    <div className="app">

      {/* SIDEBAR */}

      <aside className={`sidebar ${sidebarOpen ? "" : "closed"}`}>

        <div className="sidebar-header">
          <div className="brand">
            <div className="brand-icon">
              <BrainCircuit size={24} />
            </div>

            <div>
              <h2>MiniDocLLM</h2>
              <span>Intelligent Document AI</span>
            </div>
          </div>

          <button
            className="icon-button mobile-close"
            onClick={() => setSidebarOpen(false)}
          >
            <X size={19} />
          </button>
        </div>

        <button className="new-chat" onClick={newChat}>
          <Plus size={18} />
          New conversation
        </button>

        <div className="sidebar-section">
          <span className="section-label">WORKSPACE</span>

          <button
            className={`nav-item ${mode === "pdf" ? "active" : ""}`}
            onClick={() => setMode("pdf")}
          >
            <FileText size={18} />
            PDF Assistant
          </button>

          <button
            className={`nav-item ${mode === "llm" ? "active" : ""}`}
            onClick={() => setMode("llm")}
          >
            <Bot size={18} />
            My Own LLM
          </button>
        </div>

        <div className="sidebar-section history-section">
          <span className="section-label">RECENT QUESTIONS</span>

          {history.length === 0 ? (
            <p className="empty-history">
              Your conversations will appear here.
            </p>
          ) : (
            history.slice(0, 8).map(item => (
              <div className="history-item" key={item.id}>
                <MessageSquare size={15} />
                <span>{item.title}</span>
              </div>
            ))
          )}
        </div>

        <div className="sidebar-bottom">
          <div className="system-status">
            <span className="status-dot"></span>
            Backend connection
            <span className="status-text">API configured</span>
          </div>

          <div className="sidebar-footer">
            <Sparkles size={16} />
            <span>Built with your custom Transformer</span>
          </div>
        </div>

      </aside>

      {/* MAIN CONTENT */}

      <main className="main">

        <header className="topbar">

          <div className="top-left">
            <button
              className="icon-button"
              onClick={() => setSidebarOpen(!sidebarOpen)}
            >
              <Menu size={21} />
            </button>

            <div className="breadcrumb">
              <span>MiniDocLLM</span>
              <span>/</span>
              <strong>
                {mode === "pdf" ? "PDF Assistant" : "My Own LLM"}
              </strong>
            </div>
          </div>

          <div className="top-right">
            <div className="model-label">
              <span className="status-dot"></span>
              {mode === "pdf" ? "TF-IDF Retrieval" : "Custom Transformer"}
            </div>
          </div>

        </header>

        <div className="content">

          {/* WELCOME SCREEN */}

          {messages.length === 0 && (

            <div className="welcome">

              <div className="welcome-icon">
                {mode === "pdf" ? (
                  <BookOpen size={32} />
                ) : (
                  <BrainCircuit size={32} />
                )}
              </div>

              <div className="welcome-badge">
                <Sparkles size={14} />
                Your Personal AI Workspace
              </div>

              <h1>
                {mode === "pdf"
                  ? "Understand your documents."
                  : "Talk to your own LLM."}
                <br />
                <span>Ask anything.</span>
              </h1>

              <p className="welcome-description">
                {mode === "pdf"
                  ? "Upload a PDF and ask questions. MiniDocLLM will retrieve relevant information from your document."
                  : "Experiment with your custom PyTorch Transformer and see the text it generates."}
              </p>

              {mode === "pdf" && (

                <div className="upload-card">

                  {!file ? (

                    <div
                      className="dropzone"
                      onClick={() => fileInput.current.click()}
                      onDragOver={e => e.preventDefault()}
                      onDrop={e => {
                        e.preventDefault();
                        handleFileSelect(e.dataTransfer.files[0]);
                      }}
                    >

                      <div className="upload-icon">
                        <Upload size={25} />
                      </div>

                      <h3>Upload your PDF</h3>

                      <p>
                        Drag and drop your document here or
                        <span> browse files</span>
                      </p>

                      <small>PDF documents only</small>

                      <input
                        ref={fileInput}
                        type="file"
                        accept=".pdf,application/pdf"
                        hidden
                        onChange={e =>
                          handleFileSelect(e.target.files[0])
                        }
                      />

                    </div>

                  ) : (

                    <div className="selected-file">

                      <div className="pdf-icon">
                        <FileText size={25} />
                      </div>

                      <div className="file-info">
                        <strong>{file.name}</strong>
                        <span>
                          {(file.size / 1024 / 1024).toFixed(2)} MB
                        </span>
                      </div>

                      <button
                        className="icon-button"
                        onClick={removeFile}
                      >
                        <Trash2 size={17} />
                      </button>

                    </div>

                  )}

                  {file && !uploaded && (

                    <button
                      className="upload-button"
                      onClick={uploadPDF}
                      disabled={uploading}
                    >
                      {uploading ? (
                        <>
                          <LoaderCircle className="spin" size={17} />
                          Processing PDF...
                        </>
                      ) : (
                        <>
                          <FileUp size={17} />
                          Upload and process PDF
                        </>
                      )}
                    </button>

                  )}

                  {uploaded && (
                    <div className="success-message">
                      <CheckCircle2 size={17} />
                      PDF successfully processed
                    </div>
                  )}

                </div>

              )}

              {mode === "llm" && (

                <div className="feature-card">
                  <BrainCircuit size={25} />
                  <div>
                    <strong>Custom Transformer Model</strong>
                    <p>Powered by your trained PyTorch model and tokenizer.</p>
                  </div>
                </div>

              )}

              <div className="suggestions">

                <span className="section-label">TRY ASKING</span>

                <div className="suggestion-grid">

                  {(mode === "pdf"
                    ? [
                        "Summarize this document",
                        "What are the main topics?",
                        "Explain the important concepts",
                        "What are the key findings?"
                      ]
                    : [
                        "Hello, how are you?",
                        "Tell me about AI",
                        "What is programming?",
                        "Explain machine learning"
                      ]
                  ).map((item, index) => (

                    <button
                      key={index}
                      className="suggestion"
                      onClick={() => {
                        setQuestion(item);
                      }}
                    >
                      <MessageSquare size={15} />
                      {item}
                    </button>

                  ))}

                </div>
              </div>

            </div>

          )}

          {/* CHAT MESSAGES */}

          {messages.length > 0 && (

            <div className="chat-messages">

              {messages.map((msg, index) => (

                <div
                  key={index}
                  className={`message-row ${msg.role}`}
                >

                  <div className="message-avatar">
                    {msg.role === "assistant" ? (
                      <Bot size={19} />
                    ) : (
                      <span>U</span>
                    )}
                  </div>

                  <div className="message-body">

                    <div className="message-name">
                      {msg.role === "assistant" ? "MiniDocLLM" : "You"}
                    </div>

                    <div className={`message-text ${msg.isError ? "error-text" : ""}`}>
                      {msg.content}
                    </div>

                    {msg.sources?.length > 0 && (
                      <div className="sources">
                        <strong>
                          <BookOpen size={15} />
                          Retrieved sources
                        </strong>

                        {msg.sources.map((source, i) => (
                          <div className="source-item" key={i}>
                            <span>
                              {source.page
                                ? `Page ${source.page}`
                                : `Source ${i + 1}`}
                            </span>

                            {source.text && (
                              <p>{source.text}</p>
                            )}
                          </div>
                        ))}
                      </div>
                    )}

                  </div>

                </div>

              ))}

              {loading && (
                <div className="message-row assistant">
                  <div className="message-avatar">
                    <Bot size={19} />
                  </div>

                  <div className="message-body">
                    <div className="message-name">MiniDocLLM</div>
                    <div className="typing">
                      <LoaderCircle className="spin" size={17} />
                      Searching and generating response...
                    </div>
                  </div>
                </div>
              )}

              <div ref={chatEnd}></div>

            </div>

          )}

          {error && (
            <div className="error-banner">
              <AlertCircle size={17} />
              {error}
              <button onClick={() => setError("")}>
                <X size={15} />
              </button>
            </div>
          )}

        </div>

        {/* CHAT INPUT */}

        <div className="input-area">

          {mode === "pdf" && !uploaded && messages.length === 0 && (
            <p className="input-hint">
              <Paperclip size={14} />
              Upload a PDF to start asking questions.
            </p>
          )}

          <form className="chat-input" onSubmit={askQuestion}>

            <input
              type="text"
              placeholder={
                mode === "pdf"
                  ? "Ask something about your PDF..."
                  : "Ask your custom LLM..."
              }
              value={question}
              onChange={e => setQuestion(e.target.value)}
              disabled={loading || (mode === "pdf" && !uploaded)}
            />

            <button
              type="submit"
              disabled={
                !question.trim() ||
                loading ||
                (mode === "pdf" && !uploaded)
              }
            >
              <Send size={19} />
            </button>

          </form>

          <p className="disclaimer">
            MiniDocLLM can make mistakes. Verify important information
            against the original document.
          </p>

        </div>

      </main>

    </div>
  );
}

export default App;

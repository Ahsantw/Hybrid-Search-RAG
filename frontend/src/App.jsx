import { useEffect, useRef, useState } from 'react'
import './App.css'
import Login from './Login'
import { AuthError, streamChat } from './streamChat'

const TOKEN_KEY = 'legal-assistant-token'

const INITIAL_MESSAGE = {
  role: 'assistant',
  text: "Hello, I'm your document assistant. Ask me a question about the documents on file and I'll answer with page references.",
  sources: [],
}

function SourceList({ sources }) {
  if (!sources || sources.length === 0) return null
  return (
    <div className="sources">
      {sources.map((s, i) => (
        <span className="source-chip" key={i}>
          {s.file.split(/[\\/]/).pop()} · p.{s.page}
        </span>
      ))}
    </div>
  )
}

const VERDICT_LABELS = {
  supported: { icon: '✓', text: 'Verified against sources' },
  ambiguous: { icon: '⚠', text: 'Multiple sources may apply' },
  unsupported: { icon: '⚠', text: 'Could not verify this claim' },
  unparsed: { icon: '?', text: 'Verification inconclusive' },
}

function VerificationBadge({ verification, verifying }) {
  if (verifying) {
    return <div className="verification verifying">Verifying answer against sources...</div>
  }
  if (!verification) return null

  const label = VERDICT_LABELS[verification.verdict] || VERDICT_LABELS.unparsed
  return (
    <div className={`verification ${verification.verdict}`}>
      <span className="verification-icon">{label.icon}</span>
      <span>
        {label.text}
        {verification.note ? ` — ${verification.note}` : ''}
      </span>
    </div>
  )
}

export default function App() {
  const [token, setToken] = useState(() => sessionStorage.getItem(TOKEN_KEY))
  const [messages, setMessages] = useState([INITIAL_MESSAGE])
  const [input, setInput] = useState('')
  const [streaming, setStreaming] = useState(false)
  const bottomRef = useRef(null)

  function handleLoggedIn(newToken) {
    sessionStorage.setItem(TOKEN_KEY, newToken)
    setToken(newToken)
  }

  function handleLogout() {
    sessionStorage.removeItem(TOKEN_KEY)
    setToken(null)
    setMessages([INITIAL_MESSAGE])
  }

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  function patchLastMessage(patch) {
    setMessages((prev) => {
      const next = [...prev]
      const last = next[next.length - 1]
      next[next.length - 1] =
        typeof patch === 'function' ? patch(last) : { ...last, ...patch }
      return next
    })
  }

  async function sendQuestion() {
    const question = input.trim()
    if (!question || streaming) return

    setMessages((prev) => [
      ...prev,
      { role: 'user', text: question },
      { role: 'assistant', text: '', sources: [], streaming: true },
    ])
    setInput('')
    setStreaming(true)

    try {
      await streamChat(question, token, {
        onSources: (sources) => patchLastMessage({ sources }),
        onToken: (text) =>
          patchLastMessage((last) => ({ ...last, text: last.text + text })),
        onStatus: (stage) => {
          if (stage === 'verifying') patchLastMessage({ verifying: true })
        },
        onVerification: (verification) =>
          patchLastMessage({ verification, verifying: false }),
        onDone: () => patchLastMessage({ streaming: false }),
        onError: (detail) =>
          patchLastMessage({
            text: `Sorry, something went wrong: ${detail}`,
            error: true,
            streaming: false,
            verifying: false,
          }),
      })
    } catch (err) {
      if (err instanceof AuthError) {
        handleLogout()
        return
      }
      patchLastMessage({
        text: `Sorry, something went wrong: ${err.message}`,
        error: true,
        streaming: false,
      })
    } finally {
      setStreaming(false)
    }
  }

  function handleKeyDown(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      sendQuestion()
    }
  }

  if (!token) {
    return <Login onLoggedIn={handleLoggedIn} />
  }

  return (
    <div className="app">
      <header className="app-header">
        <div>
          <h1>Legal Document Assistant</h1>
          <p>Answers are grounded in the documents on file, with page citations.</p>
        </div>
        <button className="logout-button" onClick={handleLogout}>
          Log out
        </button>
      </header>

      <main className="chat-window">
        {messages.map((m, i) => (
          <div key={i} className={`message-row ${m.role}`}>
            <div className={`bubble ${m.role} ${m.error ? 'error' : ''}`}>
              {m.streaming && !m.text ? (
                <span className="typing-dots">
                  <span className="dot" />
                  <span className="dot" />
                  <span className="dot" />
                </span>
              ) : (
                <>
                  <p>
                    {m.text}
                    {m.streaming && !m.verifying && <span className="cursor" />}
                  </p>
                  <SourceList sources={m.sources} />
                  <VerificationBadge verification={m.verification} verifying={m.verifying} />
                </>
              )}
            </div>
          </div>
        ))}

        <div ref={bottomRef} />
      </main>

      <footer className="composer">
        <textarea
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Ask a question about the documents..."
          rows={1}
          disabled={streaming}
        />
        <button onClick={sendQuestion} disabled={streaming || !input.trim()}>
          Send
        </button>
      </footer>
    </div>
  )
}

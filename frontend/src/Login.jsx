import { useState } from 'react'

export default function Login({ onLoggedIn }) {
  const [pin, setPin] = useState('')
  const [error, setError] = useState('')
  const [submitting, setSubmitting] = useState(false)

  async function handleSubmit(e) {
    e.preventDefault()
    if (pin.length !== 4 || submitting) return

    setSubmitting(true)
    setError('')

    try {
      const res = await fetch('/api/login', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ pin }),
      })

      if (!res.ok) {
        const body = await res.json().catch(() => ({}))
        throw new Error(body.detail || `Login failed (${res.status})`)
      }

      const data = await res.json()
      onLoggedIn(data.token)
    } catch (err) {
      setError(err.message)
      setPin('')
    } finally {
      setSubmitting(false)
    }
  }

  function handlePinChange(e) {
    const digits = e.target.value.replace(/\D/g, '').slice(0, 4)
    setPin(digits)
  }

  return (
    <div className="login-screen">
      <form className="login-card" onSubmit={handleSubmit}>
        <h1>Legal Document Assistant</h1>
        <p>Enter the 4-digit PIN to continue.</p>
        <input
          type="password"
          inputMode="numeric"
          autoFocus
          value={pin}
          onChange={handlePinChange}
          maxLength={4}
          placeholder="••••"
          className="pin-input"
        />
        {error && <p className="login-error">{error}</p>}
        <button type="submit" disabled={pin.length !== 4 || submitting}>
          {submitting ? 'Checking...' : 'Enter'}
        </button>
      </form>
    </div>
  )
}

export class AuthError extends Error {}

export async function streamChat(question, token, { onSources, onToken, onDone, onError }) {
  const res = await fetch('/api/chat/stream', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Authorization: `Bearer ${token}`,
    },
    body: JSON.stringify({ question }),
  })

  if (res.status === 401) {
    throw new AuthError('Session expired. Please log in again.')
  }

  if (!res.ok || !res.body) {
    const body = await res.json().catch(() => ({}))
    throw new Error(body.detail || `Request failed (${res.status})`)
  }

  const reader = res.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''

  while (true) {
    const { value, done } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })

    let sepIndex
    while ((sepIndex = buffer.indexOf('\n\n')) !== -1) {
      const rawEvent = buffer.slice(0, sepIndex)
      buffer = buffer.slice(sepIndex + 2)
      dispatch(rawEvent, { onSources, onToken, onDone, onError })
    }
  }
}

function dispatch(rawEvent, { onSources, onToken, onDone, onError }) {
  const eventMatch = rawEvent.match(/^event: (.+)$/m)
  const dataMatch = rawEvent.match(/^data: (.+)$/m)
  if (!eventMatch || !dataMatch) return

  const event = eventMatch[1]
  const data = JSON.parse(dataMatch[1])

  if (event === 'sources') onSources?.(data.sources)
  else if (event === 'token') onToken?.(data.text)
  else if (event === 'done') onDone?.(data.response_time)
  else if (event === 'error') onError?.(data.detail)
}

import { API_BASE_URL } from '@/config'

async function request(path, { method = 'GET', body, token } = {}) {
  const headers = { Accept: 'application/json' }
  if (body !== undefined) {
    headers['Content-Type'] = 'application/json'
  }
  if (token) {
    headers.Authorization = `Bearer ${token}`
  }

  const response = await fetch(`${API_BASE_URL}${path}`, {
    method,
    headers,
    body: body !== undefined ? JSON.stringify(body) : undefined,
  })

  let data = null
  try {
    data = await response.json()
  } catch {
    data = null
  }

  if (!response.ok) {
    const detail = data?.detail
    const message =
      typeof detail === 'string'
        ? detail
        : Array.isArray(detail)
          ? detail.map((item) => item.msg || item).join(', ')
          : `Request failed (${response.status})`
    const error = new Error(message)
    error.status = response.status
    throw error
  }

  return data
}

export function login(username, password) {
  return request('/login', {
    method: 'POST',
    body: { username, password },
  })
}

export function sendChat(question, token) {
  return request('/chat', {
    method: 'POST',
    token,
    body: { question },
  })
}

export function fetchCollections(role) {
  return request(`/collections/${role}`)
}

export function fetchHealth() {
  return request('/health')
}

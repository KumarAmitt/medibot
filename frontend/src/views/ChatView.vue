<script setup>
import { nextTick, onMounted, ref } from 'vue'
import { useRouter } from 'vue-router'
import { sendChat } from '@/api/client'
import { useAuthStore } from '@/stores/auth'

const auth = useAuthStore()
const router = useRouter()
const question = ref('')
const sending = ref(false)
const thread = ref([])
const scroller = ref(null)

const suggestions = [
  'What is the emergency cashless pre-authorisation deadline?',
  'How many billing claims were escalated in 2024?',
  'What is the infection control procedure for hand hygiene?',
  'Which equipment category has the most open maintenance tickets?',
]

function roleLabel(role) {
  return (
    {
      doctor: 'Doctor',
      nurse: 'Nurse',
      billing_executive: 'Billing Executive',
      technician: 'Technician',
      admin: 'Administrator',
    }[role] || role
  )
}

function collectionLabel(name) {
  return auth.collectionLabels[name] || name
}

async function scrollToEnd() {
  await nextTick()
  if (scroller.value) {
    scroller.value.scrollTop = scroller.value.scrollHeight
  }
}

async function ask(text) {
  const trimmed = (text ?? question.value).trim()
  if (!trimmed || sending.value) {
    return
  }
  question.value = ''
  thread.value.push({
    id: crypto.randomUUID(),
    role: 'user',
    text: trimmed,
  })
  sending.value = true
  await scrollToEnd()
  try {
    const result = await sendChat(trimmed, auth.token)
    thread.value.push({
      id: crypto.randomUUID(),
      role: 'assistant',
      text: result.answer,
      sources: result.sources || [],
      retrievalType: result.retrieval_type,
      blocked: Boolean(result.blocked),
    })
  } catch (err) {
    thread.value.push({
      id: crypto.randomUUID(),
      role: 'assistant',
      text: err.message || 'The assistant could not complete that request.',
      sources: [],
      retrievalType: 'error',
      blocked: false,
      error: true,
    })
  } finally {
    sending.value = false
    await scrollToEnd()
  }
}

function logout() {
  auth.logout()
  router.push({ name: 'login' })
}

onMounted(() => {
  thread.value.push({
    id: 'welcome',
    role: 'assistant',
    text: `Welcome, ${auth.displayName}. I can search the collections assigned to a ${roleLabel(auth.role).toLowerCase()}. Ask a protocol question, or — if your role allows — an analytics question about claims and maintenance tickets.`,
    sources: [],
    retrievalType: 'system',
    blocked: false,
  })
})
</script>

<template>
  <div class="shell">
    <aside>
      <div class="brand">
        <span class="mark">MB</span>
        <div>
          <strong>MediBot</strong>
          <small>MediAssist Health Network</small>
        </div>
      </div>

      <div class="identity">
        <p class="name">{{ auth.displayName }}</p>
        <p class="handle">{{ auth.username }}</p>
        <span class="role-badge">{{ roleLabel(auth.role) }}</span>
      </div>

      <div class="panel">
        <h3>Accessible collections</h3>
        <ul>
          <li v-for="collection in auth.collections" :key="collection">
            {{ collectionLabel(collection) }}
          </li>
        </ul>
      </div>

      <p class="hint">
        Access is enforced in Qdrant before the model sees any chunk. Jailbreak prompts cannot
        surface documents outside these collections.
      </p>

      <button class="ghost" type="button" @click="logout">Sign out</button>
    </aside>

    <main>
      <header>
        <div>
          <h1>Internal knowledge assistant</h1>
          <p>Hybrid RAG with BM25 + dense search, cross-encoder reranking, and SQL analytics.</p>
        </div>
      </header>

      <section ref="scroller" class="thread">
        <article
          v-for="message in thread"
          :key="message.id"
          :class="['bubble', message.role, { blocked: message.blocked, error: message.error }]"
        >
          <p class="body">{{ message.text }}</p>
          <div v-if="message.role === 'assistant' && message.retrievalType !== 'system'" class="meta">
            <span class="pill" :class="message.retrievalType">
              {{
                message.retrievalType === 'sql_rag'
                  ? 'SQL RAG'
                  : message.retrievalType === 'hybrid_rag'
                    ? 'Hybrid RAG'
                    : 'Error'
              }}
            </span>
            <span v-if="message.blocked" class="pill blocked-pill">Access restricted</span>
          </div>
          <ul v-if="message.sources?.length" class="sources">
            <li v-for="(source, index) in message.sources" :key="index">
              <strong>{{ source.source_document }}</strong>
              <span>{{ source.section_title }}</span>
              <em>{{ collectionLabel(source.collection) }}</em>
            </li>
          </ul>
        </article>
        <p v-if="sending" class="pending">Retrieving authorised sources…</p>
      </section>

      <footer>
        <div class="suggestions">
          <button v-for="item in suggestions" :key="item" type="button" @click="ask(item)">
            {{ item }}
          </button>
        </div>
        <form @submit.prevent="ask()">
          <input
            v-model="question"
            placeholder="Ask about protocols, policies, billing, equipment, or analytics…"
            :disabled="sending"
          />
          <button type="submit" :disabled="sending || !question.trim()">Send</button>
        </form>
      </footer>
    </main>
  </div>
</template>

<style scoped>
.shell {
  min-height: 100vh;
  display: grid;
  grid-template-columns: 300px 1fr;
}

aside {
  background: var(--navy);
  color: #e8eef5;
  padding: 24px 20px;
  display: flex;
  flex-direction: column;
  gap: 20px;
}

.brand {
  display: flex;
  gap: 12px;
  align-items: center;
}

.mark {
  width: 42px;
  height: 42px;
  border-radius: 12px;
  display: grid;
  place-items: center;
  background: var(--teal);
  font-weight: 800;
}

.brand small {
  display: block;
  color: #9fb0c3;
}

.identity .name {
  margin: 0;
  font-size: 20px;
  font-weight: 700;
}

.handle {
  margin: 2px 0 10px;
  color: #9fb0c3;
  font-size: 13px;
}

.role-badge,
.pill {
  display: inline-flex;
  align-items: center;
  border-radius: 999px;
  padding: 4px 10px;
  font-size: 12px;
  font-weight: 700;
}

.role-badge {
  background: #16456d;
  color: #d7f0ef;
}

.panel {
  background: rgba(255, 255, 255, 0.05);
  border-radius: 16px;
  padding: 14px;
}

.panel h3,
.panel ul {
  margin: 0;
}

.panel h3 {
  font-size: 12px;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: #9fb0c3;
}

.panel ul {
  padding: 10px 0 0 18px;
}

.hint {
  color: #9fb0c3;
  font-size: 13px;
  margin: 0;
}

.ghost,
footer button,
form button,
.suggestions button {
  border: 0;
  border-radius: 12px;
}

.ghost {
  margin-top: auto;
  background: transparent;
  color: #e8eef5;
  border: 1px solid #33557a;
  padding: 10px 12px;
}

main {
  display: grid;
  grid-template-rows: auto 1fr auto;
  min-height: 100vh;
}

header,
footer {
  padding: 20px 28px;
}

header h1 {
  margin: 0;
  font-size: 22px;
}

header p {
  margin: 4px 0 0;
  color: var(--ink-soft);
}

.thread {
  overflow: auto;
  padding: 8px 28px 16px;
  display: grid;
  align-content: start;
  gap: 12px;
}

.bubble {
  max-width: 760px;
  background: white;
  border: 1px solid var(--line);
  border-radius: 18px;
  padding: 14px 16px;
  box-shadow: 0 8px 24px rgba(10, 37, 64, 0.04);
}

.bubble.user {
  justify-self: end;
  background: var(--teal);
  color: white;
  border-color: transparent;
}

.bubble.blocked {
  background: var(--amber-soft);
  border-color: #fdba74;
}

.bubble.error {
  background: var(--rose-soft);
}

.body {
  margin: 0;
  white-space: pre-wrap;
}

.meta {
  display: flex;
  gap: 8px;
  margin-top: 10px;
}

.pill.hybrid_rag {
  background: var(--teal-soft);
  color: var(--teal-dark);
}

.pill.sql_rag {
  background: #eee7fb;
  color: #5b21b6;
}

.pill.blocked-pill,
.pill.error {
  background: #ffedd5;
  color: var(--amber);
}

.sources {
  list-style: none;
  margin: 12px 0 0;
  padding: 0;
  display: grid;
  gap: 8px;
}

.sources li {
  display: grid;
  gap: 2px;
  background: #f8fafc;
  border-radius: 10px;
  padding: 8px 10px;
  font-size: 13px;
}

.sources span,
.sources em {
  color: var(--ink-soft);
  font-style: normal;
}

.pending {
  color: var(--ink-soft);
  font-style: italic;
}

.suggestions {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin-bottom: 10px;
}

.suggestions button {
  background: white;
  border: 1px solid var(--line);
  padding: 8px 10px;
  font-size: 12px;
  color: var(--ink-soft);
}

form {
  display: grid;
  grid-template-columns: 1fr auto;
  gap: 10px;
}

input {
  border: 1px solid var(--line);
  border-radius: 14px;
  padding: 14px 16px;
  background: white;
}

form button {
  background: var(--teal);
  color: white;
  padding: 0 18px;
  font-weight: 700;
}

form button:disabled {
  opacity: 0.6;
}

@media (max-width: 860px) {
  .shell {
    grid-template-columns: 1fr;
  }

  aside {
    min-height: auto;
  }
}
</style>

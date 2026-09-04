<script setup>
import { reactive, ref } from 'vue'
import { useRouter } from 'vue-router'
import { useAuthStore } from '@/stores/auth'

const router = useRouter()
const auth = useAuthStore()

const form = reactive({
  username: '',
  password: '',
})
const error = ref('')
const loading = ref(false)

const demos = [
  { username: 'dr.mehta', password: 'doctor', role: 'Doctor', hint: 'Clinical + nursing + general' },
  { username: 'nurse.priya', password: 'nurse', role: 'Nurse', hint: 'Nursing + general' },
  {
    username: 'billing.ravi',
    password: 'billing_executive',
    role: 'Billing',
    hint: 'Billing + general + SQL',
  },
  { username: 'tech.anand', password: 'technician', role: 'Technician', hint: 'Equipment + general' },
  { username: 'admin.sys', password: 'admin', role: 'Admin', hint: 'All collections + SQL' },
]

function fillDemo(account) {
  form.username = account.username
  form.password = account.password
  error.value = ''
}

async function onSubmit() {
  error.value = ''
  loading.value = true
  try {
    await auth.login(form.username, form.password)
    await router.push({ name: 'chat' })
  } catch (err) {
    error.value = err.message || 'Unable to sign in'
  } finally {
    loading.value = false
  }
}
</script>

<template>
  <main class="login">
    <section class="hero">
      <p class="eyebrow">MediAssist Health Network</p>
      <h1>MediBot</h1>
      <p class="lede">
        Ask natural-language questions across clinical protocols, nursing procedures, billing
        guides, and equipment manuals. Every answer is scoped to your role at retrieval time.
      </p>
    </section>

    <section class="card">
      <h2>Staff sign in</h2>
      <form @submit.prevent="onSubmit">
        <label>
          Username
          <input v-model="form.username" autocomplete="username" required />
        </label>
        <label>
          Password
          <input v-model="form.password" type="password" autocomplete="current-password" required />
        </label>
        <p v-if="error" class="error">{{ error }}</p>
        <button type="submit" :disabled="loading">
          {{ loading ? 'Signing in…' : 'Enter workspace' }}
        </button>
      </form>

      <p class="demo-label">Demo accounts — click to fill</p>
      <div class="demos">
        <button v-for="account in demos" :key="account.username" type="button" @click="fillDemo(account)">
          <strong>{{ account.role }}</strong>
          <span>{{ account.username }}</span>
          <em>{{ account.hint }}</em>
        </button>
      </div>
    </section>
  </main>
</template>

<style scoped>
.login {
  min-height: 100vh;
  display: grid;
  grid-template-columns: 1.1fr 0.9fr;
  gap: 48px;
  align-items: center;
  max-width: 1120px;
  margin: 0 auto;
  padding: 48px 24px;
}

.hero h1 {
  margin: 8px 0 16px;
  font-size: clamp(40px, 6vw, 72px);
  letter-spacing: -0.04em;
  color: var(--navy);
}

.eyebrow {
  margin: 0;
  text-transform: uppercase;
  letter-spacing: 0.16em;
  font-size: 12px;
  color: var(--teal);
  font-weight: 700;
}

.lede {
  max-width: 46ch;
  color: var(--ink-soft);
  font-size: 18px;
}

.card {
  background: var(--bg-elevated);
  border: 1px solid var(--line);
  border-radius: 24px;
  padding: 28px;
  box-shadow: var(--shadow);
}

.card h2 {
  margin: 0 0 18px;
}

form {
  display: grid;
  gap: 12px;
}

label {
  display: grid;
  gap: 6px;
  font-size: 13px;
  font-weight: 600;
}

input {
  border: 1px solid var(--line);
  border-radius: 12px;
  padding: 12px 14px;
  background: #fbfdfe;
}

button[type='submit'] {
  margin-top: 6px;
  border: 0;
  border-radius: 12px;
  padding: 12px 16px;
  background: var(--teal);
  color: white;
  font-weight: 700;
}

button[type='submit']:disabled {
  opacity: 0.7;
}

.error {
  margin: 0;
  color: var(--rose);
  background: var(--rose-soft);
  border-radius: 10px;
  padding: 8px 10px;
  font-size: 14px;
}

.demo-label {
  margin: 22px 0 10px;
  font-size: 12px;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: var(--ink-soft);
}

.demos {
  display: grid;
  gap: 8px;
}

.demos button {
  text-align: left;
  border: 1px solid var(--line);
  background: var(--teal-soft);
  border-radius: 12px;
  padding: 10px 12px;
  display: grid;
}

.demos span,
.demos em {
  font-size: 12px;
  color: var(--ink-soft);
  font-style: normal;
}

@media (max-width: 860px) {
  .login {
    grid-template-columns: 1fr;
    padding-top: 28px;
  }
}
</style>

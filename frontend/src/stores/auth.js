import { computed, ref } from 'vue'
import { defineStore } from 'pinia'
import { fetchCollections, login as loginRequest } from '@/api/client'

const STORAGE_KEY = 'medibot.session'

export const useAuthStore = defineStore('auth', () => {
  const token = ref('')
  const username = ref('')
  const role = ref('')
  const displayName = ref('')
  const collections = ref([])
  const collectionLabels = ref({})

  const isAuthenticated = computed(() => Boolean(token.value && role.value))

  function persist() {
    sessionStorage.setItem(
      STORAGE_KEY,
      JSON.stringify({
        token: token.value,
        username: username.value,
        role: role.value,
        displayName: displayName.value,
        collections: collections.value,
        collectionLabels: collectionLabels.value,
      }),
    )
  }

  function hydrate() {
    const raw = sessionStorage.getItem(STORAGE_KEY)
    if (!raw) {
      return
    }
    try {
      const saved = JSON.parse(raw)
      token.value = saved.token || ''
      username.value = saved.username || ''
      role.value = saved.role || ''
      displayName.value = saved.displayName || ''
      collections.value = saved.collections || []
      collectionLabels.value = saved.collectionLabels || {}
    } catch {
      sessionStorage.removeItem(STORAGE_KEY)
    }
  }

  async function login(usernameInput, password) {
    const session = await loginRequest(usernameInput, password)
    token.value = session.access_token
    username.value = session.username
    role.value = session.role
    displayName.value = session.display_name
    const access = await fetchCollections(session.role)
    collections.value = access.collections
    collectionLabels.value = access.collection_labels
    persist()
  }

  function logout() {
    token.value = ''
    username.value = ''
    role.value = ''
    displayName.value = ''
    collections.value = []
    collectionLabels.value = {}
    sessionStorage.removeItem(STORAGE_KEY)
  }

  hydrate()

  return {
    token,
    username,
    role,
    displayName,
    collections,
    collectionLabels,
    isAuthenticated,
    login,
    logout,
    hydrate,
  }
})

<template>
  <router-view />
</template>

<script setup>
// Use Vue Router for page management
</script>

<style>
/* Global style reset */
* {
  margin: 0;
  padding: 0;
  box-sizing: border-box;
}

#app {
  font-family: 'Inter', system-ui, -apple-system, sans-serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  color: #0A1B2A;
  background-color: #EBEDF0;
}

/* Scrollbar styles */
::-webkit-scrollbar {
  width: 10px;
  height: 10px;
}

::-webkit-scrollbar-track {
  background: #E4E7EC;
}

::-webkit-scrollbar-thumb {
  background: #56697D;
}

::-webkit-scrollbar-thumb:hover {
  background: #1E3247;
}

/* Global button styles */
button {
  font-family: inherit;
}

/* Global focus visible styles */
*:focus-visible {
  outline: 2px solid #C4751A;
  outline-offset: 2px;
}

/* Respect reduced-motion */
@media (prefers-reduced-motion: reduce) {
  *,
  *::before,
  *::after {
    animation-duration: 0.01ms !important;
    animation-iteration-count: 1 !important;
    transition-duration: 0.01ms !important;
  }
}
</style>
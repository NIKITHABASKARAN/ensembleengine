/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        'atos-bg': '#2B2B2B',
        'atos-bg-secondary': '#1E1E1E',
        'atos-card': '#363636',
        'atos-primary': '#0073E6',
        'atos-primary-hover': '#005BB5',
        'atos-border': '#404040',
        'atos-success': '#10B981',
        'atos-warning': '#F59E0B',
        'atos-danger': '#EF4444',
      },
      fontFamily: {
        'sans': ['Inter', 'system-ui', '-apple-system', 'sans-serif'],
        'mono': ['JetBrains Mono', 'monospace'],
      },
    },
  },
  plugins: [],
}
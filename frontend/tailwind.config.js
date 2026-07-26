/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  theme: {
    extend: {
      colors: {
        ink: '#17202a',
        navy: '#12263f',
        accent: '#2563eb',
        canvas: '#f5f7fa',
        line: '#dce2e9',
      },
      boxShadow: {
        panel: '0 1px 2px rgba(16, 24, 40, 0.05)',
      },
    },
  },
  plugins: [],
}

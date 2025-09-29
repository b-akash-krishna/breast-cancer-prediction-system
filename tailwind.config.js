/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  theme: {
    extend: {
      colors: {
        // Medical-themed color palette
        medical: {
          primary: {
            50: '#e6f2ff',
            100: '#cce5ff',
            200: '#99cbff',
            300: '#66b0ff',
            400: '#3396ff',
            500: '#0077cc',  // Primary blue
            600: '#0066b3',
            700: '#005599',
            800: '#004480',
            900: '#003366',
          },
          success: {
            50: '#e8f5e9',
            100: '#c8e6c9',
            200: '#a5d6a7',
            300: '#81c784',
            400: '#66bb6a',
            500: '#4caf50',  // Success green
            600: '#43a047',
            700: '#388e3c',
            800: '#2e7d32',
            900: '#1b5e20',
          },
          danger: {
            50: '#ffebee',
            100: '#ffcdd2',
            200: '#ef9a9a',
            300: '#e57373',
            400: '#ef5350',
            500: '#f44336',  // Danger red
            600: '#e53935',
            700: '#d32f2f',
            800: '#c62828',
            900: '#b71c1c',
          },
          warning: {
            50: '#fff8e1',
            100: '#ffecb3',
            200: '#ffe082',
            300: '#ffd54f',
            400: '#ffca28',
            500: '#ffc107',  // Warning amber
            600: '#ffb300',
            700: '#ffa000',
            800: '#ff8f00',
            900: '#ff6f00',
          },
          info: {
            50: '#e3f2fd',
            100: '#bbdefb',
            200: '#90caf9',
            300: '#64b5f6',
            400: '#42a5f5',
            500: '#2196f3',  // Info blue
            600: '#1e88e5',
            700: '#1976d2',
            800: '#1565c0',
            900: '#0d47a1',
          },
        },
        // Neutral grays for medical UI
        gray: {
          50: '#fafafa',
          100: '#f5f5f5',
          200: '#eeeeee',
          300: '#e0e0e0',
          400: '#bdbdbd',
          500: '#9e9e9e',
          600: '#757575',
          700: '#616161',
          800: '#424242',
          900: '#212121',
        },
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', '-apple-system', 'sans-serif'],
        medical: ['Roboto', 'system-ui', 'sans-serif'],
      },
      boxShadow: {
        'medical-sm': '0 1px 3px 0 rgba(0, 119, 204, 0.1)',
        'medical': '0 4px 6px -1px rgba(0, 119, 204, 0.1), 0 2px 4px -1px rgba(0, 119, 204, 0.06)',
        'medical-lg': '0 10px 15px -3px rgba(0, 119, 204, 0.1), 0 4px 6px -2px rgba(0, 119, 204, 0.05)',
        'medical-xl': '0 20px 25px -5px rgba(0, 119, 204, 0.1), 0 10px 10px -5px rgba(0, 119, 204, 0.04)',
      },
      borderRadius: {
        'medical': '0.75rem',
      },
    },
  },
  plugins: [],
};
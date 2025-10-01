import tailwindcss from '@tailwindcss/vite';
import { sveltekit } from '@sveltejs/kit/vite';
import { defineConfig } from 'vite';

export default defineConfig({
    plugins: [sveltekit(), tailwindcss()],
    server: {
        // Proxy configuration for development
        proxy: {
            // All requests starting with /api (e.g., /api/jobs) 
            // will be forwarded to the FastAPI server at http://127.0.0.1:8000
            '/api': {
                target: 'http://127.0.0.1:8000',
                changeOrigin: true,
                secure: false,
                // rewrite: (path) => path.replace(/^\/api/, '') // Not needed since the API path starts with /api in FastAPI
            },
        },
    },
});

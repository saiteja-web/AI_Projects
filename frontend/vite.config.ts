import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    host: true, // listen on all interfaces (needed inside Docker)
    port: 5173,
    strictPort: true,
  },
});

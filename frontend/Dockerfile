FROM node:20-alpine

WORKDIR /app

# Copy manifest first so node_modules is cached unless package.json changes
COPY package.json ./

# Install dependencies into the container
RUN npm install

# --host 0.0.0.0 makes Vite reachable from the host browser
CMD ["npm", "run", "dev", "--", "--host", "0.0.0.0"]

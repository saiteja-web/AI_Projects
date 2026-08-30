import { useState } from 'react';
import { HomePage } from './components/HomePage';
import { ChatPage } from './components/ChatPage';

function App() {
  const [route] = useState<{ page: string; sessionId?: string }>({
    page: window.location.pathname.startsWith("/chat/") ? "chat" : "home",
    sessionId: window.location.pathname.split("/chat/")[1],
  });

  return (
    <div style={{ fontFamily: "system-ui, sans-serif" }}>
      {route.page === "chat" && route.sessionId ? (
        <ChatPage sessionId={route.sessionId} />
      ) : (
        <HomePage />
      )}
    </div>
  );
}

export default App;

import { useState, useEffect, useRef } from 'react';
import { MessageList } from './MessageList';
import { getChatHistory, sendChatMessage } from '../services/api';
import type { Message } from '../types';

interface ChatPageProps {
    sessionId: string;
}

export function ChatPage({ sessionId }: ChatPageProps) {
    const [messages, setMessages] = useState<Message[]>([]);
    const [input, setInput] = useState('');
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const messagesEndRef = useRef<HTMLDivElement>(null);

    // Load chat history on mount
    useEffect(() => {
        const loadHistory = async () => {
            try {
                const history = await getChatHistory(sessionId);
                setMessages(history.messages);
            } catch (err) {
                setError(err instanceof Error ? err.message : 'Failed to load history');
            }
        };
        loadHistory();
    }, [sessionId]);

    // Auto-scroll to bottom when messages change
    useEffect(() => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, [messages]);

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        if (!input.trim() || loading) return;

        setLoading(true);
        setError(null);

        // Optimistically add user message
        const userMessage: Message = {
            id: crypto.randomUUID(),
            role: 'user',
            content: input,
            sources: [],
            created_at: new Date().toISOString(),
        };
        setMessages((prev) => [...prev, userMessage]);
        const query = input;
        setInput('');

        try {
            const response = await sendChatMessage(sessionId, query);
            const aiMessage: Message = {
                id: crypto.randomUUID(),
                role: 'ai',
                content: response.answer,
                sources: response.sources,
                created_at: new Date().toISOString(),
            };
            setMessages((prev) => [...prev, aiMessage]);
        } catch (err) {
            setError(err instanceof Error ? err.message : 'Failed to send message');
            // Remove the optimistic user message on error
            setMessages((prev) => prev.slice(0, -1));
            setInput(query);
        } finally {
            setLoading(false);
        }
    };

    return (
        <div style={{ display: 'flex', flexDirection: 'column', height: '100vh' }}>
            <header
                style={{
                    padding: '1rem 2rem',
                    borderBottom: '1px solid #e5e7eb',
                    backgroundColor: '#f9fafb',
                }}
            >
                <h2 style={{ margin: 0 }}>Document Chat</h2>
                <p style={{ margin: '0.5rem 0 0 0', fontSize: '0.875rem', color: '#6b7280' }}>
                    Session ID: <code>{sessionId}</code>
                </p>
            </header>

            <MessageList messages={messages} />

            <div ref={messagesEndRef} />

            <form
                onSubmit={handleSubmit}
                style={{
                    padding: '1rem 2rem',
                    borderTop: '1px solid #e5e7eb',
                    backgroundColor: '#f9fafb',
                }}
            >
                <div style={{ display: 'flex', gap: '1rem' }}>
                    <input
                        type="text"
                        value={input}
                        onChange={(e) => setInput(e.target.value)}
                        placeholder="Ask a question about your document..."
                        disabled={loading}
                        style={{
                            flex: 1,
                            padding: '10px 14px',
                            borderRadius: '6px',
                            border: '1px solid #d1d5db',
                            fontSize: '1rem',
                        }}
                    />
                    <button
                        type="submit"
                        disabled={loading || !input.trim()}
                        style={{
                            padding: '10px 20px',
                            backgroundColor: loading || !input.trim() ? '#9ca3af' : '#3b82f6',
                            color: 'white',
                            border: 'none',
                            borderRadius: '6px',
                            cursor: loading || !input.trim() ? 'not-allowed' : 'pointer',
                            fontSize: '1rem',
                        }}
                    >
                        {loading ? 'Sending...' : 'Send'}
                    </button>
                </div>
                {error && (
                    <p style={{ color: '#ef4444', marginTop: '0.5rem', margin: 0 }}>{error}</p>
                )}
            </form>
        </div>
    );
}

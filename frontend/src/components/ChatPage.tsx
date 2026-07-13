import { useState, useEffect, useRef } from 'react';
import { MessageList } from './MessageList';
import { getChatHistory, sendChatMessage, fetchModels } from '../services/api';
import type { Message, ModelsResponse } from '../types';

interface ChatPageProps {
    sessionId: string;
}

export function ChatPage({ sessionId }: ChatPageProps) {
    const [messages, setMessages] = useState<Message[]>([]);
    const [input, setInput] = useState('');
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [models, setModels] = useState<ModelsResponse | null>(null);
    const [selectedProvider, setSelectedProvider] = useState<string>('ollama');
    const [selectedModel, setSelectedModel] = useState<string>('llama3.2');
    const messagesEndRef = useRef<HTMLDivElement>(null);

    // Load available models on mount
    useEffect(() => {
        const loadModels = async () => {
            try {
                const modelsData = await fetchModels();
                setModels(modelsData);
                setSelectedProvider(modelsData.default_provider);
                setSelectedModel(modelsData.default_model);
            } catch (err) {
                console.error('Failed to load models:', err);
            }
        };
        loadModels();
    }, []);

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

    // Get available models for the selected provider
    const availableModels = models?.providers[selectedProvider as keyof typeof models.providers]?.models || [];

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
            const response = await sendChatMessage(
                sessionId,
                query,
                selectedProvider,
                selectedModel
            );
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
                {/* Model Selection */}
                {models && (
                    <div style={{ display: 'flex', gap: '1rem', marginBottom: '1rem' }}>
                        <div style={{ flex: 1 }}>
                            <label style={{ fontSize: '0.875rem', color: '#6b7280', marginBottom: '0.25rem', display: 'block' }}>
                                Provider
                            </label>
                            <select
                                value={selectedProvider}
                                onChange={(e) => {
                                    setSelectedProvider(e.target.value);
                                    // Auto-select first model of the new provider
                                    const firstModel = models.providers[e.target.value as keyof typeof models.providers]?.models[0];
                                    if (firstModel) setSelectedModel(firstModel.id);
                                }}
                                disabled={loading}
                                style={{
                                    width: '100%',
                                    padding: '8px 12px',
                                    borderRadius: '6px',
                                    border: '1px solid #d1d5db',
                                    fontSize: '0.875rem',
                                    backgroundColor: 'white',
                                }}
                            >
                                <option value="ollama">Ollama (Local)</option>
                                <option value="openai">OpenAI</option>
                            </select>
                        </div>
                        <div style={{ flex: 2 }}>
                            <label style={{ fontSize: '0.875rem', color: '#6b7280', marginBottom: '0.25rem', display: 'block' }}>
                                Model
                            </label>
                            <select
                                value={selectedModel}
                                onChange={(e) => setSelectedModel(e.target.value)}
                                disabled={loading}
                                style={{
                                    width: '100%',
                                    padding: '8px 12px',
                                    borderRadius: '6px',
                                    border: '1px solid #d1d5db',
                                    fontSize: '0.875rem',
                                    backgroundColor: 'white',
                                }}
                            >
                                {availableModels.map((model) => (
                                    <option key={model.id} value={model.id}>
                                        {model.name} {model.description && `- ${model.description}`}
                                    </option>
                                ))}
                            </select>
                        </div>
                    </div>
                )}

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

import { CitationBadge } from './CitationBadge';
import type { Message } from '../types';

interface MessageListProps {
    messages: Message[];
}

export function MessageList({ messages }: MessageListProps) {
    return (
        <div
            style={{
                display: 'flex',
                flexDirection: 'column',
                gap: '12px',
                padding: '16px',
                overflowY: 'auto',
                maxHeight: 'calc(100vh - 200px)',
            }}
        >
            {messages.length === 0 ? (
                <p style={{ color: '#666', textAlign: 'center' }}>
                    No messages yet. Start by asking a question!
                </p>
            ) : (
                messages.map((message) => (
                    <div
                        key={message.id}
                        style={{
                            display: 'flex',
                            flexDirection: 'column',
                            alignItems: message.role === 'user' ? 'flex-end' : 'flex-start',
                        }}
                    >
                        <div
                            style={{
                                maxWidth: '70%',
                                padding: '10px 14px',
                                borderRadius: '12px',
                                backgroundColor: message.role === 'user' ? '#3b82f6' : '#f3f4f6',
                                color: message.role === 'user' ? '#ffffff' : '#1f2937',
                            }}
                        >
                            <p style={{ margin: 0, whiteSpace: 'pre-wrap' }}>
                                {message.content}
                            </p>
                        </div>

                        {/* Show citations for AI messages */}
                        {message.role === 'ai' && message.sources.length > 0 && (
                            <div style={{ marginTop: '4px', display: 'flex', gap: '4px' }}>
                                {message.sources.map((page) => (
                                    <CitationBadge key={page} pageNumber={page} />
                                ))}
                            </div>
                        )}
                    </div>
                ))
            )}
        </div>
    );
}
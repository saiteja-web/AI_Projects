// this represents a single message in the chat history
interface Message{
    id:string;
    role:'user'|'ai';
    content:string;
    sources:number[];
    created_at:string;
}

// the response from GET /api/history/{session_id}
interface ChatHistory {
    session_id: string;
    messages: Message[];
}
// the response from POST /api/upload
interface UploadResponse{
    session_id:string;
    filename:string;
    status: string;
}
// The response from POST /api/chat
interface ChatResponse {
    answer: string;
    sources: number[];
    session_id: string;
}

// The request body for POST /api/chat
interface ChatRequest {
    session_id: string;
    query: string;
}

export type { Message, ChatHistory, ChatResponse, UploadResponse, ChatRequest };
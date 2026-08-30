// this represents a single source citation attached to an AI message
interface SourceRef {
    page: number;
    section: string;
}

// this represents a single message in the chat history
interface Message{
    id:string;
    role:'user'|'ai';
    content:string;
    sources:SourceRef[];
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
    sources: SourceRef[];
    session_id: string;
}

// The request body for POST /api/chat
interface ChatRequest {
    session_id: string;
    query: string;
    model_id?: string;
}

// LLM Model info
interface ModelInfo {
    id: string;
    name: string;
    context_tokens?: number;
    description: string;
}

// Response from GET /models/
interface ModelsResponse {
    models: ModelInfo[];
    default_model: string;
}

export type { Message, ChatHistory, ChatResponse, UploadResponse, ChatRequest, ModelInfo, ModelsResponse, SourceRef };

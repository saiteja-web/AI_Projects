import axios from 'axios';
import type { UploadResponse,ChatHistory,ChatRequest,ChatResponse, ModelsResponse } from '../types';


// Base URL for backend API
const API_URL = 'http://localhost:8001';

// Create an axios instance with base URL
const api=axios.create({
    baseURL:API_URL,
});

// Upload a PDF file
export async function uploadPDF(file:File): Promise<UploadResponse>{
    const formData=new FormData();
    formData.append('file',file)
    const response = await api.post<UploadResponse>('/api/upload',formData,{
        headers:{
            'Content-Type':'multipart/form-data',
        },
    });

    return response.data;

}
// send a chat message
export async function sendChatMessage(
    sessionId: string,
    query: string,
    provider?: string,
    modelId?: string
): Promise<ChatResponse> {
    const response = await api.post<ChatResponse>('/api/chat', {
        session_id: sessionId,
        query,
        provider,
        model_id: modelId,
    });
    return response.data;
}
// Get chat history for a session
export async function getChatHistory(sessionId: string): Promise<ChatHistory> {
    const response = await api.get<ChatHistory>(`/api/history/${sessionId}`);
    return response.data;
}

// Get available LLM models
export async function fetchModels(): Promise<ModelsResponse> {
    const response = await api.get<ModelsResponse>('/models/');
    return response.data;
}

// Health check
export async function healthCheck(): Promise<{ status: string }> {
    const response = await api.get('/health');
    return response.data;
}
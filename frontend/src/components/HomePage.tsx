import { useState } from 'react';
import { uploadPDF } from '../services/api';
import type { UploadResponse } from '../types';

export function HomePage() {
    const [file, setFile] = useState<File | null>(null);
    const [uploadResult, setUploadResult] = useState<UploadResponse | null>(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);

    const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        const selectedFile = e.target.files?.[0];
        if (selectedFile) {
            // Validate it's a PDF
            if (!selectedFile.name.toLowerCase().endsWith('.pdf')) {
                setError('Only PDF files are allowed');
                setFile(null);
                return;
            }
            setFile(selectedFile);
            setError(null);
            setUploadResult(null);
        }
    };

    const handleUpload = async () => {
        if (!file) return;

        setLoading(true);
        setError(null);

        try {
            const result = await uploadPDF(file);
            setUploadResult(result);
        } catch (err) {
            setError(err instanceof Error ? err.message : 'Upload failed');
        } finally {
            setLoading(false);
        }
    };

    return (
        <div style={{ padding: '2rem', maxWidth: '600px', margin: '0 auto' }}>
            <h1>AI Document Chat</h1>
            <p>Upload a PDF document to start chatting with it!</p>

            <div
                style={{
                    border: '2px dashed #cbd5e1',
                    borderRadius: '8px',
                    padding: '2rem',
                    textAlign: 'center',
                    marginTop: '2rem',
                }}
            >
                <input
                    type="file"
                    accept=".pdf"
                    onChange={handleFileChange}
                    style={{ display: 'none' }}
                    id="file-upload"
                />
                <label
                    htmlFor="file-upload"
                    style={{
                        display: 'inline-block',
                        padding: '10px 20px',
                        backgroundColor: '#3b82f6',
                        color: 'white',
                        borderRadius: '6px',
                        cursor: 'pointer',
                    }}
                >
                    Select PDF File
                </label>

                {file && (
                    <p style={{ marginTop: '1rem' }}>
                        Selected: <strong>{file.name}</strong>
                    </p>
                )}
            </div>

            {file && (
                <button
                    onClick={handleUpload}
                    disabled={loading}
                    style={{
                        marginTop: '1rem',
                        padding: '10px 20px',
                        backgroundColor: loading ? '#9ca3af' : '#10b981',
                        color: 'white',
                        border: 'none',
                        borderRadius: '6px',
                        cursor: loading ? 'not-allowed' : 'pointer',
                    }}
                >
                    {loading ? 'Uploading...' : 'Upload PDF'}
                </button>
            )}

            {error && <p style={{ color: '#ef4444', marginTop: '1rem' }}>{error}</p>}

            {uploadResult && (
                <div
                    style={{
                        marginTop: '2rem',
                        padding: '1rem',
                        backgroundColor: '#ecfdf5',
                        borderRadius: '8px',
                        border: '1px solid #10b981',
                    }}
                >
                    <h3 style={{ marginTop: 0 }}>Upload Successful!</h3>
                    <p>
                        Session ID: <code>{uploadResult.session_id}</code>
                    </p>
                    <p>
                        Filename: <strong>{uploadResult.filename}</strong>
                    </p>
                    <a
                        href={`/chat/${uploadResult.session_id}`}
                        style={{
                            display: 'inline-block',
                            marginTop: '1rem',
                            padding: '10px 20px',
                            backgroundColor: '#3b82f6',
                            color: 'white',
                            textDecoration: 'none',
                            borderRadius: '6px',
                        }}
                    >
                        Start Chat →
                    </a>
                </div>
            )}
        </div>
    );
}

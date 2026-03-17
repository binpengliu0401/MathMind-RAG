import { createDemoTransport } from './rag-demo-client';
import { createWebSocketTransport } from './rag-socket-client';
import type { RAGTransport } from '../types/rag';

export function createRAGTransport(): RAGTransport {
  const wsUrl = import.meta.env.VITE_RAG_WS_URL;

  if (typeof wsUrl === 'string' && wsUrl.trim().length > 0) {
    return createWebSocketTransport({ url: wsUrl });
  }

  return createDemoTransport();
}

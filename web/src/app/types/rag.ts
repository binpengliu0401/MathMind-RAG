export type ProcessingStep = 'rewriting' | 'retrieval' | 'generation' | 'grading' | 'complete';

export type RunStatus = 'idle' | 'running' | 'complete' | 'error';

export type ConnectionStatus = 'connecting' | 'connected' | 'demo' | 'disconnected' | 'error';

export type TransportMode = 'websocket' | 'demo';

export interface RetrievedDoc {
  id: string;
  source: string;
  snippet: string;
  relevant: boolean;
}

export interface HallucinationResult {
  score: number;
  explanation: string;
  unsupportedClaims?: string[];
}

export interface SessionSnapshot {
  runId?: string;
  query: string;
  runStatus: RunStatus;
  currentStep?: ProcessingStep;
  rewrittenQuery: string;
  retrievedDocs: RetrievedDoc[];
  answer: string;
  hallucinationResult?: HallucinationResult;
  error?: string;
}

export interface SessionState {
  connectionStatus: ConnectionStatus;
  transportMode: TransportMode;
  snapshot: SessionSnapshot;
}

export type RAGServerEvent =
  | { type: 'run_started'; runId?: string; query: string }
  | { type: 'step_changed'; step: ProcessingStep }
  | { type: 'query_rewritten'; rewrittenQuery: string }
  | { type: 'documents_retrieved'; retrievedDocs: RetrievedDoc[] }
  | { type: 'answer_delta'; delta: string }
  | { type: 'answer_replaced'; answer: string }
  | { type: 'grading_completed'; hallucinationResult: HallucinationResult }
  | { type: 'run_completed' }
  | { type: 'run_failed'; error: string }
  | { type: 'snapshot'; snapshot: Partial<SessionSnapshot> };

export interface TransportCallbacks {
  onConnectionStatusChange: (status: ConnectionStatus) => void;
  onEvent: (event: RAGServerEvent) => void;
  onTransportError: (message: string) => void;
}

export interface RAGTransport {
  mode: TransportMode;
  connect: (callbacks: TransportCallbacks) => void;
  disconnect: () => void;
  submitQuery: (query: string) => void;
}

export const createEmptySnapshot = (): SessionSnapshot => ({
  query: '',
  runStatus: 'idle',
  currentStep: undefined,
  rewrittenQuery: '',
  retrievedDocs: [],
  answer: '',
  hallucinationResult: undefined,
  error: undefined,
});

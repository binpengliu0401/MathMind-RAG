import { useEffect, useMemo, useReducer, useRef } from 'react';
import { createRAGTransport } from '../lib/create-rag-transport';
import {
  createEmptySnapshot,
  type ConnectionStatus,
  type RAGServerEvent,
  type RAGTransport,
  type SessionSnapshot,
  type SessionState,
} from '../types/rag';

type SessionAction =
  | { type: 'connection_status_changed'; status: ConnectionStatus }
  | { type: 'transport_error'; message: string }
  | { type: 'submit_requested'; query: string }
  | { type: 'server_event_received'; event: RAGServerEvent };

const initialState: SessionState = {
  connectionStatus: 'connecting',
  transportMode: 'demo',
  snapshot: createEmptySnapshot(),
};

function applyServerEvent(snapshot: SessionSnapshot, event: RAGServerEvent): SessionSnapshot {
  switch (event.type) {
    case 'run_started':
      return {
        ...createEmptySnapshot(),
        runId: event.runId,
        query: event.query,
        runStatus: 'running',
        currentStep: 'rewriting',
      };
    case 'step_changed':
      return {
        ...snapshot,
        currentStep: event.step,
        runStatus: event.step === 'complete' ? 'complete' : 'running',
      };
    case 'query_rewritten':
      return {
        ...snapshot,
        rewrittenQuery: event.rewrittenQuery,
      };
    case 'documents_retrieved':
      return {
        ...snapshot,
        retrievedDocs: event.retrievedDocs,
      };
    case 'answer_delta':
      return {
        ...snapshot,
        answer: `${snapshot.answer}${event.delta}`,
      };
    case 'answer_replaced':
      return {
        ...snapshot,
        answer: event.answer,
      };
    case 'grading_completed':
      return {
        ...snapshot,
        hallucinationResult: event.hallucinationResult,
      };
    case 'run_completed':
      return {
        ...snapshot,
        runStatus: 'complete',
        currentStep: 'complete',
      };
    case 'run_failed':
      return {
        ...snapshot,
        runStatus: 'error',
        error: event.error,
      };
    case 'snapshot':
      return {
        ...snapshot,
        ...event.snapshot,
      };
    default:
      return snapshot;
  }
}

function sessionReducer(state: SessionState, action: SessionAction): SessionState {
  switch (action.type) {
    case 'connection_status_changed':
      return {
        ...state,
        connectionStatus: action.status,
      };
    case 'transport_error':
      return {
        ...state,
        snapshot: {
          ...state.snapshot,
          runStatus: state.snapshot.runStatus === 'idle' ? 'idle' : 'error',
          error: action.message,
        },
      };
    case 'submit_requested':
      return {
        ...state,
        snapshot: {
          ...createEmptySnapshot(),
          query: action.query,
          runStatus: 'running',
          currentStep: 'rewriting',
        },
      };
    case 'server_event_received':
      return {
        ...state,
        snapshot: applyServerEvent(state.snapshot, action.event),
      };
    default:
      return state;
  }
}

export function useRAGSession() {
  const [state, dispatch] = useReducer(sessionReducer, initialState);
  const transportRef = useRef<RAGTransport | null>(null);
  const transport = useMemo(() => createRAGTransport(), []);

  useEffect(() => {
    transportRef.current = transport;

    dispatch({
      type: 'connection_status_changed',
      status: transport.mode === 'demo' ? 'demo' : 'connecting',
    });

    transport.connect({
      onConnectionStatusChange: (status) => {
        dispatch({ type: 'connection_status_changed', status });
      },
      onEvent: (event) => {
        dispatch({ type: 'server_event_received', event });
      },
      onTransportError: (message) => {
        dispatch({ type: 'transport_error', message });
      },
    });

    return () => {
      transport.disconnect();
      transportRef.current = null;
    };
  }, [transport]);

  const submitQuery = (query: string) => {
    const trimmedQuery = query.trim();
    if (!trimmedQuery) {
      return;
    }

    dispatch({ type: 'submit_requested', query: trimmedQuery });
    transportRef.current?.submitQuery(trimmedQuery);
  };

  return {
    connectionStatus: state.connectionStatus,
    transportMode: transport.mode,
    snapshot: state.snapshot,
    isRunning: state.snapshot.runStatus === 'running',
    submitQuery,
  };
}

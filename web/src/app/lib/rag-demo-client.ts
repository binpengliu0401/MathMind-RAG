import type { RAGServerEvent, RAGTransport, TransportCallbacks } from '../types/rag';

const DEMO_ANSWER = 'Retrieval-Augmented Generation (RAG) improves factual accuracy by grounding answers in retrieved sources. It also exposes source context to users and makes knowledge updates possible without retraining the model.';

const DEMO_DOCS = [
  {
    id: '1',
    source: 'Doc 1, Page 3',
    snippet: 'Retrieval-Augmented Generation (RAG) combines large language models with external knowledge retrieval to improve accuracy and reduce hallucinations.',
    relevant: true,
  },
  {
    id: '2',
    source: 'Doc 2, Page 7',
    snippet: 'RAG systems retrieve relevant documents from a knowledge base before generating responses, grounding the output in factual information.',
    relevant: true,
  },
  {
    id: '3',
    source: 'Doc 3, Page 12',
    snippet: 'The main benefits include improved factual accuracy, source attribution, and the ability to update knowledge without retraining the model.',
    relevant: true,
  },
  {
    id: '4',
    source: 'Doc 4, Page 5',
    snippet: 'Machine learning models can be trained on various datasets including text, images, and structured data.',
    relevant: false,
  },
];

const emitLater = (
  timeouts: number[],
  delayMs: number,
  callback: () => void
) => {
  const timeoutId = window.setTimeout(callback, delayMs);
  timeouts.push(timeoutId);
};

const chunkAnswer = (answer: string) => answer.match(/\S+\s*/g) ?? [];

export function createDemoTransport(): RAGTransport {
  let callbacks: TransportCallbacks | null = null;
  let timeouts: number[] = [];

  const clearPending = () => {
    timeouts.forEach(window.clearTimeout);
    timeouts = [];
  };

  const emit = (event: RAGServerEvent) => {
    callbacks?.onEvent(event);
  };

  return {
    mode: 'demo',
    connect(nextCallbacks) {
      callbacks = nextCallbacks;
      callbacks.onConnectionStatusChange('demo');
    },
    disconnect() {
      clearPending();
      callbacks = null;
    },
    submitQuery(query) {
      clearPending();

      emitLater(timeouts, 0, () => emit({ type: 'run_started', runId: `demo-${Date.now()}`, query }));
      emitLater(timeouts, 200, () => emit({ type: 'step_changed', step: 'rewriting' }));
      emitLater(timeouts, 900, () => emit({
        type: 'query_rewritten',
        rewrittenQuery: `Enhanced query: ${query} (with context and specificity)`,
      }));
      emitLater(timeouts, 1100, () => emit({ type: 'step_changed', step: 'retrieval' }));
      emitLater(timeouts, 1800, () => emit({ type: 'documents_retrieved', retrievedDocs: DEMO_DOCS }));
      emitLater(timeouts, 2100, () => emit({ type: 'step_changed', step: 'generation' }));

      const answerChunks = chunkAnswer(DEMO_ANSWER);
      answerChunks.forEach((chunk, index) => {
        emitLater(timeouts, 2400 + index * 90, () => emit({ type: 'answer_delta', delta: chunk }));
      });

      const gradingDelay = 2400 + answerChunks.length * 90 + 250;
      emitLater(timeouts, gradingDelay, () => emit({ type: 'step_changed', step: 'grading' }));
      emitLater(timeouts, gradingDelay + 500, () => emit({
        type: 'grading_completed',
        hallucinationResult: {
          score: 0.85,
          explanation: 'All major claims are supported by the retrieved documents shown in the trace.',
        },
      }));
      emitLater(timeouts, gradingDelay + 700, () => emit({ type: 'run_completed' }));
    },
  };
}

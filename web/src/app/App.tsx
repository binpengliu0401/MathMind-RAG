import { useState } from 'react';
import { QueryInput } from './components/query-input';
import { AnswerCard } from './components/answer-card';
import { ReasoningPanel } from './components/reasoning-panel';
import { LoadingSteps } from './components/loading-steps';
import { Card } from './components/ui/card';
import { AlertCircle, Brain } from 'lucide-react';
import { useRAGSession } from './hooks/use-rag-session';

export default function App() {
  const [query, setQuery] = useState('');
  const { snapshot, connectionStatus, transportMode, isRunning, submitQuery } = useRAGSession();

  const handleSubmit = async () => {
    if (!query.trim()) return;
    submitQuery(query);
  };

  const isTransportReady = connectionStatus === 'connected' || connectionStatus === 'demo';
  const shouldDisableInput = isRunning || !isTransportReady;
  const showLoadingCard = isRunning && snapshot.answer.length === 0;
  const showAnswerCard = snapshot.answer.length > 0 || snapshot.runStatus === 'complete';
  const shouldShowError = snapshot.runStatus === 'error' || connectionStatus === 'error' || connectionStatus === 'disconnected';
  const errorMessage = snapshot.error
    ?? (connectionStatus === 'disconnected' ? 'Connection lost. Reconnect the WebSocket to continue.' : '')
    ?? (connectionStatus === 'error' ? 'WebSocket connection error.' : '');

  return (
    <div className="h-screen bg-[#0B0F14] overflow-hidden flex flex-col">
      {/* Background gradient effects */}
      <div className="fixed inset-0 overflow-hidden pointer-events-none">
        <div className="absolute top-0 left-1/4 w-96 h-96 bg-blue-500/10 rounded-full blur-3xl" />
        <div className="absolute bottom-0 right-1/4 w-96 h-96 bg-purple-500/10 rounded-full blur-3xl" />
      </div>

      <div className="container mx-auto px-4 py-8 max-w-7xl relative z-10 flex-1 min-h-0 flex flex-col">
        {/* Header */}
        <div className="mb-8 text-center flex-shrink-0">
          <div className="flex items-center justify-center gap-3 mb-3">
            <Brain className="size-8 text-blue-400" />
            <h1 className="text-4xl text-white">Neural Observatory</h1>
          </div>
          <p className="text-gray-400">Retrieval-Augmented Generation with Transparency</p>
        </div>

        <div className="grid grid-cols-1 xl:grid-cols-3 gap-6 flex-1 min-h-0 xl:overflow-hidden">
          {/* Left Column - Output and Loading */}
          <div className="xl:col-span-2 space-y-6 min-h-0 xl:overflow-y-auto xl:pr-4 scrollbar-thin scrollbar-thumb-gray-700 scrollbar-track-transparent">
            {/* Loading State */}
            {showLoadingCard && snapshot.currentStep && (
              <Card className="p-6 bg-gray-900/50 border-gray-800 backdrop-blur-sm">
                <LoadingSteps currentStep={snapshot.currentStep} />
              </Card>
            )}

            {/* Answer Display */}
            {showAnswerCard && (
              <AnswerCard
                answer={snapshot.answer}
                hallucinationScore={snapshot.hallucinationResult?.score}
                isGenerating={isRunning}
                isComplete={snapshot.runStatus === 'complete'}
              />
            )}

            {/* Error State */}
            {shouldShowError && errorMessage && (
              <Card className="p-6 bg-red-500/10 border-red-500/30 backdrop-blur-sm">
                <div className="flex items-center gap-3 text-red-400">
                  <AlertCircle className="size-5" />
                  <p>{errorMessage}</p>
                </div>
              </Card>
            )}
          </div>

          {/* Right Column - Reasoning Panel */}
          <div className="xl:col-span-1 min-h-0 xl:overflow-y-auto scrollbar-thin scrollbar-thumb-gray-700 scrollbar-track-transparent">
            <ReasoningPanel
              snapshot={snapshot}
              connectionStatus={connectionStatus}
              transportMode={transportMode}
            />
          </div>
        </div>

        {/* Input Section */}
        <div className="flex-shrink-0 bg-gradient-to-t from-[#0B0F14] via-[#0B0F14] to-transparent pt-8 pb-6">
          <div className="grid grid-cols-1 xl:grid-cols-3 gap-6">
            <div className="xl:col-span-2">
              <div className="p-6 bg-gray-900/30 rounded-xl backdrop-blur-xl">
                <QueryInput
                  query={query}
                  onQueryChange={setQuery}
                  onSubmit={handleSubmit}
                  isLoading={shouldDisableInput}
                />
              </div>
            </div>
            <div className="hidden xl:block xl:col-span-1" />
          </div>
        </div>
      </div>
    </div>
  );
}

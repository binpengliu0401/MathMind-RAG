import { Card } from "./ui/card";
import { Badge } from "./ui/badge";
import { FileText, RefreshCw, FileSearch, MessageSquare, ShieldCheck, PlugZap } from "lucide-react";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "./ui/collapsible";
import { ChevronDown, CheckCircle2 } from "lucide-react";
import type { ConnectionStatus, ProcessingStep, SessionSnapshot, TransportMode } from "../types/rag";

interface ReasoningPanelProps {
  snapshot: SessionSnapshot;
  connectionStatus: ConnectionStatus;
  transportMode: TransportMode;
}

export function ReasoningPanel({
  snapshot,
  connectionStatus,
  transportMode,
}: ReasoningPanelProps) {
  if (
    !snapshot.query &&
    !snapshot.rewrittenQuery &&
    snapshot.retrievedDocs.length === 0 &&
    !snapshot.answer &&
    !snapshot.hallucinationResult
  ) {
    return (
      <Card className="p-6 bg-gray-900/50 border-gray-800 backdrop-blur-sm h-full">
        <div className="flex items-center justify-between gap-3 mb-6">
          <h3 className="font-medium text-gray-100">Reasoning Trace</h3>
          <ConnectionBadge connectionStatus={connectionStatus} transportMode={transportMode} />
        </div>
        <div className="text-center text-gray-500 py-12">
          <FileText className="size-10 mx-auto mb-3 opacity-30" />
          <p className="text-sm">Pipeline details will appear here</p>
        </div>
      </Card>
    );
  }

  return (
    <Card className="p-6 bg-gray-900/50 border-gray-800 backdrop-blur-sm h-full flex flex-col">
      <div className="mb-6 flex items-center justify-between gap-3 flex-shrink-0">
        <h3 className="font-medium text-gray-100 flex items-center gap-2">
          <div className="size-2 bg-purple-400 rounded-full animate-pulse" />
          Reasoning Trace
        </h3>
        <ConnectionBadge connectionStatus={connectionStatus} transportMode={transportMode} />
      </div>

      <div className="space-y-8 overflow-y-auto pr-2 scrollbar-thin scrollbar-thumb-gray-700 scrollbar-track-transparent">
        <TraceBody snapshot={snapshot} />
      </div>
    </Card>
  );
}

function ConnectionBadge({
  connectionStatus,
  transportMode,
}: {
  connectionStatus: ConnectionStatus;
  transportMode: TransportMode;
}) {
  const label = transportMode === 'demo'
    ? 'Demo mode'
    : connectionStatus === 'connected'
    ? 'Socket live'
    : connectionStatus === 'connecting'
    ? 'Connecting'
    : connectionStatus === 'disconnected'
    ? 'Disconnected'
    : 'Connection error';

  const classes = transportMode === 'demo'
    ? 'bg-blue-500/10 text-blue-400 border-blue-500/30'
    : connectionStatus === 'connected'
    ? 'bg-green-500/10 text-green-400 border-green-500/30'
    : connectionStatus === 'connecting'
    ? 'bg-yellow-500/10 text-yellow-400 border-yellow-500/30'
    : 'bg-red-500/10 text-red-400 border-red-500/30';

  return (
    <Badge variant="outline" className={`text-xs ${classes}`}>
      <PlugZap className="size-3 mr-1" />
      {label}
    </Badge>
  );
}

function TraceBody({ snapshot }: { snapshot: SessionSnapshot }) {
  const currentStep = snapshot.currentStep;
  const getStepStatus = (step: string) => {
    if (!currentStep) return 'pending';
    const steps = ['rewriting', 'retrieval', 'generation', 'grading', 'complete'];
    const currentIndex = steps.indexOf(currentStep);
    const stepIndex = steps.indexOf(step);
    
    if (currentStep === 'complete' || stepIndex < currentIndex) return 'complete';
    if (stepIndex === currentIndex) return 'active';
    return 'pending';
  };

  return (
    <div className="space-y-6 relative">
      {/* Timeline line */}
      <div className="absolute left-[19px] top-6 bottom-0 w-0.5 bg-gray-800" />

      {/* Rewritten Query */}
      <div className="relative">
        <div className="flex items-start gap-4">
          <div className={`size-10 rounded-full flex items-center justify-center z-10 ${
            getStepStatus('rewriting') === 'complete' 
              ? 'bg-green-500/20 border-2 border-green-500'
              : getStepStatus('rewriting') === 'active'
              ? 'bg-blue-500/20 border-2 border-blue-500 animate-pulse'
              : 'bg-gray-800/50 border-2 border-gray-700'
          }`}>
            <RefreshCw className={`size-5 ${
              getStepStatus('rewriting') === 'complete' ? 'text-green-400' : 
              getStepStatus('rewriting') === 'active' ? 'text-blue-400' :
              'text-gray-600'
            }`} />
          </div>
          <div className="flex-1 space-y-2 pt-1">
            <div className="text-sm text-gray-400 font-medium">Query Rewrite</div>
            {getStepStatus('rewriting') !== 'pending' && (
              <div className="bg-gray-800/50 rounded-lg p-3 text-sm text-gray-300 border border-gray-700">
                "{snapshot.rewrittenQuery || 'Waiting for backend rewrite...'}"
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Retrieved Documents */}
      <div className="relative">
        <div className="flex items-start gap-4">
          <div className={`size-10 rounded-full flex items-center justify-center z-10 ${
            getStepStatus('retrieval') === 'complete' 
              ? 'bg-green-500/20 border-2 border-green-500'
              : getStepStatus('retrieval') === 'active'
              ? 'bg-blue-500/20 border-2 border-blue-500 animate-pulse'
              : 'bg-gray-800/50 border-2 border-gray-700'
          }`}>
            <FileSearch className={`size-5 ${
              getStepStatus('retrieval') === 'complete' ? 'text-green-400' : 
              getStepStatus('retrieval') === 'active' ? 'text-blue-400' :
              'text-gray-600'
            }`} />
          </div>
          <div className="flex-1 space-y-3 pt-1">
            <div className="text-sm text-gray-400 font-medium">Retrieved Documents</div>
            {getStepStatus('retrieval') !== 'pending' && (
              <div className="space-y-2">
                {snapshot.retrievedDocs.length === 0 && (
                  <div className="bg-gray-800/50 rounded-lg p-3 border border-gray-700 text-sm text-gray-500">
                    Waiting for retrieved context...
                  </div>
                )}
                {snapshot.retrievedDocs.map((doc) => (
                  <Collapsible key={doc.id}>
                    <CollapsibleTrigger className="w-full">
                      <div className="flex items-start gap-3 bg-gray-800/50 rounded-lg p-3 border border-gray-700 hover:border-gray-600 transition-colors cursor-pointer group">
                        <ChevronDown className="size-4 text-gray-500 mt-0.5 group-data-[state=open]:rotate-180 transition-transform flex-shrink-0" />
                        <div className="flex-1 text-left min-w-0">
                          <div className="flex items-center gap-2 mb-1 flex-wrap">
                            <span className="text-xs text-gray-500">{doc.source}</span>
                            {doc.relevant && (
                              <Badge variant="secondary" className="text-xs bg-green-500/10 text-green-400 border-green-500/30">
                                <CheckCircle2 className="size-3 mr-1" />
                                Supports answer
                              </Badge>
                            )}
                          </div>
                          <p className="text-sm text-gray-300 line-clamp-2">
                            {doc.snippet}
                          </p>
                        </div>
                      </div>
                    </CollapsibleTrigger>
                    <CollapsibleContent>
                      <div className="ml-7 mt-2 p-3 bg-gray-900/50 rounded-lg border border-gray-700 text-sm text-gray-400">
                        {doc.snippet}
                      </div>
                    </CollapsibleContent>
                  </Collapsible>
                ))}
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Generation Step */}
      <div className="relative">
        <div className="flex items-start gap-4">
          <div className={`size-10 rounded-full flex items-center justify-center z-10 ${
            getStepStatus('generation') === 'complete' 
              ? 'bg-green-500/20 border-2 border-green-500'
              : getStepStatus('generation') === 'active'
              ? 'bg-blue-500/20 border-2 border-blue-500 animate-pulse'
              : 'bg-gray-800/50 border-2 border-gray-700'
          }`}>
            <MessageSquare className={`size-5 ${
              getStepStatus('generation') === 'complete' ? 'text-green-400' : 
              getStepStatus('generation') === 'active' ? 'text-blue-400' :
              'text-gray-600'
            }`} />
          </div>
          <div className="flex-1 space-y-2 pt-1">
            <div className="text-sm text-gray-400 font-medium">Generation</div>
            {getStepStatus('generation') === 'active' && (
              <div className="text-xs text-gray-500">Generating response...</div>
            )}
            {getStepStatus('generation') === 'complete' && (
              <div className="text-xs text-green-400">Answer stream completed.</div>
            )}
          </div>
        </div>
      </div>

      {/* Hallucination Detection */}
      <div className="relative">
        <div className="flex items-start gap-4">
          <div className={`size-10 rounded-full flex items-center justify-center z-10 ${
            getStepStatus('grading') === 'complete'
              ? (snapshot.hallucinationResult?.score ?? 0) >= 0.7
                ? 'bg-green-500/20 border-2 border-green-500'
                : (snapshot.hallucinationResult?.score ?? 0) >= 0.4
                ? 'bg-yellow-500/20 border-2 border-yellow-500'
                : 'bg-red-500/20 border-2 border-red-500'
              : getStepStatus('grading') === 'active'
              ? 'bg-blue-500/20 border-2 border-blue-500 animate-pulse'
              : 'bg-gray-800/50 border-2 border-gray-700'
          }`}>
            <ShieldCheck className={`size-5 ${
              getStepStatus('grading') === 'complete'
                ? (snapshot.hallucinationResult?.score ?? 0) >= 0.7 ? 'text-green-400' :
                  (snapshot.hallucinationResult?.score ?? 0) >= 0.4 ? 'text-yellow-400' :
                  'text-red-400'
                : getStepStatus('grading') === 'active' ? 'text-blue-400' :
                'text-gray-600'
            }`} />
          </div>
          <div className="flex-1 space-y-3 pt-1">
            <div className="text-sm text-gray-400 font-medium">Hallucination Detection</div>
            
            {getStepStatus('grading') !== 'pending' && (
              <div className="space-y-3">
                {!snapshot.hallucinationResult && (
                  <div className="bg-gray-800/50 rounded-lg p-3 border border-gray-700 text-sm text-gray-500">
                    Waiting for grounding assessment...
                  </div>
                )}
                {snapshot.hallucinationResult && (
                  <>
                <div className="flex items-center justify-between bg-gray-800/50 rounded-lg p-3 border border-gray-700">
                  <span className="text-sm text-gray-400">Grounding Score</span>
                  <span className={`text-sm font-medium ${
                    snapshot.hallucinationResult.score >= 0.7 ? 'text-green-400' :
                    snapshot.hallucinationResult.score >= 0.4 ? 'text-yellow-400' :
                    'text-red-400'
                  }`}>
                    {snapshot.hallucinationResult.score.toFixed(2)}
                  </span>
                </div>
                
                <div className="bg-gray-800/50 rounded-lg p-3 border border-gray-700">
                  <p className="text-sm text-gray-300">{snapshot.hallucinationResult.explanation}</p>
                </div>

                {snapshot.hallucinationResult.unsupportedClaims && snapshot.hallucinationResult.unsupportedClaims.length > 0 && (
                  <div className="bg-red-500/10 rounded-lg p-3 border border-red-500/30">
                    <p className="text-xs text-red-400 font-medium mb-2">Unsupported Claims:</p>
                    <ul className="list-disc list-inside space-y-1">
                      {snapshot.hallucinationResult.unsupportedClaims.map((claim, idx) => (
                        <li key={idx} className="text-sm text-red-300">{claim}</li>
                      ))}
                    </ul>
                  </div>
                )}
                  </>
                )}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

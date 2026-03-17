import { CheckCircle2, Loader2, RefreshCw, FileSearch, MessageSquare, ShieldCheck } from "lucide-react";

interface LoadingStepsProps {
  currentStep: 'rewriting' | 'retrieval' | 'generation' | 'grading' | 'complete';
}

const steps = [
  { id: 'rewriting', label: 'Query Rewrite', icon: RefreshCw },
  { id: 'retrieval', label: 'Retrieval', icon: FileSearch },
  { id: 'generation', label: 'Generation', icon: MessageSquare },
  { id: 'grading', label: 'Hallucination Detection', icon: ShieldCheck },
];

export function LoadingSteps({ currentStep }: LoadingStepsProps) {
  const currentIndex = steps.findIndex(s => s.id === currentStep);

  return (
    <div className="relative space-y-0">
      {/* Vertical timeline line */}
      <div className="absolute left-[19px] top-8 bottom-8 w-0.5 bg-gray-800" />
      
      {steps.map((step, index) => {
        const isComplete = currentStep === 'complete' || index < currentIndex;
        const isCurrent = index === currentIndex;
        const Icon = step.icon;

        return (
          <div key={step.id} className="flex items-center gap-4 py-3 relative">
            {/* Step indicator */}
            <div className="relative z-10">
              {isComplete ? (
                <div className="size-10 rounded-full bg-green-500/20 border-2 border-green-500 flex items-center justify-center">
                  <CheckCircle2 className="size-5 text-green-400" />
                </div>
              ) : isCurrent ? (
                <div className="size-10 rounded-full bg-blue-500/20 border-2 border-blue-500 flex items-center justify-center animate-pulse shadow-lg shadow-blue-500/50">
                  <Loader2 className="size-5 text-blue-400 animate-spin" />
                </div>
              ) : (
                <div className="size-10 rounded-full bg-gray-800/50 border-2 border-gray-700 flex items-center justify-center">
                  <Icon className="size-5 text-gray-600" />
                </div>
              )}
            </div>
            
            {/* Step label */}
            <div className="flex-1">
              <span className={`text-sm font-medium transition-colors ${
                isComplete ? 'text-green-400' :
                isCurrent ? 'text-blue-400' :
                'text-gray-500'
              }`}>
                {step.label}
              </span>
              {isCurrent && (
                <div className="text-xs text-gray-500 mt-0.5">Processing...</div>
              )}
            </div>
          </div>
        );
      })}
    </div>
  );
}

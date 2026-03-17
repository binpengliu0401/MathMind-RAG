import { Card } from "./ui/card";
import { Badge } from "./ui/badge";
import { AlertCircle, CheckCircle2, AlertTriangle } from "lucide-react";
import { TypewriterText } from "./typewriter-text";
import { useEffect, useState } from "react";
import { Progress } from "./ui/progress";

interface AnswerCardProps {
  answer: string;
  hallucinationScore?: number;
  isGenerating?: boolean;
  isComplete?: boolean;
}

export function AnswerCard({ answer, hallucinationScore, isGenerating = false, isComplete = false }: AnswerCardProps) {
  const [typewriterComplete, setTypewriterComplete] = useState(false);

  useEffect(() => {
    setTypewriterComplete(isComplete || (!isGenerating && answer.length > 0));
  }, [answer, isComplete, isGenerating]);

  const getScoreColor = (score: number) => {
    if (score >= 0.7) return 'green';
    if (score >= 0.4) return 'yellow';
    return 'red';
  };

  const getScoreIcon = (score: number) => {
    if (score >= 0.7) return <CheckCircle2 className="size-4" />;
    if (score >= 0.4) return <AlertTriangle className="size-4" />;
    return <AlertCircle className="size-4" />;
  };

  const getScoreLabel = (score: number) => {
    if (score >= 0.7) return 'Grounded';
    if (score >= 0.4) return 'Partially Grounded';
    return 'Low Confidence';
  };

  const scoreColor = hallucinationScore ? getScoreColor(hallucinationScore) : 'green';
  const scoreColorClasses = {
    green: 'bg-green-500/10 text-green-400 border-green-500/30',
    yellow: 'bg-yellow-500/10 text-yellow-400 border-yellow-500/30',
    red: 'bg-red-500/10 text-red-400 border-red-500/30'
  };

  const progressColorClasses = {
    green: 'bg-green-500',
    yellow: 'bg-yellow-500',
    red: 'bg-red-500'
  };

  return (
    <Card className="p-6 space-y-4 bg-gray-900/50 border-gray-800 backdrop-blur-sm">
      <div className="flex items-start justify-between gap-4">
        <div className="flex items-center gap-3">
          <h3 className="text-sm text-gray-400">Answer</h3>
          {isComplete && (
            <Badge variant="outline" className="bg-green-500/10 text-green-400 border-green-500/30 text-xs">
              Completed
            </Badge>
          )}
        </div>
      </div>
      
      {isGenerating && (
        <div className="flex items-center gap-2 text-sm text-gray-400 mb-2">
          <div className="size-2 bg-blue-400 rounded-full animate-ping" />
          Generating answer...
        </div>
      )}
      
      <div className="text-gray-100 leading-relaxed text-base min-h-[80px]">
        <TypewriterText 
          text={answer} 
          speed={30}
          onComplete={() => setTypewriterComplete(true)}
          isActive={isGenerating}
        />
      </div>

      {hallucinationScore !== undefined && typewriterComplete && (
        <div className="space-y-3 pt-2 border-t border-gray-800 animate-in fade-in duration-500">
          <div className="flex items-center justify-between">
            <span className="text-sm text-gray-400">Grounding Score</span>
            <Badge 
              variant="outline" 
              className={`flex items-center gap-1.5 ${scoreColorClasses[scoreColor]}`}
            >
              {getScoreIcon(hallucinationScore)}
              <span>{getScoreLabel(hallucinationScore)}</span>
              <span className="ml-1 opacity-75">{hallucinationScore.toFixed(2)}</span>
            </Badge>
          </div>
          
          <div className="relative">
            <Progress 
              value={hallucinationScore * 100} 
              className="h-2 bg-gray-800"
            />
            <div 
              className={`absolute top-0 left-0 h-2 rounded-full transition-all duration-500 ${progressColorClasses[scoreColor]}`}
              style={{ width: `${hallucinationScore * 100}%` }}
            />
          </div>
        </div>
      )}
    </Card>
  );
}

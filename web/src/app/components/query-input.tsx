import { Button } from "./ui/button";
import { Textarea } from "./ui/textarea";
import { Sparkles } from "lucide-react";

interface QueryInputProps {
  query: string;
  onQueryChange: (query: string) => void;
  onSubmit: () => void;
  isLoading: boolean;
}

export function QueryInput({ query, onQueryChange, onSubmit, isLoading }: QueryInputProps) {
  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      if (query.trim() && !isLoading) {
        onSubmit();
      }
    }
  };

  return (
    <div className="w-full">
      <label className="text-sm text-gray-400 mb-2 block">
        Ask a question
      </label>
      <div className="flex gap-3 items-start">
        <Textarea
          placeholder="e.g., What are the main benefits of retrieval-augmented generation?"
          value={query}
          onChange={(e) => onQueryChange(e.target.value)}
          onKeyDown={handleKeyDown}
          className="flex-1 min-h-[56px] max-h-[56px] resize-none bg-gray-900/50 border-gray-700 text-gray-100 placeholder:text-gray-500 focus:border-blue-500 focus:ring-2 focus:ring-blue-500/20 transition-all duration-200"
          disabled={isLoading}
        />
        <Button 
          onClick={onSubmit} 
          disabled={!query.trim() || isLoading}
          className="h-[56px] px-8 bg-blue-600/30 hover:bg-blue-600/50 text-blue-300 border border-blue-500/30 hover:border-blue-500/50 backdrop-blur-sm shadow-lg shadow-blue-500/10 hover:shadow-blue-500/20 transition-all duration-200 disabled:opacity-30 disabled:shadow-none flex-shrink-0"
        >
          {isLoading ? (
            <>
              <Sparkles className="size-4 mr-2 animate-pulse" />
              Processing...
            </>
          ) : (
            'Ask'
          )}
        </Button>
      </div>
    </div>
  );
}
import { useEffect, useState } from 'react';

interface TypewriterTextProps {
  text: string;
  speed?: number;
  onComplete?: () => void;
  isActive?: boolean;
}

export function TypewriterText({ text, speed = 30, onComplete, isActive = true }: TypewriterTextProps) {
  const [displayedText, setDisplayedText] = useState('');
  const [currentIndex, setCurrentIndex] = useState(0);
  const [showCursor, setShowCursor] = useState(true);

  useEffect(() => {
    if (!isActive) {
      setDisplayedText(text);
      setCurrentIndex(text.length);
      return;
    }

    if (text.length === 0) {
      setDisplayedText('');
      setCurrentIndex(0);
      return;
    }

    if (currentIndex > text.length) {
      setDisplayedText(text);
      setCurrentIndex(text.length);
    }
  }, [text, isActive, currentIndex]);

  useEffect(() => {
    if (!isActive) {
      setDisplayedText(text);
      setCurrentIndex(text.length);
      return;
    }

    if (currentIndex < text.length) {
      const timeout = setTimeout(() => {
        setDisplayedText(text.slice(0, currentIndex + 1));
        setCurrentIndex(currentIndex + 1);
      }, speed);

      return () => clearTimeout(timeout);
    } else if (currentIndex === text.length && onComplete) {
      onComplete();
    }
  }, [currentIndex, text, speed, onComplete, isActive]);

  // Cursor blink effect
  useEffect(() => {
    const interval = setInterval(() => {
      setShowCursor(prev => !prev);
    }, 500);

    return () => clearInterval(interval);
  }, []);

  const isComplete = currentIndex >= text.length;

  return (
    <span className="relative">
      {displayedText}
      {isActive && (
        <span 
          className={`inline-block w-0.5 h-5 ml-0.5 bg-blue-400 align-middle transition-opacity duration-100 ${
            isComplete ? 'opacity-30' : (showCursor ? 'opacity-100' : 'opacity-0')
          }`}
        />
      )}
    </span>
  );
}

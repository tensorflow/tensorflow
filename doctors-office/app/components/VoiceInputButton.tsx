"use client";

import { useVoiceInput } from "../hooks/useVoiceInput";

interface VoiceInputButtonProps {
  onTranscript: (text: string) => void;
  className?: string;
  label?: string;
}

export default function VoiceInputButton({
  onTranscript,
  className = "",
  label = "Voice Input",
}: VoiceInputButtonProps) {
  const { isListening, isSupported, startListening, stopListening, transcript, error } =
    useVoiceInput({
      continuous: false,
      interimResults: true,
      onResult: (text) => {
        onTranscript(text);
      },
    });

  const handleClick = () => {
    if (isListening) {
      stopListening();
    } else {
      startListening();
    }
  };

  if (!isSupported) {
    return (
      <button
        type="button"
        disabled
        className={`px-3 py-2 bg-gray-300 text-gray-500 rounded-lg cursor-not-allowed ${className}`}
        title="Voice input not supported in this browser"
      >
        <svg
          className="w-5 h-5"
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2}
            d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z"
          />
          <line x1="2" y1="2" x2="22" y2="22" stroke="currentColor" strokeWidth={2} />
        </svg>
      </button>
    );
  }

  return (
    <div className="relative">
      <button
        type="button"
        onClick={handleClick}
        className={`px-3 py-2 rounded-lg transition-all duration-200 ${
          isListening
            ? "bg-red-500 text-white animate-pulse"
            : "bg-blue-600 text-white hover:bg-blue-700"
        } ${className}`}
        title={isListening ? "Stop listening" : label}
        aria-label={isListening ? "Stop voice input" : "Start voice input"}
      >
        <svg
          className="w-5 h-5"
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24"
        >
          {isListening ? (
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
            />
          ) : (
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z"
            />
          )}
        </svg>
      </button>

      {isListening && transcript && (
        <div className="absolute top-full mt-2 left-0 right-0 bg-blue-50 border border-blue-200 rounded-lg p-2 text-sm text-blue-900 shadow-lg z-10 min-w-[200px]">
          <p className="font-medium text-xs text-blue-600 mb-1">Listening...</p>
          <p className="text-blue-900">{transcript}</p>
        </div>
      )}

      {error && (
        <div className="absolute top-full mt-2 left-0 right-0 bg-red-50 border border-red-200 rounded-lg p-2 text-sm text-red-900 shadow-lg z-10 min-w-[200px]">
          <p className="font-medium text-xs text-red-600 mb-1">Error</p>
          <p className="text-red-900">{error}</p>
        </div>
      )}
    </div>
  );
}

"use client";

import { useVoiceOutput } from "../hooks/useVoiceOutput";

interface VoiceOutputButtonProps {
  text: string;
  className?: string;
  label?: string;
}

export default function VoiceOutputButton({
  text,
  className = "",
  label = "Read aloud",
}: VoiceOutputButtonProps) {
  const { speak, speaking, supported, cancel } = useVoiceOutput({
    rate: 1,
    pitch: 1,
    volume: 1,
  });

  const handleClick = () => {
    if (speaking) {
      cancel();
    } else {
      speak(text);
    }
  };

  if (!supported) {
    return (
      <button
        type="button"
        disabled
        className={`px-3 py-2 bg-gray-300 text-gray-500 rounded-lg cursor-not-allowed ${className}`}
        title="Text-to-speech not supported in this browser"
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
            d="M15.536 8.464a5 5 0 010 7.072m2.828-9.9a9 9 0 010 12.728M5.586 15H4a1 1 0 01-1-1v-4a1 1 0 011-1h1.586l4.707-4.707C10.923 3.663 12 4.109 12 5v14c0 .891-1.077 1.337-1.707.707L5.586 15z"
          />
          <line x1="2" y1="2" x2="22" y2="22" stroke="currentColor" strokeWidth={2} />
        </svg>
      </button>
    );
  }

  return (
    <button
      type="button"
      onClick={handleClick}
      className={`px-3 py-2 rounded-lg transition-all duration-200 ${
        speaking
          ? "bg-green-500 text-white animate-pulse"
          : "bg-green-600 text-white hover:bg-green-700"
      } ${className}`}
      title={speaking ? "Stop speaking" : label}
      aria-label={speaking ? "Stop text-to-speech" : "Start text-to-speech"}
    >
      <svg
        className="w-5 h-5"
        fill="none"
        stroke="currentColor"
        viewBox="0 0 24 24"
      >
        {speaking ? (
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
            d="M15.536 8.464a5 5 0 010 7.072m2.828-9.9a9 9 0 010 12.728M5.586 15H4a1 1 0 01-1-1v-4a1 1 0 011-1h1.586l4.707-4.707C10.923 3.663 12 4.109 12 5v14c0 .891-1.077 1.337-1.707.707L5.586 15z"
          />
        )}
      </svg>
    </button>
  );
}

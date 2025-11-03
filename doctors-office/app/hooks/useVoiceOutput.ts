"use client";

import { useState, useEffect, useCallback, useRef } from "react";

interface UseVoiceOutputOptions {
  rate?: number; // 0.1 to 10
  pitch?: number; // 0 to 2
  volume?: number; // 0 to 1
  lang?: string;
  onEnd?: () => void;
  onError?: (error: string) => void;
}

interface UseVoiceOutputReturn {
  speak: (text: string) => void;
  speaking: boolean;
  supported: boolean;
  pause: () => void;
  resume: () => void;
  cancel: () => void;
  voices: SpeechSynthesisVoice[];
  setVoice: (voice: SpeechSynthesisVoice) => void;
}

export const useVoiceOutput = (
  options: UseVoiceOutputOptions = {}
): UseVoiceOutputReturn => {
  const {
    rate = 1,
    pitch = 1,
    volume = 1,
    lang = "en-US",
    onEnd,
    onError,
  } = options;

  const [speaking, setSpeaking] = useState(false);
  const [supported, setSupported] = useState(false);
  const [voices, setVoices] = useState<SpeechSynthesisVoice[]>([]);
  const [selectedVoice, setSelectedVoice] = useState<SpeechSynthesisVoice | null>(null);

  const utteranceRef = useRef<SpeechSynthesisUtterance | null>(null);

  useEffect(() => {
    if (typeof window !== "undefined" && window.speechSynthesis) {
      setSupported(true);

      // Load available voices
      const loadVoices = () => {
        const availableVoices = window.speechSynthesis.getVoices();
        setVoices(availableVoices);
        
        // Set default voice for the specified language
        const defaultVoice = availableVoices.find(
          (voice) => voice.lang === lang
        ) || availableVoices[0];
        setSelectedVoice(defaultVoice);
      };

      loadVoices();
      
      // Chrome loads voices asynchronously
      if (window.speechSynthesis.onvoiceschanged !== undefined) {
        window.speechSynthesis.onvoiceschanged = loadVoices;
      }
    } else {
      setSupported(false);
    }

    return () => {
      if (window.speechSynthesis) {
        window.speechSynthesis.cancel();
      }
    };
  }, [lang]);

  const speak = useCallback(
    (text: string) => {
      if (!supported || !text.trim()) {
        return;
      }

      // Cancel any ongoing speech
      window.speechSynthesis.cancel();

      const utterance = new SpeechSynthesisUtterance(text);
      utterance.rate = rate;
      utterance.pitch = pitch;
      utterance.volume = volume;
      utterance.lang = lang;

      if (selectedVoice) {
        utterance.voice = selectedVoice;
      }

      utterance.onstart = () => {
        setSpeaking(true);
      };

      utterance.onend = () => {
        setSpeaking(false);
        if (onEnd) {
          onEnd();
        }
      };

      utterance.onerror = (event) => {
        setSpeaking(false);
        const errorMessage = `Speech synthesis error: ${event.error}`;
        if (onError) {
          onError(errorMessage);
        }
      };

      utteranceRef.current = utterance;
      window.speechSynthesis.speak(utterance);
    },
    [supported, rate, pitch, volume, lang, selectedVoice, onEnd, onError]
  );

  const pause = useCallback(() => {
    if (supported && speaking) {
      window.speechSynthesis.pause();
    }
  }, [supported, speaking]);

  const resume = useCallback(() => {
    if (supported) {
      window.speechSynthesis.resume();
    }
  }, [supported]);

  const cancel = useCallback(() => {
    if (supported) {
      window.speechSynthesis.cancel();
      setSpeaking(false);
    }
  }, [supported]);

  const setVoice = useCallback((voice: SpeechSynthesisVoice) => {
    setSelectedVoice(voice);
  }, []);

  return {
    speak,
    speaking,
    supported,
    pause,
    resume,
    cancel,
    voices,
    setVoice,
  };
};

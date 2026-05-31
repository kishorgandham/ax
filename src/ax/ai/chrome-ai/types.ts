import type { AxModelConfig } from '../types.js';

/**
 * Chrome AI: Model for text generation
 * Chrome ships with Gemini Nano as its built-in AI model
 */
export enum AxAIChromeAIModel {
  GeminiNano = 'gemini-nano',
}

/**
 * Chrome AI: Model options for text generation
 */
export type AxAIChromeAIConfig = AxModelConfig & {
  model: AxAIChromeAIModel;
};

/**
 * Chrome AI: Minimal LanguageModel interface
 * Typed inline to avoid external dependency on @types/dom-chromium-ai
 * Based on the Chrome Prompt API spec:
 * https://developer.chrome.com/docs/ai/prompt-api
 */
export interface ChromeAILanguageModel {
  availability(): Promise<
    'available' | 'downloadable' | 'downloading' | 'unavailable'
  >;
  create(options?: ChromeAICreateOptions): Promise<ChromeAISession>;
}

export interface ChromeAICreateOptions {
  systemPrompt?: string;
  initialPrompts?: Array<{
    role: 'system' | 'user' | 'assistant';
    content: string;
  }>;
  temperature?: number;
  topK?: number;
  expectedInputLanguages?: string[];
  expectedOutputLanguage?: string;
}

export interface ChromeAIPromptOptions {
  responseConstraint?: object;
}

export interface ChromeAISession {
  prompt(input: string, options?: ChromeAIPromptOptions): Promise<string>;
  promptStreaming(
    input: string,
    options?: ChromeAIPromptOptions
  ): ReadableStream<string>;
  destroy(): void;
}

/**
 * Chrome AI: Synthetic chat request
 * Maps ax's multi-message format to Chrome AI's session + prompt model
 */
export type AxAIChromeAIChatRequest = {
  model: AxAIChromeAIModel;
  /** Messages to be passed as initialPrompts (all except the last user message) */
  initialPrompts: Array<{
    role: 'system' | 'user' | 'assistant';
    content: string;
  }>;
  /** The final user message to be passed to session.prompt() */
  prompt: string;
  /** JSON schema for constrained decoding via responseConstraint */
  responseConstraint?: object;
  temperature?: number;
  topK?: number;
  stream?: boolean;
};

/**
 * Chrome AI: Synthetic chat response
 * Wraps Chrome AI's text response into an OpenAI-like structure for ax
 */
export type AxAIChromeAIChatResponse = {
  id: string;
  content: string;
  finishReason: 'stop' | 'length';
};

/**
 * Chrome AI: Streaming response delta
 * Chrome AI's promptStreaming() returns cumulative text.
 * This type wraps a cumulative chunk for processing.
 */
export type AxAIChromeAIChatResponseDelta = {
  id: string;
  /** Cumulative content so far (not a delta) */
  content: string;
  done: boolean;
};

/**
 * Chrome AI doesn't support embeddings
 * Placeholders for consistency with the framework
 */
export type AxAIChromeAIEmbedModel = never;
export type AxAIChromeAIEmbedRequest = never;
export type AxAIChromeAIEmbedResponse = never;

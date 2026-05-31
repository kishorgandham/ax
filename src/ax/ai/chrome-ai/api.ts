import type { AxAPI } from '../../util/apicall.js';
import {
  AxBaseAI,
  axBaseAIDefaultConfig,
  axBaseAIDefaultCreativeConfig,
} from '../base.js';
import type {
  AxAIInputModelList,
  AxAIServiceImpl,
  AxAIServiceOptions,
  AxChatResponse,
  AxEmbedResponse,
  AxInternalChatRequest,
  AxInternalEmbedRequest,
  AxModelConfig,
  AxTokenUsage,
} from '../types.js';

import { axModelInfoChromeAI } from './info.js';
import {
  type AxAIChromeAIChatRequest,
  type AxAIChromeAIChatResponse,
  type AxAIChromeAIChatResponseDelta,
  type AxAIChromeAIConfig,
  type AxAIChromeAIEmbedModel,
  type AxAIChromeAIEmbedRequest,
  type AxAIChromeAIEmbedResponse,
  AxAIChromeAIModel,
  type ChromeAILanguageModel,
  type ChromeAISession,
} from './types.js';

export const axAIChromeAIDefaultConfig = (): AxAIChromeAIConfig =>
  structuredClone({
    model: AxAIChromeAIModel.GeminiNano,
    ...axBaseAIDefaultConfig(),
  });

export const axAIChromeAICreativeConfig = (): AxAIChromeAIConfig =>
  structuredClone({
    model: AxAIChromeAIModel.GeminiNano,
    ...axBaseAIDefaultCreativeConfig(),
  });

export interface AxAIChromeAIArgs<TModelKey> {
  name: 'chrome-ai';
  config?: Readonly<Partial<AxAIChromeAIConfig>>;
  options?: Readonly<AxAIServiceOptions>;
  models?: AxAIInputModelList<
    AxAIChromeAIModel,
    AxAIChromeAIEmbedModel,
    TModelKey
  >;
}

/**
 * Gets the LanguageModel API from the global scope.
 * Works in Chrome 138+ where the API is available on globalThis.
 */
function getLanguageModelAPI(): ChromeAILanguageModel {
  const lm = (globalThis as any).LanguageModel as
    | ChromeAILanguageModel
    | undefined;
  if (!lm) {
    throw new Error(
      'Chrome built-in AI (LanguageModel) is not available. ' +
        'Requires Chrome 138+ with the Prompt API enabled.'
    );
  }
  return lm;
}

class AxAIChromeAIImpl
  implements
    AxAIServiceImpl<
      AxAIChromeAIModel,
      AxAIChromeAIEmbedModel,
      AxAIChromeAIChatRequest,
      AxAIChromeAIEmbedRequest,
      AxAIChromeAIChatResponse,
      AxAIChromeAIChatResponseDelta,
      AxAIChromeAIEmbedResponse
    >
{
  private tokensUsed: AxTokenUsage | undefined;

  constructor(private config: AxAIChromeAIConfig) {}

  getTokenUsage(): AxTokenUsage | undefined {
    return this.tokensUsed;
  }

  getModelConfig(): AxModelConfig {
    const { config } = this;
    return {
      maxTokens: config.maxTokens,
      temperature: config.temperature,
      topK: config.topK,
      stream: config.stream,
    } as AxModelConfig;
  }

  createChatReq(
    req: Readonly<AxInternalChatRequest<AxAIChromeAIModel>>,
    _options?: Readonly<AxAIServiceOptions>
  ): [AxAPI, AxAIChromeAIChatRequest] {
    const model = req.model;

    // Separate system prompt and build initialPrompts for context
    const initialPrompts: AxAIChromeAIChatRequest['initialPrompts'] = [];
    let lastUserMessage = '';

    for (let i = 0; i < req.chatPrompt.length; i++) {
      const msg = req.chatPrompt[i];
      const isLastMessage = i === req.chatPrompt.length - 1;

      if (msg.role === 'system') {
        // System messages go into initialPrompts
        // System role content is always a string in ax
        initialPrompts.push({ role: 'system', content: msg.content });
      } else if (msg.role === 'user') {
        // Extract text content from user message
        const content =
          typeof msg.content === 'string'
            ? msg.content
            : msg.content
                .filter((c) => c.type === 'text')
                .map((c) => c.text)
                .join('\n');

        if (isLastMessage) {
          // The last user message becomes the prompt
          lastUserMessage = content;
        } else {
          // Earlier user messages go into initialPrompts for context
          initialPrompts.push({ role: 'user', content });
        }
      } else if (msg.role === 'assistant') {
        // Assistant messages go into initialPrompts for context
        const content = msg.content || '';
        initialPrompts.push({ role: 'assistant', content });
      } else if (msg.role === 'function') {
        // Chrome AI doesn't support function calling.
        // Include function results as assistant context.
        const content =
          typeof msg.result === 'string'
            ? msg.result
            : JSON.stringify(msg.result);
        initialPrompts.push({ role: 'assistant', content });
      }
    }

    // If no user message was found at the end, use empty string
    if (!lastUserMessage && req.chatPrompt.length > 0) {
      const lastMsg = req.chatPrompt[req.chatPrompt.length - 1];
      if (lastMsg.role !== 'user') {
        lastUserMessage = '';
      }
    }

    // Map responseFormat to Chrome AI's responseConstraint
    // Chrome AI takes the JSON schema object directly (not stringified)
    let responseConstraint: object | undefined;
    if (req.responseFormat?.schema) {
      if (typeof req.responseFormat.schema === 'string') {
        try {
          responseConstraint = JSON.parse(req.responseFormat.schema);
        } catch {
          // If parsing fails, skip the constraint
        }
      } else {
        // req.responseFormat.schema is { schema: <actual schema> }
        responseConstraint =
          (req.responseFormat.schema as any).schema ||
          req.responseFormat.schema;
      }
    }

    // Build the localCall that creates a fresh session per request
    const apiConfig = {
      name: '/prompt',
      localCall: async <TRequest, TResponse>(
        data: TRequest,
        stream?: boolean
      ): Promise<TResponse | ReadableStream<TResponse>> => {
        const reqData = data as unknown as AxAIChromeAIChatRequest;
        const languageModel = getLanguageModelAPI();

        // Check availability
        const availability = await languageModel.availability();
        if (availability === 'unavailable') {
          throw new Error(
            'Chrome built-in AI model is not available on this device.'
          );
        }

        // Create a fresh session per request (no session reuse)
        const session = await languageModel.create({
          ...(reqData.initialPrompts.length > 0
            ? { initialPrompts: reqData.initialPrompts }
            : {}),
          ...(reqData.temperature !== undefined
            ? { temperature: reqData.temperature }
            : {}),
          ...(reqData.topK !== undefined ? { topK: reqData.topK } : {}),
        });

        try {
          const promptOptions = reqData.responseConstraint
            ? { responseConstraint: reqData.responseConstraint }
            : undefined;

          if (stream) {
            return this.handleStreaming(
              session,
              reqData.prompt,
              promptOptions
            ) as TResponse | ReadableStream<TResponse>;
          }

          // Non-streaming: call session.prompt()
          const content = await session.prompt(reqData.prompt, promptOptions);

          const response: AxAIChromeAIChatResponse = {
            id: `chrome-ai-${Date.now()}`,
            content,
            finishReason: 'stop',
          };

          return response as TResponse | ReadableStream<TResponse>;
        } finally {
          // Always destroy the session after the request
          session.destroy();
        }
      },
    };

    const reqValue: AxAIChromeAIChatRequest = {
      model,
      initialPrompts,
      prompt: lastUserMessage,
      ...(responseConstraint ? { responseConstraint } : {}),
      ...(req.modelConfig?.temperature !== undefined
        ? { temperature: req.modelConfig.temperature }
        : this.config.temperature !== undefined
          ? { temperature: this.config.temperature }
          : {}),
      ...(req.modelConfig?.topK !== undefined
        ? { topK: req.modelConfig.topK }
        : this.config.topK !== undefined
          ? { topK: this.config.topK }
          : {}),
      stream: req.modelConfig?.stream ?? this.config.stream,
    };

    return [apiConfig, reqValue];
  }

  /**
   * Handle streaming from Chrome AI's promptStreaming().
   * Chrome AI returns cumulative text chunks, so we wrap them as
   * AxAIChromeAIChatResponseDelta for the response handler to diff.
   */
  private handleStreaming(
    session: ChromeAISession,
    prompt: string,
    options?: { responseConstraint: object }
  ): ReadableStream<AxAIChromeAIChatResponseDelta> {
    const id = `chrome-ai-${Date.now()}`;
    const chromeStream = session.promptStreaming(prompt, options);
    const reader = chromeStream.getReader();
    let sessionDestroyed = false;

    return new ReadableStream<AxAIChromeAIChatResponseDelta>({
      async pull(controller) {
        try {
          const { done, value } = await reader.read();

          if (done) {
            if (!sessionDestroyed) {
              sessionDestroyed = true;
              session.destroy();
            }
            controller.close();
            return;
          }

          const delta: AxAIChromeAIChatResponseDelta = {
            id,
            content: value ?? '',
            done: false,
          };

          controller.enqueue(delta);
        } catch (error) {
          if (!sessionDestroyed) {
            sessionDestroyed = true;
            session.destroy();
          }
          controller.error(error);
        }
      },
      cancel() {
        if (!sessionDestroyed) {
          sessionDestroyed = true;
          session.destroy();
        }
      },
    });
  }

  createEmbedReq = (
    _req: Readonly<AxInternalEmbedRequest<AxAIChromeAIEmbedModel>>
  ): [AxAPI, AxAIChromeAIEmbedRequest] => {
    throw new Error('Chrome AI does not support embeddings');
  };

  createChatResp = (
    resp: Readonly<AxAIChromeAIChatResponse>
  ): AxChatResponse => {
    const results = [
      {
        index: 0,
        id: resp.id,
        content: resp.content,
        finishReason: resp.finishReason as 'stop' | 'length',
      },
    ];

    return { results, remoteId: resp.id };
  };

  createChatStreamResp = (
    resp: Readonly<AxAIChromeAIChatResponseDelta>,
    state: object
  ): AxChatResponse => {
    const ss = state as {
      previousContent?: string;
    };

    // Chrome AI streaming returns cumulative content.
    // Compute the delta by diffing with previous content.
    const cumulativeContent = resp.content || '';
    const previousContent = ss.previousContent || '';
    const deltaContent = cumulativeContent.startsWith(previousContent)
      ? cumulativeContent.slice(previousContent.length)
      : cumulativeContent;

    // Update state with current cumulative content
    ss.previousContent = cumulativeContent;

    const finishReason = resp.done ? ('stop' as const) : undefined;

    const results = [
      {
        index: 0,
        id: resp.id,
        content: deltaContent,
        finishReason,
      },
    ];

    return { results, remoteId: resp.id };
  };

  createEmbedResp(_resp: Readonly<AxAIChromeAIEmbedResponse>): AxEmbedResponse {
    throw new Error('Chrome AI does not support embeddings');
  }
}

/**
 * AxAIChromeAI: Adapter for Chrome's built-in AI (Prompt API)
 *
 * Chrome ships with Gemini Nano, accessible via the LanguageModel API.
 * This adapter enables ax-llm to use Chrome's built-in model for
 * chat completions with structured output support.
 *
 * Key characteristics:
 * - Browser-only (Chrome 138+)
 * - No API key required — runs locally
 * - Supports structured outputs via responseConstraint (JSON schema)
 * - No function/tool calling support
 * - No embeddings support
 * - Fresh session per request (no context leakage)
 *
 * @example
 * ```typescript
 * const chromeAI = ai({ name: 'chrome-ai' });
 * const gen = ax('question -> answer');
 * const result = await gen.forward(chromeAI, { question: 'What is 2+2?' });
 * ```
 */
export class AxAIChromeAI<TModelKey = AxAIChromeAIModel> extends AxBaseAI<
  AxAIChromeAIModel,
  AxAIChromeAIEmbedModel,
  AxAIChromeAIChatRequest,
  AxAIChromeAIEmbedRequest,
  AxAIChromeAIChatResponse,
  AxAIChromeAIChatResponseDelta,
  AxAIChromeAIEmbedResponse,
  TModelKey
> {
  constructor({
    config,
    options,
    models,
  }: Readonly<Omit<AxAIChromeAIArgs<TModelKey>, 'name'>>) {
    const Config = {
      ...axAIChromeAIDefaultConfig(),
      ...config,
    };

    const aiImpl = new AxAIChromeAIImpl(Config);

    super(aiImpl, {
      name: 'ChromeAI',
      apiURL: undefined, // No URL needed for local inference
      headers: async () => ({}), // No headers needed
      modelInfo: axModelInfoChromeAI,
      defaults: { model: Config.model },
      supportFor: (_model: AxAIChromeAIModel) => ({
        functions: false, // Chrome AI doesn't support function/tool calling
        streaming: true,
        structuredOutputs: true,
        hasThinkingBudget: false,
        hasShowThoughts: false,
        media: {
          images: {
            supported: false,
            formats: [],
          },
          audio: {
            supported: false,
            formats: [],
          },
          files: {
            supported: false,
            formats: [],
            uploadMethod: 'none' as const,
          },
          urls: {
            supported: false,
            webSearch: false,
            contextFetching: false,
          },
        },
        caching: {
          supported: false,
          types: [],
        },
        thinking: false,
        multiTurn: true,
      }),
      options,
      models,
    });
  }
}

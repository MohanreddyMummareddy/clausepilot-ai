import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

const { generateContentMock, MockApiError } = vi.hoisted(() => {
  class ApiError extends Error {
    readonly status: number

    constructor(message: string, status: number) {
      super(message)
      this.name = 'ApiError'
      this.status = status
    }
  }

  return { generateContentMock: vi.fn(), MockApiError: ApiError }
})

vi.mock('@google/genai', () => {
  class MockGoogleGenAI {
    models = { generateContent: generateContentMock }
  }

  return {
    ApiError: MockApiError,
    GoogleGenAI: MockGoogleGenAI,
  }
})

import { analyzeRequestBody, askRequestBody } from '../api/gemini.js'

const sourceText = 'Sample Agreement\n\nThe notice period is thirty days.'
const document = {
  name: 'agreement.txt',
  mimeType: 'text/plain' as const,
  data: Buffer.from(sourceText).toString('base64'),
}

function modelResponse(value: unknown) {
  return {
    text: JSON.stringify(value),
    candidates: [{ finishReason: 'STOP' }],
    promptFeedback: {},
  }
}

function validAnalysis() {
  return {
    documentTitle: 'Sample Agreement',
    documentType: 'Contract',
    summary: {
      text: 'The notice period is thirty days.',
      citations: [
        {
          excerpt: 'The notice period is thirty days.',
          locator: { kind: 'text_section', section: '[unheaded]', lineStart: 3, lineEnd: 3 },
        },
      ],
    },
    keyTerms: [],
    riskFlags: [],
    suggestedQuestions: ['What notice period applies?'],
  }
}

function validAnswer() {
  return {
    answer: 'The notice period is thirty days.',
    citations: [
      {
        excerpt: 'The notice period is thirty days.',
        locator: { kind: 'text_section', section: '[unheaded]', lineStart: 3, lineEnd: 3 },
      },
    ],
    notFound: false,
    followUpQuestions: ['When does the period begin?'],
  }
}

beforeEach(() => {
  process.env.GEMINI_API_KEY = 'test-key'
  process.env.GEMINI_MODEL = 'gemini-3.5-flash-lite'
  generateContentMock.mockReset()
})

afterEach(() => {
  delete process.env.GEMINI_API_KEY
  delete process.env.GEMINI_MODEL
})

describe('Gemini request controls', () => {
  it('grounds text citations against the decoded document', async () => {
    generateContentMock.mockResolvedValueOnce(modelResponse(validAnalysis()))

    const result = await analyzeRequestBody(document)

    expect(result.summary.citations[0]?.excerpt).toBe('The notice period is thirty days.')
    expect(generateContentMock).toHaveBeenCalledTimes(1)
    const request = generateContentMock.mock.calls[0]?.[0] as {
      config: { abortSignal: AbortSignal; responseMimeType: string }
    }
    expect(request.config.abortSignal).toBeInstanceOf(AbortSignal)
    expect(request.config.responseMimeType).toBe('application/json')
  })

  it('does not retry an invalid-argument response from the default model', async () => {
    generateContentMock.mockRejectedValueOnce(new MockApiError('invalid argument', 400))

    await expect(analyzeRequestBody(document)).rejects.toMatchObject({
      code: 'DOCUMENT_REJECTED',
    })
    expect(generateContentMock).toHaveBeenCalledTimes(1)
  })

  it('uses the fallback model once for a transient service failure', async () => {
    generateContentMock
      .mockRejectedValueOnce(new MockApiError('service unavailable', 503))
      .mockResolvedValueOnce(modelResponse(validAnalysis()))

    await expect(analyzeRequestBody(document)).resolves.toMatchObject({
      documentTitle: 'Sample Agreement',
    })
    expect(generateContentMock).toHaveBeenCalledTimes(2)
    expect((generateContentMock.mock.calls[1]?.[0] as { model: string }).model).toBe(
      'gemini-3.1-flash-lite-preview',
    )
  })

  it('spends at most two calls when malformed JSON needs repair', async () => {
    generateContentMock.mockResolvedValue({ text: 'not json', candidates: [{ finishReason: 'STOP' }] })

    await expect(analyzeRequestBody(document)).rejects.toMatchObject({
      code: 'INVALID_AI_RESPONSE',
    })
    expect(generateContentMock).toHaveBeenCalledTimes(2)
  })

  it('requires citations for a grounded non-not-found answer', async () => {
    const answer = validAnswer()
    answer.citations = []
    generateContentMock
      .mockResolvedValueOnce(modelResponse(answer))
      .mockResolvedValueOnce({ text: 'not json', candidates: [{ finishReason: 'STOP' }] })

    await expect(
      askRequestBody({ ...document, question: 'What is the notice period?' }),
    ).rejects.toMatchObject({ code: 'INVALID_AI_RESPONSE' })
    expect(generateContentMock).toHaveBeenCalledTimes(2)
  })
})

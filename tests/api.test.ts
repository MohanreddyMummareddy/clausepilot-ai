import { afterEach, describe, expect, it, vi } from 'vitest'
import {
  analyzeDocument,
  ClientApiError,
  createDocumentPayload,
  validateDocumentFile,
} from '../src/api.js'
import {
  analyzeRequestBody,
  PublicError,
  requestBody,
  requireApiRequest,
  sendApiError,
  type ApiRequest,
} from '../api/gemini.js'

function createResponse() {
  const headers: Record<string, string> = {}
  return {
    headers,
    statusCode: 200,
    body: undefined as unknown,
    setHeader(name: string, value: string) {
      headers[name] = value
    },
    status(code: number) {
      this.statusCode = code
      return this
    },
    json(body: unknown) {
      this.body = body
    },
    end() {},
  }
}

function createRequest(overrides: Partial<ApiRequest> = {}): ApiRequest {
  return {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    ...overrides,
  }
}

const originalFetch = globalThis.fetch

afterEach(() => {
  vi.unstubAllGlobals()
  globalThis.fetch = originalFetch
  delete process.env.APP_ORIGIN
})

describe('API request policy', () => {
  it('requires JSON and applies no-store response headers', () => {
    const response = createResponse()

    requireApiRequest(createRequest(), response)

    expect(response.headers['Cache-Control']).toBe('no-store')
    expect(response.headers['X-Content-Type-Options']).toBe('nosniff')
    expect(response.headers['Content-Security-Policy']).toContain("default-src 'none'")
  })

  it('rejects unsupported content types', () => {
    const response = createResponse()

    expect(() =>
      requireApiRequest(createRequest({ headers: { 'content-type': 'text/plain' } }), response),
    ).toThrowError(expect.objectContaining({ code: 'UNSUPPORTED_MEDIA_TYPE' }))
  })

  it('rejects a disallowed origin when APP_ORIGIN is configured', () => {
    process.env.APP_ORIGIN = 'https://app.example'
    const response = createResponse()

    expect(() =>
      requireApiRequest(
        createRequest({ headers: { 'content-type': 'application/json', origin: 'https://evil.example' } }),
        response,
      ),
    ).toThrowError(expect.objectContaining({ code: 'ORIGIN_NOT_ALLOWED' }))
  })

  it('accepts a same-host browser origin without requiring APP_ORIGIN', () => {
    const response = createResponse()

    expect(() =>
      requireApiRequest(
        createRequest({
          headers: {
            'content-type': 'application/json',
            host: 'localhost:3000',
            origin: 'http://localhost:3000',
          },
        }),
        response,
      ),
    ).not.toThrow()
  })

  it('limits requests per forwarded client address', () => {
    const response = createResponse()
    const address = `test-${Date.now()}-${Math.random()}`
    const request = createRequest({ headers: { 'content-type': 'application/json', 'x-forwarded-for': address } })

    for (let index = 0; index < 10; index += 1) {
      expect(() => requireApiRequest(request, response)).not.toThrow()
    }

    expect(() => requireApiRequest(request, response)).toThrowError(
      expect.objectContaining({ code: 'RATE_LIMITED' }),
    )
  })

  it('rejects oversized string request bodies before parsing', () => {
    expect(() => requestBody(createRequest({ body: 'x'.repeat(4_300_000) }))).toThrowError(
      expect.objectContaining({ code: 'REQUEST_TOO_LARGE' }),
    )
  })

  it('returns public errors without exposing internal details', () => {
    const response = createResponse()

    sendApiError(response, new PublicError(400, 'INVALID_REQUEST', 'Invalid request.'))

    expect(response.statusCode).toBe(400)
    expect(response.body).toEqual({ error: { code: 'INVALID_REQUEST', message: 'Invalid request.' } })
    expect(response.headers['Cache-Control']).toBe('no-store')
  })

  it('does not expose unexpected internal errors', () => {
    const response = createResponse()

    sendApiError(response, new Error('database password should stay private'))

    expect(response.statusCode).toBe(500)
    expect(response.body).toEqual({
      error: { code: 'INTERNAL_ERROR', message: 'Something went wrong. Please try again.' },
    })
  })
})

describe('document validation', () => {
  it('rejects a MIME type that does not match the extension', async () => {
    const data = Buffer.from('This is a sufficiently long text document for validation.').toString('base64')

    await expect(
      analyzeRequestBody({ name: 'agreement.txt', mimeType: 'application/pdf', data }),
    ).rejects.toMatchObject({ code: 'INVALID_FILE' })
  })

  it('rejects unsupported client file extensions', () => {
    const file = new File(['content'], 'agreement.exe', { type: 'application/octet-stream' })

    expect(() => validateDocumentFile(file)).toThrowError(
      expect.objectContaining({ code: 'UNSUPPORTED_FILE' }),
    )
  })

  it('creates a bounded base64 document payload', async () => {
    const file = new File(['sample text'], 'agreement.txt', { type: 'text/plain' })

    await expect(createDocumentPayload(file)).resolves.toMatchObject({
      name: 'agreement.txt',
      mimeType: 'text/plain',
    })
  })
})

describe('client response validation', () => {
  const analysis = {
    documentTitle: 'Sample agreement',
    documentType: 'Contract',
    summary: { text: 'A short summary.', citations: [] },
    keyTerms: [],
    riskFlags: [],
    suggestedQuestions: ['What is the termination period?'],
  }

  it('accepts a valid analysis response', async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      json: async () => ({ analysis }),
    })
    vi.stubGlobal('fetch', fetchMock)
    const file = new File(['sample text'], 'agreement.txt', { type: 'text/plain' })
    const payload = await createDocumentPayload(file)

    await expect(analyzeDocument(payload)).resolves.toEqual(analysis)
    expect(fetchMock).toHaveBeenCalledWith(
      '/api/analyze',
      expect.objectContaining({ method: 'POST', headers: { 'Content-Type': 'application/json' } }),
    )
  })

  it('rejects malformed analysis responses', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue({
        ok: true,
        status: 200,
        json: async () => ({ analysis: { documentTitle: 'Incomplete' } }),
      }),
    )
    const file = new File(['sample text'], 'agreement.txt', { type: 'text/plain' })
    const payload = await createDocumentPayload(file)

    await expect(analyzeDocument(payload)).rejects.toMatchObject({
      code: 'INVALID_RESPONSE',
      status: 502,
    })
  })

  it('maps server errors to ClientApiError', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue({
        ok: false,
        status: 429,
        json: async () => ({ error: { code: 'RATE_LIMITED', message: 'Try later.' } }),
      }),
    )
    const file = new File(['sample text'], 'agreement.txt', { type: 'text/plain' })
    const payload = await createDocumentPayload(file)

    await expect(analyzeDocument(payload)).rejects.toEqual(
      expect.objectContaining<Partial<ClientApiError>>({ code: 'RATE_LIMITED', status: 429 }),
    )
  })
})

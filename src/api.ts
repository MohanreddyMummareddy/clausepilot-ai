import type {
  AnalysisResult,
  ApiErrorResponse,
  AskRequest,
  AskResult,
  DocumentPayload,
} from './types'

const MAX_FILE_SIZE = 3 * 1024 * 1024
const REQUEST_TIMEOUT_MS = 70_000
const ALLOWED_EXTENSIONS = ['.pdf', '.txt', '.md']
const ALLOWED_MIME_TYPES = new Set(['application/pdf', 'text/plain', 'text/markdown'])

export class ClientApiError extends Error {
  readonly code: string
  readonly status: number

  constructor(message: string, code = 'REQUEST_FAILED', status = 500) {
    super(message)
    this.name = 'ClientApiError'
    this.code = code
    this.status = status
  }
}

export function validateDocumentFile(file: File): void {
  const lowerName = file.name.toLowerCase()
  const hasAllowedExtension = ALLOWED_EXTENSIONS.some((extension) =>
    lowerName.endsWith(extension),
  )

  if (!hasAllowedExtension) {
    throw new ClientApiError('Choose a PDF, TXT, or Markdown document.', 'UNSUPPORTED_FILE')
  }

  if (file.type && !ALLOWED_MIME_TYPES.has(file.type.toLowerCase())) {
    throw new ClientApiError('Choose a PDF, TXT, or Markdown document.', 'UNSUPPORTED_FILE')
  }

  if (file.size === 0) {
    throw new ClientApiError('The selected document is empty.', 'EMPTY_FILE')
  }

  if (file.size > MAX_FILE_SIZE) {
    throw new ClientApiError('Choose a document smaller than 3 MB.', 'FILE_TOO_LARGE')
  }
}

function toBase64(bytes: Uint8Array): string {
  const chunks: string[] = []
  const chunkSize = 32_768

  for (let offset = 0; offset < bytes.length; offset += chunkSize) {
    chunks.push(String.fromCharCode(...bytes.subarray(offset, offset + chunkSize)))
  }

  return btoa(chunks.join(''))
}

export async function createDocumentPayload(file: File): Promise<DocumentPayload> {
  validateDocumentFile(file)
  const bytes = new Uint8Array(await file.arrayBuffer())

  return {
    name: file.name,
    mimeType: file.name.toLowerCase().endsWith('.pdf')
      ? 'application/pdf'
      : 'text/plain',
    data: toBase64(bytes),
  }
}

async function postJson<ResponseBody>(
  endpoint: string,
  body: DocumentPayload | AskRequest,
): Promise<ResponseBody> {
  const controller = new AbortController()
  const timeout = window.setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS)

  try {
    const response = await fetch(endpoint, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(body),
      signal: controller.signal,
    })

    const payload = (await response.json().catch(() => null)) as unknown

    if (!response.ok) {
      const apiError = payload as Partial<ApiErrorResponse> | null
      throw new ClientApiError(
        apiError?.error?.message ?? 'The request could not be completed.',
        apiError?.error?.code ?? 'REQUEST_FAILED',
        response.status,
      )
    }

    if (!payload || typeof payload !== 'object') {
      throw new ClientApiError('The server returned an invalid response.', 'INVALID_RESPONSE')
    }

    return payload as ResponseBody
  } catch (error) {
    if (error instanceof ClientApiError) {
      throw error
    }

    if (error instanceof DOMException && error.name === 'AbortError') {
      throw new ClientApiError(
        'The request took too long. Please try a smaller document.',
        'REQUEST_TIMEOUT',
      )
    }

    throw new ClientApiError(
      'Could not reach the analysis service. Check your connection and try again.',
      'NETWORK_ERROR',
    )
  } finally {
    window.clearTimeout(timeout)
  }
}

export async function analyzeDocument(
  documentPayload: DocumentPayload,
): Promise<AnalysisResult> {
  const response = await postJson<{ analysis: AnalysisResult }>('/api/analyze', documentPayload)
  return response.analysis
}

export async function askDocument(
  documentPayload: DocumentPayload,
  question: string,
  history: NonNullable<AskRequest['history']>,
): Promise<AskResult> {
  const response = await postJson<{ answer: AskResult }>('/api/ask', {
    ...documentPayload,
    question,
    history,
  })
  return response.answer
}

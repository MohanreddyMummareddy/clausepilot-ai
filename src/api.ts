import type {
  AnalysisResult,
  ApiErrorResponse,
  AskRequest,
  AskResult,
  Citation,
  DocumentPayload,
  KeyTerm,
  RiskFlag,
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

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function isNonEmptyString(value: unknown): value is string {
  return typeof value === 'string' && value.trim().length > 0
}

function isCitation(value: unknown): value is Citation {
  if (!isRecord(value) || !isNonEmptyString(value.excerpt) || !isRecord(value.locator)) {
    return false
  }

  if (value.locator.kind === 'pdf_page') {
    return Number.isInteger(value.locator.page) && (value.locator.page as number) >= 1
  }

  return (
    value.locator.kind === 'text_section' &&
    isNonEmptyString(value.locator.section) &&
    Number.isInteger(value.locator.lineStart) &&
    Number.isInteger(value.locator.lineEnd) &&
    (value.locator.lineStart as number) >= 1 &&
    (value.locator.lineEnd as number) >= (value.locator.lineStart as number)
  )
}

function isCitationList(value: unknown): value is Citation[] {
  return Array.isArray(value) && value.every(isCitation)
}

function isKeyTerm(value: unknown): value is KeyTerm {
  return (
    isRecord(value) &&
    isNonEmptyString(value.term) &&
    isNonEmptyString(value.value) &&
    isNonEmptyString(value.explanation) &&
    isCitationList(value.citations)
  )
}

function isRiskFlag(value: unknown): value is RiskFlag {
  return (
    isRecord(value) &&
    isNonEmptyString(value.title) &&
    isNonEmptyString(value.category) &&
    (value.severity === 'low' || value.severity === 'medium' || value.severity === 'high') &&
    isNonEmptyString(value.whyItMatters) &&
    isNonEmptyString(value.suggestedNextStep) &&
    isCitationList(value.citations)
  )
}

function assertAnalysisResult(value: unknown): asserts value is AnalysisResult {
  if (
    !isRecord(value) ||
    !isNonEmptyString(value.documentTitle) ||
    !isNonEmptyString(value.documentType) ||
    !isRecord(value.summary) ||
    !isNonEmptyString(value.summary.text) ||
    !isCitationList(value.summary.citations) ||
    !Array.isArray(value.keyTerms) ||
    !value.keyTerms.every(isKeyTerm) ||
    !Array.isArray(value.riskFlags) ||
    !value.riskFlags.every(isRiskFlag) ||
    !Array.isArray(value.suggestedQuestions) ||
    !value.suggestedQuestions.every(isNonEmptyString)
  ) {
    throw new ClientApiError('The server returned an invalid response.', 'INVALID_RESPONSE', 502)
  }
}

function assertAskResult(value: unknown): asserts value is AskResult {
  if (
    !isRecord(value) ||
    !isNonEmptyString(value.answer) ||
    typeof value.notFound !== 'boolean' ||
    !isCitationList(value.citations) ||
    !Array.isArray(value.followUpQuestions) ||
    !value.followUpQuestions.every(isNonEmptyString)
  ) {
    throw new ClientApiError('The server returned an invalid response.', 'INVALID_RESPONSE', 502)
  }
}

function assertAnalysisEnvelope(value: unknown): asserts value is { analysis: AnalysisResult } {
  if (!isRecord(value) || !isRecord(value.analysis)) {
    throw new ClientApiError('The server returned an invalid response.', 'INVALID_RESPONSE', 502)
  }
  assertAnalysisResult(value.analysis)
}

function assertAskEnvelope(value: unknown): asserts value is { answer: AskResult } {
  if (!isRecord(value) || !isRecord(value.answer)) {
    throw new ClientApiError('The server returned an invalid response.', 'INVALID_RESPONSE', 502)
  }
  assertAskResult(value.answer)
}

export function validateDocumentFile(file: File): void {
  const lowerName = file.name.toLowerCase()
  const hasAllowedExtension = ALLOWED_EXTENSIONS.some((extension) =>
    lowerName.endsWith(extension),
  )

  if (!hasAllowedExtension) {
    throw new ClientApiError('Choose a PDF, TXT, or Markdown document.', 'UNSUPPORTED_FILE')
  }

  if (file.type) {
    const mimeType = file.type.toLowerCase()
    const isPdf = lowerName.endsWith('.pdf')
    const isText = lowerName.endsWith('.txt') || lowerName.endsWith('.md')
    if (!ALLOWED_MIME_TYPES.has(mimeType) || (isPdf && mimeType !== 'application/pdf')) {
      throw new ClientApiError('Choose a PDF, TXT, or Markdown document.', 'UNSUPPORTED_FILE')
    }
    if (isText && mimeType !== 'text/plain' && mimeType !== 'text/markdown') {
      throw new ClientApiError('Choose a PDF, TXT, or Markdown document.', 'UNSUPPORTED_FILE')
    }
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
  validate: (value: unknown) => asserts value is ResponseBody,
): Promise<ResponseBody> {
  const controller = new AbortController()
  const timeout = globalThis.setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS)

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

    validate(payload)
    return payload
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
    globalThis.clearTimeout(timeout)
  }
}

export async function analyzeDocument(
  documentPayload: DocumentPayload,
): Promise<AnalysisResult> {
  const response = await postJson<{ analysis: AnalysisResult }>(
    '/api/analyze',
    documentPayload,
    assertAnalysisEnvelope,
  )
  return response.analysis
}

export async function askDocument(
  documentPayload: DocumentPayload,
  question: string,
  history: NonNullable<AskRequest['history']>,
): Promise<AskResult> {
  const response = await postJson<{ answer: AskResult }>(
    '/api/ask',
    {
      ...documentPayload,
      question,
      history,
    },
    assertAskEnvelope,
  )
  return response.answer
}

export type Severity = 'low' | 'medium' | 'high'

export type CitationLocator =
  | {
      kind: 'pdf_page'
      page: number
    }
  | {
      kind: 'text_section'
      section: string
      lineStart: number
      lineEnd: number
    }

export interface Citation {
  excerpt: string
  locator: CitationLocator
}

export interface KeyTerm {
  term: string
  value: string
  explanation: string
  citations: Citation[]
}

export interface RiskFlag {
  title: string
  category: string
  severity: Severity
  whyItMatters: string
  suggestedNextStep: string
  citations: Citation[]
}

export interface AnalysisResult {
  documentTitle: string
  documentType: string
  summary: {
    text: string
    citations: Citation[]
  }
  keyTerms: KeyTerm[]
  riskFlags: RiskFlag[]
  suggestedQuestions: string[]
}

export interface AskResult {
  answer: string
  citations: Citation[]
  notFound: boolean
  followUpQuestions: string[]
}

export interface DocumentPayload {
  name: string
  mimeType: 'application/pdf' | 'text/plain'
  data: string
}

export interface AnalyzeRequest extends DocumentPayload {}

export interface AskRequest extends DocumentPayload {
  question: string
  history?: Array<{
    role: 'user' | 'assistant'
    content: string
  }>
}

export interface ApiErrorResponse {
  error: {
    code: string
    message: string
  }
}

import './style.css'
import {
  ClientApiError,
  analyzeDocument,
  askDocument,
  createDocumentPayload,
} from './api'
import type {
  AnalysisResult,
  AskResult,
  Citation,
  DocumentPayload,
  Severity,
} from './types'

interface SelectedFile {
  name: string
  size: number
}

interface ChatMessage {
  id: number
  role: 'user' | 'assistant'
  content: string
  citations?: Citation[]
  notFound?: boolean
}

type AppView = 'upload' | 'loading' | 'results'

interface AppState {
  view: AppView
  selectedFile: SelectedFile | null
  document: DocumentPayload | null
  analysis: AnalysisResult | null
  messages: ChatMessage[]
  isAsking: boolean
  isPreparingSample: boolean
  loadingStep: number
  error: string
}

function getApplicationRoot(): HTMLDivElement {
  const element = document.querySelector<HTMLDivElement>('#app')
  if (!element) {
    throw new Error('Application root was not found.')
  }
  return element
}

const root = getApplicationRoot()

const loadingSteps = [
  'Reading the document structure',
  'Extracting important terms',
  'Checking clauses for practical concerns',
  'Linking every insight to its source',
]

const state: AppState = {
  view: 'upload',
  selectedFile: null,
  document: null,
  analysis: null,
  messages: [],
  isAsking: false,
  isPreparingSample: false,
  loadingStep: 0,
  error: '',
}

let loadingTimer: number | undefined
let selectionSequence = 0
let sampleSequence = 0
let messageSequence = 0
let requestEpoch = 0

const escapeCharacters: Record<string, string> = {
  '&': '&amp;',
  '<': '&lt;',
  '>': '&gt;',
  '"': '&quot;',
  "'": '&#039;',
}

function escapeHtml(value: unknown): string {
  return String(value).replace(/[&<>"']/g, (character) => escapeCharacters[character] ?? character)
}

function formatFileSize(size: number): string {
  if (size < 1024) {
    return `${size} B`
  }
  if (size < 1024 * 1024) {
    return `${(size / 1024).toFixed(1)} KB`
  }
  return `${(size / (1024 * 1024)).toFixed(1)} MB`
}

function fileTypeLabel(name: string): string {
  const lowerName = name.toLowerCase()
  if (lowerName.endsWith('.pdf')) {
    return 'PDF'
  }
  if (lowerName.endsWith('.md')) {
    return 'MD'
  }
  return 'TXT'
}

function citationLabel(citation: Citation): string {
  if (citation.locator.kind === 'pdf_page') {
    return `Page ${citation.locator.page}`
  }

  const { section, lineStart, lineEnd } = citation.locator
  const lines = lineStart === lineEnd ? `Line ${lineStart}` : `Lines ${lineStart}–${lineEnd}`
  return section === '[unheaded]' ? lines : `${section} · ${lines}`
}

function renderCitations(citations: Citation[]): string {
  if (citations.length === 0) {
    return ''
  }

  return `
    <div class="citation-list" aria-label="Source citations">
      ${citations
        .map(
          (citation, index) => `
            <details class="citation">
              <summary>
                <span class="citation-number">${index + 1}</span>
                <span>${escapeHtml(citationLabel(citation))}</span>
                <span class="citation-action">View source</span>
              </summary>
              <blockquote>${escapeHtml(citation.excerpt)}</blockquote>
            </details>
          `,
        )
        .join('')}
    </div>
  `
}

function severityLabel(severity: Severity): string {
  return severity.charAt(0).toUpperCase() + severity.slice(1)
}

function uniqueCitationCount(analysis: AnalysisResult): number {
  const citations = [
    ...analysis.summary.citations,
    ...analysis.keyTerms.flatMap((item) => item.citations),
    ...analysis.riskFlags.flatMap((item) => item.citations),
  ]
  return new Set(citations.map((citation) => JSON.stringify(citation))).size
}

function renderHeader(): string {
  return `
    <header class="site-header">
      <button class="brand brand-button" type="button" data-action="home" aria-label="ClausePilot AI home">
        <span class="brand-mark" aria-hidden="true">C</span>
        <span>ClausePilot <strong>AI</strong></span>
      </button>
      <nav class="header-nav" aria-label="Primary navigation">
        ${
          state.view === 'upload'
            ? '<a href="#how-it-works">How it works</a>'
            : state.view === 'loading'
              ? '<span class="processing-label">Analysis in progress</span>'
              : '<button type="button" data-action="home">New analysis</button>'
        }
        <span class="secure-label"><span></span> Live AI analysis</span>
      </nav>
    </header>
  `
}

function renderFooter(): string {
  return `
    <footer>
      <span>ClausePilot AI</span>
      <span>Clarity before commitment.</span>
      <span>For education, not legal advice.</span>
    </footer>
  `
}

function renderError(): string {
  if (!state.error) {
    return ''
  }

  return `
    <div class="error-banner" role="alert">
      <span class="error-symbol" aria-hidden="true">!</span>
      <p>${escapeHtml(state.error)}</p>
      <button type="button" data-action="dismiss-error" aria-label="Dismiss error">×</button>
    </div>
  `
}

function renderUploadView(): string {
  const selectedFile = state.selectedFile
  return `
    <section class="hero">
      <div class="hero-copy">
        <p class="eyebrow"><span></span> Legal assistance made understandable</p>
        <h1>Know what you’re <em>agreeing to.</em></h1>
        <p class="hero-text">
          Turn a dense agreement into a clear, practical brief. See important terms,
          understand clauses that may need attention, and trace every answer to the source.
        </p>
        <div class="trust-row">
          <span><i>✓</i> Plain-language summary</span>
          <span><i>✓</i> Source-linked answers</span>
        </div>
      </div>

      <div class="upload-panel">
        <div class="panel-heading">
          <div>
            <span class="panel-kicker">Document analysis</span>
            <h2>Start with your agreement</h2>
          </div>
          <span class="ai-badge"><i></i> Gemini powered</span>
        </div>

        <form id="analysis-form" novalidate>
          <div id="dropzone" class="dropzone ${selectedFile ? 'has-file' : ''}">
            <input id="file-input" type="file" accept=".pdf,.txt,.md,application/pdf,text/plain,text/markdown" hidden />
            <span class="upload-icon" aria-hidden="true">
              <svg viewBox="0 0 24 24" fill="none">
                <path d="M12 16V4m0 0L7.5 8.5M12 4l4.5 4.5M5 14v4a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2v-4" />
              </svg>
            </span>
            <strong>${selectedFile ? 'Document ready' : 'Drop your document here'}</strong>
            <span>${selectedFile ? 'Choose another file to replace it' : 'or select a PDF, TXT, or Markdown file'}</span>
            <button class="secondary-button" type="button" data-action="choose-file">
              ${selectedFile ? 'Change file' : 'Choose file'}
            </button>
          </div>

          <button
            class="sample-button"
            type="button"
            data-action="use-sample"
            ${state.isPreparingSample ? 'disabled' : ''}
          >
            ${state.isPreparingSample ? 'Loading sample…' : 'No document handy? Try a sample agreement'}
            <span aria-hidden="true">↗</span>
          </button>

          ${
            selectedFile
              ? `
                <div class="selected-file">
                  <span class="file-type">${fileTypeLabel(selectedFile.name)}</span>
                  <div>
                    <strong>${escapeHtml(selectedFile.name)}</strong>
                    <span>${formatFileSize(selectedFile.size)} · ready to analyze</span>
                  </div>
                  <button type="button" data-action="clear-file" aria-label="Remove selected document">×</button>
                </div>
              `
              : ''
          }

          <button
            class="primary-button analyze-button"
            type="submit"
            ${selectedFile ? '' : 'disabled'}
          >
            Analyze document <span aria-hidden="true">→</span>
          </button>
        </form>

        <div class="upload-footnote">
          <span>Maximum 3 MB</span>
          <span>No application storage</span>
        </div>
      </div>
    </section>

    <section class="feature-strip" id="how-it-works">
      <article>
        <span>01</span>
        <h2>Upload</h2>
        <p>Choose the agreement you want to understand before signing or responding.</p>
      </article>
      <article>
        <span>02</span>
        <h2>Analyze</h2>
        <p>Gemini extracts key terms, obligations, dates, and supported potential risks.</p>
      </article>
      <article>
        <span>03</span>
        <h2>Verify</h2>
        <p>Open source excerpts and ask focused questions grounded in the same document.</p>
      </article>
    </section>

    <section class="safety-note">
      <span class="safety-icon" aria-hidden="true">i</span>
      <div>
        <strong>Built for understanding, not replacing legal advice.</strong>
        <p>ClausePilot can make mistakes. Verify important clauses with the source and a qualified professional.</p>
      </div>
    </section>
  `
}

function renderProcessingView(): string {
  const file = state.selectedFile
  return `
    <section class="processing-view" aria-live="polite">
      <div class="processing-card">
        <div class="ai-orbit" aria-hidden="true">
          <span class="orbit orbit-one"></span>
          <span class="orbit orbit-two"></span>
          <span class="orbit-core">C</span>
        </div>
        <p class="panel-kicker">Live document analysis</p>
        <h1>Reading between the legal lines.</h1>
        <p class="processing-description">
          Gemini is analyzing <strong>${escapeHtml(file?.name ?? 'your document')}</strong>
          and grounding each finding in the source.
        </p>

        <ol class="processing-steps">
          ${loadingSteps
            .map(
              (step, index) => `
                <li class="${index < state.loadingStep ? 'complete' : ''} ${index === state.loadingStep ? 'active' : ''}">
                  <span>${index < state.loadingStep ? '✓' : index + 1}</span>
                  <p>${escapeHtml(step)}</p>
                </li>
              `,
            )
            .join('')}
        </ol>
      </div>
    </section>
  `
}

function renderSummary(analysis: AnalysisResult): string {
  return `
    <section class="result-section summary-section">
      <div class="section-heading">
        <div>
          <span class="section-kicker">Overview</span>
          <h2>Plain-language summary</h2>
        </div>
        <span class="grounded-badge"><i></i> Grounded in your document</span>
      </div>
      <div class="summary-card">
        <p>${escapeHtml(analysis.summary.text)}</p>
        ${renderCitations(analysis.summary.citations)}
      </div>
    </section>
  `
}

function renderRisks(analysis: AnalysisResult): string {
  return `
    <section class="result-section" id="risk-flags">
      <div class="section-heading">
        <div>
          <span class="section-kicker">Review points</span>
          <h2>Clauses to look at closely</h2>
        </div>
        <span class="count-badge">${analysis.riskFlags.length} ${analysis.riskFlags.length === 1 ? 'item' : 'items'}</span>
      </div>

      ${
        analysis.riskFlags.length
          ? `<div class="risk-list">${analysis.riskFlags
              .map(
                (risk) => `
                  <article class="risk-card risk-${escapeHtml(risk.severity)}">
                    <div class="risk-card-top">
                      <span class="severity-badge">${escapeHtml(severityLabel(risk.severity))}</span>
                      <span class="risk-category">${escapeHtml(risk.category)}</span>
                    </div>
                    <h3>${escapeHtml(risk.title)}</h3>
                    <p>${escapeHtml(risk.whyItMatters)}</p>
                    <div class="next-step">
                      <strong>Suggested next step</strong>
                      <p>${escapeHtml(risk.suggestedNextStep)}</p>
                    </div>
                    ${renderCitations(risk.citations)}
                  </article>
                `,
              )
              .join('')}</div>`
          : `
            <div class="empty-state">
              <span aria-hidden="true">✓</span>
              <div>
                <h3>No supported risk flags found</h3>
                <p>The analysis did not identify a specific concern that could be grounded in the supplied text.</p>
              </div>
            </div>
          `
      }
    </section>
  `
}

function renderKeyTerms(analysis: AnalysisResult): string {
  return `
    <section class="result-section" id="key-terms">
      <div class="section-heading">
        <div>
          <span class="section-kicker">Important details</span>
          <h2>Key terms</h2>
        </div>
        <span class="count-badge">${analysis.keyTerms.length} extracted</span>
      </div>

      ${
        analysis.keyTerms.length
          ? `<div class="term-grid">${analysis.keyTerms
              .map(
                (item) => `
                  <article class="term-card">
                    <span class="term-label">${escapeHtml(item.term)}</span>
                    <strong>${escapeHtml(item.value)}</strong>
                    <p>${escapeHtml(item.explanation)}</p>
                    ${renderCitations(item.citations)}
                  </article>
                `,
              )
              .join('')}</div>`
          : `
            <div class="empty-state">
              <span aria-hidden="true">—</span>
              <div>
                <h3>No structured terms extracted</h3>
                <p>Try asking a focused question about a particular clause in the document.</p>
              </div>
            </div>
          `
      }
    </section>
  `
}

function renderChatMessage(message: ChatMessage): string {
  if (message.role === 'user') {
    return `
      <article class="chat-message user-message">
        <span class="message-avatar" aria-hidden="true">You</span>
        <div>
          <p>${escapeHtml(message.content)}</p>
        </div>
      </article>
    `
  }

  return `
    <article class="chat-message assistant-message">
      <span class="message-avatar" aria-hidden="true">C</span>
      <div>
        <span class="message-label">ClausePilot</span>
        <p>${escapeHtml(message.content)}</p>
        ${message.notFound ? '<span class="not-found-badge">Not stated in this document</span>' : ''}
        ${renderCitations(message.citations ?? [])}
      </div>
    </article>
  `
}

function renderAskSection(analysis: AnalysisResult): string {
  return `
    <section class="result-section ask-section" id="ask-document">
      <div class="section-heading">
        <div>
          <span class="section-kicker">Document Q&amp;A</span>
          <h2>Ask a focused question</h2>
        </div>
        <span class="grounded-badge"><i></i> Answers use the uploaded document</span>
      </div>

      <div class="chat-card">
        ${
          state.messages.length
            ? `<div class="chat-messages" id="chat-messages">${state.messages.map(renderChatMessage).join('')}</div>`
            : `
              <div class="chat-empty">
                <span class="chat-empty-icon" aria-hidden="true">?</span>
                <h3>What do you want to understand?</h3>
                <p>Choose a prompt or write your own question about ${escapeHtml(analysis.documentTitle)}.</p>
              </div>
            `
        }

        ${
          state.messages.length === 0
            ? `<div class="suggested-questions">${analysis.suggestedQuestions
                .slice(0, 4)
                .map(
                  (question, index) => `
                    <button type="button" data-suggested-index="${index}">
                      <span>${escapeHtml(question)}</span><i aria-hidden="true">↗</i>
                    </button>
                  `,
                )
                .join('')}</div>`
            : ''
        }

        ${
          state.isAsking
            ? '<div class="typing-indicator" aria-label="ClausePilot is answering"><span></span><span></span><span></span></div>'
            : ''
        }

        <form id="question-form" class="question-form">
          <label for="question-input">Your question</label>
          <div>
            <textarea
              id="question-input"
              name="question"
              rows="2"
              maxlength="500"
              placeholder="For example: What notice period applies to termination?"
              ${state.isAsking ? 'disabled' : ''}
              required
            ></textarea>
            <button class="primary-button" type="submit" ${state.isAsking ? 'disabled' : ''}>
              ${state.isAsking ? 'Thinking…' : 'Ask'}
              <span aria-hidden="true">→</span>
            </button>
          </div>
          <p>AI can make mistakes. Check the cited source before making a decision.</p>
        </form>
      </div>
    </section>
  `
}

function renderResultsView(): string {
  const analysis = state.analysis
  const file = state.selectedFile
  if (!analysis || !file) {
    return renderUploadView()
  }

  return `
    <section class="results-header">
      <div class="results-file-icon" aria-hidden="true">${fileTypeLabel(file.name)}</div>
      <div class="results-title">
        <div class="results-meta">
          <span>${escapeHtml(analysis.documentType)}</span>
          <span class="analysis-complete"><i></i> Analysis complete</span>
        </div>
        <h1>${escapeHtml(analysis.documentTitle)}</h1>
        <p>${escapeHtml(file.name)} · ${formatFileSize(file.size)} · ${uniqueCitationCount(analysis)} source links</p>
      </div>
      <button class="secondary-button new-analysis-button" type="button" data-action="home">
        Analyze another
      </button>
    </section>

    <nav class="results-nav" aria-label="Analysis sections">
      <a href="#overview">Overview</a>
      <a href="#risk-flags">Review points</a>
      <a href="#key-terms">Key terms</a>
      <a href="#ask-document">Ask a question</a>
    </nav>

    <div id="overview">${renderSummary(analysis)}</div>
    ${renderRisks(analysis)}
    ${renderKeyTerms(analysis)}
    ${renderAskSection(analysis)}
  `
}

function render(): void {
  const view =
    state.view === 'loading'
      ? renderProcessingView()
      : state.view === 'results'
        ? renderResultsView()
        : renderUploadView()

  document.title =
    state.view === 'results'
      ? `${state.analysis?.documentTitle ?? 'Analysis'} — ClausePilot AI`
      : 'ClausePilot AI — Understand before you sign'

  root.innerHTML = `
    ${renderHeader()}
    <main>${view}</main>
    ${renderError()}
    ${renderFooter()}
  `

  bindEvents()
}

function clearState(): void {
  requestEpoch += 1
  selectionSequence += 1
  sampleSequence += 1
  state.view = 'upload'
  state.selectedFile = null
  state.document = null
  state.analysis = null
  state.messages = []
  state.isAsking = false
  state.isPreparingSample = false
  state.loadingStep = 0
  state.error = ''
  window.clearInterval(loadingTimer)
  loadingTimer = undefined
}

async function selectFile(
  file: File | undefined,
  options: { fromSample?: boolean } = {},
): Promise<void> {
  if (!file) {
    return
  }

  if (!options.fromSample) {
    sampleSequence += 1
    state.isPreparingSample = false
  }

  const sequence = ++selectionSequence
  state.error = ''

  try {
    const documentPayload = await createDocumentPayload(file)
    if (sequence !== selectionSequence) {
      return
    }

    state.selectedFile = { name: file.name, size: file.size }
    state.document = documentPayload
    render()
  } catch (error) {
    if (sequence !== selectionSequence) {
      return
    }

    state.selectedFile = null
    state.document = null
    state.error =
      error instanceof ClientApiError
        ? error.message
        : 'The selected document could not be read.'
    render()
  }
}

async function loadSampleAgreement(): Promise<void> {
  if (state.isPreparingSample) {
    return
  }

  const sequence = ++sampleSequence
  state.isPreparingSample = true
  state.error = ''
  render()

  try {
    const response = await fetch('/sample-agreement.txt', { cache: 'no-store' })
    if (!response.ok) {
      throw new Error('Sample document request failed.')
    }

    const file = new File([await response.blob()], 'sample-services-agreement.txt', {
      type: 'text/plain',
    })
    if (sequence !== sampleSequence) {
      return
    }

    await selectFile(file, { fromSample: true })
  } catch {
    if (sequence === sampleSequence) {
      state.error = 'The sample agreement could not be loaded. Please try again.'
    }
  } finally {
    if (sequence === sampleSequence) {
      state.isPreparingSample = false
      render()
    }
  }
}

async function startAnalysis(): Promise<void> {
  if (!state.document || !state.selectedFile) {
    state.error = 'Choose a document before starting the analysis.'
    render()
    return
  }

  state.view = 'loading'
  state.loadingStep = 0
  state.error = ''
  const epoch = requestEpoch
  render()
  window.scrollTo({ top: 0, behavior: 'smooth' })

  loadingTimer = window.setInterval(() => {
    state.loadingStep = Math.min(state.loadingStep + 1, loadingSteps.length - 1)
    render()
  }, 2600)

  try {
    const analysis = await analyzeDocument(state.document)
    if (epoch !== requestEpoch) {
      return
    }
    state.analysis = analysis
    state.messages = []
    state.view = 'results'
    render()
  } catch (error) {
    state.view = 'upload'
    state.error =
      error instanceof ClientApiError
        ? error.message
        : 'The document could not be analyzed. Please try again.'
  } finally {
    window.clearInterval(loadingTimer)
    loadingTimer = undefined
    render()
    window.scrollTo({ top: 0, behavior: 'smooth' })
  }
}

function scrollToChat(): void {
  window.requestAnimationFrame(() => {
    document.querySelector('#chat-messages')?.scrollIntoView({
      behavior: 'smooth',
      block: 'center',
    })
  })
}

async function submitQuestion(question: string): Promise<void> {
  const cleanQuestion = question.trim()
  if (!cleanQuestion || state.isAsking || !state.document) {
    return
  }

  const history = state.messages.slice(-6).map(({ role, content }) => ({ role, content }))
  const epoch = requestEpoch
  state.messages.push({
    id: ++messageSequence,
    role: 'user',
    content: cleanQuestion,
  })
  state.isAsking = true
  state.error = ''
  render()
  scrollToChat()

  try {
    const answer: AskResult = await askDocument(state.document, cleanQuestion, history)
    if (epoch !== requestEpoch) {
      return
    }
    state.messages.push({
      id: ++messageSequence,
      role: 'assistant',
      content: answer.answer,
      citations: answer.citations,
      notFound: answer.notFound,
    })
  } catch (error) {
    state.error =
      error instanceof ClientApiError
        ? error.message
        : 'The question could not be answered. Please try again.'
  } finally {
    state.isAsking = false
    render()
    scrollToChat()
  }
}

function bindEvents(): void {
  root.querySelectorAll<HTMLElement>('[data-action="home"]').forEach((element) => {
    element.addEventListener('click', () => {
      clearState()
      render()
      window.scrollTo({ top: 0, behavior: 'smooth' })
    })
  })

  root.querySelector('[data-action="dismiss-error"]')?.addEventListener('click', () => {
    state.error = ''
    render()
  })

  const chooseButton = root.querySelector<HTMLButtonElement>('[data-action="choose-file"]')
  const fileInput = root.querySelector<HTMLInputElement>('#file-input')
  const dropzone = root.querySelector<HTMLElement>('#dropzone')

  chooseButton?.addEventListener('click', () => fileInput?.click())
  root.querySelector('[data-action="use-sample"]')?.addEventListener('click', () => {
    void loadSampleAgreement()
  })
  fileInput?.addEventListener('change', () => {
    void selectFile(fileInput.files?.[0])
  })

  dropzone?.addEventListener('dragover', (event) => {
    event.preventDefault()
    dropzone.classList.add('dragging')
  })
  dropzone?.addEventListener('dragleave', () => dropzone.classList.remove('dragging'))
  dropzone?.addEventListener('drop', (event) => {
    event.preventDefault()
    dropzone.classList.remove('dragging')
    void selectFile(event.dataTransfer?.files[0])
  })

  root.querySelector('[data-action="clear-file"]')?.addEventListener('click', () => {
    requestEpoch += 1
    selectionSequence += 1
    sampleSequence += 1
    state.isPreparingSample = false
    state.selectedFile = null
    state.document = null
    state.error = ''
    render()
  })

  root
    .querySelector<HTMLFormElement>('#analysis-form')
    ?.addEventListener('submit', (event) => {
      event.preventDefault()
      void startAnalysis()
    })

  root.querySelector<HTMLFormElement>('#question-form')?.addEventListener('submit', (event) => {
    event.preventDefault()
    const input = root.querySelector<HTMLTextAreaElement>('#question-input')
    if (input) {
      void submitQuestion(input.value)
    }
  })

  root.querySelectorAll<HTMLButtonElement>('[data-suggested-index]').forEach((button) => {
    button.addEventListener('click', () => {
      const index = Number(button.dataset.suggestedIndex)
      const question = state.analysis?.suggestedQuestions[index]
      if (question) {
        void submitQuestion(question)
      }
    })
  })
}

render()

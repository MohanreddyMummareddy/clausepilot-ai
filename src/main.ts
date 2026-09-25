import './style.css'

document.querySelector<HTMLDivElement>('#app')!.innerHTML = `
  <header class="site-header">
    <a class="brand" href="#" aria-label="ClausePilot AI home">
      <span class="brand-mark" aria-hidden="true">C</span>
      <span>ClausePilot <strong>AI</strong></span>
    </a>
    <span class="header-note">Built for people, not legal jargon</span>
  </header>

  <main>
    <section class="hero">
      <div class="hero-copy">
        <p class="eyebrow"><span></span> Legal assistance made understandable</p>
        <h1>Know what you’re <em>agreeing to.</em></h1>
        <p class="hero-text">
          Turn complex agreements into a clear, practical brief. Surface important terms,
          understand risky clauses, and get answers grounded in your own document.
        </p>
        <div class="hero-actions">
          <a class="primary-button" href="#analyze">Analyze a document <span aria-hidden="true">→</span></a>
          <span class="privacy-note">Private by design · Not legal advice</span>
        </div>
      </div>

      <div class="document-card" aria-label="Example contract analysis">
        <div class="document-topbar">
          <span class="document-icon" aria-hidden="true">PDF</span>
          <div>
            <strong>Service Agreement</strong>
            <span>6 pages · analyzed</span>
          </div>
          <span class="status">Ready</span>
        </div>
        <div class="summary-block">
          <span class="section-label">Plain-language summary</span>
          <p>The agreement lasts 12 months and renews automatically unless notice is given 30 days before renewal.</p>
        </div>
        <div class="risk-heading">
          <span class="section-label">What needs attention</span>
          <span>2 items</span>
        </div>
        <div class="risk-item high">
          <span class="risk-icon" aria-hidden="true">!</span>
          <div>
            <strong>Automatic renewal</strong>
            <p>Missing the notice window may extend the agreement for another year.</p>
          </div>
        </div>
        <div class="risk-item medium">
          <span class="risk-icon" aria-hidden="true">↗</span>
          <div>
            <strong>Broad liability</strong>
            <p>One clause limits liability for several categories of loss.</p>
          </div>
        </div>
        <div class="source-note">
          <span aria-hidden="true">§</span>
          Every insight links back to the exact clause
        </div>
      </div>
    </section>

    <section class="feature-strip" id="analyze">
      <article>
        <span>01</span>
        <h2>Plain English</h2>
        <p>Understand obligations, dates, payments, and termination terms at a glance.</p>
      </article>
      <article>
        <span>02</span>
        <h2>Risk, explained</h2>
        <p>Spot clauses that may need attention, with practical context instead of legal jargon.</p>
      </article>
      <article>
        <span>03</span>
        <h2>Grounded answers</h2>
        <p>Ask questions and trace every answer back to the source text.</p>
      </article>
    </section>
  </main>

  <footer>
    <span>ClausePilot AI</span>
    <span>Clarity before commitment.</span>
  </footer>
`

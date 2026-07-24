# Dataset register

This register records the assumptions used by SupportIQ. It is not legal advice. The pipeline checks Hugging Face builder configurations and fields at download time and stores a fresh metadata manifest, because dataset cards can change.

## PolyAI/banking77

- Purpose: 77-way banking intent classification.
- Fields consumed: `text`, `label`; class names come from the dataset `ClassLabel` feature rather than a hardcoded ordering.
- Dataset card license: CC BY 4.0.
- Risks: narrow English banking domain, short single-turn queries, class ambiguity, and a taxonomy that may not map to a company's routing rules. Customer populations and production language may differ substantially.

## cardiffnlp/tweet_eval — sentiment

- Purpose: negative / neutral / positive sentiment classification.
- Configuration: explicitly `sentiment`; fields consumed are `text` and `label`.
- Dataset card: the collection has per-subset terms; the sentiment subset lists CC BY 3.0 and requires compliance with Twitter/X terms.
- Risks: social-media text has different pragmatics, topics, spelling, and prevalence from support tickets. Sentiment is not customer satisfaction, churn risk, or urgency.

## knkarthick/samsum

- Purpose: dialogue summarization.
- Fields consumed: `dialogue`, `summary`.
- Dataset card license: CC BY-NC-ND 4.0 and research/non-commercial use.
- Risks: informal constructed messenger-style conversations, one reference summary per dialogue, and names embedded in examples. It is not a support-specific factuality benchmark. Do not assume a model trained on it can be commercially deployed; review the license with counsel.

## bitext/Bitext-customer-support-llm-chatbot-training-dataset

- Purpose: support-example retrieval for response templates.
- Fields consumed: `instruction`, `response`, plus `intent` / `category` metadata.
- Dataset card license: CDLA-Sharing 1.0.
- Risks: the card describes hybrid synthetic data. Responses can be generic or conflict with a real company's policy. Retrieved content is context for a conservative draft—not authoritative truth—and drafts require a support agent's approval.

## Data governance checklist

1. Record a source revision/hash, download date, card, license, and intended purpose.
2. Keep raw data, processed data, caches, embeddings, and weights out of Git.
3. Scan for secrets and PII before ingestion; the included regex cleanup is not sufficient production DLP.
4. Evaluate by language, channel, category, dialect, message length, and other relevant slices.
5. Define deletion, retention, and data-subject-request workflows for real ticket data.
6. Require human review for urgency escalation and all outgoing response suggestions.

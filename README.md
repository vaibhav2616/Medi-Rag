# Level 1: MediRAG — Guarded Clinical Reference Assistant

A clinical Question-Answering assistant engineered to solve two fundamental flaws in naive RAG applications:
1. **Conversational Drift**: Multi-turn dialogue queries ("How is it treated?") losing context from prior turns.
2. **Clinical Hallucination**: Speculating on out-of-domain, non-medical, or unsupported queries instead of strictly refusing when verified reference evidence is absent.

---

## Architecture

```
User Input (Conversational) + Chat History
                  │
                  ▼
┌─────────────────────────────────────────┐
│ Stage 1: Query Reformulation Chain      │
│ (create_history_aware_retriever)        │
│ Formulates standalone query from history│
└─────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────┐
│ Dense Vector Search (ChromaDB)          │
│ Embeddings: BAAI/bge-small-en-v1.5      │
│ Top-K clinical evidence retrieval       │
└─────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────┐
│ Stage 2: Anti-Hallucination QA Chain    │
│ (create_stuff_documents_chain)          │
│ LLM: Groq Llama-3.1-8B-Instant          │
└─────────────────────────────────────────┘
                  │
        ┌─────────┴─────────┐
        ▼                   ▼
 [Evidence Found]    [Insufficient Evidence]
        │                   │
        ▼                   ▼
 Grounded Response   Strict Negative Rejection:
 + Citation Badges   "I am sorry, but based on
 (Focus, QType, NIH) the available clinical..."
```

---

## Dataset: MedQuAD
- **Source**: [MedQuAD on Hugging Face (`keivalya/MedQuad-MedicalQnADataset`)](https://huggingface.co/datasets/keivalya/MedQuad-MedicalQnADataset)
- **Origin**: 16,407 medical questions and answers extracted from trusted NIH/CDC institutes (National Cancer Institute, NHLBI, NIDDK, MedlinePlus, CDC).
- **Metadata Fields**:
  - `Focus`: Medical condition or topic (e.g. *Hypertension*, *Type 2 Diabetes*, *Asthma*)
  - `QType`: Question categorization (e.g. *symptoms*, *treatment*, *causes*, *complications*)
  - `Source`: NIH/CDC agency origin

---

## Quickstart

### 1. Configure Environment
Create a `.env` file or export your Groq API key:
```bash
GROQ_API_KEY="gsk_your_groq_api_key_here"
```

### 2. Ingest Dataset
```bash
python ingest.py --sample-size 1500
```
*Note: If offline or rate-limited by Hugging Face, `ingest.py` automatically falls back to an embedded high-fidelity NIH corpus covering primary chronic conditions.*

### 3. Run Guardrail Tests
```bash
pytest tests/test_guardrails.py -v
```
Verifies both:
- In-domain grounded medical queries produce evidence citations.
- Out-of-domain negative test queries strictly trigger the standardized refusal message without hallucination.

### 4. Launch Streamlit Web UI
```bash
streamlit run app.py
```

"""
MediRAG Ingestion Pipeline
Loads clinical records from MedQuAD (keivalya/MedQuad-MedicalQnADataset),
cleans clinical metadata (Focus, QType, Source, Answer), chunks documents with
RecursiveCharacterTextSplitter, embeds using BAAI/bge-small-en-v1.5 on CPU,
and persists to local ChromaDB.
"""

import argparse
import html
import json
import logging
import os
import re
from typing import Any, Dict, List, Optional

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from chain import (
    DEFAULT_COLLECTION_NAME,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_PERSIST_DIR,
    FallbackClinicalEmbeddings,
    get_embedding_model,
    is_hf_reachable,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("MediRAG.Ingest")

CORPUS_JSON_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "nih_medquad_clinical_corpus.json")
EXTENDED_CORPUS_JSON_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "medquad_extended_corpus.json")

PRIORITY_DISEASE_KEYWORDS = (
    "malaria", "dengue", "typhoid", "tuberculosis", "pneumonia", "cholera",
    "hepatitis", "chikungunya", "yellow fever", "meningitis", "sepsis",
    "asthma", "diabetes", "hypertension", "blood pressure", "anemia",
    "arthritis", "migraine", "glaucoma", "celiac", "thyroid", "influenza",
    "stroke", "heart", "kidney", "liver", "epilepsy", "bronchitis", "copd",
)


def load_verified_json_corpus(path: str = CORPUS_JSON_PATH) -> List[Dict[str, Any]]:
    """Loads bundled high-fidelity clinical dataset from NIH MedQuAD corpus JSON."""
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list) and len(data) > 0:
                    return data
        except Exception as e:
            logger.warning(f"Error reading {path}: {e}")
    return []

# Verified NIH MedQuAD Clinical Corpus
FALLBACK_NIH_RECORDS = load_verified_json_corpus()
if not FALLBACK_NIH_RECORDS:
    FALLBACK_NIH_RECORDS = [
        {
            "Focus": "High Blood Pressure (Hypertension)",
            "QType": "symptoms",
            "Source": "NHLBI",
            "Question": "What are the symptoms of High Blood Pressure (Hypertension)?",
            "Answer": (
                "High blood pressure (hypertension) is known as a 'silent killer' because it typically produces no warning signs or symptoms for years, "
                "even when blood pressure readings reach dangerously high levels. Most individuals with hypertension feel entirely normal. "
                "However, during a hypertensive crisis—a medical emergency where blood pressure spikes to 180/120 mm Hg or higher—patients may experience acute symptoms. "
                "These severe symptoms include severe throbbing headaches, shortness of breath (dyspnea), nosebleeds (epistaxis), severe anxiety, dizziness, "
                "lightheadedness, chest pain, irregular heartbeat (palpitations), and vision changes such as blurred vision or temporary vision loss. "
                "Because symptoms are usually absent until life-threatening organ damage begins, regular blood pressure screening is essential."
            ),
        },
        {
            "Focus": "High Blood Pressure (Hypertension)",
            "QType": "treatment",
            "Source": "NHLBI",
            "Question": "How is High Blood Pressure (Hypertension) treated?",
            "Answer": (
                "Treatment for high blood pressure (hypertension) combines lifestyle changes and prescription antihypertensive medications to achieve target blood pressure below 130/80 mm Hg. "
                "Core lifestyle interventions include: 1) Following the DASH (Dietary Approaches to Stop Hypertension) diet high in fruits, vegetables, whole grains, and low-fat dairy; "
                "2) Limiting dietary sodium intake to under 1,500–2,300 mg per day; 3) Engaging in at least 150 minutes of moderate aerobic exercise weekly; "
                "4) Maintaining a healthy weight (BMI 18.5–24.9); 5) Quitting smoking; and 6) Moderating alcohol consumption. "
                "First-line medications include Angiotensin-Converting Enzyme (ACE) inhibitors (lisinopril, enalapril), Angiotensin II Receptor Blockers (ARBs, losartan, valsartan), "
                "Calcium Channel Blockers (CCBs, amlodipine), and Thiazide Diuretics (hydrochlorothiazide, chlorthalidone). "
                "Many patients require two or more medication classes in combination for effective control."
            ),
        },
        {
            "Focus": "High Blood Pressure (Hypertension)",
            "QType": "causes",
            "Source": "NHLBI",
            "Question": "What causes High Blood Pressure (Hypertension)?",
            "Answer": (
                "High blood pressure is classified into primary (essential) hypertension and secondary hypertension. "
                "Primary hypertension develops gradually over decades without a single identifiable cause; it is driven by aging (arterial stiffening), "
                "genetic predisposition, high dietary sodium, sedentary lifestyle, obesity, chronic stress, and tobacco use. "
                "Secondary hypertension is triggered directly by an identifiable underlying medical disease or medication. "
                "Major secondary causes include chronic kidney disease, renal artery stenosis, obstructive sleep apnea, adrenal gland tumors, and certain medications."
            ),
        },
        {
            "Focus": "Type 2 Diabetes (Diabetes Mellitus)",
            "QType": "symptoms",
            "Source": "NIDDK",
            "Question": "What are the symptoms of Type 2 Diabetes?",
            "Answer": (
                "Symptoms of type 2 diabetes often develop insidiously over several years and include: 1) Increased thirst (polydipsia) and frequent urination (polyuria); "
                "2) Increased hunger (polyphagia); 3) Unexplained weight loss; 4) Persistent fatigue and lack of energy; 5) Blurred vision; "
                "6) Slow-healing cuts, sores, or recurrent infections; and 7) Tingling, numbness, or burning pain in the hands and feet (peripheral neuropathy)."
            ),
        },
        {
            "Focus": "Type 2 Diabetes (Diabetes Mellitus)",
            "QType": "treatment",
            "Source": "NIDDK",
            "Question": "What are the treatments for Type 2 Diabetes?",
            "Answer": (
                "Managing type 2 diabetes requires glycemic control through lifestyle therapy, blood glucose monitoring, and medications. "
                "Lifestyle management involves a personalized low-glycemic meal plan, 150 minutes of weekly aerobic exercise, and weight loss. "
                "Metformin is the standard first-line oral medication. Additional therapies include GLP-1 receptor agonists (semaglutide), "
                "SGLT2 inhibitors (empagliflozin, dapagliflozin), DPP-4 inhibitors, sulfonylureas, and subcutaneous insulin therapy when target HbA1c (<7.0%) is not attained."
            ),
        },
        {
            "Focus": "Asthma (Bronchial Asthma)",
            "QType": "symptoms",
            "Source": "NHLBI",
            "Question": "What are the symptoms of Asthma?",
            "Answer": (
                "Asthma is a chronic inflammatory disorder of the airways. Core symptoms include: 1) Wheezing, a musical whistling sound made while exhaling; "
                "2) Shortness of breath (dyspnea); 3) Chest tightness or squeezing; and 4) Chronic coughing, which often worsens at night or after exercise."
            ),
        },
        {
            "Focus": "Asthma (Bronchial Asthma)",
            "QType": "treatment",
            "Source": "NHLBI",
            "Question": "What are the treatments for Asthma?",
            "Answer": (
                "Asthma management utilizes long-term controllers to suppress inflammation and quick-relief rescue medications. "
                "First-line daily controllers are Inhaled Corticosteroids (fluticasone, budesonide), often combined with Long-Acting Beta2-Agonists (formoterol, salmeterol). "
                "Quick-relief rescue inhalers contain Short-Acting Beta2-Agonists (albuterol) to rapidly dilate airways during acute asthma attacks."
            ),
        },
        {
            "Focus": "Coronary Artery Disease (CAD / Heart Disease)",
            "QType": "symptoms",
            "Source": "NHLBI",
            "Question": "What are the symptoms of Coronary Artery Disease (CAD)?",
            "Answer": (
                "The hallmark symptom of coronary artery disease is angina pectoris, described as chest pain, tightness, pressure, aching, or squeezing behind the breastbone. "
                "Angina discomfort may radiate to the left shoulder, arms, neck, jaw, back, or upper abdomen. Associated symptoms include shortness of breath, "
                "profound fatigue, dizziness, nausea, cold sweats, and heart palpitations."
            ),
        },
        {
            "Focus": "Coronary Artery Disease (CAD / Heart Disease)",
            "QType": "treatment",
            "Source": "NHLBI",
            "Question": "How is Coronary Artery Disease treated?",
            "Answer": (
                "CAD management incorporates therapeutic lifestyle modifications, cardioprotective medications, and invasive revascularization. "
                "Medications include antiplatelet agents (aspirin, clopidogrel), high-intensity statins (atorvastatin), beta-blockers (metoprolol), "
                "ACE inhibitors, and sublingual nitroglycerin. In severe coronary obstruction, percutaneous coronary intervention (PCI stenting) or CABG is performed."
            ),
        },
    ]


def clean_text(text: Optional[str]) -> str:
    """Removes HTML tags, decodes HTML entities, and normalizes clinical text."""
    if not text:
        return ""
    text = html.unescape(text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([.,!?;:])", r"\1", text)
    return text.strip()


def extract_focus_from_question(question: str) -> str:
    """Infers clinical condition name from MedQuAD question phrasing if Focus is missing."""
    q_clean = re.sub(r"[?\s]+$", "", question).strip()
    patterns = [
        r"^what\s+is\s*\(are\)\s+(.+)$",
        r"^what\s+(?:are\s+the\s+symptoms|causes|are\s+the\s+treatments|is\s+the\s+outlook|is\s+the\s+treatment)\s+(?:of|for)\s+(.+)$",
        r"(?:of|for|about|in|with|prevent|diagnose|treat)\s+([A-Za-z0-9\s\-\(\)/']+)$",
    ]
    for pat in patterns:
        m = re.search(pat, q_clean, re.IGNORECASE)
        if m:
            candidate = re.sub(r"[?.,;:]+$", "", m.group(1)).strip()
            if len(candidate) > 2 and len(candidate.split()) <= 7:
                return candidate.title()
    return "General Clinical"


def load_medquad_records(sample_size: int = 2500, save_extended_cache: bool = True) -> List[Dict[str, Any]]:
    """
    Loads verified NIH MedQuAD clinical corpus and supplements with
    extended MedQuAD records from Hugging Face (keivalya/MedQuad-MedicalQnADataset)
    or cached extended corpus JSON, prioritizing infectious, tropical, and common diseases.
    """
    records: List[Dict[str, Any]] = list(FALLBACK_NIH_RECORDS)
    seen_questions = {r.get("Question", "").lower() for r in records}

    if is_hf_reachable(timeout=2.0):
        try:
            logger.info("Fetching MedQuAD clinical dataset from Hugging Face (keivalya/MedQuad-MedicalQnADataset)...")
            from datasets import load_dataset  # type: ignore

            dataset = load_dataset("keivalya/MedQuad-MedicalQnADataset", split="train")
            total = len(dataset)
            limit = total if sample_size <= 0 else min(sample_size, total)

            priority_items: List[Dict[str, Any]] = []
            other_items: List[Dict[str, Any]] = []

            for idx in range(total):
                item = dataset[idx]
                q = clean_text(item.get("Question") or item.get("question", ""))
                a = clean_text(item.get("Answer") or item.get("answer", ""))
                if not q or not a or q.lower() in seen_questions:
                    continue
                seen_questions.add(q.lower())

                raw_focus = clean_text(item.get("Focus") or item.get("focus", ""))
                if not raw_focus:
                    raw_focus = extract_focus_from_question(q)
                qtype = clean_text(item.get("QType") or item.get("qtype", "inquiry"))
                source = clean_text(item.get("Source") or item.get("source", "NIH MedQuAD"))

                rec = {
                    "Focus": raw_focus if raw_focus else "General Clinical",
                    "QType": qtype.lower() if qtype else "inquiry",
                    "Source": source.upper() if source else "NIH MEDQUAD",
                    "Question": q,
                    "Answer": a,
                }
                q_lower = q.lower()
                if any(kw in q_lower for kw in PRIORITY_DISEASE_KEYWORDS):
                    priority_items.append(rec)
                else:
                    other_items.append(rec)

            selected = priority_items[:limit]
            if len(selected) < limit:
                selected.extend(other_items[: limit - len(selected)])
            records.extend(selected)
            logger.info(
                f"Added {len(selected)} MedQuAD records ({len(priority_items)} priority disease matches)."
            )

            if save_extended_cache and len(records) > len(FALLBACK_NIH_RECORDS):
                os.makedirs(os.path.dirname(EXTENDED_CORPUS_JSON_PATH), exist_ok=True)
                with open(EXTENDED_CORPUS_JSON_PATH, "w", encoding="utf-8") as f:
                    json.dump(records, f, ensure_ascii=False, indent=2)
                logger.info(f"Saved extended clinical corpus ({len(records)} records) to {EXTENDED_CORPUS_JSON_PATH}")
        except Exception as e:
            logger.info(f"Supplemental Hugging Face dataset fetch bypassed ({type(e).__name__}: {e}).")
            cached = load_verified_json_corpus(EXTENDED_CORPUS_JSON_PATH)
            if len(cached) > len(records):
                records = cached
                logger.info(f"Loaded {len(records)} records from extended local cache.")
    else:
        cached = load_verified_json_corpus(EXTENDED_CORPUS_JSON_PATH)
        if len(cached) > len(records):
            records = cached
            logger.info(f"Hugging Face unreachable; loaded {len(records)} records from extended local cache.")
        else:
            logger.info("Hugging Face Hub is unreachable/offline. Using verified NIH clinical corpus.")

    logger.info(f"Loaded {len(records)} raw medical records.")
    return records


def build_documents(records: List[Dict[str, Any]]) -> List[Document]:
    """Converts raw medical records into LangChain Documents with standardized clinical metadata."""
    docs: List[Document] = []
    for idx, r in enumerate(records):
        focus = r.get("Focus", "Clinical General")
        qtype = r.get("QType", "inquiry")
        source = r.get("Source", "NIH")
        question = r.get("Question", "")
        answer = r.get("Answer", "")

        page_content = (
            f"Clinical Focus: {focus}\n"
            f"Question Type: {qtype}\n"
            f"Evidence Source: {source}\n"
            f"Medical Question: {question}\n"
            f"Clinical Evidence: {answer}"
        )

        metadata = {
            "focus": focus,
            "qtype": qtype,
            "source": source,
            "question": question,
            "doc_id": idx,
        }
        docs.append(Document(page_content=page_content, metadata=metadata))
    return docs


def chunk_documents(
    documents: List[Document], chunk_size: int = 1200, chunk_overlap: int = 150
) -> List[Document]:
    """Chunks clinical documents preserving metadata, clinical headers, and semantic boundaries."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    enhanced_chunks: List[Document] = []
    for c in chunks:
        content = c.page_content.strip()
        meta = c.metadata
        focus = meta.get("focus", "General Clinical")
        qtype = meta.get("qtype", "inquiry")
        source = meta.get("source", "NIH")
        if not content.startswith("Clinical Focus:"):
            header = f"Clinical Focus: {focus}\nQuestion Type: {qtype}\nEvidence Source: {source}\n"
            content = header + content
        enhanced_chunks.append(Document(page_content=content.strip(), metadata=meta))
    logger.info(f"Chunked {len(documents)} parent documents into {len(enhanced_chunks)} text chunks.")
    return enhanced_chunks


def build_and_persist_chroma(
    chunks: List[Document],
    persist_dir: str = DEFAULT_PERSIST_DIR,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    embedding_model: Optional[Any] = None,
) -> Any:
    """Builds and persists ChromaDB vector store, resetting existing collection to avoid duplication."""
    from langchain_chroma import Chroma  # type: ignore

    if embedding_model is None:
        embedding_model = get_embedding_model()

    logger.info(f"Persisting {len(chunks)} chunks to ChromaDB at '{persist_dir}'...")
    os.makedirs(persist_dir, exist_ok=True)

    try:
        existing = Chroma(
            persist_directory=persist_dir,
            embedding_function=embedding_model,
            collection_name=collection_name,
        )
        existing.delete_collection()
    except Exception:
        pass

    batch_size = 2000
    first_batch = chunks[:batch_size]
    vectorstore = Chroma.from_documents(
        documents=first_batch,
        embedding=embedding_model,
        persist_directory=persist_dir,
        collection_name=collection_name,
    )
    for start in range(batch_size, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        logger.info(f"Indexing batch {start}..{start + len(batch)} of {len(chunks)} chunks...")
        vectorstore.add_documents(batch)
    logger.info("ChromaDB vector store successfully created and indexed.")
    return vectorstore


def run_pipeline(
    sample_size: int = 2500,
    persist_dir: str = DEFAULT_PERSIST_DIR,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    chunk_size: int = 1000,
    chunk_overlap: int = 120,
) -> Any:
    """Full execution pipeline: load -> clean -> chunk -> embed -> persist."""
    records = load_medquad_records(sample_size=sample_size)
    docs = build_documents(records)
    chunks = chunk_documents(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    embeddings = get_embedding_model()
    vectorstore = build_and_persist_chroma(
        chunks=chunks,
        persist_dir=persist_dir,
        collection_name=collection_name,
        embedding_model=embeddings,
    )
    return vectorstore


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MediRAG Clinical Ingestion Pipeline")
    parser.add_argument(
        "--sample-size",
        type=int,
        default=2500,
        help="Number of records to sample (default: 2500, 0 for all)",
    )
    parser.add_argument(
        "--persist-dir",
        type=str,
        default=DEFAULT_PERSIST_DIR,
        help="Directory to persist ChromaDB index",
    )
    parser.add_argument(
        "--collection-name",
        type=str,
        default=DEFAULT_COLLECTION_NAME,
        help="Chroma collection name",
    )
    parser.add_argument("--chunk-size", type=int, default=1000, help="Document chunk size in chars")
    parser.add_argument(
        "--chunk-overlap", type=int, default=120, help="Chunk overlap in characters"
    )

    args = parser.parse_args()
    run_pipeline(
        sample_size=args.sample_size,
        persist_dir=args.persist_dir,
        collection_name=args.collection_name,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )

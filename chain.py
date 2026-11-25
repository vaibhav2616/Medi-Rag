"""
MediRAG Two-Stage LCEL Chain
Stage 1: Contextualize conversational query reformulation with create_history_aware_retriever
Stage 2: Anti-hallucination QA chain with create_stuff_documents_chain enforcing
         strict negative rejection when clinical evidence is absent, with metadata citations.
"""

import functools
import hashlib
import logging
import math
import os
import re
import urllib.request
from typing import Any, Dict, List, Optional, Set

from dotenv import load_dotenv
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from pydantic import Field

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("MediRAG.Chain")

NEGATIVE_REJECTION_MESSAGE = (
    "Out of scope: I can only answer medical and clinical questions "
    "grounded in my verified clinical knowledge base."
)

DEFAULT_PERSIST_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "medirag_chroma_db")
DEFAULT_COLLECTION_NAME = "medirag_collection"
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
DEFAULT_GROQ_MODEL = "llama-3.1-8b-instant"
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash-lite"
DEFAULT_OPENAI_MODEL = "gpt-4.1-nano"

# Comprehensive standard English stop words and conversational filler terms
ENGLISH_STOP_WORDS: Set[str] = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being",
    "below", "between", "both", "but", "by", "can", "can't", "cannot", "could",
    "couldn't", "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down",
    "during", "each", "few", "for", "from", "further", "had", "hadn't", "has",
    "hasn't", "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her",
    "here", "here's", "hers", "herself", "him", "himself", "his", "how", "how's",
    "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it",
    "it's", "its", "itself", "let's", "me", "more", "most", "mustn't", "my",
    "myself", "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other",
    "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "shan't",
    "she", "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such",
    "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then",
    "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've",
    "this", "those", "through", "to", "too", "under", "until", "up", "very", "was",
    "wasn't", "we", "we'd", "we'll", "we're", "we've", "were", "weren't", "what",
    "what's", "when", "when's", "where", "where's", "which", "while", "who", "who's",
    "whom", "why", "why's", "with", "won't", "would", "wouldn't", "you", "you'd",
    "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves",
    # Conversational / Prompt boilerplate
    "tell", "explain", "please", "know", "give", "help", "many", "much", "make",
    "take", "like", "just", "show", "read", "say", "said", "ask", "asked", "answer",
    "question", "current", "system", "instructions", "prompts", "developer", "secret",
    "pretend", "role", "ignore", "prior", "rules", "rule", "write", "repeat", "doctors"
}

# Generic clinical descriptor vocabulary that does not by itself identify a specific medical disease/entity
GENERIC_CLINICAL_TERMS: Set[str] = {
    "disease", "condition", "illness", "disorder", "syndrome", "problem",
    "symptom", "symptoms", "sign", "signs", "treatment", "treatments",
    "treated", "therapy", "therapies", "medication", "medications",
    "medicine", "medicines", "drug", "drugs", "cause", "causes",
    "caused", "cure", "cures", "curing", "prevention", "prevent",
    "preventive", "diagnose", "diagnosis", "diagnostic", "diagnosed",
    "manage", "management", "risk", "risks", "complication", "complications",
    "patient", "patients", "care", "health", "medical", "clinical",
    "evidence", "records", "reference", "inquiry"
}

CLINICAL_SYNONYM_EXPANSIONS: Dict[str, List[str]] = {
    "high blood pressure": ["high", "blood", "pressure", "hypertension", "hypertensive", "bp"],
    "hypertension": ["high", "blood", "pressure", "hypertension", "hypertensive", "bp"],
    "blood pressure": ["high", "blood", "pressure", "hypertension", "hypertensive", "bp"],
    "hypertensive": ["high", "blood", "pressure", "hypertension", "hypertensive", "bp"],
    "hypertensive crisis": ["high", "blood", "pressure", "hypertension", "hypertensive", "crisis"],
    "elevated blood pressure": ["high", "blood", "pressure", "hypertension", "hypertensive", "bp"],
    "elevated bp": ["high", "blood", "pressure", "hypertension", "hypertensive", "bp"],
    "dash diet": ["high", "blood", "pressure", "hypertension", "dash", "diet"],
    "dash": ["high", "blood", "pressure", "hypertension", "dash", "diet"],
    "essential hypertension": ["high", "blood", "pressure", "hypertension", "essential"],
    "bp": ["high", "blood", "pressure", "hypertension"],
    "type 2 diabetes": ["type", "2", "diabetes", "mellitus", "glucose", "insulin", "sugar", "t2d"],
    "type 1 diabetes": ["type", "1", "diabetes", "mellitus", "glucose", "insulin", "t1d"],
    "diabetes": ["diabetes", "mellitus", "glucose", "insulin", "sugar"],
    "hyperglycemia": ["type", "2", "diabetes", "hyperglycemia", "glucose", "insulin", "sugar"],
    "hypoglycemia": ["type", "2", "diabetes", "hypoglycemia", "glucose", "insulin", "sugar"],
    "blood sugar": ["diabetes", "glucose", "insulin", "sugar"],
    "asthma": ["asthma", "bronchial", "airways", "wheezing", "inhaler", "bronchospasm"],
    "bronchospasm": ["asthma", "bronchial", "airways", "wheezing", "inhaler", "bronchospasm"],
    "wheezing": ["asthma", "bronchial", "airways", "wheezing", "inhaler"],
    "inhaler": ["asthma", "bronchial", "inhaler", "albuterol"],
    "coronary artery disease": ["coronary", "artery", "disease", "cad", "heart", "angina"],
    "heart disease": ["coronary", "artery", "disease", "cad", "heart", "angina"],
    "cad": ["coronary", "artery", "disease", "cad", "heart", "angina"],
    "angina": ["coronary", "artery", "disease", "cad", "angina", "heart"],
    "angina pectoris": ["coronary", "artery", "disease", "cad", "angina", "heart"],
    "chest pain": ["coronary", "artery", "disease", "cad", "angina", "heart"],
    "heart failure": ["heart", "failure", "congestive", "chf", "cardiac", "edema"],
    "congestive heart failure": ["heart", "failure", "congestive", "chf", "cardiac", "edema"],
    "chf": ["heart", "failure", "congestive", "chf", "cardiac"],
    "stroke": ["stroke", "tia", "cerebrovascular", "fast", "infarction"],
    "tia": ["stroke", "tia", "cerebrovascular", "fast"],
    "transient ischemic attack": ["stroke", "tia", "cerebrovascular"],
    "glaucoma": ["glaucoma", "intraocular", "pressure", "optic", "eye"],
    "celiac": ["celiac", "coeliac", "gluten", "enteropathy", "wheat"],
    "celiac disease": ["celiac", "coeliac", "gluten", "enteropathy", "wheat"],
    "gluten": ["celiac", "coeliac", "gluten", "enteropathy", "wheat"],
    "migraine": ["migraine", "headache", "aura", "throbbing"],
    "chronic kidney disease": ["kidney", "renal", "ckd", "chronic", "disease"],
    "kidney disease": ["kidney", "renal", "ckd", "chronic", "disease"],
    "ckd": ["kidney", "renal", "ckd", "chronic", "disease"],
    "copd": ["copd", "chronic", "obstructive", "pulmonary", "emphysema", "bronchitis"],
    "gerd": ["gerd", "acid", "reflux", "heartburn", "esophageal"],
    "acid reflux": ["gerd", "acid", "reflux", "heartburn", "esophageal"],
    "heartburn": ["gerd", "acid", "reflux", "heartburn", "esophageal"],
    "rheumatoid arthritis": ["rheumatoid", "arthritis", "joint", "synovium"],
    "osteoarthritis": ["osteoarthritis", "arthritis", "joint", "cartilage"],
    "hypothyroidism": ["hypothyroidism", "thyroid", "underactive", "levothyroxine"],
    "underactive thyroid": ["hypothyroidism", "thyroid", "underactive", "levothyroxine"],
    "hyperthyroidism": ["hyperthyroidism", "thyroid", "overactive", "methimazole"],
    "overactive thyroid": ["hyperthyroidism", "thyroid", "overactive", "methimazole"],
    "thyroid": ["thyroid", "hypothyroidism", "hyperthyroidism"],
    "influenza": ["influenza", "flu", "respiratory", "viral"],
    "flu": ["influenza", "flu", "respiratory", "viral"],
    "pneumonia": ["pneumonia", "lung", "infection", "alveoli", "sputum"],
    "covid-19": ["covid", "covid-19", "coronavirus", "sars-cov-2"],
    "covid": ["covid", "covid-19", "coronavirus", "sars-cov-2"],
    "iron deficiency anemia": ["anemia", "iron", "deficiency", "hemoglobin", "fatigue"],
    "iron deficiency": ["anemia", "iron", "deficiency", "hemoglobin", "fatigue"],
    "anemia": ["anemia", "iron", "deficiency", "hemoglobin", "fatigue"],
    "depression": ["depression", "depressive", "mood", "sadness", "anhedonia"],
    "major depressive disorder": ["depression", "depressive", "mood", "sadness", "anhedonia"],
    "anxiety": ["anxiety", "generalized", "worry", "gad", "nervousness"],
    "generalized anxiety disorder": ["anxiety", "generalized", "worry", "gad", "nervousness"],
    "gad": ["anxiety", "generalized", "worry", "gad", "nervousness"],
    "atrial fibrillation": ["atrial", "fibrillation", "afib", "arrhythmia", "palpitations"],
    "afib": ["atrial", "fibrillation", "afib", "arrhythmia", "palpitations"],
    "arrhythmia": ["atrial", "fibrillation", "afib", "arrhythmia", "palpitations"],
    "deep vein thrombosis": ["deep", "vein", "thrombosis", "dvt", "clot", "pulmonary", "embolism"],
    "pulmonary embolism": ["pulmonary", "embolism", "pe", "clot", "thrombosis"],
    "dvt": ["deep", "vein", "thrombosis", "dvt", "clot"],
    "pe": ["pulmonary", "embolism", "pe", "clot"],
    # Infectious, tropical & common diseases
    "malaria": ["malaria", "plasmodium", "anopheles", "mosquito", "antimalarial", "chloroquine", "artemisinin"],
    "dengue": ["dengue", "dengue fever", "aedes", "hemorrhagic", "breakbone", "platelet"],
    "dengue fever": ["dengue", "dengue fever", "aedes", "hemorrhagic", "platelet"],
    "typhoid": ["typhoid", "typhoid fever", "salmonella", "enteric", "typhi"],
    "typhoid fever": ["typhoid", "typhoid fever", "salmonella", "enteric", "typhi"],
    "tuberculosis": ["tuberculosis", "tb", "mycobacterium", "pulmonary", "isoniazid", "rifampin"],
    "tb": ["tuberculosis", "tb", "mycobacterium"],
    "cholera": ["cholera", "vibrio", "diarrhea", "dehydration", "rehydration"],
    "hepatitis": ["hepatitis", "liver", "jaundice", "hbv", "hcv", "hav", "viral"],
    "hepatitis a": ["hepatitis", "hepatitis a", "hav", "liver", "jaundice"],
    "hepatitis b": ["hepatitis", "hepatitis b", "hbv", "liver", "jaundice"],
    "hepatitis c": ["hepatitis", "hepatitis c", "hcv", "liver", "jaundice"],
    "chikungunya": ["chikungunya", "mosquito", "aedes", "arthralgia", "joint"],
    "meningitis": ["meningitis", "meninges", "cerebrospinal", "neck", "stiffness"],
    "sepsis": ["sepsis", "septic", "bacteremia", "infection", "shock"],
    "yellow fever": ["yellow", "fever", "flavivirus", "mosquito", "jaundice"],
    # Foundational clinical biology & infectious concepts
    "virus": ["virus", "viral", "viruses", "capsid", "antiviral", "host", "infection"],
    "viruses": ["virus", "viral", "viruses", "capsid", "antiviral"],
    "viral": ["virus", "viral", "viruses", "antiviral", "infection"],
    "viral infection": ["virus", "viral", "infection", "antiviral"],
    "bacteria": ["bacteria", "bacterial", "prokaryotes", "antibiotic", "antibiotics", "infection"],
    "bacterial": ["bacteria", "bacterial", "antibiotic", "antibiotics", "infection"],
    "bacterial infection": ["bacteria", "bacterial", "infection", "antibiotics"],
    "infection": ["infection", "infections", "infectious", "virus", "bacteria", "pathogen"],
    "infections": ["infection", "infections", "infectious", "pathogen"],
    "fever": ["fever", "pyrexia", "temperature", "chills", "antipyretic"],
    "pyrexia": ["fever", "pyrexia", "temperature"],
    "inflammation": ["inflammation", "inflammatory", "immune", "swelling", "redness"],
    "immunity": ["immunity", "immune", "vaccine", "vaccines", "antibodies", "vaccination"],
    "vaccine": ["vaccine", "vaccines", "vaccination", "immunization", "immunity", "antibodies"],
    "vaccines": ["vaccine", "vaccines", "vaccination", "immunization", "immunity"],
    "vaccination": ["vaccine", "vaccines", "vaccination", "immunization", "immunity"],
    "immunization": ["vaccine", "vaccines", "vaccination", "immunization", "immunity"],
}

# Precision canonical condition mapping for clinical citation attribution
CONDITION_CANONICAL_ALIASES: Dict[str, List[str]] = {
    "High Blood Pressure (Hypertension)": [
        "high blood pressure", "hypertension", "hypertensive", "bp", "dash"
    ],
    "Type 2 Diabetes (Diabetes Mellitus)": [
        "type 2 diabetes", "diabetes mellitus", "t2d", "type 2 diabetes mellitus"
    ],
    "Type 1 Diabetes": [
        "type 1 diabetes", "t1d", "type 1 diabetes mellitus"
    ],
    "Asthma (Bronchial Asthma)": [
        "asthma", "bronchial asthma", "bronchospasm"
    ],
    "Coronary Artery Disease (CAD / Heart Disease)": [
        "coronary artery disease", "cad", "angina"
    ],
    "Heart Failure (Congestive Heart Failure)": [
        "heart failure", "congestive heart failure", "chf"
    ],
    "Stroke and TIA (Cerebrovascular Accident)": [
        "stroke", "tia", "transient ischemic attack", "cerebrovascular accident"
    ],
    "Glaucoma": [
        "glaucoma"
    ],
    "Celiac Disease": [
        "celiac disease", "celiac", "coeliac"
    ],
    "Migraine": [
        "migraine"
    ],
    "Chronic Kidney Disease (CKD)": [
        "chronic kidney disease", "ckd"
    ],
    "Chronic Obstructive Pulmonary Disease (COPD)": [
        "chronic obstructive pulmonary disease", "copd"
    ],
    "Gastroesophageal Reflux Disease (GERD / Acid Reflux)": [
        "gastroesophageal reflux disease", "gerd", "acid reflux"
    ],
    "Rheumatoid Arthritis (RA)": [
        "rheumatoid arthritis"
    ],
    "Osteoarthritis (OA)": [
        "osteoarthritis"
    ],
    "Hypothyroidism (Underactive Thyroid)": [
        "hypothyroidism", "underactive thyroid"
    ],
    "Hyperthyroidism (Overactive Thyroid)": [
        "hyperthyroidism", "overactive thyroid"
    ],
    "Influenza (Flu)": [
        "influenza", "flu"
    ],
    "Pneumonia": [
        "pneumonia"
    ],
    "COVID-19 (SARS-CoV-2)": [
        "covid-19", "covid", "sars-cov-2", "coronavirus"
    ],
    "Iron Deficiency Anemia": [
        "iron deficiency anemia"
    ],
    "Depression (Major Depressive Disorder)": [
        "major depressive disorder", "depression", "depressive"
    ],
    "Anxiety (Generalized Anxiety Disorder - GAD)": [
        "generalized anxiety disorder", "anxiety", "gad"
    ],
    "Atrial Fibrillation (AFib)": [
        "atrial fibrillation", "afib"
    ],
    "Deep Vein Thrombosis and Pulmonary Embolism (DVT / PE)": [
        "deep vein thrombosis", "pulmonary embolism", "dvt", "pe"
    ],
    "Malaria": ["malaria", "plasmodium"],
    "Dengue": ["dengue", "dengue fever"],
    "Typhoid Fever": ["typhoid", "typhoid fever"],
    "Tuberculosis": ["tuberculosis", "tb"],
    "Cholera": ["cholera"],
    "Hepatitis": ["hepatitis", "hepatitis a", "hepatitis b", "hepatitis c"],
    "Chikungunya": ["chikungunya"],
    "Meningitis": ["meningitis"],
    "Sepsis": ["sepsis"],
    "Yellow Fever": ["yellow fever"],
    "Virus (Viral Infections)": ["virus", "viral", "viral infection", "viruses"],
    "Bacteria (Bacterial Infections)": ["bacteria", "bacterial", "bacterial infection"],
    "Infection (Infectious Diseases)": ["infection", "infections", "infectious disease"],
    "Fever (Pyrexia)": ["fever", "pyrexia"],
    "Inflammation": ["inflammation", "inflammatory"],
    "Immunity and Vaccines": ["immunity", "immune system", "vaccine", "vaccines", "vaccination", "immunization"],
}


def is_hf_reachable(timeout: float = 1.0) -> bool:
    """Checks whether HuggingFace Hub is accessible without blocking."""
    if os.environ.get("HF_HUB_OFFLINE") == "1":
        return False
    try:
        req = urllib.request.Request("https://huggingface.co", headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status < 400
    except Exception:
        return False


class FallbackClinicalEmbeddings(Embeddings):
    """
    Deterministic clinical feature embeddings generating 384-dimensional normalized vectors.
    Uses clinical term weighting, stop-word elimination, and synonym expansion to ensure
    robust vector retrieval even without GPU or network access.
    """

    def __init__(self, dimension: int = 384):
        self.dimension = dimension

    def _embed(self, text: str) -> List[float]:
        vec = [0.0] * self.dimension
        text_lower = text.lower()
        raw_words = re.findall(r"\w+", text_lower)
        if not raw_words:
            return vec

        # Expand clinical synonyms to bridge vocabulary mismatches
        expanded_tokens: List[str] = []
        for phrase, synonyms in CLINICAL_SYNONYM_EXPANSIONS.items():
            if phrase in text_lower:
                expanded_tokens.extend(synonyms)

        # Filter stop words and keep substantive clinical terms
        for w in raw_words:
            if w not in ENGLISH_STOP_WORDS:
                expanded_tokens.append(w)

        if not expanded_tokens:
            expanded_tokens = raw_words

        for token in expanded_tokens:
            h = int(hashlib.sha256(token.encode("utf-8")).hexdigest(), 16)
            idx = h % self.dimension
            weight = 2.5 if token in CLINICAL_SYNONYM_EXPANSIONS else 1.0
            vec[idx] += weight

        norm = math.sqrt(sum(x * x for x in vec))
        if norm > 0:
            vec = [x / norm for x in vec]
        return vec

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._embed(text)


class MockClinicalLLM(BaseChatModel):
    """
    Deterministic clinical reasoning engine for offline verification and testing
    when GROQ_API_KEY is not configured. Emulates strict negative rejection guardrails,
    grounded clinical evidence extraction, and comprehensive response synthesis.
    """

    model_name: str = Field(default="mock-clinical-llama-3.1")

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        system_text = ""
        user_query = ""
        history_text = ""
        prior_human_queries: List[str] = []

        for m in messages:
            content = m.content if isinstance(m.content, str) else str(m.content)
            if isinstance(m, SystemMessage) or getattr(m, "type", "") == "system":
                system_text += content + "\n"
            elif isinstance(m, HumanMessage) or getattr(m, "type", "") == "human":
                user_query = content.strip()
                prior_human_queries.append(user_query)
            elif isinstance(m, AIMessage) or getattr(m, "type", "") == "ai":
                history_text += content + "\n"

        is_reformulation = "standalone question" in system_text.lower()
        is_qa = "medirag" in system_text.lower() or "clinical context" in system_text.lower()

        # Clinical condition candidates for conversational pronoun resolution
        candidate_conditions = [
            "high blood pressure",
            "hypertension",
            "hypertensive",
            "blood pressure",
            "type 2 diabetes",
            "type 1 diabetes",
            "diabetes",
            "asthma",
            "glaucoma",
            "celiac disease",
            "celiac",
            "migraine",
            "coronary artery disease",
            "heart disease",
            "heart failure",
            "stroke",
            "chronic kidney disease",
            "copd",
            "gerd",
            "rheumatoid arthritis",
            "osteoarthritis",
            "hypothyroidism",
            "hyperthyroidism",
            "influenza",
            "flu",
            "pneumonia",
            "covid-19",
            "iron deficiency anemia",
            "anemia",
            "depression",
            "anxiety",
            "atrial fibrillation",
            "deep vein thrombosis",
            "pulmonary embolism",
            "malaria",
            "dengue",
            "dengue fever",
            "typhoid",
            "typhoid fever",
            "tuberculosis",
            "cholera",
            "hepatitis",
            "chikungunya",
            "meningitis",
            "sepsis",
            "yellow fever",
            "virus",
            "viral infection",
            "bacteria",
            "bacterial infection",
            "infection",
            "fever",
            "inflammation",
            "immunity",
            "vaccine",
        ]
        history_corpus = (history_text + " " + " ".join(prior_human_queries[:-1])).lower()
        matched_history_condition = None
        for cond in candidate_conditions:
            if cond in history_corpus:
                matched_history_condition = cond
                break

        # ---------------------------------------------------------------------
        # Stage 1: Conversational Query Reformulation
        # ---------------------------------------------------------------------
        if is_reformulation:
            query_lower = user_query.lower()

            has_reference = bool(
                re.search(
                    r"\b(it|this|that|these|the condition|this condition|the disease|the illness|the disorder|the treatment|the medication)\b",
                    query_lower,
                )
            )

            if has_reference and matched_history_condition:
                reformulated = re.sub(
                    r"\b(it|this|that|these|the condition|this condition|the disease|the illness|the disorder)\b",
                    matched_history_condition.title(),
                    user_query,
                    flags=re.IGNORECASE,
                )
                return ChatResult(generations=[ChatGeneration(message=AIMessage(content=reformulated))])

            return ChatResult(generations=[ChatGeneration(message=AIMessage(content=user_query))])

        # ---------------------------------------------------------------------
        # Stage 2: Anti-Hallucination Grounded QA Chain
        # ---------------------------------------------------------------------
        if is_qa:
            ctx_match = re.search(r"Clinical Context:\s*(.*?)(?:\Z)", system_text, flags=re.DOTALL)
            context_text = ctx_match.group(1).strip() if ctx_match else ""

            # If no context was retrieved or context is empty, strictly reject
            if not context_text:
                return ChatResult(
                    generations=[ChatGeneration(message=AIMessage(content=NEGATIVE_REJECTION_MESSAGE))]
                )

            query_lower = user_query.lower()

            # Explicit adversarial, prompt-injection, and pseudoscientific / non-medical guardrail
            non_medical_or_adversarial_patterns = [
                r"\b(ignore\s+(all\s+)?prior|system\s+instructions|secret\s+developer|pretend\s+you\s+are)\b",
                r"\b(cryptocurrency|bitcoin|tesla\s+stock|call\s+and\s+put\s+options|financial\s+advisor)\b",
                r"\b(crystal\s+energy|energy\s+pyramids|colloidal\s+silver|diy\s+appendectomy)\b",
            ]
            if any(re.search(pat, query_lower) for pat in non_medical_or_adversarial_patterns):
                return ChatResult(
                    generations=[ChatGeneration(message=AIMessage(content=NEGATIVE_REJECTION_MESSAGE))]
                )

            # Conversational reference inheritance
            has_reference = bool(
                re.search(
                    r"\b(it|this|that|these|the condition|this condition|the disease|the illness|the disorder)\b",
                    query_lower,
                )
            )
            effective_query = f"{user_query} {matched_history_condition}" if (has_reference and matched_history_condition) else user_query
            effective_query_lower = effective_query.lower()

            # Parse retrieved clinical documents from context
            doc_blocks = re.split(r"\n*=== CLINICAL_DOC_BOUNDARY ===\n*|\n\n(?=Clinical Focus:)", context_text)
            doc_entries: List[Dict[str, Any]] = []

            for block in doc_blocks:
                b = block.strip()
                if not b:
                    continue
                focus_m = re.search(r"Clinical Focus:\s*([^\n]+)", b, re.IGNORECASE)
                qtype_m = re.search(r"Question Type:\s*([^\n]+)", b, re.IGNORECASE)
                src_m = re.search(r"Evidence Source:\s*([^\n]+)", b, re.IGNORECASE)
                q_m = re.search(r"Medical Question:\s*([^\n]+)", b, re.IGNORECASE)
                ans_m = re.search(r"Clinical Evidence:\s*(.*?)(?:\Z)", b, re.IGNORECASE | re.DOTALL)

                focus_val = focus_m.group(1).strip() if focus_m else ""
                qtype_val = qtype_m.group(1).strip().lower() if qtype_m else "general"
                src_val = src_m.group(1).strip() if src_m else "NIH"
                q_val = q_m.group(1).strip() if q_m else ""

                if ans_m:
                    ans_val = ans_m.group(1).strip()
                else:
                    ans_val = re.sub(
                        r"^(?:Clinical Focus|Question Type|Evidence Source|Medical Question):[^\n]*\n*",
                        "",
                        b,
                        flags=re.MULTILINE,
                    ).strip()

                doc_entries.append(
                    {
                        "focus": focus_val,
                        "qtype": qtype_val,
                        "source": src_val,
                        "question": q_val,
                        "evidence": ans_val,
                        "full_text": b.lower(),
                    }
                )

            # Match target clinical condition from query and synonyms
            matched_condition = None
            for phrase in sorted(CLINICAL_SYNONYM_EXPANSIONS.keys(), key=len, reverse=True):
                if re.search(rf"\b{re.escape(phrase)}\b", effective_query_lower):
                    matched_condition = phrase
                    break

            matching_docs: List[Dict[str, Any]] = []
            if matched_condition:
                condition_synonyms = CLINICAL_SYNONYM_EXPANSIONS[matched_condition]
                for doc in doc_entries:
                    doc_focus_lower = doc["focus"].lower()
                    doc_ev_lower = doc["evidence"].lower()
                    doc_q_lower = doc["question"].lower()
                    if any(re.search(rf"\b{re.escape(syn)}\b", doc_focus_lower) for syn in condition_synonyms):
                        matching_docs.append(doc)
                    elif re.search(rf"\b{re.escape(matched_condition)}\b", doc_ev_lower) or re.search(rf"\b{re.escape(matched_condition)}\b", doc_q_lower):
                        matching_docs.append(doc)

                # Guard against broad foundational terms (e.g., 'virus', 'infection', 'fever')
                # masking an uncataloged specific disease like 'Ebola virus disease'.
                broad_foundational_terms = {
                    "virus", "viruses", "viral", "viral infection",
                    "bacteria", "bacterial", "bacterial infection",
                    "infection", "infections", "fever", "pyrexia",
                    "inflammation", "immunity", "vaccine", "vaccines",
                    "vaccination", "immunization",
                }
                if matched_condition in broad_foundational_terms and matching_docs:
                    filler_verbs = {
                        "work", "works", "working", "happen", "happens", "mean", "means",
                        "spread", "spreads", "affect", "affects", "body", "human", "people",
                        "differ", "difference", "between", "types", "type", "common", "define",
                    }
                    distinguishing_words = (
                        set(re.findall(r"\b[a-z]{3,}\b", effective_query_lower))
                        - ENGLISH_STOP_WORDS
                        - GENERIC_CLINICAL_TERMS
                        - set(condition_synonyms)
                        - filler_verbs
                    )
                    if distinguishing_words:
                        specific_docs = [
                            d for d in matching_docs
                            if any(re.search(rf"\b{re.escape(w)}\b", d["full_text"]) for w in distinguishing_words)
                        ]
                        matching_docs = specific_docs
            else:
                # Fallback non-stopword token overlap (strict threshold to prevent false positives on large corpora)
                query_words = set(re.findall(r"\b[a-z]{3,}\b", effective_query_lower))
                content_words = query_words - ENGLISH_STOP_WORDS - GENERIC_CLINICAL_TERMS
                if content_words:
                    min_overlap = max(2, (len(content_words) + 1) // 2)
                    for doc in doc_entries:
                        if len(doc["evidence"].strip()) < 20:
                            continue
                        focus_lower = doc["focus"].lower()
                        focus_words = (
                            set(re.findall(r"\b[a-z]{3,}\b", focus_lower))
                            - ENGLISH_STOP_WORDS
                            - GENERIC_CLINICAL_TERMS
                        )
                        q_words = set(re.findall(r"\b[a-z]{3,}\b", doc["question"].lower()))
                        ev_words = set(re.findall(r"\b[a-z]{3,}\b", doc["evidence"].lower()))
                        if focus_lower != "general clinical" and focus_words and (
                            len(content_words & focus_words) >= min(2, len(focus_words))
                            or (len(content_words) == 1 and content_words <= focus_words)
                        ):
                            matching_docs.append(doc)
                        elif len(content_words & q_words) >= min_overlap:
                            matching_docs.append(doc)
                        elif len(content_words & ev_words) >= min_overlap:
                            matching_docs.append(doc)

            # Strict Negative Rejection if no document in context matches clinical topic
            if not matching_docs:
                return ChatResult(
                    generations=[ChatGeneration(message=AIMessage(content=NEGATIVE_REJECTION_MESSAGE))]
                )

            # Determine clinical intent: symptoms, treatment, causes, prevention
            is_symptom_inquiry = bool(
                re.search(
                    r"\b(symptom|symptoms|sign|signs|warning|presentation|feel|look like|crisis)\b",
                    effective_query_lower,
                )
            )
            is_treatment_inquiry = bool(
                re.search(
                    r"\b(treat|treatment|treatments|cure|medication|medications|drug|drugs|therapy|manage|management|prescription|diet|dash|lifestyle|inhaler|insulin)\b",
                    effective_query_lower,
                )
            )
            is_cause_inquiry = bool(
                re.search(
                    r"\b(cause|causes|caused|risk|risks|why|develop|trigger|triggers|salt|etiology)\b",
                    effective_query_lower,
                )
            )
            is_prevention_inquiry = bool(
                re.search(r"\b(prevent|prevention|preventative|preventing|avoid)\b", effective_query_lower)
            )

            selected_doc = None
            if is_treatment_inquiry:
                for d in matching_docs:
                    if d["qtype"] in ("treatment", "treatments"):
                        selected_doc = d
                        break
            elif is_symptom_inquiry:
                for d in matching_docs:
                    if d["qtype"] == "symptoms":
                        selected_doc = d
                        break
            elif is_cause_inquiry:
                for d in matching_docs:
                    if d["qtype"] in ("causes", "cause"):
                        selected_doc = d
                        break
            elif is_prevention_inquiry:
                for d in matching_docs:
                    if d["qtype"] in ("prevention", "prevent"):
                        selected_doc = d
                        break

            if not selected_doc:
                selected_doc = matching_docs[0]

            primary_focus = selected_doc["focus"]
            primary_source = selected_doc["source"]
            primary_evidence = selected_doc["evidence"]

            is_general_disease_query = not (
                is_symptom_inquiry or is_treatment_inquiry or is_cause_inquiry or is_prevention_inquiry
            )

            if is_general_disease_query and len(matching_docs) > 1:
                # Synthesize a clear, structured clinical summary (Overview, Symptoms, Causes, Treatment)
                same_focus_docs = [
                    d for d in matching_docs
                    if d["focus"].lower() == primary_focus.lower() and d["evidence"]
                ] or matching_docs

                section_labels = {
                    "information": "Overview",
                    "overview": "Overview",
                    "symptoms": "Symptoms",
                    "causes": "Causes & Risk Factors",
                    "cause": "Causes & Risk Factors",
                    "susceptibility": "Risk & Susceptibility",
                    "treatment": "Treatment & Management",
                    "treatments": "Treatment & Management",
                    "prevention": "Prevention",
                }
                seen_sections: Set[str] = set()
                sections_md: List[str] = []
                for d in same_focus_docs:
                    label = section_labels.get(d["qtype"], d["qtype"].title() or "Clinical Details")
                    if label in seen_sections:
                        continue
                    seen_sections.add(label)
                    sections_md.append(f"**{label}:**\n{d['evidence']}")
                    if len(sections_md) >= 4:
                        break

                clinical_response = (
                    f"Based on verified clinical reference records ({primary_source}):\n\n"
                    f"**Clinical Focus:** {primary_focus}\n"
                    f"**Evidence Source:** {primary_source}\n\n"
                    + "\n\n".join(sections_md)
                )
            else:
                # Construct comprehensive, verified clinical response
                clinical_response = (
                    f"Based on clinical reference records from the National Institutes of Health ({primary_source}):\n\n"
                    f"**Clinical Focus:** {primary_focus}\n"
                    f"**Evidence Source:** {primary_source}\n\n"
                    f"{primary_evidence}"
                )

                # Append supplementary clinical context if available for the EXACT SAME condition
                supplementary_docs = [
                    d
                    for d in matching_docs
                    if d["focus"] == selected_doc["focus"]
                    and d["qtype"] != selected_doc["qtype"]
                    and d["evidence"]
                ]
                if supplementary_docs:
                    supp = supplementary_docs[0]
                    supp_type = supp["qtype"].title()
                    clinical_response += f"\n\n**Additional Clinical Context ({supp_type}):**\n{supp['evidence']}"

            return ChatResult(generations=[ChatGeneration(message=AIMessage(content=clinical_response))])

        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=NEGATIVE_REJECTION_MESSAGE))])

    @property
    def _llm_type(self) -> str:
        return "mock-clinical-llm"


def detect_provider_from_key(api_key: Optional[str]) -> str:
    """
    Infers the AI provider from a pasted API key prefix:
    - 'AIza...' -> 'gemini'
    - 'gsk_...' -> 'groq'
    - 'sk-...' or other non-empty keys -> 'openai' (OpenAI / OpenAI-compatible)
    - Empty/None -> 'local'
    """
    if not api_key or not api_key.strip():
        return "local"
    key = api_key.strip()
    if key.startswith("AIza"):
        return "gemini"
    if key.startswith("gsk_"):
        return "groq"
    return "openai"


def get_llm(
    model_name: Optional[str] = None,
    temperature: float = 0.0,
    provider: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    max_tokens: int = 512,
) -> BaseChatModel:
    """
    Returns an LLM client based on provider/model_name and configured API keys:
    - Local Offline Mode ('mock-clinical-fallback' / 'local'): MockClinicalLLM (no key needed)
    - Auto-detect ('auto'): infers Gemini / Groq / OpenAI-compatible from API key prefix or base_url
    - Google Gemini API ('gemini-*' / provider='gemini'): ChatGoogleGenerativeAI (default: gemini-2.5-flash-lite)
    - Groq Cloud API ('llama-*' / provider='groq'): ChatGroq (default: llama-3.1-8b-instant)
    - OpenAI / OpenAI-Compatible (provider='openai' / 'openai-compatible'): ChatOpenAI (default: gpt-4.1-nano)
    Falls back gracefully to deterministic MockClinicalLLM if no API key is configured.
    """
    if model_name in ("mock-clinical-fallback", "mock-clinical-llama-3.1", "local-offline") or provider == "local":
        logger.info("Local offline MockClinicalLLM selected.")
        return MockClinicalLLM()

    resolved_provider = provider
    if resolved_provider == "auto":
        if base_url and base_url.strip():
            resolved_provider = "openai"
        elif api_key and api_key.strip():
            resolved_provider = detect_provider_from_key(api_key)
        else:
            resolved_provider = None

    # Check Google Gemini routing
    google_api_key = (
        api_key.strip()
        if (resolved_provider == "gemini" and api_key and api_key.strip())
        else (os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"))
    )
    is_gemini_requested = resolved_provider == "gemini" or (
        resolved_provider is None and model_name is not None and model_name.lower().startswith("gemini")
    )

    if is_gemini_requested or (
        model_name is None
        and resolved_provider is None
        and google_api_key
        and not os.getenv("GROQ_API_KEY")
        and not os.getenv("OPENAI_API_KEY")
    ):
        if google_api_key and google_api_key.strip():
            try:
                from langchain_google_genai import ChatGoogleGenerativeAI  # type: ignore

                gemini_model = (
                    model_name
                    if (model_name and not model_name.startswith("mock-"))
                    else DEFAULT_GEMINI_MODEL
                )
                logger.info(f"Initializing Google Gemini LLM ('{gemini_model}') with temperature={temperature}...")
                return ChatGoogleGenerativeAI(
                    model=gemini_model,
                    google_api_key=google_api_key.strip(),
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                )
            except Exception as e:
                logger.warning(f"Could not initialize ChatGoogleGenerativeAI ({e}). Falling back to MockClinicalLLM.")
                return MockClinicalLLM()
        else:
            logger.info("GOOGLE_API_KEY/GEMINI_API_KEY not configured. Falling back to MockClinicalLLM.")
            return MockClinicalLLM()

    # Check OpenAI / OpenAI-compatible routing
    is_openai_requested = resolved_provider in ("openai", "openai-compatible") or (
        resolved_provider is None and model_name is not None and model_name.lower().startswith(("gpt-", "o1", "o3", "o4"))
    )
    openai_api_key = (
        api_key.strip()
        if (resolved_provider in ("openai", "openai-compatible") and api_key and api_key.strip())
        else os.getenv("OPENAI_API_KEY")
    )
    if is_openai_requested or (
        model_name is None and resolved_provider is None and openai_api_key and not os.getenv("GROQ_API_KEY")
    ):
        if openai_api_key and openai_api_key.strip():
            try:
                from langchain_openai import ChatOpenAI  # type: ignore

                openai_model = model_name or DEFAULT_OPENAI_MODEL
                resolved_base_url = base_url.strip() if (base_url and base_url.strip()) else None
                if not resolved_base_url and openai_api_key.strip().startswith("sk-or-"):
                    resolved_base_url = "https://openrouter.ai/api/v1"
                kwargs: Dict[str, Any] = {
                    "model": openai_model,
                    "api_key": openai_api_key.strip(),
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                }
                if resolved_base_url:
                    kwargs["base_url"] = resolved_base_url
                logger.info(f"Initializing OpenAI-compatible LLM ('{openai_model}') with temperature={temperature}...")
                return ChatOpenAI(**kwargs)
            except Exception as e:
                logger.warning(f"Could not initialize ChatOpenAI ({e}). Falling back to MockClinicalLLM.")
                return MockClinicalLLM()
        else:
            logger.info("OPENAI_API_KEY not configured. Falling back to MockClinicalLLM.")
            return MockClinicalLLM()

    # Check Groq routing
    groq_api_key = (
        api_key.strip()
        if (resolved_provider == "groq" and api_key and api_key.strip())
        else os.getenv("GROQ_API_KEY")
    )
    groq_model = model_name or DEFAULT_GROQ_MODEL
    if groq_api_key and groq_api_key.strip():
        try:
            from langchain_groq import ChatGroq  # type: ignore

            logger.info(f"Initializing Groq LLM ('{groq_model}') with temperature={temperature}...")
            return ChatGroq(
                groq_api_key=groq_api_key.strip(),
                model_name=groq_model,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except Exception as e:
            logger.warning(f"Could not initialize ChatGroq ({e}). Falling back to MockClinicalLLM.")
            return MockClinicalLLM()

    logger.info("No cloud API key configured. Initializing deterministic MockClinicalLLM.")
    return MockClinicalLLM()


@functools.lru_cache(maxsize=4)
def get_embedding_model(model_name: str = DEFAULT_EMBEDDING_MODEL) -> Embeddings:
    """
    Initializes local HuggingFace BGE embeddings on CPU.
    1. First tries local cached model with local_files_only=True to prevent network latency/blocking.
    2. Tries online hub if reachable.
    3. Falls back gracefully to enhanced FallbackClinicalEmbeddings.
    """
    # 1. Attempt loading from local disk cache
    try:
        from langchain_huggingface import HuggingFaceEmbeddings  # type: ignore

        logger.info(f"Loading embedding model '{model_name}' from local cache (CPU)...")
        return HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": "cpu", "local_files_only": True},
            encode_kwargs={"normalize_embeddings": True},
        )
    except Exception as e:
        logger.debug(f"Local files only embedding initialization failed ({e}).")

    # 2. Attempt online hub fetch if reachable
    if is_hf_reachable(timeout=1.0):
        try:
            from langchain_huggingface import HuggingFaceEmbeddings  # type: ignore

            logger.info(f"Loading embedding model '{model_name}' from Hugging Face Hub (CPU)...")
            return HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )
        except Exception as e:
            logger.warning(
                f"HuggingFace embedding initialization failed ({type(e).__name__}: {e}). "
                f"Using local FallbackClinicalEmbeddings."
            )

    logger.info("Using enhanced local FallbackClinicalEmbeddings.")
    return FallbackClinicalEmbeddings()


def get_vectorstore(
    persist_dir: str = DEFAULT_PERSIST_DIR,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    embedding_model: Optional[Embeddings] = None,
) -> Chroma:
    """Loads existing Chroma vector store with embeddings."""
    if embedding_model is None:
        embedding_model = get_embedding_model()

    vectorstore = Chroma(
        persist_directory=persist_dir,
        embedding_function=embedding_model,
        collection_name=collection_name,
    )
    return vectorstore


def build_medirag_chain(
    retriever: Any,
    llm: Optional[BaseChatModel] = None,
) -> Any:
    """
    Assembles the two-stage LCEL Retrieval Chain:
    Stage 1: Conversational query reformulation via create_history_aware_retriever
    Stage 2: Strict anti-hallucination QA chain via create_stuff_documents_chain
    """
    if llm is None:
        llm = get_llm()

    # Stage 1: Contextualize question prompt
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question which might reference context "
        "in the chat history, formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, just reformulate it if "
        "needed and otherwise return it as is."
    )
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    history_aware_retriever = create_history_aware_retriever(
        llm=llm,
        retriever=retriever,
        prompt=contextualize_q_prompt,
    )

    # Stage 2: Anti-hallucination QA Prompt
    qa_system_prompt = (
        "You are MediRAG, an evidence-based clinical AI assistant.\n"
        "RULES:\n"
        "1. Answer ONLY using the Clinical Context below. Zero hallucination or extrapolation.\n"
        "2. For short 1-2 word disease queries (e.g., 'malaria', 'dengue'), provide a clear, "
        "well-structured summary (Overview, Symptoms, Causes, Treatment) from the context.\n"
        "3. OUT-OF-SCOPE REFUSAL: If the question is non-medical or the Clinical Context lacks "
        "relevant evidence, output ONLY this exact short sentence (save tokens, no extra text):\n"
        f"{NEGATIVE_REJECTION_MESSAGE}\n\n"
        "Clinical Context:\n"
        "{context}"
    )

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    from langchain_core.prompts import PromptTemplate

    doc_prompt = PromptTemplate.from_template("{page_content}")

    question_answer_chain = create_stuff_documents_chain(
        llm,
        qa_prompt,
        document_prompt=doc_prompt,
        document_separator="\n\n=== CLINICAL_DOC_BOUNDARY ===\n\n",
    )
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    return rag_chain


def extract_citations(documents: List[Document]) -> List[Dict[str, str]]:
    """Extracts unique clinical citations (Focus, QType, Source) from retrieved documents."""
    citations: List[Dict[str, str]] = []
    seen = set()
    for doc in documents:
        meta = doc.metadata or {}
        focus = meta.get("focus", "Clinical Evidence")
        qtype = meta.get("qtype", "inquiry")
        source = meta.get("source", "NIH")
        key = (focus.lower(), qtype.lower(), source.lower())
        if key not in seen:
            seen.add(key)
            citations.append({"focus": focus, "qtype": qtype, "source": source})
    return citations


def is_negative_rejection_answer(answer: str) -> bool:
    """Detects whether an LLM generation signifies negative rejection / refusal."""
    clean = answer.strip().strip('"').strip("'").lower()
    if NEGATIVE_REJECTION_MESSAGE.lower() in clean:
        return True
    refusal_patterns = [
        r"^out of scope\b",
        r"can only answer medical and clinical questions",
        r"^i am sorry, but based on the available clinical reference records",
        r"^based on the (available )?(provided )?clinical (reference )?(records|context), there is insufficient (clinical )?evidence",
        r"^there is insufficient (clinical )?evidence to answer this question",
        r"^i cannot answer this question based on the provided",
        r"not provided in the clinical context",
        r"not mentioned in the provided clinical",
        r"no clinical evidence (is |was )?provided to answer",
    ]
    for pat in refusal_patterns:
        if re.search(pat, clean):
            return True
    return False


def run_medirag_query(
    chain: Any,
    query: str,
    chat_history: Optional[List[BaseMessage]] = None,
) -> Dict[str, Any]:
    """
    Executes a query through the MediRAG LCEL chain, formats citations,
    and enforces strict negative rejection consistency.
    """
    if chat_history is None:
        chat_history = []

    result = chain.invoke({"input": query, "chat_history": chat_history})
    answer = result.get("answer", "").strip()
    docs = result.get("context", [])

    is_rejected = is_negative_rejection_answer(answer) or len(docs) == 0

    if is_rejected:
        answer = NEGATIVE_REJECTION_MESSAGE
        citations: List[Dict[str, str]] = []
    else:
        all_citations = extract_citations(docs)
        answer_lower = answer.lower()

        # 1. First check if explicit Clinical Focus was declared in answer
        focus_match = re.search(r"\*\*Clinical Focus:\*\*\s*([^\n]+)", answer, re.IGNORECASE)
        explicit_focus = focus_match.group(1).strip() if focus_match else ""

        if explicit_focus:
            relevant_citations = [
                c for c in all_citations if c["focus"].lower() == explicit_focus.lower()
            ]
        else:
            relevant_citations = []
            for c in all_citations:
                c_focus = c["focus"]
                aliases = CONDITION_CANONICAL_ALIASES.get(c_focus, [c_focus.lower()])
                matched_alias = any(
                    re.search(rf"\b{re.escape(alias)}\b", answer_lower)
                    for alias in aliases
                )
                if matched_alias:
                    relevant_citations.append(c)

        citations = relevant_citations if relevant_citations else ([all_citations[0]] if all_citations else [])

    return {
        "answer": answer,
        "context": docs,
        "citations": citations,
        "is_rejected": is_rejected,
    }

#!/usr/bin/env python3
"""
Round 1B: Persona-Driven Document Intelligence (Offline & Dockerized)
Author: Gemini
Description: This script is adapted for the Adobe India Hackathon 2025.
It processes a collection of PDF documents based on a user persona and a job-to-be-done,
extracting the most relevant sections. It is designed to run in a containerized,
offline environment. Models are pre-loaded from a local directory.
"""
import os
import json
import re
import logging
from datetime import datetime
from typing import Dict, List, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, Counter
import warnings
warnings.filterwarnings("ignore")

import fitz  # PyMuPDF
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import spacy
from sentence_transformers import SentenceTransformer

# --- Static Configuration ---
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define fixed paths for models and IO, compliant with Docker environment
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'models')
INPUT_DIR = '/app/input'  # Fixed path as per Docker volume mount
OUTPUT_DIR = '/app/output'  # Fixed path as per Docker volume mount

# Define model names to be loaded from the local directory
SENTENCE_MODEL_NAME = 'all-MiniLM-L6-v2'
SPACY_MODEL_NAME = 'en_core_web_sm'

# --- Data Structures ---
@dataclass
class DocumentSection:
    document: str
    page_number: int
    section_title: str
    content: str
    level: str

@dataclass
class PersonaProfile:
    role: str
    expertise_areas: List[str]
    domain: str
    keywords: List[str]

@dataclass
class JobTask:
    description: str
    task_type: str
    keywords: List[str]
    priority_areas: List[str] = field(default_factory=list)

# --- Core Logic Classes (Adapted for Offline Use) ---
class DocumentProcessor:
    """Extracts structured content from PDF documents."""
    def __init__(self, spacy_model_path: str):
        self.nlp = self._load_local_spacy_model(spacy_model_path)
        self.common_headings = {
            'abstract', 'introduction', 'methodology', 'methods', 'results',
            'discussion', 'conclusion', 'references', 'literature review',
            'background', 'related work', 'evaluation', 'analysis'
        }

    def _load_local_spacy_model(self, model_path: str):
        try:
            logger.info(f"Loading local SpaCy model from: {model_path}")
            return spacy.load(model_path)
        except Exception as e:
            logger.error(f"Fatal: Could not load SpaCy model from '{model_path}'. Error: {e}")
            raise

    def extract_text_from_pdf(self, pdf_path: str) -> Dict[int, str]:
        """Extracts plain text from each page of a PDF."""
        try:
            doc = fitz.open(pdf_path)
            page_texts = {i + 1: page.get_text() for i, page in enumerate(doc)}
            doc.close()
            return page_texts
        except Exception as e:
            logger.error(f"Could not extract text from {pdf_path}: {e}")
            return {}

    def extract_outline_from_toc(self, doc: fitz.Document) -> List[Dict]:
        """Extracts outline from the PDF's Table of Contents."""
        toc = doc.get_toc()
        if not toc:
            return []
        return [
            {"level": f"H{min(level, 3)}", "text": title.strip(), "page": page}
            for level, title, page in toc if title.strip() and page > 0
        ]

    def extract_outline_from_formatting(self, doc: fitz.Document) -> List[Dict]:
        """Fallback to extract headings based on font size heuristics."""
        font_counts = defaultdict(int)
        for page_num in range(doc.page_count):
            page = doc.load_page(page_num)
            blocks = page.get_text("dict")["blocks"]
            for b in blocks:
                if "lines" in b:
                    for l in b["lines"]:
                        for s in l["spans"]:
                            font_counts[round(s["size"])] += 1
        
        sorted_sizes = sorted(font_counts.keys(), reverse=True)
        if not sorted_sizes:
            return []
        
        # Simple heuristic: Top 3 font sizes are potential headings
        h1_size, h2_size, h3_size = (sorted_sizes + [None, None, None])[:3]
        
        outline = []
        for page_num in range(doc.page_count):
            page = doc.load_page(page_num)
            blocks = page.get_text("dict")["blocks"]
            for b in blocks:
                if "lines" in b:
                    for l in b["lines"]:
                        span = l["spans"][0]
                        text = " ".join([s["text"] for s in l["spans"]]).strip()
                        size = round(span["size"])
                        if not text or len(text) > 150: # Avoid long paragraphs
                            continue

                        level = None
                        if size == h1_size: level = "H1"
                        elif size == h2_size: level = "H2"
                        elif size == h3_size: level = "H3"
                        
                        if level:
                            outline.append({"level": level, "text": text, "page": page_num + 1})
        return outline

    def extract_sections(self, pdf_path: str) -> List[DocumentSection]:
        """Extracts a list of DocumentSection objects from a PDF."""
        filename = os.path.basename(pdf_path)
        page_texts = self.extract_text_from_pdf(pdf_path)
        if not page_texts:
            return []

        doc = fitz.open(pdf_path)
        outline = self.extract_outline_from_toc(doc)
        if len(outline) < 3: # If TOC is poor, try formatting
            outline.extend(self.extract_outline_from_formatting(doc))
        doc.close()
        
        # Deduplicate and sort outline
        seen = set()
        dedup_outline = []
        for item in sorted(outline, key=lambda x: x["page"]):
            key = (item["text"].lower(), item["page"])
            if key not in seen:
                seen.add(key)
                dedup_outline.append(item)
        
        if not dedup_outline:
             # Fallback: Treat each page as a section if no outline is found
            return [
                DocumentSection(filename, pn, f"Page {pn}", text, "H1")
                for pn, text in page_texts.items()
            ]

        sections = []
        for i, heading in enumerate(dedup_outline):
            start_page = heading["page"]
            end_page = dedup_outline[i + 1]["page"] - 1 if i + 1 < len(dedup_outline) else len(page_texts)
            
            content_parts = []
            for page_num in range(start_page, end_page + 1):
                if page_num in page_texts:
                    page_content = page_texts[page_num]
                    # On the first page, take content after the heading
                    if page_num == start_page:
                        heading_pos = page_content.lower().find(heading["text"].lower())
                        if heading_pos != -1:
                            page_content = page_content[heading_pos + len(heading["text"]):]
                    content_parts.append(page_content)
            
            content = self._clean_content("\n".join(content_parts))
            if content:
                sections.append(DocumentSection(filename, start_page, heading["text"], content, heading["level"]))
        
        logger.info(f"Extracted {len(sections)} sections from {filename}")
        return sections

    def _clean_content(self, text: str) -> str:
        """Basic cleaning of extracted text."""
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'-\n', '', text) # De-hyphenate
        return text.strip()


class PersonaAnalyzer:
    """Parses persona and job descriptions."""
    def parse_persona(self, persona_text: str) -> PersonaProfile:
        text_lower = persona_text.lower()
        role = "Professional" # Default role
        if "researcher" in text_lower: role = "Researcher"
        elif "student" in text_lower: role = "Student"
        elif "analyst" in text_lower: role = "Analyst"
        
        keywords = self._extract_keywords(text_lower)
        
        domain = "general" # Default domain
        if any(k in keywords for k in ['financial', 'revenue', 'market', 'investment']):
            domain = "business"
        elif any(k in keywords for k in ['methodology', 'datasets', 'benchmarks', 'computational']):
            domain = "research"
        
        return PersonaProfile(
            role=role,
            expertise_areas=[k for k in keywords if k in persona_text][:5],
            domain=domain,
            keywords=keywords
        )

    def parse_job(self, job_text: str) -> JobTask:
        text_lower = job_text.lower()
        task_type = "analysis" # Default
        if "literature review" in text_lower: task_type = "literature_review"
        elif "summarize" in text_lower: task_type = "summarization"
        elif "prepare" in text_lower or "exam" in text_lower: task_type = "preparation"
        
        keywords = self._extract_keywords(text_lower)
        
        priority_areas = []
        match = re.search(r'focusing on (.*?)(?:\.|,|$)', job_text, re.IGNORECASE)
        if match:
            priority_areas = [p.strip() for p in match.group(1).split(',')]

        return JobTask(
            description=job_text,
            task_type=task_type,
            keywords=keywords,
            priority_areas=priority_areas
        )
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extracts meaningful keywords from text."""
        return list(set(re.findall(r'\b[a-zA-Z-]{4,}\b', text.lower())))


class RelevanceEngine:
    """Scores and ranks document sections based on relevance."""
    def __init__(self, sentence_model_path: str):
        self.model = self._load_local_sentence_model(sentence_model_path)
        self.vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))

    def _load_local_sentence_model(self, model_path: str):
        try:
            logger.info(f"Loading local Sentence Transformer model from: {model_path}")
            return SentenceTransformer(model_path)
        except Exception as e:
            logger.error(f"Fatal: Could not load Sentence Transformer model from '{model_path}'. Error: {e}")
            raise

    def calculate_relevance_score(self, section: DocumentSection, persona: PersonaProfile, job: JobTask) -> float:
        """Calculates a composite relevance score for a section."""
        section_text = f"{section.section_title}. {section.content}"
        query_text = f"{persona.role}. {job.description}. {' '.join(persona.keywords + job.keywords + job.priority_areas)}"
        
        # Semantic Score
        try:
            section_emb = self.model.encode(section_text[:512]) # Limit text length for model
            query_emb = self.model.encode(query_text)
            semantic_score = cosine_similarity([section_emb], [query_emb])[0][0]
        except Exception as e:
            logger.warning(f"Could not compute semantic score for section '{section.section_title}': {e}")
            semantic_score = 0.0

        # Keyword Score
        try:
            tfidf_matrix = self.vectorizer.fit_transform([section_text, query_text])
            keyword_score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        except Exception:
            keyword_score = 0.0
        
        # Structural Importance Score
        level_score = {"H1": 1.0, "H2": 0.8, "H3": 0.6}.get(section.level, 0.4)
        page_score = max(0, 1 - (section.page_number / 50.0)) # Early pages are more important
        importance_score = (level_score + page_score) / 2
        
        # Weighted average
        final_score = (0.5 * semantic_score) + (0.3 * keyword_score) + (0.2 * importance_score)
        return float(max(0.0, min(1.0, final_score)))

    def rank_sections(self, sections: List[DocumentSection], persona: PersonaProfile, job: JobTask) -> List[Tuple[DocumentSection, float]]:
        """Ranks sections and returns the top results."""
        if not sections:
            return []
        
        scored_sections = [
            (sec, self.calculate_relevance_score(sec, persona, job))
            for sec in sections
        ]
        
        # Sort by score in descending order
        scored_sections.sort(key=lambda x: x[1], reverse=True)
        return scored_sections


class DocumentIntelligenceSystem:
    """Main orchestrator for the document intelligence pipeline."""
    def __init__(self, model_dir: str):
        logger.info("Initializing Document Intelligence System...")
        sentence_model_path = os.path.join(model_dir, SENTENCE_MODEL_NAME)
        spacy_model_path = os.path.join(model_dir, SPACY_MODEL_NAME)

        self.doc_processor = DocumentProcessor(spacy_model_path)
        self.persona_analyzer = PersonaAnalyzer()
        self.relevance_engine = RelevanceEngine(sentence_model_path)
        logger.info("System initialized successfully.")

    def run_pipeline(self, input_dir: str, output_dir: str):
        """Executes the full document processing pipeline."""
        start_time = datetime.now()
        
        # 1. Load Inputs
        pdf_files, persona_text, job_text = self._load_inputs(input_dir)
        if not pdf_files:
            logger.error("No PDF files found in the input directory. Aborting.")
            return

        logger.info(f"Processing {len(pdf_files)} documents for persona: '{persona_text.split('.')[0]}...'")

        # 2. Extract Sections from all documents
        all_sections = []
        for pdf_file in pdf_files:
            pdf_path = os.path.join(input_dir, pdf_file)
            all_sections.extend(self.doc_processor.extract_sections(pdf_path))
        
        if not all_sections:
            logger.error("Failed to extract any sections from the documents. Aborting.")
            return
            
        # 3. Analyze Persona and Job
        persona = self.persona_analyzer.parse_persona(persona_text)
        job = self.persona_analyzer.parse_job(job_text)
        
        # 4. Rank Sections
        ranked_sections = self.relevance_engine.rank_sections(all_sections, persona, job)
        
        # 5. Generate and Save Output
        output_data = self._generate_output_json(ranked_sections, pdf_files, persona_text, job_text)
        self._save_output(output_data, output_dir)
        
        end_time = datetime.now()
        logger.info(f"Pipeline completed in {(end_time - start_time).total_seconds():.2f} seconds.")

    def _load_inputs(self, input_dir: str) -> Tuple[List[str], str, str]:
        """Loads PDFs, persona, and job from the input directory."""
        pdf_files = sorted([f for f in os.listdir(input_dir) if f.lower().endswith('.pdf')])
        
        try:
            with open(os.path.join(input_dir, 'persona.txt'), 'r', encoding='utf-8') as f:
                persona_text = f.read().strip()
        except FileNotFoundError:
            persona_text = "PhD Researcher in Computational Biology"
            logger.warning("persona.txt not found. Using default persona.")
            
        try:
            with open(os.path.join(input_dir, 'job.txt'), 'r', encoding='utf-8') as f:
                job_text = f.read().strip()
        except FileNotFoundError:
            job_text = "Prepare a comprehensive literature review focusing on methodologies, datasets, and performance benchmarks"
            logger.warning("job.txt not found. Using default job description.")
            
        return pdf_files, persona_text, job_text

    def _generate_output_json(self, ranked_sections: List[Tuple[DocumentSection, float]],
                              pdf_files: List[str], persona_text: str, job_text: str) -> Dict:
        """Generates the final JSON output in the required format."""
        
        output = {
            "metadata": {
                "input_documents": pdf_files,
                "persona": persona_text,
                "job_to_be_done": job_text,
                "processing_timestamp": datetime.now().isoformat(),
            },
            "extracted_sections": [],
            "sub_section_analysis": []
        }
        
        for i, (section, score) in enumerate(ranked_sections[:20]): # Limit to top 20 sections
            output["extracted_sections"].append({
                "document": section.document,
                "page_number": section.page_number,
                "section_title": section.section_title,
                "importance_rank": i + 1,
            })
            
            output["sub_section_analysis"].append({
                "document": section.document,
                "refined_text": self._refine_text(section.content),
                "page_number": section.page_number
            })
            
        return output

    def _refine_text(self, content: str) -> str:
        """Creates a concise summary of the section content."""
        sentences = re.split(r'(?<=[.!?])\s+', content)
        # Return first few sentences up to a reasonable length
        summary = ""
        for s in sentences:
            if len(summary) + len(s) > 500:
                break
            summary += s + " "
        
        return summary.strip() if summary else content[:500]

    def _save_output(self, data: Dict, output_dir: str):
        """Saves the final data to output.json."""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        output_path = os.path.join(output_dir, 'output.json')
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            logger.info(f"Successfully wrote output to {output_path}")
        except Exception as e:
            logger.error(f"Failed to write output JSON file: {e}")


def main():
    """Main entry point for the script."""
    logger.info("=" * 60)
    logger.info("Starting Persona-Driven Document Intelligence System")
    logger.info(f"Input Directory: {INPUT_DIR}")
    logger.info(f"Output Directory: {OUTPUT_DIR}")
    logger.info(f"Model Directory: {MODEL_DIR}")
    logger.info("=" * 60)
    
    # Validate that directories and models exist
    if not os.path.isdir(INPUT_DIR):
        logger.error(f"Input directory '{INPUT_DIR}' not found. Please mount it correctly.")
        return
    if not os.path.isdir(os.path.join(MODEL_DIR, SENTENCE_MODEL_NAME)):
         logger.error(f"Sentence model not found at '{os.path.join(MODEL_DIR, SENTENCE_MODEL_NAME)}'")
         return
    if not os.path.isdir(os.path.join(MODEL_DIR, SPACY_MODEL_NAME)):
         logger.error(f"SpaCy model not found at '{os.path.join(MODEL_DIR, SPACY_MODEL_NAME)}'")
         return
         
    system = DocumentIntelligenceSystem(model_dir=MODEL_DIR)
    system.run_pipeline(input_dir=INPUT_DIR, output_dir=OUTPUT_DIR)

if __name__ == "__main__":
    main()
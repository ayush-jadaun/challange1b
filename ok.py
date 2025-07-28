#!/usr/bin/env python3
"""
Round 1B: Persona-Driven Document Intelligence
Modified to work directly with PDFs in input folder (no config.json required)
Fixed JSON serialization issue with float32 values
"""

import os
import json
import re
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
from collections import defaultdict
import warnings
warnings.filterwarnings("ignore")

import fitz  # PyMuPDF
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import spacy
from sentence_transformers import SentenceTransformer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class DocumentSection:
    """Represents a document section with metadata"""
    document: str
    page_number: int
    section_title: str
    content: str
    level: str
    subsections: List[Dict] = None

@dataclass
class PersonaProfile:
    """Represents parsed persona information"""
    role: str
    expertise_areas: List[str]
    domain: str
    keywords: List[str]

@dataclass
class JobTask:
    """Represents the job-to-be-done"""
    description: str
    task_type: str
    keywords: List[str]
    deliverables: List[str]

class DocumentProcessor:
    """Handles PDF processing and section extraction"""
    
    def __init__(self):
        self.nlp = None
        self._load_nlp_model()
    
    def _load_nlp_model(self):
        """Load spaCy model for text processing"""
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.warning("spaCy model not found, using basic text processing")
            self.nlp = None
    
    def extract_text_from_pdf(self, pdf_path: str) -> Dict[int, str]:
        """Extract text from PDF, organized by page"""
        try:
            doc = fitz.open(pdf_path)
            page_texts = {}
            
            for page_num in range(doc.page_count):
                page = doc[page_num]
                text = page.get_text()
                page_texts[page_num + 1] = text
            
            doc.close()
            return page_texts
        except Exception as e:
            logger.error(f"Error extracting text from {pdf_path}: {e}")
            return {}
    
    def extract_outline_structure(self, pdf_path: str) -> List[Dict]:
        """Extract document outline structure"""
        try:
            doc = fitz.open(pdf_path)
            outline = []
            
            # Try to get TOC first
            toc = doc.get_toc()
            if toc:
                for level, title, page in toc:
                    heading_level = f"H{min(level, 3)}"
                    outline.append({
                        "level": heading_level,
                        "text": title.strip(),
                        "page": page
                    })
            else:
                # Fallback: analyze text formatting
                outline = self._extract_headings_from_formatting(doc)
            
            doc.close()
            return outline
            
        except Exception as e:
            logger.error(f"Error extracting outline from {pdf_path}: {e}")
            return []
    
    def _extract_headings_from_formatting(self, doc) -> List[Dict]:
        """Extract headings based on text formatting"""
        outline = []
        font_sizes = defaultdict(list)
        
        # Analyze font sizes across document
        for page_num in range(min(doc.page_count, 10)):  # Sample first 10 pages
            page = doc[page_num]
            blocks = page.get_text("dict")
            
            for block in blocks.get("blocks", []):
                if block.get("type") == 0:  # Text block
                    for line in block.get("lines", []):
                        for span in line.get("spans", []):
                            text = span.get("text", "").strip()
                            if text and len(text) > 5:
                                font_size = span.get("size", 12)
                                font_sizes[font_size].append(text)
        
        # Determine heading levels based on font sizes
        sorted_sizes = sorted(font_sizes.keys(), reverse=True)
        heading_sizes = sorted_sizes[:3]  # Top 3 sizes for H1, H2, H3
        
        # Extract headings
        for page_num in range(doc.page_count):
            page = doc[page_num]
            blocks = page.get_text("dict")
            
            for block in blocks.get("blocks", []):
                if block.get("type") == 0:
                    for line in block.get("lines", []):
                        for span in line.get("spans", []):
                            text = span.get("text", "").strip()
                            font_size = span.get("size", 12)
                            
                            if font_size in heading_sizes and len(text) > 5:
                                level_idx = heading_sizes.index(font_size)
                                level = f"H{level_idx + 1}"
                                
                                outline.append({
                                    "level": level,
                                    "text": text,
                                    "page": page_num + 1
                                })
        
        return outline
    
    def extract_sections(self, pdf_path: str) -> List[DocumentSection]:
        """Extract structured sections from PDF"""
        outline = self.extract_outline_structure(pdf_path)
        page_texts = self.extract_text_from_pdf(pdf_path)
        
        if not outline or not page_texts:
            return []
        
        sections = []
        filename = os.path.basename(pdf_path)
        
        for i, heading in enumerate(outline):
            # Get content between current heading and next heading
            current_page = heading["page"]
            next_page = outline[i + 1]["page"] if i + 1 < len(outline) else max(page_texts.keys())
            
            content_parts = []
            for page_num in range(current_page, next_page + 1):
                if page_num in page_texts:
                    page_text = page_texts[page_num]
                    
                    # Extract content after current heading
                    if page_num == current_page:
                        heading_pos = page_text.find(heading["text"])
                        if heading_pos >= 0:
                            content_parts.append(page_text[heading_pos + len(heading["text"]):])
                        else:
                            content_parts.append(page_text)
                    else:
                        content_parts.append(page_text)
            
            content = "\n".join(content_parts).strip()
            
            # Clean up content
            content = self._clean_section_content(content)
            
            if content and len(content) > 50:  # Minimum content length
                section = DocumentSection(
                    document=filename,
                    page_number=current_page,
                    section_title=heading["text"],
                    content=content,
                    level=heading["level"]
                )
                sections.append(section)
        
        return sections
    
    def _clean_section_content(self, content: str) -> str:
        """Clean and normalize section content"""
        # Remove excessive whitespace
        content = re.sub(r'\s+', ' ', content)
        
        # Remove page numbers and common PDF artifacts
        content = re.sub(r'\b\d+\b(?=\s*$)', '', content, flags=re.MULTILINE)
        content = re.sub(r'^\s*\d+\s*$', '', content, flags=re.MULTILINE)
        
        # Remove URLs and email addresses
        content = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', content)
        content = re.sub(r'\S+@\S+', '', content)
        
        return content.strip()

class PersonaAnalyzer:
    """Analyzes persona and job requirements"""
    
    def __init__(self):
        self.domain_keywords = {
            'research': ['methodology', 'analysis', 'study', 'research', 'findings', 'results', 'conclusion'],
            'business': ['revenue', 'profit', 'market', 'strategy', 'growth', 'investment', 'financial'],
            'education': ['concept', 'theory', 'principle', 'example', 'exercise', 'practice', 'exam'],
            'technical': ['implementation', 'algorithm', 'system', 'architecture', 'performance', 'optimization']
        }
    
    def parse_persona(self, persona_text: str) -> PersonaProfile:
        """Parse persona description into structured profile"""
        persona_lower = persona_text.lower()
        
        # Extract role
        role_patterns = [
            r'(phd researcher|researcher|analyst|student|engineer|scientist|manager|consultant)',
            r'(undergraduate|graduate|postdoc|professor|developer|specialist)'
        ]
        
        role = "General Professional"
        for pattern in role_patterns:
            match = re.search(pattern, persona_lower)
            if match:
                role = match.group(1).title()
                break
        
        # Determine domain
        domain = "general"
        if any(word in persona_lower for word in ['research', 'phd', 'academic', 'scientist']):
            domain = "research"
        elif any(word in persona_lower for word in ['business', 'analyst', 'investment', 'financial']):
            domain = "business"
        elif any(word in persona_lower for word in ['student', 'undergraduate', 'education']):
            domain = "education"
        elif any(word in persona_lower for word in ['engineer', 'developer', 'technical', 'system']):
            domain = "technical"
        
        # Extract expertise areas and keywords
        expertise_areas = []
        keywords = []
        
        # Domain-specific keyword extraction
        for word in persona_text.split():
            clean_word = re.sub(r'[^\w]', '', word.lower())
            if len(clean_word) > 3:
                keywords.append(clean_word)
        
        # Add domain-specific keywords
        if domain in self.domain_keywords:
            keywords.extend(self.domain_keywords[domain])
        
        return PersonaProfile(
            role=role,
            expertise_areas=expertise_areas,
            domain=domain,
            keywords=list(set(keywords))
        )
    
    def parse_job(self, job_text: str) -> JobTask:
        """Parse job-to-be-done into structured task"""
        job_lower = job_text.lower()
        
        # Determine task type
        task_type = "analysis"
        if "review" in job_lower or "literature" in job_lower:
            task_type = "literature_review"
        elif "summary" in job_lower or "summarize" in job_lower:
            task_type = "summarization"
        elif "analyze" in job_lower or "analysis" in job_lower:
            task_type = "analysis"
        elif "identify" in job_lower or "find" in job_lower:
            task_type = "identification"
        elif "prepare" in job_lower or "study" in job_lower:
            task_type = "preparation"
        
        # Extract keywords
        keywords = []
        important_words = re.findall(r'\b[a-zA-Z]{4,}\b', job_text.lower())
        keywords.extend([word for word in important_words if word not in ['that', 'with', 'from', 'this', 'they', 'were', 'been', 'have']])
        
        # Extract potential deliverables
        deliverables = []
        if "review" in job_lower:
            deliverables.append("comprehensive review")
        if "summary" in job_lower:
            deliverables.append("summary report")
        if "analysis" in job_lower:
            deliverables.append("analytical report")
        
        return JobTask(
            description=job_text,
            task_type=task_type,
            keywords=list(set(keywords)),
            deliverables=deliverables
        )

class RelevanceEngine:
    """Scores and ranks sections based on persona and job requirements"""
    
    def __init__(self):
        self.sentence_model = None
        self.tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self._load_models()
    
    def _load_models(self):
        """Load semantic similarity models"""
        try:
            self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("Loaded sentence transformer model")
        except Exception as e:
            logger.warning(f"Could not load sentence transformer: {e}")
    
    def calculate_relevance_score(self, section: DocumentSection, 
                                persona: PersonaProfile, job: JobTask) -> float:
        """Calculate composite relevance score"""
        
        # Combine section title and content for analysis
        section_text = f"{section.section_title} {section.content}"
        
        # 1. Semantic similarity score
        semantic_score = self._calculate_semantic_similarity(section_text, persona, job)
        
        # 2. Keyword matching score
        keyword_score = self._calculate_keyword_score(section_text, persona, job)
        
        # 3. Section importance score (based on level and position)
        importance_score = self._calculate_importance_score(section)
        
        # 4. Domain alignment score
        domain_score = self._calculate_domain_score(section_text, persona)
        
        # Weighted combination
        final_score = (
            0.35 * semantic_score +
            0.25 * keyword_score +
            0.20 * importance_score +
            0.20 * domain_score
        )
        
        # Convert to Python float to avoid JSON serialization issues
        return float(final_score)
    
    def _calculate_semantic_similarity(self, section_text: str, 
                                     persona: PersonaProfile, job: JobTask) -> float:
        """Calculate semantic similarity using embeddings"""
        if not self.sentence_model:
            return 0.5  # Fallback score
        
        try:
            # Create combined query from persona and job
            query_text = f"{persona.role} {job.description} {' '.join(persona.keywords[:5])}"
            
            # Get embeddings
            section_embedding = self.sentence_model.encode([section_text])
            query_embedding = self.sentence_model.encode([query_text])
            
            # Calculate cosine similarity
            similarity = cosine_similarity(section_embedding, query_embedding)[0][0]
            
            # Convert numpy float32 to Python float
            return float(max(0, min(1, similarity)))
            
        except Exception as e:
            logger.warning(f"Error in semantic similarity calculation: {e}")
            return 0.5
    
    def _calculate_keyword_score(self, section_text: str, 
                               persona: PersonaProfile, job: JobTask) -> float:
        """Calculate keyword matching score"""
        section_lower = section_text.lower()
        
        # Combine all relevant keywords
        all_keywords = persona.keywords + job.keywords
        
        if not all_keywords:
            return 0.5
        
        # Count keyword matches
        matches = 0
        for keyword in all_keywords:
            if keyword.lower() in section_lower:
                matches += 1
        
        # Normalize by total keywords
        score = matches / len(all_keywords)
        return min(1.0, score * 2)  # Boost score slightly
    
    def _calculate_importance_score(self, section: DocumentSection) -> float:
        """Calculate section importance based on structural position"""
        # H1 sections are more important than H2, H3
        level_scores = {"H1": 1.0, "H2": 0.8, "H3": 0.6}
        level_score = level_scores.get(section.level, 0.5)
        
        # Earlier sections might be more important (introduction, methodology)
        position_score = max(0.5, 1.0 - (section.page_number - 1) * 0.05)
        
        # Sections with certain keywords are more important
        important_keywords = ['introduction', 'methodology', 'results', 'conclusion', 
                            'analysis', 'findings', 'discussion', 'summary']
        
        title_lower = section.section_title.lower()
        keyword_bonus = 0.2 if any(kw in title_lower for kw in important_keywords) else 0
        
        return min(1.0, level_score * position_score + keyword_bonus)
    
    def _calculate_domain_score(self, section_text: str, persona: PersonaProfile) -> float:
        """Calculate domain-specific alignment score"""
        if persona.domain == "general":
            return 0.5
        
        domain_indicators = {
            'research': ['study', 'research', 'methodology', 'analysis', 'findings', 'hypothesis', 'experiment'],
            'business': ['revenue', 'profit', 'market', 'strategy', 'investment', 'growth', 'financial', 'business'],
            'education': ['concept', 'theory', 'principle', 'example', 'definition', 'explanation', 'practice'],
            'technical': ['algorithm', 'implementation', 'system', 'architecture', 'performance', 'technical', 'method']
        }
        
        indicators = domain_indicators.get(persona.domain, [])
        if not indicators:
            return 0.5
        
        section_lower = section_text.lower()
        matches = sum(1 for indicator in indicators if indicator in section_lower)
        
        return min(1.0, matches / len(indicators) * 2)
    
    def rank_sections(self, sections: List[DocumentSection], 
                     persona: PersonaProfile, job: JobTask, 
                     top_k: int = 10) -> List[Tuple[DocumentSection, float]]:
        """Rank sections by relevance score"""
        
        # Calculate scores for all sections
        scored_sections = []
        for section in sections:
            score = self.calculate_relevance_score(section, persona, job)
            scored_sections.append((section, score))
        
        # Sort by score (descending)
        scored_sections.sort(key=lambda x: x[1], reverse=True)
        
        # Return top k
        return scored_sections[:top_k]

class DocumentIntelligenceSystem:
    """Main system that orchestrates the document intelligence pipeline"""
    
    def __init__(self):
        self.doc_processor = DocumentProcessor()
        self.persona_analyzer = PersonaAnalyzer()
        self.relevance_engine = RelevanceEngine()
    
    def process_document_collection(self, input_dir: str, output_dir: str, 
                                   persona_text: str = None, job_text: str = None):
        """Process all PDFs in the input directory"""
        try:
            # Find all PDF files in input directory
            pdf_files = []
            if os.path.exists(input_dir):
                for file in os.listdir(input_dir):
                    if file.lower().endswith('.pdf'):
                        pdf_files.append(file)
            
            if not pdf_files:
                logger.error("No PDF files found in input directory")
                return
            
            logger.info(f"Found {len(pdf_files)} PDF files: {pdf_files}")
            
            # Use default persona and job if not provided
            if not persona_text:
                persona_text = "Travel Planner"
            
            if not job_text:
                job_text = "Plan a trip of 4 days for a group of 10 college friends."
            
            # Process documents
            all_sections = []
            for pdf_file in pdf_files:
                pdf_path = os.path.join(input_dir, pdf_file)
                if os.path.exists(pdf_path):
                    logger.info(f"Processing {pdf_file}")
                    sections = self.doc_processor.extract_sections(pdf_path)
                    all_sections.extend(sections)
                    logger.info(f"Extracted {len(sections)} sections from {pdf_file}")
            
            if not all_sections:
                logger.error("No sections extracted from documents")
                return
            
            # Analyze persona and job
            persona = self.persona_analyzer.parse_persona(persona_text)
            job = self.persona_analyzer.parse_job(job_text)
            
            logger.info(f"Persona: {persona.role} in {persona.domain}")
            logger.info(f"Job type: {job.task_type}")
            
            # Rank sections
            ranked_sections = self.relevance_engine.rank_sections(all_sections, persona, job)
            
            # Generate output
            output = self._generate_output(ranked_sections, pdf_files, persona_text, job_text)
            
            # Save output
            output_path = os.path.join(output_dir, 'output.json')
            os.makedirs(output_dir, exist_ok=True)
            
            with open(output_path, 'w') as f:
                json.dump(output, f, indent=2)
            
            logger.info(f"Generated output with {len(ranked_sections)} ranked sections")
            logger.info(f"Output saved to: {output_path}")
            
        except Exception as e:
            logger.error(f"Error processing document collection: {e}")
            raise
    
    def _generate_output(self, ranked_sections: List[Tuple[DocumentSection, float]], 
                        pdf_files: List[str], persona_text: str, job_text: str) -> Dict:
        """Generate output JSON in required format"""
        
        output = {
            "metadata": {
                "input_documents": pdf_files,
                "persona": persona_text,
                "job_to_be_done": job_text,
                "processing_timestamp": datetime.now().isoformat()
            },
            "extracted_sections": [],
            "sub_section_analysis": []
        }
        
        # Add extracted sections with ranking
        for i, (section, score) in enumerate(ranked_sections):
            output["extracted_sections"].append({
                "document": section.document,
                "page_number": section.page_number,
                "section_title": section.section_title,
                "importance_rank": i + 1,
                "relevance_score": round(score, 4)  # round() returns Python float
            })
            
            # Add subsection analysis (refined text)
            refined_text = self._refine_section_text(section.content, persona_text, job_text)
            
            output["sub_section_analysis"].append({
                "document": section.document,
                "refined_text": refined_text,
                "page_number": section.page_number,
                "section_title": section.section_title,
                "relevance_score": round(score, 4)  # round() returns Python float
            })
        
        return output
    
    def _refine_section_text(self, content: str, persona_text: str, job_text: str) -> str:
        """Refine section text to extract most relevant parts"""
        # Simple refinement: take first few sentences that seem most relevant
        sentences = re.split(r'[.!?]+', content)
        
        # Score sentences based on keyword presence
        persona_keywords = set(re.findall(r'\b[a-zA-Z]{4,}\b', persona_text.lower()))
        job_keywords = set(re.findall(r'\b[a-zA-Z]{4,}\b', job_text.lower()))
        all_keywords = persona_keywords.union(job_keywords)
        
        scored_sentences = []
        for sentence in sentences[:10]:  # Limit to first 10 sentences
            sentence = sentence.strip()
            if len(sentence) > 20:  # Minimum sentence length
                score = sum(1 for keyword in all_keywords if keyword in sentence.lower())
                scored_sentences.append((sentence, score))
        
        # Sort by score and take top sentences
        scored_sentences.sort(key=lambda x: x[1], reverse=True)
        top_sentences = [s[0] for s in scored_sentences[:3]]  # Top 3 sentences
        
        refined = '. '.join(top_sentences)
        
        # Ensure reasonable length
        if len(refined) > 500:
            refined = refined[:500] + "..."
        
        return refined if refined else content[:300] + "..."

def main():
    """Main execution function with command line interface"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Document Intelligence System')
    parser.add_argument('--input_dir', default='./input', help='Input directory containing PDF files')
    parser.add_argument('--output_dir', default='./output', help='Output directory for results')
    parser.add_argument('--persona', help='Persona description (optional)')
    parser.add_argument('--job', help='Job-to-be-done description (optional)')
    
    args = parser.parse_args()
    
    logger.info("Starting Document Intelligence System")
    logger.info(f"Input directory: {args.input_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    
    # Initialize system
    system = DocumentIntelligenceSystem()
    
    # Process documents
    system.process_document_collection(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        persona_text=args.persona,
        job_text=args.job
    )
    
    logger.info("Processing completed successfully")

if __name__ == "__main__":
    main()
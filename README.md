# Enhanced Document Intelligence System

## Overview

This enhanced document intelligence system addresses the key issues in the original code and provides significant improvements for persona-driven document analysis. The system extracts structured information from PDF documents and ranks sections based on user personas and job requirements.

## Key Improvements

### 1. Fixed spaCy Model Loading
- **Problem**: Original code failed with "spaCy model not found" warning
- **Solution**: 
  - Multiple fallback options for spaCy models
  - Automatic model download attempt
  - Graceful fallback to basic text processing if spaCy unavailable
  - Enhanced error handling and logging

### 2. Enhanced PDF Processing
- **Improved heading detection** using multiple strategies:
  - Table of Contents extraction
  - Font-based analysis with better heuristics
  - Pattern-based detection (numbered sections, common headings)
  - Duplicate removal and validation
- **Better text extraction** with formatting preservation
- **Robust content cleaning** with OCR artifact removal

### 3. Advanced Persona & Job Analysis
- **Enhanced persona parsing** with:
  - Better role extraction using regex patterns
  - Domain classification with confidence scoring
  - Seniority level detection
  - Comprehensive keyword generation
- **Improved job analysis** with:
  - Task type classification
  - Priority area extraction
  - Deliverable identification
  - Context-aware keyword extraction

### 4. Sophisticated Relevance Scoring
- **Multi-factor scoring algorithm**:
  - Semantic similarity (with sentence transformers)
  - Enhanced keyword matching with weighted importance
  - Structural importance analysis
  - Domain alignment scoring
  - Priority area alignment
  - Contextual relevance assessment
- **Adaptive weighting** based on persona and job characteristics
- **Diversity filtering** to ensure varied content selection

### 5. Better Error Handling & Robustness
- Comprehensive exception handling
- Graceful degradation when models unavailable
- Input validation and sanitization
- Progress tracking and detailed logging
- Output validation before saving

### 6. Enhanced Output Format
- Additional metadata including processing statistics
- Content previews and key phrase extraction
- Section-level metadata (heading level, content length)
- Processing quality indicators

## Installation & Setup

### Using Docker (Recommended)

1. **Build the Docker image:**
```bash
docker build --platform linux/amd64 -t enhanced-doc-intelligence:latest .
```

2. **Run the container:**
```bash
docker run --rm \
  -v $(pwd)/input:/app/input \
  -v $(pwd)/output:/app/output \
  --network none \
  enhanced-doc-intelligence:latest
```

### Local Installation

1. **Install Python dependencies:**
```bash
pip install -r requirements.txt
```

2. **Download spaCy model:**
```bash
python -m spacy download en_core_web_sm
```

3. **Run the system:**
```bash
python enhanced_doc_intelligence.py --input_dir ./input --output_dir ./output
```

## Usage

### Basic Usage
```bash
python enhanced_doc_intelligence.py \
  --input_dir ./pdfs \
  --output_dir ./results
```

### With Custom Persona and Job
```bash
python enhanced_doc_intelligence.py \
  --input_dir ./pdfs \
  --output_dir ./results \
  --persona "PhD researcher in computational biology specializing in drug discovery" \
  --job "Prepare comprehensive literature review focusing on methodologies and performance benchmarks"
```

### Command Line Options
- `--input_dir`: Directory containing PDF files (default: `./input`)
- `--output_dir`: Output directory for results (default: `./output`)
- `--persona`: Persona description (optional)
- `--job`: Job-to-be-done description (optional)
- `--setup-deps`: Setup dependencies (download models)
- `--verbose`: Enable verbose logging

## Architecture

### Core Components

1. **EnhancedDocumentProcessor**
   - Multi-strategy heading detection
   - Robust text extraction with formatting
   - Content cleaning and validation

2. **EnhancedPersonaAnalyzer**
   - Advanced persona parsing with domain classification
   - Comprehensive job analysis with task categorization
   - Context-aware keyword extraction

3. **EnhancedRelevanceEngine**
   - Multi-factor relevance scoring
   - Adaptive weighting algorithms
   - Diversity-aware section ranking

4. **EnhancedDocumentIntelligenceSystem**
   - Orchestrates the complete pipeline
   - Progress tracking and error handling
   - Enhanced output generation

### Scoring Algorithm

The relevance score combines multiple factors:

- **Semantic Similarity (25-35%)**: Using sentence transformers or TF-IDF fallback
- **Keyword Matching (20-25%)**: Weighted by keyword importance
- **Structural Importance (15-25%)**: Based on heading level and position
- **Domain Alignment (15-20%)**: Domain-specific indicator matching
- **Priority Alignment (10-25%)**: Alignment with job priorities
- **Contextual Relevance (5-15%)**: Document structure and quality indicators

Weights are dynamically adjusted based on:
- Persona characteristics (role, domain, seniority)
- Job requirements (task type, priorities)
- Content availability and quality

## Output Format

The system generates:

1. **output.json**: Main results with ranked sections
2. **processing_stats.json**: Processing statistics and quality metrics

### Sample Output Structure
```json
{
  "metadata": {
    "input_documents": ["doc1.pdf", "doc2.pdf"],
    "persona": "PhD researcher...",
    "job_to_be_done": "Literature review...",
    "processing_timestamp": "2025-07-28T20:58:42",
    "persona_analysis": {
      "role": "PhD Researcher",
      "domain": "research",
      "seniority_level": "senior"
    },
    "job_analysis": {
      "task_type": "literature_review",
      "priority_areas": ["methodology", "benchmarks"]
    }
  },
  "extracted_sections": [
    {
      "document": "doc1.pdf",
      "page_number": 3,
      "section_title": "Methodology",
      "section_level": "H2",
      "importance_rank": 1,
      "relevance_score": 0.8945,
      "content_length": 1234,
      "content_preview": "This section describes..."
    }
  ],
  "sub_section_analysis": [
    {
      "document": "doc1.pdf",
      "refined_text": "Enhanced extracted text...",
      "page_number": 3,
      "section_title": "Methodology",
      "relevance_score": 0.8945,
      "key_phrases": ["neural networks", "performance evaluation"]
    }
  ]
}
```

## Performance Optimizations

- **Efficient PDF processing**: Selective page analysis for large documents
- **Model caching**: Reuse of loaded models across documents
- **Memory management**: Streaming processing for large document collections
- **Parallel processing**: Ready for multi-threading (can be enabled)

## Constraints Compliance

- ✅ **CPU only**: No GPU dependencies
- ✅ **Model size**: ≤ 1GB (sentence transformer ~90MB, spaCy ~50MB)
- ✅ **Processing time**: Optimized for ≤60 seconds per collection
- ✅ **Offline operation**: No internet calls during processing
- ✅ **AMD64 compatibility**: Docker platform specification included

## Testing

The system has been tested with:
- Research papers (10-50 pages)
- Business reports and financial documents
- Educational textbooks and course materials
- Technical documentation

## Troubleshooting

### Common Issues

1. **spaCy model not found**
   - Run: `python enhanced_doc_intelligence.py --setup-deps`
   - Or manually: `python -m spacy download en_core_web_sm`

2. **PDF processing errors**
   - Check PDF file integrity
   - Ensure files are not password-protected
   - Verify sufficient disk space

3. **Memory issues**
   - Reduce batch size for large document collections
   - Ensure sufficient RAM (recommended: 4GB+)

4. **Poor relevance scores**
   - Provide more specific persona descriptions
   - Include detailed job requirements with priorities
   - Check document quality and content relevance

### Logging

Enable verbose logging for debugging:
```bash
python enhanced_doc_intelligence.py --verbose
```

## Future Enhancements

- Multi-language support
- Advanced NLP techniques (named entity recognition)
- Interactive web interface
- Real-time processing capabilities
- Integration with external knowledge bases

## License

This project is developed for the Adobe India Hackathon 2025.
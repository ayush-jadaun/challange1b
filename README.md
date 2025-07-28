# Enhanced Document Intelligence System - Round 1B

**Team: FrontendGrinders**  
**Members:** Ayush Jadaun, Shreeya Srivastava, Ashish Singh  
**Challenge:** Adobe India Hackathon 2025 - Round 1B: Persona-Driven Document Intelligence

## Overview

This enhanced document intelligence system addresses the key requirements for Round 1B of the Adobe India Hackathon 2025. The system extracts structured information from PDF documents and ranks sections based on user personas and job requirements, providing intelligent document analysis for diverse use cases.

## Key Features

### 1. Persona-Driven Intelligence
- **Advanced persona parsing** with role detection (Researcher, Student, Analyst, etc.)
- **Domain classification** (research, business, education)
- **Expertise area identification** and keyword extraction
- **Context-aware job analysis** with task type classification

### 2. Multi-Strategy PDF Processing
- **Robust heading detection** using:
  - Table of Contents extraction
  - Font-based analysis with intelligent heuristics
  - Pattern-based detection for numbered sections
  - Fallback to page-based segmentation
- **Enhanced text extraction** with formatting preservation
- **Content cleaning** with OCR artifact removal

### 3. Sophisticated Relevance Scoring
- **Multi-factor scoring algorithm**:
  - Semantic similarity using sentence transformers
  - Keyword matching with weighted importance
  - Structural importance (heading level, page position)
  - Domain alignment scoring
  - Priority area alignment
- **Adaptive weighting** based on persona and job characteristics
- **Diversity filtering** for comprehensive content selection

### 4. Offline-First Architecture
- **No internet dependencies** during processing
- **Pre-downloaded models** embedded in Docker image
- **Robust error handling** with graceful fallbacks
- **CPU-optimized** for AMD64 architecture

## Architecture & Setup

### Docker-Based Deployment (Recommended)

Our solution uses a multi-stage Docker build process that downloads models during image creation:

#### Project Structure
```
frontendgrinders-solution/
├── Dockerfile
├── requirements.txt
├── download_models.py
├── challange2b.py
└── README.md
```

#### Build Process
```bash
# Build the Docker image (downloads models during build)
docker build --platform linux/amd64 -t frontendgrinders-solution:v1 .
```

#### Execution (As Per Challenge Requirements)
```bash
# Run the container with volume mounts
docker run --rm \
  -v $(pwd)/input:/app/input \
  -v $(pwd)/output:/app/output \
  --network none \
  frontendgrinders-solution:v1
```

### Input Directory Structure
Your `input` directory should contain:
```
input/
├── document1.pdf
├── document2.pdf
├── document3.pdf (3-10 PDFs)
├── persona.txt
└── job.txt
```

#### Sample Files

**persona.txt**
```
PhD Researcher in Computational Biology specializing in machine learning applications for drug discovery and molecular analysis
```

**job.txt**
```
Prepare a comprehensive literature review focusing on methodologies, datasets, and performance benchmarks for neural network approaches in pharmaceutical research
```

## Technical Implementation

### Core Components

1. **DocumentProcessor**
   - Multi-strategy heading extraction
   - Robust PDF text extraction using PyMuPDF
   - Content cleaning and normalization

2. **PersonaAnalyzer**
   - Role classification (Researcher, Student, Analyst, etc.)
   - Domain detection (research, business, education)
   - Keyword extraction and expertise mapping

3. **RelevanceEngine**
   - Sentence transformer-based semantic similarity
   - TF-IDF keyword matching
   - Structural importance weighting
   - Priority alignment scoring

4. **DocumentIntelligenceSystem**
   - Pipeline orchestration
   - Progress tracking and logging
   - Output generation and validation

### Model Configuration
- **Sentence Transformer**: `all-MiniLM-L6-v2` (~90MB)
- **SpaCy Model**: `en_core_web_sm` (~50MB)
- **Total Model Size**: ~140MB (well under 1GB limit)

### Performance Optimizations
- **Efficient text processing**: Chunked processing for large documents
- **Model caching**: Single model load per container run
- **Memory management**: Optimized for 16GB RAM constraint
- **Processing time**: Optimized for <60 seconds per collection

## Output Format

The system generates `output.json` in the specified format:

```json
{
  "metadata": {
    "input_documents": ["doc1.pdf", "doc2.pdf", "doc3.pdf"],
    "persona": "PhD Researcher in Computational Biology...",
    "job_to_be_done": "Prepare a comprehensive literature review...",
    "processing_timestamp": "2025-07-28T20:58:42.123456",
    "persona_analysis": {
      "role": "Researcher",
      "domain": "research",
      "keywords": ["computational", "biology", "machine", "learning"]
    },
    "job_analysis": {
      "task_type": "literature_review",
      "priority_areas": ["methodologies", "datasets", "benchmarks"]
    }
  },
  "extracted_sections": [
    {
      "document": "paper1.pdf",
      "page_number": 3,
      "section_title": "Methodology",
      "importance_rank": 1
    },
    {
      "document": "paper2.pdf",
      "page_number": 5,
      "section_title": "Performance Evaluation",
      "importance_rank": 2
    }
  ],
  "sub_section_analysis": [
    {
      "document": "paper1.pdf",
      "refined_text": "This section presents a novel neural network architecture for drug discovery applications. The methodology combines convolutional layers with attention mechanisms to process molecular structures...",
      "page_number": 3
    },
    {
      "document": "paper2.pdf",
      "refined_text": "Performance benchmarks were established using standard datasets including ChEMBL and PubChem. The evaluation metrics include accuracy, precision, recall, and F1-scores...",
      "page_number": 5
    }
  ]
}
```

## Test Cases Supported

### 1. Academic Research
- **Documents**: Research papers on specialized topics
- **Persona**: PhD researchers, academic professionals
- **Job**: Literature reviews, methodology analysis

### 2. Business Analysis
- **Documents**: Annual reports, financial statements
- **Persona**: Investment analysts, business consultants
- **Job**: Market analysis, financial trend identification

### 3. Educational Content
- **Documents**: Textbook chapters, course materials
- **Persona**: Students at various levels
- **Job**: Exam preparation, concept understanding

## Constraints Compliance

✅ **CPU Only**: No GPU dependencies, optimized for AMD64  
✅ **Model Size**: ~140MB total (well under 1GB limit)  
✅ **Processing Time**: Optimized for <60 seconds per collection  
✅ **Offline Operation**: No internet calls during processing  
✅ **Memory Efficient**: Works within 16GB RAM constraint  
✅ **Platform Compatible**: AMD64 Docker platform specified  

## Dependencies

### Python Packages (requirements.txt)
```
PyMuPDF==1.23.14
numpy==1.24.3
scikit-learn==1.3.0
spacy==3.7.2
sentence-transformers==2.2.2
torch==2.0.1
transformers==4.35.0
```

### System Requirements
- **Platform**: AMD64 (x86_64)
- **RAM**: 16GB
- **CPU**: 8 cores
- **Storage**: ~2GB for models and processing

## Scoring Optimization

Our solution is optimized for the challenge scoring criteria:

### Section Relevance (60 points)
- Semantic similarity matching between sections and persona+job
- Weighted keyword matching with domain-specific terms
- Structural importance analysis (H1 > H2 > H3)
- Priority area alignment with job requirements

### Sub-Section Relevance (40 points)
- Granular text extraction with context preservation
- Content refinement and summarization
- Key phrase extraction and relevance scoring
- Quality assessment and filtering

## Team Approach

**FrontendGrinders** focused on creating a robust, production-ready solution that:

1. **Handles edge cases** gracefully with comprehensive error handling
2. **Scales efficiently** across different document types and domains
3. **Provides consistent results** through deterministic algorithms
4. **Meets all constraints** while maximizing performance
5. **Follows best practices** in code organization and documentation

## Troubleshooting

### Common Issues

1. **Docker build fails**
   ```bash
   # Ensure platform specification
   docker build --platform linux/amd64 -t frontendgrinders-solution:v1 .
   ```

2. **Models not loading**
   - Check that models were downloaded during build
   - Verify model directory structure in container

3. **PDF processing errors**
   - Ensure PDFs are not password-protected
   - Check file integrity and format compatibility

4. **Memory issues**
   - Monitor container memory usage
   - Reduce document collection size if needed

### Debug Mode
For development and testing, you can run with verbose logging by modifying the script to include debug information.

## Future Enhancements

- Multi-language document support
- Advanced named entity recognition
- Interactive visualization of relevance scores
- Integration with external knowledge bases
- Real-time processing capabilities

## Acknowledgments

This solution was developed by **Team FrontendGrinders** for the Adobe India Hackathon 2025. We appreciate Adobe's commitment to innovation in document intelligence and are excited to contribute to the future of PDF interaction technology.

---

**Team FrontendGrinders**  
*Connecting the Dots Through Intelligent Document Analysis*
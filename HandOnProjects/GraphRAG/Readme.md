\# OCR-to-GraphRAG Pipeline



This repository contains a multi-stage pipeline designed to transform raw document data into a structured \*\*Neo4j Knowledge Graph\*\*. By leveraging the latest DeepSeek OCR models and strategic LLM-based chunking, this workflow ensures high-fidelity data extraction and relationship mapping for advanced RAG applications.



\## 🔄 Workflow Overview



The pipeline follows a linear path from raw document processing to graph database insertion:







1\.  \*\*OCR Extraction\*\*: `dpsk\_ocr\_pipeline.py` extracts text, tables, and captions.

2\.  \*\*Refinement\*\*: Data is saved and expanded in `final\_result.md`.

3\.  \*\*Chunking\*\*: `chunk\_stra\_ppline\_v3.py` prepares the text for LLM processing.

4\.  \*\*Graph Construction\*\*: `graph\_construct\_neo4j\_v2.py` builds and merges the graph into Neo4j.



---



\## 🛠 Component Details



\### 1. Data Extraction \& Pre-processing

\* \*\*`dpsk\_ocr\_pipeline.py`\*\*: Utilizes the latest \*\*DeepSeek OCR model\*\* to perform comprehensive document analysis. It accurately identifies and extracts text, image captions, and complex table structures.

\* \*\*`final\_result.md`\*\*: The primary output of the OCR stage. This file undergoes targeted expansion to serve as the high-quality source text for subsequent chunking and embedding.



\### 2. Strategic Chunking

\* \*\*`chunk\_stra\_ppline\_v3.py`\*\*: A segmentation script that breaks down `final\_result.md` into optimized batches. This process is tailored for LLM consumption to maximize the accuracy of relationship extraction.

\* \*\*`human\_check\_final\_v3.md`\*\*: The intermediate output produced after chunking. This file contains the segmented data ready for graph construction and serves as a checkpoint for quality assurance.



\### 3. Graph Construction

\* \*\*`graph\_construct\_neo4j\_v2.py`\*\*: The final stage of the pipeline. It processes the chunked content from `human\_check\_final\_v3.md`, identifies entities and relationships, and constructs a property graph that is merged directly into a \*\*Neo4j database\*\*.



---



\## 🚀 Key Features

\* \*\*DeepSeek Integration\*\*: High-accuracy OCR for multimodal document elements.

\* \*\*Optimized LLM Batching\*\*: Strategic chunking to prevent context loss during relationship extraction.

\* \*\*Graph-Ready Output\*\*: Seamless integration with Neo4j for GraphRAG applications.



---



\## 📦 Requirements

\* Python 3.10+

\* Neo4j Database (Local or AuraDB)

\* DeepSeek API Credentials


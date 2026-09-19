# Multimodal RAG for PDFs

A multimodal RAG pipeline that answers questions from PDFs containing **text, images, diagrams, and charts**.

## Architecture

```text
                         PDF
                          │
                   ┌──────┴──────┐
                   │             │
                 Text          Images
                   │             │
             Text Chunks       GPT-4o
                   │             │
                   │      Image Descriptions
                   │             │
                   └──────┬──────┘
                          │
                          ▼
               OpenAI Embeddings
             text-embedding-3-small
                          │
                          ▼
                       Chroma
                          │
                      User Query
                          │
                          ▼
                  Similarity Search
                          │
                  ┌───────┴───────┐
                  ▼               ▼
             Text Chunks    Image Descriptions
                                  │
                                  ▼
                           Original Images
                                  │
                  ┌───────────────┴──────────────┐
                  ▼                              ▼
             Text Context                  Images
                  │                              │
                  └──────────────┬───────────────┘
                                 ▼
                               GPT-4o
                                 │
                                 ▼
                              Answer
```

## Tech Stack

* **PDF Parsing:** PyMuPDF
* **Text Chunking:** LangChain RecursiveCharacterTextSplitter
* **Image Understanding:** OpenAI GPT-4o
* **Embeddings:** OpenAI `text-embedding-3-small`
* **Vector Store:** Chroma
* **Answer Generation:** OpenAI GPT-4o

## How It Works

1. Extract text and images from the PDF using PyMuPDF.
2. Split extracted text into chunks.
3. Generate searchable descriptions for images using GPT-4o.
4. Embed both text chunks and image descriptions.
5. Store embeddings in Chroma.
6. Retrieve relevant text/image representations for a user query.
7. Pass retrieved text and original images to GPT-4o to generate the answer.

## Setup

```bash
pip install pymupdf langchain langchain-openai langchain-chroma chromadb
```

Set your OpenAI API key:

```bash
export OPENAI_API_KEY="your-api-key"
```

Update the PDF path in the code:

```python
PDF_PATH = "Azure-Kubernetes-Service.pdf"
```

Run:

```bash
python multimodal_rag.py
```

Then ask questions directly from the terminal:

```text
Ask a question (type 'exit' to quit): What does the architecture diagram show?

Answer:
...
```

## Future Improvements

* LangChain `MultiVectorRetriever`
* Direct image embeddings using CLIP
* Table-aware PDF extraction
* Persistent/reusable vector store
* Better document layout parsing with Docling

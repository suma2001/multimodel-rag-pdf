'''
Title: Building a multimodal RAG pipeline for PDFs with text and images using LangChain

WorkFlow:

PDf --> PDF Parser --> Text Elements --> Text chunks --> OpenAI Embeddings --> Chroma DB
            |
            |--------> Images --> Image description(using LLM) --> Text description --> OpenAI EMbeddings --> Chroma DB

Retrieval Query --> Chroma similarity ---> Text chunks -------------
                            |                                       | --> LLM Model --> Answer
                            |-------------> Image descriptions -----

Components:-
1. PDF parsing       -  PyMuPDF 
2. Text splitting    -  Langchain text splitter
3. Image summary     -  OpenAI gpt-4o
4. Embeddings        -  OpenAI text-embedding-3-small
5. Vector DB         -  Chroma/FAISS
6. Output Generation - gpt-4o

'''

import os
import fitz

from pathlib import Path
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from uuid import uuid4
import torch
import open_clip
from PIL import Image

import base64
from dotenv import load_dotenv

PDF_PATH = "Azure-Kubernetes-Service.pdf"
IMAGE_DIR = "extracted_images"
CHROMA_DIR = "chroma_db"
CLIP_CHROMA_DIR = "clip_chroma_db"
CLIP_MODEL, _, CLIP_PREPROCESS = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
CLIP_TOKENIZER = open_clip.get_tokenizer("ViT-B-32")

load_dotenv()

# Extract text and images from PDF
def extract_pdf_elements(pdf_path: str, image_dir):
    '''
    Extracts the text blocks and images in the pdf.
    Returns text_elements and image_elements
    '''

    Path(image_dir).mkdir(parents=True, exist_ok=True)
    pdf = fitz.open(pdf_path)

    text_elements = []
    image_elements = []

    for page_number, page in enumerate(pdf, start=1):
        # Extract text
        text = page.get_text("text").strip()
        if text:
            text_elements.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": pdf_path,
                        "page": page_number,
                        "type": "text"
                    }
                )
            )
        
        # Extract images
        images = page.get_images(full=True)
        for image_index, image in enumerate(images):
            xref = image[0]

            image_data = pdf.extract_image(xref)

            image_bytes = image_data["image"]
            image_ext = image_data["ext"]

            image_path = (
                Path(image_dir)
                / f"page_{page_number}_image_{image_index}.{image_ext}"
            )

            with open(image_path, "wb") as f:
                f.write(image_bytes)

            image_elements.append(
                {
                    "image_path": str(image_path),
                    "source": pdf_path,
                    "page": page_number,
                    "type": "image",
                }
            )

    pdf.close()
    return text_elements, image_elements

# Chunk text elements
def chunk_text(text_elements: list) -> list:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
    )

    chunks = []

    for doc in text_elements:
        doc_id = str(uuid4())
        split_docs = splitter.split_documents([doc])
        for chunk in split_docs:
            chunk.metadata["type"] = "text"
            chunk.metadata["doc_id"] = doc_id
        chunks.extend(split_docs)
    return chunks

def encode_image(image_path: str) -> str:
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

# Describe image from gpt model
def describe_image(image_path: str, page: int, source: str) -> Document:
    image_base64 = encode_image(image_path)

    llm = ChatOpenAI(model="gpt-4o", temperature=0,)

    response = llm.invoke(
        [
            {
                "role": "system",
                "content": """
                You are analyzing an image extracted from a PDF.

                Describe the image in detail so that the description can be used
                for semantic search in a RAG system.

                Include:
                - What the image represents
                - Important objects or components
                - Text visible in the image
                - Relationships between components
                - Tables, charts, diagrams, or workflows
                - Important numbers or labels
                - Technical concepts shown in the image

                Do not mention that you are an AI.
                Do not make assumptions about information that is not visible.
                """,
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Describe this PDF image for retrieval.",
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_base64}"
                        },
                    },
                ],
            },
        ]
    )

    description = response.content

    return Document(
        page_content=description,
        metadata={
            "source": source,
            "page": page,
            "type": "image",
            "image_path": image_path,
        },
    )

# CLIP image embeddings
def get_clip_image_embedding(image_path: str) -> list:

    image = Image.open(image_path).convert("RGB")
    image = CLIP_PREPROCESS(image).unsqueeze(0)

    with torch.no_grad():
        embedding = CLIP_MODEL.encode_image(image)

    embedding = embedding / embedding.norm(
        dim=-1,
        keepdim=True,
    )

    return embedding[0].cpu().tolist()

# CLIP text embeddings
def get_clip_text_embedding(query: str) -> list:

    text = CLIP_TOKENIZER([query])
    with torch.no_grad():
        embedding = CLIP_MODEL.encode_text(text)
    embedding = embedding / embedding.norm(
        dim=-1,
        keepdim=True,
    )
    return embedding[0].cpu().tolist()

# Build image documents
def build_image_documents(image_elements: list) -> list:

    image_docs = []
    for image in image_elements:
        doc = describe_image(
            image_path=image["image_path"],
            page=image["page"],
            source=image["source"],
        )
        doc.metadata["doc_id"] = str(uuid4())
        image_docs.append(doc)
    return image_docs

# Fetch CLIP images
def retrieve_clip_images(clip_store, query: str, k: int = 3) -> list:
    query_embedding = get_clip_text_embedding(query)
    results = clip_store._collection.query(
        query_embeddings=[query_embedding],
        n_results=k,
    )
    documents = []
    for metadata in results["metadatas"][0]:
        documents.append(
            Document(
                page_content=metadata.get("description", ""),
                metadata=metadata,
            )
        )
    return documents

# Build CLIP store
def build_clip_store(image_docs: list):
    clip_store = Chroma(
        collection_name="clip_images",
        embedding_function=None,
        persist_directory=CLIP_CHROMA_DIR,
    )

    for doc in image_docs:
        embedding = get_clip_image_embedding(doc.metadata["image_path"])
        metadata = { **doc.metadata, "description": doc.page_content,}
        clip_store._collection.add(
            ids=[str(uuid4())],
            embeddings=[embedding],
            documents=[doc.page_content],
            metadatas=[metadata],
        )
    return clip_store

# Build vector store
def build_vector_store(text_chunks: list,image_docs: list) -> Chroma:
    documents = text_chunks + image_docs
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_store = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
        collection_name="multimodal_pdf",
    )
    return vector_store

# Retrieve documents
def retrieve_documents(vector_store: Chroma, query: str, k: int = 5,) -> list:
    results = vector_store.similarity_search(query, k=k)
    return results

# Build RAG pipeline
def build_multimodal_rag():
    # 1. Extract PDF
    text_elements, image_elements = extract_pdf_elements(PDF_PATH, IMAGE_DIR)
    print(f"Extracted text elements: {len(text_elements)}")
    print(f"Extracted images: {len(image_elements)}")

    # 2. Chunk text
    text_chunks = chunk_text(text_elements)
    print(f"Text chunks: {len(text_chunks)}")

    # 3. Generate image descriptions
    image_docs = build_image_documents(image_elements)
    print(f"Image descriptions: {len(image_docs)}")

    # 4. Existing OpenAI vector store
    vector_store = build_vector_store(text_chunks, image_docs)

    # 5. New CLIP image vector store
    clip_store = build_clip_store(image_docs)
    return vector_store, clip_store

def build_multimodal_context(documents: list) -> tuple[list, list]:

    text_context = []
    image_paths = []

    for doc in documents:
        if doc.metadata.get("type") == "image":
            image_paths.append(doc.metadata["image_path"])
            text_context.append(f"[Image description]\n{doc.page_content}")
        else:
            text_context.append(f"[Text]\n{doc.page_content}")
    return text_context, image_paths

def generate_answer(query: str, documents: list,) -> str:

    text_context, image_paths = build_multimodal_context(documents)
    context = "\n\n".join(text_context)
    content = [
        {
            "type": "text",
            "text": f"""
            Answer the user's question using the provided PDF context.

            User question:
            {query}

            Retrieved context:
            {context}

            Use the original images when they provide useful visual
            information.

            Do not invent information that is not present in the
            retrieved context or images.
            """,
        }
    ]

    # Add original images
    for image_path in image_paths:
        image_base64 = encode_image(image_path)
        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{image_base64}"
                },
            }
        )

    llm = ChatOpenAI(model="gpt-4o", temperature=0,)

    response = llm.invoke(
        [
            {
                "role": "user",
                "content": content,
            }
        ]
    )
    return response.content

def run_qna(vector_store: Chroma, clip_store):
    while True:
        query = input("\nAsk a question (type 'exit' to quit): ").strip()
        if query.lower() == "exit":
            print("Exiting...")
            break
        if not query:
            continue

        # Retrieve relevant text/image descriptions
        semantic_documents = retrieve_documents(vector_store, query, k=5)
        clip_documents = retrieve_clip_images(clip_store, query, k=3)

        documents = semantic_documents + clip_documents

        # Generate answer using retrieved context
        answer = generate_answer(query, documents)

        print("\nAnswer:")
        print(answer)

if __name__ == "__main__":
    vector_store, clip_storage = build_multimodal_rag()
    run_qna(vector_store, clip_storage)
    
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
import base64
from dotenv import load_dotenv

PDF_PATH = "Azure-Kubernetes-Service.pdf"
IMAGE_DIR = "extracted_images"
CHROMA_DIR = "chroma_db"

load_dotenv()

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

def chunk_text(text_elements: list) -> list:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
    )

    chunks = []

    for doc in text_elements:
        split_docs = splitter.split_documents([doc])
        for chunk in split_docs:
            chunk.metadata["type"] = "text"
        chunks.extend(split_docs)
    return chunks

def encode_image(image_path: str) -> str:
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")
    
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

def build_image_documents(image_elements: list) -> list:

    image_docs = []
    for image in image_elements:
        doc = describe_image(
            image_path=image["image_path"],
            page=image["page"],
            source=image["source"],
        )
        image_docs.append(doc)
    return image_docs

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

def retrieve_documents(vector_store: Chroma, query: str, k: int = 5,) -> list:
    results = vector_store.similarity_search(
        query,
        k=k,
    )
    return results

def build_multimodal_context(documents: list,) -> tuple[list, list]:

    text_context = []
    image_paths = []

    for doc in documents:
        if doc.metadata.get("type") == "image":
            image_paths.append(
                doc.metadata["image_path"]
            )
            text_context.append(
                f"[Image description]\n{doc.page_content}"
            )
        else:
            text_context.append(
                f"[Text]\n{doc.page_content}"
            )
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

def build_multimodal_rag():

    # 1. Extract PDF
    text_elements, image_elements = extract_pdf_elements(PDF_PATH, IMAGE_DIR,)
    print(f"Extracted text elements: {len(text_elements)}")
    print(f"Extracted images: {len(image_elements)}")

    # 2. Chunk text
    text_chunks = chunk_text(text_elements)
    print(f"Text chunks: {len(text_chunks)}")

    # 3. Describe images
    image_docs = build_image_documents(image_elements)
    print(f"Image descriptions: {len(image_docs)}")

    # 4. Build vector store
    vector_store = build_vector_store(text_chunks, image_docs,)
    return vector_store

def run_qna(vector_store: Chroma):
    while True:
        query = input("\nAsk a question (type 'exit' to quit): ").strip()
        if query.lower() == "exit":
            print("Exiting...")
            break
        if not query:
            continue

        # Retrieve relevant text/image descriptions
        documents = retrieve_documents(
            vector_store,
            query,
            k=5,
        )

        # Generate answer using retrieved context
        answer = generate_answer(
            query,
            documents,
        )

        print("\nAnswer:")
        print(answer)

if __name__ == "__main__":
    vector_store = build_multimodal_rag()
    run_qna(vector_store)
    
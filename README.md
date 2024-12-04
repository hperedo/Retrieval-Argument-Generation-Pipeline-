# RAG Pipeline for LaTeX Documents
This project implements a Retrieval-Augmented Generation (RAG) pipeline for querying and generating responses based on internal LaTeX documents. The solution uses AWS services, OpenSearch, and OpenAI GPT to process and query documents.

# Features
- Preprocess LaTeX documents into plain text.
- Store embeddings in AWS OpenSearch for semantic retrieval.
- Retrieve relevant content for user queries using semantic similarity.
- Generate responses using OpenAI GPT or AWS Bedrock.
- Deployable on AWS with Lambda and API Gateway.

# Architecture
- Document Upload: Store LaTeX files in an S3 bucket.
- Preprocessing: A Lambda function extracts and processes text from LaTeX files.
- Embedding: Generate embeddings using a Sentence Transformer model.
- Indexing: Store embeddings in OpenSearch for retrieval.
- Querying: Users query the system via an API Gateway endpoint.
- Generation: Retrieve relevant content and generate answers using OpenAI GPT or AWS Bedrock.

# Requirements
- Python Libraries
   - boto3: AWS SDK for Python.
   - sentence-transformers: For embedding generation.
   - opensearch-py: OpenSearch client.
   - openai: OpenAI GPT API.
   - re: For LaTeX preprocessing.

 # AWS Services
   - S3: Store LaTeX documents.
   - Lambda: Handle preprocessing, embedding, and querying.
   - OpenSearch: Store and retrieve embeddings.
   - API Gateway: Expose the pipeline via an HTTP endpoint.




# Retrieval Augmented Generation (RAG) pipeline on top of internal LaTeX documents! Here’s a step by step guide tailored to this setup:

# 1. Understand RAG
RAG combines:

- Retriever: Searches through documents for relevant context.
- Generator: Generates responses based on retrieved context (e.g., using a large language model).

For LaTeX documents, the pipeline needs:

  - 1. Conversion of LaTeX content into a searchable format (text, embeddings).
  - 2. A retriever (like vector search).
  - 3. A generator that integrates retrieved context.

# 2. Pipeline Components

## Step 1: Preprocessing LaTeX Documents
Extract text from LaTeX:

- Use Python libraries like pytx, PyLaTeX, or pdflatex to process .tex files.
- Strip out LaTeX-specific commands (\section{}, \cite{}, etc.) unless relevant.
- Preserve meaningful structure (headings, equations, etc.) in metadata or structured text.

Optionally include:

- Math Extraction: Convert math expressions to MathML or keep LaTeX notation.
- Metadata: Titles, sections, keywords for more structured queries.

## Step 2: Indexing the Extracted Text
Embed the processed text for search:

- Use a text embedding model, e.g., OpenAI’s embeddings, Hugging Face models, or Sentence Transformers.
- Split documents into chunks (e.g., ~500 tokens) for embedding.

Store embeddings in a vector database:

- Popular options: Pinecone, Weaviate, FAISS, or Milvus.

## Step 3: Retriever
- Query the vector database for relevant chunks using semantic similarity.
- Retrieve top-N matches to pass to the generator.
## Step 4: Generator
- Use a language model (e.g., GPT or similar) to:
- Take the user query + retrieved context.
- Generate a coherent response incorporating the provided context.

## Step 5: Post-Processing
Optionally include:
- Formatting LaTeX-based outputs (if required by your application).
- Citations or traceability to original documents.

 # 3. Implementation Details
Libraries & Tools:
- Text extraction: pytx, textract, or custom LaTeX parsers.
- Embedding models:
  - OpenAI’s text-embedding-ada-002.
  -  Hugging Face models like all-MiniLM-L6-v2.
- Vector database: FAISS for local use; Pinecone or Weaviate for scalable cloud solutions.
- Generator: OpenAI's GPT API or open-source alternatives like LLaMA, Falcon, or Flan-T5.

  Example Workflow (Python):
```python
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI

# Step 1: Load and preprocess LaTeX files (convert to text)
loader = PyPDFLoader("example.pdf")  # Alternatively, preprocess LaTeX manually
documents = loader.load()

# Step 2: Embed and store in a vector database
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(documents, embeddings)

# Step 3: Create retriever and generator pipeline
retriever = vectorstore.as_retriever()
llm = OpenAI(model="gpt-4")  # Generator
qa_chain = RetrievalQA(llm=llm, retriever=retriever)

# Step 4: Query the system
query = "Explain the equation in Section 2."
response = qa_chain.run(query)

print(response)
```

# 4. Challenges and Solutions
- LaTeX Complexity: Extract equations properly or tokenize math for semantic understanding.
  - Solution: Retain math expressions as-is or translate them to MathML.
- Search Relevance: Tuning embedding models for domain-specific LaTeX content.
  - Solution: Fine-tune embeddings or leverage domain-specific pre-trained models.
- Traceability: Ensuring answers reference exact sections.
  - Solution: Add metadata (e.g., section headers, document IDs).

# 5. Enhancements
- Include a GUI for document upload and query submission (e.g., Streamlit or Flask).
- Allow multi-modal queries (e.g., text + equations).
- Support citation generation with links to sections or equations.

# Implementation of a Retrieval-Augmented Generation (RAG) pipeline using Python and AWS services, covering all five steps

## Step 1: Preprocessing LaTeX Documents
Goal:
Extract readable text from .tex files and structure it for indexing.

Code:
```python
import os
import re

def preprocess_latex(file_path):
    """
    Extract readable text from a LaTeX document.
    Removes LaTeX commands and retains plain text with some structure.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        latex_content = f.read()

    # Remove LaTeX commands and keep text
    text = re.sub(r"\\[a-zA-Z]+\{.*?\}", "", latex_content)  # Remove commands like \section{}
    text = re.sub(r"%.*", "", text)  # Remove comments
    text = re.sub(r"\s+", " ", text)  # Normalize whitespace

    return text.strip()

# Example usage:
file_path = "example.tex"
processed_text = preprocess_latex(file_path)

# Split into chunks (useful for embedding later)
def chunk_text(text, chunk_size=500):
    words = text.split()
    return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

chunks = chunk_text(processed_text)

```
## Step 2: Embedding and Indexing
Goal:
Create embeddings for chunks of text and store them in a vector database using AWS OpenSearch or DynamoDB + Faiss.

Code:

```python
from sentence_transformers import SentenceTransformer
from opensearchpy import OpenSearch

# Step 1: Generate Embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')  # Choose your embedding model
embeddings = [model.encode(chunk) for chunk in chunks]

# Step 2: Index Embeddings in AWS OpenSearch
def index_to_opensearch(embeddings, chunks, index_name="latex_index"):
    client = OpenSearch(
        hosts=[{"host": "your-opensearch-endpoint", "port": 443}],
        http_auth=('username', 'password'),
        use_ssl=True,
        verify_certs=True,
    )

    # Create the index if it doesn't exist
    if not client.indices.exists(index=index_name):
        client.indices.create(index=index_name)

    # Add embeddings and text chunks to the index
    for i, (embedding, text) in enumerate(zip(embeddings, chunks)):
        doc = {
            "id": i,
            "embedding": embedding.tolist(),
            "text": text
        }
        client.index(index=index_name, body=doc)

# Call the function to index
index_to_opensearch(embeddings, chunks)
```

## Step 3: Retriever
Goal:
Query the vector database for relevant chunks.

Code:

 ```python
import numpy as np

def search_opensearch(query, index_name="latex_index", top_k=3):
    query_embedding = model.encode(query)

    # Query OpenSearch
    response = client.search(
        index=index_name,
        body={
            "query": {
                "script_score": {
                    "query": {"match_all": {}},
                    "script": {
                        "source": "cosineSimilarity(params.query_vector, doc['embedding']) + 1.0",
                        "params": {"query_vector": query_embedding.tolist()}
                    }
                }
            },
            "size": top_k
        }
    )

    results = [
        {
            "text": hit["_source"]["text"],
            "score": hit["_score"]
        }
        for hit in response["hits"]["hits"]
    ]
    return results

# Example usage
query = "Explain the equation for Fourier transform."
retrieved_docs = search_opensearch(query)
```

## Step 4: Generator
Goal:
Use AWS Lambda or an external API (e.g., OpenAI's GPT) to generate responses based on retrieved context.

Code:

 ```python
import openai

openai.api_key = "your_openai_api_key"

def generate_response(query, context):
    """
    Generate a response using OpenAI's GPT model.
    """
    prompt = f"Answer the query based on the context provided.\n\nQuery: {query}\n\nContext: {context}\n\nAnswer:"
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=300
    )
    return response["choices"][0]["text"].strip()

# Example usage:
context = " ".join([doc["text"] for doc in retrieved_docs])
response = generate_response(query, context)
print(response)
 ```

## Step 5: Post-Processing and Deployment
Goal:
Wrap the pipeline into a reusable and deployable architecture with AWS Lambda and API Gateway.

Deployment Architecture:
  - Upload documents to an S3 bucket.
  - AWS Lambda triggers preprocessing, embedding, and indexing into OpenSearch.
  - API Gateway handles user queries, invoking a Lambda function to:
     - Retrieve relevant chunks from OpenSearch.
     - Generate responses using OpenAI GPT or AWS Bedrock.
Example Lambda Handler:

 ```python
import json

def lambda_handler(event, context):
    query = event["queryStringParameters"]["query"]

    # Step 1: Retrieve context
    retrieved_docs = search_opensearch(query)
    context = " ".join([doc["text"] for doc in retrieved_docs])

    # Step 2: Generate response
    response = generate_response(query, context)

    # Return response
    return {
        "statusCode": 200,
        "body": json.dumps({"response": response})
    }
 ```

# Complete Workflow
- LaTeX documents: Preprocess and chunk into text.
- Embeddings: Use a model to generate embeddings and store them in OpenSearch.
- Retriever: Query OpenSearch for relevant context.
- Generator: Use OpenAI’s GPT or an AWS service to generate answers.
- Deployment: Deploy on AWS with S3, Lambda, and API Gateway.

# Setup Instructions

## 1. Clone the Repository
```bash
git clone https://github.com/your-repo/rag-latex-pipeline.git
cd rag-latex-pipeline
```

## 2. Preprocessing LaTeX Documents
Extract text from .tex files:

```python
import os
import re

def preprocess_latex(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        latex_content = f.read()
    text = re.sub(r"\\[a-zA-Z]+\{.*?\}", "", latex_content)  # Remove commands
    text = re.sub(r"%.*", "", text)  # Remove comments
    text = re.sub(r"\s+", " ", text)  # Normalize whitespace
    return text.strip()

```

## 3. Embedding and Indexing
Generate embeddings and store them in OpenSearch:
```python
from sentence_transformers import SentenceTransformer
from opensearchpy import OpenSearch

model = SentenceTransformer('all-MiniLM-L6-v2')
chunks = ["chunk1 text", "chunk2 text"]  # Replace with actual chunks
embeddings = [model.encode(chunk) for chunk in chunks]

client = OpenSearch(
    hosts=[{"host": "your-opensearch-endpoint", "port": 443}],
    http_auth=('username', 'password'),
    use_ssl=True,
    verify_certs=True,
)

for i, (embedding, text) in enumerate(zip(embeddings, chunks)):
    client.index(index="latex_index", body={"id": i, "embedding": embedding.tolist(), "text": text})

```
## 4. Querying
Retrieve relevant chunks based on a query:


```python
def search_opensearch(query):
    query_embedding = model.encode(query)
    response = client.search(
        index="latex_index",
        body={
            "query": {
                "script_score": {
                    "query": {"match_all": {}},
                    "script": {
                        "source": "cosineSimilarity(params.query_vector, doc['embedding']) + 1.0",
                        "params": {"query_vector": query_embedding.tolist()}
                    }
                }
            },
            "size": 3
        }
    )
    return [hit["_source"]["text"] for hit in response["hits"]["hits"]]



```

## 5. Response Generation
Use OpenAI GPT for generation:
```python
import openai
openai.api_key = "your_openai_api_key"

def generate_response(query, context):
    prompt = f"Answer the query based on the context provided.\n\nQuery: {query}\n\nContext: {context}\n\nAnswer:"
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=300
    )
    return response["choices"][0]["text"].strip()

```

## 6. Deploy on AWS
 - Lambda: Upload functions to AWS Lambda for:
   - Preprocessing and embedding.
   - Querying and response generation.
 - API Gateway: Set up an endpoint to handle user queries.
   - Example Lambda handler:

```python
def lambda_handler(event, context):
    query = event["queryStringParameters"]["query"]
    retrieved_docs = search_opensearch(query)
    context = " ".join(retrieved_docs)
    response = generate_response(query, context)
    return {"statusCode": 200, "body": json.dumps({"response": response})}
```

# Usage
  - Upload LaTeX documents to the S3 bucket.
  - Query the system via API Gateway
    
```bash
curl -X GET "https://your-api-gateway-endpoint?query=Explain Fourier Transform."
```

  - Receive a JSON response with the generated answer.

 # Future Enhancements
 - Support for equations (e.g., MathML extraction).
 - Interactive front-end for queries and uploads.
 - Optimize retrieval and embeddings for mathematical content

 # License
This project is licensed under the MIT License.

# # rag_utils.py
# import os
# import tempfile
# from langchain_community.document_loaders import PyPDFLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.embeddings import HuggingFaceEmbeddings
# from langchain.vectorstores import Chroma
# from openai import OpenAI
# from dotenv import load_dotenv

# load_dotenv()
# client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# def process_pdf_and_ask(uploaded_pdf, question):
#     # Save the uploaded PDF to a temporary file
#     with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
#         tmp_file.write(uploaded_pdf.read())
#         tmp_pdf_path = tmp_file.name

#     # Load PDF
#     loader = PyPDFLoader(tmp_pdf_path)
#     documents = loader.load()

#     # Chunking
#     tokenizer = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
#     splitter = RecursiveCharacterTextSplitter(
#         chunk_size=512,
#         chunk_overlap=50,
#         separators=["\n\n", "\n", " ", ""]
#     )
#     docs = splitter.split_documents(documents)

#     # Embedding + saving to Chroma (in-memory)
#     vectordb = Chroma.from_documents(
#         documents=docs,
#         embedding=tokenizer,
#         collection_name="rag_pdf_temp",
#     )

#     # Retrieve similar chunks
#     results = vectordb.similarity_search(question, k=3)
#     context = "\n\n".join([doc.page_content for doc in results])

#     # Use LLM to answer
#     prompt = f"""Based on the following document excerpts, answer the question at the end as clearly and accurately as possible.

# Document Chunks:
# {context}

# Question: {question}
# Answer:"""

#     response = client.chat.completions.create(
#         model="gpt-3.5-turbo",
#         messages=[
#             {"role": "user", "content": prompt}
#         ]
#     )

#     final_answer = response.choices[0].message.content.strip()
#     return final_answer


#---------------------------------------------------------------------------------------------------

import os
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

def process_pdf_and_ask(uploaded_pdf, question):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_pdf.read())
        tmp_pdf_path = tmp_file.name

    loader = PyPDFLoader(tmp_pdf_path)
    documents = loader.load()
    return rag_answer_from_text(question, documents)

def rag_answer_from_text(question, documents):
    tokenizer = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=50,
        separators=["\n\n", "\n", " ", ""]
    )
    docs = splitter.split_documents(documents)

    vectordb = Chroma.from_documents(
        documents=docs,
        embedding=tokenizer,
        collection_name="rag_pdf_temp",
    )

    results = vectordb.similarity_search(question, k=3)
    context = "\n\n".join([doc.page_content for doc in results])

    prompt = f"""Based on the following document excerpts, answer the question at the end as clearly and accurately as possible.

Document Chunks:
{context}

Question: {question}
Answer:"""

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message.content.strip()

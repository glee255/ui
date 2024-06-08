
import streamlit as st import streamlit_session_state as SessionState session_state = SessionState.get(data=None)
import subprocess
import sys

# Install PyPDF2 if not already installed
subprocess.check_call([sys.executable, "-m", "pip", "install", "PyPDF2"])
import PyPDF2
import pandas as pd

file_path = "Resume_Gayeon Lee.pdf"

# Open the PDF file
with open(file_path, "rb") as file:
    # Create a PDF file reader object
    pdf_reader = PyPDF2.PdfReader(file)

    # Initialize an empty string to store the text
    text = ""

    # Iterate through each page of the PDF
    for page_num in range(len(pdf_reader.pages)):
        # Get the page object
        page = pdf_reader.pages[page_num]

        # Extract text from the page
        text += page.extract_text()

df1 = pd.DataFrame({'Text': [text]})

import string
import re

def clean(doc):
    # Remove specific characters
    doc = re.sub(r'•', ' ', doc, flags=re.IGNORECASE)
    doc = re.sub(r'\uf0b7', ' ', doc, flags=re.IGNORECASE)
    # Remove blank spaces
    doc = ' '.join(doc.split())

    return doc

df1['cleaned'] = df1['Text'].apply(clean)

"""Hi, I am Gayeon Lee. I am a passionate Data Professional.
   Explore my projects and skills to learn more about my potentials."""

# from sentence_transformers import SentenceTransformer
# import streamlit as sl

# vectorizer = SentenceTransformer('all-MiniLM-L12-v2')
# sl.header("Explore my projects and skills to learn more about my potentials")

# Retrieve
# q =sl.text_input("enter question") # "Does she have SQL skills?"
# query_vector = vectorizer.encode([q])


from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document
import pandas as pd
import PyPDF2

# Convert DataFrame rows to Document objects
# document_objects = [Document(page_content=row['cleaned']) for _, row in df1.iterrows()]

# Initialize CharacterTextSplitter
# text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)

# Split the documents into chunks
# chunks = text_splitter.split_documents(document_objects)
# data_list = [chunk.dict() for chunk in chunks]
# df_chunks = pd.DataFrame(data_list)

columns_to_include = ['cleaned']
data1 = df1[columns_to_include]
data_list = list(data1.to_records(index=False))
data = [f"{columns_to_include[0]}:{e[0]}\n" for e in data_list]

@st.cache 
def load_data(): # 데이터 로딩 함수 예시 
    
    return data


# Display the chunks
#for chunk in chunks:
    #print(chunk.page_content)

from sentence_transformers import SentenceTransformer
import numpy as np
import os
import pickle
vectorizer = SentenceTransformer('all-MiniLM-L12-v2')


if os.path.isfile("vector_cache"):
    with open("vector_cache", 'rb') as f:
        comment_vectors = pickle.load(f)
    f.close()
else:
    comment_vectors = vectorizer.encode(data)
    with open("vector_cache", 'wb') as f:
        pickle.dump(comment_vectors,f)
    f.close()


import faiss

dimension = comment_vectors.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(comment_vectors)

def retrieve(query, k=1):
    query_vector = vectorizer.encode([q], convert_to_tensor=True).cpu().numpy()
    distances, indices = index.search(query_vector, k)
    return [data[i] for i in indices[0]]


# Retrieve
import streamlit as sl

sl.header("Enter any questions you would like to know.")

# Input box for the question
q = sl.text_input("Your question")

#q =sl.text_input("enter question") # "Does she have SQL skills?"
if q=='':
    sl.write('')
else:
    retrieved_documents = retrieve(q, k=1)

# q = "Does she have SQL skills?"
# retriever = comment_vectors.as_retriever(search_kwargs={'k':1})

    from langchain_community.llms import HuggingFaceEndpoint

    repo_id = "mistralai/Mistral-7B-Instruct-v0.2"

    # auth to huggingface
    HUGGINGFACEHUB_API_TOKEN = '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'
    import os
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = HUGGINGFACEHUB_API_TOKEN

    llm = HuggingFaceEndpoint(
    repo_id=repo_id,
    max_length=1024,
    temperature=0.1,
    token=HUGGINGFACEHUB_API_TOKEN
    )

    from langchain.prompts import PromptTemplate
    from langchain.chains import LLMChain

    # Entire document
    # info = ""
    #for document in documents:
    #    info += document.page_content + "\n"

    info=""

    if q:
        documents = retrieved_documents
        info = ""
        for document in documents:
            info += document + "\n\n"

    # create template for LLM
    template = """Question: {question}
    Answer: Let's think step by step. The information can be used \n###\n{info}"""
    prompt = PromptTemplate.from_template(template)

    # post q + info to LLM
    llm_chain = LLMChain(prompt=prompt, llm=llm)
    result = llm_chain.invoke({"question": q, "info": info}, temperature=0.1)
    sl.write(result['text'])


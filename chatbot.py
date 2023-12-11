import os
from langchain.document_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import time
from langchain.embeddings import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer

print("Script is running")
os.environ['OPENAI_API_KEY'] = "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"

loader = CSVLoader(file_path="C:/Users/Snigdha Mundra/Documents/Dummy Data/work_dummy_data.csv")
data = loader.load()

#start_time = time.time()
#text_splitter_csv = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
#all_splits_csv = text_splitter_csv.split_documents(data)
#end_time = time.time()
#elapsed_time = end_time - start_time
#print(f"Time taken: {elapsed_time} seconds")

vector_store2 = FAISS.from_documents(data, HuggingFaceEmbeddings())
retriever2 = vector_store2.as_retriever()

template2 = """Answer the question based only on the following context:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template2)
llm = ChatOpenAI()
llm_chain = LLMChain(prompt=prompt, llm=llm)

chain = (
    {"context": retriever2, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

result = llm_chain.invoke("How many software engineers are there? ")
print(result)

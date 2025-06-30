import os

import pandas as pd
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.document_loaders import DataFrameLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import (ChatGoogleGenerativeAI,
                                    GoogleGenerativeAIEmbeddings)

# Load environment variables
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

vectordb_pdf_path = "pdf_db"
vectordb_faq_path = "faq_db"

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.7,google_api_key=api_key)
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

def load_pdf():
    loader = PyPDFLoader('./data/bsrm.pdf')
    data = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(data)
    return docs

def update_csv(question,answer ):

    new_row = {
        "Question": question,
        "Answer": answer,
        "category": "",  # Optional: replace if needed
        "link": ""       # Optional: replace if needed
    }

    pd.DataFrame([new_row]).to_csv('./data/faq_bsrm.csv', mode='a', header=False, index=False)  
 


def load_csv_combined():
    df = pd.read_csv('./data/faq_bsrm.csv')
    df.dropna(subset=["Question", "Answer"], inplace=True)

    # Combine Question and Answer into one string for embedding
    df["content"] = "Question: " + df["Question"] + "\n\nAnswer: " + df["Answer"]
    # Use DataFrameLoader with the combined content column
    loader = DataFrameLoader(df, page_content_column="content")
    documents = loader.load()
    return documents




def create_vector_db(question,answer):
    update_csv(question,answer)
    docs =  load_csv_combined()
    vectordb_file_path =  vectordb_faq_path

    # Create a FAISS instance for vector database from 'data'
    vectordb = FAISS.from_documents(documents=docs,
                                    embedding=embeddings)
    
    # Save vector database locally
    vectordb.save_local(vectordb_file_path)


def get_qa_chain(type):
    vectordb_file_path = vectordb_pdf_path if type == "pdf" else vectordb_faq_path
    index_path = os.path.join(vectordb_file_path, "index.faiss")
    if not (os.path.exists(index_path)) :
        return None
    # Load the vector database from the local folder
    vectordb = FAISS.load_local(vectordb_file_path, embeddings,allow_dangerous_deserialization=True)

    # Initialize retriever
    retriever = vectordb.as_retriever(score_threshold=0.7)


    # Define the system prompt
    prompt_template = """Given the following context and a question, generate an answer based on this context only. 
    response should be a good looking sentence in HTML format. If link is available in the context, include the link in response with labeling Learn more here in a new line. 
    If found list use bullet points in the response.
    Do not use P tag in the response.
    If the answer is not found in the context, strictly return "DATA_NOT_FOUND" as code.

    CONTEXT: {context}

    QUESTION: {question}"""


    # Create the chat prompt template
    prompt = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )


    chain = RetrievalQA.from_chain_type(llm=llm,
                                    chain_type="stuff",
                                    retriever=retriever,
                                    input_key="question",
                                    return_source_documents=True,
                                    chain_type_kwargs={"prompt": prompt})

    return chain

# async def generate_content(query,chat_history,session_id):
#     try:
#         response = rag_chain.invoke({
#             "input": query,
#             "chat_history": chat_history  # Include chat history
#         })
#         cleaned_output = response["answer"].replace("```html", "").replace("```", "").replace("\n", "").strip()
#         return {"response": cleaned_output,"session_id": session_id}
#     except Exception as e:
#         logging.error(f"Error in generate_content: {e}")
#         return {"response": str(e),"session_id": session_id}

import streamlit as st

from langchain_helper import create_vector_db, get_qa_chain

st.sidebar.title("Knowledgebase")
btnFaq = st.sidebar.button("Generate FAQ Knowledgebase")
btnPdf = st.sidebar.button("Generate PDF Knowledgebase")
status_box = st.sidebar.empty()

if btnFaq:
    status_box.info("Generating knowledgebase from FAQ... This may take a few minutes.")
    create_vector_db("faq")
    status_box.success("FAQ Knowledgebase generated successfully!")

if btnPdf:
    status_box.info("Generating knowledgebase from PDF... This may take a few minutes.")
    create_vector_db("pdf")
    status_box.success("PDF Knowledgebase generated successfully!")

st.title("Frequently Asked Questions 🌱")
question = st.text_input("Ask your question:")

NO_DATA = "DATA_NOT_FOUND"

def get_response(question, source_type, box):
    chain = get_qa_chain(source_type)
    if chain is None:
        box.error(f"Knowledgebase for {source_type} not generated yet. Please generate it from the sidebar.")
        st.stop()
    box.info("Generating answer...")
    response = chain(question)
    if response["result"] == NO_DATA:
        return NO_DATA
    return f"{response['result']}\n\n_Source: {source_type}_"

if question:
    answer_box = st.empty()
    result = get_response(question, "faq", answer_box)

    if result == NO_DATA:
        result = get_response(question, "pdf", answer_box)

    if result == NO_DATA:
        answer_box.warning("Sorry, no data found in FAQ or PDF knowledgebase.")
    else:
        answer_box.markdown(f"## Answer\n\n{result}", unsafe_allow_html=True)

import time

import streamlit as st

from langchain_helper import create_vector_db, get_qa_chain

st.sidebar.title("Admin Panel:")
st.sidebar.subheader("Update your knowledgebase:-")
st.sidebar.write(" ")
with st.sidebar.container():
    question =st.sidebar.text_input("Question", key="question")
    answer=st.sidebar.text_area("Answer:", height=200,key="answer")

btnFaq = st.sidebar.button("Add to Knowledgebase")
status_box = st.sidebar.empty()




if btnFaq:
    if question == "" or answer == "":
        status_box.error("Please fill the knowledgebase.")
        time.sleep(2)
        status_box.empty()
    else:
        status_box.status("Updating knowledgebase... This may take a few minutes.")
        create_vector_db(question, answer)
        status_box.success("✅ Knowledgebase added successfully!")
        time.sleep(1)
        status_box.empty()


# if btnPdf:
#     status_box.info("Generating knowledgebase from PDF... This may take a few minutes.")
#     create_vector_db("pdf")
#     status_box.success("PDF Knowledgebase generated successfully!")

st.title("🤖 BSRM Chatbot")
st.text(" ")
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
    # return f"{response['result']}\n\n_Source: {source_type}_"
    return f"{response['result']}"

if question:
    answer_box = st.empty()
    result = get_response(question, "faq", answer_box)

    if result == NO_DATA:
        result = get_response(question, "pdf", answer_box)

    if result == NO_DATA:
        answer_box.warning("Sorry, no data found in FAQ or PDF knowledgebase.")
    else:
        answer_box.markdown(f"## Answer\n\n{result}", unsafe_allow_html=True)

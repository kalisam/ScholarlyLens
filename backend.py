import streamlit as st
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains import RetrievalQA
import tempfile

# Initialize session state
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'embeddings' not in st.session_state:
    st.session_state.embeddings = HuggingFaceEmbeddings()

# Upload PDF
upload_pdf_file = st.file_uploader("Upload a PDF file", type="pdf")

if upload_pdf_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(upload_pdf_file.getvalue())
        temp_pdf_path = temp_file.name

    try:
        # Load and process the PDF
        loader = PDFPlumberLoader(temp_pdf_path)
        docs = loader.load()

        # Create text chunks
        text_splitter = SemanticChunker(st.session_state.embeddings)
        documents = text_splitter.split_documents(docs)

        # Create vector store
        st.session_state.vector_store = FAISS.from_documents(documents, st.session_state.embeddings)
        st.success("PDF processed successfully!")
    except Exception as e:
        st.error(f"Error processing PDF: {str(e)}")

# Only proceed if vector store exists
if st.session_state.vector_store:
    retriever = st.session_state.vector_store.as_retriever(search_kwargs={"k": 3})
    llm = Ollama(model="deepseek-r1")

    # Prompt templates
    QA_CHAIN_PROMPT = PromptTemplate.from_template(
        """  
        1. Use ONLY the context below.  
        2. If unsure, say "I don't know".  
        3. Keep answers under 4 sentences.  

        Context: {context}  

        Question: {question}  

        Answer:  
        """
    )

    llm_chain = LLMChain(llm=llm, prompt=QA_CHAIN_PROMPT)
    
    document_prompt = PromptTemplate(
        template="{page_content}",
        input_variables=["page_content"]
    )

    # # Create QA chain
    # qa = RetrievalQA.from_chain_type(
    #     retriever=retriever,
    #     combine_documents_chain=StuffDocumentsChain(
    #         llm_chain=llm_chain,
    #         document_prompt=document_prompt,
    #         document_variable_name="context"
    #     )
    # )

    # Create QA chain
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever
    )

    # Streamlit UI
    user_input = st.text_input("Ask your PDF a question:")

    if user_input:
        with st.spinner("Thinking..."):
            try:
                response = qa.run(user_input)
                st.write(response)
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
else:
    st.info("Please upload a PDF file first.")

import streamlit as st
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.text_splitter import CharacterTextSplitter
from PyPDF2 import PdfReader
import os
from dotenv import load_dotenv

# Load API key from .env
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# Initialize LangChain with OpenAI
llm = OpenAI(api_key=api_key, temperature=0.7)
prompt_template = PromptTemplate(
    input_variables=["text"],
    template="Summarize the following text:\n\n{text}"
)
chain = LLMChain(llm=llm, prompt=prompt_template)

# Streamlit app
st.title("PDF Summarizer")
st.write("Upload a PDF, and I'll summarize it for you!")

# File uploader
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:
    # Read PDF content
    pdf_reader = PdfReader(uploaded_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()

    # Split text for summarization
    text_splitter = CharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    chunks = text_splitter.split_text(text)

    # Summarize each chunk
    summary = []
    for chunk in chunks:
        response = chain.run({"text": chunk})
        summary.append(response)

    # Display summary
    st.subheader("Summary")
    st.write(" ".join(summary))

    # Download summary as a text file
    summary_text = "\n\n".join(summary)
    st.download_button("Download Summary", summary_text, "summary.txt", "text/plain")

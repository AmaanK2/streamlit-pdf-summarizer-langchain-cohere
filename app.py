import streamlit as st
from PyPDF2 import PdfReader
from langchain.llms import Cohere
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.text_splitter import CharacterTextSplitter
from dotenv import load_dotenv
import os

# Load API key from .env
load_dotenv()
cohere_api_key = os.getenv("COHERE_API_KEY")

# Initialize Cohere LLM through LangChain
llm = Cohere(cohere_api_key=cohere_api_key, model="command-xlarge", temperature=0.5)
prompt_template = PromptTemplate(
    input_variables=["text"],
    template="Summarize the following text:\n\n{text}\n\nSummary:"
)
chain = LLMChain(llm=llm, prompt=prompt_template)

# Streamlit app
st.title("PDF Summarizer with Cohere and LangChain")
st.write("Upload a PDF file, and I'll summarize its content using LangChain and Cohere!")

# Upload a PDF file
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:
    # Extract text from the PDF
    st.info("Extracting text from PDF...")
    pdf_reader = PdfReader(uploaded_file)
    text = "".join(page.extract_text() for page in pdf_reader.pages)

    if len(text.strip()) == 0:
        st.error("No readable text found in the PDF.")
    else:
        st.success("Text successfully extracted from the PDF!")

        # Display the first 500 characters of the PDF content
        st.subheader("Extracted Text (Preview)")
        st.write(text[:500] + "...")

        # Summarize the text
        st.subheader("Summary")
        st.info("Summarizing the PDF content with Cohere and LangChain...")
        try:
            # Split text into manageable chunks
            text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = text_splitter.split_text(text)

            # Summarize each chunk
            summaries = []
            for chunk in chunks:
                summary = chain.run({"text": chunk})
                summaries.append(summary)

            # Combine summaries
            final_summary = " ".join(summaries)

            # Display the summary
            st.write(final_summary)

            # Allow download of the summary as a text file
            st.download_button(
                "Download Summary",
                final_summary,
                file_name="summary.txt",
                mime="text/plain",
            )
        except Exception as e:
            st.error(f"An error occurred during summarization: {str(e)}")

import streamlit as st
import fitz  # PyMuPDF
import openai
import numpy as np
import os
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from io import BytesIO
from dotenv import load_dotenv
load_dotenv()

# Set your OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

def extract_text_from_pdf(pdf_path):
    """Load PDF and extract text."""
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        st.error(f"Error opening {pdf_path}: {e}")
        return ""
    
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def split_text(text, max_tokens=3000):
    """Split text into chunks of max_tokens size."""
    sentences = text.split('. ')
    current_chunk = []
    current_length = 0
    chunks = []
    
    for sentence in sentences:
        sentence_length = len(sentence.split())
        if current_length + sentence_length > max_tokens:
            chunks.append('. '.join(current_chunk) + '.')
            current_chunk = [sentence]
            current_length = sentence_length
        else:
            current_chunk.append(sentence)
            current_length += sentence_length
    
    if current_chunk:
        chunks.append('. '.join(current_chunk) + '.')
    
    return chunks

def get_embedding(text, model="text-embedding-ada-002"):
    """Generate embeddings for the given text using OpenAI's API."""
    response = openai.Embedding.create(
        input=text,
        model=model
    )
    return response['data'][0]['embedding']

def generate_embeddings_for_chunks(chunks):
    """Generate embeddings for each chunk."""
    embeddings = []
    for chunk in chunks:
        embedding = get_embedding(chunk)
        embeddings.append(embedding)
    return np.array(embeddings)

def extract_information_with_openai(text_chunk, prompt):
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt + text_chunk}
        ],
        max_tokens=500  # Adjusting max tokens to handle the output size
    )
    return response['choices'][0]['message']['content'].strip()

def generate_summary_and_extract_info(chunks):
    summary_prompt = "I am a contractor and want to know if I can bid for this project or not. So please generate a 500-word technical summary of the project from the following text. The summary should include the name of the project, location of the project site, and important dates (start, end, other key dates) of the project. Make sure to give information in concise points."
    summary = ""
    for chunk in chunks:
        if not summary:
            summary = extract_information_with_openai(chunk, summary_prompt)
    return summary

def save_chat_to_pdf(chat_history):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []

    for entry in chat_history:
        paragraphs = entry.split('\n')
        for para in paragraphs:
            story.append(Paragraph(para, styles['Normal']))
            story.append(Spacer(1, 0.2 * inch))

    doc.build(story)
    buffer.seek(0)
    return buffer

# Streamlit App
def main():
    # Sidebar contents
    with st.sidebar:
        st.image("https://via.placeholder.com/150", use_column_width=True)
        st.title('💬 Tender App')
        st.markdown('''
        ## About
        This app is an LLM-powered chatbot built for analyzing tenders and helping companies make decisions faster and better.

        ### Key Features:
        1. **Company-specific solution**: Provides a similarity score to prioritize tenders, helping organizations stay ahead of competitors by analyzing and making decisions faster.
        2. **Offline Suite for Enterprises**: Ensures complete secrecy and control within the company as no information is collected or saved.

        ### Beta Version:
        The Beta version contains 2 functions: Summary and ChatBOT.
        - **Summary**: Provides a quick summary of the document and work to be done.
        - **ChatBOT**: Ask questions about the document and get answers from within the document.

        **Note**: AI can make mistakes. Please verify important information. This is a demo shared for research purposes.
        ''')
        st.write('Made with ❤️ by Gak and Ashar')

    st.markdown(
        """
        <style>
        .main {
            background-color: #f0f2f6;
        }
        .stButton>button {
            color: white;
            background-color: #007BFF;
            border-radius: 5px;
            border: none;
            padding: 10px 20px;
            font-size: 16px;
        }
        .stButton>button:hover {
            background-color: #0056b3;
        }
        .header-title {
            font-size: 2em;
            font-weight: bold;
            color: #007BFF;
        }
        .summary-section {
            background-color: white;
            padding: 20px;
            border-radius: 5px;
            box-shadow: 0px 4px 6px rgba(0,0,0,0.1);
        }
        </style>
        """, unsafe_allow_html=True
    )

    st.markdown('<div class="header-title">Summarize and Chat with your PDF Tender Document 💬</div>', unsafe_allow_html=True)

    # Upload a PDF file
    pdf = st.file_uploader("Upload your Tender here", type='pdf')

    if pdf is not None:
        with st.spinner('Extracting text from PDF...'):
            pdf_path = pdf.name
            with open(pdf_path, "wb") as f:
                f.write(pdf.getbuffer())

            text = extract_text_from_pdf(pdf_path)
            chunks = split_text(text)

        with st.spinner('Generating summary...'):
            if "summary" not in st.session_state:
                st.session_state.summary = generate_summary_and_extract_info(chunks)
        
        st.markdown('<div class="summary-section">', unsafe_allow_html=True)
        st.markdown("### Summary of the Project:")
        st.write(st.session_state.summary)
        st.markdown('</div>', unsafe_allow_html=True)

        if st.button("Regenerate Summary"):
            with st.spinner('Regenerating summary...'):
                st.session_state.summary = generate_summary_and_extract_info(chunks)
                st.write(st.session_state.summary)

        if st.button("Save Summary as PDF"):
            chat_history = [st.session_state.summary]
            pdf_buffer = save_chat_to_pdf(chat_history)
            st.download_button("Download Summary as PDF", data=pdf_buffer, file_name="summary.pdf", mime="application/pdf")

if __name__ == '__main__':
    main()

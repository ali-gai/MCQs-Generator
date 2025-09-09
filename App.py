import streamlit as st
import PyPDF2
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from io import BytesIO
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
import os

#Load API key
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY")

#Streamlit UI config
st.set_page_config(page_title="MCQs Generator", layout="wide")
st.title("MCQs Generator from PDF")
st.markdown("Upload a PDF, choose number of MCQs, and download the output!")

#Upload PDF
uploaded_file = st.file_uploader("📤 Upload your PDF file", type=["pdf"])
num_mcqs = st.slider("🔢 Number of MCQs", min_value=1, max_value=20, value=5)

#LLM setup
llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="gemma2-9b-it"
)

#Prompt Template
prompt_template = ChatPromptTemplate.from_template("""
You are an expert teacher. Based on the following content, generate {num_mcqs} multiple-choice questions (MCQs).
Each question must have:
- A clear question
- 4 options (A to D)
- One correct answer marked clearly
- Format the output like:
Q1: ...
A. ...
B. ...
C. ...
D. ...
Correct Answer: B

Here is the content:
-------------------
{input_text}
-------------------
""")

#MCQ Generation
def generate_mcqs(text: str, num_mcqs: int):
    prompt = prompt_template.format_messages(input_text=text, num_mcqs=num_mcqs)
    response = llm.invoke(prompt)
    return response.content

#PDF Export
def create_pdf(mcq_text):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer)
    styles = getSampleStyleSheet()
    story = []

    for line in mcq_text.strip().split("\n"):
        story.append(Paragraph(line, styles["Normal"]))
        story.append(Spacer(1, 6))

    doc.build(story)
    buffer.seek(0)
    return buffer

#TXT Export
def create_txt(mcq_text):
    return BytesIO(mcq_text.encode("utf-8"))

#Main Logic
if uploaded_file is not None:
    pdf_reader = PyPDF2.PdfReader(uploaded_file)
    extracted_text = ""

    for page in pdf_reader.pages:
        extracted_text += page.extract_text() or ""

    text_length = len(extracted_text.strip())

    if text_length < 200:
        st.error("❗ The uploaded PDF is too short to generate meaningful MCQs. Try a longer document.")
    elif text_length > 8000:
        st.error("⚠️ The PDF content is too long. Please upload a shorter file or split the document.")
    else:
        st.subheader("Extracted Text")
        with st.expander("🔍 View Extracted Content"):
            st.write(extracted_text)

        if st.button("Generate MCQs"):
            with st.spinner("Thinking really hard..."):
                try:
                    mcqs = generate_mcqs(extracted_text, num_mcqs)
                    st.success("✅ MCQs generated successfully!")
                    st.subheader("📋 Generated MCQs")
                    st.text(mcqs)

                    # Download PDF
                    pdf_file = create_pdf(mcqs)
                    st.download_button(
                        label="📥 Download as PDF",
                        data=pdf_file,
                        file_name="generated_mcqs.pdf",
                        mime="application/pdf"
                    )

                    # Download TXT
                    txt_file = create_txt(mcqs)
                    st.download_button(
                        label="📄 Download as TXT",
                        data=txt_file,
                        file_name="generated_mcqs.txt",
                        mime="text/plain"
                    )

                except Exception as e:
                    st.error(f"❌ An error occurred while generating MCQs: {str(e)}")
else:
    st.info("⬆️ Upload a PDF file to get started.")

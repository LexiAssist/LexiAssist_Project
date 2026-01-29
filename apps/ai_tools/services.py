import os
from django.conf import settings
import PyPDF2
import json

# --- NO GOOGLE IMPORT HERE ---

def get_ai_response(text, image=None):
    """
    Multimodal AI assistant using Gemini Pro Vision logic.
    """
    # --- LAZY LOAD: Only import when needed ---
    import google.generativeai as genai
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

    # Initialize the model
    model = genai.GenerativeModel(
        model_name="gemini-2.5-flash",
        system_instruction=(
            "You are LexiAssist AI, an assistive learning companion for students with dyslexia. "
            "Use short sentences. Prefer bullet points over paragraphs. Avoid complex academic language. "
            "Explain one idea at a time and use an encouraging tone."
        )
    )

    # Prepare the content list
    content = []
    if text:
        content.append(text)

    if image:
        content.append({
            "mime_type": image.content_type,
            "data": image.read()
        })

    try:
        response = model.generate_content(content)
        return response.text
    except Exception as e:
        return f"AI Error: {str(e)}"


def extract_text_from_pdf(pdf_file):
    """Helper to pull text from a PDF file"""
    try:
        reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text[:10000]
    except Exception as e:
        return ""

def extract_text_from_docx(docx_file):
    """Extract text from a Word document (.docx)"""
    try:
        from docx import Document
        doc = Document(docx_file)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text
    except Exception as e:
        return ""


def generate_flashcards(topic, text_content, num_cards=5):
    """Asks Gemini to create flashcards in JSON format."""
    # --- LAZY LOAD ---
    import google.generativeai as genai
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

    prompt = f"""
    You are a teacher. Create {num_cards} flashcards about "{topic}".
    Base them on this content:
    {text_content[:5000]}

     STRICT RESPONSE FORMAT:
    You must return ONLY a raw JSON array. Do not use Markdown blocks.
    Example:
    [
        {{"question": "What is X?", "answer": "X is Y"}},
        {{"question": "Who did Z?", "answer": "Person A"}}
    ]
    """

    model = genai.GenerativeModel("gemini-2.5-flash")

    try:
        response = model.generate_content(prompt)
        clean_text = response.text.replace("```json", "").replace("```", "").strip()
        return json.loads(clean_text)
    except Exception as e:
        print(f"Error: {e}")
        return []


def generate_quiz_data(topic, text_content, num_questions=5):
    """Asks Gemini to create a quiz in strict JSON format."""
    # --- LAZY LOAD ---
    import google.generativeai as genai
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

    prompt = f"""
    Create a {num_questions}-question multiple choice quiz about "{topic}".
    Base the questions on this content:
    {text_content[:8000]}

    STRICT JSON FORMAT REQUIRED:
    Return ONLY a raw JSON array. Do not use Markdown blocks.
    Structure:
    [
        {{
            "question": "Question text here?",
            "options": ["Option A", "Option B", "Option C", "Option D"],
            "correct_index": 0
        }}
    ]
    """

    model = genai.GenerativeModel("gemini-2.5-flash")

    try:
        response = model.generate_content(prompt)
        clean_text = response.text.replace("```json", "").replace("```", "").strip()
        return json.loads(clean_text)
    except Exception as e:
        print(f"Quiz Gen Error: {e}")
        return []

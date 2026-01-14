import os
import google.generativeai as genai
from django.conf import settings
import PyPDF2
import json


# Configure Gemini with your key
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

def get_ai_response(text, image=None):
    """
    Multimodal AI assistant using Gemini Pro Vision logic.
    """
    # Initialize the model with your specific accessibility instructions
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
        # Pass the image directly to Gemini
        content.append({
            "mime_type": image.content_type,
            "data": image.read()
        })

    try:
        # Generate the response
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
        return text[:10000] # Limit to 10k chars to save quota
    except Exception as e:
        return ""

def generate_flashcards(topic, text_content, num_cards=5):
    """
    Asks Gemini to create flashcards in JSON format.
    """
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

    model = genai.GenerativeModel("gemini-1.5-flash")
    try:
        response = model.generate_content(prompt)
        # Clean up if Gemini adds markdown ```json ... ```
        clean_text = response.text.replace("```json", "").replace("```", "").strip()
        return json.loads(clean_text)
    except Exception as e:
        print(f"Error: {e}")
        return []

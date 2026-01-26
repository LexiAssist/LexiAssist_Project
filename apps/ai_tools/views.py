import logging
import os
from django.shortcuts import render, get_object_or_404
from django.http import JsonResponse
from django.contrib.auth.decorators import login_required
from django.conf import settings
from apps.classroom.models import UserActivity

# Consolidated imports from services
from .models import ChatSession, ChatMessage
from .services import (
    get_ai_response,
    generate_flashcards,
    generate_quiz_data,
    extract_text_from_pdf
)

from django.views.decorators.csrf import csrf_exempt
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile

# --- AI ENGINE IMPORTS (For RAG) ---
from ai_engine.doc_loader import load_and_split_pdf
from ai_engine.embeddings import create_vector_store
from ai_engine.rag import workflow
import ai_engine.retriever as retriever_module

logger = logging.getLogger(__name__)

# ============================
# 1. CHAT ASSISTANT
# ============================
@login_required
def chat_assistant_view(request, session_id=None):
    # 1. Get all previous chats for the sidebar
    my_sessions = ChatSession.objects.filter(user=request.user).order_by('-created_at')

    # 2. If a specific session is requested (clicking history), load its messages
    current_session = None
    existing_messages = []

    if session_id:
        current_session = get_object_or_404(ChatSession, id=session_id, user=request.user)
        existing_messages = current_session.messages.all().order_by('created_at')

    # 3. Handle the AJAX Chat (When you click Send)
    if request.method == "POST":
        user_query = request.POST.get('user_query', '').strip()
        uploaded_image = request.FILES.get('image')
        active_session_id = request.POST.get('session_id')

        if not user_query and not uploaded_image:
            return JsonResponse({'status': 'error', 'reply': 'Empty message.'})

        try:
            # A. Get or Create the Session
            if active_session_id:
                session = ChatSession.objects.get(id=active_session_id, user=request.user)
            else:
                title_text = user_query[:30] + "..." if user_query else "Image Analysis"
                session = ChatSession.objects.create(user=request.user, title=title_text)

            # B. Save USER Message
            ChatMessage.objects.create(session=session, is_user=True, text=user_query)

            # C. Get AI Response
            ai_reply = get_ai_response(text=user_query, image=uploaded_image)

            # D. Save AI Message
            ChatMessage.objects.create(session=session, is_user=False, text=ai_reply)

            return JsonResponse({
                'status': 'success',
                'reply': ai_reply,
                'session_id': session.id,
                'session_title': session.title
            })

        except Exception as e:
            logger.error(f"Error: {e}")
            return JsonResponse({'status': 'error', 'reply': "An error occurred."})

    context = {
        'chat_sessions': my_sessions,
        'active_session': current_session,
        'existing_messages': existing_messages
    }
    return render(request, 'chat_assistant.html', context)


# ============================
# 2. FLASHCARD GENERATOR
# ============================
def flashcard_generator_view(request):
    if request.method == "POST":
        topic = request.POST.get('topic')
        num_cards = request.POST.get('num_cards', 5)
        upload_file = request.FILES.get('file')

        # 1. Get Text from File or Paste
        content_text = ""
        if upload_file:
            if upload_file.name.endswith('.pdf'):
                content_text = extract_text_from_pdf(upload_file)
            else:
                content_text = "Unsupported file type. Please upload PDF."

        # 2. Call Gemini
        cards_data = generate_flashcards(topic, content_text, num_cards)

        return JsonResponse({'status': 'success', 'cards': cards_data})

    return render(request, 'flashcards.html')


# ============================
# 3. QUIZ GENERATOR
# ============================
def quizzes_view(request):
    if request.method == "POST":
        topic = request.POST.get('topic')
        num_questions = 5  # Default to 5

        # 1. Find the user's most recent uploaded PDF/Doc
        last_activity = UserActivity.objects.filter(
            user=request.user
        ).exclude(file_name='').order_by('-timestamp').first()

        context_text = ""
        if last_activity:
            file_path = os.path.join(settings.MEDIA_ROOT, last_activity.file_name)
            # Check if file actually exists on disk
            if os.path.exists(file_path):
                if file_path.endswith('.pdf'):
                    with open(file_path, 'rb') as f:
                        context_text = extract_text_from_pdf(f)
                # You can add elif for .docx here later

        # If no file found, use the topic as context
        if not context_text:
            context_text = f"General knowledge about {topic}"

        # 2. Call AI
        quiz_data = generate_quiz_data(topic, context_text, num_questions)

        return JsonResponse({'status': 'success', 'quiz': quiz_data})

    return render(request, 'quizzes.html')


# ============================
# 4. RAG / STUDY BUDDY API
# ============================

@csrf_exempt
def upload_note_api(request):
    """
    API View to handle PDF uploads and process them for the AI.
    """
    if request.method == 'POST' and request.FILES.get('document'):
        uploaded_file = request.FILES['document']

        # 1. Save the file temporarily
        file_path = default_storage.save(f"temp_notes/{uploaded_file.name}", ContentFile(uploaded_file.read()))
        full_path = os.path.join(default_storage.location, file_path)

        try:
            # 2. Load and Split the PDF
            print(f"\n{'='*60}")
            print(f"📄 UPLOAD NOTE API - Processing file: {full_path}")
            print(f"{'='*60}\n")

            documents = load_and_split_pdf(full_path)
            print(f"✅ PDF loaded and split into {len(documents)} chunks")

            # 3. Create Vector Store (The Brain)
            print("🧠 Creating embeddings... (this might take a moment)")
            vector_store = create_vector_store(documents)
            print("✅ Vector store created and saved to disk")

            # 4. Update the global variable in retriever.py so the AI uses THIS document
            retriever_module.active_vector_store = vector_store
            print("✅ Active vector store updated\n")

            return JsonResponse({
                "status": "success",
                "message": "Note processed successfully! You can now chat."
            })

        except Exception as e:
            logger.error(f"Error processing RAG file: {e}")
            import traceback
            traceback.print_exc()
            return JsonResponse({"status": "error", "message": str(e)}, status=500)

    return JsonResponse({"status": "error", "message": "No file uploaded"}, status=400)


@csrf_exempt
def chat_with_ai_api(request):
    """
    API View to handle RAG chat messages (Specific to the uploaded document).
    """
    if request.method == 'POST':
        user_message = request.POST.get('message')

        if not user_message:
            return JsonResponse({"error": "Message is required"}, status=400)

        print(f"\n{'='*60}")
        print(f"💬 CHAT WITH AI API")
        print(f"📩 User Message: {user_message}")
        print(f"{'='*60}\n")

        # Check if FAISS index exists (use same path as embeddings.py)
        faiss_path = os.path.join(settings.BASE_DIR, "faiss_index")
        print(f"📁 Checking for FAISS index at: {faiss_path}")
        print(f"📁 Index exists: {os.path.exists(faiss_path)}")

        if not os.path.exists(faiss_path):
            print("❌ No FAISS index found")
            return JsonResponse({
                "answer": "No document uploaded yet. Please upload a PDF file first."
            })

        try:
            # Import the retrieve_data tool directly
            from ai_engine.retriever import retrieve_data

            print("🔄 Calling retrieve_data directly...")

            # Call the retriever directly - it returns a string
            ai_answer = retrieve_data.invoke({"query": user_message})

            print(f"\n✅ Got answer from retrieve_data")
            print(f"📝 Answer preview: {ai_answer[:200]}...\n")

            return JsonResponse({"answer": ai_answer})

        except Exception as e:
            logger.error(f"RAG Chat Error: {e}")
            import traceback
            traceback.print_exc()

            return JsonResponse({
                "answer": f"I'm having trouble reading that document. Error: {str(e)}"
            })

    return JsonResponse({"error": "Invalid request"}, status=400)

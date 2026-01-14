import logging
from django.shortcuts import render, get_object_or_404
from django.http import JsonResponse
from django.contrib.auth.decorators import login_required
from .services import get_ai_response
from .models import ChatSession, ChatMessage
from .services import generate_flashcards, extract_text_from_pdf

logger = logging.getLogger(__name__)

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
        active_session_id = request.POST.get('session_id') # Hidden input we will add

        if not user_query and not uploaded_image:
            return JsonResponse({'status': 'error', 'reply': 'Empty message.'})

        try:
            # A. Get or Create the Session
            if active_session_id:
                session = ChatSession.objects.get(id=active_session_id, user=request.user)
            else:
                # Create a title from the first few words
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

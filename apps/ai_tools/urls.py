from django.urls import path
from . import views

urlpatterns = [
    # Chat Assistant
    path('chat/', views.chat_assistant_view, name='chat_assistant'),
    path('chat/<int:session_id>/', views.chat_assistant_view, name='chat_history'),

    # Flashcards
    path('flashcards/', views.flashcard_generator_view, name='flashcards'),

    # RAG/Study Buddy
    path('api/upload-note/', views.upload_note_api, name='upload_note_api'),
    path('api/chat/', views.chat_with_ai_api, name='chat_with_ai_api'),

    # TEXT TO SPEECH
    path('text-to-speech/', views.tts_dashboard_view, name='text2speech'),  # Upload page
    path('api/tts/', views.text_to_speech, name='text_to_speech'),  # API for MP3 generation

    # READING ASSISTANT (NEW)
    path('reading-assistant/', views.reading_assistant_view, name='reading_assistant'),
]

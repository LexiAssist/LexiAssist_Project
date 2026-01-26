from django.urls import path
from . import views

urlpatterns = [
    # This makes the chat accessible at /ai/chat/
    path('chat/', views.chat_assistant_view, name='chat_assistant'),
    path('chat/<int:session_id>/', views.chat_assistant_view, name='chat_history'),
    path('flashcards/', views.flashcard_generator_view, name='flashcards'),

    
    path('api/upload-note/', views.upload_note_api, name='upload_note_api'),
    path('api/chat/', views.chat_with_ai_api, name='chat_with_ai_api'),
]

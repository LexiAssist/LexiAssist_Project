from django.urls import path
from . import views

urlpatterns = [
    # This makes the chat accessible at /ai/chat/
    path('chat/', views.chat_assistant_view, name='chat_assistant'),
    path('chat/<int:session_id>/', views.chat_assistant_view, name='chat_history'),
    path('flashcards/', views.flashcard_generator_view, name='flashcards'),
]

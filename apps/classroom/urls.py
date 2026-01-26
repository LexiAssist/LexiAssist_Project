from django.urls import path
from .views import (
    dashboard_view,
    text2speech_view,
    # reading_assistant_view,  <-- Remove this one, it's redundant
    writing_assistant_view,
    landing_view,
    clean_text_api,
    reading_assistant_upload,
    reading_reader,
    tts_reader,
    delete_activity,

)
from apps.ai_tools import views as ai_views

# apps/classroom/urls.py
# apps/classroom/urls.py

urlpatterns = [
    path('dashboard/', dashboard_view, name='dashboard'),
    path('text-to-speech/', text2speech_view, name='text2speech'),
    path('writing-assistant/', writing_assistant_view, name='writing_assistant'),
    path('', landing_view, name='landing'),
    path('api/clean-text/', clean_text_api, name='clean_text_api'),


    # FIXED: Changed 'reading-assistant' to 'reading_assistant'
    path('reading-assistant/', reading_assistant_upload, name='reading_assistant'),

    path('reading-reader/', reading_reader, name='reading_reader'),

    path('text-to-speech/reader/',tts_reader, name='tts_reader'),
    path('activity/delete/<int:activity_id>/',delete_activity, name='delete_activity'),


    path('chat/', ai_views.chat_assistant_view, name='chat_assistant'),

    # This specifically fixes the missing reference for flashcards too
    path('flashcards/', ai_views.flashcard_generator_view, name='flashcards'),

    path('quizzes/', ai_views.quizzes_view, name='quizzes'),
]

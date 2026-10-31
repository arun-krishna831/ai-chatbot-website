import os
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq

from django.conf import settings
from django.shortcuts import render, redirect
from django.contrib import messages as django_messages
from django.core.files.storage import FileSystemStorage

from .rag import store_pdf_chunks, generate_rag_answer


BASE_DIR = Path(__file__).resolve().parent.parent

load_dotenv(BASE_DIR / ".env")

client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def home(request):

    if "messages" not in request.session:
        request.session["messages"] = [
            {
                "role": "system",
                "content": "You are a helpful assistant."
            }
        ]

    if request.method == "POST":

        # New Chat button
        if request.POST.get("action") == "new_chat":

            request.session["messages"] = [
                {
                    "role": "system",
                    "content": "You are a helpful assistant."
                }
            ]

            request.session.modified = True

            return redirect("home")

        # Normal chat message
        user_message = request.POST.get("message", "").strip()

        if user_message:

            chat_history = request.session["messages"]

            chat_history.append({
                "role": "user",
                "content": user_message
            })

            try:

                # Get answer using RAG
                ai_response = generate_rag_answer(
                    user_message,
                    client
                )

                chat_history.append({
                    "role": "assistant",
                    "content": ai_response
                })

            except Exception as e:

                chat_history.pop()

                django_messages.error(
                    request,
                    f"Something went wrong: {str(e)}"
                )

            request.session["messages"] = chat_history
            request.session.modified = True

            return redirect("home")

    chat_history = request.session.get("messages", [])

    visible_messages = [
        item for item in chat_history
        if item["role"] in ("user", "assistant")
    ]

    return render(
        request,
        "chatbot/index.html",
        {
            "chat_messages": visible_messages,
            "notifications": django_messages.get_messages(request)
        }
    )


def upload_pdf(request):

    if request.method != "POST":
        return redirect("home")

    uploaded_file = request.FILES.get("pdf_file")

    if not uploaded_file:
        django_messages.error(request, "Please select a PDF file.")
        return redirect("home")

    if not uploaded_file.name.lower().endswith(".pdf"):
        django_messages.error(request, "Only PDF files are allowed.")
        return redirect("home")

    if uploaded_file.content_type != "application/pdf":
        django_messages.error(request, "Invalid PDF file.")
        return redirect("home")

    try:

        fs = FileSystemStorage(
            location=settings.MEDIA_ROOT
        )

        filename = fs.save(
            uploaded_file.name,
            uploaded_file
        )

        pdf_path = Path(settings.MEDIA_ROOT) / filename

        total_chunks = store_pdf_chunks(pdf_path)

        if total_chunks > 0:

            django_messages.success(
                request,
                f"PDF uploaded successfully! {total_chunks} chunks stored."
            )

        else:

            django_messages.error(
                request,
                "No readable text was found in the PDF."
            )

    except Exception as e:

        django_messages.error(
            request,
            f"PDF processing failed: {str(e)}"
        )

    return redirect("home")
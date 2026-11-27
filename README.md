# SmartGuru – Curriculum Chatbot

AI study assistant for **SmartGuru**, a learning app for school students (published on the Play Store and App Store).
Students ask questions about their class syllabus and get answers that stay within their standard and subject.

## What it does

- **Curriculum-aware answers:** the app sends the syllabus (standard → subject → chapters), and the bot uses it to keep answers relevant to the student's class.
- **Conversation memory:** remembers the last messages in each session, so follow-up questions work naturally.
- **Student context:** stores the student's details per session (for example their standard) to personalise replies.

## Tech stack

Python · Flask · OpenAI API (GPT-4) · Gunicorn

## Endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| POST | `/api/curriculum/set` | Load the syllabus data |
| POST | `/api/chat` | Ask a question |
| POST | `/api/chat/user` | Save the student's details for the session |
| GET | `/api/chat/history` | Get the conversation history |
| POST | `/api/chat/clear` | Clear the conversation |

A sample syllabus file is in `chabot.json`.

## Run locally

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then add your OpenAI key
python app.py          # runs on port 5000
```

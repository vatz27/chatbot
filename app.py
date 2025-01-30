from flask import Flask, request, jsonify
from flask_cors import CORS
from typing import List, Dict, Tuple
from pydantic import BaseModel, Field
from datetime import datetime
import uuid
import os
import json
from io import BytesIO
import re
from openai import OpenAI
from dotenv import load_dotenv
from langchain.docstore.document import Document
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from pypdf import PdfReader

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Load environment variables
load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')
client = OpenAI(api_key=api_key)

# Global variables
vectordb = None
chat_histories = {}

class ChatMessage(BaseModel):
    text: str
    is_user: bool
    message_type: str = "text"
    file_path: str = None
    curriculum_data: dict = None
    timestamp: datetime = Field(default_factory=datetime.now)

    def dict(self):
        return {
            "text": self.text,
            "is_user": self.is_user,
            "message_type": self.message_type,
            "file_path": self.file_path,
            "curriculum_data": self.curriculum_data,
            "timestamp": self.timestamp.isoformat()
        }

class ChatHistory(BaseModel):
    id: str
    messages: List[ChatMessage]
    timestamp: datetime = Field(default_factory=datetime.now)

def parse_pdf(file: BytesIO, filename: str) -> Tuple[List[str], str]:
    pdf = PdfReader(file)
    output = []
    for page in pdf.pages:
        text = page.extract_text()
        text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)
        text = re.sub(r"(?<!\n\s)\n(?!\s\n)", " ", text.strip())
        text = re.sub(r"\n\s*\n", "\n\n", text)
        output.append(text)
    return output, filename

def text_to_docs(text: List[str], filename: str) -> List[Document]:
    if isinstance(text, str):
        text = [text]
    
    page_docs = [Document(page_content=page) for page in text]
    for i, doc in enumerate(page_docs):
        doc.metadata["page"] = i + 1

    doc_chunks = []
    for doc in page_docs:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=4000,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
            chunk_overlap=0,
        )
        chunks = text_splitter.split_text(doc.page_content)
        for i, chunk in enumerate(chunks):
            doc = Document(
                page_content=chunk,
                metadata={
                    "page": doc.metadata["page"],
                    "chunk": i,
                    "source": f"{doc.metadata['page']}-{i}",
                    "filename": filename
                }
            )
            doc_chunks.append(doc)
    return doc_chunks

def docs_to_index(docs, openai_api_key):
    return FAISS.from_documents(docs, OpenAIEmbeddings(openai_api_key=openai_api_key))

@app.route('/api/chat/new-chat', methods=['POST'])
def new_chat():
    chat_id = str(uuid.uuid4())
    chat_histories[chat_id] = ChatHistory(
        id=chat_id,
        messages=[
            ChatMessage(
                text="Hello! I'm your AI Tutor. How can I help you today?",
                is_user=False
            )
        ]
    )
    return jsonify({"chat_id": chat_id})

@app.route('/api/chat/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data received"}), 400

        message = data.get('message')
        chat_id = data.get('chat_id')

        if not chat_id or not message:
            return jsonify({"error": "Chat ID and message are required"}), 400

        if chat_id not in chat_histories:
            chat_histories[chat_id] = ChatHistory(id=chat_id, messages=[])

        # Add user message to history
        chat_histories[chat_id].messages.append(
            ChatMessage(text=message, is_user=True)
        )

        # Use OpenAI for all queries
        messages = [{"role": "system", "content": "You are a helpful AI tutor. When responding to questions about class subjects and chapters, be clear and informative. For curriculum-related questions, provide structured information about subjects and chapters."}]
        
        for msg in chat_histories[chat_id].messages:
            role = "user" if msg.is_user else "assistant"
            messages.append({"role": role, "content": msg.text})

        completion = client.chat.completions.create(
            model="gpt-4",
            messages=messages
        )
        
        response = {
            "text": completion.choices[0].message.content,
            "message_type": "text",
        }

        # Add AI response to history
        chat_histories[chat_id].messages.append(
            ChatMessage(
                text=response["text"],
                is_user=False,
                message_type=response["message_type"]
            )
        )

        return jsonify({
            "response": response["text"],
            "message_type": response["message_type"],
            "chat_id": chat_id
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/chat/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    try:
        upload_dir = os.path.join(app.root_path, 'uploads')
        os.makedirs(upload_dir, exist_ok=True)

        filename = str(uuid.uuid4()) + '_' + file.filename
        file_path = os.path.join(upload_dir, filename)
        file.save(file_path)

        if file.filename.endswith('.pdf'):
            file_content = file.read()
            file.seek(0)
            parsed_text, _ = parse_pdf(BytesIO(file_content), file.filename)
            docs = text_to_docs(parsed_text, file.filename)
            global vectordb
            vectordb = docs_to_index(docs, api_key)

        return jsonify({
            "message": "File uploaded successfully",
            "file_path": file_path
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/chat/history', methods=['GET'])
def get_chat_history():
    chat_id = request.args.get('chat_id')
    if not chat_id:
        return jsonify({"error": "Chat ID is required"}), 400

    if chat_id not in chat_histories:
        return jsonify({"error": "Chat history not found"}), 404

    history = chat_histories[chat_id]
    return jsonify({
        "id": history.id,
        "messages": [msg.dict() for msg in history.messages],
        "timestamp": history.timestamp.isoformat()
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

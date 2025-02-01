from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import os
import openai
from dotenv import load_dotenv
import logging
from datetime import datetime
from typing import Dict, List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')

class ConversationMemory:
    def __init__(self):
        self.conversations: Dict[str, List[Dict]] = {}
        self.user_data = {}
    
    def add_message(self, session_id: str, role: str, content: str):
        if session_id not in self.conversations:
            self.conversations[session_id] = []
        message = {
            'role': role,
            'content': content,
            'timestamp': datetime.now().isoformat()
        }
        self.conversations[session_id].append(message)
    
    def add_user_data(self, session_id: str, user_data: dict):
        self.user_data[session_id] = user_data

    def get_user_data(self, session_id: str) -> dict:
        return self.user_data.get(session_id, {})
        
    def get_conversation_history(self, session_id: str, max_messages: int = 10) -> List[Dict]:
        if session_id not in self.conversations:
            return []
        return self.conversations[session_id][-max_messages:]
    
    def clear_conversation(self, session_id: str):
        if session_id in self.conversations:
            self.conversations[session_id] = []

memory = ConversationMemory()

def load_curriculum_data():
    """Load curriculum data from JSON file"""
    json_path = r"D:\StudioProjects\school\assets\data\chatbot.json"
    
    try:
        if os.path.exists(json_path):
            with open(json_path, 'r', encoding='utf-8') as f:
                curriculum_data = json.load(f)
                logger.info("Successfully loaded curriculum data")
                return curriculum_data
        else:
            logger.error(f"Curriculum file not found at: {json_path}")
            return {}
    except Exception as e:
        logger.error(f"Error loading curriculum data: {str(e)}")
        return {}

def is_standards_query(query: str) -> bool:
    query_lower = query.lower()
    keywords = [
        'what standards', 'which standards', 'available standards',
        'what classes', 'which classes', 'available classes', 
        'list standards', 'show standards', 'tell me standards',
        'what standard', 'which standard', 'tell me the standards'
    ]
    return any(keyword in query_lower for keyword in keywords)

def get_standards_response() -> dict:
    standards = sorted(curriculum_data.keys())
    if standards:
        response = "Here are the available standards:\n\n"
        for std in standards:
            response += f"• Standard {std}\n"
        return {
            "response": response,
            "type": "text"
        }
    return {
        "response": "Sorry, no curriculum data is currently available.",
        "type": "text"
    }

def extract_query_info(query: str):
    query_lower = query.lower()
    standard = None
    
    for std in curriculum_data.keys():
        if std.lower() in query_lower:
            standard = std
            break

    subjects = set()
    for std_data in curriculum_data.values():
        subjects.update(std_data.keys())
    
    subject = None
    for sub in subjects:
        if sub.lower() in query_lower:
            subject = sub
            break

    return {
        'standard': standard,
        'subject': subject
    }

def is_curriculum_query(query: str) -> bool:
    query_lower = query.lower()
    
    curriculum_keywords = [
        'what chapters', 'list chapters', 'show chapters',
        'which chapters', 'chapters in', 'tell me chapters',
        'syllabus', 'curriculum', 'what are the chapters',
        'tell me the chapters'
    ]
    
    has_standard_mention = any(std.lower() in query_lower for std in curriculum_data.keys())
    subjects = set()
    for std_data in curriculum_data.values():
        subjects.update(std_data.keys())
    has_subject_mention = any(subject.lower() in query_lower for subject in subjects)
    
    has_curriculum_keyword = any(keyword in query_lower for keyword in curriculum_keywords)
    
    return has_curriculum_keyword or (has_standard_mention and has_subject_mention)

def get_chapters_response(standard: str, subject: str) -> str:
    try:
        if standard in curriculum_data and subject in curriculum_data[standard]:
            chapters = curriculum_data[standard][subject]
            chapter_list = '\n'.join([f"• {chapter}" for chapter in chapters['chapters']])
            return f"Here are the chapters for {subject} in Standard {standard}:\n\n{chapter_list}"
        else:
            return f"Sorry, I couldn't find chapter information for {subject} in Standard {standard}."
    except Exception as e:
        logger.error(f"Error getting chapters: {str(e)}")
        return "Sorry, I encountered an error while fetching the chapter information."

def handle_curriculum_query(query: str, session_id: str) -> dict:
    try:
        if is_standards_query(query):
            response = get_standards_response()
            memory.add_message(session_id, "assistant", response["response"])
            return response
        
        query_info = extract_query_info(query)
        standard = query_info['standard']
        subject = query_info['subject']

        if not standard:
            user_data = memory.get_user_data(session_id)
            standard = user_data.get('standard', '')
            
            if standard:
                if '11th' in standard.lower() or '12th' in standard.lower():
                    # No stream division for 11th and 12th standards
                    pass

        if not standard:
            return {
                "response": "Please specify which class you're asking about.",
                "type": "text"
            }

        if subject:
            response = {
                "response": get_chapters_response(standard, subject),
                "type": "text"
            }
        elif standard in curriculum_data:
            subjects = list(curriculum_data[standard].keys())
            response = {
                "response": f"Available subjects for Standard {standard}: {', '.join(subjects)}",
                "type": "text"
            }
        else:
            response = {
                "response": f"No curriculum data found for Standard {standard}.",
                "type": "text"
            }
            
        memory.add_message(session_id, "assistant", response["response"])
        return response

    except Exception as e:
        logger.error(f"Error: {str(e)}")
        error_response = {
            "response": "Error processing request. Please try again.",
            "type": "text"
        }
        memory.add_message(session_id, "assistant", error_response["response"])
        return error_response

def handle_educational_query(query: str, session_id: str) -> dict:
    try:
        conversation_history = memory.get_conversation_history(session_id)
        user_data = memory.get_user_data(session_id)
        
        system_message = """You are a helpful AI tutor for school students. Use this student information:
        Name: {name}
        Standard: {standard} 
        Stream: {stream}
        
        Focus on:
        - Providing clear, accurate explanations
        - Using age-appropriate language
        - Breaking down complex topics
        - Explaining step-by-step solutions
        - Providing relevant examples
        - Maintaining context from previous messages"""

        messages = [
            {
                "role": "system",
                "content": system_message.format(
                    name=user_data.get('name', 'Student'),
                    standard=user_data.get('standard', 'Unknown'),
                    stream=user_data.get('stream', '')
                )
            }
        ]
        
        for msg in conversation_history:
            messages.append({
                "role": msg["role"],
                "content": msg["content"]
            })
        
        messages.append({
            "role": "user",
            "content": query
        })
        
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=messages
        )
        
        assistant_response = response.choices[0].message['content']
        memory.add_message(session_id, "assistant", assistant_response)
        
        return {
            "response": assistant_response,
            "type": "text"
        }
    except Exception as e:
        logger.error(f"Error in OpenAI response: {str(e)}")
        error_response = "I encountered an error while processing your question. Please try again."
        memory.add_message(session_id, "assistant", error_response)
        return {
            "response": error_response,
            "type": "text"
        }

@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        if not data or 'message' not in data:
            return jsonify({"error": "No message provided"}), 400
        
        user_message = data['message']
        session_id = data.get('session_id', 'default_session')
        
        logger.info(f"Received message: {user_message} for session: {session_id}")
        
        memory.add_message(session_id, "user", user_message)
        
        if is_standards_query(user_message):
            logger.info("Processing as standards query")
            response = get_standards_response()
        elif is_curriculum_query(user_message):
            logger.info("Processing as curriculum query")
            response = handle_curriculum_query(user_message, session_id)
        else:
            logger.info("Processing as educational query")
            response = handle_educational_query(user_message, session_id)

        return jsonify(response)

    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/chat/user', methods=['POST'])
def set_user_data():
    try:
        data = request.json
        session_id = data.get('session_id', 'default_session')
        user_data = {
            'name': data.get('name'),
            'standard': data.get('standard'),
            'stream': data.get('stream')
        }
        memory.add_user_data(session_id, user_data)
        return jsonify({"message": "User data stored successfully"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/chat/history', methods=['GET'])
def get_chat_history():
    session_id = request.args.get('session_id', 'default_session')
    history = memory.get_conversation_history(session_id)
    return jsonify({"history": history})

@app.route('/api/chat/clear', methods=['POST'])
def clear_chat():
    session_id = request.json.get('session_id', 'default_session')
    memory.clear_conversation(session_id)
    return jsonify({"message": "Conversation cleared successfully"})

if __name__ == '__main__':
    logger.info("Loading curriculum data...")
    curriculum_data = load_curriculum_data()
    if not curriculum_data:
        logger.warning("No curriculum data loaded. Check if JSON file exists.")
    else:
        logger.info(f"Loaded curriculum data for standards: {list(curriculum_data.keys())}")
        
    logger.info("Starting server...")
    app.run(debug=True, host='0.0.0.0', port=5000)

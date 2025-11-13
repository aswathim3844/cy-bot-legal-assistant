from flask import Flask, render_template, request, jsonify, session, send_from_directory
import os
import uuid
import json
from werkzeug.utils import secure_filename
from bot_backend import get_bot_response, process_pdf, remove_pdf_from_memory

app = Flask(__name__)
app.secret_key = "cyberlaw_secret"

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Create upload directory if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# File to store chat sessions
CHAT_SESSIONS_FILE = 'chat_sessions.json'

def load_chat_sessions():
    """Load chat sessions from file"""
    try:
        if os.path.exists(CHAT_SESSIONS_FILE):
            with open(CHAT_SESSIONS_FILE, 'r') as f:
                return json.load(f)
    except Exception as e:
        print(f"Error loading chat sessions: {e}")
    return {}

def save_chat_sessions(sessions):
    """Save chat sessions to file"""
    try:
        with open(CHAT_SESSIONS_FILE, 'w') as f:
            json.dump(sessions, f)
        return True
    except Exception as e:
        print(f"Error saving chat sessions: {e}")
        return False

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/")
def index():
    # Initialize session ID if it doesn't exist
    if "session_id" not in session:
        session["session_id"] = str(uuid.uuid4())
    
    # Initialize history only if it doesn't exist - don't clear on refresh
    if "history" not in session:
        session["history"] = []
    
    # Load chat sessions for sidebar
    chat_sessions = load_chat_sessions()
    
    return render_template("index.html", chat_history=session["history"], chat_sessions=chat_sessions)

@app.route("/get_response", methods=["POST"])
def get_response():
    user_msg = request.json["message"]
    has_pdf = request.json.get("has_pdf", False)
    
    bot_msg = get_bot_response(user_msg, has_pdf)

    entry = {"user": user_msg, "bot": bot_msg}
    history = session.get("history", [])
    history.append(entry)
    session["history"] = history
    
    # Save the updated session to persistent storage
    if "session_id" in session:
        chat_sessions = load_chat_sessions()
        # Only save if we have at least one message
        if history:
            chat_sessions[session["session_id"]] = {
                "history": history,
                "title": history[0]["user"][:50] + "..." if history else "New Chat",
                "timestamp": request.json.get("timestamp", None)
            }
            save_chat_sessions(chat_sessions)
    
    return jsonify({"bot": bot_msg})

@app.route("/clear", methods=["POST"])
def clear_chat():
    # Get current session ID before clearing
    current_session_id = session.get("session_id")
    
    # Clear session history
    session.pop("history", None)
    
    # Remove the session from persistent storage
    if current_session_id:
        chat_sessions = load_chat_sessions()
        if current_session_id in chat_sessions:
            del chat_sessions[current_session_id]
            save_chat_sessions(chat_sessions)
    
    # Remove all uploaded PDF files
    try:
        upload_folder = app.config['UPLOAD_FOLDER']
        for filename in os.listdir(upload_folder):
            file_path = os.path.join(upload_folder, filename)
            if os.path.isfile(file_path) and filename.endswith('.pdf'):
                os.remove(file_path)
                print(f"Removed PDF: {filename}")
    except Exception as e:
        print(f"Error cleaning uploads folder: {e}")
    
    return jsonify({"status": "cleared"})

@app.route("/new_chat", methods=["POST"])
def new_chat():
    # Generate a new session ID
    old_session_id = session.get("session_id")
    session["session_id"] = str(uuid.uuid4())
    session["history"] = []
    
    # Save the old session if it exists and has content
    if old_session_id and "history" in session:
        chat_sessions = load_chat_sessions()
        old_history = session.get("history", [])
        if old_history:
            chat_sessions[old_session_id] = {
                "history": old_history,
                "title": old_history[0]["user"][:50] + "..." if old_history else "Previous Chat",
                "timestamp": request.json.get("timestamp") if request.json else None
            }
            save_chat_sessions(chat_sessions)
    
    return jsonify({"status": "new_chat_created", "session_id": session["session_id"]})

@app.route("/load_chat", methods=["POST"])
def load_chat():
    session_id = request.json.get("session_id")
    if not session_id:
        return jsonify({"success": False, "error": "No session ID provided"})
    
    chat_sessions = load_chat_sessions()
    if session_id in chat_sessions:
        session["session_id"] = session_id
        session["history"] = chat_sessions[session_id]["history"]
        return jsonify({"success": True, "history": chat_sessions[session_id]["history"]})
    
    return jsonify({"success": False, "error": "Chat session not found"})

@app.route("/delete_session", methods=["POST"])
def delete_session():
    session_id = request.json.get("session_id")
    if not session_id:
        return jsonify({"success": False, "error": "No session ID provided"})
    
    chat_sessions = load_chat_sessions()
    if session_id in chat_sessions:
        del chat_sessions[session_id]
        save_chat_sessions(chat_sessions)
        
        # If we're deleting the current session, clear it from the session too
        if session.get("session_id") == session_id:
            session.pop("history", None)
        
        return jsonify({"success": True})
    
    return jsonify({"success": False, "error": "Chat session not found"})

@app.route("/get_sessions", methods=["GET"])
def get_sessions():
    chat_sessions = load_chat_sessions()
    return jsonify({"sessions": chat_sessions})

@app.route("/get_history", methods=["GET"])
def get_history():
    return jsonify({"history": session.get("history", [])})

@app.route("/upload_pdf", methods=["POST"])
def upload_pdf():
    if 'pdf' not in request.files:
        return jsonify({"success": False, "error": "No file part"})
    
    file = request.files['pdf']
    if file.filename == '':
        return jsonify({"success": False, "error": "No selected file"})
    
    if file and allowed_file(file.filename):
        # Generate unique ID for the PDF
        pdf_id = str(uuid.uuid4())
        filename = secure_filename(f"{pdf_id}.pdf")
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # Process the PDF asynchronously - just save the file for now
        # The actual processing will happen when needed
        try:
            # Start processing in background but don't wait for completion
            process_pdf(pdf_id, file_path)
            pdf_url = f"/uploads/{filename}"
            return jsonify({
                "success": True, 
                "pdf_id": pdf_id,
                "pdf_url": pdf_url
            })
        except Exception as e:
            # Clean up the file if processing failed
            if os.path.exists(file_path):
                os.remove(file_path)
            return jsonify({"success": False, "error": str(e)})
    
    return jsonify({"success": False, "error": "Invalid file type"})

@app.route("/remove_pdf", methods=["POST"])
def remove_pdf():
    pdf_id = request.json.get("pdf_id")
    if not pdf_id:
        return jsonify({"success": False, "error": "No PDF ID provided"})
    
    try:
        # Remove from memory
        remove_pdf_from_memory(pdf_id)
        
        # Remove file from disk
        filename = secure_filename(f"{pdf_id}.pdf")
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if os.path.exists(file_path):
            os.remove(file_path)
        
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# Add this to the BOTTOM of your app.py
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port, debug=False)
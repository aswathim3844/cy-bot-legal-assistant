import os
import joblib
import uuid
import fitz  # PyMuPDF for PDF processing
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder
from typing import List, Tuple
import ollama
from langchain_core.embeddings import Embeddings
import threading
import time

# --- Create the same Custom Embeddings Class for Ollama ---
class OllamaLocalEmbeddings(Embeddings):
    def __init__(self, model: str):
        self.model = model

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return ollama.embed(model=self.model, input=texts).embeddings

    def embed_query(self, text: str) -> List[float]:
        return ollama.embed(model=self.model, input=text).embeddings[0]

# Load environment variables
load_dotenv()

# Load your ML classifier
intent_classifier = joblib.load("intent_classifier.pkl")

# --- Initialize the custom Ollama Embedding Model ---
embeddings = OllamaLocalEmbeddings(model="nomic-embed-text")

# Load the FAISS index built with the same model
vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# --- Initialize Groq's Cloud LLM ---
llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.3)

# --- Initialize the Re-ranker Model ---
reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

# Store PDF vectorstores in memory
pdf_vectorstores = {}
# Track PDF processing status
pdf_processing_status = {}

# Enhanced System Prompt with Scope Control
SYSTEM_PROMPT = """
You are Cy-Bot, a friendly and empathetic guide for Kerala's cyber laws.

GREETING RESPONSES:
1. Theses queries must be simple very short     and direct and end with the question and aim to help the user .

CRITICAL SCOPE RULES:
1. Your PRIMARY function is to answer questions specifically about Kerala's cyber laws, cybersecurity, digital rights, and related legal matters.
2. If a question is CLEARLY outside this scope (e.g., about national politics, general knowledge, personal advice, other states' laws), you MUST politely decline to answer.
3. You can answer general cybersecurity questions that are relevant to Kerala context.
4. You MUST indicate the source of your information in your response.

SOURCE ATTRIBUTION FORMAT:
- If information comes from Kerala cyber laws knowledge base: "Based on Kerala's cyber laws..."
- If information comes from uploaded PDF: "According to the document you provided..."
- If information comes from both: "Based on Kerala's cyber laws and the document you provided..."
- If no relevant information found: "I couldn't find specific information about this in Kerala's cyber laws or your uploaded documents."

CRITICAL FORMATTING RULES:
1. You MUST format your response using HTML for display in a web browser.
2. Use <p> and </p> tags for paragraphs.
3. Use <strong> and </strong> tags for important terms.
4. Do NOT use asterisks (*) for bolding.
5. There should be a blank line between each paragraph.

OTHER RULES:
- Be empathetic and polite.
- You are NOT a lawyer. Do not give legal advice.
- Base your answers ONLY on the provided context.

Always start with source attribution, then provide the answer if within scope, or politely decline if out of scope.
"""

def predict_intent(query: str) -> str:
    try:
        return intent_classifier.predict([query])[0]
    except Exception as e:
        return f"IntentError: {e}"

def is_question_in_scope(query: str, context: str) -> Tuple[bool, str]:
    """
    Check if the question is within the scope of Kerala cyber laws
    Returns: (is_in_scope, reason)
    """
    # List of out-of-scope topics
    out_of_scope_keywords = [
        'national politics', 'central government', 'prime minister', 'president',
        'other state', 'delhi', 'mumbai', 'chennai', 'bangalore', 'tamil nadu', 'karnataka',
        'general knowledge', 'history', 'geography', 'science', 'mathematics',
        'personal advice', 'medical', 'health', 'relationship',
        'cooking', 'recipes', 'sports', 'entertainment', 'movies'
    ]
    
    query_lower = query.lower()
    
    # Check for obvious out-of-scope topics
    for keyword in out_of_scope_keywords:
        if keyword in query_lower:
            return False, f"this question about '{keyword}' is outside my scope of Kerala cyber laws"
    
    # If context is empty or very generic, question is likely out of scope
    if not context or "No relevant context found" in context or len(context.strip()) < 50:
        return False, "I couldn't find relevant information about this in Kerala's cyber laws knowledge base"
    
    return True, "Question is within scope of Kerala cyber laws"

def get_relevant_documents(user_question: str, has_pdf: bool = False, pdf_id: str = None) -> Tuple[List[Document], List[Document]]:
    """
    Retrieve relevant documents from both knowledge base and PDF
    Returns: (kb_docs, pdf_docs)
    """
    # Get documents from knowledge base
    try:
        kb_docs = retriever.invoke(user_question)
    except AttributeError:
        kb_docs = retriever.get_relevant_documents(user_question)
    
    pdf_docs = []
    # Get documents from PDF if available and processed
    if has_pdf and pdf_id:
        # Check if PDF is still processing
        if pdf_id in pdf_processing_status and pdf_processing_status[pdf_id] == "processing":
            return kb_docs, pdf_docs
            
        if pdf_id in pdf_vectorstores:
            pdf_retriever = pdf_vectorstores[pdf_id].as_retriever(search_kwargs={"k": 5})
            try:
                pdf_docs = pdf_retriever.invoke(user_question)
            except AttributeError:
                pdf_docs = pdf_retriever.get_relevant_documents(user_question)
    
    # Re-rank all documents if we have any
    all_docs = kb_docs + pdf_docs
    if all_docs:
        rerank_pairs = [[user_question, d.page_content] for d in all_docs]
        scores = reranker.predict(rerank_pairs)
        scored_docs = list(zip(all_docs, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        all_docs = [doc for doc, score in scored_docs[:3]]
        
        # Separate them back
        kb_docs = [doc for doc in all_docs if doc.metadata.get("source") != "pdf"]
        pdf_docs = [doc for doc in all_docs if doc.metadata.get("source") == "pdf"]
    
    return kb_docs, pdf_docs

def generate_source_attribution(kb_docs: List[Document], pdf_docs: List[Document], pdf_id: str = None) -> str:
    """Generate appropriate source attribution based on where information came from"""
    has_kb = len(kb_docs) > 0
    has_pdf = len(pdf_docs) > 0
    
    # Check if PDF is still processing
    pdf_processing = pdf_id and pdf_id in pdf_processing_status and pdf_processing_status[pdf_id] == "processing"
    
    if has_kb and has_pdf:
        return "Based on Kerala's cyber laws and the document you provided"
    elif has_kb and pdf_processing:
        return "Based on Kerala's cyber laws (PDF is still processing)"
    elif has_kb:
        return "Based on Kerala's cyber laws"
    elif has_pdf:
        return "According to the document you provided"
    else:
        return "I couldn't find specific information about this in Kerala's cyber laws or your uploaded documents"

def process_pdf_background(pdf_id: str, file_path: str):
    """Process PDF in background thread"""
    try:
        print(f"Starting background processing for PDF: {pdf_id}")
        doc = fitz.open(file_path)
        text = ""
        
        # Extract text from first 10 pages only for faster processing
        max_pages = min(10, len(doc))
        for page_num in range(max_pages):
            page = doc[page_num]
            text += page.get_text()
        
        # Use larger chunks for faster processing
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,  # Larger chunks
            chunk_overlap=100  # Smaller overlap
        )
        chunks = text_splitter.split_text(text)
        
        documents = [Document(page_content=chunk, metadata={"source": "pdf", "pdf_id": pdf_id}) for chunk in chunks]
        
        # Create vectorstore
        pdf_vectorstore = FAISS.from_documents(documents, embeddings)
        pdf_vectorstores[pdf_id] = pdf_vectorstore
        
        # Mark as completed
        pdf_processing_status[pdf_id] = "completed"
        print(f"Completed background processing for PDF: {pdf_id}")
        
    except Exception as e:
        print(f"Error processing PDF in background: {e}")
        pdf_processing_status[pdf_id] = "error"

def process_pdf(pdf_id: str, file_path: str):
    """Start PDF processing in background"""
    # Mark as processing
    pdf_processing_status[pdf_id] = "processing"
    
    # Start background thread
    thread = threading.Thread(target=process_pdf_background, args=(pdf_id, file_path))
    thread.daemon = True
    thread.start()
    
    return True

def remove_pdf_from_memory(pdf_id: str):
    if pdf_id in pdf_vectorstores:
        del pdf_vectorstores[pdf_id]
    if pdf_id in pdf_processing_status:
        del pdf_processing_status[pdf_id]

def get_bot_response(user_question: str, has_pdf: bool = False, pdf_id: str = None) -> str:
    try:
        intent = predict_intent(user_question)

        # Get relevant documents from both sources
        kb_docs, pdf_docs = get_relevant_documents(user_question, has_pdf, pdf_id)
        
        # Combine context
        kb_context = "\n\n".join([d.page_content for d in kb_docs]) if kb_docs else ""
        pdf_context = "\n\n".join([d.page_content for d in pdf_docs]) if pdf_docs else ""
        
        full_context = ""
        if kb_context and pdf_context:
            full_context = f"KNOWLEDGE BASE CONTEXT:\n{kb_context}\n\nPDF CONTEXT:\n{pdf_context}"
        elif kb_context:
            full_context = f"KNOWLEDGE BASE CONTEXT:\n{kb_context}"
        elif pdf_context:
            full_context = f"PDF CONTEXT:\n{pdf_context}"
        else:
            full_context = "No relevant context found in knowledge base or uploaded documents."
        
        # Check if question is in scope
        is_in_scope, scope_reason = is_question_in_scope(user_question, full_context)
        
        # Generate source attribution
        source_attribution = generate_source_attribution(kb_docs, pdf_docs, pdf_id)
        
        if not is_in_scope:
            # For out-of-scope questions, provide polite refusal
            return f'<p><strong>Note:</strong> I specialize in Kerala cyber laws and cybersecurity matters.</p><p>I\'m unable to answer questions about {scope_reason}.</p><p>Please ask me about Kerala\'s cyber laws, digital rights, or cybersecurity issues relevant to Kerala.</p>'

        # For in-scope questions, generate detailed response
        prompt = f"""
{SYSTEM_PROMPT}

Intent: {intent}
Source Attribution: {source_attribution}

Context:
{full_context}

Question: {user_question}

Answer (start with source attribution, then provide helpful information):
"""

        # Generate answer using the Groq LLM
        response = llm.invoke(prompt)
        return response.content.strip()

    except Exception as e:
        return f'<p>Sorry, I encountered an error while processing your question. Please try again.</p><p>Error: {str(e)}</p>'

# Local testing
if __name__ == "__main__":
    print("✅ Enhanced Cy-Bot backend loaded successfully.")
    print("Testing scope detection and source attribution...")
    
    # Test cases
    test_questions = [
        "What are the penalties for cyber bullying in Kerala?",
        "Who is the prime minister of India?",
        "How to make biryani?",
        "What cybersecurity measures should Kerala businesses take?"
    ]
    
    for question in test_questions:
        print(f"\nYou: {question}")
        print("Cy-Bot:", get_bot_response(question))
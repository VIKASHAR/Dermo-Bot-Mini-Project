from flask import Flask, render_template, request, jsonify
from sklearnex import patch_sklearn
patch_sklearn()  # Use optimized scikit-learn
import fitz  # PyMuPDF for PDF handling
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize the Flask app
app = Flask(__name__)

# Get the Google API key from the environment variable
api_key = os.getenv("GOOGLE_API_KEY")  # Ensure to set this in your .env file

# Path to the single PDF file
pdf_file = "Prevention of Skin Cancer.pdf"

def get_pdf_text(pdf_file_path):
    """Extract text from a PDF file."""
    text = ""
    with fitz.open(pdf_file_path) as doc:
        for page in doc:
            text += page.get_text()
    return text

def get_text_chunks(text):
    """Split text into chunks for processing."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks, api_key):
    """Create and save a FAISS vector store from text chunks."""
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def initialize_vector_store():
    """Initialize vector store from the PDF file if it doesn't already exist."""
    if not os.path.exists("faiss_index"):
        print("Creating FAISS vector store from the PDF file...")
        raw_text = get_pdf_text(pdf_file)
        text_chunks = get_text_chunks(raw_text)
        get_vector_store(text_chunks, api_key)
        print("Vector store created successfully!")

def get_conversational_chain():
    """Set up the conversational chain for Q&A."""
    prompt_template = """
        You are a healthcare assistant specializing in dermatology. When a user asks about a specific skin issue, respond in a conversational manner. Provide a comprehensive answer that covers the symptoms they might experience, effective treatments, home remedies, and practical tips for prevention. Your goal is to create a supportive dialogue, ensuring the user feels heard and understood while also offering valuable insights for their skin health.

        Context: {context}
        Question: {question}
        Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3, google_api_key=api_key)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question):
    """Handle user input and generate a response."""
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    if not docs:
        return "No matches found."

    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)

    return response.get("output_text", "No response generated.")

@app.route('/')
def index():
    """Render the main chat interface."""
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    """Handle chat requests."""
    user_question = request.form.get('user_question')
    
    if not user_question:
        return jsonify({'error': 'No question provided.'})
    
    response = user_input(user_question)
    
    return jsonify({'response': response})

# Initialize the vector store when the app starts
initialize_vector_store()

if __name__ == '__main__':
    app.run(debug=True)  # Run the app in debug mode

# ===================================================================================
# Project: ChatTensorFlow
# File: app/main.py
# Description: This file Orchasterates the main TensorFlow Application.
# Author: LALAN KUMAR
# Created: [16-05-2025]
# Updated: [16-05-2025]
# LAST MODIFIED BY: LALAN KUMAR [https://github.com/kumar8074]
# Version: 1.0.0
# ===================================================================================

from flask import Flask, render_template, request, jsonify, session
import os
import sys
from uuid import uuid4
import asyncio
from langchain_core.messages import HumanMessage
from langsmith import Client
from langchain.callbacks.tracers import LangChainTracer

# Add root path
current_file_path = os.path.abspath(__file__)
project_root = os.path.abspath(os.path.join(current_file_path, "../.."))
if project_root not in sys.path:
    sys.path.append(project_root)

# Import the router graph and settings
from app.graphs.states import AgentState
from app.graphs.router import create_router_graph
from config import settings

from langchain.callbacks.base import AsyncCallbackHandler

class StepStreamer(AsyncCallbackHandler):
    """Collect every token the LLM emits and stash it in `self.tokens`."""
    def __init__(self):
        super().__init__()
        self.tokens: list[str] = []

    async def on_llm_new_token(self, token: str, **kwargs):
        # called for every new token
        self.tokens.append(token)



app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', str(uuid4()))

# Create the router graph
router_graph = create_router_graph()

langsmith_client = Client(api_key=os.environ.get("LANGSMITH_API_KEY"))
tracer = LangChainTracer(client=langsmith_client, project_name="TensorFlowAssistantProject")

# Define models that are available and their corresponding LLM providers
MODELS = {
    "gpt-4o": "OpenAI GPT-4o",
    "claude-3-sonnet": "Claude 3 Sonnet",
    "gemini-2.0-flash": "Gemini 2.0 flash",
    "command": "Cohere Command"
}

# Map frontend model selections to LLM providers
MODEL_TO_PROVIDER = {
    "gpt-4o": settings.LLM_PROVIDER_OPENAI,
    "claude-3-sonnet": settings.LLM_PROVIDER_ANTHROPIC,
    "gemini-2.0-flash": settings.LLM_PROVIDER_GEMINI,
    "command": settings.LLM_PROVIDER_COHERE
}

# Sample suggestions
SUGGESTIONS = [
    "How do I compile and train a Keras model?",
    "How do I use tf.data.Dataset to load and preprocess large datasets?",
    "What are the parameters for a Conv2D layer?",
    "How do I create and use a custom loss function?"
]

# In-memory chat history storage
# In a production app, you'd use a database instead
chat_history = {}


@app.route('/')
def index():
    # Set default model if none is selected
    if 'selected_model' not in session:
        session['selected_model'] = 'gemini-2.0-flash'
    
    # Generate a unique session ID if none exists
    if 'session_id' not in session:
        session['session_id'] = str(uuid4())
        chat_history[session['session_id']] = []
    
    return render_template('index.html', 
                          models=MODELS, 
                          selected_model=session['selected_model'],
                          selected_model_name=MODELS[session['selected_model']],
                          suggestions=SUGGESTIONS,
                          chat_history=chat_history.get(session['session_id'], []))

@app.route('/select_model', methods=['POST'])
def select_model():
    model_id = request.form.get('model')
    if model_id in MODELS:
        session['selected_model'] = model_id
        
        # Update the LLM provider based on the selected model
        llm_provider = MODEL_TO_PROVIDER[model_id]
        os.environ["LLM_PROVIDER"] = llm_provider
        os.environ["EMBEDDING_PROVIDER"] = llm_provider
        
        # Update settings
        settings.LLM_PROVIDER = llm_provider
        settings.EMBEDDING_PROVIDER = llm_provider
        
        return jsonify({"status": "success", "model": MODELS[model_id]})
    return jsonify({"status": "error", "message": "Invalid model selection"})

@app.route('/send_message', methods=['POST'])
def send_message():
    # Support both JSON and form data
    if request.is_json:
        message = request.json.get('message', '').strip()
    else:
        message = request.form.get('message', '').strip()
    model_id = session.get('selected_model', 'gemini-2.0-flash')
    session_id = session.get('session_id')

    if not message:
        return jsonify({"status": "error", "message": "Message cannot be empty"})

    streamer = StepStreamer()

    # Prepare config with thread_id for persistence
    config = {
        "configurable": {"thread_id": session_id},
        "callbacks":[tracer, streamer],
        "streaming": True
    }

    # Try to get current state from graph
    try:
        latest_state = router_graph.get_state(config)
        current_messages = latest_state.values.get("messages", [])
    except Exception:
        current_messages = []

    # Append new user message
    current_messages.append(HumanMessage(content=message))

    # Prepare input state
    state = AgentState(messages=current_messages)

    # Set LLM provider environment variables
    llm_provider = MODEL_TO_PROVIDER[model_id]
    os.environ["LLM_PROVIDER"] = llm_provider
    os.environ["EMBEDDING_PROVIDER"] = llm_provider
    settings.LLM_PROVIDER = llm_provider
    settings.EMBEDDING_PROVIDER = llm_provider

    # Run the graph asynchronously
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        result = loop.run_until_complete(router_graph.ainvoke(state, config))
        full_streamed_text="".join(streamer.tokens)
        #print("STREAMED:" ,full_streamed_text)
        assistant_messages = result.get("messages", [])
        if assistant_messages:
            current_messages.extend(assistant_messages)
        # Update chat history for UI (optional)
        chat_history.setdefault(session_id, []).append(message)
        if assistant_messages:
            chat_history[session_id].append(assistant_messages[-1].content)
        response_content = assistant_messages[-1].content if assistant_messages else ""
        return jsonify({
            "status": "success",
            "message": message,
            "response": response_content,
            "streamed_text": full_streamed_text,
            "model": MODELS[model_id]
        })
    finally:
        loop.close()

@app.route('/get_suggestions')
def get_suggestions():
    return jsonify({"suggestions": SUGGESTIONS})

@app.route('/get_chat_history')
def get_chat_history():
    session_id = session.get('session_id')
    return jsonify({"history": chat_history.get(session_id, [])})

@app.route('/clear_chat_history', methods=['POST'])
def clear_chat_history():
    session_id = session.get('session_id')
    if session_id in chat_history:
        chat_history[session_id] = []
    return jsonify({"status": "success"})

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8000,debug=True)

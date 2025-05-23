import pytest
import os
import sys

# Add root path
current_file_path = os.path.abspath(__file__)
project_root = os.path.abspath(os.path.join(current_file_path, "../.."))
if project_root not in sys.path:
    sys.path.append(project_root)

from app.main import app

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_index_route(client):
    res = client.get('/')
    assert res.status_code == 200
    assert b"Chat TensorFlow" in res.data

def test_get_suggestions(client):
    res = client.get('/get_suggestions')
    assert res.status_code == 200
    assert res.json["suggestions"]

def test_model_selection(client):
    res = client.post('/select_model', data={"model": "gpt-4o"})
    assert res.status_code == 200
    assert res.json["status"] == "success"

def test_send_message(client):
    res = client.post('/send_message', data={"message": "How to build convolutional neural network?"})
    assert res.status_code == 200
    assert res.json["status"] == "success"
    assert "neural network" in res.json["response"]

def test_clear_history(client):
    res = client.post('/clear_chat_history')
    assert res.status_code == 200
    assert res.json["status"] == "success"

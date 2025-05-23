import pytest
import os
import sys

# Add root path
current_file_path = os.path.abspath(__file__)
project_root = os.path.abspath(os.path.join(current_file_path, "../.."))
if project_root not in sys.path:
    sys.path.append(project_root)

from app.graphs.router import create_router_graph
from app.graphs.states import InputState, AgentState
from langchain_core.messages import HumanMessage

@pytest.mark.asyncio
async def test_sklean_graph_respond_to_sklearn_question():
    graph = create_router_graph()
    input_state = InputState(messages=[HumanMessage(content="How do I create and use a custom loss function?")])
    result = await graph.ainvoke(input_state)
    
    assert isinstance(result, dict)
    assert "messages" in result
    assert any("loss function" in msg.content for msg in result["messages"] if hasattr(msg, "content"))

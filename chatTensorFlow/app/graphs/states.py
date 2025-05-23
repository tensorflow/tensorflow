# ===================================================================================
# Project: ChatTensorFlow
# File: app/graphs/states.py
# Description: This file contains the state schemas used by the graphs.
# Author: LALAN KUMAR
# Created: [15-05-2025]
# Updated: [15-05-2025]
# LAST MODIFIED BY: LALAN KUMAR [https://github.com/kumar8074]
# Version: 1.0.0
# ===================================================================================

import os
import sys
from dataclasses import dataclass, field
from typing_extensions import Annotated, Literal, TypedDict
from langchain_core.documents import Document
from langgraph.graph import add_messages
from langchain_core.messages import AnyMessage

# Dynamically add the project root directory to sys.path
current_file_path = os.path.abspath(__file__)
project_root = os.path.abspath(os.path.join(current_file_path, "../../.."))
if project_root not in sys.path:
    sys.path.append(project_root)

from app.core.utils import reduce_docs

@dataclass(kw_only=True)
class QueryState:
    """Private state for the retrieve_documents node in the researcher graph"""
    query: str

@dataclass(kw_only=True)
class ResearcherState:
    """State of the researcher graph"""
    question: str
    queries: list[str] = field(default_factory=list)
    documents: Annotated[list[Document], reduce_docs] = field(default_factory=list)

@dataclass(kw_only=True)
class InputState:
    """Represents the input state for the agent"""
    messages: Annotated[list[AnyMessage], add_messages]

class Router(TypedDict):
    """Classify user query"""
    logic: str
    type: Literal["more-info", "tensorflow", "general"]

@dataclass(kw_only=True)
class AgentState(InputState):
    """State of the retrieval graph"""
    router: Router = field(default_factory=lambda: Router(type="general", logic=""))
    steps: list[str] = field(default_factory=list)
    documents: Annotated[list[Document], reduce_docs] = field(default_factory=list)
    summary: str=""

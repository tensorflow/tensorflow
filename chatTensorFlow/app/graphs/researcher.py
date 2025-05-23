# ===================================================================================
# Project: ChatTensorFlow
# File: app/graphs/researcher.py
# Description: This file contains the implementation of Researcher sub-Graph.
# Author: LALAN KUMAR
# Created: [15-05-2025]
# Updated: [15-05-2025]
# LAST MODIFIED BY: LALAN KUMAR [https://github.com/kumar8074]
# Version: 1.0.0
# ===================================================================================

import os
import sys
import asyncio
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send
from typing_extensions import TypedDict, cast
from langchain_core.runnables import RunnableConfig

# Dynamically add the project root directory to sys.path
current_file_path = os.path.abspath(__file__)
project_root = os.path.abspath(os.path.join(current_file_path, "../../.."))
if project_root not in sys.path:
    sys.path.append(project_root)

from app.graphs.states import ResearcherState, QueryState
from app.graphs.prompts import GENERATE_QUERIES_SYSTEM_PROMPT
from app.core.llm import get_llm
from app.retriever.retriever import get_retriever
from langchain_core.documents import Document

async def generate_queries(
    state: ResearcherState, *, config: RunnableConfig
) -> dict[str, list[str]]:
    """Generate search queries based on the question (a step in the research plan)."""
    
    class Response(TypedDict):
        queries: list[str]
    
    llm = get_llm(
        streaming=config.get("streaming", False), 
        callbacks=config.get("callbacks", [])
    )
    model = llm.with_structured_output(Response)
    messages = [
        {"role": "system", "content": GENERATE_QUERIES_SYSTEM_PROMPT},
        {"role": "human", "content": state.question},
    ]
    response = cast(Response, await model.ainvoke(messages, config=config))
    return {"queries": response["queries"]}

async def retrieve_documents(
    state: QueryState, *, config: RunnableConfig
) -> dict[str, list[Document]]:
    """Retrieve documents based on a given query."""
    
    retriever = get_retriever()
    response = await retriever.ainvoke(state.query, config)
    return {"documents": response}

def retrieve_in_parallel(state: ResearcherState) -> list[Send]:
    """Create parallel retrieval tasks for each generated query."""
    return [
        Send("retrieve_documents", QueryState(query=query)) for query in state.queries
    ]

def create_researcher_graph():
    """Create and return the researcher graph."""
    builder = StateGraph(ResearcherState)
    builder.add_node("generate_queries", generate_queries)
    builder.add_node("retrieve_documents", retrieve_documents)
    builder.add_edge(START, "generate_queries")
    builder.add_conditional_edges(
        "generate_queries",
        retrieve_in_parallel,  
        path_map=["retrieve_documents"],
    )
    builder.add_edge("retrieve_documents", END)
    
    # Compile into a graph object
    researcher_graph = builder.compile()
    researcher_graph.name = "ResearcherGraph"
    
    return researcher_graph

# Example usage:
#graph=create_researcher_graph()
#print(graph)
#print(graph.name)
#print(graph.nodes)

#state = ResearcherState(question="How to build a convolutional neural network?")
#result = asyncio.run(graph.ainvoke(state))
#print(result)

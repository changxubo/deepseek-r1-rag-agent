"""Main entrypoint for the conversational retrieval graph.

This module defines the core structure and functionality of the conversational
retrieval graph. It includes the main graph definition, state management,
and key functions for processing & routing user queries, generating research plans to answer user questions,
conducting research, and formulating responses.
"""

import json
import time
from typing import Any, Literal, cast

from langchain_core.messages import BaseMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from retrieval_graph.configuration import AgentConfiguration
from retrieval_graph.researcher_graph.graph import graph as researcher_graph
from retrieval_graph.state import AgentState, InputState, Router
from shared.utils import format_docs, load_chat_model
from pydantic import BaseModel, Field
import streamlit as st


class RouterResult(BaseModel):
    """Classify user query."""

    logic: str = Field(
        description="The logic to use for the query. This is the name of the function to call."
    )
    type: Literal["more-info", "langchain", "general"] = Field(
        description="The type of query. This is the type of query to route to."
    )


async def analyze_and_route_query(
    state: AgentState, *, config: RunnableConfig
) -> dict[str, Router]:
    """Analyze the user's query and determine the appropriate routing.

    This function uses a language model to classify the user's query and decide how to route it
    within the conversation flow.

    Args:
        state (AgentState): The current state of the agent, including conversation history.
        config (RunnableConfig): Configuration with the model used for query analysis.

    Returns:
        dict[str, Router]: A dictionary containing the 'router' key with the classification result (classification type and logic).
    """

    configuration = AgentConfiguration.from_runnable_config(config)
    model = load_chat_model(configuration.query_model)
    messages = [
        {"role": "system", "content": configuration.router_system_prompt}
    ] + state.messages
    response = cast(
        RouterResult, await model.with_structured_output(RouterResult).ainvoke(messages)
    )
    router = Router(
        logic=response.logic,
        type=response.type,
    )

    return {"router": router}


def route_query(
    state: AgentState,
) -> Literal["create_research_plan", "ask_for_more_info", "respond_to_general_query"]:
    """Determine the next step based on the query classification.

    Args:
        state (AgentState): The current state of the agent, including the router's classification.

    Returns:
        Literal["create_research_plan", "ask_for_more_info", "respond_to_general_query"]: The next step to take.

    Raises:
        ValueError: If an unknown router type is encountered.
    """
    _type = state.router["type"]
    if _type == "langchain":
        return "create_research_plan"
    elif _type == "more-info":
        return "ask_for_more_info"
    elif _type == "general":
        return "respond_to_general_query"
    else:
        raise ValueError(f"Unknown router type {_type}")


async def ask_for_more_info(
    state: AgentState, *, config: RunnableConfig
) -> dict[str, list[BaseMessage]]:
    """Generate a response asking the user for more information.

    This node is called when the router determines that more information is needed from the user.

    Args:
        state (AgentState): The current state of the agent, including conversation history and router logic.
        config (RunnableConfig): Configuration with the model used to respond.

    Returns:
        dict[str, list[str]]: A dictionary with a 'messages' key containing the generated response.
    """

    configuration = AgentConfiguration.from_runnable_config(config)
    model = load_chat_model(configuration.query_model)
    system_prompt = configuration.more_info_system_prompt.format(
        logic=state.router["logic"]
    )
    messages = [{"role": "system", "content": system_prompt}] + state.messages

    response = await model.ainvoke(messages)

    return {"messages": [response]}


async def respond_to_general_query(
    state: AgentState, *, config: RunnableConfig
) -> dict[str, list[BaseMessage]]:
    """Generate a response to a general query not related to LangChain.

    This node is called when the router classifies the query as a general question.

    Args:
        state (AgentState): The current state of the agent, including conversation history and router logic.
        config (RunnableConfig): Configuration with the model used to respond.

    Returns:
        dict[str, list[str]]: A dictionary with a 'messages' key containing the generated response.
    """

    configuration = AgentConfiguration.from_runnable_config(config)
    model = load_chat_model(configuration.query_model)
    system_prompt = configuration.general_system_prompt.format(
        logic=state.router["logic"]
    )
    messages = [{"role": "system", "content": system_prompt}] + state.messages

    response = await model.ainvoke(messages)
    return {"messages": [response]}


async def create_research_plan(
    state: AgentState, *, config: RunnableConfig
) -> dict[str, list[str] | str]:
    """Create a step-by-step research plan for answering a LangChain-related query.

    Args:
        state (AgentState): The current state of the agent, including conversation history.
        config (RunnableConfig): Configuration with the model used to generate the plan.

    Returns:
        dict[str, list[str]]: A dictionary with a 'steps' key containing the list of research steps.
    """

    # class Plan(TypedDict):
    #    """Generate research plan."""

    #    steps: list[str]
    class PlanResult(BaseModel):
        """Generate research plan."""

        steps: list[str]

    configuration = AgentConfiguration.from_runnable_config(config)
    model = load_chat_model(configuration.query_model).with_structured_output(
        PlanResult
    )
    messages = [
        {"role": "system", "content": configuration.research_plan_system_prompt}
    ] + state.messages

    response = cast(PlanResult, await model.ainvoke(messages))

    return {"steps": response.steps, "documents": "delete"}


async def conduct_research(state: AgentState) -> dict[str, Any]:
    """Execute the first step of the research plan.

    This function takes the first step from the research plan and uses it to conduct research.

    Args:
        state (AgentState): The current state of the agent, including the research plan steps.

    Returns:
        dict[str, list[str]]: A dictionary with 'documents' containing the research results and
                              'steps' containing the remaining research steps.

    Behavior:
        - Invokes the researcher_graph with the first step of the research plan.
        - Updates the state with the retrieved documents and removes the completed step.
    """

    result = await researcher_graph.ainvoke({"question": state.steps[0]})
    return {"documents": result["documents"], "steps": state.steps[1:]}


def check_finished(state: AgentState) -> Literal["respond", "conduct_research"]:
    """Determine if the research process is complete or if more research is needed.

    This function checks if there are any remaining steps in the research plan:
        - If there are, route back to the `conduct_research` node
        - Otherwise, route to the `respond` node

    Args:
        state (AgentState): The current state of the agent, including the remaining research steps.

    Returns:
        Literal["respond", "conduct_research"]: The next step to take based on whether research is complete.
    """
    if len(state.steps or []) > 0:
        return "conduct_research"
    else:
        return "respond"


async def respond(
    state: AgentState, *, config: RunnableConfig
) -> dict[str, list[BaseMessage]]:
    """Generate a final response to the user's query based on the conducted research.

    This function formulates a comprehensive answer using the conversation history and the documents retrieved by the researcher.

    Args:
        state (AgentState): The current state of the agent, including retrieved documents and conversation history.
        config (RunnableConfig): Configuration with the model used to respond.

    Returns:
        dict[str, list[str]]: A dictionary with a 'messages' key containing the generated response.
    """

    configuration = AgentConfiguration.from_runnable_config(config)
    model = load_chat_model(configuration.response_model)
    context = format_docs(state.documents)
    prompt = configuration.response_system_prompt.format(context=context)
    messages = [{"role": "system", "content": prompt}] + state.messages
    response = await model.ainvoke(messages)
    return {"messages": [response]}


# Define the graph
builder = StateGraph(AgentState, input=InputState, config_schema=AgentConfiguration)
builder.add_node(analyze_and_route_query)
builder.add_node(ask_for_more_info)
builder.add_node(respond_to_general_query)
builder.add_node(conduct_research)
builder.add_node(create_research_plan)
builder.add_node(respond)

builder.add_edge(START, "analyze_and_route_query")
builder.add_conditional_edges("analyze_and_route_query", route_query)
builder.add_edge("create_research_plan", "conduct_research")
builder.add_conditional_edges("conduct_research", check_finished)
builder.add_edge("ask_for_more_info", END)
builder.add_edge("respond_to_general_query", END)
builder.add_edge("respond", END)
# Create a memory saver to store graph states by thread and allow state recovery
memory = MemorySaver()
# Compile into a graph object that you can invoke and deploy.
graph = builder.compile(checkpointer=memory)
graph.name = "RetrievalGraph"


# Asynchronous function to process events from the graph and update Streamlit UI
async def invoke_graph(st_messages, st_placeholder):
    # Set up placeholders for displaying updates in the Streamlit app
    container = (
        st_placeholder  # This container will hold the dynamic Streamlit UI components
    )
    thoughts_placeholder = (
        container.container()
    )  # Container for displaying status messages
    
    # Placeholder for displaying status messages
    token_placeholder = (
        container.empty()
    )  # Placeholder for displaying progressive token updates
    
    final_text = ""
    thoughts = [""]  # Initialize an empty list to store thoughts
    # Invoke the graph with the current messages and callback configuration
    async for event in graph.astream_events(
        {"messages": st_messages},
        version="v2",
        config={
            "configurable": {
                "thread_id": "12",
                "checkpoint_ns": "retrieval_graph",
                "checkpoint_id": "12",
            }
        },
    ):
        kind = event["name"]  # Determine the type of event received
        trigger = event["event"]
        meta = event["metadata"]
        #print(kind, trigger)
        if kind == "analyze_and_route_query" and trigger == "on_chain_start":
            start_time = time.time()  # Record the start time
            thinked_time =time.time()-start_time
            with thoughts_placeholder:
                status_placeholder = st.empty()  
                with status_placeholder.status(
                    "Thinking ...", expanded=False
                ) as s:
                    output_placeholder = st.empty()
        elif kind == "RetrievalGraph" and trigger == "on_chain_end":
            with thoughts_placeholder:
                with status_placeholder.status(
                    "Thinking ...", expanded=False
                ) as s:
                    thinked_time =time.time()-start_time
                    s.update(label=f"Thinked in {int(thinked_time)} seconds.",expanded=False)  # Update the status message with total tokens consumed
                    output_placeholder.write("\n\n".join(thoughts)) 
        elif kind == "analyze_and_route_query" and trigger == "on_chain_end":
            if "output" in event["data"] and "router" in event["data"]["output"]:
                thoughts.append(f"Analyze and route query: {event["data"]["output"]["router"]["type"]}.")
                output_placeholder.write("\n\n".join(thoughts))         
        elif kind == "create_research_plan" and trigger == "on_chain_end":
            if "output" in event["data"] and "steps" in event["data"]["output"]:
                thoughts.append(f"Create research plan: {" ".join(event["data"]["output"]["steps"])}.")
                output_placeholder.write("\n\n".join(thoughts))    
        elif kind == "generate_queries" and trigger == "on_chain_end":
            if "output" in event["data"] and "queries" in event["data"]["output"]:
                thoughts.append(f"Generate queries: {" && ".join(event["data"]["output"]["queries"])}.")
                output_placeholder.write("\n\n".join(thoughts))    
        elif kind == "conduct_research" and trigger == "on_chain_end":
            if "output" in event["data"] and "documents" in event["data"]["output"]:
                thoughts.append(f"Conduct research and retrived {len(event["data"]["output"]["documents"])} documents.")
                output_placeholder.write("\n\n".join(thoughts))    
        elif kind in ["respond","ask_for_more_info","respond_to_general_query"]and trigger == "on_chain_start":
            if "input" in event["data"] :
                thoughts.append(f"Generate response ...")
                output_placeholder.write("\n\n".join(thoughts))   
        elif trigger == "on_chat_model_stream" and meta["langgraph_node"] in ["respond","ask_for_more_info","respond_to_general_query"]:
            if "chunk" in event["data"]:
                final_text += event["data"]["chunk"].content
                token_placeholder.write(final_text)
            
    return final_text  # Return the final accumulated text

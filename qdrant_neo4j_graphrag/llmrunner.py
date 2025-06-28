from langchain.agents import initialize_agent, Tool, AgentType
from langchain.utilities import DuckDuckGoSearchAPIWrapper
from langchain.schema import SystemMessage, HumanMessage


def global_search_query(query,llm):
    search = DuckDuckGoSearchAPIWrapper()
    
    tools = [
        Tool(
            name="Intermediate_Answer",
            func=search.run,
            description="useful for when you need to ask with search",
        )
    ]

    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=False  # optional: set to False to reduce console output
    )

    result = agent.run(query)
    print(result)


def graphRAG_run(graph_context, user_query , llm):
    nodes_str = ", ".join(graph_context["nodes"])
    edges_str = "; ".join(graph_context["edges"])
    prompt = f"""
    You are an intelligent assistant with access to the following knowledge graph:

    Nodes: {nodes_str}

    Edges: {edges_str}

    Using this graph, Answer the following question:

    User Query: "{user_query}"
    """
    
    try:
        messages = [
            SystemMessage(content="Provide the answer for the following question:"),
            HumanMessage(content=prompt)
        ]

        # Invoke the LLM
        response = llm.invoke(messages)

        # Return the content
        return response.content
    
    except Exception as e:
        return f"Error querying LLM: {str(e)}"
import uuid
import json
from langchain.schema import SystemMessage, HumanMessage
from pydantic import BaseModel

class single(BaseModel):
    node: str
    target_node: str
    relationship: str

class GraphComponents(BaseModel):
    graph: list[single]

def ingest_to_neo4j(nodes, relationships,neo4j_driver):
    """
    Ingest nodes and relationships into Neo4j.
    """

    with neo4j_driver.session() as session:
        # Create nodes in Neo4j
        for name, node_id in nodes.items():
            session.run(
                "CREATE (n:Entity {id: $id, name: $name})",
                id=node_id,
                name=name
            )

        # Create relationships in Neo4j
        for relationship in relationships:
            session.run(
                "MATCH (a:Entity {id: $source_id}), (b:Entity {id: $target_id}) "
                "CREATE (a)-[:RELATIONSHIP {type: $type}]->(b)",
                source_id=relationship["source"],
                target_id=relationship["target"],
                type=relationship["type"]
            )

    return nodes


def is_openai_llm(llm) -> bool:
    md = getattr(llm, "metadata", None) or {}
    name = str(getattr(md, "model_name", "") or "").lower()
    provider = getattr(md, "provider", "") or ""
    if isinstance(provider, (list, tuple)):
        provider = provider[0]
    provider = str(provider).lower()
    return (
        provider in ("openai", "azure_openai")
        or "gpt-" in name
        or "openai" in name
    )

def extract_graph_components(raw_data , llm):
    prompt = f"Extract nodes and relationships from the following text:\n{raw_data}"

    messages = [
        SystemMessage(
            content=(
                "You are a precise graph relationship extractor. "
                "Extract all relationships from the text and format them as a JSON object "
                "with this exact structure:\n"
                '{"graph":[{"node":"Person/Entity","target_node":"Related Entity","relationship":"Type of Relationship"}]}\n'
                "Include ALL relationships mentioned, even implicit ones."
                "If the output is invalid JSON or contains extra text, **do not attempt to self-correct**—simply output **only** the JSON."
            )
        ),
        HumanMessage(content=prompt),
    ]

    try:
        if is_openai_llm(llm):
            response = llm.invoke(messages, response_format={"type": "json_object"})
        else:
            response = llm.invoke(messages)
    except:
        response = llm.invoke(messages)
    # response.content is a JSON string
    parsed = json.loads(response.content)
    
    # Validate with your Pydantic model
    parsed_response =  GraphComponents.model_validate(parsed)

    # parsed_response = openai_llm_parser(prompt)  # Assuming this returns a list of dictionaries
    parsed_response = parsed_response.graph  # Assuming the 'graph' structure is a key in the parsed response

    nodes = {}
    relationships = []

    for entry in parsed_response:
        node = entry.node
        target_node = entry.target_node  # Get target node if available
        relationship = entry.relationship  # Get relationship if available

        # Add nodes to the dictionary with a unique ID
        if node not in nodes:
            nodes[node] = str(uuid.uuid4())

        if target_node and target_node not in nodes:
            nodes[target_node] = str(uuid.uuid4())

        # Add relationship to the relationships list with node IDs
        if target_node and relationship:
            relationships.append({
                "source": nodes[node],
                "target": nodes[target_node],
                "type": relationship
            })

    return nodes, relationships

def fetch_related_graph(neo4j_client, entity_ids):
    query = """
    MATCH (e:Entity)-[r1]-(n1)-[r2]-(n2)
    WHERE e.id IN $entity_ids
    RETURN e, r1 as r, n1 as related, r2, n2
    UNION
    MATCH (e:Entity)-[r]-(related)
    WHERE e.id IN $entity_ids
    RETURN e, r, related, null as r2, null as n2
    """
    with neo4j_client.session() as session:
        result = session.run(query, entity_ids=entity_ids)
        subgraph = []
        for record in result:
            subgraph.append({
                "entity": record["e"],
                "relationship": record["r"],
                "related_node": record["related"]
            })
            if record["r2"] and record["n2"]:
                subgraph.append({
                    "entity": record["related"],
                    "relationship": record["r2"],
                    "related_node": record["n2"]
                })
    return subgraph


def format_graph_context(subgraph):
    nodes = set()
    edges = []

    for entry in subgraph:
        entity = entry["entity"]
        related = entry["related_node"]
        relationship = entry["relationship"]

        nodes.add(entity["name"])
        nodes.add(related["name"])

        edges.append(f"{entity['name']} {relationship['type']} {related['name']}")

    return {"nodes": list(nodes), "edges": edges}
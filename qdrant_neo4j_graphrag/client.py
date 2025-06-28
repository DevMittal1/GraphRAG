import os
import logging
from typing import Optional, Any
from neo4j import GraphDatabase

from .clientqgrant import retriever_search, ingest_to_qdrant
from .llmrunner import global_search_query, graphRAG_run
from .client_neo4j import (
    fetch_related_graph,
    format_graph_context,
    extract_graph_components,
    ingest_to_neo4j,
)

# Setup logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def get_env_var(key: str, default: Optional[str] = None, required: bool = False) -> Optional[str]:
    value = os.getenv(key, default)
    if required and value is None:
        raise EnvironmentError(f"Missing required environment variable: {key}")
    return value


class Graphragclient:
    def __init__(
        self,
        neo4j_uri: Optional[str] = None,
        neo4j_username: Optional[str] = None,
        neo4j_password: Optional[str] = None,
        llm: Any = None,
        qdrant_client: Any = None,
        embedding_model: Any = None,
        global_search: Optional[str] = None,
        use_env_fallback: bool = True,
        qdrant_collection: Any = None,
    ):
        get = lambda k, d=None: get_env_var(k, d) if use_env_fallback else d

        self.neo4j_uri = neo4j_uri or get("NEO4J_URI")
        self.neo4j_username = neo4j_username or get("NEO4J_USERNAME")
        self.neo4j_password = neo4j_password or get("NEO4J_PASSWORD")
        self.global_search = global_search or get("GLOBAL_SEARCH", "OFF")

        try:
            self.neo4j_driver = GraphDatabase.driver(
                self.neo4j_uri, auth=(self.neo4j_username, self.neo4j_password)
            )
            logger.info("Connected to Neo4j successfully.")
        except Exception as e:
            logger.exception("Failed to initialize Neo4j driver.")
            raise

        self.llm = llm
        self.qdrant_client = qdrant_client
        self.embedding_model = embedding_model
        self.qdrant_collection = qdrant_collection

    def __repr__(self):
        return f"<Graphragclient neo4j_uri={self.neo4j_uri}>"

    def search_query(self, query: str) -> str:
        """
        Performs a semantic + graph-based retrieval and returns the final answer from GraphRAG.
        Falls back to global search if no results found.
        """
        try:
            logger.info("Running hybrid retriever for query: %s", query)
            retriever_result = retriever_search(
                self.global_search, self.qdrant_client, self.qdrant_collection, query, self.embedding_model
            )

            if retriever_result is None:
                logger.warning("No results from hybrid retriever. Using global search.")
                return global_search_query(query, self.llm)

            entity_ids = [payload['id'] for _, payload, _ in retriever_result]
            logger.info("Entity IDs retrieved: %s", entity_ids)

            subgraph = fetch_related_graph(self.neo4j_driver, entity_ids)
            graph_context = format_graph_context(subgraph)

            logger.info("Running GraphRAG...")
            return graphRAG_run(graph_context, query, self.llm)

        except Exception as e:
            logger.exception("Error during search query processing.")
            return "An error occurred while processing your query."

    def ingest_data(self, document_to_insert: Any) -> None:
        """
        Ingests document data into Neo4j and Qdrant.
        """
        try:
            logger.info("Extracting graph components...")
            nodes, relationships = extract_graph_components(document_to_insert, self.llm)

            logger.info("Ingesting data into Neo4j...")
            node_id_mapping = ingest_to_neo4j(nodes, relationships, self.neo4j_driver)
            logger.info("Neo4j ingestion complete.")

            logger.info("Ingesting data into Qdrant...")
            ingest_to_qdrant(
                self.qdrant_client,
                self.qdrant_collection,
                document_to_insert,
                node_id_mapping,
                self.embedding_model
            )
            logger.info("Qdrant ingestion complete.")

        except Exception as e:
            logger.exception("Data ingestion failed.")

    def close(self):
        try:
            self.neo4j_driver.close()
            logger.info("Neo4j driver closed successfully.")
        except Exception as e:
            logger.warning("Error closing Neo4j driver: %s", str(e))

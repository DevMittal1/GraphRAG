import uuid

def ingest_to_qdrant(qdrant_client , collection_name, raw_data, node_id_mapping, embedding_model):
    embeddings = [embedding_model.embed_query(paragraph) for paragraph in raw_data.split("\n")]

    qdrant_client.upsert(
        collection_name=collection_name,
        points=[
            {
                "id": str(uuid.uuid4()),
                "vector": embedding,
                "payload": {"id": node_id}
            }
            for node_id, embedding in zip(node_id_mapping.values(), embeddings)
        ]
    )


def retriever_search(global_search, qdrant_client, collection_name, query, embedding_model):

    vector = embedding_model.embed_query(query)

    score_threshold = 0.85 if global_search == "ON" else None

    search_args = dict(
        collection_name=collection_name,
        query_vector=vector,
        limit=5,
        with_payload=True,
    )

    # Only add score_threshold if it's set
    if score_threshold is not None:
        search_args["score_threshold"] = score_threshold

    search_results = qdrant_client.search(**search_args)
    
    if not search_results:
        return None
    # Each hit in search_results is a ScoredPoint
    return [(hit.id, hit.payload, hit.score) for hit in search_results]
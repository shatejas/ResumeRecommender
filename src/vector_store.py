"""OpenSearch vector store with nested chunks per resume."""

from opensearchpy import OpenSearch
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import BaseRetriever, Document
from src.config import (
    OPENSEARCH_HOST, OPENSEARCH_PORT, OPENSEARCH_INDEX, EMBEDDING_MODEL
)

DIMENSION = 384  # all-MiniLM-L6-v2 output dimension
SEARCH_PIPELINE = "resume-hybrid-pipeline"

INDEX_MAPPING = {
    "mappings": {
        "properties": {
            "source": {"type": "keyword"},
            "chunks": {
                "type": "nested",
                "properties": {
                    "text": {"type": "text"},
                    "embedding": {
                        "type": "knn_vector",
                        "dimension": DIMENSION,
                        "method": {
                            "name": "hnsw",
                            "engine": "lucene",
                            "space_type": "l2",
                        },
                    },
                },
            },
        }
    },
    "settings": {
        "index": {"knn": True}
    },
}


def get_client() -> OpenSearch:
    return OpenSearch(
        hosts=[{"host": OPENSEARCH_HOST, "port": OPENSEARCH_PORT}],
        use_ssl=False,
    )


def get_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def ensure_search_pipeline():
    """Create the hybrid search pipeline if it doesn't exist."""
    client = get_client()
    try:
        client.http.get(f"/_search/pipeline/{SEARCH_PIPELINE}")
    except Exception:
        pipeline_body = {
            "description": "Normalization pipeline for hybrid resume search",
            "phase_results_processors": [
                {
                    "normalization-processor": {
                        "normalization": {"technique": "min_max"},
                        "combination": {
                            "technique": "arithmetic_mean",
                            "parameters": {"weights": [0.6, 0.4]},
                        },
                    }
                }
            ],
        }
        client.http.put(
            f"/_search/pipeline/{SEARCH_PIPELINE}",
            body=pipeline_body,
        )


def ensure_index():
    """Delete and recreate the index, and ensure search pipeline exists."""
    client = get_client()
    if client.indices.exists(OPENSEARCH_INDEX):
        client.indices.delete(OPENSEARCH_INDEX)
    client.indices.create(OPENSEARCH_INDEX, body=INDEX_MAPPING)
    ensure_search_pipeline()


def index_resume(source: str, chunks: list[Document]):
    """Index a single resume as one doc with nested chunks."""
    client = get_client()
    embeddings = get_embeddings()

    texts = [c.page_content for c in chunks]
    vectors = embeddings.embed_documents(texts)

    doc = {
        "source": source,
        "chunks": [
            {"text": t, "embedding": v}
            for t, v in zip(texts, vectors)
        ],
    }
    client.index(index=OPENSEARCH_INDEX, body=doc)


def search_resumes(query: str, k: int = 3) -> list[Document]:
    """Hybrid search: combines nested kNN (semantic) + nested BM25 (keyword)."""
    ensure_search_pipeline()
    client = get_client()
    embeddings = get_embeddings()
    query_vector = embeddings.embed_query(query)

    body = {
        "size": k,
        "query": {
            "hybrid": {
                "queries": [
                    {
                        "nested": {
                            "path": "chunks",
                            "query": {
                                "knn": {
                                    "chunks.embedding": {
                                        "vector": query_vector,
                                        "k": k,
                                        "expand_nested_docs": True,
                                    }
                                }
                            },
                            "inner_hits": {"size": 1, "_source": False, "name": "knn_hits"},
                        }
                    },
                    {
                        "nested": {
                            "path": "chunks",
                            "query": {
                                "match": {
                                    "chunks.text": query
                                }
                            },
                            "inner_hits": {"size": 1, "_source": False, "name": "bm25_hits"},
                        }
                    },
                ]
            }
        },
    }

    resp = client.search(
        index=OPENSEARCH_INDEX,
        body=body,
        params={"search_pipeline": SEARCH_PIPELINE},
    )

    print("--- OpenSearch Raw Response ---")
    for hit in resp["hits"]["hits"]:
        print(f"\nScore: {hit['_score']}")
        print(f"Source: {hit['_source']['source']}")
        print(f"Chunks: {len(hit['_source']['chunks'])}")
        for i, chunk in enumerate(hit['_source']['chunks']):
            print(f"  Chunk {i}: {chunk['text'][:100]}...")
    print("--- End Raw Response ---\n")

    results = []
    for hit in resp["hits"]["hits"]:
        source = hit["_source"]["source"]
        all_texts = [chunk["text"] for chunk in hit["_source"]["chunks"]]
        results.append(Document(
            page_content="\n\n".join(all_texts),
            metadata={"source": source},
        ))
    return results


class ResumeRetriever(BaseRetriever):
    """Custom retriever that searches nested resume chunks."""
    k: int = 3

    def _get_relevant_documents(self, query: str, **kwargs) -> list[Document]:
        return search_resumes(query, self.k)

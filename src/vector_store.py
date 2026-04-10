"""OpenSearch vector store with nested chunks per resume."""

from opensearchpy import OpenSearch
from langchain_ollama import OllamaEmbeddings
from langchain.schema import BaseRetriever, Document
from src.config import (
    OPENSEARCH_HOST, OPENSEARCH_PORT, OPENSEARCH_INDEX, EMBEDDING_MODEL
)

DIMENSION = 768
SEARCH_PIPELINE = "resume-hybrid-pipeline"

INDEX_MAPPING = {
    "mappings": {
        "properties": {
            "source": {"type": "keyword"},
            "skills": {"type": "text"},
            "experience": {"type": "text"},
            "chunks": {
                "type": "nested",
                "properties": {
                    "text": {"type": "text"},
                    "embedding": {
                        "type": "knn_vector",
                        "dimension": DIMENSION,
                        "method": {"name": "hnsw", "engine": "lucene", "space_type": "l2"},
                    },
                },
            },
        }
    },
    "settings": {"index": {"knn": True}},
}


def get_client() -> OpenSearch:
    return OpenSearch(hosts=[{"host": OPENSEARCH_HOST, "port": OPENSEARCH_PORT}], use_ssl=False)


def get_embeddings():
    return OllamaEmbeddings(model=EMBEDDING_MODEL)


def ensure_search_pipeline():
    """Create the hybrid search pipeline if it doesn't exist."""
    client = get_client()
    try:
        client.http.get(f"/_search/pipeline/{SEARCH_PIPELINE}")
    except Exception:
        client.http.put(
            f"/_search/pipeline/{SEARCH_PIPELINE}",
            body={
                "description": "Normalization pipeline for hybrid resume search",
                "phase_results_processors": [{
                    "normalization-processor": {
                        "normalization": {"technique": "min_max"},
                        "combination": {
                            "technique": "arithmetic_mean",
                            "parameters": {"weights": [0.6, 0.4]},
                        },
                    }
                }],
            },
        )


def ensure_index():
    """Delete and recreate the index, and ensure search pipeline exists."""
    client = get_client()
    if client.indices.exists(OPENSEARCH_INDEX):
        client.indices.delete(OPENSEARCH_INDEX)
    client.indices.create(OPENSEARCH_INDEX, body=INDEX_MAPPING)
    ensure_search_pipeline()


def index_resume(source: str, skills: str, experience: str, chunks: list[Document]):
    """Index a single resume with structured skills/experience and nested chunks."""
    client = get_client()
    embeddings = get_embeddings()
    texts = [c.page_content for c in chunks]
    vectors = embeddings.embed_documents(texts)

    client.index(
        index=OPENSEARCH_INDEX,
        body={
            "source": source,
            "skills": skills,
            "experience": experience,
            "chunks": [{"text": t, "embedding": v} for t, v in zip(texts, vectors)],
        },
    )


def search_resumes(query: str, k: int = 3) -> list[dict]:
    """Hybrid search. Returns list of dicts with source, skills, experience, content."""
    ensure_search_pipeline()
    client = get_client()
    query_vector = get_embeddings().embed_query(query)

    resp = client.search(
        index=OPENSEARCH_INDEX,
        body={
            "size": k,
            "query": {
                "hybrid": {
                    "queries": [
                        {
                            "nested": {
                                "path": "chunks",
                                "query": {"knn": {"chunks.embedding": {
                                    "vector": query_vector, "k": k, "expand_nested_docs": True,
                                }}},
                                "inner_hits": {"size": 1, "_source": False, "name": "knn_hits"},
                            }
                        },
                        {
                            "nested": {
                                "path": "chunks",
                                "query": {"match": {"chunks.text": query}},
                                "inner_hits": {"size": 1, "_source": False, "name": "bm25_hits"},
                            }
                        },
                    ]
                }
            },
        },
        params={"search_pipeline": SEARCH_PIPELINE},
    )

    return [
        {
            "source": hit["_source"]["source"],
            "skills": hit["_source"].get("skills", ""),
            "experience": hit["_source"].get("experience", ""),
            "content": "\n\n".join(c["text"] for c in hit["_source"]["chunks"]),
        }
        for hit in resp["hits"]["hits"]
    ]


class ResumeRetriever(BaseRetriever):
    """Custom retriever for the RAG chain. Returns content as Documents."""
    k: int = 3

    def _get_relevant_documents(self, query: str, **kwargs) -> list[Document]:
        results = search_resumes(query, self.k)
        return [
            Document(
                page_content=r["content"],
                metadata={"source": r["source"], "skills": r["skills"], "experience": r["experience"]},
            )
            for r in results
        ]

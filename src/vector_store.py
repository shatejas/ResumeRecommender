"""OpenSearch vector store with nested chunks per resume."""

import re
from opensearchpy import OpenSearch
from langchain_ollama import OllamaEmbeddings
from langchain.schema import Document
from src.config import (
    OPENSEARCH_HOST, OPENSEARCH_PORT, OPENSEARCH_INDEX, EMBEDDING_MODEL
)

DIMENSION = 768
SEARCH_PIPELINE = "resume-hybrid-pipeline"

INDEX_MAPPING = {
    "mappings": {
        "properties": {
            "source": {"type": "keyword"},
            "original_path": {"type": "keyword"},
            "skills": {"type": "text"},
            "experience": {"type": "text"},
            "education": {"type": "text"},
            "certifications": {"type": "text"},
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


def ensure_index_exists():
    """Create the index only if it doesn't exist. Does not delete existing data."""
    client = get_client()
    if not client.indices.exists(OPENSEARCH_INDEX):
        client.indices.create(OPENSEARCH_INDEX, body=INDEX_MAPPING)
    ensure_search_pipeline()


def resume_exists(source: str) -> bool:
    """Check if a resume with this source name is already indexed."""
    client = get_client()
    if not client.indices.exists(OPENSEARCH_INDEX):
        return False
    resp = client.search(
        index=OPENSEARCH_INDEX,
        body={"size": 0, "query": {"term": {"source": source}}},
    )
    return resp["hits"]["total"]["value"] > 0


def index_resume(source: str, skills: str, experience: str, chunks: list[Document],
                 education: str = "", certifications: str = "", original_path: str = ""):
    """Index a single resume with structured data and nested chunks."""
    client = get_client()
    embeddings = get_embeddings()
    texts = [c.page_content for c in chunks]
    vectors = embeddings.embed_documents(texts)

    client.index(
        index=OPENSEARCH_INDEX,
        body={
            "source": source,
            "original_path": original_path or source,
            "skills": skills,
            "experience": experience,
            "education": education,
            "certifications": certifications,
            "chunks": [{"text": t, "embedding": v} for t, v in zip(texts, vectors)],
        },
    )


# Common filler words to exclude from BM25 query
_STOP_WORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with",
    "by", "from", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
    "do", "does", "did", "will", "would", "could", "should", "may", "might", "can",
    "this", "that", "these", "those", "it", "its", "we", "our", "you", "your", "they",
    "their", "them", "he", "she", "his", "her", "who", "which", "what", "where", "when",
    "how", "not", "no", "if", "as", "about", "into", "through", "during", "before",
    "after", "above", "below", "between", "such", "each", "every", "all", "any", "both",
    "more", "most", "other", "some", "than", "too", "very", "just", "also", "over",
    "able", "across", "well", "including", "etc", "e.g", "i.e", "must", "shall",
}


def _extract_keywords(text: str, max_words: int = 80) -> str:
    """Extract meaningful keywords from text, removing stop words and short terms."""
    words = re.findall(r"[a-zA-Z0-9#+./-]{2,}", text)
    keywords = [w for w in words if w.lower() not in _STOP_WORDS]
    # Deduplicate while preserving order
    seen = set()
    unique = []
    for w in keywords:
        low = w.lower()
        if low not in seen:
            seen.add(low)
            unique.append(w)
    return " ".join(unique[:max_words])


def search_resumes(query: str, k: int = 5) -> list[dict]:
    """Hybrid search. Returns list of dicts with source, skills, experience, content."""
    ensure_search_pipeline()
    client = get_client()
    query_vector = get_embeddings().embed_query(query)

    # Extract keywords for BM25 to avoid maxClauseCount limit
    bm25_query = _extract_keywords(query)

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
                                "query": {"match": {"chunks.text": bm25_query}},
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
            "original_path": hit["_source"].get("original_path", ""),
            "skills": hit["_source"].get("skills", ""),
            "experience": hit["_source"].get("experience", ""),
            "education": hit["_source"].get("education", ""),
            "certifications": hit["_source"].get("certifications", ""),
            "content": "\n\n".join(c["text"] for c in hit["_source"]["chunks"]),
        }
        for hit in resp["hits"]["hits"]
    ]


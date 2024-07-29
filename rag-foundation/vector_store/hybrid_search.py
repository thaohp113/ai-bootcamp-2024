from typing import List
from collections import defaultdict

from .base import BaseVectorStore
from .node import TextNode, VectorStoreQueryResult
from .semantic_vector_store import SemanticVectorStore
from .sparse_vector_store import SparseVectorStore

class HybridSearch():
    
    def __init__(self, **data):
        super().__init__(**data)
        self.dense_vector_store = SparseVectorStore(
            persist=True,
            saved_file="data/sparse.csv",
            metadata_file="data/sparse_metadata.json",
            force_index=True
        )
        self.sparse_vector_store = SemanticVectorStore(
            persist=True,
            saved_file="data/dense.csv",
            force_index=True
        )
        self.vector_store = [self.dense_vector_store, self.sparse_vector_store]
        
    def get_vector_store(self):
        return self.vector_store

    # def add(self, nodes: List[TextNode]) -> List[str]:
    #     self.dense_vector_store.add(nodes)
    #     self.sparse_vector_store.add(nodes)
    
    def combine_search_results(
        retrieved_results: List[VectorStoreQueryResult],
        maximum_results: int = 3,
        fusion_constant: float = 60. ) -> VectorStoreQueryResult:
        # Stores the combined RRF scores for each document.
        document_scores = defaultdict(float)
        # Keeps track of each document object by its identifier.
        documents_registry = {}

        # Update scores and store documents using list comprehension for efficiency
        [(
            documents_registry.setdefault(document.id_, document), 
            document_scores.update({document.id_: document_scores[document.id_] + 1. / (fusion_constant + position + 1)})
        ) for single_result_set in retrieved_results for position, document in enumerate(single_result_set.nodes)]

        # Order documents by their scores from highest to lowest.
        ranked_documents = sorted(document_scores.items(), key=lambda item: item[1], reverse=True)
        top_ranked_documents = ranked_documents[:maximum_results]

        # Construct the final result object using list comprehensions.
        return VectorStoreQueryResult(
            nodes=[documents_registry[doc_id] for doc_id, _ in top_ranked_documents],
            similarities=[score for _, score in top_ranked_documents],
            ids=[doc_id for doc_id, _ in top_ranked_documents]
        )

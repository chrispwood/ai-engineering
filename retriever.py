from typing import Callable, Any, List, Dict
from pinecone_db import PineconeDB

class Retriever:
    """
    A class that handles retrieving relevant document chunks based on a text query.
    """

    def __init__(self, db: PineconeDB, embedder: Callable[[str], List[float]]):
        """
        Initialize the Retriever with a database instance and an embedder function.

        :param db: An instance of PineconeDB.
        :param embedder: A function that takes a string and returns a vector.
        """
        self.db = db
        self.embedder = embedder

    def retrieve(self, query: str, top_k: int = 5, filter: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Retrieve document chunks relevant to the query.

        :param query: The search query.
        :param top_k: The number of results to return.
        :param filter: Optional metadata filter for Pinecone query.
        :return: A list of result dictionaries containing metadata and scores.
        """
        # Embed the query
        query_vector = self.embedder(query)

        # Query the database
        results = self.db.query_by_vector(
            vector=query_vector,
            top_k=top_k,
            include_metadata=True,
            filter=filter
        )

        # Format and return the results
        formatted_results = []
        for match in results.matches:
            formatted_results.append({
                "id": match.id,
                "score": match.score,
                "metadata": match.metadata,
                "text": match.metadata.get("chunk_text", ""),
                "file_name": match.metadata.get("file_name", ""),
                "subject": match.metadata.get("subject", ""),
            })

        return formatted_results

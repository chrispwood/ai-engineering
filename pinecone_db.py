from pinecone import Pinecone, ServerlessSpec
import os
import time
from typing import Callable, Iterable


class PineconeDB:
    """
    A store for indexing and querying chunk records in Pinecone.
    """

    def __init__(self, api_key: str, index_name: str = "ai-rag", dimension: int = 1536):
        """
        Initialize the Pinecone client and ensure the index exists.

        :param api_key: The Pinecone API key.
        :param index_name: The name of the index to use.
        :param dimension: Vector dimension.
        """
        self.pc = Pinecone(api_key=api_key)
        self.index_name = index_name
        self.dimension = dimension

        self._ensure_index_exists()
        self.index = self.pc.Index(self.index_name)

    def _ensure_index_exists(self):
        """
        Checks if the index exists; if not, creates a Serverless index.
        """
        existing_indexes = [index.name for index in self.pc.list_indexes()]
        if self.index_name not in existing_indexes:
            print(f"Creating Pinecone index: {self.index_name}...")
            self.pc.create_index(
                name=self.index_name,
                dimension=self.dimension,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                )
            )
            while not self.pc.describe_index(self.index_name).status["ready"]:
                time.sleep(1)
            print(f"Index {self.index_name} is ready.")
        else:
            print(f"Using existing Pinecone index: {self.index_name}.")


    def index_records(
        self,
        records: Iterable,
        embedder: Callable[[list[str]], list[list[float]]],
        batch_size: int = 50,
        skip_if_exists: bool = False,
    ) -> int:
        """
        Embed and write chunk records to Pinecone in batches.

        :param records: An iterable of records to index.
        :param embedder: A function that takes a list of strings and returns a list of vectors.
        :param batch_size: Number of vectors to upsert in one batch.
        :param skip_if_exists: If True, skips documents that already have at least one chunk indexed.
        """
        records_to_process = []
        total_indexed = 0
        skipped_documents = set()

        for record in records:
            if skip_if_exists:
                if record.document_id in skipped_documents:
                    continue
                
                # Check Pinecone for any existing chunks from this document
                results = self.index.query(
                    vector=[0.0] * self.dimension,
                    top_k=1,
                    filter={"document_id": {"$eq": record.document_id}}
                )
                if results.matches:
                    print(f"Skipping document {record.document_id} because it's already indexed.")
                    skipped_documents.add(record.document_id)
                    continue

            records_to_process.append(record)

            if len(records_to_process) >= batch_size:
                total_indexed += self._upsert_batch(records_to_process, embedder)
                records_to_process = []

        if records_to_process:
            total_indexed += self._upsert_batch(records_to_process, embedder)

        print(f"Finished indexing {total_indexed} records.")
        return total_indexed

    def _upsert_batch(self, records, embedder: Callable[[list[str]], list[list[float]]]) -> int:
        """
        Helper method to embed and upsert a batch of records.
        """
        texts = [r.chunk_text for r in records]
        embeddings = embedder(texts)

        vectors = []
        for record, values in zip(records, embeddings):
            if len(values) != self.dimension:
                raise ValueError(
                    f"Embedding dimension mismatch: expected {self.dimension}, got {len(values)} "
                    f"for record {record.id!r}."
                )

            vectors.append({
                "id": record.id,
                "values": values,
                "metadata": {
                    "document_id": record.document_id,
                    "file_name": record.file_name,
                    "subject": record.subject,
                    "chunk_index": record.chunk_index,
                    "chunk_text": record.chunk_text,
                },
            })

        self.index.upsert(vectors=vectors)
        print(f"Indexed {len(vectors)} records...")
        return len(vectors)

    def query_by_vector(
        self,
        vector: list[float],
        top_k: int = 5,
        include_metadata: bool = True,
        filter: dict | None = None,
    ):
        """
        Query the Pinecone index with a vector.
        """
        return self.index.query(
            vector=vector,
            top_k=top_k,
            include_metadata=include_metadata,
            filter=filter
        )

    def delete_ids(self, ids: list[str]):
        """
        Delete vectors from the Pinecone index by ID.
        """
        return self.index.delete(ids=ids)

    def delete_document(self, document_id: str):
        """
        Delete all chunks belonging to a document.
        """
        return self.index.delete(filter={"document_id": {"$eq": document_id}})


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()

    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        print("PINECONE_API_KEY not found in environment. Skipping test.")

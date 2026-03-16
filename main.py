import os
import hashlib
from dotenv import load_dotenv

from data_loader import DataLoader
from pinecone_db import PineconeDB


def embed_text(text: str) -> list[float]:
    """
    Generates a deterministic dummy embedding for text.
    Returns a 1536-dimensional vector.
    """
    dimension = 1536
    # Use MD5 to get a stable, pseudo-random vector for the same text
    hash_object = hashlib.md5(text.encode())
    hash_hex = hash_object.hexdigest()
    seed = int(hash_hex, 16)
    vector = []
    for i in range(dimension):
        # Deterministic but diverse numbers
        val = ((seed >> (i % 128)) & 0xFF) / 255.0
        vector.append(val)
    return vector


def main():
    load_dotenv()

    # The issue description uses PDF_DIRECTORY (uppercase) in the recommended main.py
    # but my previous implementation used pdf_directory (lowercase).
    # I'll support both, defaulting to 'attachments'.
    pdf_directory = os.getenv("PDF_DIRECTORY") or os.getenv("pdf_directory") or "attachments"
    pinecone_api_key = os.getenv("PINECONE_API_KEY")

    if not pinecone_api_key:
        raise ValueError("PINECONE_API_KEY is not set.")

    print(f"Loading PDFs from: {pdf_directory}")

    loader = DataLoader(pdf_directory=pdf_directory, subject="ai")
    db = PineconeDB(api_key=pinecone_api_key)

    total_indexed = db.index_records(
        records=loader.iter_records(),
        embedder=embed_text,
        batch_size=50,
        skip_if_exists=False,
    )

    print(f"Total records indexed: {total_indexed}")


if __name__ == "__main__":
    main()

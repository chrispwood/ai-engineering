import os
import sys
import argparse
from dotenv import load_dotenv
from openai import OpenAI

from data_loader import DataLoader
from pinecone_db import PineconeDB
from retriever import Retriever


def get_openai_client():
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY is not set.")
    return OpenAI(api_key=openai_api_key)


def embed_texts(openai_client: OpenAI, texts: list[str]) -> list[list[float]]:
    """
    Generates embeddings for a list of texts using OpenAI's text-embedding-3-small.
    Returns a list of 1536-dimensional vectors.
    """
    cleaned = [t.replace("\n", " ") for t in texts]
    response = openai_client.embeddings.create(
        input=cleaned,
        model="text-embedding-3-small"
    )
    # Results are returned in order, but sorting by index is safer
    return [e.embedding for e in sorted(response.data, key=lambda e: e.index)]


def embed_text(openai_client: OpenAI, text: str) -> list[float]:
    """
    Generates an embedding for text using OpenAI's text-embedding-3-small.
    Returns a 1536-dimensional vector.
    """
    return embed_texts(openai_client, [text])[0]


def main():
    load_dotenv()

    parser = argparse.ArgumentParser(description="AI RAG CLI")
    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser("load",  help="Index PDFs from the PDF directory.")
    subparsers.add_parser("wipe",  help="Delete all vectors from the Pinecone index.")
    subparsers.add_parser("query", help="Enter the interactive query REPL.")
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    # Defer API setup until we know a subcommand was given
    openai_client  = get_openai_client()
    embedder       = lambda text:  embed_text(openai_client, text)
    batch_embedder = lambda texts: embed_texts(openai_client, texts)

    pdf_directory    = os.getenv("PDF_DIRECTORY") or "attachments"
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    if not pinecone_api_key:
        raise ValueError("PINECONE_API_KEY is not set.")

    db        = PineconeDB(api_key=pinecone_api_key)
    retriever = Retriever(db=db, embedder=embedder)

    if args.command == "load":
        print(f"Loading PDFs from: {pdf_directory}")
        loader = DataLoader(pdf_directory=pdf_directory, subject="ai")
        total_indexed = db.index_records(
            records=loader.iter_records(),
            embedder=batch_embedder,
            batch_size=50,
            skip_if_exists=False,
        )
        print(f"Total records indexed: {total_indexed}")

    elif args.command == "wipe":
        print(f"WARNING: This will permanently delete all vectors from index '{db.index_name}'.")
        confirm = input("Type 'yes' to confirm: ").strip()
        if confirm != "yes":
            print("Wipe cancelled. Exiting.")
            return
        db.wipe()
        print("Index wiped.")

    elif args.command == "query":
        print("\n--- AI Retrieval Loop (Type 'exit' to quit) ---")
        while True:
            query = input("\nEnter your query: ").strip()
            if query.lower() in ("exit", "quit", "q"):
                print("Exiting retrieval loop.")
                break
            if not query:
                continue
            results = retriever.retrieve(query=query, top_k=10)
            if not results:
                print("No relevant results found.")
                continue
            print(f"Top {len(results)} results for query: {query!r}")
            for i, res in enumerate(results, 1):
                print(f"\nResult {i}:")
                print(f"Score: {res['score']:.4f}")
                print(f"File: {res['file_name']}")
                print(f"Text: {res['text'][:200]}...")


if __name__ == "__main__":
    main()

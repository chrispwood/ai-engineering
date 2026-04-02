# ai-engineering
Exemplary ai systems project

## Setup
1. Install dependencies:
   ```bash
   uv sync
   ```
2. Configure environment variables in a `.env` file:
   ```env
   PINECONE_API_KEY=your_pinecone_api_key
   OPENAI_API_KEY=your_openai_api_key
   PDF_DIRECTORY=attachments
   ```

## Usage

```bash
uv run main.py load    # Index PDFs from PDF_DIRECTORY into Pinecone
uv run main.py query   # Start interactive query REPL
uv run main.py wipe    # Delete all vectors from the Pinecone index
```



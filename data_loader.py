from dataclasses import dataclass
from pathlib import Path
from typing import Generator, Iterable

from pdf_loader import PDFLoader


@dataclass(frozen=True)
class ChunkRecord:
    id: str
    document_id: str
    file_name: str
    subject: str
    chunk_index: int
    chunk_text: str


class DataLoader:
    """
    Converts PDF chunks into structured chunk records.
    """

    def __init__(
        self,
        pdf_directory: str,
        chunk_size: int = 1000,
        chunk_overlap: int = 100,
        subject: str = "ai",
    ):
        self.pdf_directory = pdf_directory
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.subject = subject

    @staticmethod
    def _build_document_id(source_path: str) -> str:
        return Path(source_path).name

    def iter_records(self) -> Generator[ChunkRecord, None, None]:
        loader = PDFLoader(
            self.pdf_directory,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )

        for chunk in loader:
            source = chunk["source"]
            file_name = Path(source).name
            document_id = self._build_document_id(source)
            chunk_index = chunk["chunk_index"]
            chunk_text = chunk["content"].strip()

            if not chunk_text:
                continue

            record_id = f"{document_id}#chunk-{chunk_index}"

            yield ChunkRecord(
                id=record_id,
                document_id=document_id,
                file_name=file_name,
                subject=self.subject,
                chunk_index=chunk_index,
                chunk_text=chunk_text,
            )

    def load_records(self) -> list[ChunkRecord]:
        return list(self.iter_records())

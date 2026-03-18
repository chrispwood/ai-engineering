from typing import Generator
from pdf_loader import PDFLoader, ChunkRecord


class DataLoader:
    """
    Converts PDF chunks into structured chunk records.
    """

    def __init__(
        self,
        pdf_directory: str,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        subject: str = "ai",
    ):
        self.pdf_directory = pdf_directory
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.subject = subject

    def iter_records(self) -> Generator[ChunkRecord, None, None]:
        loader = PDFLoader(
            self.pdf_directory,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            subject=self.subject,
        )

        for record in loader:
            if not record.chunk_text.strip():
                continue
            yield record

    def load_records(self) -> list[ChunkRecord]:
        return list(self.iter_records())

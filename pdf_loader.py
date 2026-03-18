import os
from typing import Generator, Iterator
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dataclasses import dataclass


@dataclass
class ChunkRecord:
    id: str
    document_id: str
    file_name: str
    subject: str
    chunk_index: int
    chunk_text: str


class PDFLoader:
    """
    Loads PDF documents from a directory and yields ChunkRecords using
    LangChain's PyPDFLoader and RecursiveCharacterTextSplitter.

    PyPDFLoader is preferred over using pypdf directly because it integrates with
    LangChain's Document format, which carries metadata (page number, source path)
    alongside each chunk. This makes debugging and citation tracking easier, and
    keeps the pipeline compatible with other LangChain tools like text splitters
    and vector store connectors without extra conversion steps.

    This also supports splitting along natural boundaries rather than mid-word.
    """

    def __init__(
        self,
        directory_path: str,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        subject: str = "general",
    ):
        self.directory_path = directory_path
        self.subject = subject
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],  # semantic-first splitting
        )

    def _get_pdf_files(self) -> list[str]:
        pdf_files = []
        for root, _, files in os.walk(self.directory_path):
            for file in files:
                if file.lower().endswith(".pdf"):
                    pdf_files.append(os.path.join(root, file))
        return pdf_files

    def __iter__(self) -> Iterator[ChunkRecord]:
        yield from self.load_chunks()

    def load_chunks(self) -> Generator[ChunkRecord, None, None]:
        for pdf_path in self._get_pdf_files():
            try:
                file_name = os.path.basename(pdf_path)
                document_id = os.path.splitext(file_name)[0]

                # LangChain loader — handles text extraction per page
                loader = PyPDFLoader(pdf_path)
                pages = loader.load()  # list of LangChain Document objects

                # Split into semantic chunks
                chunks = self.splitter.split_documents(pages)

                for chunk_index, chunk in enumerate(chunks):
                    yield ChunkRecord(
                        id=f"{document_id}_{chunk_index}",
                        document_id=document_id,
                        file_name=file_name,
                        subject=self.subject,
                        chunk_index=chunk_index,
                        chunk_text=chunk.page_content,
                    )

            except Exception as e:
                print(f"Error processing {pdf_path}: {e}")

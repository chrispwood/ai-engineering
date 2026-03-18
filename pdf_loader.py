import os
from typing import Generator, Iterator
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
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
                print(f"Loading: {file_name}...")
                loader = PyPDFLoader(pdf_path)

                chunk_index = 0

                # Sliding window: accumulate up to 2 pages at a time so the splitter
                # can create chunks that span page boundaries, while never holding
                # the entire PDF in memory.
                page_buffer: list[str] = []

                # Text carried forward from the previous window flush.  It is the
                # last (potentially incomplete) chunk that was held back because it
                # might be extended by text on the next page.
                carry_over: str = ""

                for page in loader.lazy_load():
                    page_buffer.append(page.page_content)

                    # Wait until we have a 2-page window before splitting.
                    if len(page_buffer) < 2:
                        continue

                    # Prepend carry_over so the splitter sees text that crosses the
                    # previous window boundary, enabling true cross-page chunks.
                    sep = "\n\n" if carry_over else ""
                    window_text = carry_over + sep + "\n\n".join(page_buffer)
                    chunks = self.splitter.split_text(window_text)

                    # Yield every chunk except the last: the final chunk may be a
                    # partial semantic unit that continues on the next page.
                    for chunk_text in chunks[:-1]:
                        yield ChunkRecord(
                            id=f"{document_id}_{chunk_index}",
                            document_id=document_id,
                            file_name=file_name,
                            subject=self.subject,
                            chunk_index=chunk_index,
                            chunk_text=chunk_text,
                        )
                        chunk_index += 1

                    # Hold the last chunk as the bridge into the next window.
                    carry_over = chunks[-1] if chunks else carry_over
                    page_buffer.clear()

                # Final flush: combine any buffered pages with the carry_over and
                # yield all remaining chunks (nothing more to extend them).
                if page_buffer or carry_over:
                    sep = "\n\n" if carry_over and page_buffer else ""
                    tail = carry_over + sep + "\n\n".join(page_buffer)
                    for chunk_text in self.splitter.split_text(tail):
                        yield ChunkRecord(
                            id=f"{document_id}_{chunk_index}",
                            document_id=document_id,
                            file_name=file_name,
                            subject=self.subject,
                            chunk_index=chunk_index,
                            chunk_text=chunk_text,
                        )
                        chunk_index += 1

            except Exception as e:
                print(f"Error processing {pdf_path}: {e}")

import os
from typing import Generator, List, Dict
from pypdf import PdfReader

class PDFLoader:
    """
    A class to load PDF documents from a directory and provide chunks using a generator pattern.
    """
    def __init__(self, directory_path: str, chunk_size: int = 1000, chunk_overlap: int = 100):
        """
        Initialize the PDFLoader.

        :param directory_path: The directory containing PDF documents.
        :param chunk_size: The number of characters in each chunk.
        :param chunk_overlap: The number of overlapping characters between chunks.
        """
        self.directory_path = directory_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def _get_pdf_files(self) -> List[str]:
        """
        Walk the directory and find all PDF files.
        """
        pdf_files = []
        for root, _, files in os.walk(self.directory_path):
            for file in files:
                if file.lower().endswith(".pdf"):
                    pdf_files.append(os.path.join(root, file))
        return pdf_files


    def __iter__(self) -> Generator[Dict[str, str], None, None]:
        """
        Allows the PDFLoader to be used as an iterator directly.
        """
        yield from self.load_chunks()

    def load_chunks(self) -> Generator[Dict[str, str], None, None]:
        """
        Generator that yields chunks of text from PDF files in the directory.
        Yields a dictionary with metadata (source file, chunk index) and text.
        """
        pdf_files = self._get_pdf_files()
        
        for pdf_path in pdf_files:
            try:
                reader = PdfReader(pdf_path)
                buffer = ""
                chunk_index = 0
                
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        buffer += page_text + "\n"
                    
                    # Yield chunks as they become available in the buffer
                    while len(buffer) >= self.chunk_size:
                        chunk_content = buffer[:self.chunk_size]
                        yield {
                            "source": pdf_path,
                            "chunk_index": chunk_index,
                            "content": chunk_content
                        }
                        chunk_index += 1
                        # Keep the overlap in the buffer for the next chunk
                        buffer = buffer[self.chunk_size - self.chunk_overlap:]
                
                # After all pages, yield any remaining text in the buffer as the final chunk
                if buffer:
                    yield {
                        "source": pdf_path,
                        "chunk_index": chunk_index,
                        "content": buffer
                    }
                
            except Exception as e:
                print(f"Error processing {pdf_path}: {e}")

if __name__ == "__main__":
    # Example usage:
    loader = PDFLoader("path/to/pdfs")
    for chunk in loader.load_chunks():
        print(chunk)
    pass

from pathlib import Path
from pypdf import PdfReader

def readFile(file_path: Path) -> str:
    if file_path.suffix == ".md":
        return file_path.read_text(encoding="utf-8")
    else:
        reader = PdfReader(file_path)
        text = ""

        for page in reader.pages:
            text += page.extract_text()

        return text

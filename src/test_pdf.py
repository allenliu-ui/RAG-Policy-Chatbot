"""
Test 2: PDF parsing
- Reads a PDF from the data/ folder
- Prints the first 500 characters of extracted text
Run: python src/test_pdf.py
Drop any PDF into the data/ folder first and update PDF_PATH below.
"""

import fitz  # PyMuPDF

PDF_PATH = "data/sample.pdf"  # Update this to your PDF filename

def extract_text(pdf_path: str) -> str:
    doc = fitz.open(pdf_path)
    full_text = ""
    for page_num, page in enumerate(doc):
        text = page.get_text()
        full_text += f"\n--- Page {page_num + 1} ---\n{text}"
    return full_text

print(f"Parsing PDF: {PDF_PATH}")
text = extract_text(PDF_PATH)
print(f"Total characters extracted: {len(text)}")
print("\nFirst 500 characters:")
print(text[:500])
print("\nPDF parsing test passed.")

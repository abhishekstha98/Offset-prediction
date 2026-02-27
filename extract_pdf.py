from pypdf import PdfReader
import re

def analyze_pdf(filename):
    reader = PdfReader(filename)
    print(f"Total Pages: {len(reader.pages)}")
    for i, page in enumerate(reader.pages):
        print(f"--- Page {i+1} ---")
        text = page.extract_text()
        print(text[:500]) # First 500 chars of each page

if __name__ == "__main__":
    analyze_pdf("Published_Station_List_2019_Final1.pdf")





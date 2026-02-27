import os
from pypdf import PdfReader

pdfs = [
    r"d:\Offset Prediction Research\atmosphere-12-01462.pdf",
    r"d:\Offset Prediction Research\clim-JCLI-D-20-0966.1.pdf",
    r"d:\Offset Prediction Research\f8fdb67cf9c93affe1044566cd69f7331478.pdf",
    r"d:\Offset Prediction Research\fenvs-10-921659.pdf",
    r"d:\Offset Prediction Research\s10712-024-09827-x.pdf"
]

def extract_summary(pdf_path):
    try:
        reader = PdfReader(pdf_path)
        num_pages = len(reader.pages)
        
        print(f"\n{'='*50}\nAnalyzing: {os.path.basename(pdf_path)} (Pages: {num_pages})")
        
        # Extract first 2 pages (Title, Abstract, Introduction)
        first_pages = ""
        for i in range(min(2, num_pages)):
            first_pages += reader.pages[i].extract_text() + "\n"
        
        # Extract last 2 pages (Conclusion, Discussion)
        last_pages = ""
        for i in range(max(0, num_pages-3), num_pages):
            last_pages += reader.pages[i].extract_text() + "\n"
            
        print("--- ABSTRACT / INTRO (Snippets) ---")
        # Print a snippet to keep output manageable, or write to file
        with open(f"{os.path.basename(pdf_path)}.txt", 'w', encoding='utf-8') as f:
            f.write("--- FIRST PAGES ---\n")
            f.write(first_pages)
            f.write("\n--- LAST PAGES ---\n")
            f.write(last_pages)
        print(f"Saved text to {os.path.basename(pdf_path)}.txt")
            
    except Exception as e:
        print(f"Error processing {pdf_path}: {e}")

if __name__ == "__main__":
    for pdf in pdfs:
        extract_summary(pdf)

import fitz  # PyMuPDF

print(fitz.__version__)
pdf = fitz.open("static/8d01d877-45c7-4b8d-8fd1-a0aea6be69db.pdf")
print(f"Number of pages: {pdf.page_count}")
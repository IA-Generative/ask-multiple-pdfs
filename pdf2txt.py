import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
#        pdf_reader = PdfReader(pdf)
        with open(pdf, 'rb') as f:
           pdf_reader = PdfReader(f)

        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


# Function to read multiple PDF files and return them as a list
def read_pdf_files_from_directory(directory):
    pdf_docs = []
    for file in os.listdir(directory):
        if file.endswith('.pdf'):
            file_path = os.path.join(directory, file)
            with open(file_path, 'rb') as f:
                pdf_docs.append(f.read())
    return pdf_docs


load_dotenv()

# Directory where your PDFs are stored on the server
pdf_directory = '/home/ailab/pdf-demo/'

# List PDF files in the directory
pdf_files = [f for f in os.listdir(pdf_directory) if f.endswith('.pdf')]

raw_text = ""

# Process each PDF file
for pdf_file in pdf_files:
  pdf_path = os.path.join(pdf_directory, pdf_file)

  # Open and process the PDF file
  with open(pdf_path, 'rb') as pdf_doc:
    # Process the PDF files
    # Get PDF text
    pdf_reader = PdfReader(pdf_doc)

    for page in pdf_reader.pages:
      raw_text += page.extract_text()

print(raw_text)

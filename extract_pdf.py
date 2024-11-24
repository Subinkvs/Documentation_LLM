import pdfplumber
from transformers import pipeline

pdf_path = "C:/Users/USER/Downloads/google_terms_of_service_en_in.pdf"

output_text_file = "extracted_text.txt"

with pdfplumber.open(pdf_path) as pdf:
    extracted_text = ""
    for page in pdf.pages:
        extracted_text += page.extract_text()
        
with open(output_text_file, "w") as text_file:
    text_file.write(extracted_text)
    
# print(f"Text extracted and saved to {output_text_file}")

# Open and read the extracted text file
with open(output_text_file, "r") as text_file:
    extracted_text = text_file.read()

# Print the content of the text file
# print(extracted_text[:500])

summarizer = pipeline("summarization", model="t5-small")

summary = summarizer(extracted_text[:1000], max_length=150, min_length=30, do_sample=False)
print("Summary:", summary[0]['summary_text'])
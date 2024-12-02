import pdfplumber
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
from nltk.tokenize import sent_tokenize
from transformers import pipeline

pdf_path = "C:/Users/USER/Downloads/google_terms_of_service_en_in.pdf"

output_text_file = "extracted_text.txt"

# Load the question generation pipeline
qg_pipeline = pipeline("text2text-generation", model="valhalla/t5-base-qg-hl")

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

sentences = sent_tokenize(extracted_text)

passages = []
current_passage = ""

for sentence in sentences:
    if len(current_passage.split()) + len(sentence.split()) < 200:
        current_passage += " " + sentence
    else:
        passages.append(current_passage.strip())
        current_passage = sentence
        
if current_passage:
    passages.append(current_passage.strip())
    
    
# function to generate questions using the pipeline
def generate_questions_pipeline(passage, min_questions=3):
    input_text = f"generate questions: {passage}"
    results = qg_pipeline(input_text)
    questions = results[0]['generated_text'].split('<sep>')
    
    # ensure we have atleat 3 questions
    questions = [q.strip() for q in questions if q.strip()]
    
    # if fewer than 3 questions, try to regenerate from smaller parts of the passage
    if len(questions) < min_questions:
        passage_sentences = passage.split('. ')
        for i in range(len(passage_sentences)):
            if len(questions) >= min_questions:
                break
            
            additional_input = ' '.join(passage_sentences[i:i+2])
            additional_results = qg_pipeline(f"generate questions: {additional_input}")
            additional_questions = additional_results[0]['generated_text'].split('<sep>')
            questions.extend([q.strip() for q in additional_questions if q.strip()])
    
    return questions[:min_questions] # return only top 3 questions


# generate questions from passages
for idx, passage in enumerate(passages):
    questions = generate_questions_pipeline(passage)
    print(f"Passage {idx+1}:\n{passage}\n")
    print(f"Generated Questions: ")
    for q in questions:
        print(f"- {q}")
    print(f"\n{'-'*50}\n")
        
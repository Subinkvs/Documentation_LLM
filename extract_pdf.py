"""
Script for processing a PDF document, extracting text, summarizing content,
and generating questions using NLP models.
"""

import pdfplumber
import nltk
from nltk.tokenize import sent_tokenize
from transformers import pipeline

# Ensure NLTK dependencies are downloaded
nltk.download('punkt')

# File paths
PDF_PATH = "C:/Users/USER/Downloads/google_terms_of_service_en_in.pdf"
OUTPUT_TEXT_FILE = "extracted_text.txt"

# Load NLP pipelines
qg_pipeline = pipeline("text2text-generation", model="valhalla/t5-base-qg-hl")  # Question generation pipeline
summarizer = pipeline("summarization", model="t5-small")  # Summarization pipeline

qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2") # Load the QA pipeline

def extract_text_from_pdf(pdf_path):
    """
    Extract text from a PDF file using pdfplumber.

    Args:
        pdf_path (str): Path to the PDF file.

    Returns:
        str: Extracted text from the PDF.
    """
    with pdfplumber.open(pdf_path) as pdf:
        extracted_text = "".join([page.extract_text() for page in pdf.pages])
    return extracted_text


def save_text_to_file(text, file_path):
    """
    Save text to a specified file.

    Args:
        text (str): The text content to save.
        file_path (str): Path to the output file.
    """
    with open(file_path, "w") as text_file:
        text_file.write(text)


def summarize_text(text, max_length=150, min_length=30):
    """
    Summarize a given text using the summarization pipeline.

    Args:
        text (str): The text to summarize.
        max_length (int): Maximum length of the summary.
        min_length (int): Minimum length of the summary.

    Returns:
        str: The summarized text.
    """
    summary = summarizer(text[:1000], max_length=max_length, min_length=min_length, do_sample=False)
    return summary[0]['summary_text']


def split_text_into_passages(text, max_words=200):
    """
    Split text into smaller passages, each containing a maximum number of words.

    Args:
        text (str): The input text.
        max_words (int): Maximum number of words per passage.

    Returns:
        list: List of text passages.
    """
    sentences = sent_tokenize(text)
    passages, current_passage = [], ""

    for sentence in sentences:
        if len(current_passage.split()) + len(sentence.split()) < max_words:
            current_passage += " " + sentence
        else:
            passages.append(current_passage.strip())
            current_passage = sentence

    if current_passage:
        passages.append(current_passage.strip())

    return passages


def generate_questions_pipeline(passage, min_questions=3):
    """
    Generate questions from a given passage using the question generation pipeline.

    Args:
        passage (str): The input passage.
        min_questions (int): Minimum number of questions to generate.

    Returns:
        list: List of generated questions.
    """
    input_text = f"generate questions: {passage}"
    results = qg_pipeline(input_text)
    questions = results[0]['generated_text'].split('<sep>')
    questions = [q.strip() for q in questions if q.strip()]

    if len(questions) < min_questions:
        passage_sentences = passage.split('. ')
        for i in range(len(passage_sentences)):
            if len(questions) >= min_questions:
                break
            additional_input = ' '.join(passage_sentences[i:i+2])
            additional_results = qg_pipeline(f"generate questions: {additional_input}")
            additional_questions = additional_results[0]['generated_text'].split('<sep>')
            questions.extend([q.strip() for q in additional_questions if q.strip()])

    return questions[:min_questions]


# Main execution
if __name__ == "__main__":
    # Step 1: Extract text from the PDF
    extracted_text = extract_text_from_pdf(PDF_PATH)

    # Step 2: Save extracted text to a file
    save_text_to_file(extracted_text, OUTPUT_TEXT_FILE)

    # Step 3: Summarize the extracted text
    summary = summarize_text(extracted_text)
    print("Summary:", summary)

    # Step 4: Split the text into passages
    passages = split_text_into_passages(extracted_text)

    # Step 5: Generate questions for each passage
    for idx, passage in enumerate(passages):
        questions = generate_questions_pipeline(passage)
        print(f"Passage {idx + 1}:\n{passage}\n")
        print("Generated Questions:")
        for q in questions:
            print(f"- {q}")
        print(f"\n{'-' * 50}\n")
        
# function to track and answer only unique questions    
def answer_unique_questions(passages, qa_pipeline):
    answered_questions = set() # to store unique questions
    
    for idx, passage in enumerate(passages):
        questions = generate_questions_pipeline(passage) 
    
    for question in questions:
        if question not in answered_questions: # check if the question has already been answered
            answer =qa_pipeline({'question': question, 'context': passage})
            print(f"Q: {question}")
            print(f"A: {answer['answer']}\n")
            answered_questions.add(question)  # add the question to the set to avoid repetition
    print(f"{'='*50}\n")
    
answer_unique_questions(passages, qa_pipeline)
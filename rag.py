import faiss
import numpy as np
from transformers import DPRContextEncoder, DPRQuestionEncoder, DPRContextEncoderTokenizer, DPRQuestionEncoderTokenizer



# Load the encoders
context_encoder = DPRContextEncoder.from_pretrained('facebook/dpr-ctx_encoder-single-nq-base')
question_encoder = DPRQuestionEncoder.from_pretrained('facebook/dpr-question_encoder-single-nq-base')

# Tokenizers for both encoders
context_tokenizer = DPRContextEncoderTokenizer.from_pretrained('facebook/dpr-ctx_encoder-single-nq-base')
question_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained('facebook/dpr-question_encoder-single-nq-base')

# Example documents (your corpus)
documents = ["The cat sits on the mat.", "The dog runs fast.", "AI is the future of technology."]

# Encode documents
context_embeddings = []
for doc in documents:
    inputs = context_tokenizer(doc, return_tensors="pt", truncation=True, padding=True)
    embeddings = context_encoder(**inputs).pooler_output.detach().numpy()
    context_embeddings.append(embeddings)

context_embeddings = np.vstack(context_embeddings)

# Create the FAISS index
index = faiss.IndexFlatL2(context_embeddings.shape[1])  # L2 distance for similarity
index.add(context_embeddings)  # Add document embeddings to the index

from transformers import BartForConditionalGeneration, BartTokenizer

# Load the model and tokenizer
generator = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
generator_tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")

def retrieve_and_generate(query, index, context_encoder, question_encoder, generator, context_tokenizer, question_tokenizer, generator_tokenizer):
    # Encode the query
    inputs = question_tokenizer(query, return_tensors="pt", truncation=True, padding=True)
    query_embedding = question_encoder(**inputs).pooler_output.detach().numpy()
    
    # Retrieve relevant documents from the FAISS index
    _, indices = index.search(query_embedding, k=2)  # Retrieve top 2 documents
    retrieved_docs = [documents[i] for i in indices[0]]
    
    # Combine query and retrieved docs to pass into the generator
    input_text = query + " " + " ".join(retrieved_docs)
    
    # Generate an answer using BART
    inputs = generator_tokenizer(input_text, return_tensors="pt", truncation=True, padding=True)
    summary_ids = generator.generate(inputs['input_ids'], max_length=150, num_beams=4, early_stopping=True)
    
    # Decode and return the generated text
    return generator_tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# Test the pipeline
query = "Tell me about AI technology."
response = retrieve_and_generate(query, index, context_encoder, question_encoder, generator, context_tokenizer, question_tokenizer, generator_tokenizer)
print(response)
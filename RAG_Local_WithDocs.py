from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline
import torch
import numpy as np
from PyPDF2 import PdfReader
import time
import re
import nltk
nltk.download('punkt_tab')
from nltk.tokenize import sent_tokenize
import os
import json


# implements RAG system 
# TinyLLama-1.1B-Chat for answer generation
# all-MiniLM-L6-v2 for doc retrieval


#  STORING + RETRIEVING RELEVANT DOCS BASED ON USER QUERY
class VectorStore:
    def __init__(self, cache_path="vector_cache"):
        self.documents = [] # list of raw text docs
        self.embeddings = [] # vector representations (embeddings) of docs
        self.cache_path = cache_path
        
        print("> Loading embedding model...")
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2') # loads a pre-trained model from sentence-transformers to generate embeddings
        print("> Embedding model loaded!")

        self.load_cache()

    
    # adds doc from store, compute embedding
    def add_document(self, text):
        self.documents.append(text)
        embedding = self.embedding_model.encode(text)
        self.embeddings.append(embedding)
    
    # encodes query, compares to stored embeddings using cosine sim, returns top-k most similar docs
    def search(self, query, top_k=3):
        query_embedding = self.embedding_model.encode(query) # encode query
        similarities = cosine_similarity([query_embedding], self.embeddings)[0] # compares query to embeddings using cosine similarity
        top_indices = np.argsort(similarities)[-top_k:][::-1] # finds top-3 most similar docs

        return [(self.documents[i], similarities[i]) for i in top_indices]
    
    def save_cache(self):
        print("> Saving vector cache...")
        os.makedirs(self.cache_path, exist_ok=True)

        with open(os.path.join(self.cache_path, "docs.json"), "w", encoding="utf-8") as f:
            json.dump(self.documents, f)

        # Convert embeddings (numpy arrays) to lists before saving
        embeddings_as_lists = [embedding.tolist() for embedding in self.embeddings]
        with open(os.path.join(self.cache_path, "embeddings.json"), "w", encoding="utf-8") as f:
            json.dump(embeddings_as_lists, f)

        print("> Vector cache saved.")

    def load_cache(self):
        try:
            with open(os.path.join(self.cache_path, "docs.json"), "r", encoding="utf-8") as f:
                self.documents = json.load(f)

            with open(os.path.join(self.cache_path, "embeddings.json"), "r", encoding="utf-8") as f:
                embeddings_lists = json.load(f)
                self.embeddings = [np.array(emb) for emb in embeddings_lists]

            print(f"> Loaded {len(self.documents)} cached vectors.")
        except FileNotFoundError:
            print("> No cached vectors found, starting fresh.")

class RAGSystem:
    def __init__(self, model_name):
        self.vector_store = VectorStore() # loads embedding model 
        print("> Loading language model...")
        
        # Initialize the pipeline - convenience function from Transformers Library
        self.pipe = pipeline(
            "text-generation",
            model=model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        print("> Language model loaded successfully!")

    def load_docs(self, pdf_path="./Docs/France.pdf"):
        """Load documents from a PDF file and chunk them."""
        print(f"> Loading documents from {pdf_path}...")
        
        # Extract text from PDF
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n\n"

        # Use sentence chunking
        chunks = self.chunk_by_sentences(text, window_size=3)

        # Filter out very short chunks (likely headers or page numbers)
        chunks = [chunk for chunk in chunks if len(chunk.split()) > 10]
        
        print(f"> Created {len(chunks)} chunks from the document")
        return chunks
    
    
    def chunk_by_sentences(self, text, window_size=3):
        sentences = sent_tokenize(text)
        chunks = []
        for i in range(0, len(sentences), window_size):
            chunk = " ".join(sentences[i:i+window_size])
            if len(chunk.split()) > 10:  # skip tiny chunks
                chunks.append(chunk)
        return chunks
    
    # indexing documents - converting each doc into an vector (embedding), storing it for retrieval
    def index_documents(self, documents):
        for doc in documents:
            self.vector_store.add_document(doc)
        print(f"> Indexed {len(documents)} documents")
        self.vector_store.save_cache() # Save cache after indexing
    
    def generate_response(self, query, max_new_tokens=150):

        try:
            print("\n> Retrieving Relevant Documents...")
            # Retrieve relevant documents 
            relevant_docs = self.vector_store.search(query)

            print(f"\nTop matches for '{query}':")
            for i, (doc, score) in enumerate(relevant_docs):
                print(f"{i+1}. Score: {score:.2f}\n{doc[:200]}...\n")
            
            # Limit context to avoid exceeding model's max length
            context = ""
            context_sources = []  # Store the source chunks and their similarity scores
            for doc, score in relevant_docs:
                if score > 0.6:
                    if len(context) + len(doc) < 1000:  # Keep context manageable
                        context += doc + "\n\n"
                        context_sources.append((doc, score))  # Store the chunk and its score
                else:
                    break     

            # Prompt structure
            prompt = (
                "Answer the question based on the context below. "
                "Keep the answer concise and factual.\n\n"
                f"Context:\n{context}\n\n"
                f"Question: {query}\n"
                "Answer:"
            )
            
            # Conservative parameters
            outputs = self.pipe(
                prompt,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.5,  # Low temperature for more factual answers
                top_k=30,
                top_p=0.9,
                repetition_penalty=1.1,
                truncation=True  # Can't exceed max length
            )
            
            full_response = outputs[0]["generated_text"]
            response = full_response[len(prompt):].strip()
            
            # Post-process to clean up any artifacts
            response = response.split("\n")[0]  # Take first line if multiple
            response = response.replace("</s>", "").replace("<|endoftext|>", "")
            
            # Return the response along with the context sources
            cleaned_sources = []
            for doc, score in context_sources:
                # Clean text and ensure complete sentences
                clean_text = re.sub(r'\s+', ' ', doc).strip()
                if not clean_text.endswith(('.', '!', '?')):
                    clean_text += '...'  # Mark incomplete sentences
                cleaned_sources.append((clean_text, round(float(score), 2)))
                                    
            return {
                "answer": response,
                "sources": cleaned_sources  # List of (chunk, similarity_score) tuples
            }
            
        except Exception as e:
            print(f"Error during generation: {e}")
            return "Sorry, I encountered an error while generating a response."

if __name__ == "__main__":
    print("\n> Initializing RAG system...")
    try:
        rag = RAGSystem(model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
        
        print("> Loading and indexing documents...")
        documents = rag.load_docs() # LOAD DOCS
        rag.index_documents(documents) # INDEX DOCS
        
        print("\n> Ready for questions. Press Enter with no input to quit.")
        while True:
            question = input("\nENTER YOUR QUESTION: ").strip()
            if not question:
                break
                
            raw_answer = rag.generate_response(question) # GENERATE RESPONSE
            answer = raw_answer["answer"]
            sources = raw_answer["sources"]
            print(f"\nANSWER: {answer}")
            print(f"\nSources:{sources}")
        
    except Exception as e:
        print(f"Failed to initialize RAG system: {e}")
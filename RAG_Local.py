from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline
import torch
import numpy as np

# implements RAG system 
# TinyLLama-1.1B-Chat for answer generation
# all-MiniLM-L6-v2 for doc retrieval


#  STORING + RETRIEVING RELEVANT DOCS BASED ON USER QUERY
class VectorStore:
    def __init__(self):
        self.documents = [] # list of raw text docs
        self.embeddings = [] # vector representations (embeddings) of docs
        
        print("Loading embedding model...")
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2') # loads a pre-trained model from sentence-transformers to generate embeddings
        print("Embedding model loaded!")
    
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

class RAGSystem:
    def __init__(self, model_name):
        self.vector_store = VectorStore() # loads embedding model 
        print("Loading language model...")
        
        # Initialize the pipeline - convenience function from Transformers Library
        self.pipe = pipeline(
            "text-generation",
            model=model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        print("Language model loaded successfully!")
    
    # indexing documents - converting each doc into an vector (embedding), storing it for retrieval
    def index_documents(self, documents):
        for doc in documents:
            self.vector_store.add_document(doc)
        print(f"Indexed {len(documents)} documents")
    
    def generate_response(self, query, max_new_tokens=150):
        print("\nGenerating response...")
        try:
            # Retrieve relevant documents 
            relevant_docs = self.vector_store.search(query)
            context = "\n".join([doc[0] for doc in relevant_docs])
            
            # Format the prompt using this chat template
            messages = [
                {"role": "system", "content": "Answer the question using the provided context."},
                {"role": "user", "content": f"Context: {context}\n\nQuestion: {query}"}
            ]
            
            # turns messages into prompt string for LLM understanding (like "|user| and |assistant|")
            prompt = self.pipe.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            # Generate response
            outputs = self.pipe(
                prompt,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_k=50,
                top_p=0.95
            )
            
            # Extract just the assistant's response
            full_response = outputs[0]["generated_text"]
            response = full_response.split("<|assistant|>")[-1].strip()
            return response
            
        except Exception as e:
            print(f"Error during generation: {e}")
            return "Sorry, I encountered an error while generating a response."

if __name__ == "__main__":
    print("Initializing RAG system...")
    try:
        rag = RAGSystem(model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
        
        # demo documents
        documents = [
            "The capital of France is Paris.",
            "The Eiffel Tower is located in Paris.",
            "France is known for its wine and cheese.",
            "The French Revolution took place from 1789 to 1799."
        ]

        # index documents (turn into vector + store)
        rag.index_documents(documents)
        
        question = '"' + str(input("Enter your question: ")) + '"'
        print(f"\nQuestion: {question}")
        answer = rag.generate_response(question)
        print(f"Answer: {answer}")
        
    except Exception as e:
        print(f"Failed to initialize RAG system: {e}")
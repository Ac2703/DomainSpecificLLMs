import requests  # For making HTTP requests to web pages and files
from bs4 import BeautifulSoup  # To parse HTML content easily
from urllib.parse import urljoin, urlparse  # To handle and manipulate URLs
import re  # Regular expressions (though not used here, imported anyway)
import urllib3  # To manage HTTP warnings
import csv  # To save data in CSV format
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)  # Disable insecure SSL warnings

from langchain_community.document_loaders import PyPDFLoader  # To load PDFs into documents
from langchain.text_splitter import RecursiveCharacterTextSplitter  # To split large texts into smaller chunks
from langchain_community.vectorstores import FAISS  # FAISS vector store for fast similarity search

from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline  # For embeddings and model pipelines

from langchain.chains import RetrievalQA  # Chain for question answering using retrieval
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM  # Huggingface tools for models and tokenizers
from langchain.schema import Document  # Standard Document object in Langchain

import os  # For file and path management


class WebScrape:
    def __init__(self):
        self.webpage_url = "https://tinman.cs.gsu.edu/~raj/past2.html"  # Starting URL for scraping
        self.filter_keyword = "syllabus"  # Keyword filter to focus on syllabus-related content
        response = requests.get(self.webpage_url, verify=False)  # Fetch the webpage ignoring SSL verification
        self.soup = BeautifulSoup(response.text, "html.parser")  # Parse the HTML content using BeautifulSoup
        self.all_docs = []  # List to hold all loaded documents (webpages + PDFs)
        self.vectorstore = None  # Placeholder for FAISS vector store (embedding index)
        self.qa_chain = None  # Placeholder for the question-answering chain

    def load_webpage_text(self, url):
        try:
            resp = requests.get(url, verify=False, timeout=10)  # Request the webpage with 10s timeout
            soup = BeautifulSoup(resp.text, "html.parser")  # Parse the webpage HTML

            # Get page title if exists, else fallback to "No Title"
            title_tag = soup.title.string.strip() if soup.title else "No Title"

            texts = soup.stripped_strings  # Extract all visible text parts, stripped of whitespace
            full_text = " ".join(texts)  # Join all text parts into a single string

            if len(full_text) < 50:  # Skip pages that are too short (likely no useful content)
                print(f"Skipped {url} due to short content")
                return []  # Return empty list to indicate no document

            # Create a Langchain Document with content and metadata about source and title
            doc = Document(
                page_content=full_text,
                metadata={
                    "source": url,
                    "title": title_tag
                }
            )
            print(f"Loaded webpage '{title_tag}' from {url} (length {len(full_text)})")
            return [doc]  # Return list containing the single Document
        
        except Exception as e:
            print(f"Failed to load webpage text from {url} - {e}")  # Print error if something goes wrong
            return []  # Return empty list on failure


    ### DOWNLOADS PDF FILE FROM PDF_URL AND SPLITS IT INTO DOCS ###
    def load_pdf(self, pdf_url):
        try:
            response = requests.get(pdf_url, verify=False, timeout=15)  # Download PDF ignoring SSL, 15s timeout
            pdf_path = "./temp.pdf"  # Temporary file path to save PDF locally

            with open(pdf_path, "wb") as f:
                f.write(response.content)  # Write the downloaded PDF content to file

            loader = PyPDFLoader(pdf_path)  # Load PDF with PyPDFLoader to split into pages
            docs = loader.load()  # Load returns a list of Document objects, one per page

            print(f"Loaded {len(docs)} pages from PDF {pdf_url}")

            # Add metadata about source URL and a fixed title "Syllabus PDF" to each page
            for d in docs:
                d.metadata["source"] = pdf_url
                d.metadata["title"] = "Syllabus PDF"
            return docs  # Return list of PDF page documents
        
        except Exception as e:
            print(f"Failed to load PDF {pdf_url} - {e}")  # Print error on failure
            return []  # Return empty list if error


    def ensure_dir(self, url: str) -> str:
        # Check if last segment of URL path looks like a directory (no dot/extension)
        path = urlparse(url).path
        if not url.endswith('/') and '.' not in path.split('/')[-1]:
            return url + '/'  # Add trailing slash to treat as directory URL
        return url  # Otherwise return URL as-is

    def crawl_and_load(self, url, depth=0, max_depth=1, logs=None, visited=None):
        if visited is None:
            visited = set()  # Initialize visited URL set on first call
        if url in visited:
            logs.append(f"Already visited {url}, skipping.")  # Avoid revisiting URLs to prevent loops
            return logs
        visited.add(url)  # Mark current URL as visited

        if logs is None:
            logs = []  # Initialize logs list on first call
        if depth > max_depth:
            return logs  # Stop crawling if max depth exceeded

        logs.append(f"Crawling URL: {url} (depth {depth})")  # Log current crawling action

        url_lower = url.lower()  # Normalize URL to lowercase for filtering

        # === Handle PDF URLs === #
        if url_lower.endswith(".pdf"):
            if (self.filter_keyword in url_lower or "syllabus" in url_lower):  # Filter PDFs by keyword
                try:
                    docs = self.load_pdf(url)  # Load and split PDF into documents
                    self.all_docs.extend(docs)  # Add PDF docs to overall list
                    logs.append(f"Loaded {len(docs)} pages from PDF {url}")
                except Exception as e:
                    logs.append(f"Failed to load PDF {url} - {e}")
            else:
                logs.append(f"Skipped PDF without filter keyword '{self.filter_keyword}' in URL: {url}")
            logs.append("-" * 40)  # Separator in logs
            return logs

        # === Handle HTML pages ===
        try:
            webpage_docs = self.load_webpage_text(url)  # Load webpage text as document(s)
            self.all_docs.extend(webpage_docs)  # Add docs to all_docs
            logs.append(f"Loaded webpage text from {url} (length {len(webpage_docs[0].page_content) if webpage_docs else 0})")
        except Exception as e:
            logs.append(f"Failed to load webpage text from {url} - {e}")

        # === Crawl internal links from this page ===
        try:
            resp = requests.get(url, verify=False, timeout=10)  # Request page again to find links
            soup = BeautifulSoup(resp.text, "html.parser")  # Parse HTML

            for link in soup.find_all("a", href=True):  # Find all anchor tags with href attribute
                base = self.ensure_dir(url)  # Get base URL, add trailing slash if needed
                next_url = urljoin(base, link["href"])  # Resolve relative URL to absolute

                next_url_lower = next_url.lower()  # Lowercase next URL
                next_domain = urlparse(next_url).netloc  # Extract domain part
                next_path = urlparse(next_url).path.lower()  # Extract path part lowercase

                is_html_page = next_path.endswith(".html")  # Check if link points to .html page

                # Only crawl internal links within the same domain and specific user path
                is_internal = (
                    next_domain == "tinman.cs.gsu.edu" and
                    "/~raj/" in urlparse(next_url).path
                )

                # Check if link is a syllabus PDF either by URL or link text
                is_syllabus_pdf = (
                    next_url_lower.endswith(".pdf") and
                    ("syllabus" in next_url_lower or "syllabus" in (link.get_text() or "").lower())
                )

                url_contains_keyword = self.filter_keyword in next_url_lower  # Keyword in URL
                link_text_contains_keyword = self.filter_keyword in (link.get_text() or "").lower()  # Keyword in anchor text

                # If internal HTML page and matches keyword filters and not exceeding max depth, crawl recursively
                if is_internal and is_html_page and depth < max_depth and (url_contains_keyword or link_text_contains_keyword):
                    self.crawl_and_load(next_url, depth=depth+1, max_depth=max_depth, logs=logs, visited=visited)
                # If syllabus PDF link, load PDF directly
                elif is_syllabus_pdf:
                    try:
                        docs = self.load_pdf(next_url)  # Load syllabus PDF
                        self.all_docs.extend(docs)  # Add PDF docs
                        logs.append(f"Loaded {len(docs)} pages from PDF {next_url}")
                    except Exception as e:
                        logs.append(f"Failed to load PDF {next_url} - {e}")
                    logs.append("-" * 20)  # Separator in logs
                else:
                    logs.append(f"Skipping external or non-syllabus/non-internal link: {next_url}")  # Skip unrelated links

            logs.append("-" * 40)  # Separator after crawling all links
        except Exception as e:
            logs.append(f"Error crawling links in {url}: {e}")  # Log any error during link crawling
            logs.append("-" * 40)

        return logs  # Return all logs collected

    def scrape_all_content(self):
        visited=set()  # Track visited URLs globally
        # Iterate through all links on the initial webpage
        for link in self.soup.find_all("a", href=True):
            full_url = urljoin(self.webpage_url, link["href"])  # Convert relative link to full URL
            print("\n" + "=" * 60)  # Visual separation
            print(f"Main link: {full_url}")  # Show which main link is being crawled
            print("=" * 60)
            logs = self.crawl_and_load(full_url, depth=0, max_depth=2, logs=[])  # Crawl starting from main link with depth limit 2
            for log in logs:
                print(f"  {log}")  # Print all logs for that crawl
            print("\n" + "=" * 60 + "\n")  # Visual separation between main links

    def build_faiss_index(self):
        if not self.all_docs:
            print("No docs found to index!")  # Alert if no documents to index
            return None

        # Split all documents into smaller chunks for embeddings
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=150, chunk_overlap=20)
        chunks = text_splitter.split_documents(self.all_docs)
        print(f"Split into {len(chunks)} chunks")

        embeddings = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2")  # Load embedding model
        vectorstore = FAISS.from_documents(chunks, embeddings)  # Build FAISS vector index from chunks
        vectorstore.save_local("faiss_index")  # Save index locally

        self.vectorstore = vectorstore  # Store index internally
        print("FAISS vector index built and saved.")

    def load_faiss_index(self):
        if not os.path.exists("faiss_index"):
            print("No FAISS index directory found!")
            return None
        embeddings = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2")  # Same embedding model as before
        self.vectorstore = FAISS.load_local("faiss_index", embeddings)  # Load FAISS index from disk
        print("FAISS vector index loaded.")

    def setup_qa_chain(self):
        if self.vectorstore is None:
            print("Vectorstore not initialized. Build or load it first.")
            return None

        # Load a T5-based model pipeline from Huggingface, set to run on GPU if available
        tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
        model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
        pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer, device=0)

        hf_pipeline = HuggingFacePipeline(pipeline=pipe)  # Wrap Huggingface pipeline for Langchain

        self.qa_chain = RetrievalQA.from_chain_type(
            llm=hf_pipeline,
            retriever=self.vectorstore.as_retriever(search_type="similarity"),
            return_source_documents=True,
            chain_type="stuff",
        )
        print("QA chain is ready.")

    def ask_question(self, question, course_code=None, term=None, top_k=20, use_k=3):
        """
        Run a filtered RAG query.
        - question: your natural‑language question
        - course_code: e.g. "4998"
        - term: e.g. "sp06" or "f06" (depending on how your URLs are structured)
        - top_k: how many raw hits to retrieve before filtering
        - use_k: how many filtered hits to actually feed into the LLM
        """
        # 1) Retrieve a broad candidate set
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": top_k})
        candidates = retriever.invoke(question)

        # 2) If metadata filters specified, apply them
        if course_code and term:
            filtered = [
                d for d in candidates
                if course_code in d.metadata.get("source", "")
                and term in d.metadata.get("source", "")
            ]
        else:
            filtered = candidates

        # 3) Fall back if no docs survived filtering
        if not filtered:
            print("⚠️  No docs matched your course_code/term filters—using unfiltered candidates.")
            filtered = candidates

        # 4) Truncate to the top `use_k` for the QA chain
        to_use = filtered[:use_k]

        # 5) Invoke the QA chain **directly passing** our filtered docs
        #    RetrievalQA supports passing `input_documents` into its LLM step
        result = self.qa_chain.invoke({
            "query": question,
            "input_documents": to_use
        })

        # 6) Package up for display
        return {
            "query": question,
            "answer": result["result"],
            "sources": [
                {
                    "title": d.metadata.get("title", "No Title"),
                    "url":   d.metadata.get("source", "No Source")
                }
                for d in to_use
            ]
        }



if __name__ == "__main__":
    scraper = WebScrape()  # Create scraper object

    print("Starting to scrape all content...")
    scraper.scrape_all_content()  # Crawl and load content

    print("Building FAISS index...")
    scraper.build_faiss_index()  # Build and save vector index

    print("Setting up QA chain...")
    scraper.setup_qa_chain()  # Prepare QA system

    # Example question to test
    question = input("Ask a question: ")
    answer = scraper.ask_question(
        question,
        course_code="4998",
        term="sp06",     # or "f06" if your Spring‑2006 URLs use “f06”
        top_k=20,
        use_k=3
    )

    # Print the answer
    print("\n=== ANSWER ===")
    print(answer["answer"])

    # Print the filtered sources
    print("\n=== SOURCES ===")
    for src in answer["sources"]:
        print(f"- Title: {src['title']}")
        print(f"  URL:   {src['url']}")

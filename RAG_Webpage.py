import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import re
import urllib3
import csv
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline

from langchain.chains import RetrievalQA
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from langchain.schema import Document

import os



class WebScrape:
    def __init__(self):
        self.webpage_url = "https://tinman.cs.gsu.edu/~raj/past2.html"
        self.filter_keyword = "syllabus"
        response = requests.get(self.webpage_url, verify=False)
        self.soup = BeautifulSoup(response.text, "html.parser")
        self.all_docs = []
        self.vectorstore = None
        self.qa_chain = None

    def load_webpage_text(self, url):
        try:
            resp = requests.get(url, verify=False, timeout=10)
            soup = BeautifulSoup(resp.text, "html.parser")

            # Get page title from <title> tag or fallback
            title_tag = soup.title.string.strip() if soup.title else "No Title"

            texts = soup.stripped_strings
            full_text = " ".join(texts)

            if len(full_text) < 50:  # Skip pages with too little text
                print(f"Skipped {url} due to short content")
                return []

            # Document Object from Langchain.Schema 
            doc = Document(
                page_content=full_text,
                metadata={
                    "source": url,
                    "title": title_tag
                }
            )
            print(f"Loaded webpage '{title_tag}' from {url} (length {len(full_text)})")
            return [doc] # RETURN LIST CONTAINING SINGLE DOC
        
        except Exception as e:
            print(f"Failed to load webpage text from {url} - {e}")
            return []


    ### DOWNLOADS PDF FILE FROM PDF_URL AND SPLITS IT INTO DOCS ###
    def load_pdf(self, pdf_url):
        try:
            response = requests.get(pdf_url, verify=False, timeout=15)
            pdf_path = "./temp.pdf" # Save PDF locally as Temp.PDF

            with open(pdf_path, "wb") as f:
                f.write(response.content)
            loader = PyPDFLoader(pdf_path)
            docs = loader.load() # PDF is split into Document pages

            print(f"Loaded {len(docs)} pages from PDF {pdf_url}")

            # Attach source URL and a title "Syllabus PDF" to metadata
            for d in docs:
                d.metadata["source"] = pdf_url
                d.metadata["title"] = "Syllabus PDF"
            return docs # Return PDF docs
        
        except Exception as e:
            print(f"Failed to load PDF {pdf_url} - {e}")
            return []


    def ensure_dir(self, url: str) -> str:
        # if last segment has no “.”, treat it as a directory
        path = urlparse(url).path
        if not url.endswith('/') and '.' not in path.split('/')[-1]:
            return url + '/'
        return url

    def crawl_and_load(self, url, depth=0, max_depth=1, logs=None, visited=None):
        if visited is None:
            visited = set()
        if url in visited:
            logs.append(f"Already visited {url}, skipping.")
            return logs
        visited.add(url)

        
        if logs is None:
            logs = []
        if depth > max_depth:
            return logs

        logs.append(f"Crawling URL: {url} (depth {depth})")
       

        url_lower = url.lower()

        # === Handle PDF == #
        if url_lower.endswith(".pdf"):
            if (self.filter_keyword in url_lower or "syllabus" in url_lower):
                try:
                    docs = self.load_pdf(url)
                    self.all_docs.extend(docs)
                    logs.append(f"Loaded {len(docs)} pages from PDF {url}")
                except Exception as e:
                    logs.append(f"Failed to load PDF {url} - {e}")
            else:
                logs.append(f"Skipped PDF without filter keyword '{self.filter_keyword}' in URL: {url}")
            logs.append("-" * 40)
            return logs

        # === Handle HTML Page ===
        try:
            webpage_docs = self.load_webpage_text(url)
            self.all_docs.extend(webpage_docs)
            logs.append(f"Loaded webpage text from {url} (length {len(webpage_docs[0].page_content) if webpage_docs else 0})")
        except Exception as e:
            logs.append(f"Failed to load webpage text from {url} - {e}")

        # === Crawl Internal Links ===
        try:
            resp = requests.get(url, verify=False, timeout=10)
            soup = BeautifulSoup(resp.text, "html.parser")

            for link in soup.find_all("a", href=True):
                base = self.ensure_dir(url)
                next_url = urljoin(base, link["href"])

                
                next_url_lower = next_url.lower()
                next_domain = urlparse(next_url).netloc
                next_path = urlparse(next_url).path.lower()


                is_html_page = next_path.endswith(".html")


                is_internal = (
                    next_domain == "tinman.cs.gsu.edu" and
                    "/~raj/" in urlparse(next_url).path
                )

                is_syllabus_pdf = (
                    next_url_lower.endswith(".pdf") and
                    ("syllabus" in next_url_lower or "syllabus" in (link.get_text() or "").lower())
                )

                url_contains_keyword = self.filter_keyword in next_url_lower
                link_text_contains_keyword = self.filter_keyword in (link.get_text() or "").lower()

                if is_internal and is_html_page and depth < max_depth and (url_contains_keyword or link_text_contains_keyword):
                    self.crawl_and_load(next_url, depth=depth+1, max_depth=max_depth, logs=logs, visited=visited)
                elif is_syllabus_pdf:
                    try:
                        docs = self.load_pdf(next_url)
                        self.all_docs.extend(docs)
                        logs.append(f"Loaded {len(docs)} pages from PDF {next_url}")
                    except Exception as e:
                        logs.append(f"Failed to load PDF {next_url} - {e}")
                    logs.append("-" * 20)
                else:
                    logs.append(f"Skipping external or non-syllabus/non-internal link: {next_url}")

            logs.append("-" * 40)
        except Exception as e:
            logs.append(f"Error crawling links in {url}: {e}")
            logs.append("-" * 40)

        return logs


    def scrape_all_content(self):
        visited=set()
        for link in self.soup.find_all("a", href=True):
            full_url = urljoin(self.webpage_url, link["href"])
            print("\n" + "=" * 60)
            print(f"Main link: {full_url}")
            print("=" * 60)
            logs = self.crawl_and_load(full_url, depth=0, max_depth=2, logs=[])
            for log in logs:
                print(f"  {log}")
            print("\n" + "=" * 60 + "\n")  # Clear separation between main links

    def build_faiss_index(self):
        if not self.all_docs:
            print("No docs found to index!")
            return None

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=150, chunk_overlap=20)
        chunks = text_splitter.split_documents(self.all_docs)
        print(f"Split into {len(chunks)} chunks")

        embeddings = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2")
        vectorstore = FAISS.from_documents(chunks, embeddings)
        vectorstore.save_local("faiss_index")
        return vectorstore
    
    def load_faiss_index(self):
        if os.path.exists("faiss_index"):
            embeddings = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2")
            self.vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
            print("Loaded FAISS index from disk.")
            return True
        return False

    def save_chunks_to_csv(self, filename="scraped_chunks.csv"):
        """
        Save all chunks with their source, title, and snippet to a CSV for review.
        """
        if not self.all_docs:
            print("No documents to save!")
            return

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100)
        chunks = text_splitter.split_documents(self.all_docs)

        with open(filename, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Source URL", "Title", "Content Snippet"])

            for chunk in chunks:
                source = chunk.metadata.get("source", "unknown")
                title = chunk.metadata.get("title", "No Title")
                snippet = chunk.page_content[:200].replace("\n", " ") + ("..." if len(chunk.page_content) > 200 else "")
                writer.writerow([source, title, snippet])

        print(f"Saved {len(chunks)} chunks to {filename}")

    def setup(self):
        print("Checking for existing FAISS index...")
        if not self.load_faiss_index():
            print("No saved index found. Scraping all content (webpages + PDFs)...")
            self.scrape_all_content()
            print(f"Total documents collected: {len(self.all_docs)}")
            print("Splitting documents into chunks...")
            
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
            chunks = text_splitter.split_documents(self.all_docs)
            print(f"Split into {len(chunks)} chunks")
            
            print("Building FAISS index...")
            embeddings = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2")
            self.vectorstore = FAISS.from_documents(chunks, embeddings)
            self.vectorstore.save_local("faiss_index")
        else:
            print("Skipping crawling and indexing since index was loaded.")
        
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": 1})
        llm_name = "google/flan-t5-large"
        tokenizer = AutoTokenizer.from_pretrained(llm_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(llm_name)
        pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer, max_new_tokens=300)
        llm = HuggingFacePipeline(pipeline=pipe)

        self.qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
        print("Setup complete. Ready to answer questions!")

        # Save chunks for review
        self.save_chunks_to_csv()

    def query(self, question):
        if self.qa_chain is None:
            raise Exception("QA chain not initialized. Run setup() first.")

        retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})
        docs = retriever.invoke(question)

        answer = self.qa_chain.invoke(question)

        sources = []
        for d in docs:
            source_link = d.metadata.get("source", "unknown source")
            title = d.metadata.get("title", "No Title")
            snippet = d.page_content[:200].replace("\n", " ") + ("..." if len(d.page_content) > 200 else "")
            sources.append(f"Title: {title}\nSource: {source_link}\nContent snippet: {snippet}\n")

        return {
            "query": question,
            "answer": answer,
            "sources": sources
        }


if __name__ == "__main__":
    scraper = WebScrape()
    scraper.setup()

    print("\n")
    print("-" * 40)

    while True:
        print("=" * 40)
        q = input("Ask a question (or 'quit' to exit): ").strip()
        if q.lower() in ["q", ""]:
            break
        result = scraper.query(q)
        print('-' * 40)
        print("Sources:")
        for src in result["sources"]:
            print(src)
            print("-" * 40)

        print("Query:", result['query'])
        print("Answer:", result["answer"]["result"])

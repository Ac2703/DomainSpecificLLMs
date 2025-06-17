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


class WebScrape:
    def __init__(self):
        self.webpage_url = "https://tinman.cs.gsu.edu/~raj/past2.html"
        response = requests.get(self.webpage_url, verify=False)
        self.soup = BeautifulSoup(response.text, "html.parser")
        self.all_docs = []
        self.vectorstore = None
        self.qa_chain = None

    def load_webpage_text(self, url):
        try:
            resp = requests.get(url, verify=False, timeout=10)
            soup = BeautifulSoup(resp.text, "html.parser")

            # Get page title from <title> tag or make it "No Title"
            title_tag = soup.title.string.strip() if soup.title else "No Title"


            texts = soup.stripped_strings
            full_text = " ".join(texts)

            if len(full_text) < 50:  # Skip pages with too little text
                print(f"Skipped {url} due to short content")
                return []

            # Make a document object
            doc = Document(
                page_content=full_text,
                metadata={
                    "source": url,
                    "title": title_tag
                }
            )
            print(f"Loaded webpage '{title_tag}' from {url} (length {len(full_text)})")
            return [doc]
        except Exception as e:
            print(f"Failed to load webpage text from {url} - {e}")
            return []
        
    def load_pdf(self, pdf_url):
        try:
            response = requests.get(pdf_url, verify=False, timeout=15)
            pdf_path = "./temp.pdf"
            with open(pdf_path, "wb") as f:
                f.write(response.content)
            loader = PyPDFLoader(pdf_path)
            docs = loader.load()
            print(f"Loaded {len(docs)} pages from PDF {pdf_url}")
            # Attach source URL and a title "Syllabus PDF" to metadata
            for d in docs:
                d.metadata["source"] = pdf_url
                d.metadata["title"] = "Syllabus PDF"
            return docs
        except Exception as e:
            print(f"Failed to load PDF {pdf_url} - {e}")
            return []

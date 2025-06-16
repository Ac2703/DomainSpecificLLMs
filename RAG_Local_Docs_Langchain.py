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


# 'Local' --> all models running locally

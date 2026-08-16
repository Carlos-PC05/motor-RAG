"""Fase 1 — chunking: divide los documentos en fragmentos indexables."""
import sys
from pathlib import Path
from dotenv import load_dotenv
import os

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.stdout.reconfigure(encoding="utf-8")

from loader import cargar_vault

from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer

#Para usar el access token de Hugging face
from huggingface_hub import login

load_dotenv()
hf_token = os.getenv("HF_TOKEN")
if hf_token:
    login(token=hf_token)


MODELO = "sentence-transformers/all-MiniLM-L6-v2"
LIMITE_MODELO = 256
CHUNK_TOKENS = 200
SOLAPAMIENTO_TOKENS = 40

# Cadena de creación de chunks
def crear_chunks():

    # Inicializar tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODELO)
    tokenizer.model_max_length = 10_000  # solo se usa para contar: evita avisos
    # Inicializar Splitter
    splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer, chunk_size=CHUNK_TOKENS, chunk_overlap=SOLAPAMIENTO_TOKENS
    )

    docs = cargar_vault()
    #origen = {d.metadata["fuente"]: d.page_content for d in docs}
    return splitter.split_documents(docs)


#print(f"Documentos: {len(docs)}  ->  Chunks: {len(chunks)}")

# Comprobación
"""
max_tokens, max_doc = 0, ""
violaciones = []
for c in chunks:
    n = len(tokenizer.encode(c.page_content))
    if n > max_tokens:
        max_tokens, max_doc = n, c.metadata["titulo"]
    if n > LIMITE_MODELO:
        violaciones.append((c.metadata["titulo"], n))

print(f"Chunk mas grande: {max_tokens} tokens ({max_doc})")
print(f"Chunks sobre el limite de {LIMITE_MODELO} tokens: {len(violaciones)}")

falta = sum(
    1 for c in chunks if c.page_content not in origen[c.metadata["fuente"]]
)
print(f"Chunks sin integridad textual: {falta}")
"""
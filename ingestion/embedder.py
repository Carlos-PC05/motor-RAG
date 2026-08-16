"""Fase 1 — embeddings: convierte cada chunk en un vector de 384 numeros."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.stdout.reconfigure(encoding="utf-8")

from chunker import crear_chunks

import numpy as np
from sentence_transformers import SentenceTransformer

MODELO = "sentence-transformers/all-MiniLM-L6-v2"
DIMENSIONES = 384

# 1.Inicializar modelo embeddings
modelo = SentenceTransformer(MODELO)

# 2.Creamos chunks
chunks = crear_chunks()
textos = [c.page_content for c in chunks]

# 3.Creamos vectores (Embeddings)
vectors = modelo.encode(textos, batch_size=32, show_progress_bar=True)

###########################

print(f"Chunks: {len(textos)}")
print(f"Forma de la matriz: {vectors.shape}")

normas = np.linalg.norm(vectors, axis=1)
print(f"Norma media de los vectores: {normas.mean():.3f}")

# Consulta de ejemplo
consulta = "receta de paella"
v_query = modelo.encode(consulta)
similitudes = vectors @ v_query / (normas * np.linalg.norm(v_query))

top = np.argsort(similitudes)[-3:][::-1]
print(f"\nConsulta: {consulta}\n")
for i in top:
    c = chunks[i]
    print(f"[{similitudes[i]:.3f}] {c.metadata['titulo']} - {c.page_content[:90].strip()}")
"""
Loader del Vault de Obsidian basado en LangChain.

Migrado del loader manual a LangChain: cada nota se convierte en un
`langchain_core.documents.Document` con metadatos (fuente, titulo, carpeta,
etiquetas, fechas) que el resto del pipeline (chunking, embeddings, ChromaDB)
consume de forma nativa.

Correcciones sobre el loader manual anterior:
- Las etiquetas se extraen fuera de los bloques de codigo: comentarios de
  codigo como ``# Import ImageIO`` ya no se cuentan como etiquetas.
- Se soportan acentos (``#matematicas``) y etiquetas anidadas (``#padre/hijo``).
- Los enlaces a adjuntos ``[archivo](attachment:...)`` se limpian del texto.
- Se ignoran carpetas de configuracion (``.obsidian``, ``.trash``, ...).
"""

import re
import sys
from pathlib import Path
from typing import Iterator

from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document

sys.stdout.reconfigure(encoding="utf-8")

VAULT = Path(r"C:\Users\carlo\OneDrive\Documentos\Obsidian Vault")

IGNORAR = {".obsidian", ".trash", ".git", ".github"}

# Enlaces a adjuntos [nombre](attachment:uuid:archivo.pdf) -> ruido para embeddings
ADJUNTO = re.compile(r"\[[^\]]*\]\(attachment:[^)]*\)")
# Bloques de codigo: los comentarios '# ...' NO son etiquetas
BLOQUE_CODIGO = re.compile(r"```.*?```", re.DOTALL)
# Etiqueta Obsidian: '#' pegada al texto, con acentos, guiones y anidadas 'padre/hijo'
ETIQUETA = re.compile(r"(?<!\S)#([A-Za-zÁÉÍÓÚáéíóúÑñ0-9_/-]+)")


class VaultLoader(BaseLoader):
    """
    Carga las notas del Vault de Obsidian como `Document` de LangChain.

    Implementa `BaseLoader` de LangChain (langchain-core): cada nota es un
    `Document` con `page_content` (texto limpio) y metadatos (fuente, titulo,
    carpeta, etiquetas, fechas).

    @args:
        path (str | Path): Ruta del Vault de Obsidian.
        encoding (str): Codificacion de lectura de las notas (por defecto UTF-8).
    """

    def __init__(self, path: str | Path, encoding: str = "utf-8") -> None:
        self.file_path = Path(path)
        self.encoding = encoding

    def lazy_load(self) -> Iterator[Document]:
        """Itera sobre las notas del vault, ignorando carpetas de configuracion."""
        for ruta in self.file_path.rglob("*.md"):
            if any(parte in IGNORAR for parte in ruta.parts):
                continue
            yield self._cargar_nota(ruta)

    def _cargar_nota(self, ruta: Path) -> Document:
        """Convierte una nota Markdown en un `Document` de LangChain."""
        texto = ruta.read_text(encoding=self.encoding)
        texto = ADJUNTO.sub("", texto)

        # Las etiquetas se buscan fuera de los bloques de codigo
        sin_codigo = BLOQUE_CODIGO.sub("", texto)
        etiquetas = sorted(set(ETIQUETA.findall(sin_codigo)))

        stat = ruta.stat()
        return Document(
            page_content=texto,
            metadata={
                "fuente": str(ruta),
                "titulo": ruta.stem,
                "carpeta": ruta.parent.name,
                "etiquetas": etiquetas,
                "path": str(ruta),
                "created": stat.st_ctime,
                "last_modified": stat.st_mtime,
            },
        )


def cargar_vault(raiz: Path = VAULT) -> list[Document]:
    """
    Devuelve las notas del vault como lista de `Document` de LangChain.

    @args:
        raiz (Path): Ruta absoluta del Vault.
    @returns:
        list[Document]: Lista de documentos del Vault.
    """
    return VaultLoader(raiz).load()


if __name__ == "__main__":
    docs = cargar_vault()
    print("=== RESUMEN DEL VAULT ===")
    print(f"Total documentos cargados: {len(docs)}")

    # 1. Comprobacion de Etiquetas
    docs_con_etiquetas = [d for d in docs if d.metadata["etiquetas"]]
    print(f"Documentos con etiquetas: {len(docs_con_etiquetas)}")

    print("\n--- EJEMPLOS DE NOTAS CON ETIQUETAS ---")
    for d in docs_con_etiquetas[:3]:
        print(f"\n[Nota] {d.metadata['titulo']} ({d.metadata['carpeta']})")
        print(f"  Etiquetas extraidas: {d.metadata['etiquetas']}")
        print(f"  Texto (fragmento): {d.page_content[:120].strip()}...")

    # 2. Comprobacion de Adjuntos
    print("\n--- COMPROBACION DE ADJUNTOS ---")
    docs_con_adjunto_remanente = [d for d in docs if "attachment:" in d.page_content]
    print(
        "Documentos con enlaces 'attachment:' en texto procesado: "
        f"{len(docs_con_adjunto_remanente)}"
    )

    if len(docs_con_adjunto_remanente) == 0:
        print(
            "OK Limpieza verificada: Todos los enlaces de tipo "
            "[archivo](attachment:...) se han eliminado del texto."
        )
    else:
        print("Atencion: Se encontraron referencias a attachment: no eliminadas.")
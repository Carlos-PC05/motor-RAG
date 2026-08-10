# Roadmap — Motor de RAG (proyecto open source de portfolio)

> Backend de búsqueda semántica + generación de respuestas sobre documentos propios.
> Objetivo: aprender bien cada concepto, sin prisa. Cada fase es autocontenida: puedes parar y retomar sin perder el hilo.

## Stack

- **API**: FastAPI
- **Embeddings**: `sentence-transformers` (CPU) — modelo `all-MiniLM-L6-v2` para empezar
- **Base vectorial**: ChromaDB (más simple de arrancar que FAISS, buena para aprender)
- **Generación**: Ollama (ya lo tienes instalado) con un modelo pequeño tipo Qwen o Llama 3.2
- **Contenerización**: Docker (al final, cuando el proyecto ya funcione)

---

## Fase 0 — Setup y decisión de dominio (1 sesión)

- [x] Crear repo en GitHub con licencia (MIT recomendado para portfolio) y `.gitignore` de Python
- [x] Decidir sobre qué documentos vas a hacer RAG (ideas: tus propios apuntes de la carrera, documentación de una librería, artículos de Wikipedia de un tema concreto). Esto determina qué vas a poder enseñar en las demos.
- [x] Entorno virtual + `requirements.txt` inicial (`fastapi`, `uvicorn`, `sentence-transformers`, `chromadb`)

**Concepto clave a entender antes de seguir:** qué es un embedding y por qué convertir texto en vectores permite "buscar por significado" en vez de por palabras exactas.

---

## Fase 1 — Ingesta y embeddings (semana 1)

- [ ] Cargar documentos (texto plano / PDF / markdown, tú eliges)
- [ ] Implementar **chunking**: dividir documentos largos en fragmentos manejables
  - Entender el trade-off: chunks muy pequeños pierden contexto, muy grandes pierden precisión
  - Probar chunking simple (por nº de caracteres/tokens con solapamiento) antes de nada más sofisticado
- [ ] Generar embeddings de cada chunk con `sentence-transformers`
- [ ] Guardar chunks + embeddings en ChromaDB

**Concepto clave:** similitud coseno — cómo se mide "cercanía" entre vectores y por qué eso equivale a cercanía semántica entre textos.

**Checkpoint de aprendizaje:** deberías poder explicar con tus palabras qué le pasa a una frase desde que entra como texto hasta que se guarda como vector.

---

## Fase 2 — Recuperación (retrieval) y API de consulta (semana 2)

- [ ] Endpoint `POST /query`: recibe una pregunta, la convierte en embedding, busca los N chunks más similares
- [ ] Probar distintos valores de `top_k` y ver cómo afecta a la relevancia de resultados
- [ ] Endpoint `POST /ingest` (o script CLI) para añadir nuevos documentos sin reiniciar todo
- [ ] Manejo básico de errores (documento vacío, query vacía, etc.)

**Concepto clave:** la diferencia entre _retrieval_ (encontrar información relevante) y _generation_ (redactar una respuesta) — son dos problemas distintos que se combinan en RAG.

**Checkpoint de aprendizaje:** en este punto ya tienes un buscador semántico funcional, aunque no "hable". Es un buen momento para probarlo con preguntas reales y ver si recupera lo que esperas.

---

## Fase 3 — Generación (RAG completo) (semana 3)

- [ ] Construir el prompt: inyectar los chunks recuperados como contexto + la pregunta del usuario
- [ ] Llamar a Ollama con ese prompt y devolver la respuesta generada
- [ ] Endpoint final `POST /ask`: pregunta → retrieval → generación → respuesta con fuentes citadas (qué chunks se usaron)
- [ ] Evaluación básica: crea un pequeño set de 10-15 preguntas de prueba y anota manualmente si las respuestas son correctas/relevantes

**Concepto clave:** prompt engineering para RAG — cómo estructurar el contexto para que el modelo no "alucine" y cite bien las fuentes.

**Checkpoint de aprendizaje:** deberías poder explicar por qué RAG reduce alucinaciones comparado con preguntar directamente al LLM sin contexto.

---

## Fase 4 — Pulido para portfolio (opcional, sin límite de tiempo)

- [ ] Tests básicos con `pytest` (al menos del pipeline de ingesta y de un query end-to-end)
- [ ] README con: diagrama simple de arquitectura, cómo instalar, cómo usar, ejemplos de queries reales
- [ ] Dockerfile + `docker-compose.yml` (API + Ollama)
- [ ] GitHub Actions básico (lint + tests en cada push)
- [ ] (Extra) Métricas de evaluación más serias: precision@k, o comparar contra un baseline de búsqueda por palabras clave (BM25) para tener un "antes y después" que enseñar

---

## Notas de seguimiento

_(usa esta sección para anotar dónde lo dejaste cada vez que retomes el proyecto)_

- Última sesión:
- Próximo paso:
- Dudas pendientes:

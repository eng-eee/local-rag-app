File structure

                    main.py
                       │
        ┌──────────────┼──────────────┐
        │              │              │
        ▼              ▼              ▼
    ingestion      retrieval        llm
        │              │              │
        ▼              ▼              ▼
     chunks       PostgreSQL        Ollama
                       │
                       ▼
                    pgvector

                    
Potential final architecture:


Python prototype
       ↓
understand architecture
       ↓
C++ RAG engine
       ↓
llama.cpp
       ↓
optimized local AI application


                   Browser
                      │
                      ▼
                 FastAPI/C++
                      │
                      ▼
              ┌───────────────┐
              │ RAG Engine     │
              │ C++            │
              └───────┬───────┘
                      │
        ┌─────────────┼─────────────┐
        ▼             ▼             ▼
   PostgreSQL     Embedding      llama.cpp
    pgvector        model           LLM


"""
UW-Madison CS Course Assistant
A RAG pipeline using Cohere's Embed, Rerank, and Command R+ APIs
"""

import cohere
import numpy as np
import os

# Initialize Cohere client
# Set your API key as environment variable: export COHERE_API_KEY="your_key"
API_KEY = os.environ.get("COHERE_API_KEY", "your_api_key_here")
co = cohere.ClientV2(API_KEY)

# ─────────────────────────────────────────────
# KNOWLEDGE BASE: UW-Madison CS Course Descriptions
# ─────────────────────────────────────────────
COURSES = [
    {
        "id": "CS300",
        "title": "Programming II",
        "description": "Introduction to object-oriented programming using Java. Topics include classes, inheritance, interfaces, generics, recursion, and basic data structures like linked lists and binary trees."
    },
    {
        "id": "CS354",
        "title": "Machine Organization and Programming",
        "description": "Introduction to C programming, assembly language, and computer organization. Topics include memory management, pointers, system calls, and low-level programming."
    },
    {
        "id": "CS400",
        "title": "Programming III",
        "description": "Advanced data structures and algorithms in Java. Topics include hash tables, red-black trees, graphs, dynamic programming, and algorithm complexity analysis."
    },
    {
        "id": "CS537",
        "title": "Introduction to Operating Systems",
        "description": "Principles of modern operating systems including process management, memory management, file systems, concurrency, and distributed systems fundamentals."
    },
    {
        "id": "CS540",
        "title": "Introduction to Artificial Intelligence",
        "description": "Foundational AI concepts including search algorithms, constraint satisfaction, Bayesian networks, machine learning, neural networks, and natural language processing."
    },
    {
        "id": "CS541",
        "title": "Applied Machine Learning",
        "description": "Practical machine learning techniques including supervised learning, unsupervised learning, neural networks, model evaluation, and real-world applications using Python and scikit-learn."
    },
    {
        "id": "CS544",
        "title": "Introduction to Big Data Systems",
        "description": "Large-scale data processing systems including Hadoop, Spark, distributed storage, data pipelines, SQL at scale, and cloud computing platforms like AWS."
    },
    {
        "id": "CS559",
        "title": "Computer Graphics",
        "description": "Fundamentals of 2D and 3D computer graphics including rendering pipelines, transformations, shading, texture mapping, and WebGL programming."
    },
    {
        "id": "CS564",
        "title": "Database Management Systems",
        "description": "Relational database design, SQL, query optimization, transaction management, indexing, and NoSQL systems. Includes hands-on projects with PostgreSQL."
    },
    {
        "id": "CS571",
        "title": "Building User Interfaces",
        "description": "Modern UI development using React, REST APIs, and mobile frameworks. Topics include state management, accessibility, user testing, and full-stack web development."
    },
    {
        "id": "CS577",
        "title": "Introduction to Algorithms",
        "description": "Algorithm design and analysis including divide and conquer, greedy algorithms, dynamic programming, graph algorithms, NP-completeness, and approximation algorithms."
    },
    {
        "id": "CS639",
        "title": "Topics in Database Systems: Deep Learning",
        "description": "Advanced topics in deep learning including convolutional neural networks, transformers, attention mechanisms, fine-tuning large language models, and deployment."
    },
    {
        "id": "CS640",
        "title": "Introduction to Computer Networks",
        "description": "Principles of computer networking including TCP/IP, routing protocols, network security, HTTP, DNS, and socket programming."
    },
    {
        "id": "STAT479",
        "title": "Machine Learning",
        "description": "Statistical approaches to machine learning including linear models, SVMs, ensemble methods, deep learning, and probabilistic graphical models with mathematical foundations."
    },
    {
        "id": "CS506",
        "title": "Data Science with Python",
        "description": "Practical data science using Python. Covers data wrangling with Pandas, visualization with Matplotlib, statistical analysis, and introductory machine learning with scikit-learn."
    },
]


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a list of texts using Cohere's embed-english-v3.0 model."""
    response = co.embed(
        texts=texts,
        model="embed-english-v3.0",
        input_type="search_document",
        embedding_types=["float"]
    )
    return response.embeddings.float


def embed_query(query: str) -> list[float]:
    """Embed a search query."""
    response = co.embed(
        texts=[query],
        model="embed-english-v3.0",
        input_type="search_query",
        embedding_types=["float"]
    )
    return response.embeddings.float[0]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Calculate cosine similarity between two vectors."""
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def retrieve_top_k(query_embedding, doc_embeddings, k=5) -> list[int]:
    """Retrieve top-k most similar documents by cosine similarity."""
    similarities = [
        cosine_similarity(query_embedding, doc_emb)
        for doc_emb in doc_embeddings
    ]
    return sorted(range(len(similarities)), key=lambda i: similarities[i], reverse=True)[:k]


def rerank_results(query: str, candidates: list[dict], top_n=3) -> list[dict]:
    """Rerank candidates using Cohere's Rerank API for improved relevance."""
    docs = [f"{c['title']}: {c['description']}" for c in candidates]
    response = co.rerank(
        query=query,
        documents=docs,
        model="rerank-english-v3.0",
        top_n=top_n
    )
    return [candidates[r.index] for r in response.results]


def generate_answer(query: str, context_courses: list[dict]) -> str:
    """Generate a natural language answer using Command R+."""
    context = "\n\n".join([
        f"**{c['id']} - {c['title']}**\n{c['description']}"
        for c in context_courses
    ])

    messages = [
        {
            "role": "user",
            "content": f"""You are a helpful UW-Madison course advisor. 
Answer the student's question based only on the course information provided.
Be specific, mention course numbers, and be helpful but concise.

Course Information:
{context}

Student Question: {query}"""
        }
    ]

    response = co.chat(
        model="command-r-plus-08-2024",
        messages=messages,
    )
    return response.message.content[0].text


def build_index():
    """Pre-compute embeddings for all courses."""
    print("Building course index with Cohere embeddings...")
    texts = [f"{c['title']}: {c['description']}" for c in COURSES]
    embeddings = embed_texts(texts)
    print(f"Indexed {len(COURSES)} courses successfully.\n")
    return embeddings


def answer_question(query: str, doc_embeddings: list) -> str:
    """Full RAG pipeline: embed -> retrieve -> rerank -> generate."""
    print(f"Question: {query}\n")

    # Step 1: Embed the query
    query_embedding = embed_query(query)

    # Step 2: Retrieve top-5 candidates by vector similarity
    top_indices = retrieve_top_k(query_embedding, doc_embeddings, k=5)
    candidates = [COURSES[i] for i in top_indices]

    # Step 3: Rerank to get top-3 most relevant
    reranked = rerank_results(query, candidates, top_n=3)

    print("Most relevant courses found:")
    for c in reranked:
        print(f"  - {c['id']}: {c['title']}")

    # Step 4: Generate answer with Command R+
    print("\nGenerating answer...\n")
    answer = generate_answer(query, reranked)
    print(f"Answer:\n{answer}\n")
    print("-" * 60 + "\n")
    return answer


def main():
    print("=" * 60)
    print("UW-Madison CS Course Assistant")
    print("Powered by Cohere Embed + Rerank + Command R+")
    print("=" * 60 + "\n")

    # Build the index once
    doc_embeddings = build_index()

    # Example queries
    queries = [
        "What courses should I take to learn machine learning?",
        "Which course covers databases and SQL?",
        "I want to build web applications, what should I take?",
        "What's the best course for understanding how computers work at a low level?",
        "Which courses involve Python programming?",
    ]

    for query in queries:
        answer_question(query, doc_embeddings)

    # Interactive mode
    print("\n" + "=" * 60)
    print("Interactive Mode - Ask your own questions!")
    print("Type 'quit' to exit")
    print("=" * 60 + "\n")

    while True:
        user_query = input("Your question: ").strip()
        if user_query.lower() in ["quit", "exit", "q"]:
            print("Thanks for using the UW Course Assistant!")
            break
        if user_query:
            answer_question(user_query, doc_embeddings)


if __name__ == "__main__":
    main()

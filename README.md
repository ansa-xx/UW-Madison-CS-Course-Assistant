# UW-Madison-CS-Course-Assistant
A semantic Q&A system for UW-Madison Computer Science courses, built using Cohere's full RAG stack.
What It Does
Ask natural language questions about UW-Madison CS courses and get accurate, specific answers grounded in real course descriptions.
Example queries:

"What courses should I take to learn machine learning?"
"Which course covers databases and SQL?"
"I want to build web applications, what should I take?"

How It Works
This project demonstrates a complete Retrieval-Augmented Generation (RAG) pipeline using three Cohere APIs:

Cohere Embed (embed-english-v3.0) — converts course descriptions and queries into semantic vector embeddings
Cohere Rerank (rerank-english-v3.0) — improves retrieval quality by reranking the top candidates
Cohere Command R+ (command-r-plus-08-2024) — generates natural language answers grounded in retrieved course context

User Question
     │
     ▼
Embed Query (Cohere Embed)
     │
     ▼
Vector Search → Top 5 Candidates
     │
     ▼
Rerank → Top 3 Most Relevant (Cohere Rerank)
     │
     ▼
Generate Answer (Cohere Command R+)
     │
     ▼
Natural Language Response
Setup
bash# Clone the repo
git clone https://github.com/ansa-xx/UW-Madison-CS-Course-Assistant
cd UW-Madison-CS-Course-Assistant

# Install dependencies
pip install cohere numpy

# Set your Cohere API key
export COHERE_API_KEY="your_api_key_here"

# Run
python uw_course.py
Get a free API key at dashboard.cohere.com
Courses Covered
15 UW-Madison CS and Data Science courses including CS540 (AI), CS541 (ML), CS544 (Big Data), CS564 (Databases), CS571 (UI), CS577 (Algorithms), and more.

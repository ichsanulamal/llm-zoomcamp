# README

## **Setup Guide**

### **Step 1: Setting Up the Database**

Once your Docker setup is complete, run the following `docker-compose.yml` file to set up a TimescaleDB instance. This will also install extensions like `pg_vector` and `pgai`.

```yaml
version: '3.8'

services:
  timescaledb:
    image: timescale/timescaledb:latest-pg16
    container_name: timescaledb
    environment:
      - POSTGRES_DB=postgres
      - POSTGRES_USER=postgres
      - POSTGRES_PASSWORD=postgres
    ports:
      - "5444:5432"
    volumes:
      - timescaledb_data:/var/lib/postgresql/data
    restart: unless-stopped

volumes:
  timescaledb_data:
    driver: local
```

After running this Docker container, connect to the database and run the following SQL commands to enable the required extensions:

```sql
-- Enable the pgai extension
CREATE EXTENSION IF NOT EXISTS ai CASCADE;

-- Verify active extensions
SELECT * FROM pg_extension;
```

### **Step 2: Installing and Running Ollama**

1. **Install Ollama:**

   Run the following command to install Ollama:

   ```bash
   curl -fsSL https://ollama.com/install.sh | sh
   ```

2. **Pull and Run LLM Model:**

   Pull the Llama3.2 model and run it:

   ```bash
   ollama pull llama3.2
   ollama run llama3.2
   ```

3. **Pull the Embedding Model:**

   Pull in the embedding model:

   ```bash
   ollama pull all-minilm
   ```

Now your LLM should be running and accessible at:
**[http://localhost:11434/api/generate](http://localhost:11434/api/generate)**

---

## **Local RAG (Retrieval-Augmented Generation) Setup**

### **Step 1: Create a Table to Store Text and Embeddings**

In your database, create a table to store documents and their corresponding embeddings:

```sql
CREATE TABLE IF NOT EXISTS documents (
    id SERIAL PRIMARY KEY,
    title TEXT,
    content TEXT,
    embedding VECTOR(384)
);
```

### **Step 2: Store Data as Embeddings**

Use the following Python script to insert documents into the database with their embeddings.

```python
import psycopg2
from psycopg2.extras import execute_values
import requests
import json

# Ollama API endpoint
OLLAMA_API = "http://localhost:11434/api/embeddings"

# Database connection parameters
DB_PARAMS = {
    "dbname": "postgres",
    "user": "postgres",
    "password": "postgres",
    "host": "localhost",
    "port": "5444"
}

def get_embedding(text):
    response = requests.post(OLLAMA_API, json={"model": "all-minilm", "prompt": text})
    return response.json()['embedding']

def insert_documents(documents):
    conn = psycopg2.connect(**DB_PARAMS)
    cur = conn.cursor()

    try:
        data = [(doc['title'], doc['content'], get_embedding(doc['content'])) for doc in documents]
        execute_values(cur, """
            INSERT INTO documents (title, content, embedding)
            VALUES %s
        """, data)
        conn.commit()
    except Exception as e:
        print(f"An error occurred: {e}")
        conn.rollback()
    finally:
        cur.close()
        conn.close()

# Example usage
documents = [
    {"title": "Seoul Tower", "content": "Seoul Tower is a communication and observation tower located on Namsan Mountain in central Seoul, South Korea."},
    {"title": "Gwanghwamun Gate", "content": "Gwanghwamun is the main and largest gate of Gyeongbokgung Palace, in Jongno-gu, Seoul, South Korea."},
    {"title": "Bukchon Hanok Village", "content": "Bukchon Hanok Village is a Korean traditional village in Seoul with a long history."},
    {"title": "Myeong-dong Shopping Street", "content": "Myeong-dong is one of the primary shopping districts in Seoul, South Korea."},
    {"title": "Dongdaemun Design Plaza", "content": "The Dongdaemun Design Plaza is a major urban development landmark in Seoul, South Korea."}
]

insert_documents(documents)
```

### **Step 3: Implement RAG for Query-Based Content Generation**

This Python script retrieves relevant documents from the database based on the query and generates a response using the LLM.

```python
import psycopg2
import requests
import json

# Ollama API endpoint
OLLAMA_API = "http://localhost:11434/api"

# Database connection parameters
DB_PARAMS = {
    "dbname": "postgres",
    "user": "postgres",
    "password": "postgres",
    "host": "localhost",
    "port": "5444"
}

def connect_db():
    return psycopg2.connect(**DB_PARAMS)

def get_embedding(text):
    response = requests.post(f"{OLLAMA_API}/embeddings", json={"model": "all-minilm", "prompt": text})
    return response.json()['embedding']

def generate_response(prompt):
    response = requests.post(f"{OLLAMA_API}/generate", json={
        "model": "llama3.2",
        "prompt": prompt,
        "stream": False
    }, stream=True)
    
    full_response = ""
    for line in response.iter_lines():
        if line:
            json_response = json.loads(line)
            if 'response' in json_response:
                full_response += json_response['response']
    
    return full_response

def retrieve_and_generate_response(query):
    conn = connect_db()
    cur = conn.cursor()
    
    # Embed the query
    query_embedding = get_embedding(query)
    
    # Format the embedding for pg_vector
    embedding_string = f"[{','.join(map(str, query_embedding))}]"
    
    # Retrieve relevant documents using cosine similarity
    cur.execute("""
        SELECT title, content, 1 - (embedding <=> %s::vector) AS similarity
        FROM documents
        ORDER BY similarity DESC
        LIMIT 5;
    """, (embedding_string,))
    
    rows = cur.fetchall()
    
    # Prepare the context for generating the response
    context = "\n\n".join([f"Title: {row[0]}\nContent: {row[1]}" for row in rows])
    
    # Generate the response
    prompt = f"Query: {query}\nContext: {context}\nPlease provide a concise answer based on the given context."
    response = generate_response(prompt)
    
    print(f"Response: {response}")
    
    cur.close()
    conn.close()

def main():
    retrieve_and_generate_response("Tell me about landmarks in Seoul")

if __name__ == "__main__":
    main()
```

### **Summary:**

1. **Set up the TimescaleDB instance** via Docker, enabling the required extensions.
2. **Install Ollama** and pull both the LLM and embedding models.
3. **Store documents** in the database with embeddings generated from Ollama.
4. **Implement the Retrieval-Augmented Generation (RAG)** pattern, where you query the database for relevant documents and generate a concise response using the LLM.

---

This format should be much easier to follow, with the steps clearly outlined and the code organized for readability. If you have any further questions or need more clarification, feel free to ask!

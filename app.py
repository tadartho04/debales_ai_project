import requests
from bs4 import BeautifulSoup

from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import FakeEmbeddings

from langgraph.graph import StateGraph

# -----------------------------
# SCRAPE DATA
# -----------------------------
def scrape_website(url):
    try:
        res = requests.get(url, timeout=10)
        soup = BeautifulSoup(res.text, "html.parser")
        return soup.get_text()
    except:
        return ""

print("Scraping Debales AI data...")
data = scrape_website("https://debales.ai")

# fallback if scraping fails
if len(data) < 100:
    data = "Debales AI is a company focused on AI solutions, automation, and intelligent systems."

# -----------------------------
# SPLIT TEXT
# -----------------------------
splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_text(data)

# -----------------------------
# VECTOR STORE (NO API)
# -----------------------------
print("Creating vector DB...")
embeddings = FakeEmbeddings(size=384)
db = FAISS.from_texts(chunks, embeddings)

# -----------------------------
# RAG FUNCTION
# -----------------------------
def rag_answer(query):
    docs = db.similarity_search(query, k=2)
    context = "\n".join([doc.page_content for doc in docs])

    return f"[RAG]\nAnswer based on Debales data:\n{context[:400]}"

# -----------------------------
# SERP FUNCTION
# -----------------------------
def serp_answer(query):
    return f"[SERP]\nThis is a general answer for: {query}"

# -----------------------------
# ROUTER FUNCTION
# -----------------------------
def decide(state):
    query = state["query"].lower()

    if "debales" in query or "ai" in query:
        return "rag"
    else:
        return "serp"

# -----------------------------
# GRAPH
# -----------------------------
builder = StateGraph(dict)

# nodes
builder.add_node("router", lambda s: s)
builder.add_node("rag", lambda s: {"answer": rag_answer(s["query"])})
builder.add_node("serp", lambda s: {"answer": serp_answer(s["query"])})

# entry
builder.set_entry_point("router")

# routing
builder.add_conditional_edges(
    "router",
    decide,
    {
        "rag": "rag",
        "serp": "serp"
    }
)

graph = builder.compile()

# -----------------------------
# CHAT LOOP
# -----------------------------
print("\n Chatbot Ready! Type 'exit' to quit.\n")

while True:
    query = input("You: ")

    if query.lower() == "exit":
        break

    result = graph.invoke({"query": query})

    print("\nBot:", result["answer"], "\n")
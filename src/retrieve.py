"""
retrieve.py — Load FAISS index and answer queries through Groq.
Usage: from src.retrieve import RAGChain; chain = RAGChain(); answer = chain.query("your question")
"""
import os
from pathlib import Path
from dotenv import load_dotenv
 
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
 
from src.config import EMBED_MODEL, GROQ_MODEL, INDEX_DIR, TOP_K
 
load_dotenv()
 
 
class RAGChain:
    def __init__(self, index_dir: str = INDEX_DIR):
        """Load FAISS index and initialize Groq LLM."""
        self.index_dir = Path(index_dir)
        if not self.index_dir.exists():
            raise FileNotFoundError(f"FAISS index not found at {self.index_dir}. Run ingest first.")
 
        # Load embeddings and vector store
        self.embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
        self.vectorstore = FAISS.load_local(str(self.index_dir), self.embeddings, allow_dangerous_deserialization=True)
 
        # Initialize Groq LLM
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            raise ValueError("GROQ_API_KEY not set in .env")
 
        self.llm = ChatGroq(
            groq_api_key=groq_api_key,
            model_name=GROQ_MODEL,
            temperature=0.1,
            max_tokens=512
        )
 
        # RAG prompt
        self.prompt = PromptTemplate(
            input_variables=["context", "question"],
            template="""You are a helpful assistant. Answer the question based on the provided context.
If the context doesn't contain relevant information, say "I don't have enough information to answer that."
 
Context:
{context}
 
Question: {question}
 
Answer:"""
        )
 
    def query(self, question: str) -> dict:
        """
        Query the RAG chain.
        Returns: {"answer": str, "sources": list of relevant docs}
        """
        # Retrieve top-k relevant documents
        docs = self.vectorstore.similarity_search(question, k=TOP_K)
 
        if not docs:
            return {
                "answer": "No relevant documents found.",
                "sources": []
            }
 
        # Format context from retrieved docs
        context = "\n\n".join([f"[Doc {i+1}]\n{doc.page_content}" for i, doc in enumerate(docs)])
 
        # Generate answer
        prompt_text = self.prompt.format(context=context, question=question)
        response = self.llm.invoke(prompt_text)
        answer = response.content if hasattr(response, 'content') else str(response)
 
        return {
            "answer": answer,
            "sources": [{"content": doc.page_content[:200], "page": doc.metadata.get("page", 0)} for doc in docs]
        }
 
 
if __name__ == "__main__":
    # Test the chain
    chain = RAGChain()
    result = chain.query("What is this document about?")
    print(f"Answer: {result['answer']}")
    print(f"Sources: {len(result['sources'])} documents retrieved")
 
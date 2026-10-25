"""
retrieve.py — Load FAISS index and answer queries through Groq.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import JinaEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate

from src.config import GROQ_MODEL, INDEX_DIR, TOP_K

load_dotenv()


class RAGChain:
    def __init__(self, index_dir: str = INDEX_DIR):
        self.index_dir = Path(index_dir)
        if not self.index_dir.exists():
            raise FileNotFoundError(f"FAISS index not found at {self.index_dir}. Run ingest first.")

        self.embeddings = JinaEmbeddings(
            jina_api_key=os.getenv("JINA_API_KEY"),
            model_name="jina-embeddings-v2-base-en"
        )
        self.vectorstore = FAISS.load_local(
            str(self.index_dir),
            self.embeddings,
            allow_dangerous_deserialization=True
        )

        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            raise ValueError("GROQ_API_KEY not set in .env")

        self.llm = ChatGroq(
            groq_api_key=groq_api_key,
            model_name=GROQ_MODEL,
            temperature=0.1,
            max_tokens=512
        )

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
        docs = self.vectorstore.similarity_search(question, k=TOP_K)

        if not docs:
            return {"answer": "No relevant documents found.", "sources": []}

        context = "\n\n".join([f"[Doc {i+1}]\n{doc.page_content}" for i, doc in enumerate(docs)])
        prompt_text = self.prompt.format(context=context, question=question)
        response = self.llm.invoke(prompt_text)
        answer = response.content if hasattr(response, 'content') else str(response)

        return {
            "answer": answer,
            "sources": [{"content": doc.page_content[:200], "page": doc.metadata.get("page", 0)} for doc in docs]
        }


if __name__ == "__main__":
    chain = RAGChain()
    result = chain.query("What is this document about?")
    print(f"Answer: {result['answer']}")
    print(f"Sources: {len(result['sources'])} documents retrieved")
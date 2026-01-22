from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from src.prompt_template import get_anime_prompt


class AnimeRecommender:
    def __init__(self, retriever, api_key: str, model_name: str):
        # LLM
        self.llm = ChatGroq(
            api_key=api_key,
            model=model_name,
            temperature=0
        )

        # Prompt
        self.prompt = get_anime_prompt()

        # ✅ LCEL RAG chain (Replacement for RetrievalQA)
        self.qa_chain = (
            {
                "context": retriever,
                "question": RunnablePassthrough()
            }
            | self.prompt
            | self.llm
            | StrOutputParser()
        )

    def get_recommendation(self, query: str) -> str:
        return self.qa_chain.invoke(query)

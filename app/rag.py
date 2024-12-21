from typing import Optional
from langchain_core.output_parsers import StrOutputParser
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_community.vectorstores.faiss import FAISS
from langchain_core.prompts import load_prompt
from langchain_ollama import OllamaEmbeddings, ChatOllama

from base import BaseChain


# 문서 포맷팅
def format_docs(docs):
    return "\n\n".join(
        f"<document><content>{doc.page_content}</content><page>{doc.metadata['page']}</page><source>{doc.metadata['source']}</source></document>"
        for doc in docs
    )


class RagChain(BaseChain):

    def __init__(
        self,
        model: str = "exaone",
        temperature: float = 0.3,
        system_prompt: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(model, temperature, **kwargs)
        self.system_prompt = (
            system_prompt
            or "You are a helpful AI Assistant. Your name is '테디'. You must answer in Korean."
        )
        if "file_path" in kwargs:
            self.file_path = kwargs["file_path"]

    def setup(self):
        if not self.file_path:
            raise ValueError("file_path is required")

        # Splitter 설정
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

        # 문서 로드
        loader = PDFPlumberLoader(self.file_path)
        docs = loader.load_and_split(text_splitter=text_splitter)

        # 캐싱을 지원하는 임베딩 설정
        EMBEDDING_MODEL = "bge-m3"
        embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)

        # 벡터 DB 저장
        vectorstore = FAISS.from_documents(docs, embedding=embeddings)

        # 문서 검색기 설정
        retriever = vectorstore.as_retriever()

        # 프롬프트 로드
        prompt = load_prompt("prompts/rag-exaone.yaml", encoding="utf-8")

        # Ollama 모델 지정
        llm = ChatOllama(
            model="exaone",
            temperature=0,
        )

        # 체인 생성
        chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        return chain

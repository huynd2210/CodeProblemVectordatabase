from typing import List

from langchain.document_loaders import TextLoader
from langchain.schema import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings

class VectorDB:
    def __init__(self):
        self.chroma = None
        self.textSplitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)

    def initChroma(self, collectionName, persistDirectory, embeddingFunction=None) -> None:
        if embeddingFunction is None:
            modelName = "sentence-transformers/all-mpnet-base-v2"
            embeddingFunction = HuggingFaceEmbeddings(model_name=modelName)
        self.chroma = Chroma(collection_name=collectionName, embedding_function=embeddingFunction, persist_directory=persistDirectory)

    def addDocument(self, documentPath, metadata=None) -> None:
        loader = TextLoader(documentPath)
        document = loader.load()
        texts = self.textSplitter.split_documents(document)
        self.chroma.add_texts(texts, metadatas=metadata)

    def updateDocument(self):
        pass
    def search(self, query) -> List[Document]:
        return self.chroma.similarity_search(query)
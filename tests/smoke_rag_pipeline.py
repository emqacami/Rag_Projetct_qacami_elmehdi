import sys
import tempfile
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from climate_tweet_rag import CHROMA_SETTINGS, RagConfig, format_docs, split_documents


class SimpleEmbeddings(Embeddings):
    def _embed(self, text: str) -> list[float]:
        lowered = text.lower()
        return [
            float(len(lowered)),
            float(lowered.count("climate")),
            float(lowered.count("change")),
            float(lowered.count("warming")),
            float(lowered.count("policy")),
            float(lowered.count("science")),
        ]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


def main() -> None:
    config = RagConfig(chunk_size=200, chunk_overlap=20)
    documents = [
        Document(page_content="Climate change is making extreme weather worse.", metadata={"row": 0}),
        Document(page_content="People are debating climate policy and clean energy.", metadata={"row": 1}),
        Document(page_content="Some tweets mention global warming and scientific evidence.", metadata={"row": 2}),
    ]
    chunks = split_documents(documents, config)

    assert len(chunks) == len(documents)

    with tempfile.TemporaryDirectory(dir=ROOT) as chroma_dir:
        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=SimpleEmbeddings(),
            collection_name="climate_smoke_test",
            persist_directory=chroma_dir,
            client_settings=CHROMA_SETTINGS,
        )

        assert vector_store._collection.count() == len(chunks)

        results = vector_store.similarity_search("climate change science", k=2)
        assert results
        assert "Tweet 1:" in format_docs(results)

    print("smoke test ok")
    print(f"documents={len(documents)} chunks={len(chunks)}")


if __name__ == "__main__":
    main()

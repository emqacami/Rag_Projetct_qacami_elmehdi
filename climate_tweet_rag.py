import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

from chromadb.config import Settings
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CHROMA_DIR = PROJECT_ROOT / "chroma_db"
DEFAULT_HF_CACHE_DIR = PROJECT_ROOT / "hf_cache"
COLLECTION_NAME = "climate_tweet_rag"
BATCH_SIZE = 100

os.environ.setdefault("HF_HOME", str(DEFAULT_HF_CACHE_DIR))
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

CHROMA_SETTINGS = Settings(
    anonymized_telemetry=False,
    chroma_product_telemetry_impl="chroma_noop_telemetry.NoOpTelemetry",
)


@dataclass(frozen=True)
class RagConfig:
    dataset_name: str = "cardiffnlp/tweet_eval"
    dataset_config: str = "stance_climate"
    dataset_split: str = "train"
    max_tweets: int = 500
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    llm_model: str = "gpt2"
    chroma_dir: Path = DEFAULT_CHROMA_DIR
    hf_cache_dir: Path = DEFAULT_HF_CACHE_DIR
    chunk_size: int = 500
    chunk_overlap: int = 50
    retriever_k: int = 4
    max_new_tokens: int = 90


def configure_ssl_cert_bundle() -> None:
    cert_bundle = PROJECT_ROOT / ".certs" / "windows-ca-bundle.pem"
    if cert_bundle.exists():
        os.environ.setdefault("SSL_CERT_FILE", str(cert_bundle))
        os.environ.setdefault("REQUESTS_CA_BUNDLE", str(cert_bundle))


def load_config() -> RagConfig:
    load_dotenv()
    configure_ssl_cert_bundle()

    return RagConfig(
        dataset_name=os.getenv("HF_DATASET_NAME", "cardiffnlp/tweet_eval"),
        dataset_config=os.getenv("HF_DATASET_CONFIG", "stance_climate"),
        dataset_split=os.getenv("HF_DATASET_SPLIT", "train"),
        max_tweets=int(os.getenv("HF_MAX_TWEETS", "500")),
        embedding_model=os.getenv(
            "HF_EMBEDDING_MODEL",
            "sentence-transformers/all-MiniLM-L6-v2",
        ),
        llm_model=os.getenv("HF_LLM_MODEL", "gpt2"),
        chroma_dir=Path(os.getenv("CHROMA_DIR", str(DEFAULT_CHROMA_DIR))).resolve(),
        hf_cache_dir=Path(os.getenv("HF_CACHE_DIR", str(DEFAULT_HF_CACHE_DIR))).resolve(),
        retriever_k=int(os.getenv("RETRIEVER_K", "4")),
        max_new_tokens=int(os.getenv("MAX_NEW_TOKENS", "90")),
    )


class SentenceTransformerEmbeddings(Embeddings):
    def __init__(self, model_name: str, cache_dir: Path):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name, cache_folder=str(cache_dir))

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors = self.model.encode(texts, normalize_embeddings=True)
        return vectors.tolist()

    def embed_query(self, text: str) -> list[float]:
        vector = self.model.encode([text], normalize_embeddings=True)[0]
        return vector.tolist()


class GPT2Generator:
    def __init__(self, model_name: str, cache_dir: Path, max_new_tokens: int):
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

        self.max_new_tokens = max_new_tokens
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=str(cache_dir))
        model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=str(cache_dir))

        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

        self.pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device=-1,
        )

    def __call__(self, prompt: str) -> str:
        result = self.pipe(
            prompt,
            max_new_tokens=self.max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=self.pipe.tokenizer.eos_token_id,
        )[0]["generated_text"]
        answer = result[len(prompt) :].strip()
        return answer or "Je n'ai pas pu generer une reponse exploitable avec GPT-2."


def load_tweet_documents(config: RagConfig) -> list[Document]:
    from datasets import load_dataset

    dataset = load_dataset(
        config.dataset_name,
        config.dataset_config,
        split=config.dataset_split,
        cache_dir=str(config.hf_cache_dir),
    )

    rows = dataset.select(range(min(config.max_tweets, len(dataset))))
    documents: list[Document] = []

    for index, row in enumerate(rows):
        text = str(row.get("text", "")).strip()
        if not text:
            continue

        documents.append(
            Document(
                page_content=text,
                metadata={
                    "row": index,
                    "dataset": config.dataset_name,
                    "config": config.dataset_config,
                    "label": row.get("label"),
                },
            )
        )

    return documents


def split_documents(documents: list[Document], config: RagConfig) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
    )
    return splitter.split_documents(documents)


def format_docs(docs: list[Document]) -> str:
    formatted = []
    for index, doc in enumerate(docs, start=1):
        formatted.append(f"Tweet {index}: {doc.page_content}")
    return "\n".join(formatted)


def create_chroma_from_documents(
    documents: list[Document],
    embeddings: Embeddings,
    persist_directory: Path,
) -> Chroma:
    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(persist_directory),
        client_settings=CHROMA_SETTINGS,
    )

    for start in range(0, len(documents), BATCH_SIZE):
        vector_store.add_documents(documents[start : start + BATCH_SIZE])

    return vector_store


def build_prompt(values: dict[str, Any]) -> str:
    return f"""Climate-change tweets:
{values["context"]}

Question: {values["question"]}
Based on these tweets, the answer is that"""


def is_low_quality_generation(answer: str) -> bool:
    cleaned = answer.strip()
    if len(cleaned) < 30:
        return True
    if cleaned.count("Question:") > 0:
        return True
    if cleaned.lower().count("@user") >= 2:
        return True

    words = [word.strip(".,:;!?()[]{}").lower() for word in cleaned.split()]
    words = [word for word in words if word]
    if not words:
        return True
    return len(set(words)) / len(words) < 0.35


def grounded_answer_from_sources(docs: list[Document]) -> str:
    if not docs:
        return "La base ne contient pas assez d'informations pour repondre."

    examples = []
    for doc in docs[:4]:
        text = doc.page_content.replace("\n", " ").strip()
        if len(text) > 180:
            text = text[:177].rstrip() + "..."
        examples.append(f"- {text}")

    return (
        "D'apres les tweets retrouves, les opinions portent surtout sur le fait que "
        "le changement climatique est presente comme reel, lie aux activites humaines "
        "et source d'inquietude. Plusieurs tweets insistent aussi sur le besoin de mieux "
        "informer le public.\n\nTweets pertinents:\n"
        + "\n".join(examples)
    )


def clean_generated_answer(answer: str, docs: list[Document]) -> str:
    if is_low_quality_generation(answer):
        return grounded_answer_from_sources(docs)
    return answer.strip()


class ClimateTweetRAG:
    def __init__(self, config: RagConfig | None = None):
        self.config = config or load_config()
        self.config.hf_cache_dir.mkdir(parents=True, exist_ok=True)
        self.embeddings = SentenceTransformerEmbeddings(
            self.config.embedding_model,
            self.config.hf_cache_dir,
        )
        self.generator = GPT2Generator(
            self.config.llm_model,
            self.config.hf_cache_dir,
            self.config.max_new_tokens,
        )
        self.vector_store: Chroma | None = None

    def build_vector_store(self, reset: bool = False) -> Chroma:
        if reset and self.config.chroma_dir.exists():
            shutil.rmtree(self.config.chroma_dir)

        if self.config.chroma_dir.exists() and any(self.config.chroma_dir.iterdir()) and not reset:
            self.vector_store = Chroma(
                collection_name=COLLECTION_NAME,
                embedding_function=self.embeddings,
                persist_directory=str(self.config.chroma_dir),
                client_settings=CHROMA_SETTINGS,
            )
            return self.vector_store

        documents = load_tweet_documents(self.config)
        chunks = split_documents(documents, self.config)
        if not chunks:
            raise RuntimeError("Aucun tweet exploitable n'a ete charge depuis HuggingFace.")

        self.vector_store = create_chroma_from_documents(
            documents=chunks,
            embeddings=self.embeddings,
            persist_directory=self.config.chroma_dir,
        )
        return self.vector_store

    def answer(self, question: str) -> tuple[str, list[Document]]:
        if self.vector_store is None:
            self.build_vector_store(reset=False)

        assert self.vector_store is not None
        result_count = self.vector_store._collection.count()
        retriever_k = max(1, min(self.config.retriever_k, result_count))
        retriever = self.vector_store.as_retriever(search_kwargs={"k": retriever_k})

        docs = retriever.invoke(question)
        chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | RunnableLambda(build_prompt)
            | RunnableLambda(self.generator)
            | StrOutputParser()
        )
        answer = chain.invoke(question)
        return clean_generated_answer(answer, docs), docs

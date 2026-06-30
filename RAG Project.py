import argparse
import sys

from climate_tweet_rag import ClimateTweetRAG


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Questionner une base HuggingFace de tweets climat avec LangChain, Chroma et GPT-2."
    )
    parser.add_argument(
        "question",
        nargs="?",
        default="What are people saying about climate change?",
        help="Question a poser a la base de tweets.",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Reconstruire la base vectorielle Chroma depuis HuggingFace.",
    )
    parser.add_argument(
        "--show-sources",
        action="store_true",
        help="Afficher les tweets retrouves par le retriever.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rag = ClimateTweetRAG()
    rag.build_vector_store(reset=args.reset)
    answer, sources = rag.answer(args.question)

    print(answer)

    if args.show_sources:
        print("\nSources retrouvees:")
        for index, doc in enumerate(sources, start=1):
            print(f"\n[{index}] {doc.page_content}")


if __name__ == "__main__":
    try:
        main()
    except (ImportError, ModuleNotFoundError) as exc:
        print(f"Dependance manquante: {exc}", file=sys.stderr)
        print("Installe les dependances avec: python -m pip install -r requirements.txt", file=sys.stderr)
        raise SystemExit(1)
    except Exception as exc:
        print(f"Erreur: {exc}", file=sys.stderr)
        raise SystemExit(1)

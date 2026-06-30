import gradio as gr

from climate_tweet_rag import ClimateTweetRAG, load_config


rag_instance: ClimateTweetRAG | None = None


def get_rag(reset: bool = False) -> ClimateTweetRAG:
    global rag_instance
    if rag_instance is None or reset:
        rag_instance = ClimateTweetRAG(load_config())
        rag_instance.build_vector_store(reset=reset)
    return rag_instance


def ask(question: str, reset_index: bool) -> tuple[str, str]:
    question = question.strip()
    if not question:
        return "Pose une question pour interroger la base de tweets.", ""

    rag = get_rag(reset=reset_index)
    answer, sources = rag.answer(question)

    source_text = "\n\n".join(
        f"Source {index}\n{doc.page_content}" for index, doc in enumerate(sources, start=1)
    )
    return answer, source_text


with gr.Blocks(title="RAG Tweets Climat") as demo:
    gr.Markdown("# RAG sur tweets climat avec GPT-2")
    gr.Markdown(
        "Pose une question sur la base HuggingFace `cardiffnlp/tweet_eval/stance_climate`. "
        "LangChain recupere les tweets pertinents, puis GPT-2 genere une reponse."
    )

    with gr.Row():
        question = gr.Textbox(
            label="Question",
            value="What opinions appear in the tweets about climate change?",
            lines=3,
        )

    reset_index = gr.Checkbox(
        label="Reconstruire l'index Chroma avant de repondre",
        value=False,
    )
    submit = gr.Button("Questionner le modele", variant="primary")

    answer = gr.Textbox(label="Reponse GPT-2", lines=8)
    sources = gr.Textbox(label="Tweets retrouves", lines=10)

    submit.click(
        fn=ask,
        inputs=[question, reset_index],
        outputs=[answer, sources],
    )


if __name__ == "__main__":
    demo.launch()

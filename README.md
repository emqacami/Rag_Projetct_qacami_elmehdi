# RAG sur tweets climat avec HuggingFace, LangChain, GPT-2 et Gradio

Objectif : disposer d'un LLM capable de repondre a des questions sur une base de donnees textuelle specifique.

Ce projet utilise :

- HuggingFace Datasets pour recuperer une base de tweets sur le changement climatique : `cardiffnlp/tweet_eval`, configuration `stance_climate`.
- HuggingFace Transformers pour charger le LLM `gpt2`.
- LangChain pour relier la question, la recherche dans la base vectorielle et la generation de reponse.
- Chroma pour stocker localement les embeddings des tweets.
- Gradio pour fournir une interface graphique simple.

## Installation

Active l'environnement virtuel existant :

```powershell
.\.venv\Scripts\Activate.ps1
```

Installe les dependances :

```powershell
python -m pip install -r requirements.txt
```

Copie le fichier d'exemple de configuration :

```powershell
Copy-Item .env.example .env
```

Les valeurs par defaut suffisent pour lancer le projet. Tu peux modifier `.env` pour changer le nombre de tweets, le modele d'embedding, le modele LLM ou les dossiers de cache.

## Lancer en ligne de commande

Reconstruire l'index Chroma et poser une question :

```powershell
python ".\RAG Project.py" --reset "What are people saying about climate change?"
```

Afficher aussi les tweets retrouves par le retriever :

```powershell
python ".\RAG Project.py" --show-sources "Are the tweets worried about climate change?"
```

## Lancer l'interface Gradio

```powershell
python app.py
```

Gradio affichera une URL locale, generalement :

```text
http://127.0.0.1:7860
```

## Tester le projet

Test rapide local, sans appeler HuggingFace :

```powershell
python tests\smoke_rag_pipeline.py
```

Test complet avec HuggingFace, GPT-2 et Chroma :

```powershell
python ".\RAG Project.py" --reset "What opinions appear in the tweets about climate change?"
```

## Notes

GPT-2 est un modele ancien et non instruction-tune. Il sert ici a respecter le contexte du projet, mais ses reponses peuvent etre moins fiables qu'un modele plus recent.

Les dossiers `hf_cache/` et `chroma_db/` sont generes localement et ignores par git.

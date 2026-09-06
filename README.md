# Parole Locale

Application web locale de transcription automatique de la parole (ASR/STT) et de traduction. Elle ne fait pas de TTS : le microphone est transcrit, puis le texte final est traduit.

Tout le traitement tourne sur le Mac : MLX/Metal pour Whisper et Ollama pour TranslateGemma. Aucune cle API, aucun service cloud payant et aucun audio transmis a un service distant.

## Requirements

- Mac Apple Silicon sous macOS 14 ou plus recent. L application est optimisee pour un MacBook Pro M5 Pro avec 64 Go de memoire unifiee.
- Python 3.10+ natif ARM64. Le projet a ete verifie avec Python 3.13.
- Node.js 20.19+.
- [Ollama](https://ollama.com/download) pour macOS.

Il n y a pas de dependance systeme audio supplementaire : le navigateur envoie du PCM mono 16 kHz directement au backend.

## Installation

Depuis un clone propre :

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -r server/requirements.txt

npm ci

ollama pull translategemma:4b
# Optionnel : meilleure qualite, davantage de memoire et de latence.
ollama pull translategemma:12b
```

Le modele ASR par defaut est `mlx-community/whisper-large-v3-turbo`. Il est telecharge localement par `mlx-whisper` au premier lancement du backend et conserve dans le cache Hugging Face. Aucun token Hugging Face n est necessaire pour ce modele public.

## Models

### ASR

L interface propose deux modeles MLX compatibles avec les langues francaises, anglaises et japonaises :

| Modele | Usage |
| --- | --- |
| `small` | Benchmark ou latence minimale |
| `large-v3-turbo` | Defaut, meilleure qualite multilingue |

Whisper detecte automatiquement la langue pour chaque segment. Il n y a aucune langue source globale forcee, donc une conversation peut passer du francais a l anglais puis au japonais.

### Traduction

| Preset | Modele par defaut | Usage |
| --- | --- | --- |
| `FAST` | `translategemma:4b` | Defaut, faible latence |
| `QUALITY` | `translategemma:12b` | Qualite maximale, plus lourd |
| `Qwen 3.5 MLX` | `qwen3.5:0.8b-mlx` | Latence minimale, qualite de traduction inferieure |
| `Qwen 3` | `qwen3:0.6b` | Latence minimale, qualite de traduction inferieure |

TranslateGemma est appele uniquement une fois un segment ASR finalise. La langue source est affichee directement : aucune traduction inutile `fr -> fr`, `en -> en` ou `ja -> ja` n est envoyee a Ollama.

## Development

Lancer Ollama si le service n est pas deja demarre :

```sh
ollama serve
```

Dans un premier terminal :

```sh
source .venv/bin/activate
uvicorn server.app:app --reload
```

Dans un second terminal :

```sh
npm run dev
```

Ouvrir l URL Vite, habituellement `http://localhost:5173`, puis autoriser le microphone. Le proxy Vite transmet `/ws/transcribe` au backend FastAPI.

## Build Production

```sh
npm run build
source .venv/bin/activate
uvicorn server.app:app
```

Le backend sert alors `dist/` sur `http://127.0.0.1:8000`.

## Configuration

Les valeurs par defaut sont adaptees au Mac Apple Silicon cible.

| Variable | Defaut | Usage |
| --- | --- | --- |
| `ASR_BACKEND` | `mlx` | Backend ASR local. Seul MLX est active dans cette version. |
| `WHISPER_MODEL` | `large-v3-turbo` | Modele ASR charge au demarrage. |
| `OLLAMA_MODEL` | `translategemma:4b` | Modele du preset FAST. |
| `OLLAMA_QUALITY_MODEL` | `translategemma:12b` | Modele du preset QUALITY. |
| `OLLAMA_QWEN_MLX_MODEL` | `qwen3.5:0.8b-mlx` | Modele du preset Qwen 3.5 MLX. |
| `OLLAMA_QWEN_MODEL` | `qwen3:0.6b` | Modele du preset Qwen 3. |
| `TRANSLATION_CONCURRENCY` | `1` | Requetes Ollama simultanees. Garder `1` reduit la contention sur un seul modele local. |
| `OLLAMA_URL` | `http://127.0.0.1:11434` | URL du serveur Ollama local. |
| `LOG_LEVEL` | `INFO` | Niveau des logs de latence. |

Exemple :

```sh
WHISPER_MODEL=small OLLAMA_MODEL=translategemma:4b uvicorn server.app:app --reload
```

## Architecture

```text
Microphone getUserMedia
  -> AudioWorklet (mono PCM 16 kHz)
  -> WebSocket
  -> VAD WebRTC (trames 20 ms, pauses de 500 ms)
  -> Whisper Large v3 Turbo / MLX / Metal
  -> transcript.partial ou transcript.final
  -> TranslateGemma / Ollama seulement pour les finals
  -> WebSocket
  -> UI
```

Les trames audio sont decoupees par VAD, avec 300 ms de pre-roll. Un segment continu est borne a huit secondes et conserve un overlap court avec deduplication exacte pour limiter la latence sans couper les mots. La file ASR par client est bornee a trois travaux : les partials excedentaires sont abandonnes, les finals appliquent du backpressure plutot que de faire croitre la memoire.

## Messages WebSocket

```json
{"type":"transcript.partial","segmentId":"3","text":"I want to go","language":"en","isFinal":false}
{"type":"transcript.final","segmentId":"3","text":"I want to go to Japan.","language":"en","isFinal":true}
{"type":"translation","segmentId":"3","sourceLanguage":"en","targetLanguage":"ja","text":"日本に行きたいです。","isFinal":true}
```

Les translations gardent le meme `segmentId`; le frontend met donc a jour le segment au lieu d ajouter une ligne a chaque partial.

## Metal Et Latence

Verifier MLX/Metal :

```sh
source .venv/bin/activate
python -c "import mlx.core as mx; print('Metal:', mx.metal.is_available()); print('Device:', mx.default_device())"
```

Le resultat attendu est `Metal: True` et un peripherique `gpu`. Au demarrage, le backend logge par exemple :

```text
ASR backend=mlx model=large-v3-turbo warmed in 4.21s
ASR segment=3 audio=2.30s processing=0.31s RTF=0.13 language=ja final=True
Translation segment=3 target=fr model=qwen3:0.6b queue=0.00s request=0.06s ollama_total=0.06s load=0.00s prompt=0.02s generation=0.03s tokens=24/7
End-to-end segment=3 target=fr latency=0.61s
```

`RTF` est le rapport temps de traitement / duree audio; inferieur a 1 signifie que l ASR est plus rapide que le temps reel. Les logs de traduction separent l attente de semaphore (`queue`), le temps HTTP (`request`), le chargement du modele (`load`), l evaluation du prompt (`prompt`) et la generation (`generation`). Ces mesures identifient si le goulot est l ASR, Ollama a froid, les tokens generes ou une concurrence excessive.

## Benchmark ASR

Le benchmark compare les deux modeles sur un meme WAV PCM16 mono 16 kHz :

```sh
source .venv/bin/activate
python -m server.benchmark chemin/vers/extrait-16khz-mono.wav
```

Il affiche la duree audio, le temps de transcription, le RTF et la langue detectee. L abstraction `ASREngine` dans `server/asr.py` isole le backend actuel `WhisperMLXEngine`; une future implementation `Qwen3ASREngine` peut reutiliser la meme pipeline VAD/WebSocket sans modifier le reste du backend.

## Tests

```sh
source .venv/bin/activate
python server/app.py --check
python -m unittest server.test_pipeline
npm run build
```

Les tests couvrent les langues cibles multiples, l absence de traduction vers la source, le changement de langue entre segments, les messages partial/final, le VAD, la deduplication, la configuration ASR et une erreur Ollama.

## Troubleshooting

| Probleme | Resolution |
| --- | --- |
| Microphone non autorise | Autoriser le navigateur a utiliser le microphone, puis recharger la page. |
| Ollama absent | Installer Ollama puis lancer `ollama serve`. |
| Modele Ollama absent | Executer `ollama pull translategemma:4b`; pour QUALITY, `ollama pull translategemma:12b`. |
| Modele Whisper absent | Laisser le premier demarrage terminer le telechargement du modele MLX; verifier la connexion reseau initiale et l espace disque. |
| MLX ou Metal indisponible | Utiliser un Python ARM64 natif sur macOS Apple Silicon, puis lancer la commande de verification Metal ci-dessus. |
| WebSocket inaccessible | Verifier que Uvicorn tourne sur le port 8000 et que Vite est lance depuis ce projet, ou utiliser directement le build sur le port 8000. |

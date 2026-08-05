# Proxy FastAPI pour Gemini

Cet ajout fournit un proxy FastAPI minimal pour centraliser les appels vers l'API Gemini et protéger la clé serveur.

Endpoints
- POST /api/gemini/generate
  - Body JSON: { "prompt": "...", "model": "gemini-2.5-flash" }
  - Réponse: { "text": "..." }

Instructions rapides (dev)
1. Copier .env.example -> .env et remplir GEMINI_API_KEY
2. Installer: pip install -r requirements.txt
3. Lancer: uvicorn server:app --reload --host 0.0.0.0 --port 8000

Remarques
- Remplacez l'URL d'appel et la logique de parsing par le SDK officiel Google GenAI si vous l'utilisez.
- N'exposez jamais GEMINI_API_KEY côté client. Stockez‑la dans les secrets de votre plateforme de déploiement.

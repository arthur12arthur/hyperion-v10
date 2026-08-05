from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import logging
import requests

app = FastAPI()
logger = logging.getLogger("uvicorn.error")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    logger.warning("GEMINI_API_KEY non défini — l'API ne fonctionnera pas en production")

class GenerateRequest(BaseModel):
    prompt: str
    model: str = "gemini-2.5-flash"

@app.post("/api/gemini/generate")
async def generate(req: GenerateRequest):
    if not GEMINI_API_KEY:
        raise HTTPException(status_code=500, detail="Clé serveur non configurée")

    # Exemple générique d'appel HTTP vers l'API provider — ADAPTÉR selon l'API réelle / SDK Google
    url = "https://api.example.com/v1/generate"  # <-- remplacer par l'endpoint réel
    headers = {
        "Authorization": f"Bearer {GEMINI_API_KEY}",
        "Content-Type": "application/json"
    }
    body = {
        "model": req.model,
        "prompt": req.prompt
    }

    try:
        resp = requests.post(url, json=body, headers=headers, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        # Adaptez l'accès au texte selon la structure de la réponse
        return {"text": data.get("text") or data}
    except requests.RequestException as e:
        logger.exception("Erreur lors de l'appel à l'API Gemini")
        raise HTTPException(status_code=502, detail="Erreur lors de l'appel au provider")

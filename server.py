from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import logging

# SDK officiel Google GenAI (google-genai) — utilisé si installé
try:
    from google_genai import GoogleGenAI
    HAS_GOOGLE_GENAI = True
except Exception:
    HAS_GOOGLE_GENAI = False

# Fallback HTTP client si le SDK n'est pas disponible
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

    # Utiliser le SDK officiel si disponible
    if HAS_GOOGLE_GENAI:
        try:
            # Initialisation du client SDK — adapte selon la version du package
            ai = GoogleGenAI(api_key=GEMINI_API_KEY)
            # Exemple d'appel via le SDK — adaptez les noms de méthodes selon la version réelle
            # Ici on tente d'appeler une méthode "models.generate_content" similaire à l'exemple TypeScript
            resp = ai.models.generate_content(model=req.model, contents=req.prompt)

            # Extraire le texte selon la structure renvoyée par le SDK
            text = None
            if isinstance(resp, dict):
                text = resp.get("text") or resp.get("output")
            else:
                # objet SDK : tenter des attributs courants
                text = getattr(resp, "text", None) or getattr(resp, "output", None)

            return {"text": text or resp}
        except Exception as e:
            logger.exception("Erreur lors de l'appel via google-genai SDK")
            raise HTTPException(status_code=502, detail="Erreur via SDK google-genai: %s" % str(e))

    # Sinon fallback HTTP (ancien comportement)
    url = "https://api.example.com/v1/generate"  # <-- remplacer par l'endpoint réel si nécessaire
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
        return {"text": data.get("text") or data}
    except requests.RequestException as e:
        logger.exception("Erreur lors de l'appel HTTP au provider")
        raise HTTPException(status_code=502, detail="Erreur lors de l'appel au provider")

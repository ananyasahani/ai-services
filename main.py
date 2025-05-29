from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from typing import List
import httpx
import os

app = FastAPI()

# Initialize BERT tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModel.from_pretrained("distilbert-base-uncased")

# Environment variables
USER_SERVICE_URL = os.getenv("USER_SERVICE_URL", "http://user-service:4000")
FREELANCE_SERVICE_URL = os.getenv("FREELANCE_SERVICE_URL", "http://freelance-service:4001")

class UserRequest(BaseModel):
    user_id: str

async def validate_user(user_id: str):
    """Validate user with user-service"""
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(f"{USER_SERVICE_URL}/auth/validate/{user_id}")
            if response.status_code != 200:
                raise HTTPException(status_code=401, detail="Invalid user")
            return response.json()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error validating user: {str(e)}")

def get_bert_embeddings(text: str) -> np.ndarray:
    """Generate BERT embeddings for a given text."""
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state[:, 0, :].numpy()
    return embeddings.flatten()

def compute_cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    return dot_product / (norm1 * norm2) if norm1 > 0 and norm2 > 0 else 0.0

@app.post("/api/recommend/gigs")
async def recommend_gigs(request: UserRequest):
    try:
        # Validate user
        await validate_user(request.user_id)

        # Fetch user profile from freelance service
        async with httpx.AsyncClient() as client:
            user_response = await client.get(f"{FREELANCE_SERVICE_URL}/api/freelancers/{request.user_id}")
            if user_response.status_code != 200:
                raise HTTPException(status_code=404, detail="User profile not found")
            user = user_response.json()["data"]

        # Fetch available gigs from freelance service
        gigs_response = await client.get(f"{FREELANCE_SERVICE_URL}/api/gigs")
        if gigs_response.status_code != 200:
            raise HTTPException(status_code=500, detail="Error fetching gigs")
        gigs = gigs_response.json()["data"]

        # Combine user skills and bio
        user_text = " ".join(user["skills"] + [user.get("bio", "")]).lower()
        user_embedding = get_bert_embeddings(user_text)

        # Compute embeddings for gigs and similarities
        similarities = []
        for gig in gigs:
            gig_text = " ".join([gig["title"], gig["description"]] + gig["skills"]).lower()
            gig_embedding = get_bert_embeddings(gig_text)
            similarity = compute_cosine_similarity(user_embedding, gig_embedding)
            similarities.append({"gig": gig, "score": similarity})

        # Sort by similarity and select top 5
        top_gigs = sorted(similarities, key=lambda x: x["score"], reverse=True)[:5]
        recommended_gigs = [item["gig"] for item in top_gigs]

        return {"success": True, "data": recommended_gigs}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"} 
import streamlit as st
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch

st.title("BERT Gig Recommendation Model Tester")

st.write("Enter a user ID to test the BERT-based gig recommendation model.")

# Initialize BERT tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModel.from_pretrained("distilbert-base-uncased")

# Dummy data from main.py
dummy_gigs = [
    {"id": "1", "title": "Build a Web App", "description": "Develop a full-stack web application using React and Node.js", "skills": ["React", "Node.js", "MongoDB"]},
    {"id": "2", "title": "Mobile App Development", "description": "Create a cross-platform mobile app with Flutter", "skills": ["Flutter", "Dart", "Firebase"]},
    {"id": "3", "title": "API Integration", "description": "Integrate third-party APIs into an existing system", "skills": ["Node.js", "REST API", "Express"]},
    {"id": "4", "title": "Data Analysis Dashboard", "description": "Build a dashboard for data visualization using Python and Dash", "skills": ["Python", "Dash", "Pandas"]},
    {"id": "5", "title": "E-commerce Platform", "description": "Develop an e-commerce website with Shopify and JavaScript", "skills": ["Shopify", "JavaScript", "HTML"]},
    {"id": "6", "title": "Machine Learning Model", "description": "Develop a predictive model using TensorFlow", "skills": ["Python", "TensorFlow", "Machine Learning"]}
]

mock_users = {
    "freelancer1": {"id": "freelancer1", "skills": ["React", "Node.js", "JavaScript"], "bio": "Experienced full-stack developer with expertise in web applications"}
}

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

# Input for user_id
user_id = st.text_input("User ID", value="freelancer1")

# Button to trigger recommendation
if st.button("Test Model"):
    if not user_id:
        st.error("Please enter a user ID.")
    else:
        try:
            # Fetch user profile
            user = mock_users.get(user_id)
            if not user:
                st.error("User not found.")
            else:
                # Combine user skills and bio
                user_text = " ".join(user["skills"] + [user["bio"]]).lower()
                user_embedding = get_bert_embeddings(user_text)

                # Compute embeddings for gigs and similarities
                similarities = []
                for gig in dummy_gigs:
                    gig_text = " ".join([gig["title"], gig["description"]] + gig["skills"]).lower()
                    gig_embedding = get_bert_embeddings(gig_text)
                    similarity = compute_cosine_similarity(user_embedding, gig_embedding)
                    similarities.append({"gig": gig, "score": similarity})

                # Sort by similarity and select top 5
                top_gigs = sorted(similarities, key=lambda x: x["score"], reverse=True)[:5]
                recommended_gigs = [item["gig"] for item in top_gigs]

                # Display results
                st.success("Model executed successfully!")
                st.subheader("Top 5 Recommended Gigs")
                for i, gig in enumerate(recommended_gigs, 1):
                    st.write(f"**Gig {i}: {gig['title']}**")
                    st.write(f"- **Description**: {gig['description']}")
                    st.write(f"- **Skills**: {', '.join(gig['skills'])}")
                    st.write(f"- **Similarity Score**: {next(item['score'] for item in similarities if item['gig']['id'] == gig['id']):.4f}")
                    st.write("---")
        except Exception as e:
            st.error(f"Error running model: {str(e)}")
# BERT Gig Recommendation Model - Streamlit App

This is a simple web-based demo application built with **Streamlit** that utilizes a **BERT-based model** to recommend freelance gigs based on a user's profile.

## 🔗 Web App

This recommendation service powers the web application developed for **Hack With Gujarat 2025**.

🚀 **Try it live:** [v0-hwg.vercel.app](https://v0-hwg.vercel.app)

## 🔍 Overview

Freelancers can enter their **User ID**, and the app will analyze their skills and bio to recommend the most relevant freelance gigs. The recommendation is powered by **DistilBERT**, a lightweight version of BERT.

## 🙅 What the Freelancer Should Enter

* **User ID**: A unique identifier representing the freelancer.

  * The app uses a mock user database, so ensure your user ID exists (e.g., `freelancer1`).
  * Each user profile includes a `skills` list and a short `bio`.

## 🧰 How It Works

1. The app uses the `transformers` library to load a pre-trained `distilbert-base-uncased` model.
2. The user's profile (skills + bio) is converted into a BERT embedding.
3. Each gig is represented by an embedding of its title, description, and required skills.
4. Cosine similarity is calculated between the user's embedding and each gig embedding.
5. The top 5 most similar gigs are displayed as recommendations.

## 📊 Example User

```json
freelancer1: {
  "id": "freelancer1",
  "skills": ["React", "Node.js", "JavaScript"],
  "bio": "Experienced full-stack developer with expertise in web applications"
}
```

## 📈 All Available Gigs

```json
[
  {
    "id": "1",
    "title": "Build a Web App",
    "description": "Develop a full-stack web application using React and Node.js",
    "skills": ["React", "Node.js", "MongoDB"]
  },
  {
    "id": "2",
    "title": "Mobile App Development",
    "description": "Create a cross-platform mobile app with Flutter",
    "skills": ["Flutter", "Dart", "Firebase"]
  },
  {
    "id": "3",
    "title": "API Integration",
    "description": "Integrate third-party APIs into an existing system",
    "skills": ["Node.js", "REST API", "Express"]
  },
  {
    "id": "4",
    "title": "Data Analysis Dashboard",
    "description": "Build a dashboard for data visualization using Python and Dash",
    "skills": ["Python", "Dash", "Pandas"]
  },
  {
    "id": "5",
    "title": "E-commerce Platform",
    "description": "Develop an e-commerce website with Shopify and JavaScript",
    "skills": ["Shopify", "JavaScript", "HTML"]
  },
  {
    "id": "6",
    "title": "Machine Learning Model",
    "description": "Develop a predictive model using TensorFlow",
    "skills": ["Python", "TensorFlow", "Machine Learning"]
  }
]
```

## 🏆 Recommended Gigs (Model Results)

```
Gig 1: Mobile App Development
Description: Create a cross-platform mobile app with Flutter
Skills: Flutter, Dart, Firebase
Similarity Score: 0.9391

Gig 2: Build a Web App
Description: Develop a full-stack web application using React and Node.js
Skills: React, Node.js, MongoDB
Similarity Score: 0.9166

Gig 3: Data Analysis Dashboard
Description: Build a dashboard for data visualization using Python and Dash
Skills: Python, Dash, Pandas
Similarity Score: 0.9166

Gig 4: E-commerce Platform
Description: Develop an e-commerce website with Shopify and JavaScript
Skills: Shopify, JavaScript, HTML
Similarity Score: 0.9136
```

## ⚙️ Dependencies

Before running the app, install the following Python packages:

```bash
pip install streamlit transformers torch numpy
```

## 🚀 How to Run

1. Save the Python script as `app.py`
2. Open a terminal and navigate to the directory containing `app.py`
3. Run the app using Streamlit:

```bash
streamlit run app.py
```

4. The browser will open the app at `http://localhost:8501`

---

## 📄 Notes

* This is a **demo** and uses hardcoded mock data for both gigs and users.
* In a production version, user data and gigs would be retrieved from a real database.
* BERT embeddings are computed dynamically; depending on hardware, performance may vary.

## ✈️ Future Improvements

* Connect to a real backend database
* Add user authentication
* Allow user to update skills and bio in-app
* Use caching to reduce re-computation of embeddings

# app.py
import os
import json
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS
from sqlalchemy import create_engine, Column, String, Float, Text, DateTime, Integer
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from textblob import TextBlob
import requests

DATABASE_URL = os.environ.get("DATABASE_URL")  # set by Railway (postgres URI)
HF_API_KEY = os.environ.get("HF_API_KEY")      # optional: HuggingFace Inference API key
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")  # optional

if not DATABASE_URL:
    raise RuntimeError("Please set DATABASE_URL environment variable")

engine = create_engine(
    DATABASE_URL.replace("postgres://", "postgresql+pg8000://"),
    connect_args={"sslmode": "require"}
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class Feedback(Base):
    __tablename__ = "feedback"
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    feedback_id = Column(String, unique=True, index=True, nullable=False)
    platform = Column(String)
    post_id = Column(String)
    text = Column(Text)
    user = Column(String)
    timestamp = Column(DateTime)
    location = Column(String, index=True)  # store_id
    category = Column(String, index=True)
    category_confidence = Column(Float)
    sentiment = Column(String, index=True)
    sentiment_score = Column(Float)
    raw_json = Column(Text)

Base.metadata.create_all(bind=engine)

# Rule based keywords
CATEGORY_KEYWORDS = {
    "Food": ["burger", "meal", "taste", "spicy", "bland", "cold", "overcooked", "delicious"],
    "Service": ["staff", "waiter", "manager", "rude", "friendly", "helpful"],
    "Speed": ["slow", "quick", "waiting", "wait", "speed", "time", "long"],
    "Ambience": ["ambience", "decor", "music", "noisy", "clean", "dirty"]
}

def map_category(text):
    txt = (text or "").lower()
    scores = {k: 0 for k in CATEGORY_KEYWORDS}
    for k, keywords in CATEGORY_KEYWORDS.items():
        for kw in keywords:
            if kw in txt:
                scores[k] += 1
    top = max(scores.items(), key=lambda x: x[1])
    if top[1] == 0:
        return "Unmapped", 0.0
    total = sum(scores.values()) or 1
    return top[0], float(top[1]) / float(total)

def analyze_sentiment_textblob(text):
    if not text:
        return "neutral", 0.0
    tb = TextBlob(text)
    score = tb.sentiment.polarity  # -1 .. 1
    if score > 0.1:
        s = "positive"
    elif score < -0.1:
        s = "negative"
    else:
        s = "neutral"
    return s, float(score)

def analyze_sentiment_hf(text):
    """Optional: call Hugging Face text-sentiment if HF_API_KEY provided.
       Returns (label, score) or fallback to None if fails."""
    if not HF_API_KEY:
        return None
    url = "https://api-inference.huggingface.co/models/sentence-transformers/distilbert-base-uncased"
    headers = {"Authorization": f"Bearer {HF_API_KEY}"}
    # Use a small sentiment model endpoint - if you have a model name, set it here.
    # For this template, we'll call a text-classification endpoint:
    url = "https://api-inference.huggingface.co/models/distilbert-base-uncased-finetuned-sst-2-english"
    try:
        resp = requests.post(url, headers=headers, json={"inputs": text}, timeout=20)
        resp.raise_for_status()
        out = resp.json()
        if isinstance(out, list) and len(out) > 0:
            label = out[0].get("label", "").lower()
            score = float(out[0].get("score", 0.0))
            return label, score
    except Exception as e:
        print("HF sentiment call failed:", e)
    return None

app = Flask(__name__)
CORS(app)  # allow cross origin for frontend

@app.route("/health")
def health():
    return jsonify({"status": "ok", "time": datetime.utcnow().isoformat() + "Z"})

@app.route("/ingest", methods=["POST"])
def ingest():
    """
    Accepts a JSON list of records:
    [
      {"feedback_id":"c1","platform":"csv","post_id":"1","text":"...","user":"john","timestamp":"2025-11-01T10:15:00Z","location":"Store_101"},
      ...
    ]
    """
    data = request.get_json()
    if not data:
        return jsonify({"error": "no json body"}), 400
    session = SessionLocal()
    ingested = 0
    for r in data:
        text = r.get("text","")
        category, cat_conf = map_category(text)
        # prefer HF or OpenAI if configured, else textblob
        sent_label, sent_score = None, None
        if HF_API_KEY:
            try:
                out = analyze_sentiment_hf(text)
                if out:
                    sent_label, sent_score = out
            except Exception as e:
                print("hf error", e)
        if sent_label is None:
            sent_label, sent_score = analyze_sentiment_textblob(text)
        # parse timestamp
        ts = None
        try:
            if r.get("timestamp"):
                ts = datetime.fromisoformat(r.get("timestamp").replace("Z", "+00:00"))
        except Exception:
            ts = None
        # store
        fb = session.query(Feedback).filter_by(feedback_id=str(r.get("feedback_id"))).first()
        if not fb:
            fb = Feedback(
                feedback_id=str(r.get("feedback_id")),
                platform=r.get("platform"),
                post_id=str(r.get("post_id")) if r.get("post_id") else None,
                text=text,
                user=r.get("user"),
                timestamp=ts,
                location=r.get("location"),
                category=category,
                category_confidence=cat_conf,
                sentiment=sent_label,
                sentiment_score=sent_score or 0.0,
                raw_json=json.dumps(r)
            )
            session.add(fb)
            ingested += 1
        else:
            # update minimal fields (idempotency)
            fb.text = text
            fb.category = category
            fb.category_confidence = cat_conf
            fb.sentiment = sent_label
            fb.sentiment_score = sent_score or 0.0
            fb.raw_json = json.dumps(r)
    session.commit()
    session.close()
    return jsonify({"status":"ok","ingested":ingested})

@app.route("/metrics", methods=["GET"])
def metrics():
    """
    /metrics?store_id=Store_101
    returns aggregated counts per category and sentiment and a recent feed
    """
    store = request.args.get("store_id")
    limit = int(request.args.get("limit", 20))
    session = SessionLocal()
    q = session.query(Feedback)
    if store:
        q = q.filter(Feedback.location == store)
    # totals per category & sentiment
    rows = q.all()
    totals = {}
    sentiment = {}
    recent = []
    # sort by timestamp desc:
    rows_sorted = sorted(rows, key=lambda r: r.timestamp or datetime.min, reverse=True)
    for r in rows_sorted:
        totals.setdefault(r.category or "Unmapped", 0)
        totals[r.category or "Unmapped"] += 1
        sentiment.setdefault(r.sentiment or "neutral", 0)
        sentiment[r.sentiment or "neutral"] += 1
    for r in rows_sorted[:limit]:
        recent.append({
            "feedback_id": r.feedback_id,
            "text": r.text,
            "category": r.category,
            "sentiment": r.sentiment,
            "timestamp": r.timestamp.isoformat() if r.timestamp else None,
            "location": r.location
        })
    session.close()
    return jsonify({"store_id": store, "totals": totals, "sentiment": sentiment, "recent": recent})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))

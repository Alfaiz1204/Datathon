from flask import Flask, render_template, request, jsonify, redirect, url_for
from flask_cors import CORS
from chat import get_response
import nltk
from nltk.data import find
try:
    find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

# ────────────────────────────── Flask setup ──────────────────────────────
app = Flask(
    __name__,
    static_folder="static",
    template_folder="templates"
)
CORS(app)

# ────────────────────────────── ROUTES UI ────────────────────────────────
@app.get("/")
def home():
    """Landing‑page."""
    return render_template("home.html")

@app.get("/chat")
def chat_page():
    return render_template("chatbot.html")

@app.get("/chatbot.html")
def chat_legacy():
    return redirect(url_for("chat_page"), code=301)

# ────────────────────────  API  /predict  ───────────────────────────────
@app.post("/predict")
def predict():
    """
    Body JSON  : {"message": "<teks user>"}
    Response   : {"answer": "<balasan bot>"}
    """
    data = request.get_json(silent=True) or {}
    user_msg = str(data.get("message", "")).strip()

    if not user_msg:
        return jsonify({"answer": "Silakan ketik sesuatu."}), 200

    bot_reply = get_response(user_msg)

    if isinstance(bot_reply, list):
        bot_reply = bot_reply[1] if len(bot_reply) > 1 else bot_reply[0]

    return jsonify({"answer": bot_reply})

# ─────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

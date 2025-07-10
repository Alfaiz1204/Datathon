"""
Chatbot kesehatan:
• Ketik gejala satu‑satu (boleh kalimat penuh), lalu ketik 'done' untuk memproses.
• Model prediksi  =  SVC (models/svc_model.joblib)
• Menampilkan description & precaution (di‑translate) setelah diagnosis.
"""

import json, math, re, csv
from pathlib import Path
from typing import List, Dict
import pandas as pd

import geocoder, joblib, numpy as np
from nltk_utils import tokenize           # tokenizer dasar (spasi & tanda baca)
from deep_translator import GoogleTranslator
from difflib import get_close_matches          # ← NEW
from sentence_transformers import SentenceTransformer, models
import torch
from transformers import BertTokenizer, AutoModel

# ─────────────────── manual translation (ringkas) ───────────────────
manual_translation_en_to_id: Dict[str, str] = {}
manual_translation_id_to_en: Dict[str, str] = {}

def _alias_keys(txt: str) -> List[str]:
    base = txt.strip().lower()
    return list({base, base.replace('_',' '), base.replace(' ','_')})

def add_manual_translation(en_text: str, id_text: str):
    for k in _alias_keys(en_text):
        manual_translation_en_to_id[k] = id_text.strip()
    for k in _alias_keys(id_text):
        manual_translation_id_to_en[k] = en_text.strip()

# ✍️ — tambahkan sebagian mapping penting (bisa diperluas)
add_manual_translation("muscle_wasting", "penyusutan otot")
add_manual_translation("burning_micturition", "rasa terbakar saat buang air kecil")
add_manual_translation("ulcers_on_tongue", "luka pada lidah")
add_manual_translation("spotting_urination", "bercak darah saat buang air kecil")        # ← typo fixed
add_manual_translation("cold_hands_and_feets", "tangan dan kaki terasa dingin")
add_manual_translation("patches_in_throat", "bercak di tenggorokan")
add_manual_translation("yellowing_of_eyes", "mata menguning")
add_manual_translation("malaise", "rasa tidak enak badan")
add_manual_translation("redness_of_eyes", "kemerahan pada mata")
add_manual_translation("weakness_in_limbs", "kelemahan pada anggota tubuh")
add_manual_translation("bloody_stool", "tinja berdarah")
add_manual_translation("enlarged_thyroid", "pembesaran kelenjar tiroid")
add_manual_translation("swollen_extremeties", "pembengkakan pada tungkai")
add_manual_translation("extra_marital_contacts", "kontak seksual di luar pernikahan")
add_manual_translation("slurred_speech", "bicara pelo")
add_manual_translation("spinning_movements", "gerakan berputar")
add_manual_translation("loss_of_balance", "kehilangan keseimbangan")
add_manual_translation("loss_of_smell", "kehilangan indera penciuman")
add_manual_translation("continuous_feel_of_urine", "terus-menerus merasa ingin buang air kecil")
add_manual_translation("passage_of_gases", "buang gas (kentut)")
add_manual_translation("toxic_look_(typhos)", "penampakan toksik (demam tifoid)")
add_manual_translation("irritability", "mudah tersinggung")
add_manual_translation("altered_sensorium", "perubahan kesadaran")
add_manual_translation("dischromic _patches", "bercak perubahan warna kulit")
add_manual_translation("watering_from_eyes", "keluar air mata terus-menerus")
add_manual_translation("mucoid_sputum", "dahak berlendir")
add_manual_translation("rusty_sputum", "dahak berwarna karat")
add_manual_translation("prominent_veins_on_calf", "pembuluh darah menonjol pada betis")
add_manual_translation("pus_filled_pimples", "jerawat berisi nanah")
add_manual_translation("blackheads", "komedo hitam")
add_manual_translation("scurring", "keropeng")
add_manual_translation("silver_like_dusting", "sisik seperti debu perak")
add_manual_translation("inflammatory_nails", "peradangan pada kuku")
add_manual_translation("red_sore_around_nose", "luka merah di sekitar hidung")
add_manual_translation("yellow_crust_ooze", "kerak kuning yang mengeluarkan cairan")
add_manual_translation("diziness", "pusing")
add_manual_translation("Hypertension", "Hipertensi")

# ───────── helper translate ─────────
def to_en(text: str) -> str:
    for k in _alias_keys(text.lower()):
        if k in manual_translation_id_to_en:
            return manual_translation_id_to_en[k]
    try:
        return GoogleTranslator(source="auto", target="en").translate(text)
    except Exception:
        return text

def to_id(text: str) -> str:
    for k in _alias_keys(text.lower()):
        if k in manual_translation_en_to_id:
            return manual_translation_en_to_id[k]
    try:
        # terjemahkan per kalimat (lebih stabil)
        parts = re.split(r'(?<=[.!?]) +', text.strip())
        return ' '.join(GoogleTranslator(source="en", target="id").translate(p) for p in parts if p)
    except Exception:
        return text

# ───────────────────── konfigurasi dasar ─────────────────────
BASE       = Path(__file__).resolve().parent
MODEL_DIR  = BASE / "models"
MASTER_DIR = BASE / "MasterData"
MIN_SYMPT  = 3

COLUMNS    = json.loads((MODEL_DIR / "columns.json").read_text())
MODEL      = joblib.load(MODEL_DIR / "svc_model.joblib")
ENCODER    = joblib.load(MODEL_DIR / "label_encoder.joblib")

VALID_SYMPTOMS = set(COLUMNS)                   # set cepat pencocokan

# ────────── regex pattern each symptom (space form) ──────────
SYMPTOM_PATTERNS = {
    s: re.compile(r'\b' + re.escape(s.replace('_', ' ')) + r'\b')
    for s in VALID_SYMPTOMS
}

# ────────── load description & precaution CSV ──────────
description_list: Dict[str,str] = {}
precaution_dict : Dict[str,List[str]] = {}
obat_list: Dict[str,str] = {}

desc_csv = MASTER_DIR / "symptom_Description.csv"
prec_csv = MASTER_DIR / "symptom_precaution.csv"
obat_xlsx = MASTER_DIR / "obat.xlsx"

if desc_csv.exists():
    with desc_csv.open(encoding="utf-8") as f:
        for dis, desc in csv.reader(f):
            description_list[dis.strip()] = desc.strip()

if obat_xlsx.exists():
    df_obat = pd.read_excel(obat_xlsx)

for _, row in df_obat.iterrows():
    disease = str(row["Nama Penyakit"]).strip().lower()
    obat_list.setdefault(disease, []).append({
        "drug"   : str(row["Nama Obat"]).strip(),
        "age"    : str(row.get("Usia","")).strip(),
        "dosage"   : str(row["Takaran"]).strip(),
        "content": str(row["Kandungan Obat"]).strip()
    })


if prec_csv.exists():
    with prec_csv.open(encoding="utf-8") as f:
        for row in csv.reader(f):
            if len(row) >= 5:
                precaution_dict[row[0].strip()] = [c.strip() for c in row[1:5]]

# ────────── memori sesi ──────────
symptom_session: List[str] = []

# status percakapan: 'ask_name' → 'ask_age' → 'collect_symptoms'
bot_state = "ask_name"
user_profile = {"name": None, "age": None}

try:
    # 1) Ambil encoder BERT mentah
    bert = models.Transformer("indobenchmark/indobert-base-p1")     # dimensi 768
    # 2) Tambahkan pooling 'mean' (agar bisa jadi embedding kalimat)
    pooling = models.Pooling(
        bert.get_word_embedding_dimension(),
        pooling_mode_mean_tokens=True,
        pooling_mode_cls_token=False,
        pooling_mode_max_tokens=False,
    )
    # 3) Gabungkan menjadi satu pipeline SentenceTransformer
    indo_bert_model = SentenceTransformer(modules=[bert, pooling])
except Exception as e:
    # Fallback multilingual yang sudah siap pakai bila download gagal
    print("⚠️  IndoBERT tidak tersedia, fallback ke MiniLM.\n", e)
    indo_bert_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

# ────────── fungsi util utama ──────────
def extract_symptoms(sentence: str) -> List[str]:
    """
    Terjemahkan kalimat → bagi ke dalam n-gram → cocokkan:
    1. Dengan semantic similarity IndoBERT (paling utama).
    2. Dengan regex (exact match).
    3. Jika tidak ketemu, fallback ke fuzzy match.
    """
    from difflib import get_close_matches

    sent = to_en(sentence).lower()
    sent = re.sub(r"[^a-z0-9 ]+", " ", sent)
    sent = re.sub(r"\s+", " ", sent).strip()

    tokens = sent.split()
    ngrams = []

    # Buat n-gram dari 1 sampai 4 kata (sliding window)
    for n in range(1, 5):
        ngrams += [' '.join(tokens[i:i+n]) for i in range(len(tokens)-n+1)]

    found = set()

    # --- IndoBERT semantic matching ---
    # Siapkan daftar gejala (dalam bentuk string, spasi, bukan underscore)
    symptom_texts = [s.replace('_', ' ') for s in VALID_SYMPTOMS]
    # Encode semua gejala hanya sekali (cache di global jika perlu)
    symptom_embeds = indo_bert_model.encode(symptom_texts, convert_to_tensor=True)

    for phrase in ngrams:
        # Encode phrase
        phrase_embed = indo_bert_model.encode(phrase, convert_to_tensor=True)
        # Hitung similarity ke semua gejala
        cos_scores = torch.nn.functional.cosine_similarity(phrase_embed, symptom_embeds)
        # Ambil index dengan similarity tertinggi
        best_idx = torch.argmax(cos_scores).item()
        best_score = cos_scores[best_idx].item()
        if best_score > 0.80:  # threshold bisa diatur
            found.add(symptom_texts[best_idx].replace(' ', '_'))
            continue
        # Coba exact match dengan regex
        for sym, pat in SYMPTOM_PATTERNS.items():
            if pat.search(phrase):
                found.add(sym)
                break
        else:
            # Coba fuzzy match jika regex gagal
            close = get_close_matches(phrase, [s.replace("_", " ") for s in VALID_SYMPTOMS], n=1, cutoff=0.75)
            if close:
                found.add(close[0].replace(" ", "_"))

    return list(found)



def classify(symptoms: List[str]) -> str:
    vec = np.zeros(len(COLUMNS), dtype=int)
    for s in symptoms:
        if s in VALID_SYMPTOMS:
            vec[COLUMNS.index(s)] = 1
    idx = MODEL.predict([vec])[0]
    return ENCODER.inverse_transform([idx])[0]

def centres():
    def hav(lat1,lon1,lat2,lon2):
        R=6371; lat1,lon1,lat2,lon2=map(math.radians,[lat1,lon1,lat2,lon2])
        a=(math.sin((lat2-lat1)/2)**2+
           math.cos(lat1)*math.cos(lat2)*math.sin((lon2-lon1)/2)**2)
        return R*2*math.atan2(math.sqrt(a),math.sqrt(1-a))
    loc = geocoder.ip('me').latlng or [0.0,0.0]
    meds = json.load(open("medical_centers.json",encoding="utf-8"))["intents"]
    nearest = sorted([(m,hav(loc[0],loc[1],*m["location"])) for m in meds], key=lambda x:x[1])[:5]
    return ["center"] + [[m["tag"], f"{d:.2f} km", m["Address"]] for m,d in nearest]

# ────────── inti percakapan ──────────
def classify_top3(symptoms: List[str]) -> List[tuple]:
    """
    Mengembalikan 3 prediksi penyakit teratas dalam bentuk:
    [(nama_penyakit, probabilitas float 0‑1), ...]
    """
    vec = np.zeros(len(COLUMNS), dtype=int)
    for s in symptoms:
        if s in VALID_SYMPTOMS:
            vec[COLUMNS.index(s)] = 1
    probs = MODEL.predict_proba([vec])[0]
    top3_idx = np.argsort(probs)[::-1][:3]  # indeks dari probabilitas tertinggi
    top3 = [(ENCODER.inverse_transform([i])[0], probs[i]) for i in top3_idx]
    return top3

def get_response(user_msg: str):
    global bot_state, user_profile

    # ── Tahap 1: minta NAMA ───────────────────────────────────────────────
    if bot_state == "ask_name":
        # Jika pengguna belum memberikan nama, minta nama
        if not user_profile["name"]:
            # Asumsikan nama adalah kata alfabet pertama yg bukan stop‑word
            token_name = next((t for t in tokenize(user_msg) if t.isalpha()), None)
            if token_name:                           # nama ditemukan
                user_profile["name"] = token_name.capitalize()
                bot_state = "ask_age"
                return f"Hai {user_profile['name']}! Berapa usia Anda?"
            else:
                return "Boleh saya tahu nama Anda?"

    # ── Tahap 2: minta USIA ───────────────────────────────────────────────
    if bot_state == "ask_age":
        number = re.search(r'\d+', user_msg)
        if number:
            user_profile["age"] = int(number.group())
            bot_state = "collect_symptoms"
            return ("Terima kasih. Silakan ketik gejala Anda satu‑per‑satu "
                    "(boleh kalimat penuh), lalu ketik *done* jika selesai.")
        else:
            return "Berapa usia Anda (angka)?"

    # ── Mulai tahap pengumpulan gejala ────────────────────────────────────

    global symptom_session
    lowered = user_msg.strip().lower()

    # pusat medis
    tokens = tokenize(user_msg)
    if {"medical","hospital","hospitals","center"} & {t.lower() for t in tokens}:
        return centres()

    # kumpulkan gejala (jika bukan 'done')
    if lowered != "done":
        new_syms = extract_symptoms(user_msg)
        if new_syms:
            for s in new_syms:
                if s not in symptom_session:
                    symptom_session.append(s)
        else:
            return ["unknown_symptom",
                    "Maaf, saya tidak mengenali gejala pada kalimat tersebut. "
                    "Silakan sebutkan gejala lain atau ketik *done* jika selesai."]

        return ["collect",
                f"Noted: {', '.join(symptom_session)}. "
                f"Tambahkan gejala lain atau ketik *done* jika selesai."]

    # user mengetik 'done'
    if len(symptom_session) < MIN_SYMPT:
        return ["need_more",
                f"Anda baru memasukkan {len(symptom_session)} gejala. "
                f"Silakan tambahkan setidaknya {MIN_SYMPT-len(symptom_session)} lagi."]

    top3 = classify_top3(symptom_session)
    symptom_session = []
    lines = ["Berdasarkan gejala, berikut kemungkinan penyakit Anda:\n"]

    for i, (name, prob) in enumerate(top3, start=1):
        lines.append(f"{i}. 🩺 **{to_id(name)}** ({prob*100:.1f}%)")

        if name in description_list:
            desc_id = to_id(description_list[name])
            lines.append(f"   📌 {desc_id}")

        if name in precaution_dict:
            lines.append("   🛡️ Pencegahan:")
            for p in precaution_dict[name]:
                lines.append(f"     • {to_id(p)}")

        meds = obat_list.get(name.lower()) or obat_list.get(to_id(name).lower())
        if meds:
            lines.append("   💊 Rekomendasi obat:")
            for m in meds:
                line = f"     • {m['drug']} – {m['content']} (usia: {m['age']}) {m['dosage']}"
                lines.append(to_id(line))

        lines.append("")

    lines.append("Silakan konsultasi ke tenaga medis untuk diagnosis pasti.")

    return "\n".join(lines)

# ────────── CLI demo ──────────
if __name__ == "__main__":
    print("Chatbot is ready! (type 'quit' to exit)\n")
    print("Bot: Boleh saya tahu nama Anda?")
    while True:
        s = input("You: ")
        if s.strip().lower() == "quit":
            break
        res = get_response(s)
        if isinstance(res, list):
            print("Bot:", res[1])
        else:
            print("Bot:", res)


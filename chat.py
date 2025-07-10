import json, re, csv
from pathlib import Path
from typing import List, Dict
import pandas as pd
import joblib, numpy as np

from nltk_utils import tokenize
from translation_utils import (
    to_en,
    to_id,
    add_manual_translation,
    manual_translation_en_to_id,
    manual_translation_id_to_en,
)

# ───────────────────── konfigurasi dasar ─────────────────────
BASE       = Path(__file__).resolve().parent
MODEL_DIR  = BASE / "models"
MASTER_DIR = BASE / "MasterData"
MIN_SYMPT  = 3

# ╭─ 1. Artefak ML – lebih tangguh ..............................................
def _load_pickle(name: str):
    p = MODEL_DIR / name
    if not p.exists():
        raise FileNotFoundError(
            f"File {p} tidak ditemukan. Jalankan notebook pelatihan terlebih dulu."
        )
    return joblib.load(p)

MODEL   = _load_pickle("logistic_regression_model.pkl")
ENCODER = _load_pickle("encoder.pkl")

# Dukungan dua skenario:
#   • symptom_columns.pkl ada  → pakai itu,
#   • kalau tidak, ambil feature_names_in_ dari model.
try:
    COLUMNS: List[str] = _load_pickle("symptom_columns.pkl")
except FileNotFoundError:
    try:
        COLUMNS = list(MODEL.feature_names_in_)
    except AttributeError as e:
        raise RuntimeError(
            "Daftar fitur tidak ditemukan—"
            "simpan symptom_columns.pkl atau latih model memakai DataFrame."
        ) from e

VALID_SYMPTOMS: set[str] = set(COLUMNS)
SYMPT_IDX: Dict[str, int] = {s: i for i, s in enumerate(COLUMNS)}  # O(1) lookup

# ────────── regex pattern tiap gejala (pakai spasi) ──────────
SYMPTOM_PATTERNS = {
    s: re.compile(r'\b' + re.escape(s.replace('_', ' ')) + r'\b', flags=re.I)
    for s in VALID_SYMPTOMS
}

# ────────── muat deskripsi, pencegahan & obat ──────────
description_list: Dict[str, str]        = {}
precaution_dict : Dict[str, List[str]]  = {}
obat_list       : Dict[str, List[Dict]] = {}

desc_csv  = MASTER_DIR / "symptom_Description.xlsx"
prec_csv  = MASTER_DIR / "symptom_precaution.xlsx"
obat_xlsx = MASTER_DIR / "obat.xlsx"

if desc_csv.exists():
    try:
        df_desc = pd.read_excel(desc_csv)
        for _, row in df_desc.iterrows():
            disease = str(row[0]).strip()
            desc = str(row[1]).strip()
            description_list[disease] = desc
    except Exception as e:
        print(f"Error loading description: {e}")

if obat_xlsx.exists():
    df_obat = pd.read_excel(obat_xlsx)
    for _, row in df_obat.iterrows():
        disease = str(row["Nama Penyakit"]).strip().lower()
        obat_list.setdefault(disease, []).append(
            {
                "drug"   : str(row["Nama Obat"]).strip(),
                "age"    : str(row.get("Usia", "")).strip(),
                "dosage" : str(row["Takaran"]).strip(),
                "content": str(row["Kandungan Obat"]).strip(),
            }
        )

# Load tindakan pencegahan
if prec_csv.exists():
    try:
        df_prec = pd.read_excel(prec_csv)
        for _, row in df_prec.iterrows():
            disease = str(row[0]).strip()
            # Ambil 4 kolom, gabungkan jadi satu string, lalu pisahkan jadi list kalimat
            raw_precautions = " ".join([str(row.get(i, "")).strip() for i in range(1, 5)]).strip()
            # Pecah berdasarkan titik atau titik koma (tergantung format datanya)
            precautions = [p.strip(" .;-") for p in re.split(r"[.;]\s*", raw_precautions) if p.strip()]
            precaution_dict[disease] = precautions
    except Exception as e:
        print(f"Error loading precautions: {e}")

# ────────── memori sesi ──────────
symptom_session: List[str] = []
bot_state = "ask_name"
user_profile = {"name": None, "age": None}

# ────────── util utama ──────────
def list_all_diseases() -> List[str]:
    """Mengembalikan daftar semua penyakit yang dikenali oleh model."""
    return sorted(to_id(d) for d in ENCODER.classes_)

def extract_symptoms(sentence: str) -> List[str]:
    """Keluarkan nama‑nama gejala (dalam EN/ID) dari kalimat."""
    from difflib import get_close_matches

    found: set[str] = set()
    parts = re.split(r",|\bdan\b|&|/|\\", sentence)

    for part in parts:
        sent = to_en(part).lower()
        sent = re.sub(r"[^a-z0-9 ]+", " ", sent)
        sent = re.sub(r"\s+", " ", sent).strip()

        tokens = sent.split()
        ngrams = [
            " ".join(tokens[i : i + n])
            for n in range(1, 5)
            for i in range(len(tokens) - n + 1)
        ]

        for phrase in ngrams:
            for sym, pat in SYMPTOM_PATTERNS.items():
                if pat.search(phrase):
                    found.add(sym)
                    break
            else:  # belum persis, coba fuzzy
                close = get_close_matches(
                    phrase,
                    [s.replace("_", " ") for s in VALID_SYMPTOMS],
                    n=1,
                    cutoff=0.70,
                )
                if close:
                    found.add(close[0].replace(" ", "_"))
    return list(found)

def _vectorize(symptoms: List[str]) -> pd.DataFrame:
    vec = np.zeros(len(COLUMNS), dtype=int)
    for s in symptoms:
        idx = SYMPT_IDX.get(s)
        if idx is not None:
            vec[idx] = 1
    return pd.DataFrame([vec], columns=COLUMNS)

def classify(symptoms: List[str]) -> str:
    vec_df = _vectorize(symptoms)
    pred = MODEL.predict(vec_df)[0]              # numeric label
    return ENCODER.inverse_transform([pred])[0]  # penyakit

# ────────── inti percakapan ──────────
def get_response(user_msg: str):
    global bot_state, user_profile, symptom_session

    lowered = user_msg.strip().lower()

    # ────────── fitur: tampilkan semua penyakit ──────────
    if lowered in {"list", "daftar penyakit", "semua penyakit"}:
        diseases = list_all_diseases()
        return "📋 Berikut daftar penyakit yang dikenali sistem:\n" + "\n".join(f"- {d}" for d in diseases)

    # ────────── fase 1: tanya nama ──────────
    if bot_state == "ask_name":
        if not user_profile["name"]:
            token_name = next((t for t in tokenize(user_msg) if t.isalpha()), None)
            if token_name:
                user_profile["name"] = token_name.capitalize()
                bot_state = "ask_age"
                return f"Hai {user_profile['name']}! Berapa usia Anda?"
            return "Boleh saya tahu nama Anda?"

    # ────────── fase 2: tanya usia ──────────
    if bot_state == "ask_age":
        number = re.search(r'\d+', user_msg)
        if number:
            user_profile["age"] = int(number.group())
            bot_state = "collect_symptoms"
            return ("Terima kasih. Silakan ketik gejala Anda satu‑per‑satu "
                    "(boleh kalimat penuh), lalu ketik *done* jika selesai.")
        return "Berapa usia Anda (angka)?"

    # ────────── fase 3: input gejala ──────────
    if lowered != "done":
        new_syms = extract_symptoms(user_msg)
        if new_syms:
            symptom_session += [s for s in new_syms if s not in symptom_session]
        else:
            return [
                "unknown_symptom",
                "Maaf, saya tidak mengenali gejala pada kalimat tersebut. "
                "Silakan sebutkan gejala lain atau ketik *done* jika selesai."
            ]

        return [
            "collect",
            f"Noted: {', '.join(symptom_session)}. "
            f"Tambahkan gejala lain atau ketik *done* jika selesai."
        ]

    # ────────── user mengetik 'done' ──────────
    if len(symptom_session) < MIN_SYMPT:
        return [
            "need_more",
            f"Anda baru memasukkan {len(symptom_session)} gejala. "
            f"Silakan tambahkan setidaknya {MIN_SYMPT - len(symptom_session)} lagi."
        ]

    pred_label = classify(symptom_session)  # hanya 1 prediksi
    symptom_session = []  # reset setelah klasifikasi

    lines: List[str] = ["Berdasarkan gejala, kemungkinan penyakit Anda:\n"]
    lines.append(f"🩺 **{to_id(pred_label)}**")

    # Deskripsi
    if pred_label in description_list:
        lines.append(f"📌 {to_id(description_list[pred_label])}")

    # Pencegahan
# 🛡️ Tampilkan Pencegahan
    if pred_label in precaution_dict:
        lines.append("🛡️ Pencegahan:")
        for p in precaution_dict[pred_label]:
            lines.append(f"{to_id(p)}")
    # Obat
    meds = obat_list.get(pred_label.lower()) or obat_list.get(to_id(pred_label).lower())
    if meds:
        lines.append("💊 Rekomendasi obat:")
        for m in meds:
            line = (f"• {m['drug']} – {m['content']} "
                    f"(usia: {m['age']}) {m['dosage']}")
            lines.append(to_id(line))

    lines.append("")  # spasi
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
        print("Bot:", res[1] if isinstance(res, list) else res)

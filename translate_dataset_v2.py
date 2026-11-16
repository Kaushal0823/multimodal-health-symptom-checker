"""
Multilingual Symptom Dataset Builder (v2 - Multiple Sentence Styles)
------------------------------------------------------------------------
PEHLE KYA PROBLEM THI:
Har row ek hi fixed sentence banata tha: "I am experiencing X, Y, Z."
Model ne isi EXACT pattern ko yaad kar liya, real/natural sentences
par confuse ho gaya.

AB KYA FIX HAI:
Har row ke liye RANDOM sentence style choose hota hai (8 English
styles + unke translated versions + symptoms ka random order).
Isse model sirf ek wording yaad nahi karega, balki symptom-words
aur disease ke beech ka asli connection seekhega.

STEP 1 (ek baar): pip install deep-translator pandas tqdm indic-transliteration --break-system-packages
STEP 2: Isi folder mein "unique_symptoms.csv" aur "symptom_lists_full.csv" rakho
STEP 3: python translate_dataset_v2.py
STEP 4: Output milega -> multilingual_symptom_dataset_v2.csv
"""

import pandas as pd
import random
from deep_translator import GoogleTranslator
from tqdm import tqdm
from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate
import time

random.seed(42)

# ---------------- CONFIG ----------------
SYMPTOMS_FILE = "unique_symptoms.csv"
SYMPTOM_LISTS_FILE = "symptom_lists_full.csv"
OUTPUT_FILE = "multilingual_symptom_dataset_v2.csv"

TARGET_LANGUAGES = {
    "hindi": "hi",
    "bengali": "bn",
    "tamil": "ta",
    "telugu": "te",
    "marathi": "mr",
}

# English mein hum khud confident hain -- isliye symptoms ko sentence
# ke BEECH mein bhi natural tarike se embed kar sakte hain.
ENGLISH_TEMPLATES = [
    ("I am experiencing ", "."),
    ("I have been suffering from ", " for the past few days."),
    ("I have ", "."),
    ("My symptoms include ", "."),
    ("I have had ", " lately."),
    ("I am having ", "."),
    ("I notice ", " in myself."),
    ("Lately I feel like I have ", "."),
]

# Doosri languages ke liye SAFE pattern: "lead-in phrase: symptom list."
# (word-order issues se bachne ke liye list ko sentence ke beech mein
# force nahi karte, colon ke baad simple list rakhte hain)
LEAD_IN_PHRASES_EN = [
    "I am experiencing the following symptoms",
    "I have been suffering from these symptoms",
    "My symptoms are as follows",
    "I have the following health problems",
    "I have had these symptoms lately",
    "I notice the following in myself",
]

# Hinglish khud confident hain likhne mein -- Google Translate ki zaroorat nahi
HINGLISH_LEAD_INS = [
    "Mujhe ye lakshan mehsoos ho rahe hain",
    "Mujhe kuch dinon se ye problem ho rahi hai",
    "Mere symptoms kuch is tarah hain",
    "Mujhe ye takleef ho rahi hai",
]


def format_english_list(symptoms):
    symptoms = list(symptoms)
    random.shuffle(symptoms)
    if len(symptoms) == 1:
        return symptoms[0]
    elif len(symptoms) == 2:
        return f"{symptoms[0]} and {symptoms[1]}"
    return ", ".join(symptoms[:-1]) + f", and {symptoms[-1]}"


def translate_symptom_list(symptoms, lang_code, lang_name):
    translator = GoogleTranslator(source="en", target=lang_code)
    mapping = {}
    for s in tqdm(symptoms, desc=f"Translating symptom names -> {lang_name}"):
        try:
            mapping[s] = translator.translate(s)
        except Exception:
            mapping[s] = s
        time.sleep(0.05)
    return mapping


def translate_lead_ins(lang_code, lang_name):
    """Sirf 6 chhoti lead-in phrases translate karta hai (fast, kuch second mein)."""
    translator = GoogleTranslator(source="en", target=lang_code)
    localized = []
    for phrase in tqdm(LEAD_IN_PHRASES_EN, desc=f"Translating lead-ins -> {lang_name}"):
        try:
            localized.append(translator.translate(phrase))
        except Exception:
            localized.append(phrase)
        time.sleep(0.05)
    return localized


def build_localized_sentence(symptom_names, symptom_mapping, lead_ins):
    translated = [symptom_mapping.get(s, s) for s in symptom_names]
    random.shuffle(translated)
    joined = ", ".join(translated)
    lead_in = random.choice(lead_ins)
    return f"{lead_in}: {joined}."


def build_hinglish_symptom_mapping(hindi_mapping):
    hinglish_mapping = {}
    for eng_symptom, hindi_text in hindi_mapping.items():
        try:
            roman = transliterate(hindi_text, sanscript.DEVANAGARI, sanscript.ITRANS)
            hinglish_mapping[eng_symptom] = roman.lower().replace("_", " ")
        except Exception:
            hinglish_mapping[eng_symptom] = eng_symptom
    return hinglish_mapping


def main():
    symptoms_df = pd.read_csv(SYMPTOMS_FILE)
    symptoms = symptoms_df["symptom"].tolist()

    lists_df = pd.read_csv(SYMPTOM_LISTS_FILE)
    print(f"Loaded {len(lists_df)} rows and {len(symptoms)} unique symptoms.")

    # Symptom names translate karo (jaise pehle, per-language)
    symptom_mappings = {}
    for lang_name, lang_code in TARGET_LANGUAGES.items():
        print(f"\n--- {lang_name.upper()} symptom names ---")
        symptom_mappings[lang_name] = translate_symptom_list(symptoms, lang_code, lang_name)

    # Lead-in phrases bhi translate karo (fast -- sirf 6 chhoti lines per language)
    localized_lead_ins = {}
    for lang_name, lang_code in TARGET_LANGUAGES.items():
        localized_lead_ins[lang_name] = translate_lead_ins(lang_code, lang_name)

    hinglish_symptom_mapping = build_hinglish_symptom_mapping(symptom_mappings["hindi"])

    output_rows = []
    for _, row in tqdm(lists_df.iterrows(), total=len(lists_df), desc="Building sentences"):
        symptom_names = str(row["symptoms"]).split("|")
        entry = {"disease": row["disease"]}

        eng_prefix, eng_suffix = random.choice(ENGLISH_TEMPLATES)
        entry["text"] = f"{eng_prefix}{format_english_list(symptom_names)}{eng_suffix}"

        for lang_name in TARGET_LANGUAGES:
            entry[lang_name] = build_localized_sentence(
                symptom_names, symptom_mappings[lang_name], localized_lead_ins[lang_name]
            )

        entry["hinglish"] = build_localized_sentence(
            symptom_names, hinglish_symptom_mapping, HINGLISH_LEAD_INS
        )

        output_rows.append(entry)

    output_df = pd.DataFrame(output_rows)
    output_df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nDone! Saved: {OUTPUT_FILE}")
    print(f"Columns: {list(output_df.columns)}")
    print(f"Total rows: {len(output_df)}")


if __name__ == "__main__":
    main()

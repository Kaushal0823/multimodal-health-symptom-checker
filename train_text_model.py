# =====================================================================
# Multilingual Symptom -> Disease Classifier (Text Model)
# Google Colab ke liye ready hai
# =====================================================================
#
# KAISE USE KARNA HAI:
# 1. Colab kholo: colab.research.google.com
# 2. Runtime -> Change runtime type -> GPU (T4) select karo
# 3. Neeche wale saare cells ek-ek karke Colab mein paste karo aur Run karo
#    (Shift+Enter se ek cell run hoti hai)
# 4. Jaha "UPLOAD" likha hai, wahan apni file Colab mein upload karo
#
# =====================================================================

# ---------------- CELL 1: Google Drive mount karo ----------------
from google.colab import drive
drive.mount('/content/drive')

# Apna dataset Google Drive mein ek folder banake usme daal do, jaise:
# MyDrive/health_project/multilingual_symptom_dataset.csv
# Neeche path ko apne hisaab se badal lena.

DATASET_PATH = "/content/drive/MyDrive/health_project/multilingual_symptom_dataset.csv"


# ---------------- CELL 2: Zaroori libraries install karo ----------------
!pip install -q transformers datasets scikit-learn torch accelerate


# ---------------- CELL 3: Dataset load aur prepare karo ----------------
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv(DATASET_PATH)
print("Total rows:", len(df))
print("Columns:", df.columns.tolist())
print("Unique diseases:", df['disease'].nunique())

# Har language ke rows ko ek column mein combine karte hain
# (taaki model ek saath sab languages se seekhe -- ye hi "multilingual" training hai)
lang_columns = ['text', 'hindi', 'bengali', 'tamil', 'telugu', 'marathi', 'hinglish']

rows = []
for _, row in df.iterrows():
    for lang_col in lang_columns:
        if lang_col in df.columns and pd.notna(row[lang_col]):
            rows.append({'text': row[lang_col], 'disease': row['disease'], 'language': lang_col})

combined_df = pd.DataFrame(rows)
print(f"\nTotal training examples (sab languages milakar): {len(combined_df)}")
print(combined_df['language'].value_counts())


# ---------------- CELL 4: Labels ko numbers mein convert karo ----------------
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
combined_df['label'] = label_encoder.fit_transform(combined_df['disease'])
num_labels = len(label_encoder.classes_)
print(f"Total disease classes: {num_labels}")

# Label mapping save kar lo -- baad mein prediction ko wapas disease name mein convert karne ke liye zaroori hai
import pickle
with open('/content/drive/MyDrive/health_project/label_encoder.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)


# ---------------- CELL 5: Train/Validation split ----------------
train_df, val_df = train_test_split(
    combined_df, test_size=0.15, random_state=42, stratify=combined_df['label']
)
print(f"Train size: {len(train_df)}, Validation size: {len(val_df)}")


# ---------------- CELL 6: Tokenizer aur Model load karo ----------------
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_NAME = "xlm-roberta-base"  # multilingual model -- 100+ languages samajhta hai

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=num_labels)


# ---------------- CELL 7: Dataset ko HuggingFace format mein convert karo ----------------
from datasets import Dataset

train_dataset = Dataset.from_pandas(train_df[['text', 'label']])
val_dataset = Dataset.from_pandas(val_df[['text', 'label']])

def tokenize_function(examples):
    return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=64)

train_dataset = train_dataset.map(tokenize_function, batched=True)
val_dataset = val_dataset.map(tokenize_function, batched=True)


# ---------------- CELL 8: Training setup ----------------
from transformers import TrainingArguments, Trainer
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        'accuracy': accuracy_score(labels, predictions),
        'f1_weighted': f1_score(labels, predictions, average='weighted')
    }

training_args = TrainingArguments(
    output_dir="/content/drive/MyDrive/health_project/model_checkpoints",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,   # RTX 2050 jaisa chhota GPU ho toh 8 kar dena
    per_device_eval_batch_size=16,
    num_train_epochs=5,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="f1_weighted",
    logging_steps=50,
    save_total_limit=2,
    fp16=True,   # Colab GPU par fast training ke liye
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)


# ---------------- CELL 9: Training shuru karo ----------------
trainer.train()


# ---------------- CELL 10: Final model save karo ----------------
FINAL_MODEL_PATH = "/content/drive/MyDrive/health_project/final_text_model"
trainer.save_model(FINAL_MODEL_PATH)
tokenizer.save_pretrained(FINAL_MODEL_PATH)
print(f"Model saved to: {FINAL_MODEL_PATH}")


# ---------------- CELL 11: Test karo -- kisi bhi language mein try karke dekho ----------------
from transformers import pipeline

classifier = pipeline(
    "text-classification",
    model=FINAL_MODEL_PATH,
    tokenizer=FINAL_MODEL_PATH,
    device=0  # GPU use karo
)

def predict_disease(text):
    result = classifier(text)[0]
    label_id = int(result['label'].split('_')[-1])  # LABEL_0, LABEL_1 format se number nikalo
    disease_name = label_encoder.inverse_transform([label_id])[0]
    confidence = result['score']
    return disease_name, confidence

# Test examples -- kisi bhi language mein try karo
test_texts = [
    "I have fever and headache",              # English
    "मुझे बुखार और सिरदर्द है",                  # Hindi
    "Mujhe bukhar aur sir dard hai",           # Hinglish
]

for text in test_texts:
    disease, confidence = predict_disease(text)
    print(f"Input: {text}")
    print(f"Predicted: {disease} (confidence: {confidence:.2%})\n")

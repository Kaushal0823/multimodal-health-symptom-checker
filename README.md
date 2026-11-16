# Multimodal Multilingual Health Symptom Checker

A final-year B.Tech (CSE) project that takes a symptom description **in any language (including Hinglish)** and an **optional skin photograph**, and returns a ranked list of possible conditions with confidence values. The answer is returned in the language of the question.

> ⚠️ **Disclaimer:** This is a research prototype and screening aid. It is **not** a medical device and does **not** provide a diagnosis. Always consult a qualified doctor.

## Features
- **Multilingual text input**: 7 language varieties, including Hinglish (Hindi in Latin script); language is auto-detected
- **Optional skin image input**: 7 lesion classes (HAM10000)
- **Fusion** of text and image evidence using keyword overlap between the two label sets
- **Back-translation**: the result is shown in the user's language
- **Gradio demo** with a shareable link

## System Overview
```
Text input ─► Language detection ─► XLM-RoBERTa (698 diseases) ─┐
                                                                 ├─► Fusion ─► Ranked result ─► Back-translation
Skin image ─► ResNet50 (7 lesion classes) ──────────────────────┘
```

## Results
| Component | Metric | Value |
|---|---|---|
| Text model (XLM-RoBERTa-base) | Validation accuracy | 88.7% |
| Image model (ResNet50, frozen backbone) | Accuracy | 67.36% |
| Image model | Weighted F1 / Macro F1 | 0.7039 / 0.48 |

Dataset: 246,945 raw rows → 189,543 after cleaning → 25,739 unique records (698 diseases, 377 symptoms) → 180,173 training sentences. A single-template version of the dataset gave misleadingly low real-world confidence (17.1–32.0% on test sentences); using 8 varied templates raised it (up to 97.3% on the same sentences).

## Tech Stack
Python, PyTorch, Hugging Face Transformers, torchvision, scikit-learn, langdetect, deep-translator, Gradio. Training was done on Google Colab (T4 GPU).

## Repository Contents
| File | Purpose |
|---|---|
| `clean_images.py` | Cleans and resizes HAM10000 images |
| `translate_dataset_v2.py` | Builds the multilingual symptom dataset |
| `train_text_model.py` | Fine-tunes the XLM-RoBERTa text model |
| `train_image_model.py` | Trains the ResNet50 skin image model |
| `unique_symptoms.csv` | The 377 symptom names |

Large files (HAM10000 images, trained model weights) are **not** stored in this repository because of GitHub's size limit. Add your links here:
- Trained models: `<Hugging Face / Google Drive link>`
- HAM10000 dataset: https://doi.org/10.7910/DVN/DBW86T

## How to Run the Demo (Google Colab)
1. Open the demo notebook in Google Colab and set the runtime to **T4 GPU**.
2. Mount Google Drive and install requirements:
   `pip install transformers torch torchvision scikit-learn langdetect deep-translator gradio`
3. Load the text and image models from the saved model folders.
4. Run the Gradio cell: `demo.launch(share=True)`
5. Open the generated `https://xxxxx.gradio.live` link.

For best results, describe symptoms as a full sentence (e.g. "I have itching and red rash on my skin"). Single words may be less reliable.

## Limitations
- Training sentences are synthetic and machine-translated.
- The image model is trained on dermatoscopic images and may not transfer to phone photos.
- Fusion is a keyword-based heuristic, not a learned joint model.
- Not clinically validated.

## Team
- **Kaushal Rajak**
- **Arpit Pal**
- **Guide:** Mr. Gautam

Department of Computer Science & Engineering, L.D.A.H. Rajkiya Engineering College, Mainpuri.

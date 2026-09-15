# 🚗 Road Safety Intelligence

A multimodal machine learning project for road-safety analysis using **numerical, text, audio, image, and video data**.

The project compares two techniques for each modality and integrates the best-performing models into a Streamlit web application.

## Live Demo

🌐 https://road-safety-intelligence.streamlit.app/

## Results

| Modality | Best Model | Accuracy |
|---|---|---:|
| Numerical | Random Forest | **71.59%** |
| Text | TF-IDF + Logistic Regression | **90.74%** |
| Audio | MFCC + Random Forest | **73.84%** |
| Image | MobileNetV2 | **87.00%** |
| Video | MobileNetV2 + LSTM | **98.00%** |

## What I Built

- **Numerical:** accident severity prediction using weather, road, time and location features.
- **Text:** accident severity prediction from accident descriptions using TF-IDF.
- **Audio:** environmental sound classification using MFCC features.
- **Image:** accident vs. non-accident detection using MobileNetV2 transfer learning.
- **Video:** accident vs. normal traffic detection using MobileNetV2 features with LSTM/GRU.

For every modality, two techniques were compared and the better-performing model was selected.

## Tech Stack

**Python · Scikit-learn · TensorFlow/Keras · Streamlit · Hugging Face · GitHub**

## Datasets

- US Accidents
- UrbanSound8K
- Accident Detection From CCTV Footage
- HWID12 Highway Incidents Detection Dataset

The project uses **real dataset samples only**. Class balancing was performed by selecting real samples from the datasets.

Trained model files are hosted separately on Hugging Face:
🤗 https://huggingface.co/Black-Bolt/road-safety-intelligence-models

##Future Improvements

-Dedicated crash-audio dataset
-Multimodal prediction fusion
-Larger real-world datasets
-Model explainability

Author

Himanshu Singh
GitHub: https://github.com/Maniac-extraordinaire


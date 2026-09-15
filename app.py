import os
import tempfile

import cv2
import joblib
import librosa
import numpy as np
import pandas as pd
import streamlit as st

from huggingface_hub import hf_hub_download
from PIL import Image

from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Road Safety Intelligence",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ============================================================
# LOAD EXTERNAL CSS
# ============================================================

CSS_PATH = os.path.join("assets", "style.css")

try:
    with open(CSS_PATH, "r", encoding="utf-8") as css_file:
        st.markdown(
            f"<style>{css_file.read()}</style>",
            unsafe_allow_html=True
        )
except FileNotFoundError:
    st.warning(
        "assets/style.css was not found. "
        "The application will continue without custom styling."
    )


# ============================================================
# HERO
# ============================================================

st.title("🚗 Road Safety Intelligence")

st.write(
    "A multimodal Machine Learning system for road-safety analysis "
    "using numerical, text, audio, image and video data."
)

st.caption(
    "Multimodal AI • Accident Analysis • Severity Prediction • "
    "Environmental Sound Classification"
)


# ============================================================
# PROJECT PERFORMANCE
# ============================================================

st.subheader("📈 Model Performance Overview")

k1, k2, k3, k4, k5 = st.columns(5)

with k1:
    st.metric(
        "📊 Numerical",
        "71.59%",
        "Random Forest"
    )

with k2:
    st.metric(
        "📝 Text",
        "90.74%",
        "Logistic Regression"
    )

with k3:
    st.metric(
        "🔊 Audio",
        "73.84%",
        "Random Forest"
    )

with k4:
    st.metric(
        "🖼️ Image",
        "87.00%",
        "MobileNetV2"
    )

with k5:
    st.metric(
        "🎥 Video",
        "98.00%",
        "LSTM"
    )

st.caption(
    "Accuracy values come from the held-out evaluation procedures "
    "used for the individual modalities."
)


# ============================================================
# MODEL COMPARISON
# ============================================================

st.subheader("⚖️ Technique Comparison")

comparison_df = pd.DataFrame(
    [
        {
            "Modality": "📊 Numerical",
            "Technique 1": "Logistic Regression + Time",
            "Accuracy 1": "47.00%",
            "Technique 2": "Random Forest + Time + Location",
            "Accuracy 2": "71.59%",
            "Better": "Random Forest",
            "Gain": "+24.59 pp"
        },
        {
            "Modality": "📝 Text",
            "Technique 1": "TF-IDF + Logistic Regression",
            "Accuracy 1": "90.74%",
            "Technique 2": "TF-IDF + Linear SVM",
            "Accuracy 2": "90.61%",
            "Better": "Logistic Regression",
            "Gain": "+0.13 pp"
        },
        {
            "Modality": "🔊 Audio",
            "Technique 1": "MFCC + Random Forest",
            "Accuracy 1": "73.84%",
            "Technique 2": "MFCC + SVM",
            "Accuracy 2": "66.19%",
            "Better": "Random Forest",
            "Gain": "+7.65 pp"
        },
        {
            "Modality": "🖼️ Image",
            "Technique 1": "MobileNetV2",
            "Accuracy 1": "87.00%",
            "Technique 2": "EfficientNetB0",
            "Accuracy 2": "80.00%",
            "Better": "MobileNetV2",
            "Gain": "+7.00 pp"
        },
        {
            "Modality": "🎥 Video",
            "Technique 1": "MobileNetV2 Features + LSTM",
            "Accuracy 1": "98.00%",
            "Technique 2": "MobileNetV2 Features + GRU",
            "Accuracy 2": "97.00%",
            "Better": "LSTM",
            "Gain": "+1.00 pp"
        }
    ]
)

st.dataframe(
    comparison_df,
    width="stretch",
    hide_index=True
)

st.caption(
    "pp = percentage points. Numerical comparison includes feature "
    "engineering; the final Random Forest additionally uses location "
    "features, so this row represents the evaluated pipeline comparison."
)


# ============================================================
# IMPROVEMENT
# ============================================================

st.subheader("📊 Improvement of Selected Models")

improvement_df = pd.DataFrame(
    [
        {
            "Modality": "Numerical",
            "Comparison": "47.00% → 71.59%",
            "Absolute Improvement": "+24.59 pp",
            "Relative Improvement": "+52.32%"
        },
        {
            "Modality": "Text",
            "Comparison": "90.61% → 90.74%",
            "Absolute Improvement": "+0.13 pp",
            "Relative Improvement": "+0.14%"
        },
        {
            "Modality": "Audio",
            "Comparison": "66.19% → 73.84%",
            "Absolute Improvement": "+7.65 pp",
            "Relative Improvement": "+11.56%"
        },
        {
            "Modality": "Image",
            "Comparison": "80.00% → 87.00%",
            "Absolute Improvement": "+7.00 pp",
            "Relative Improvement": "+8.75%"
        },
        {
            "Modality": "Video",
            "Comparison": "97.00% → 98.00%",
            "Absolute Improvement": "+1.00 pp",
            "Relative Improvement": "+1.03%"
        }
    ]
)

st.dataframe(
    improvement_df,
    width="stretch",
    hide_index=True
)


# ============================================================
# DATASET OVERVIEW
# ============================================================

with st.expander("📚 Dataset & Feature Overview"):

    dataset_df = pd.DataFrame(
        [
            {
                "Modality": "Numerical",
                "Dataset": "US Accidents",
                "Real Samples": "200,000 balanced",
                "Main Features": (
                    "Weather, road flags, time, latitude/longitude"
                ),
                "Task": "Severity 1–4"
            },
            {
                "Modality": "Text",
                "Dataset": "US Accidents",
                "Real Samples": "200,000 balanced",
                "Main Features": (
                    "Accident Description → TF-IDF 1–2 grams"
                ),
                "Task": "Severity 1–4"
            },
            {
                "Modality": "Audio",
                "Dataset": "UrbanSound8K",
                "Real Samples": "8,732 clips",
                "Main Features": (
                    "40 MFCC means + 40 MFCC standard deviations"
                ),
                "Task": "10 sound classes"
            },
            {
                "Modality": "Image",
                "Dataset": "Accident Detection From CCTV Footage",
                "Real Samples": "989 images",
                "Main Features": "224×224 RGB images",
                "Task": "Accident / Non-Accident"
            },
            {
                "Modality": "Video",
                "Dataset": "HWID12 Highway Incidents",
                "Real Samples": "500 sampled real videos",
                "Main Features": (
                    "8 frames/video → MobileNetV2 1280-D features"
                ),
                "Task": "Accident / Normal"
            }
        ]
    )

    st.dataframe(
        dataset_df,
        width="stretch",
        hide_index=True
    )

    st.info(
        "No synthetic samples are used. Real dataset samples were used "
        "throughout the project; balancing was performed by selecting "
        "real samples."
    )


# ============================================================
# HUGGING FACE REPOSITORY
# ============================================================

HF_REPO = (
    "Black-Bolt/road-safety-intelligence-models"
)


# ============================================================
# LAZY MODEL LOADERS
# ============================================================

@st.cache_resource
def load_numerical_models():

    model_path = hf_hub_download(
        repo_id=HF_REPO,
        filename="numerical_model.pkl"
    )

    preprocessor_path = hf_hub_download(
        repo_id=HF_REPO,
        filename="numerical_preprocessor.pkl"
    )

    model = joblib.load(model_path)
    preprocessor = joblib.load(preprocessor_path)

    return model, preprocessor


@st.cache_resource
def load_text_models():

    tfidf_path = hf_hub_download(
        repo_id=HF_REPO,
        filename="text_tfidf.pkl"
    )

    model_path = hf_hub_download(
        repo_id=HF_REPO,
        filename="text_model.pkl"
    )

    tfidf = joblib.load(tfidf_path)
    model = joblib.load(model_path)

    return tfidf, model


@st.cache_resource
def load_audio_model():

    model_path = hf_hub_download(
        repo_id=HF_REPO,
        filename="audio_model.pkl"
    )

    return joblib.load(model_path)


@st.cache_resource
def load_image_model():

    model_path = hf_hub_download(
        repo_id=HF_REPO,
        filename="image_mobilenetv2.keras"
    )

    return load_model(model_path)


@st.cache_resource
def load_video_models():

    cnn_path = hf_hub_download(
        repo_id=HF_REPO,
        filename="video_mobilenetv2.keras"
    )

    lstm_path = hf_hub_download(
        repo_id=HF_REPO,
        filename="video_lstm.keras"
    )

    gru_path = hf_hub_download(
        repo_id=HF_REPO,
        filename="video_gru.keras"
    )

    cnn = load_model(cnn_path)
    lstm = load_model(lstm_path)
    gru = load_model(gru_path)

    return cnn, lstm, gru


# ============================================================
# TABS
# ============================================================

tab_num, tab_text, tab_audio, tab_image, tab_video = st.tabs(
    [
        "📊 Numerical",
        "📝 Text",
        "🔊 Audio",
        "🖼️ Image",
        "🎥 Video"
    ]
)


# ============================================================
# NUMERICAL
# ============================================================

with tab_num:

    st.header("📊 Accident Severity Prediction")

    st.info(
        "This model predicts accident severity from road, weather, "
        "location and time-related conditions. It does not determine "
        "whether an accident has actually occurred."
    )

    st.subheader("🌦️ Weather & Road Conditions")

    col1, col2, col3 = st.columns(3)

    with col1:

        distance = st.number_input(
            "Distance (miles)",
            min_value=0.0,
            value=0.5
        )

        temperature = st.number_input(
            "Temperature (°F)",
            value=60.0
        )

        humidity = st.number_input(
            "Humidity (%)",
            min_value=0.0,
            max_value=100.0,
            value=60.0
        )

        pressure = st.number_input(
            "Pressure (in)",
            value=29.5
        )

    with col2:

        visibility = st.number_input(
            "Visibility (miles)",
            min_value=0.0,
            value=10.0
        )

        wind_speed = st.number_input(
            "Wind Speed (mph)",
            min_value=0.0,
            value=5.0
        )

        precipitation = st.number_input(
            "Precipitation (in)",
            min_value=0.0,
            value=0.0
        )

    with col3:

        hour = st.number_input(
            "Hour of Incident",
            min_value=0,
            max_value=23,
            value=12,
            help="0 = midnight, 12 = noon, 23 = 11 PM."
        )

        day_of_week = st.selectbox(
            "Day of Week",
            options=list(range(7)),
            format_func=lambda x: [
                "Monday",
                "Tuesday",
                "Wednesday",
                "Thursday",
                "Friday",
                "Saturday",
                "Sunday"
            ][x],
            index=2
        )

        month_names = [
            "January",
            "February",
            "March",
            "April",
            "May",
            "June",
            "July",
            "August",
            "September",
            "October",
            "November",
            "December"
        ]

        month = st.selectbox(
            "Month",
            options=list(range(1, 13)),
            format_func=lambda x: month_names[x - 1],
            index=5
        )

    st.subheader("📍 Accident Location")

    loc1, loc2 = st.columns(2)

    with loc1:

        latitude = st.number_input(
            "Start Latitude",
            value=39.0,
            format="%.6f"
        )

    with loc2:

        longitude = st.number_input(
            "Start Longitude",
            value=-95.0,
            format="%.6f"
        )

    st.subheader("🛣️ Road Features")

    r1, r2, r3 = st.columns(3)

    with r1:

        amenity = st.checkbox("Amenity")
        bump = st.checkbox("Bump")
        crossing = st.checkbox("Crossing")

    with r2:

        junction = st.checkbox("Junction")
        railway = st.checkbox("Railway")
        roundabout = st.checkbox("Roundabout")

    with r3:

        station = st.checkbox("Station")
        stop = st.checkbox("Stop")
        traffic_signal = st.checkbox("Traffic Signal")

    sunrise_sunset = st.selectbox(
        "Time of Day",
        ["Day", "Night"]
    )

    with st.expander("📌 What do Severity 1–4 mean?"):

        st.markdown(
            """
            **Severity 1 — Minor**  
            Lower-severity accident with relatively limited impact.

            **Severity 2 — Moderate**  
            Accident with a moderate level of disruption.

            **Severity 3 — Serious**  
            Accident associated with substantial traffic disruption.

            **Severity 4 — Severe**  
            Highest severity category in the dataset.
            """
        )

    if st.button(
        "🚨 Predict Accident Severity",
        key="numerical_predict"
    ):

        try:

            with st.spinner(
                "Loading numerical model..."
            ):

                numerical_model, numerical_preprocessor = (
                    load_numerical_models()
                )

            is_weekend = int(
                day_of_week >= 5
            )

            hour_sin = np.sin(
                2 * np.pi * hour / 24
            )

            hour_cos = np.cos(
                2 * np.pi * hour / 24
            )

            day_sin = np.sin(
                2 * np.pi * day_of_week / 7
            )

            day_cos = np.cos(
                2 * np.pi * day_of_week / 7
            )

            month_sin = np.sin(
                2 * np.pi * month / 12
            )

            month_cos = np.cos(
                2 * np.pi * month / 12
            )

            numerical_input = pd.DataFrame(
                [{
                    "Distance(mi)": distance,
                    "Temperature(F)": temperature,
                    "Humidity(%)": humidity,
                    "Pressure(in)": pressure,
                    "Visibility(mi)": visibility,
                    "Wind_Speed(mph)": wind_speed,
                    "Precipitation(in)": precipitation,

                    "Hour": hour,
                    "DayOfWeek": day_of_week,
                    "Month": month,
                    "IsWeekend": is_weekend,

                    "Hour_sin": hour_sin,
                    "Hour_cos": hour_cos,
                    "Day_sin": day_sin,
                    "Day_cos": day_cos,
                    "Month_sin": month_sin,
                    "Month_cos": month_cos,

                    "Start_Lat": latitude,
                    "Start_Lng": longitude,

                    "Amenity": amenity,
                    "Bump": bump,
                    "Crossing": crossing,
                    "Junction": junction,
                    "Railway": railway,
                    "Roundabout": roundabout,
                    "Station": station,
                    "Stop": stop,
                    "Traffic_Signal": traffic_signal,

                    "Sunrise_Sunset": sunrise_sunset
                }]
            )

            processed = (
                numerical_preprocessor.transform(
                    numerical_input
                )
            )

            prediction = int(
                numerical_model.predict(
                    processed
                )[0]
            )

            severity_info = {

                1: (
                    "Minor",
                    "Lower-severity accident with relatively limited impact."
                ),

                2: (
                    "Moderate",
                    "Accident with a moderate level of disruption."
                ),

                3: (
                    "Serious",
                    "Accident associated with substantial traffic disruption."
                ),

                4: (
                    "Severe",
                    "Highest severity category in the dataset."
                )
            }

            severity_name, severity_description = (
                severity_info[prediction]
            )

            st.subheader("Prediction Result")

            st.success(
                f"🚨 Predicted Severity: "
                f"**{prediction} — {severity_name}**"
            )

            st.write(
                severity_description
            )

            st.caption(
                "Random Forest • 29 numerical/contextual features • "
                "Test Accuracy: 71.59% • Macro F1: 71.00%"
            )

        except Exception as e:

            st.error(
                f"Numerical prediction error: {e}"
            )


# ============================================================
# TEXT
# ============================================================

with tab_text:

    st.header("📝 Accident Severity from Description")

    st.info(
        "Enter a road-accident description. TF-IDF converts the text "
        "into numerical features and Logistic Regression predicts "
        "the severity category."
    )

    text_input = st.text_area(
        "Accident Description",
        height=180,
        placeholder=(
            "Example: Accident on I-70 westbound with two lanes "
            "blocked due to a collision."
        )
    )

    if st.button(
        "🔍 Predict Text Severity",
        key="text_predict"
    ):

        if not text_input.strip():

            st.warning(
                "Please enter an accident description."
            )

        else:

            try:

                with st.spinner(
                    "Loading text model..."
                ):

                    text_tfidf, text_model = (
                        load_text_models()
                    )

                vectorized_text = (
                    text_tfidf.transform(
                        [text_input]
                    )
                )

                prediction = int(
                    text_model.predict(
                        vectorized_text
                    )[0]
                )

                probabilities = (
                    text_model.predict_proba(
                        vectorized_text
                    )[0]
                )

                confidence = float(
                    np.max(probabilities)
                )

                severity_info = {

                    1: (
                        "Minor",
                        "Lower-severity accident with relatively limited impact."
                    ),

                    2: (
                        "Moderate",
                        "Accident with a moderate level of disruption."
                    ),

                    3: (
                        "Serious",
                        "Accident associated with substantial traffic disruption."
                    ),

                    4: (
                        "Severe",
                        "Highest severity category in the dataset."
                    )
                }

                severity_name, severity_description = (
                    severity_info[prediction]
                )

                st.subheader("Prediction Result")

                st.success(
                    f"🚨 Predicted Severity: "
                    f"**{prediction} — {severity_name}**"
                )

                st.write(
                    severity_description
                )

                st.metric(
                    "Model Confidence",
                    f"{confidence * 100:.1f}%"
                )

                st.caption(
                    "TF-IDF + Logistic Regression • "
                    "Test Accuracy: 90.74% • Macro F1: 91.00%"
                )

            except Exception as e:

                st.error(
                    f"Text prediction error: {e}"
                )


# ============================================================
# AUDIO
# ============================================================

with tab_audio:

    st.header("🎙️ Environmental Sound Classification")

    st.info(
        "This model classifies environmental sounds such as sirens, "
        "car horns, dog barks and engine sounds. It does not directly "
        "detect vehicle crashes."
    )

    st.subheader("Choose Audio Source")

    audio_source_type = st.radio(
        "Select input method:",
        ["Upload Audio", "Record Audio"],
        horizontal=True,
        key="audio_source_type"
    )

    audio_file = None
    recorded_audio = None

    if audio_source_type == "Upload Audio":

        audio_file = st.file_uploader(
            "Upload an audio file",
            type=["wav", "mp3", "ogg"],
            key="audio_upload"
        )

        if audio_file is not None:
            st.audio(audio_file)

    else:

        recorded_audio = st.audio_input(
            "Record audio",
            key="audio_record"
        )

        if recorded_audio is not None:
            st.audio(recorded_audio)

    if st.button(
        "🔍 Analyze Audio",
        key="audio_predict"
    ):

        audio_source = (
            audio_file
            if audio_source_type == "Upload Audio"
            else recorded_audio
        )

        if audio_source is None:

            st.warning(
                "Please upload an audio file or record a sound."
            )

        else:

            temp_audio = None

            try:

                with st.spinner(
                    "Loading audio model..."
                ):

                    audio_model = (
                        load_audio_model()
                    )

                temp_audio = tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=".wav"
                )

                temp_audio.write(
                    audio_source.getbuffer()
                )

                temp_audio.close()

                y_audio, sr_audio = librosa.load(
                    temp_audio.name,
                    sr=22050,
                    mono=True
                )

                mfcc = librosa.feature.mfcc(
                    y=y_audio,
                    sr=sr_audio,
                    n_mfcc=40
                )

                audio_features = np.hstack(
                    [
                        np.mean(mfcc, axis=1),
                        np.std(mfcc, axis=1)
                    ]
                ).reshape(1, -1)

                prediction = int(
                    audio_model.predict(
                        audio_features
                    )[0]
                )

                audio_labels = {

                    0: "Air Conditioner",
                    1: "Car Horn",
                    2: "Children Playing",
                    3: "Dog Bark",
                    4: "Drilling",
                    5: "Engine Idling",
                    6: "Gun Shot",
                    7: "Jackhammer",
                    8: "Siren",
                    9: "Street Music"
                }

                predicted_class = audio_labels.get(
                    prediction,
                    str(prediction)
                )

                st.subheader("Prediction Result")

                st.success(
                    f"🔊 Predicted Class: **{predicted_class}**"
                )

                st.caption(
                    "MFCC features • Random Forest • "
                    "Test Accuracy: 73.84% • Macro F1: 75.00%"
                )

            except Exception as e:

                st.error(
                    f"Audio prediction error: {e}"
                )

            finally:

                if temp_audio is not None:

                    try:
                        os.remove(
                            temp_audio.name
                        )
                    except OSError:
                        pass


# ============================================================
# IMAGE
# ============================================================

with tab_image:

    st.header("🖼️ Accident Detection from Image")

    st.info(
        "Upload a road image. MobileNetV2 transfer learning predicts "
        "whether the image contains an accident."
    )

    image_file = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png"],
        key="image_upload"
    )

    image = None

    if image_file is not None:

        image = Image.open(
            image_file
        ).convert("RGB")

        st.image(
            image,
            caption="Uploaded Image",
            width="stretch"
        )

    if st.button(
        "🔍 Analyze Image",
        key="image_predict"
    ):

        if image is None:

            st.warning(
                "Please upload an image."
            )

        else:

            try:

                with st.spinner(
                    "Loading image model..."
                ):

                    image_model = (
                        load_image_model()
                    )

                resized_image = image.resize(
                    (224, 224)
                )

                image_array = np.array(
                    resized_image,
                    dtype=np.float32
                )

                image_array = np.expand_dims(
                    image_array,
                    axis=0
                )

                image_array = preprocess_input(
                    image_array
                )

                probability = float(
                    image_model.predict(
                        image_array,
                        verbose=0
                    )[0][0]
                )

                if probability >= 0.5:

                    result = "Non Accident"
                    confidence = probability

                else:

                    result = "Accident"
                    confidence = 1 - probability

                st.subheader("Prediction Result")

                if result == "Accident":

                    st.error(
                        "🚨 **Accident detected**"
                    )

                else:

                    st.success(
                        "✅ **Non Accident detected**"
                    )

                st.metric(
                    "Prediction Confidence",
                    f"{confidence * 100:.1f}%"
                )

                st.caption(
                    "MobileNetV2 Transfer Learning • "
                    "Test Accuracy: 87.00%"
                )

            except Exception as e:

                st.error(
                    f"Image prediction error: {e}"
                )


# ============================================================
# VIDEO
# ============================================================

with tab_video:

    st.header("🎥 Accident Detection from Video")

    st.info(
        "The system samples 8 frames from the video, extracts "
        "MobileNetV2 visual features, and analyzes the temporal "
        "sequence using LSTM and GRU models."
    )

    video_file = st.file_uploader(
        "Upload a video",
        type=["mp4", "avi", "mov"],
        key="video_upload"
    )

    if video_file is not None:

        st.subheader("Video Preview")

        st.video(video_file)

    if st.button(
        "🔍 Analyze Video",
        key="video_predict"
    ):

        if video_file is None:

            st.warning(
                "Please upload a video first."
            )

        else:

            temp_video = None

            try:

                with st.spinner(
                    "Loading video models..."
                ):

                    video_cnn, video_lstm, video_gru = (
                        load_video_models()
                    )

                temp_video = tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=os.path.splitext(
                        video_file.name
                    )[1]
                )

                temp_video.write(
                    video_file.getbuffer()
                )

                temp_video.close()

                cap = cv2.VideoCapture(
                    temp_video.name
                )

                total_frames = int(
                    cap.get(
                        cv2.CAP_PROP_FRAME_COUNT
                    )
                )

                if total_frames <= 0:

                    raise ValueError(
                        "Unable to read frames from this video."
                    )

                frame_indices = np.linspace(
                    0,
                    total_frames - 1,
                    8,
                    dtype=int
                )

                frames = []

                for index in frame_indices:

                    cap.set(
                        cv2.CAP_PROP_POS_FRAMES,
                        int(index)
                    )

                    success, frame = cap.read()

                    if not success:
                        continue

                    frame = cv2.cvtColor(
                        frame,
                        cv2.COLOR_BGR2RGB
                    )

                    frame = cv2.resize(
                        frame,
                        (112, 112)
                    )

                    frames.append(
                        frame
                    )

                cap.release()

                if len(frames) != 8:

                    raise ValueError(
                        "Could not extract all 8 required frames."
                    )

                frames = np.array(
                    frames,
                    dtype=np.float32
                )

                frames = preprocess_input(
                    frames
                )

                frame_features = video_cnn.predict(
                    frames,
                    verbose=0
                )

                frame_features = frame_features.reshape(
                    1,
                    8,
                    1280
                )

                lstm_probability = float(
                    video_lstm.predict(
                        frame_features,
                        verbose=0
                    )[0][0]
                )

                gru_probability = float(
                    video_gru.predict(
                        frame_features,
                        verbose=0
                    )[0][0]
                )

                final_probability = (
                    lstm_probability
                )

                if final_probability >= 0.5:

                    result = "Accident"
                    confidence = final_probability

                else:

                    result = "Normal Traffic"
                    confidence = 1 - final_probability

                st.subheader("Prediction Result")

                if result == "Accident":

                    st.error(
                        "🚨 **Accident detected**"
                    )

                else:

                    st.success(
                        "✅ **Normal Traffic detected**"
                    )

                st.metric(
                    "Prediction Confidence",
                    f"{confidence * 100:.1f}%"
                )

                st.subheader("Model Comparison")

                v1, v2 = st.columns(2)

                with v1:

                    st.metric(
                        "LSTM Score",
                        f"{lstm_probability:.3f}"
                    )

                with v2:

                    st.metric(
                        "GRU Score",
                        f"{gru_probability:.3f}"
                    )

                st.caption(
                    "The final video prediction uses LSTM because it "
                    "achieved 98.00% test accuracy versus 97.00% for GRU."
                )

            except Exception as e:

                st.error(
                    f"Video prediction error: {e}"
                )

            finally:

                if temp_video is not None:

                    try:
                        os.remove(
                            temp_video.name
                        )
                    except OSError:
                        pass
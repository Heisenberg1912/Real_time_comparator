import os
import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import requests

# Ensure that set_page_config is the first Streamlit command executed
st.set_page_config(page_title="Chassis Inspector", layout="wide", page_icon="🔧")

# Define model URL (Replace with your actual Hugging Face or Google Drive link)
MODEL_URL = "https://huggingface.co/tusharb007/model/resolve/main/model.pt"
MODEL_PATH = "model.pt"

# Download the model if it doesn't exist
def download_model():
    if not os.path.exists(MODEL_PATH):
        st.info("Downloading model... Please wait ⏳")
        response = requests.get(MODEL_URL, stream=True)
        if response.status_code == 200:
            with open(MODEL_PATH, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            st.success("Model downloaded successfully! ✅")
        else:
            st.error("Failed to download model. Check the URL.")

# Load YOLO model
def load_model():
    return YOLO(MODEL_PATH)

# Part nomenclature mapping
NOMENCLATURE = {
    "07": "IA348549", "08": "IA349043", "09": "IA353945", "25": "IC318330",
    "34": "IC379896", "35": "IC382160", "40": "IC391070", "41": "IC391071",
    "42": "IC392312", "43": "IC392313", "47": "IC399170", "64": "IC411673",
    "74": "IC518851", "102": "IC800958", "108": "IC801489", "117": "ID366902",
    "118": "ID369862", "120": "ID602785", "121": "ID602786", "122": "ID603820",
    "123": "ID606124", "130": "IE303129"
}

EXPECTED = {
    "left": ["43", "64", "130", "41", "25", "108", "35", "117", "102", "123", "09"],
    "right": ["74", "34", "35", "122", "08", "47", "118", "121", "120", "40", "64", "130", "07"]
}

def normalize_part_number(part):
    return f"{int(part):02d}" if part.isdigit() else part

def preprocess_image(image):
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

def detect_side(detected_classes):
    detected_classes = [normalize_part_number(p) for p in detected_classes]
    left_markers, right_markers = {"09", "123", "102"}, {"07", "08", "121"}
    left_score = sum(cls in left_markers for cls in detected_classes)
    right_score = sum(cls in right_markers for cls in detected_classes)
    return "LEFT" if left_score > right_score else "RIGHT"

def detect_objects(model, image):
    results, detected = model(image), []
    for result in results:
        for box in result.boxes:
            cls = normalize_part_number(model.names[int(box.cls.item())])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            color = tuple(np.random.randint(0, 255, size=3).tolist())
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)
            cv2.putText(image, f"{cls} ({NOMENCLATURE.get(cls, 'Unknown')})", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
            detected.append(cls)
    return image, list(set(detected))

def show_model_metrics():
    st.title("Model Metrics and Training Graphs")
    st.write("This page provides an overview of key model evaluation metrics and training performance graphs.")
    
    graph_names = [
        "Precision-Recall Curve", "Recall-Confidence Curve", "F1-Confidence Curve", "Confusion Matrix",
        "Confusion Matrix (Normalized)", "Instance Distribution and Bounding Box Overlays",
        "Pairwise Scatter Plots of Bounding Box Coordinates", "Training and Validation Loss Curves"
    ]
    
    col1, col2 = st.columns(2)
    for i, graph in enumerate(graph_names):
        with (col1 if i % 2 == 0 else col2):
            st.image(f"./graphs/{graph}.png", caption=graph, width=400, use_container_width=True)

def show_chassis_inspector():
    st.title("Automotive Chassis Component Inspector")
    uploaded_file = st.file_uploader("Upload Chassis Image", type=["jpg", "png", "jpeg"])

    if uploaded_file:
        model, img = load_model(), Image.open(uploaded_file)
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Original Image")
            st.image(img, width=400, use_container_width=True)
        
        img_cv = preprocess_image(img)
        processed_img, detected = detect_objects(model, img_cv)
        detected = [normalize_part_number(p) for p in detected]
        
        if len(detected) < 2:
            st.error("Invalid image! Please upload a valid chassis image.")
        else:
            chassis_side = detect_side(detected)
            expected, missing = EXPECTED[chassis_side.lower()], list(set(EXPECTED[chassis_side.lower()]) - set(detected))

            reference_image = f"./reference images/{chassis_side.lower()}_reference.jpg"
            if os.path.exists(reference_image):
                with col2:
                    st.subheader("Reference Image")
                    st.image(reference_image, width=400, use_container_width=True)
            
            st.subheader("Processed Image")
            st.image(cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB), use_container_width=True)
            st.markdown(f"""<h3 style='text-align:center; font-weight:bold;'>IDENTIFIED CHASSIS SIDE: {chassis_side}</h3>""", unsafe_allow_html=True)
            
            col3, col4 = st.columns(2)

            with col3:
                st.error("### Missing Components")
                if missing:
                    missing_data = [(m, NOMENCLATURE.get(m, "Unknown")) for m in missing]
                    st.table({"Part Number": [m[0] for m in missing_data], "Component ID": [m[1] for m in missing_data]})
                else:
                    st.markdown("**No missing components**")

            with col4:
                st.success("### Detected Components")
                if detected:
                    detected_data = [(d, NOMENCLATURE.get(d, "Unknown")) for d in detected]
                    st.table({"Part Number": [d[0] for d in detected_data], "Component ID": [d[1] for d in detected_data]})
                else:
                    st.markdown("**No detected components**")

def main():
    # Now that the page config is set, we can safely call other Streamlit commands.
    download_model()
    
    st.sidebar.title("Chassis Inspector")
    page = st.sidebar.radio("Select Page", ["Chassis Inspector", "Model Metrics"])
    
    if page == "Chassis Inspector":
        show_chassis_inspector()
    elif page == "Model Metrics":
        show_model_metrics()

if __name__ == "__main__":
    main()

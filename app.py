import os
import streamlit as st
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO
import requests

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

# Run model download before loading YOLO
download_model()

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

def detect_side(detected_classes):
    detected_classes = [normalize_part_number(p) for p in detected_classes]
    left_markers, right_markers = {"09", "123", "102"}, {"07", "08", "121"}
    left_score = sum(cls in left_markers for cls in detected_classes)
    right_score = sum(cls in right_markers for cls in detected_classes)
    return "LEFT" if left_score > right_score else "RIGHT"

def draw_bounding_boxes(image, results):
    """ Draws bounding boxes using PIL instead of OpenCV """
    draw = ImageDraw.Draw(image)
    detected = []
    
    for result in results:
        for box in result.boxes:
            cls = normalize_part_number(result.names[int(box.cls.item())])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            detected.append(cls)

            # Generate random colors for boxes
            color = tuple(np.random.randint(100, 255, size=3).tolist())

            # Draw rectangle
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

            # Label text
            label = f"{cls} ({NOMENCLATURE.get(cls, 'Unknown')})"
            draw.text((x1, y1 - 10), label, fill=color)

    return image, list(set(detected))

def show_chassis_inspector():
    st.title("Automotive Chassis Component Inspector")
    uploaded_file = st.file_uploader("Upload Chassis Image", type=["jpg", "png", "jpeg"])

    if uploaded_file:
        model = load_model()
        img = Image.open(uploaded_file).convert("RGB")

        col1, col2 = st.columns([1, 1])
        with col1:
            st.subheader("Original Image")
            st.image(img, width=400, use_container_width=True)
        
        results = model(np.array(img))
        processed_img, detected = draw_bounding_boxes(img, results)
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
            st.image(processed_img, use_container_width=True)
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
    st.set_page_config(page_title="Chassis Inspector", layout="wide", page_icon="🔧")
    with st.sidebar:
        page = st.radio("Select Page", ["Chassis Inspector"])
    
    if page == "Chassis Inspector":
        show_chassis_inspector()

if __name__ == "__main__":
    main()

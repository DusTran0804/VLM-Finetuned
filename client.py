import streamlit as st
import cv2
import numpy as np
import requests
import base64
import time
import os

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="LPR Client", layout="wide")
st.title("License Plate Recognition - API Client")

st.sidebar.header("Mode")
mode = st.sidebar.radio("Choose mode", ("Image Upload", "Video Upload"))

max_boxes = st.sidebar.slider("Max plates to display", 1, 10, 1)

if mode == "Image Upload":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        col1, col2 = st.columns([1, 1])
        
        with st.spinner("Processing image on server..."):
            try:
                # Reset file pointer
                uploaded_file.seek(0)
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                data = {"max_boxes": max_boxes}
                
                response = requests.post(f"{API_URL}/upload/image", files=files, data=data)
                
                if response.status_code == 200:
                    result = response.json()
                    plates = result.get("plates", [])
                    process_time = result.get("process_time", 0.0)
                    
                    if not plates:
                        with col1:
                            st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="Original image", use_column_width=True)
                        with col2:
                            st.warning("No plates detected.")
                    else:
                        annotated_image = image.copy()
                        
                        for p in plates:
                            text_clean = p["plate_text"]
                            x1, y1, x2, y2 = p["bbox"]
                            
                            # Draw box and label on annotated_image
                            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(annotated_image, text_clean, (x1, max(25, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                        
                        with col1:
                            st.image(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB), caption="Processed image", use_column_width=True)
                        
                        with col2:
                            for i, p in enumerate(plates):
                                plate_bytes = base64.b64decode(p["plate_image_base64"])
                                plate_np = np.frombuffer(plate_bytes, np.uint8)
                                plate_img = cv2.imdecode(plate_np, cv2.IMREAD_COLOR)
                                
                                st.image(cv2.cvtColor(plate_img, cv2.COLOR_BGR2RGB))
                                st.markdown(
                                    f"<h3 style='color:red; text-align:left;'>Plate #{i+1}: {p['plate_text']}</h3>",
                                    unsafe_allow_html=True
                                )
                            
                            st.write(f"Thời gian xử lý trên server: {process_time:.2f}s")
                else:
                    st.error(f"Error from server: {response.status_code} - {response.text}")
            except Exception as e:
                st.error(f"Failed to connect to server: {e}")

elif mode == "Video Upload":
    process_every_n_frame = st.sidebar.slider("Process every N-th frame (video)", 1, 30, 5)
    
    uploaded_video = st.file_uploader("Upload a video", type=["mp4", "avi", "mov", "mkv"])
    if uploaded_video is not None:
        status_placeholder = st.empty()
        status_placeholder.info("⏳ Đang gửi video lên server để xử lý, vui lòng chờ...")
        
        try:
            uploaded_video.seek(0)
            files = {"file": (uploaded_video.name, uploaded_video.getvalue(), uploaded_video.type)}
            data = {"max_boxes": max_boxes, "process_every_n_frame": process_every_n_frame}
            
            response = requests.post(f"{API_URL}/upload/video", files=files, data=data)
            
            if response.status_code == 200:
                result = response.json()
                detected_plates = result.get("plates", [])
                process_time = result.get("process_time", 0.0)
                
                if detected_plates:
                    st.markdown("### Biển số nhận diện được")
                    
                    cols_per_row = 4 
                    rows = (len(detected_plates) + cols_per_row - 1) // cols_per_row
                    idx = 0
                    for r in range(rows):
                        cols = st.columns(cols_per_row)
                        for c in range(cols_per_row):
                            if idx < len(detected_plates):
                                p = detected_plates[idx]
                                plate_bytes = base64.b64decode(p["plate_image_base64"])
                                plate_np = np.frombuffer(plate_bytes, np.uint8)
                                plate_img = cv2.imdecode(plate_np, cv2.IMREAD_COLOR)
                                
                                with cols[c]:
                                    st.image(
                                        cv2.cvtColor(plate_img, cv2.COLOR_BGR2RGB),
                                        caption=f"**{p['plate_text']}**",
                                        use_column_width=True,
                                    )
                                idx += 1
                                
                status_placeholder.success(f"Thời gian xử lý trên server: {process_time:.2f}s")
            else:
                status_placeholder.error(f"Error from server: {response.status_code} - {response.text}")
        except Exception as e:
            status_placeholder.error(f"Failed to connect to server: {e}")

import streamlit as st
from ultralytics import YOLO
import cv2
import time
from PIL import Image
import numpy as np
import re
import threading
import tempfile
import os

@st.cache_resource
def load_models(yolo_path="Models/license_plate_detector_yolov8.pt", unsloth_path="Models/unsloth_finetune"):
    yolo = YOLO(yolo_path)
    try:
        import torch
        if not torch.cuda.is_available():
            raise ImportError("CUDA is not available, falling back to standard transformers + peft")
        from unsloth import FastVisionModel
        ocr_model, ocr_tokenizer = FastVisionModel.from_pretrained(model_name=unsloth_path, load_in_4bit=True)
        FastVisionModel.for_inference(ocr_model)
    except (ImportError, ModuleNotFoundError):
        import torch
        from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
        from peft import PeftModel

        device = "mps" if torch.backends.mps.is_available() else "cpu"

        ocr_tokenizer = AutoProcessor.from_pretrained(unsloth_path)

        torch_dtype = torch.float16 if device == "mps" else torch.float32
        
        base_model = Qwen2VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2-VL-2B-Instruct",
            torch_dtype=torch_dtype,
            device_map=None
        ).to(device)

        ocr_model = PeftModel.from_pretrained(base_model, unsloth_path)
        
    return yolo, ocr_model, ocr_tokenizer

class LicensePlateRecognizer:
    def __init__(self, yolo, ocr_model, ocr_tokenizer, device=None):
        self.yolo = yolo
        self.ocr_model = ocr_model
        self.ocr_tokenizer = ocr_tokenizer
        
        if device is None:
            import torch
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device

    def detect_plates(self, image):
        results = self.yolo.predict(image, device=self.device)[0]
        plates = []
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            h, w = image.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            plate_img = image[y1:y2, x1:x2]
            plates.append((plate_img, (x1, y1, x2, y2)))
        return plates

    def extract_text(self, plate_img):
        if plate_img is None or plate_img.size == 0:
            return ""
        image_rgb = cv2.cvtColor(plate_img, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)

        instruction = (
            "You are a world-class OCR expert specializing in recognizing all types of vehicle license plates. "
            "Extract ONLY the exact license plate text using digits (0-9), uppercase letters (A-Z), hyphen (-), and dot (.)."
        )

        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": instruction}]}]
        input_text = self.ocr_tokenizer.apply_chat_template(messages, add_generation_prompt=True)

        inputs = self.ocr_tokenizer(pil_image, input_text, add_special_tokens=False, return_tensors="pt").to(self.device)
        outputs = self.ocr_model.generate(**inputs, max_new_tokens=32, temperature=1.0, min_p=0.1)
        output_text = self.ocr_tokenizer.decode(outputs[0], skip_special_tokens=True)
        return output_text.split("assistant")[-1].strip()

    def preprocess_plate_text(self, text: str) -> str:
        if not text:
            return ""
        text = text.strip().upper()
        return re.sub(r'[^A-Z0-9\-.]', '', text)

class VideoCaptureThread:
    def __init__(self, src=0):
        self.src = src
        self.cap = None
        self.running = False
        self.frame = None
        self.lock = threading.Lock()

    def start(self):
        self.cap = cv2.VideoCapture(self.src)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video source {self.src}")
        self.running = True
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.1)
                continue
            with self.lock:
                self.frame = frame
        if self.cap:
            self.cap.release()

    def read(self):
        with self.lock:
            return None if self.frame is None else self.frame.copy()

    def stop(self):
        self.running = False

st.set_page_config(page_title="LPR - Real-time", layout="wide")
st.title("License Plate Recognition - Image & Real-time Stream")

with st.spinner("Loading models (YOLO + OCR)... this can take a while"):
    yolo_model, ocr_model, ocr_tokenizer = load_models()
    recognizer = LicensePlateRecognizer(yolo_model, ocr_model, ocr_tokenizer)

st.sidebar.header("Mode")
mode = st.sidebar.radio("Choose mode", ("Image Upload", "Video Upload", "Webcam (local)", "RTSP / IP Camera"))

display_fps = st.sidebar.checkbox("Show FPS", value=True)
show_boxes = st.sidebar.checkbox("Show bounding boxes & text", value=True)
max_boxes = st.sidebar.slider("Max plates to display per frame", 1, 10, 1)
process_every_n_frame = st.sidebar.slider("Process every N-th frame (video)", 1, 30, 5)

if mode == "Image Upload":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        plates = recognizer.detect_plates(image)
        
        col1, col2 = st.columns([1, 1])
        
        if not plates:
            with col1:
                st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="Original image", use_column_width=True)
            with col2:
                st.warning("No plates detected.")
        else:
            start = time.time()
            annotated_image = image.copy()
            processed_plates_info = []
            
            for i, (plate_img, (x1, y1, x2, y2)) in enumerate(plates[:max_boxes]):
                text = recognizer.extract_text(plate_img)
                text_clean = recognizer.preprocess_plate_text(text)
                processed_plates_info.append((plate_img, text_clean, (x1, y1, x2, y2)))

                cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(annotated_image, text_clean, (x1, max(25, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            
            elapsed = time.time() - start
            
            with col1:
                st.image(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB), caption="Processed image", use_column_width=True)
            
            with col2:
                for i, (plate_img, text_clean, (x1, y1, x2, y2)) in enumerate(processed_plates_info):
                    st.image(cv2.cvtColor(plate_img, cv2.COLOR_BGR2RGB))
                    st.markdown(
                        f"<h3 style='color:red; text-align:left;'>Plate #{i+1}: {text_clean}</h3>",
                        unsafe_allow_html=True
                    )
                
                st.write('\nThời gian xử lý: {:02d}:{:02d}:{:02d}'.format(
                    int(elapsed // 3600),
                    int((elapsed % 3600) // 60),
                    int(elapsed % 60)
                ))

                try:
                    import csv
                    from datetime import datetime

                    os.makedirs("Result", exist_ok=True)
                    
                    base_name = os.path.splitext(uploaded_file.name)[0]

                    annotated_save_path = f"Result/{base_name}_annotated.jpg"
                    cv2.imwrite(annotated_save_path, annotated_image)
                    
                    csv_path = "Result/results_log.csv"
                    file_exists = os.path.exists(csv_path)
                    
                    with open(csv_path, mode="a", newline="", encoding="utf-8") as f:
                        writer = csv.writer(f)
                        if not file_exists:
                            writer.writerow(["Timestamp", "Source File", "Plate Index", "Plate Text", "Bounding Box", "Annotated Image Path", "Cropped Plate Path"])
                        
                        for i, (plate_img, text_clean, (x1, y1, x2, y2)) in enumerate(processed_plates_info):
                            plate_save_path = f"Result/{base_name}_plate_{i+1}_{text_clean}.jpg"
                            cv2.imwrite(plate_save_path, plate_img)

                            writer.writerow([
                                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                uploaded_file.name,
                                i + 1,
                                text_clean,
                                f"({x1},{y1},{x2},{y2})",
                                annotated_save_path,
                                plate_save_path
                            ])
                    
                    st.success(f"Đã lưu kết quả vào thư mục `Result/`!")
                    st.info(f"Xem ảnh đã vẽ khung tại: `{annotated_save_path}`")
                except Exception as e:
                    st.error(f"Lỗi khi lưu kết quả: {e}")

elif mode == "Video Upload":
    uploaded_video = st.file_uploader("Upload a video", type=["mp4", "avi", "mov", "mkv"])
    if uploaded_video is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_video.read())
        tfile.flush()

        cap = cv2.VideoCapture(tfile.name)
        fps = cap.get(cv2.CAP_PROP_FPS)

        status_placeholder = st.empty()
        status_placeholder.info("Đang xử lý video, vui lòng chờ...")

        frame_count = 0
        start_time = time.time()

        detected_plates = []      
        seen_texts = set()     
        plates = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1

            if frame_count % process_every_n_frame == 0:
                plates = recognizer.detect_plates(frame)
                for (plate_img, (x1, y1, x2, y2)) in plates[:max_boxes]:
                    text = recognizer.extract_text(plate_img)
                    text_clean = recognizer.preprocess_plate_text(text)

                    if text_clean.strip() != "":
                        if text_clean not in seen_texts:
                            seen_texts.add(text_clean)
                            detected_plates.append((plate_img.copy(), text_clean))

        cap.release()

        if detected_plates:
            st.markdown("### Biển số nhận diện được")

            try:
                import csv
                from datetime import datetime
                
                os.makedirs("Result", exist_ok=True)
                
                base_name = os.path.splitext(uploaded_video.name)[0]
                csv_path = "Result/results_log.csv"
                file_exists = os.path.exists(csv_path)
                
                with open(csv_path, mode="a", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    if not file_exists:
                        writer.writerow(["Timestamp", "Source File", "Plate Index", "Plate Text", "Bounding Box", "Annotated Image Path", "Cropped Plate Path"])
                    
                    for idx, (plate_img, text_clean) in enumerate(detected_plates):
                        plate_save_path = f"Result/{base_name}_video_plate_{idx+1}_{text_clean}.jpg"
                        cv2.imwrite(plate_save_path, plate_img)

                        writer.writerow([
                            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            uploaded_video.name,
                            idx + 1,
                            text_clean,
                            "N/A (video detection)",
                            "N/A",
                            plate_save_path
                        ])
                st.success(f"Đã lưu {len(detected_plates)} biển số nhận diện vào thư mục `Result/` và file nhật ký `results_log.csv`!")
            except Exception as e:
                st.error(f"Lỗi khi lưu kết quả video: {e}")

            cols_per_row = 4 
            rows = (len(detected_plates) + cols_per_row - 1) // cols_per_row
            idx = 0
            for r in range(rows):
                cols = st.columns(cols_per_row)
                for c in range(cols_per_row):
                    if idx < len(detected_plates):
                        plate_img, text_clean = detected_plates[idx]
                        with cols[c]:
                            st.image(
                                cv2.cvtColor(plate_img, cv2.COLOR_BGR2RGB),
                                caption=f"**{text_clean}**",
                                use_column_width=True,
                            )
                        idx += 1

        elapsed = time.time() - start_time
        status_placeholder.success(
            '\nThời gian xử lý: {:02d}:{:02d}:{:02d}'.format(
                int(elapsed // 3600),
                int((elapsed % 3600) // 60),
                int(elapsed % 60),
            )
        )

        print("\nDone!")
                 
elif mode in ("Webcam (local)", "RTSP / IP Camera"):
    if mode == "Webcam (local)":
        st.warning("Cảnh báo: Tùy chọn Webcam này sẽ KHÔNG hoạt động khi chạy trên Google Colab, do máy chủ không thể truy cập camera của bạn.")
        src = st.sidebar.text_input("Webcam index", "0")
    else:
        src = st.sidebar.text_input("RTSP/HTTP URL", "rtsp://username:password@192.168.x.x:554/stream")

    start_button = st.button("Start Stream")
    stop_button = st.button("Stop Stream")

    video_slot = st.empty()
    info_slot = st.empty()

    if "video_thread" not in st.session_state:
        st.session_state.video_thread = None

    if start_button:
        try:
            source = int(src) if mode == "Webcam (local)" and str(src).isdigit() else src
            vt = VideoCaptureThread(source)
            vt.start()
            st.session_state.video_thread = vt
            info_slot.success("Streaming started")
        except Exception as e:
            st.session_state.video_thread = None
            info_slot.error(f"Failed to start stream: {e}")

    if stop_button and st.session_state.video_thread is not None:
        st.session_state.video_thread.stop()
        st.session_state.video_thread = None
        info_slot.info("Streaming stopped")

    if st.session_state.video_thread is not None:
        last_time = time.time()
        fps = 0.0
        try:
            while st.session_state.video_thread is not None and st.session_state.video_thread.running:
                frame = st.session_state.video_thread.read()
                if frame is None:
                    time.sleep(0.05)
                    continue

                start_proc = time.time()
                plates = recognizer.detect_plates(frame)

                for i, (plate_img, (x1, y1, x2, y2)) in enumerate(plates[:max_boxes]):
                    text = recognizer.extract_text(plate_img)
                    text_clean = recognizer.preprocess_plate_text(text)
                    if show_boxes:
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, text_clean, (x1, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

                if display_fps:
                    now = time.time()
                    fps = 0.9 * fps + 0.1 * (1.0 / max(1e-6, now - last_time))
                    last_time = now
                    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

                video_slot.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), use_column_width=True)

                time.sleep(0.03)
        except Exception as e:
            info_slot.error(f"Stream error: {e}")
            st.session_state.video_thread = None

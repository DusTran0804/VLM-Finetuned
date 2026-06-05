import cv2
import time
import numpy as np
import re
import tempfile
import os
import base64
from fastapi import FastAPI, UploadFile, File, Form
from pydantic import BaseModel
from typing import List, Optional
from ultralytics import YOLO
from PIL import Image

# Initialize FastAPI app
app = FastAPI(title="License Plate Recognition API")

# Global variables for models
yolo_model = None
ocr_model = None
ocr_tokenizer = None
recognizer = None

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
        
        # Determine device (mps for Apple Silicon, cpu otherwise)
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        
        # Load processor
        ocr_tokenizer = AutoProcessor.from_pretrained(unsloth_path)
        
        # Load base model in float16 for Apple Silicon MPS, or float32 for CPU
        torch_dtype = torch.float16 if device == "mps" else torch.float32
        
        base_model = Qwen2VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2-VL-2B-Instruct",
            torch_dtype=torch_dtype,
            device_map=None
        ).to(device)
        
        # Load LoRA adapter
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

@app.on_event("startup")
async def startup_event():
    global yolo_model, ocr_model, ocr_tokenizer, recognizer
    print("Loading models... This may take a while.")
    yolo_model, ocr_model, ocr_tokenizer = load_models()
    recognizer = LicensePlateRecognizer(yolo_model, ocr_model, ocr_tokenizer)
    print("Models loaded successfully.")

def encode_image_base64(image):
    _, buffer = cv2.imencode('.jpg', image)
    return base64.b64encode(buffer).decode('utf-8')

class PlateResult(BaseModel):
    plate_text: str
    bbox: List[int]  # [x1, y1, x2, y2]
    plate_image_base64: str

class ImageResponse(BaseModel):
    plates: List[PlateResult]
    process_time: float

@app.post("/upload/image", response_model=ImageResponse)
async def process_image(file: UploadFile = File(...), max_boxes: int = Form(10)):
    start_time = time.time()
    
    # Read image
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if image is None:
        return {"error": "Invalid image file"}

    plates = recognizer.detect_plates(image)
    results = []
    
    for plate_img, (x1, y1, x2, y2) in plates[:max_boxes]:
        text = recognizer.extract_text(plate_img)
        text_clean = recognizer.preprocess_plate_text(text)
        
        base64_img = encode_image_base64(plate_img)
        
        results.append(PlateResult(
            plate_text=text_clean,
            bbox=[x1, y1, x2, y2],
            plate_image_base64=base64_img
        ))
        
    process_time = time.time() - start_time
    return ImageResponse(plates=results, process_time=process_time)

@app.post("/upload/video")
async def process_video(file: UploadFile = File(...), max_boxes: int = Form(10), process_every_n_frame: int = Form(5)):
    start_time = time.time()
    
    # Save uploaded video to temp file
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(await file.read())
    tfile.flush()
    
    cap = cv2.VideoCapture(tfile.name)
    frame_count = 0
    detected_plates = []
    seen_texts = set()
    
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
                        base64_img = encode_image_base64(plate_img)
                        detected_plates.append({
                            "plate_text": text_clean,
                            "plate_image_base64": base64_img
                        })
                        
    cap.release()
    os.unlink(tfile.name)
    
    process_time = time.time() - start_time
    return {"plates": detected_plates, "process_time": process_time}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

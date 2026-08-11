import cv2
import torch
from PIL import Image
from model import CancerCNN
from utils import preprocess
import os

# ---------------- CONFIG ----------------
MODEL_PATH = "models/oral_model.pth"
OUTPUT_VIDEO = "oral_webcam_record.avi"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- LOAD MODEL ----------------
model = CancerCNN().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

print("✅ Oral cancer model loaded")
print("📷 Webcam + recording started (press 'q' to stop)")

# ---------------- OPEN WEBCAM ----------------
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Could not access webcam")
    exit()

# ---------------- GET FRAME SIZE ----------------
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = 20.0

# ---------------- VIDEO WRITER ----------------
fourcc = cv2.VideoWriter_fourcc(*"XVID")
out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (width, height))

print(f"🎥 Recording video to: {OUTPUT_VIDEO}")

# ---------------- LOOP ----------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to RGB for model
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)
    input_tensor = preprocess(pil_img).to(DEVICE)

    with torch.no_grad():
        logits = model(input_tensor)
        prob = torch.sigmoid(logits).item()

    confidence = int(prob * 100)

    # Higher threshold to avoid flagging healthy as cancer too easily
    if prob > 0.7:
        label = "Cancer Detected"
        color = (0, 0, 255)
    else:
        label = "Normal"
        color = (0, 255, 0)

    cv2.putText(
        frame,
        f"{label} ({confidence}%)",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        color,
        2
    )

    # Show webcam
    cv2.imshow("Oral Cancer Webcam Detection", frame)

    # 🔴 WRITE FRAME TO VIDEO FILE
    out.write(frame)

    # Quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# ---------------- CLEANUP ----------------
cap.release()
out.release()
cv2.destroyAllWindows()

print("🛑 Webcam stopped, video saved.")
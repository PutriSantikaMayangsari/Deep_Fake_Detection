# modules/debug_preprocessing.py
# ini buat nge-debug preprocessing, karena pas mau prepare dataset suka error

import os
import cv2
import torch
import numpy as np
from PIL import Image
from facenet_pytorch import MTCNN

# ambil config path sederhana (langsung ke beberapa video saja)
REAL_PATH = "./datasets/ff-c23/FaceForensics++_C23/original"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

mtcnn = MTCNN(keep_all=False, post_process=False, min_face_size=20, device=device)

# ambil 1 video untuk debug
videos = [os.path.join(REAL_PATH, f) for f in os.listdir(REAL_PATH) if f.endswith(".mp4")]
if not videos:
    raise FileNotFoundError("Tidak ada file .mp4 di folder REAL_PATH")

video_path = videos[0]
print(f"Debugging video: {video_path}")

cap = cv2.VideoCapture(video_path)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"Frame count: {frame_count}")

# ambil 3 frame pertama untuk sample
frame_indices = np.linspace(0, frame_count - 1, 3, dtype=int)

for i in frame_indices:
    cap.set(cv2.CAP_PROP_POS_FRAMES, i)
    ret, frame = cap.read()
    if not ret:
        continue

    frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    face_tensor = mtcnn(frame_pil)

    print(f"\nFrame {i}:")
    print("  type:", type(face_tensor))
    if isinstance(face_tensor, torch.Tensor):
        print("  shape:", face_tensor.shape, "dtype:", face_tensor.dtype, "min:", face_tensor.min().item(), "max:", face_tensor.max().item())
    elif isinstance(face_tensor, np.ndarray):
        print("  shape:", face_tensor.shape, "dtype:", face_tensor.dtype, "min:", face_tensor.min(), "max:", face_tensor.max())
    elif isinstance(face_tensor, Image.Image):
        print("  mode:", face_tensor.mode, "size:", face_tensor.size)
    else:
        print("  value:", face_tensor)

cap.release()

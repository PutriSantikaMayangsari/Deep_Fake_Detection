# modules/preprocessing.py

import os
import cv2
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from sklearn.model_selection import train_test_split
from skimage.feature import local_binary_pattern


# gaussian + LBP preprocessing
def apply_gaussian_and_lbp(image: Image.Image) -> Image.Image:
    """Convert image ke grayscale, Gaussian blur, lalu LBP, hasilkan 3-channel PIL.Image."""
    img_np_gray = np.array(image.convert("L"))
    img_blur = cv2.GaussianBlur(img_np_gray, (3, 3), 0)
    lbp = local_binary_pattern(img_blur, 24, 3, method="uniform")

    max_val = np.max(lbp)
    if max_val > 0:
        lbp = (lbp / max_val * 255).astype(np.uint8)
    else:
        lbp = lbp.astype(np.uint8)

    lbp_rgb = cv2.cvtColor(lbp, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(lbp_rgb)


# data preparation
def prepare_sequence_dataset(config, mtcnn):
    all_sequences, all_labels = [], []

    def center_crop(frame_pil: Image.Image, size: int = 224) -> Image.Image:
        """Fallback crop tengah kalau MTCNN gagal."""
        w, h = frame_pil.size
        min_side = min(w, h)
        left = (w - min_side) // 2
        top = (h - min_side) // 2
        right = left + min_side
        bottom = top + min_side
        cropped = frame_pil.crop((left, top, right, bottom))
        return cropped.resize((size, size))

    def process_videos(video_paths, label, desc):
        for video_path in tqdm(video_paths, desc=desc):
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"Cannot open video: {video_path}")
                continue

            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if frame_count < config["min_frames_threshold"]:
                print(f"Skipping {video_path}, only {frame_count} frames")
                cap.release()
                continue

            frame_indices = np.linspace(
                0, frame_count - 1, config["frames_per_video"], dtype=int
            )

            current_sequence = []
            for i in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ret, frame = cap.read()
                if not ret:
                    print(f"Failed to read frame {i} from {video_path}")
                    continue

                try:
                    # convert frame → PIL RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame_pil = Image.fromarray(frame_rgb).convert("RGB")

                    # detect face
                    face_tensor = mtcnn(frame_pil)
                    if face_tensor is not None:
                        # convert tensor float [0,1] → numpy uint8 → PIL
                        face_tensor = face_tensor.float().clamp(0, 1)
                        face_np = (
                            face_tensor.permute(1, 2, 0).cpu().numpy() * 255
                        ).astype(np.uint8)
                        face_pil = Image.fromarray(face_np)
                    else:
                        # fallback ke center crop
                        print(f"No face detected in frame {i} of {video_path}, using center crop")
                        face_pil = center_crop(frame_pil, config["image_size"])

                    # apply LBP preprocessing
                    lbp_image = apply_gaussian_and_lbp(face_pil)
                    current_sequence.append(lbp_image)

                except Exception as e:
                    print(f"Error processing frame {i} in {video_path}: {e}")
                    continue

            cap.release()

            # simpan sequence kalau cukup panjang
            if len(current_sequence) >= config["min_frames_threshold"]:
                processed_sequence = current_sequence[: config["frames_per_video"]]
                while len(processed_sequence) < config["frames_per_video"]:
                    processed_sequence.append(processed_sequence[-1])

                all_sequences.append(processed_sequence)
                all_labels.append(label)
            else:
                print(f"Sequence too short for {video_path}, got {len(current_sequence)} frames")

    # real videos
    real_videos = [
        os.path.join(config["real_data_path"], f)
        for f in os.listdir(config["real_data_path"])
        if f.endswith(".mp4")
    ][: config["max_videos_per_class"]]
    process_videos(real_videos, 0, "Processing real videos")

    # fake videos
    fake_videos = [
        os.path.join(config["fake_data_path"], f)
        for f in os.listdir(config["fake_data_path"])
        if f.endswith(".mp4")
    ][: config["max_videos_per_class"]]
    process_videos(fake_videos, 1, "Processing fake videos")

    if not all_sequences:
        print("\nWARNING: Dataset is empty after processing. Check video paths and quality.")
        return [], [], [], [], [], []

    # split dataset
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        all_sequences, all_labels,
        test_size=0.20, random_state=config["seed"], stratify=all_labels
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val,
        test_size=0.20, random_state=config["seed"], stratify=y_train_val
    )

    return X_train, X_val, X_test, y_train, y_val, y_test

# config.py

CONFIG = {
    "real_data_path": "./datasets/ff-c23/FaceForensics++_C23/original",
    "fake_data_path": "./datasets/ff-c23/FaceForensics++_C23/Deepfakes",
    
    # Parameter Data
    "max_videos_per_class": 1000, 
    "frames_per_video": 10,   # bestnya 20    
    "min_frames_threshold": 10, # Minimal frame valid agar video diproses

    # Parameter Model & Training
    "image_size": 224, # Sesuaikan dengan model (224 untuk Xception)
    "batch_size": 4, # Sesuaikan dengan VRAM GPU         
    "epochs": 30,
    "learning_rate": 1e-4,
    "model_name": "xception",
    "lstm_hidden_size": 512,
    
    # Lain-lain
    "seed": 42,
    "model_save_path": "./model_checkpoints_spatiotemporal",
    "early_stopping_patience": 5,
    "weight_decay": 1e-5,
}
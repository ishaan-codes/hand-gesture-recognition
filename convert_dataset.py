import os
import cv2
import h5py
import numpy as np
import mediapipe as mp
from tqdm import tqdm

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.5
)

DATASET_PATH = "./SignAlphaSet"
OUTPUT_FILE = "hand_gesture_dataset.h5"
IMG_SIZE = (296, 296)
TEST_SPLIT = 0.2
SEED = 42

class_names = sorted(os.listdir(DATASET_PATH))
class_to_idx = {cls_name: idx for idx, cls_name in enumerate(class_names)}
num_classes = len(class_names)

print(f"Found {num_classes} classes: {class_names}")

with h5py.File(OUTPUT_FILE, 'w') as hf:
    hf.create_dataset('train/landmarks', shape=(0, 63), maxshape=(None, 63), dtype='float32')
    hf.create_dataset('train/labels', shape=(0,), maxshape=(None,), dtype='int32')
    hf.create_dataset('test/landmarks', shape=(0, 63), maxshape=(None, 63), dtype='float32')
    hf.create_dataset('test/labels', shape=(0,), maxshape=(None,), dtype='int32')

    dt = h5py.special_dtype(vlen=str)
    hf.create_dataset('class_names', data=class_names, dtype=dt)

    train_count = 0
    test_count = 0

    for class_name in class_names:
        class_dir = os.path.join(DATASET_PATH, class_name)
        if not os.path.isdir(class_dir):
            continue

        image_paths = []
        for file in os.listdir(class_dir):
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_paths.append(os.path.join(class_dir, file))

        np.random.seed(SEED)
        np.random.shuffle(image_paths)
        split_idx = int(len(image_paths) * (1 - TEST_SPLIT))
        train_paths = image_paths[:split_idx]
        test_paths = image_paths[split_idx:]
        
        print(f"\nProcessing class '{class_name}': {len(image_paths)} images")
        print(f"  Train: {len(train_paths)}, Test: {len(test_paths)}")

        skipped_train = 0
        for path in tqdm(train_paths, desc=f"  Training - {class_name}"):
            img = cv2.imread(path)
            if img is None:
                skipped_train += 1
                continue
                
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, IMG_SIZE)
            results = hands.process(img)
            
            landmarks = []
            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                wrist = hand_landmarks.landmark[0]
                for lm in hand_landmarks.landmark:
                    landmarks.extend([lm.x - wrist.x, lm.y - wrist.y, lm.z - wrist.z])
            
            if landmarks:                
                new_size = train_count + 1
                hf['train/landmarks'].resize((new_size, 63))
                hf['train/labels'].resize((new_size,))
                
                hf['train/landmarks'][train_count] = landmarks
                hf['train/labels'][train_count] = class_to_idx[class_name]
                train_count += 1
            else:
                skipped_train += 1
        
        skipped_test = 0
        for path in tqdm(test_paths, desc=f"  Testing  - {class_name}"):
            img = cv2.imread(path)
            if img is None:
                skipped_test += 1
                continue
                
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, IMG_SIZE)
            results = hands.process(img)
            
            landmarks = []
            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                wrist = hand_landmarks.landmark[0]
                
                for lm in hand_landmarks.landmark:
                    landmarks.extend([lm.x - wrist.x, lm.y - wrist.y, lm.z - wrist.z])
            
            if landmarks:
                new_size = test_count + 1
                hf['test/landmarks'].resize((new_size, 63))
                hf['test/labels'].resize((new_size,))
                
                hf['test/landmarks'][test_count] = landmarks
                hf['test/labels'][test_count] = class_to_idx[class_name]
                test_count += 1
            else:
                skipped_test += 1
                
        print(f"  Skipped: {skipped_train} train, {skipped_test} test (no hands detected)")
    
    hf.attrs['description'] = f"Hand gesture dataset with {num_classes} classes"
    hf.attrs['creation_date'] = np.string_(str(np.datetime64('now')))
    hf.attrs['image_size'] = IMG_SIZE
    hf.attrs['landmark_format'] = "21 landmarks (x,y,z relative to wrist)"
    
    print("\nDataset creation complete!")
    print(f"Final counts - Train: {train_count}, Test: {test_count}")
    print(f"Saved to: {OUTPUT_FILE}")

hands.close()
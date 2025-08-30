import cv2
import numpy as np
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed
from models_loader import face_model, emotion_labels

def predict_face_frame(image):
    image = image.convert("L")
    image = image.resize((48, 48))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    img_array = np.expand_dims(img_array, axis=-1)
    prediction = face_model.predict(img_array)
    idx = np.argmax(prediction)
    return emotion_labels[idx], float(np.max(prediction)) * 100

def process_faces(file_path: str):
    results = []
    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        raise Exception("Error opening video file")
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = 0
    frame_interval = int(fps)
    futures = []
    with ThreadPoolExecutor(max_workers=10) as executor:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_count % frame_interval == 0:
                img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                futures.append(executor.submit(predict_face_frame, img))
                start_time = frame_count / fps
                end_time = (frame_count + frame_interval) / fps
                results.append({"start": round(start_time, 2), "end": round(end_time, 2)})
            frame_count += 1

        for i, future in enumerate(as_completed(futures)):
            emotion, confidence = future.result()
            results[i]["emotion"] = emotion
            results[i]["confidence_of_model"] = confidence
    cap.release()
    return results

# # # model = load_model(r"D:/GenAI-ML/Backend/Face/Model/model_78.h5")
# # # model.load_weights(r"D:/GenAI-ML/Backend/Face/Model/model_weights_78.h5")


# import logging
# from fastapi import FastAPI, UploadFile, File
# from fastapi.responses import JSONResponse
# from fastapi.middleware.cors import CORSMiddleware
# import numpy as np
# import base64
# import io
# from PIL import Image
# from tensorflow.keras.models import load_model
# import os  # Ensure os is imported
# import cv2  # OpenCV for handling video frames
# from concurrent.futures import ThreadPoolExecutor, as_completed
# import asyncio

# # Set up logging
# logging.basicConfig(level=logging.DEBUG)
# logger = logging.getLogger(__name__)

# # Load model architecture and weights
# model = load_model(r"D:/GenAI-ML/Backend/Face/Model/Face_Emotion_Recognition.keras")
# model.load_weights(r"D:/GenAI-ML/Backend/Face/Model/Face_Emotion_Recognition_Weights.weights.h5")

# # model = load_model(r"D:/GenAI-ML/Backend/Face/Model/model_78.h5")
# # model.load_weights(r"D:/GenAI-ML/Backend/Face/Model/model_weights_78.h5")

# # Emotion labels
# emotion_labels = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

# app = FastAPI()

# # Allow frontend (React) to connect
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Function to handle prediction with timeout
# def predict_with_timeout(img_array):
#     try:
#         # Perform prediction synchronously (remove async)
#         prediction = model.predict(img_array)
#         return prediction
#     except Exception as e:
#         logger.error(f"Prediction failed: {str(e)}")
#         return None

# # Helper: Process each frame from video
# def process_frame(image):
#     try:
#         image = image.convert("L")  # Grayscale
#         image = image.resize((48, 48))  # Resize to match model's input size

#         img_array = np.array(image) / 255.0  # Normalize the image
#         img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
#         img_array = np.expand_dims(img_array, axis=-1)  # Add channel dimension for grayscale (1)

#         # Run prediction with timeout (no async)
#         prediction = predict_with_timeout(img_array)

#         # Get the emotion with the highest confidence
#         emotion_index = np.argmax(prediction)  # Index with the highest confidence
#         emotion = emotion_labels[emotion_index]  # Emotion label corresponding to the index

#         confidence = float(np.max(prediction)) * 100  # Convert to percentage

#         return emotion, confidence
#     except Exception as e:
#         logger.error(f"Error in processing frame: {str(e)}")
#         return "error", 0

# @app.post("/predict/")
# async def predict(file: UploadFile = File(...)):
#     try:
#         # Read the uploaded file
#         file_data = await file.read()
#         temp_path = "temp_video.mp4"
#         with open(temp_path, 'wb') as f:
#             f.write(file_data)

#         # Initialize video capture (OpenCV)
#         cap = cv2.VideoCapture(temp_path)

#         # Check if video opened successfully
#         if not cap.isOpened():
#             raise Exception("Error opening video file")

#         # Frame rate of the video
#         fps = cap.get(cv2.CAP_PROP_FPS)

#         results = []
#         frame_count = 0
#         frame_interval = int(fps)  # Extract one frame per second

#         # Create ThreadPoolExecutor with max 10 workers
#         with ThreadPoolExecutor(max_workers=10) as executor:
#             futures = []

#             # Process the video frame by frame
#             while True:
#                 ret, frame = cap.read()
#                 if not ret:
#                     break  # End of video

#                 # Process every second (frame_interval = 1 second)
#                 if frame_count % frame_interval == 0:
#                     # Convert frame to PIL image
#                     img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

#                     # Submit frame processing to executor
#                     futures.append(executor.submit(process_frame, img))

#                     # Calculate start and end times
#                     start_time = frame_count // fps  # Seconds
#                     end_time = start_time + 1  # Add 1 second for end time

#                     # Prepare to collect results later
#                     results.append({
#                         "start": start_time,
#                         "end": end_time
#                     })

#                 frame_count += 1

#             # Collect results from the futures as they complete
#             for i, future in enumerate(as_completed(futures)):
#                 emotion, confidence = future.result()  # Use result() to get the result from the future
#                 # Update the results with emotion and confidence values
#                 results[i]["emotion"] = emotion
#                 results[i]["confidence"] = confidence

#         # Release video capture and remove the temporary file
#         cap.release()
#         os.remove(temp_path)  # Make sure the temp file is deleted

#         return JSONResponse(content={"frames": results})

#     except Exception as e:
#         logger.error(f"Error occurred: {str(e)}")
#         return {"error": str(e)}

#     except asyncio.CancelledError as ce:
#         logger.error(f"Task was canceled: {str(ce)}")
#         return {"error": "Request was canceled."}


import logging
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
import os
import cv2
from concurrent.futures import ThreadPoolExecutor, as_completed
import asyncio

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Load model architecture and weights
model = load_model(r"D:/GenAI-ML/Backend/Face/Model/Face_Emotion_Recognition.keras")
model.load_weights(r"D:/GenAI-ML/Backend/Face/Model/Face_Emotion_Recognition_Weights.weights.h5")

# Emotion labels
emotion_labels = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

app = FastAPI()

# Allow frontend (React) to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Function to handle prediction
def predict_with_timeout(img_array):
    try:
        prediction = model.predict(img_array)
        return prediction
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        return None

# Helper: Process each frame from video
def process_frame(image):
    try:
        image = image.convert("L")  # Grayscale
        image = image.resize((48, 48))  # Resize to match model's input size

        img_array = np.array(image) / 255.0  # Normalize
        img_array = np.expand_dims(img_array, axis=0)  # Batch dimension
        img_array = np.expand_dims(img_array, axis=-1)  # Channel dim (1 for grayscale)

        prediction = predict_with_timeout(img_array)

        emotion_index = np.argmax(prediction)
        emotion = emotion_labels[emotion_index]
        confidence = float(np.max(prediction)) * 100

        return emotion, confidence
    except Exception as e:
        logger.error(f"Error in processing frame: {str(e)}")
        return "error", 0

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        # Save uploaded file temporarily
        file_data = await file.read()
        temp_path = "temp_video.mp4"
        with open(temp_path, 'wb') as f:
            f.write(file_data)

        cap = cv2.VideoCapture(temp_path)

        if not cap.isOpened():
            raise Exception("Error opening video file")

        fps = cap.get(cv2.CAP_PROP_FPS)

        results = []
        frame_count = 0
        frame_interval = int(fps)  # process 1 frame per second

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = []

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_count % frame_interval == 0:
                    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

                    futures.append(executor.submit(process_frame, img))

                    # Exact start and end times (in seconds, can be fractional)
                    start_time = frame_count / fps
                    end_time = (frame_count + frame_interval) / fps

                    results.append({
                        "start": round(start_time, 2),
                        "end": round(end_time, 2)
                    })

                frame_count += 1

            # Attach results back with emotion and confidence_of_model
            for i, future in enumerate(as_completed(futures)):
                emotion, confidence = future.result()
                results[i]["emotion"] = emotion
                results[i]["confidence_of_model"] = confidence

        cap.release()
        os.remove(temp_path)

        return JSONResponse(content={"face_model_prediction": results})

    except Exception as e:
        logger.error(f"Error occurred: {str(e)}")
        return {"error": str(e)}

    except asyncio.CancelledError as ce:
        logger.error(f"Task was canceled: {str(ce)}")
        return {"error": "Request was canceled."}

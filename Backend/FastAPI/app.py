# from fastapi import FastAPI
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# import numpy as np
# import base64
# import io
# from PIL import Image
# from tensorflow.keras.models import load_model

# # Load model architecture
# model = load_model(r"D:/GenAI-ML/Backend/Face/Model/Face_Emotion_Recognition.keras")

# # Load weights separately from another file
# model.load_weights(r"D:/GenAI-ML/Backend/Face/Model/Face_Emotion_Recognition_Weights.weights.h5")

# # Emotion labels
# emotion_labels = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

# app = FastAPI()

# # Allow frontend (React) to connect
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # in production restrict this
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# class Frame(BaseModel):
#     image: str  # base64 string

# @app.post("/predict/")
# async def predict(frame: Frame):
#     try:
#         # Decode base64 image
#         image_data = base64.b64decode(frame.image.split(",")[1])
#         image = Image.open(io.BytesIO(image_data)).convert("L")  # Grayscale
#         image = image.resize((48, 48))  # Resize to match model's input size

#         img_array = np.array(image) / 255.0  # Normalize the image
#         img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
#         img_array = np.expand_dims(img_array, axis=-1)  # Add channel dimension for grayscale (1)

#         # Predict the emotion
#         prediction = model.predict(img_array)
#         emotion_index = np.argmax(prediction)
#         emotion = emotion_labels[emotion_index]

#         return {"emotion": emotion, "confidence": float(np.max(prediction))}
    
#     except Exception as e:
#         return {"error": str(e)}


# import logging
# from fastapi import FastAPI
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# import numpy as np
# import base64
# import io
# from PIL import Image
# from tensorflow.keras.models import load_model
# import asyncio

# # Set up logging
# logging.basicConfig(level=logging.DEBUG)  # DEBUG level to ensure logs are captured
# logger = logging.getLogger(__name__)

# # Load model architecture and weights
# model = load_model(r"D:/GenAI-ML/Backend/Face/Model/Face_Emotion_Recognition.keras")
# model.load_weights(r"D:/GenAI-ML/Backend/Face/Model/Face_Emotion_Recognition_Weights.weights.h5")

# # Emotion labels
# emotion_labels = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

# app = FastAPI()

# # Allow frontend (React) to connect
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # in production restrict this
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# class Frame(BaseModel):
#     image: str  # base64 string

# # Function to handle prediction with timeout
# async def predict_with_timeout(img_array):
#     try:
#         loop = asyncio.get_event_loop()
#         # Use a future to handle prediction asynchronously
#         prediction = await loop.run_in_executor(None, lambda: model.predict(img_array))
#         return prediction
#     except asyncio.TimeoutError:
#         raise Exception("Prediction took too long")

# @app.post("/predict/")
# async def predict(frame: Frame):
#     try:
#         # Decode base64 image
#         image_data = base64.b64decode(frame.image.split(",")[1])
#         image = Image.open(io.BytesIO(image_data)).convert("L")  # Grayscale
#         image = image.resize((48, 48))  # Resize to match model's input size

#         img_array = np.array(image) / 255.0  # Normalize the image
#         img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
#         img_array = np.expand_dims(img_array, axis=-1)  # Add channel dimension for grayscale (1)

#         # Log the shape of the input image
#         logger.debug(f"Input image shape: {img_array.shape}")

#         # Run prediction with timeout
#         prediction = await predict_with_timeout(img_array)

#         # Log the raw prediction for debugging
#         logger.debug(f"Raw Prediction: {prediction}")

#         # Check if prediction is valid (i.e., not NaN or empty)
#         if np.any(np.isnan(prediction)) or np.all(prediction == 0):
#             raise Exception("Invalid prediction data")

#         # Get the emotion with the highest confidence
#         emotion_index = np.argmax(prediction)
#         emotion = emotion_labels[emotion_index]
#         confidence = float(np.max(prediction)) * 100  # Convert the highest probability to a percentage

#         # Ensure confidence is between 0 and 100%
#         if confidence > 100:
#             confidence = 100

#         # Return the result
#         return {"emotion": emotion, "confidence": confidence}

#     except Exception as e:
#         logger.error(f"Error occurred: {str(e)}")  # Log any errors for better debugging
#         return {"error": str(e)}

#     except asyncio.CancelledError as ce:
#         logger.error(f"Task was canceled: {str(ce)}")  # Handle cancellation errors explicitly
#         return {"error": "Request was canceled."}

import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import base64
import io
from PIL import Image
from tensorflow.keras.models import load_model
import asyncio

# Set up logging
logging.basicConfig(level=logging.DEBUG)  # DEBUG level to ensure logs are captured
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
    allow_origins=["*"],  # in production restrict this
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Frame(BaseModel):
    image: str  # base64 string

# Function to handle prediction with timeout
async def predict_with_timeout(img_array):
    try:
        loop = asyncio.get_event_loop()
        # Use a future to handle prediction asynchronously
        prediction = await loop.run_in_executor(None, lambda: model.predict(img_array))
        return prediction
    except asyncio.TimeoutError:
        raise Exception("Prediction took too long")

@app.post("/predict/")
async def predict(frame: Frame):
    try:
        # Decode base64 image
        image_data = base64.b64decode(frame.image.split(",")[1])
        image = Image.open(io.BytesIO(image_data)).convert("L")  # Grayscale
        image = image.resize((48, 48))  # Resize to match model's input size

        img_array = np.array(image) / 255.0  # Normalize the image
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        img_array = np.expand_dims(img_array, axis=-1)  # Add channel dimension for grayscale (1)

        # Log the shape of the input image
        logger.debug(f"Input image shape: {img_array.shape}")

        # Run prediction with timeout
        prediction = await predict_with_timeout(img_array)

        # Log the raw prediction for debugging
        logger.debug(f"Raw Prediction (before scaling): {prediction}")

        # Check if prediction is valid (i.e., not NaN or empty)
        if np.any(np.isnan(prediction)) or np.all(prediction == 0):
            raise Exception("Invalid prediction data")

        # Get the emotion with the highest confidence
        emotion_index = np.argmax(prediction)  # Index with the highest confidence
        emotion = emotion_labels[emotion_index]  # Emotion label corresponding to the index

        # Log the individual prediction values
        logger.debug(f"Individual prediction values: {prediction[0]}")  # Log all emotion probabilities

        confidence = float(np.max(prediction)) * 100  # Convert to percentage

        # Log the confidence value before returning
        logger.debug(f"Confidence (before clamping): {confidence}%")

        # Ensure confidence is between 0 and 100
        if confidence > 100:
            confidence = 100

        # Return the result
        return {"emotion": emotion, "confidence": confidence}

    except Exception as e:
        logger.error(f"Error occurred: {str(e)}")  # Log any errors for better debugging
        return {"error": str(e)}

    except asyncio.CancelledError as ce:
        logger.error(f"Task was canceled: {str(ce)}")  # Handle cancellation errors explicitly
        return {"error": "Request was canceled."}

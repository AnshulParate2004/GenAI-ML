from tensorflow.keras.models import load_model

face_model = load_model(r"D:/GenAI-ML/Backend/Face/Model/Face_Emotion_Recognition.keras")
face_model.load_weights(r"D:/GenAI-ML/Backend/Face/Model/Face_Emotion_Recognition_Weights.weights.h5")

emotion_labels = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

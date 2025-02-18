import pathlib
from fastai.vision.all import load_learner, PILImage

# 🔥 FIX: Override PosixPath with WindowsPath for compatibility
pathlib.PosixPath = pathlib.WindowsPath

# ✅ Load trained FastAI model once when the module is imported
MODEL_PATH = "animal_classifier_resnet34.pkl"
learn = load_learner(str(MODEL_PATH))  # Ensure path is string format

def predict_image(image_path):
    
    try:
        # Open the image
        img = PILImage.create(image_path)
        # Get prediction from the model
        pred_class, pred_idx, probs = learn.predict(img)
        confidence = float(probs[pred_idx])
        return {str(pred_class), confidence}
    except Exception as e:
        return f"Error: {e}", 0.0  # Return error message and 0 confidence
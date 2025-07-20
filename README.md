# 🐾 Animal Classifier with Google Teachable Machine
A Small AI Model That Predict If The Photo You Upload Is a Cat, Dog, Lion or Tigger
This project is a simple and interactive image classification system trained using Google Teachable Machine. It can recognize four types of animals: cats, dogs, lions, and tigers, using images sized 224x224 pixels.

The model was exported as a .h5 file and tested using Python + TensorFlow in Google Colab with a user-friendly image upload interface.

### 🚀 Features:

🔧 Trained on Google Teachable Machine

📷 Recognizes: cat, dog, lion, tiger

⚙️ Integrated with Keras/TensorFlow for inference

☁️ Tested in Google Colab with live image uploads

📊 Shows both predicted class and confidence score

### 🧠 How It Works:

1- Upload an image of an animal.

2- The model processes it and returns:

✅ The predicted class (e.g., "tiger")

📈 The confidence score as a percentage

### 📦 Files Included:

keras_model.h5 – Trained model file

labels.txt – Label list (Teachable Machine export)

animal_classifier.ipynb – Google Colab notebook for testing

### 🧾 Code:

```python

# 🐾 Animal Classifier - Tested in Google Colab

!pip install -q tensorflow==2.12.1
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
from google.colab import files

# Load model and class names
model = load_model("/content/keras_model.h5", compile=False)
class_names = open("/content/labels.txt", "r").readlines()

print("🧠 AI is ready to classify! Upload images one by one...")

while True:
    print("\n📤 Please upload an image (dog, cat, lion, tiger):")
    uploaded = files.upload()

    for filename in uploaded.keys():
        try:
            # Load and prepare image
            image = Image.open(filename).convert("RGB")
            image = ImageOps.fit(image, (224, 224), Image.Resampling.LANCZOS)
            image_array = np.asarray(image)
            normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
            data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
            data[0] = normalized_image_array

            # Predict
            prediction = model.predict(data)
            index = np.argmax(prediction)
            class_name = class_names[index].strip()
            confidence_score = prediction[0][index]

            # Output result
            print("\n✅ Prediction complete:")
            print(f"🐾 Class: {class_name}")
            print(f"📈 Confidence Score: {confidence_score:.2%}")

        except Exception as e:
            print(f"⚠️ Error with file {filename}: {e}")

    print("\n🔁 Ready for the next image. Upload again when ready.")













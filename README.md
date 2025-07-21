# 🐾 Animal Classifier with Google Teachable Machine

## 📚 How I Built It

I built this animal classifier using Google Teachable Machine, where I trained a custom image recognition model with pictures of cats, dogs, lions, and tigers, each resized to 224×224 pixels. After training, I exported the model (.h5 format) and used TensorFlow/Keras in Google Colab to load the model and make predictions. I implemented an interactive Python script that lets users upload an image and get real-time predictions along with confidence scores.


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

```

<img width="1920" height="927" alt="HOW IT WORK 0" src="https://github.com/user-attachments/assets/e263d463-2a60-415d-abc7-bb5307b0b775" />

<img width="1920" height="935" alt="HOW IT WORK 1" src="https://github.com/user-attachments/assets/3b4d48d2-8e20-41b2-b52d-90425bdc549e" />


<img width="1920" height="930" alt="HOW IT WORK 2" src="https://github.com/user-attachments/assets/9c39eb32-53fd-4db8-8dc8-e2c3d65ff91f" />


<img width="1920" height="926" alt="HOW IT WORK 3" src="https://github.com/user-attachments/assets/05a6712e-6526-48f1-a0f5-8dc0d761bd3b" />







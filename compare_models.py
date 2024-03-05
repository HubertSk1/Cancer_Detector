import joblib
from PIL import Image
import os
import numpy as np
from sklearn.preprocessing import StandardScaler
from svm_separator import load_images_from_folder
from base_on_color_tag import predict_base_on_color

loaded_model = joblib.load('models/svm_model.pkl')
loaded_scaler = joblib.load('models/scaler.pkl')

def load_images_from_folder(folder, label):
    images = []
    patches = []
    for filename in os.listdir(folder):
        img = Image.open(os.path.join(folder, filename))
        if img is not None:
            images.append(np.array(img).flatten())
            patches.append(img)
    return images, patches

def run_test():
    test_images, test_patches = load_images_from_folder("photos/test_photos", label=None)
    os.makedirs("photos/test_diffrences", exist_ok=True)
    test_images_scaled = loaded_scaler.transform(test_images)
    test_svm_predictions = loaded_model.predict(test_images_scaled)
    print(test_svm_predictions)
    for index,image in enumerate(test_images):
       if test_svm_predictions[index] != predict_base_on_color(test_patches[index]):
           tag ="Cancer" if test_svm_predictions[index] == 1 else "No_Cancer" 
           test_patches[index].save(f"photos/test_diffrences/patch_{index}_tagged_as_{tag}_by_SVM.png")
    
run_test()
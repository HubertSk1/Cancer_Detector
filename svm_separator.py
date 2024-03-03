import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from PIL import Image
import configparser

# Create a ConfigParser object
config = configparser.ConfigParser()

# Load the configuration from the file
config.read('cfg.conf')


general_image_folder = config["images"]["general_path"]

cancer_folder = os.path.join(general_image_folder, 'separated')
cancer_folder = os.path.join(cancer_folder, 'Cancer')

no_cancer_folder = os.path.join(general_image_folder, 'separated')
no_cancer_folder = os.path.join(no_cancer_folder, 'No-Cancer')


def load_images_from_folder(folder, label):
    images = []
    labels = []
    for filename in os.listdir(folder):
        img = Image.open(os.path.join(folder, filename))
        if img is not None:
            images.append(np.array(img).flatten())
            labels.append(label)
    return images, labels


no_cancer_images, no_cancer_labels = load_images_from_folder(no_cancer_folder, 0) 
cancer_images, cancer_labels = load_images_from_folder(cancer_folder, 1) 


X = np.array(no_cancer_images + cancer_images)
y = np.array(no_cancer_labels + cancer_labels)


X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_valid_scaled = scaler.transform(X_valid)


svm_model = SVC(kernel='linear')
svm_model.fit(X_train_scaled, y_train)


y_pred = svm_model.predict(X_valid_scaled)


accuracy = accuracy_score(y_valid, y_pred)
print(f'Dokładność modelu na zbiorze walidacyjnym: {accuracy:.2f}')

import tensorflow as tf
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

print("Mentett modell betöltése...")
# Betöltjük a korábban betanított "okos" modellt
model = tf.keras.models.load_model('brain_tumor_cnn_model.keras')

# Teszt adatok betöltése SZIGORÚAN shuffle=False beállítással!
test_dir = r"data\raw\agyikepek_4_osztaly\Testing"
test_dataset = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    image_size=(128, 128),
    batch_size=32,
    shuffle=False  # <-- EZ OLDJA MEG A PROBLÉMÁT
)
class_names = test_dataset.class_names

# Kigyűjtjük a címkéket és a tippeket biztonságosan, egyszerre
y_true = []
y_pred = []

print("Predikciók generálása a teszt adatokon (ez eltarthat pár másodpercig)...")
for images, labels in test_dataset:
    preds = model.predict(images, verbose=0)
    y_true.extend(labels.numpy())
    y_pred.extend(np.argmax(preds, axis=1))

# A VALÓDI mátrix kirajzolása
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title('Tévesztési Mátrix (Javított - 95% Pontosság)')
plt.ylabel('Valós osztály')
plt.xlabel('Gép tippje')
plt.show()
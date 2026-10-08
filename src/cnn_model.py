import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt


# 1. beállítások
IMG_SIZE = (128, 128) #
BATCH_SIZE = 32

train_dir = r"data\raw\agyikepek_4_osztaly\Training"
test_dir = r"data\raw\agyikepek_4_osztaly\Testing"

# 2. Tanító és Teszt adathalmazok betöltése
print("--- TANÍTÓ ADATOK BETÖLTÉSE ---")
train_dataset = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    image_size = IMG_SIZE,
    batch_size = BATCH_SIZE,
    shuffle = True # keverjük hogy lásson minden tumort
)

print("\n--- TESZTELŐ ADATOK BETÖLTÉSE ---")
test_dataset = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    image_size = IMG_SIZE,
    batch_size = BATCH_SIZE,
    shuffle = True
)

# Kiírjuk az osztályok neveit, amiket a mappa alapján talált
class_names = train_dataset.class_names
print(f"\nFelismerendő osztályok ({len(class_names)} db): {class_names}")
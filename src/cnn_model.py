import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix


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


# 3. A neurális hálózat (CNN) Architektúrája

model = models.Sequential([
    # Modern bemeneti réteg deklaráció a Keras figyelmeztetés elkerülésére:
    layers.Input(shape=(128, 128, 3)),
    layers.Rescaling(1./255),

    # 1. Konvolúciós blokk (itt javítva az 'activation'):
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    # 2. Konvolúciós blokk:
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    # 3. Konvolúciós blokk:
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    # Döntéshozó sűrű rétegek:
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(4)
])

# 4. a modell összeállítása (compile)
# - optimizer='adam': az egyik legnépszerűbb, önmagát gyorsító tanuló algoritmus
# - loss: SparseCategoricalCrossentropy, mert többosztályos egész számos címkéink vannak
model.compile(
    optimizer = 'adam',
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics =['accuracy']
)

# Kiírjuk a hálózat összegzését (hány százezer paramétertt fog hangolni)
print("\n--- A Modell felépítése ---")
model.summary()


#5. Model tanítás
print("\n--- A Modell tanítása ---")
EPOCHS = 10 # hányszor menjen végig az adathalmazon

#A fit() parancs indítja el a tényleges tanulást
history = model.fit(
    train_dataset,
    validation_data=test_dataset,
    epochs = EPOCHS

)


# 6. Eredmények vizualizálása
#kirajzoljuk egy grafikonra, hogyan fejlődött a gép 10 kör alatt.
plt.figure(figsize=(10,5))

#Pontosság grafikon
plt.plot(history.history['accuracy'], label='Tanítási pontosság')
plt.plot(history.history['val_accuracy'], label='Tesztelési pontosság')
plt.title('A modell pontosságának fejlődése')
plt.xlabel('Korszak (Epoch)')
plt.ylabel('Pontosság')
plt.legend()
plt.grid(True)
plt.show()


# 7. Modell mentése
print("\n--- Modell mentése ---")
model.save('brain_tumor_cnn_model.keras')
print("A modell sikeresen elmentve 'brain_tumor_cnn_model.keras' néven!")


# 8. tévesztési mátrix
print("\n--- Kiértékelés és vizualizáció ---")
# kigyüjtjük a valós címkéket és a gép tippjeit a teszt halmazból
y_true = np.concatenate([y for x, y in test_dataset], axis=0)
predictions = model.predict(test_dataset)
y_pred = np.argmax(predictions, axis=1)

# Mátrix kiszámítása és kirajzolása
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title('Tévesztési Mátrix (Deep Learning - 4 osztály)')
plt.ylabel('Valós osztály')
plt.xlabel('Gép tippje')
plt.show()




# ----------------------------------------------------
# 9. PÉLDA KÉPEK PREDÍKCIÓKKAL (A gép "szemével")
# ----------------------------------------------------
# Kiveszünk egyetlen köteget (32 képet) a teszt halmazból
for images, labels in test_dataset.take(1):
    sample_images = images.numpy()
    sample_labels = labels.numpy()
    # A gép tippel erre a 32 képre
    sample_preds = model.predict(images)

plt.figure(figsize=(12, 12))
for i in range(9):  # Az első 9 képet rajzoljuk ki egy 3x3-as rácsban
    plt.subplot(3, 3, i + 1)
    
    # Kép kirajzolása (a matplotlib szereti az egész számokat képeknél)
    plt.imshow(sample_images[i].astype("uint8"))
    
    # Valós és tippelt osztályok meghatározása
    true_label = class_names[sample_labels[i]]
    pred_idx = np.argmax(sample_preds[i])
    pred_label = class_names[pred_idx]
    
    # Magabiztosság (százalék) kiszámítása Softmax függvénnyel
    confidence = 100 * np.max(tf.nn.softmax(sample_preds[i]))
    
    # Zöld felirat, ha eltalálta, Piros, ha tévedett
    color = 'green' if true_label == pred_label else 'red'
    plt.title(f"Valós: {true_label}\nTipp: {pred_label} ({confidence:.1f}%)", color=color)
    plt.axis("off")

plt.tight_layout()
plt.show()
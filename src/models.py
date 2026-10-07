import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

def train_and_evaluate():
    # 1. Adatok betöltése
    csv_path = r"data\processed\features.csv"
    print(f"Adatok betöltése innen: {csv_path}...")
    
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print("Hiba: Nem található a features.csv fájl!")
        return

    # 2. Bemeneti jellemzők (X) és a célváltozó/címke (y) szétválasztása
    # A 'filename' nem kell a tanításhoz (az csak szöveg), a 'label' pedig a cél
    X = df.drop(columns=['filename', 'label'])
    y = df['label']
    
    print(f"Adathalmaz mérete: {X.shape[0]} minta, {X.shape[1]} jellemző.")

    # 3. Tanító (80%) és teszt (20%) halmazra osztás
    # A stratify=y biztosítja, hogy mindkét halmazban arányosan legyenek a különböző tumor típusok
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print("Felosztás kész: 80% tanítás, 20% tesztelés.")

    # 4. Modell kiválasztása és tanítása
    print("\nRandom Forest modell betanítása indul...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    print("Tanítás befejeződött!")

    # 5. Predikció (jóslás) a teszt adatokon, amiket a modell még sosem látott
    y_pred = model.predict(X_test)

    # 6. Kiértékelés
    acc = accuracy_score(y_test, y_pred)
    print("\n" + "="*45)
    print(f"MODELL PONTOSSÁGA (Accuracy): {acc * 100:.2f}%")
    print("="*45)
    
    print("\nRészletes osztályozási jelentés:")
    print(classification_report(y_test, y_pred))

if __name__ == "__main__":
    train_and_evaluate()
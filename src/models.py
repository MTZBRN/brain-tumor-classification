import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix

def train_and_evaluate():
    # 1. Adatok betöltése
    csv_path = r"data\processed\features.csv"
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print("Hiba: Nem található a features.csv fájl!")
        return

    X = df.drop(columns=['filename', 'label'])
    y = df['label']
    
    # 2. Adatok skálázása (Az SVM és a KNN nagyon érzékeny arra, ha a Terület mondjuk 10000, a Kontraszt meg 0.01. Ezzel közös léptékre hozzuk őket.)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)
    
    # Eredeti X megőrzése a Random Forest feature importance grafikonhoz
    X_train_unscaled, X_test_unscaled, _, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    svm_param_grid = {
        'C': [0.1, 1, 10, 100],
        'gamma': [1, 0.1, 0.01, 0.001],
        'kernel': ['rbf']
    }


    # 3. Modellek definiálása a "versenyhez"
    models = {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "KNN (K-Nearest Neighbors)": KNeighborsClassifier(n_neighbors=5),
        "SVM (Support Vector Machine)": GridSearchCV(SVC(random_state=42), svm_param_grid, cv=3, verbose=1)
    }

    print("\n--- MODELLEK VERSENYE ---")
    best_model_name = ""
    best_accuracy = 0
    best_predictions = None
    
    # 4. Modellek tanítása és kiértékelése
    for name, model in models.items():
        if name == "Random Forest":
            model.fit(X_train_unscaled, y_train) # A RF nem igényli a skálázást
            y_pred = model.predict(X_test_unscaled)
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
        acc = accuracy_score(y_test, y_pred)
        print(f"{name} pontossága: {acc * 100:.2f}%")
        
        # Elmentjük a legjobbat a vizualizációhoz
        if acc > best_accuracy:
            best_accuracy = acc
            best_model_name = name
            best_predictions = y_pred

    print(f"\nA győztes modell: {best_model_name} ({best_accuracy * 100:.2f}%)")

    # 5. VIZUALIZÁCIÓ 1: Tévesztési Mátrix a győztes modellre
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    cm = confusion_matrix(y_test, best_predictions)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=[1, 2, 3], yticklabels=[1, 2, 3])
    plt.title(f'Tévesztési Mátrix\n({best_model_name})')
    plt.xlabel('Gép által tippelt osztály')
    plt.ylabel('Valódi osztály')

    # 6. VIZUALIZÁCIÓ 2: Jellemzők fontossága (Csak a Random Forest tud ilyet könnyen)
    plt.subplot(1, 2, 2)
    rf_model = models["Random Forest"]
    importances = rf_model.feature_importances_
    features = X.columns
    
    # Sorba rendezés
    indices = importances.argsort()
    
    plt.barh(range(len(indices)), importances[indices], color='b', align='center')
    plt.yticks(range(len(indices)), [features[i] for i in indices])
    plt.title('Mi alapján dönt a Random Forest?')
    plt.xlabel('Jellemző fontossága (%)')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    train_and_evaluate()
# Agydaganat Osztályozó Rendszer (Brain Tumor Classification)

Ez a projekt egy klasszikus gépi tanulási (Machine Learning) csővezetéket (pipeline) valósít meg, amely MRI felvételek alapján képes három különböző típusú agydaganat (meningioma, glioma, agyalapi mirigy tumor) automatikus osztályozására.

## A projekt felépítése és mérnöki lépések

A szoftver a feladatot három fő modulra bontva oldja meg:

### 1. Képfeldolgozás és Jellemzőkinyerés (`src/preprocessor.py`)
A nyers MRI képek és a hozzájuk tartozó bináris maszkok alapján a szkript strukturált (táblázatos) adatokat generál. A felhasznált számítógépes látás (Computer Vision) technikák:
- **Alak jellemzők:** A tumor kiterjedésének (Area) és kerületének (Perimeter) meghatározása.
- **Geometriai jellemzők:** Forgás- és méretfüggetlen Hu-momentumok (7 db) számítása a daganat alakjának precíz leírásához.
- **Textúra jellemzők (GLCM):** A tumor felületi homogenitásának, kontrasztjának és energiájának kinyerése szürkeárnyalatos (Grayscale) konverzió után.
*Kimenet: Egy 15 jellemzőt tartalmazó, tisztított CSV fájl, ami a modellek bemeneteként szolgál.*

### 2. Gépi Tanulás és Optimalizáció (`src/models.py`)
A strukturált adatokon három különböző felügyelt tanulási (Supervised Learning) algoritmust versenyeztettünk meg:
- **K-Nearest Neighbors (KNN)**
- **Random Forest Classifier** (Feature Importance vizualizációval)
- **Support Vector Machine (SVM)**

**Kiemelt eredmény:** Az alapértelmezett beállítások tesztelése után az SVM modellen **hiperparaméter-hangolást (GridSearchCV)** végeztünk (C és gamma paraméterekre, RBF kernellel, 3-szoros keresztvalidációval). Az automatizált tesztpad 48 iteráció után megtalálta az optimális beállításokat, amivel a modell pontossága **86.95%-ra** nőtt.

### 3. Mélytanulás és Konvolúciós Neurális Hálózat (`src/cnn_model.py` és `src/evaluate.py`)
A feladatot kiterjesztettük egy 4 osztályos adathalmazra is (egészséges agyakat is beleértve), ahol egy saját építésű Konvolúciós Neurális Hálózat (CNN) tanul közvetlenül a nyers MRI képekből, manuális maszkok nélkül.
- **Architektúra:** TensorFlow/Keras alapú modell. 3 konvolúciós blokk (`Conv2D` + `MaxPooling2D`) felel a jellemzőkinyerésért, amit egy 128 neuronos sűrű réteg (Dropout-tal a túltanulás ellen) és egy 4 kimenetű Softmax réteg követ.
- **Eredmény:** 10 korszak (Epoch) tanítás után a modell **~95%-os tesztelési pontosságot** ért el.
- **Kiértékelés:** A rendszer Tévesztési Mátrixot és vizuális predikciókat generál, kiemelkedő magabiztossággal azonosítva az egészséges eseteket.

## Futtatási útmutató

1. **Függőségek telepítése:**
   ```bash
   pip install -r requirements.txt
   pip install tensorflow

2. **Jellemzők kinyerése (Adatelőkészítés):**
   ```bash
   python src/preprocessor.py

3. **Modellek betanítása és kiértékelése:**
    ```bash
   python src/models.py

4. **Mélytanuló hálózat (CNN) tanítása:**
    ```bash
    python src/cnn_model.py

5. **CNN modell kiértékelése (Tévesztési Mátrix és vizualizáció):**
    ```bash
    python src/evaluate.py

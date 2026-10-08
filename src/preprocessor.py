import os
import glob
import cv2
import numpy as np
import pandas as pd
from skimage.feature import graycomatrix, graycoprops

def extract_features(img_path, mask_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    

    if mask is None or img is None:
        return None

    #Maszk binarizálás (biztos ami biztos)
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    # Fájlnév és címke kinyerése
    filename = os.path.basename(img_path)
    label = filename.split('_')[-1].split('.')[0]
    
    # Csak a tumor pixeleinek kigyűjtése
    tumor_pixels = img[mask > 0]
    if len(tumor_pixels) == 0:
        return None # Ha a maszk teljesen fekete (nincs tumor)

    # 1. ALAK JELLEMZŐK (Shape features)
    area = np.sum(mask > 0)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    perimeter = cv2.arcLength(contours[0], True) if len(contours) > 0 else 0
    moments = cv2.moments(mask)
    hu_moments = cv2.HuMoments(moments).flatten()

    # 2. INTENZITÁS JELLEMZŐK (Intensity features)
    mean_intensity = np.mean(tumor_pixels)
    std_intensity = np.std(tumor_pixels)
    
    # 3. TEXTÚRA JELLEMZŐK (GLCM)
    # Hogy a fekete háttér ne zavarjon, körbevágjuk a képet a tumor méretére (Bounding Box)
    x, y, w, h = cv2.boundingRect(mask)
    tumor_roi = img[y:y+h, x:x+w]
    
    # Szürkeárnyalati együttelőfordulási mátrix kiszámítása
    glcm = graycomatrix(tumor_roi, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
    
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0]
    correlation = graycoprops(glcm, 'correlation')[0, 0]
    
    return {
        'filename': filename,
        'label': int(label),
        'area': area,
        'perimeter': perimeter,
        'hu_1': hu_moments[0],
        'hu_2': hu_moments[1],
        'hu_3': hu_moments[2],
        'hu_4': hu_moments[3],
        'hu_5': hu_moments[4],
        'hu_6': hu_moments[5],
        'hu_7': hu_moments[6],
        'mean_intensity': mean_intensity,
        'std_intensity': std_intensity,
        'contrast': contrast,
        'homogeneity': homogeneity,
        'energy': energy,
        'correlation': correlation
    }

if __name__ == "__main__":
    
    images_dir = r"data\raw\agyikepek_3_osztaly\kepek"
    masks_dir = r"data\raw\agyikepek_3_osztaly\maskkep"
    output_csv = r"data\processed\features.csv"
    
    image_paths = sorted(glob.glob(os.path.join(images_dir, "*.*")))
    print(f"Feldolgozás megkezdése: {len(image_paths)} kép...")
    
    features_list = []
    
    for i, img_path in enumerate(image_paths):
        filename = os.path.basename(img_path)
        mask_name_no_ext = os.path.splitext(filename)[0]
        
        # Maszk útvonal
        mask_path = os.path.join(masks_dir, mask_name_no_ext + ".png")
        if not os.path.exists(mask_path):
            mask_path = os.path.join(masks_dir, mask_name_no_ext + ".jpg")
            
        if os.path.exists(mask_path):
            feats = extract_features(img_path, mask_path)
            if feats is not None:
                features_list.append(feats)
                
        # 500 képenként írjuk ki, hol tartunk
        if (i + 1) % 500 == 0:
            print(f"Feldolgozva: {i + 1} / {len(image_paths)}")
            
    # Pandas DataFrame létrehozása és mentése CSV fájlba
    df = pd.DataFrame(features_list)
    df.to_csv(output_csv, index=False)
    print(f"\nSikeresen kinyertünk jellemzőket {len(df)} képről.")
    print(f"Az eredmény elmentve ide: {output_csv}")
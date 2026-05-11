import os
import pandas as pd
import numpy as np
import cv2
from sklearn.model_selection import train_test_split

def load_data_to_df(data_dir='../data/'):
    """
    Carga las rutas de las imágenes y sus etiquetas en un DataFrame.
    """
    filepaths = []
    labels = []
    
    classes = os.listdir(data_dir)
    for class_name in classes:
        class_path = os.path.join(data_dir, class_name)
        if os.path.isdir(class_path):
            images = os.listdir(class_path)
            for img_name in images:
                filepaths.append(os.path.join(data_dir, class_name, img_name))
                labels.append(class_name)
                
    df = pd.DataFrame({'filepath': filepaths, 'label': labels})
    return df

def create_artificial_imbalance(df, target_counts=None, random_state=42):
    """
    Genera desbalance artificial reduciendo la cantidad de imágenes en ciertas clases.
    
    Por defecto:
    - Turtle_Tortoise y Dolphin: Mantienen todas.
    - Seal, Penguin, Clams: Reducir a ~100.
    """
    if target_counts is None:
        target_counts = {
            'Seal': 100,
            'Penguin': 100,
            'Clams': 100
        }
    
    df_balanced = []
    for class_name in df['label'].unique():
        class_df = df[df['label'] == class_name]
        
        if class_name in target_counts:
            n_samples = min(len(class_df), target_counts[class_name])
            class_df = class_df.sample(n=n_samples, random_state=random_state)
            
        df_balanced.append(class_df)
        
    return pd.concat(df_balanced, axis=0).reset_index(drop=True)

def add_gaussian_noise(image):
    """
    Agrega ruido gaussiano a una imagen.
    """
    row, col, ch = image.shape
    mean = 0
    var = 0.1
    sigma = var**0.5
    gauss = np.random.normal(mean, sigma, (row, col, ch))
    gauss = gauss.reshape(row, col, ch)
    noisy = image + gauss
    return np.clip(noisy, 0, 255).astype(np.uint8)

def simulate_low_lighting(image):
    """
    Simula baja iluminación oscureciendo la imagen aleatoriamente.
    """
    factor = np.random.uniform(0.3, 0.7)
    low_light = image * factor
    return np.clip(low_light, 0, 255).astype(np.uint8)

def custom_preprocessing(image):
    """
    Aplica ruido y baja iluminación como preprocesamiento adicional.
    """
    # 50% de probabilidad de aplicar ruido
    if np.random.rand() > 0.5:
        image = add_gaussian_noise(image)
    
    # 50% de probabilidad de bajar iluminación
    if np.random.rand() > 0.5:
        image = simulate_low_lighting(image)
        
    return image.astype(np.float32)

def split_data(df, train_size=0.7, val_size=0.15, test_size=0.15, random_state=42):
    """
    Divide el dataframe en conjuntos de entrenamiento, validación y prueba.
    """
    train_df, rem_df = train_test_split(df, train_size=train_size, random_state=random_state, stratify=df['label'])
    
    # El resto se divide en partes iguales para val y test (0.15 / 0.30 = 0.5)
    relative_val_size = val_size / (val_size + test_size)
    val_df, test_df = train_test_split(rem_df, train_size=relative_val_size, random_state=random_state, stratify=rem_df['label'])
    
    return train_df, val_df, test_df

<div align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&height=120&section=header&animation=fadeIn" />
</div>

<h1 align="center">🐡 Clasificación de Animales Marinos con Deep Learning</h1>

<h3 align="center">🚀 Análisis Comparativo: EfficientNetB0 vs MobileNetV2</h3>

<p align="center">
  Proyecto de Computer Vision para la clasificación de 23 especies marinas mediante Transfer Learning y Fine-Tuning.<br>
  Implementado con Python, TensorFlow y Keras, abordando retos de clases desbalanceadas e imágenes de baja calidad.
</p>

---

## 📋 **Descripción del Proyecto**

**Sea Animals Classification** es un proyecto enfocado en la aplicación práctica de redes neuronales convolucionales (CNNs) para identificar especies de animales marinos a partir de imágenes. El estudio incluye la simulación de escenarios del mundo real como el **desbalance de datos** (reducción artificial de clases específicas) y condiciones de **baja calidad** (ruido gaussiano y baja iluminación) para evaluar la robustez de los modelos preentrenados.

> ⚠️ **Estado del Proyecto:** Terminado
> Modelos entrenados, evaluados y comparados satisfactoriamente.

---

## 🏗️ **Arquitecturas Evaluadas**

Se implementó un flujo de dos fases (Transfer Learning puro + Fine-Tuning) para cada modelo:
- **EfficientNetB0:** Arquitectura basada en escalamiento compuesto que busca un equilibrio óptimo entre eficiencia y alta precisión.
- **MobileNetV2:** Arquitectura ligera con bloques residuales invertidos, altamente optimizada para dispositivos móviles y entornos con recursos limitados.

Como parte experimental, se evaluó el impacto de variar la cantidad de capas congeladas (5, 10, 20) y distintos rangos de hiperparámetros (Learning Rate).

---

## 🔧 **Stack Tecnológico**

### **Lenguaje y Frameworks Core**
<div align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img width="8" />
  <img src="https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white" />
  <img width="8" />
  <img src="https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=Keras&logoColor=white" />
</div>

### **Procesamiento y Visualización de Datos**
<div align="center">
  <img src="https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white" />
  <img width="8" />
  <img src="https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white" />
  <img width="8" />
  <img src="https://img.shields.io/badge/Matplotlib-11557c?style=for-the-badge&logo=python&logoColor=white" />
  <img width="8" />
  <img src="https://img.shields.io/badge/Seaborn-4C4C4C?style=for-the-badge&logo=python&logoColor=white" />
  <img width="8" />
  <img src="https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white" />
</div>

### **Herramientas de Desarrollo**
<div align="center">
  <img src="https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=Jupyter&logoColor=white" />
  <img width="8" />
  <img src="https://img.shields.io/badge/Git-F05032?style=for-the-badge&logo=git&logoColor=white" />
</div>

---

## 🚀 **Estructura de Ejecución**

| Etapa | Archivo/Notebook | Objetivo Principal |
|-------|------------------|--------------------|
| **1. Exploración de Datos** | `01_exploracion_datos.ipynb` | Desbalance de datos y simulación visual de baja calidad |
| **2. Entrenamiento 1** | `02_efficientnet.ipynb` | Transfer Learning y Fine-Tuning de EfficientNetB0 |
| **3. Entrenamiento 2** | `03_mobilenet.ipynb` | Transfer Learning y Fine-Tuning de MobileNetV2 |
| **4. Resultados** | `04_comparacion_modelos.ipynb` | Comparación de métricas, gráficas y análisis final |

---

## 🏪 **Características Técnicas Implementadas**

### **🔍 Técnicas de Deep Learning**
- Callbacks avanzados (`ModelCheckpoint`, `EarlyStopping` con patience=5).
- Data Augmentation adaptativo (rotación, zoom, brillo, ruido gaussiano, simulación de baja luz).
- Fine-Tuning progresivo (descongelamiento de las últimas 10 capas y reducción del learning rate).

### **📊 Análisis del Desempeño**
- Curvas de Aprendizaje (Accuracy vs Épocas, Loss vs Épocas).
- Matrices de Confusión visuales.
- Reportes detallados de clasificación (Precision, Recall, F1-Score por clase).
- Detección exhaustiva de patrones de Overfitting y Underfitting.

---

👨‍💻 Desarrollador
<div align="center">
Santiago Arbelaez Contreras

Junior Full Stack Developer

Estudiante de Ingeniería de Sistemas – Universidad del Quindío

<br> <a href="https://github.com/santiagoarbelaezc"> <img src="https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white" /> </a> <img width="10" /> <a href="https://www.linkedin.com/in/santiago-arbelaez-contreras-9830b5290/"> <img src="https://img.shields.io/badge/LinkedIn-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white" /> </a> <img width="10" /> <a href="https://portfolio-santiagoa.web.app/portfolio"> <img src="https://img.shields.io/badge/Portfolio-6C63FF?style=for-the-badge&logo=sparkles&logoColor=white" /> </a></div>

<div align="center"> <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&height=90&section=footer&animation=fadeIn" /> </div>

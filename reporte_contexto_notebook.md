# Contexto Técnico del Proyecto: Clasificación de Animales Marinos (6 Clases)

Este documento contiene el contexto completo, estructura y decisiones técnicas implementadas en el notebook final del proyecto (`notebook_unificado/Proyecto_Final_6Clases_Colab.ipynb`). Está diseñado para que otro agente de IA entienda de inmediato la arquitectura, el flujo de trabajo y las soluciones aplicadas.

## 1. Información General del Proyecto
- **Objetivo Académico:** Comparar dos arquitecturas preentrenadas (EfficientNetB0 y MobileNetV2) aplicando técnicas de Transfer Learning y Fine-Tuning en un entorno de datos desbalanceados y de baja calidad.
- **Dataset:** Imágenes de 6 clases marinas (`Whale`, `Sharks`, `Fish`, `Jelly Fish`, `Starfish`, `Dolphin`).
- **Framework:** TensorFlow / Keras.
- **Dificultades artificiales inducidas:**
  - Desbalance de clases severo (se recortaron intencionalmente clases a solo 50 muestras).
  - Imágenes de baja calidad (simulación de ruido gaussiano y baja iluminación vía Data Augmentation).

---

## 2. Decisiones Arquitectónicas y Correcciones Críticas

### 2.1. Preprocesamiento e Ingesta de Datos (`ImageDataGenerator`)
**Decisión crítica:** Se eliminó el `rescale=1./255` global del generador.
- **Razón:** `EfficientNetB0` en TensorFlow incluye su propia capa interna de rescaling y espera recibir tensores en el rango `[0, 255]`. Hacer rescaling manual destruía las activaciones (las reducía a casi 0).
- **Solución para MobileNetV2:** Como MobileNet sí requiere entradas en el rango `[-1, 1]`, se le inyectó una capa `Lambda` con su propio `preprocess_input` directamente en la arquitectura del modelo.

```python
# Celdas Clave: Generador de Datos
train_datagen = ImageDataGenerator(
    rotation_range=30, width_shift_range=0.2, height_shift_range=0.2,
    shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode='nearest', brightness_range=[0.5, 1.5]
)
val_datagen = ImageDataGenerator() # Sin rescale

train_gen = train_datagen.flow_from_dataframe(...)
```

### 2.2. Balanceo de Clases
Se calculan pesos algorítmicos utilizando `sklearn.utils.class_weight` y se pasan al parámetro `class_weight` del `model.fit()` para obligar a la red a prestar atención a las clases minoritarias (ej. *Starfish*).

---

## 3. Estructura de Entrenamiento (Modelos)

El notebook sigue un flujo idéntico para ambos modelos (`EfficientNetB0` y `MobileNetV2`):

### Fase 1: Transfer Learning (Cabecera Entrenable)
- **Congelamiento:** `base_model.trainable = False`
- **Cabecera (Head):** `GlobalAveragePooling2D` -> `Dense(256)` -> `Dropout(0.5)` -> `Dense(6, softmax)`
- **Hiperparámetros:** Optimizador Adam, Learning Rate de `1e-3`, 15 épocas.
- **Callbacks:** `EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)`

```python
# Celda Clave: Arquitectura MobileNetV2 con preprocesamiento inyectado
from tensorflow.keras.layers import Lambda
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

input_tensor = tf.keras.Input(shape=(224, 224, 3))
x2 = Lambda(preprocess_input)(input_tensor)
base_mob = MobileNetV2(weights='imagenet', include_top=False, input_tensor=x2)
base_mob.trainable = False

x2 = GlobalAveragePooling2D()(base_mob.output)
x2 = Dense(256, activation='relu')(x2)
x2 = Dropout(0.5)(x2)
predictions_mob = Dense(len(selected_classes), activation='softmax')(x2)
model_mob = Model(inputs=input_tensor, outputs=predictions_mob)
```

### Fase 2: Fine-Tuning (Descongelando últimas 10 capas)
- **Descongelamiento Parcial:** Iteramos por las capas y descongelamos exclusivamente las últimas 10, tal como lo exige la rúbrica.
- **Hiperparámetros:** El Learning Rate se reduce drásticamente a `1e-4` para evitar destruir los pesos originales.

```python
# Celda Clave: Fine-Tuning
base_eff.trainable = True
for layer in base_eff.layers[:-10]:
    layer.trainable = False

model_eff.compile(optimizer=Adam(1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
h_eff_ft = model_eff.fit(..., epochs=10, callbacks=[es], class_weight=class_weights)
```

---

## 4. Laboratorios (Experimentos)
El notebook incluye dos bloques de experimentos intencionalmente diseñados para probar malos hiperparámetros y observar sus efectos destructivos:
1. **Experimento de Learning Rate:** Se usa un LR altísimo (`1e-1`), lo que provoca que el Loss explote (llegando a >500) y el modelo olvide lo aprendido.
2. **Experimento de Capas:** Se descongelan demasiadas capas (30) para mostrar inestabilidad sin un ajuste previo.

*(Nota para el agente: Las evaluaciones de clasificación oficiales deben ejecutarse ANTES de estos laboratorios para no evaluar los pesos destruidos).*

---

## 5. Evaluación Final y Comparación
El notebook concluye extrayendo las métricas finales de `val_accuracy` y `val_loss` de los historiales de ambos modelos.
- Genera gráficas superpuestas comparando la evolución del Accuracy y Loss entre `EfficientNet` y `MobileNet`.
- Imprime `classification_report` (Scikit-Learn) y mapas de calor (`sns.heatmap`) para la matriz de confusión.

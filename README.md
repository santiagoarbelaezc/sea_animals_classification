# Clasificación de Animales Marinos: EfficientNetB0 vs MobileNetV2

Este proyecto realiza un análisis comparativo de dos arquitecturas populares de Deep Learning para la clasificación de 23 especies de animales marinos.

## Estructura del Proyecto
- `data/`: Imágenes organizadas por carpetas de clases.
- `utils/`: Scripts de soporte para preprocesamiento, aumentos y métricas.
- `notebooks/`: Flujo de trabajo dividido en 4 pasos lógicos.
- `models/`: Almacenamiento de pesos (.keras).
- `results/`: Gráficas y archivos de historial.

## Flujo de Ejecución
1. **01_exploracion_datos.ipynb**: Análisis inicial y generación de desbalance artificial.
2. **02_efficientnet.ipynb**: Entrenamiento y Fine-Tuning de EfficientNetB0.
3. **03_mobilenet.ipynb**: Entrenamiento y Fine-Tuning de MobileNetV2.
4. **04_comparacion_modelos.ipynb**: Comparación final de métricas y conclusiones.

## Características Técnicas
- **Simulación de Baja Calidad**: Agregado de ruido gaussiano e iluminación variable.
- **Transfer Learning**: Uso de pesos preentrenados de ImageNet.
- **Fine-Tuning**: Descongelamiento selectivo de capas para optimización.
- **Métricas**: Matriz de confusión, reporte de clasificación y análisis de overfitting.

## Requisitos
Instalar dependencias con:
```bash
pip install -r requirements.txt
```

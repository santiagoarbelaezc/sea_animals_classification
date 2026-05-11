import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

def plot_history(history, title="Entrenamiento"):
    """
    Grafica Accuracy y Loss de un historial de entrenamiento.
    """
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))

    # Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, 'bo-', label='Training Acc')
    plt.plot(epochs, val_acc, 'ro-', label='Validation Acc')
    plt.title(f'Accuracy - {title}')
    plt.xlabel('Épocas')
    plt.ylabel('Precisión')
    plt.legend()

    # Loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, 'bo-', label='Training Loss')
    plt.plot(epochs, val_loss, 'ro-', label='Validation Loss')
    plt.title(f'Loss - {title}')
    plt.xlabel('Épocas')
    plt.ylabel('Pérdida')
    plt.legend()

    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(y_true, y_pred, classes):
    """
    Grafica la matriz de confusión.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(15, 12))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    plt.title('Matriz de Confusión')
    plt.ylabel('Real')
    plt.xlabel('Predicho')
    plt.show()

def print_classification_report(y_true, y_pred, classes):
    """
    Imprime el reporte de clasificación.
    """
    print(classification_report(y_true, y_pred, target_names=classes))

def compare_phases(h1, h2, metric='accuracy'):
    """
    Compara dos fases de entrenamiento en una misma gráfica.
    """
    m1 = h1.history[metric]
    m1_val = h1.history[f'val_{metric}']
    m2 = h2.history[metric]
    m2_val = h2.history[f'val_{metric}']
    
    total_m = m1 + m2
    total_m_val = m1_val + m2_val
    
    plt.figure(figsize=(10, 6))
    plt.plot(total_m, label=f'Train {metric}')
    plt.plot(total_m_val, label=f'Val {metric}')
    plt.axvline(x=len(m1)-1, color='grey', linestyle='--', label='Inicio Fine-Tuning')
    plt.title(f'Comparación de Fases: {metric}')
    plt.xlabel('Épocas Totales')
    plt.ylabel(metric.capitalize())
    plt.legend()
    plt.show()

def analyze_overfitting(history):
    """
    Analiza visualmente si hay overfitting o underfitting.
    """
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    
    diff = np.array(acc) - np.array(val_acc)
    
    print("--- Análisis de Overfitting/Underfitting ---")
    if acc[-1] > 0.95 and val_acc[-1] < 0.80:
        print("ALTA probabilidad de Overfitting: Gran brecha entre Train y Val Accuracy.")
    elif acc[-1] < 0.60:
        print("Probabilidad de Underfitting: El modelo no está aprendiendo lo suficiente.")
    else:
        print("El modelo parece tener un comportamiento balanceado.")
    print(f"Brecha final: {diff[-1]:.4f}")

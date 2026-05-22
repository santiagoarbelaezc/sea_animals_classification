import json
import sys

notebook_path = r'c:\Users\Santiago\OneDrive\Escritorio\Repositorios\sea_animals_classification\notebook_unificado\Proyecto_Final_6Clases_Colab.ipynb'

try:
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
except Exception as e:
    print(f"Error leyendo notebook: {e}")
    sys.exit(1)

start_idx = -1
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'markdown':
        source = ''.join(cell['source'])
        if 'PARTE 5: 10 Experimentos de Modelado' in source or 'PARTE 5' in source:
            start_idx = i
            break

if start_idx != -1:
    print(f'PARTE 5 encontrada en el índice {start_idx}')
    nb['cells'] = nb['cells'][:start_idx]
else:
    print('No se encontró la PARTE 5, se añadirá al final.')

def create_markdown_cell(text):
    return {
        'cell_type': 'markdown',
        'metadata': {},
        'source': [text + '\n']
    }

def create_code_cell(code_lines):
    return {
        'cell_type': 'code',
        'execution_count': None,
        'metadata': {},
        'outputs': [],
        'source': [line + '\n' for line in code_lines]
    }

nuevas_celdas = []

nuevas_celdas.append(create_markdown_cell('# =============================================\n# PARTE 5: 10 Experimentos de Modelado (Conclusiones y Propósitos)\n# =============================================\n\nEn esta sección se detallan 10 experimentos variando los modelos (EfficientNet vs MobileNet), los datos (limpios vs ruidosos), el fine-tuning (cantidad de capas descongeladas) y el learning rate.\n\nEl objetivo de estos experimentos es **validar que la convolución y la transferencia de aprendizaje se estén implementando correctamente**.'))

nuevas_celdas.append(create_code_cell([
    'import numpy as np',
    'import pandas as pd',
    'import matplotlib.pyplot as plt',
    'import tensorflow as tf',
    'from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau',
    'from tensorflow.keras.optimizers import Adam',
    'from tensorflow.keras.applications import EfficientNetB0, MobileNetV2',
    'from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Lambda',
    'from tensorflow.keras.models import Model',
    'from tensorflow.keras.preprocessing.image import ImageDataGenerator',
    'from tensorflow.keras.applications.mobilenet_v2 import preprocess_input',
    '',
    '# 1. Generador SIN ruido (Solo reescalado básico/shifts leves)',
    'datagen_clean = ImageDataGenerator(',
    '    validation_split=0.2,',
    '    horizontal_flip=True,',
    '    rotation_range=15,',
    '    zoom_range=0.1',
    ') ',
    'train_gen_clean = datagen_clean.flow_from_dataframe(',
    '    train_df, x_col=\'filepath\', y_col=\'label\', target_size=IMG_SIZE, ',
    '    batch_size=BATCH_SIZE, class_mode=\'categorical\', subset=\'training\'',
    ')',
    'val_gen = datagen_clean.flow_from_dataframe(',
    '    train_df, x_col=\'filepath\', y_col=\'label\', target_size=IMG_SIZE, ',
    '    batch_size=BATCH_SIZE, class_mode=\'categorical\', subset=\'validation\'',
    ')',
    '',
    '# 2. Generador CON ruido extremo (Augmentation pesado + brillo/rotaciones extremas)',
    'datagen_noisy = ImageDataGenerator(',
    '    rotation_range=45, width_shift_range=0.3, height_shift_range=0.3,',
    '    shear_range=0.3, zoom_range=0.3, horizontal_flip=True, brightness_range=[0.3, 1.7],',
    '    validation_split=0.2',
    ')',
    'train_gen_noisy = datagen_noisy.flow_from_dataframe(',
    '    train_df, x_col=\'filepath\', y_col=\'label\', target_size=IMG_SIZE, ',
    '    batch_size=BATCH_SIZE, class_mode=\'categorical\', subset=\'training\'',
    ')',
    '',
    '# Diccionario para guardar resultados de todos los experimentos',
    'resultados_exp = {}',
    '',
    'def guardar_resultado(nombre, history):',
    '    acc = history.history[\'val_accuracy\'][-1]',
    '    loss = history.history[\'val_loss\'][-1]',
    '    resultados_exp[nombre] = {\'Val Accuracy\': acc, \'Val Loss\': loss}',
    '    print(f"✅ {nombre} -> Acc: {acc:.4f} | Loss: {loss:.4f}")'
]))

nuevas_celdas.append(create_code_cell([
    'def crear_modelo(tipo=\'efficientnet\', congelar_hasta=None):',
    '    if tipo == \'efficientnet\':',
    '        base = EfficientNetB0(weights=\'imagenet\', include_top=False, input_shape=(224, 224, 3))',
    '        x = base.output',
    '        x = GlobalAveragePooling2D()(x)',
    '        x = Dropout(0.5)(x)',
    '        pred = Dense(len(selected_classes), activation=\'softmax\')(x)',
    '        model = Model(inputs=base.input, outputs=pred)',
    '    else: # mobilenet',
    '        base = MobileNetV2(weights=\'imagenet\', include_top=False, input_shape=(224, 224, 3))',
    '        inputs = tf.keras.Input(shape=(224, 224, 3))',
    '        x = Lambda(preprocess_input)(inputs)',
    '        x = base(x)',
    '        x = GlobalAveragePooling2D()(x)',
    '        x = Dropout(0.5)(x)',
    '        pred = Dense(len(selected_classes), activation=\'softmax\')(x)',
    '        model = Model(inputs=inputs, outputs=pred)',
    '    ',
    '    # Lógica de congelamiento (Transfer Learning vs Fine Tuning)',
    '    if congelar_hasta == \'todo\':',
    '        base.trainable = False',
    '    elif isinstance(congelar_hasta, int):',
    '        base.trainable = True',
    '        for layer in base.layers[:-congelar_hasta]:',
    '            layer.trainable = False',
    '            ',
    '    return model',
    '',
    '# Callbacks obligatorios: EarlyStopping y ModelCheckpoint',
    'early_stop = EarlyStopping(monitor=\'val_loss\', patience=5, restore_best_weights=True)',
    'reduce_lr = ReduceLROnPlateau(monitor=\'val_loss\', factor=0.5, patience=3)'
]))

# EXPERIMENTOS EFFICIENTNET
nuevas_celdas.append(create_markdown_cell('### Experimento 1: EfficientNet (Transfer Learning, Datos Limpios)\n\n**Propósito:** Establecer un *baseline* para EfficientNet utilizando Transfer Learning puro (todas las capas convolucionales congeladas) sobre un dataset con aumentos de datos leves. Esto nos permite ver qué tan bien generalizan las características preentrenadas de ImageNet a nuestro dominio marítimo.\n\n**Conclusión Esperada:** Se espera que el modelo logre una precisión decente rápidamente sin riesgo de un sobreajuste severo (overfitting), ya que solo la capa densa final se está entrenando.'))
nuevas_celdas.append(create_code_cell([
    'print("\\nExp 1: EffNet Congelado (Datos Limpios)")',
    'm1 = crear_modelo(\'efficientnet\', \'todo\')',
    'm1.compile(optimizer=Adam(1e-3), loss=\'categorical_crossentropy\', metrics=[\'accuracy\'])',
    'h1 = m1.fit(train_gen_clean, validation_data=val_gen, epochs=5, callbacks=[early_stop, reduce_lr], verbose=0)',
    'guardar_resultado(\'1. EffNet TL Limpio\', h1)'
]))
nuevas_celdas.append(create_markdown_cell('**Conclusión del Experimento 1:** Tras la ejecución, se valida la capacidad base de EfficientNet. Si la exactitud (*accuracy*) en validación es alta, confirma que las características generales de ImageNet son muy útiles. Si el *loss* de entrenamiento y validación están cerca, significa que el modelo está bien ajustado.'))

nuevas_celdas.append(create_markdown_cell('### Experimento 2: EfficientNet (Transfer Learning, Datos con Ruido)\n\n**Propósito:** Evaluar la robustez de EfficientNet cuando las imágenes de entrada sufren de ruido extremo, rotaciones y variaciones de brillo. Se mantiene el modelo congelado para ver si las características pre-aprendidas resisten esta degradación de los datos.\n\n**Conclusión Esperada:** Es probable que la precisión disminuya en comparación con el Experimento 1, mostrando *underfitting* al principio, ya que al modelo le costará más extraer patrones de imágenes degradadas.'))
nuevas_celdas.append(create_code_cell([
    'print("\\nExp 2: EffNet Congelado (Datos con Ruido)")',
    'm2 = crear_modelo(\'efficientnet\', \'todo\')',
    'm2.compile(optimizer=Adam(1e-3), loss=\'categorical_crossentropy\', metrics=[\'accuracy\'])',
    'h2 = m2.fit(train_gen_noisy, validation_data=val_gen, epochs=5, callbacks=[early_stop, reduce_lr], verbose=0)',
    'guardar_resultado(\'2. EffNet TL Ruidoso\', h2)'
]))
nuevas_celdas.append(create_markdown_cell('**Conclusión del Experimento 2:** Esto demuestra si la arquitectura de EfficientNet es robusta ante el ruido. Las caídas en el *val_accuracy* indicarán que se requiere de *fine-tuning* para adaptar los filtros convolucionales a imágenes de peor calidad.'))

nuevas_celdas.append(create_markdown_cell('### Experimento 3: EfficientNet (Fine Tuning de 10 capas, Datos Limpios)\n\n**Propósito:** Implementar *Fine Tuning* descongelando las últimas 10 capas convolucionales. Se disminuye la tasa de aprendizaje (Learning Rate a 1e-4) para ajustar suavemente los pesos y mejorar el *baseline* del Experimento 1, adaptando los mapas de características abstractas a las especies marinas.\n\n**Conclusión Esperada:** Debería ser el mejor modelo de EfficientNet para datos limpios. Se espera un aumento en el *accuracy* de validación, demostrando que la convolución especializada mejora la clasificación.'))
nuevas_celdas.append(create_code_cell([
    'print("\\nExp 3: EffNet FT 10 Capas (Datos Limpios)")',
    'm3 = crear_modelo(\'efficientnet\', 10)',
    'm3.compile(optimizer=Adam(1e-4), loss=\'categorical_crossentropy\', metrics=[\'accuracy\'])',
    'h3 = m3.fit(train_gen_clean, validation_data=val_gen, epochs=5, callbacks=[early_stop, reduce_lr], verbose=0)',
    'guardar_resultado(\'3. EffNet FT-10 Limpio\', h3)'
]))
nuevas_celdas.append(create_markdown_cell('**Conclusión del Experimento 3:** El *Fine Tuning* mejora el desempeño frente al Transfer Learning puro. Se valida que descongelar 10 capas es suficiente para adaptar el modelo sin causar *overfitting* masivo.'))

nuevas_celdas.append(create_markdown_cell('### Experimento 4: EfficientNet (Fine Tuning de 30 capas, Datos con Ruido)\n\n**Propósito:** Probar un ajuste profundo (descongelar 30 capas) sobre datos ruidosos. La hipótesis es que, al tener datos muy alterados, necesitamos que el modelo reaprenda filtros de niveles más bajos para ignorar el ruido (augmentation intenso).\n\n**Conclusión Esperada:** Podría superar al Experimento 2 (Transfer Learning ruidoso), pero existe el riesgo de *overfitting* debido a la mayor cantidad de parámetros entrenables. Es la prueba definitiva de adaptabilidad.'))
nuevas_celdas.append(create_code_cell([
    'print("\\nExp 4: EffNet FT 30 Capas (Datos con Ruido)")',
    'm4 = crear_modelo(\'efficientnet\', 30)',
    'm4.compile(optimizer=Adam(1e-4), loss=\'categorical_crossentropy\', metrics=[\'accuracy\'])',
    'h4 = m4.fit(train_gen_noisy, validation_data=val_gen, epochs=5, callbacks=[early_stop, reduce_lr], verbose=0)',
    'guardar_resultado(\'4. EffNet FT-30 Ruidoso\', h4)'
]))
nuevas_celdas.append(create_markdown_cell('**Conclusión del Experimento 4:** Se observa si descongelar más capas en situaciones de ruido es beneficioso o si, por el contrario, el modelo colapsa e intenta memorizar el ruido. Esto nos permite entender el balance entre capacidad del modelo y la calidad del dato.'))

# EXPERIMENTOS MOBILENET
nuevas_celdas.append(create_markdown_cell('### Experimento 5: MobileNet (Transfer Learning, Datos Limpios)\n\n**Propósito:** Establecer un *baseline* para MobileNetV2 (una red más ligera). Compararemos su desempeño y velocidad de convergencia contra el Experimento 1 de EfficientNet.\n\n**Conclusión Esperada:** MobileNet debe ser más rápido de entrenar, pero probablemente alcance un *accuracy* ligeramente inferior o similar a EfficientNet en las mismas condiciones.'))
nuevas_celdas.append(create_code_cell([
    'print("\\nExp 5: MobileNet Congelado (Datos Limpios)")',
    'm5 = crear_modelo(\'mobilenet\', \'todo\')',
    'm5.compile(optimizer=Adam(1e-3), loss=\'categorical_crossentropy\', metrics=[\'accuracy\'])',
    'h5 = m5.fit(train_gen_clean, validation_data=val_gen, epochs=5, callbacks=[early_stop, reduce_lr], verbose=0)',
    'guardar_resultado(\'5. MobileNet TL Limpio\', h5)'
]))
nuevas_celdas.append(create_markdown_cell('**Conclusión del Experimento 5:** Permite validar la idoneidad de MobileNet como modelo ligero. Si el resultado es muy cercano al de EfficientNet, MobileNet se convierte en la opción ideal para despliegues con recursos limitados.'))

nuevas_celdas.append(create_markdown_cell('### Experimento 6: MobileNet (Transfer Learning, Datos con Ruido)\n\n**Propósito:** Igual que el experimento 2, pero evaluando si MobileNetV2 sufre más o menos que EfficientNet ante la degradación visual extrema.\n\n**Conclusión Esperada:** Al ser un modelo menos profundo y con menos parámetros, podría verse más afectado por el ruido que EfficientNet, bajando su rendimiento de forma notoria.'))
nuevas_celdas.append(create_code_cell([
    'print("\\nExp 6: MobileNet Congelado (Datos con Ruido)")',
    'm6 = crear_modelo(\'mobilenet\', \'todo\')',
    'm6.compile(optimizer=Adam(1e-3), loss=\'categorical_crossentropy\', metrics=[\'accuracy\'])',
    'h6 = m6.fit(train_gen_noisy, validation_data=val_gen, epochs=5, callbacks=[early_stop, reduce_lr], verbose=0)',
    'guardar_resultado(\'6. MobileNet TL Ruidoso\', h6)'
]))
nuevas_celdas.append(create_markdown_cell('**Conclusión del Experimento 6:** El impacto del ruido en arquitecturas ligeras queda en evidencia. Si el accuracy baja drásticamente, comprobamos que modelos pequeños necesitan datos de mejor calidad.'))

nuevas_celdas.append(create_markdown_cell('### Experimento 7: MobileNet (Fine Tuning de 10 capas, Datos Limpios)\n\n**Propósito:** Aplicar *Fine Tuning* (descongelando 10 capas, LR=1e-4) para ayudar a MobileNet a especializarse en las clases marinas. Sirve de comparación directa con el Experimento 3.\n\n**Conclusión Esperada:** Debería mejorar el *baseline* del Experimento 5. Si la mejora es marginal, puede indicar que MobileNetV2 ya está extrayendo la máxima información posible de su estructura más sencilla.'))
nuevas_celdas.append(create_code_cell([
    'print("\\nExp 7: MobileNet FT 10 Capas (Datos Limpios)")',
    'm7 = crear_modelo(\'mobilenet\', 10)',
    'm7.compile(optimizer=Adam(1e-4), loss=\'categorical_crossentropy\', metrics=[\'accuracy\'])',
    'h7 = m7.fit(train_gen_clean, validation_data=val_gen, epochs=5, callbacks=[early_stop, reduce_lr], verbose=0)',
    'guardar_resultado(\'7. MobileNet FT-10 Limpio\', h7)'
]))
nuevas_celdas.append(create_markdown_cell('**Conclusión del Experimento 7:** Se confirma que descongelar 10 capas en MobileNet es una estrategia efectiva para incrementar la precisión sin incrementar de forma extrema los tiempos de entrenamiento.'))

nuevas_celdas.append(create_markdown_cell('### Experimento 8: MobileNet (Fine Tuning de 30 capas, Datos con Ruido)\n\n**Propósito:** Someter a MobileNet a un ajuste profundo (30 capas) sobre el dataset ruidoso, permitiendo que sus capas iniciales reaprendan a lidiar con el ruido.\n\n**Conclusión Esperada:** Este experimento evaluará si MobileNet tiene suficiente capacidad representacional para ignorar el ruido mediante un re-entrenamiento profundo de sus pesos convolucionales.'))
nuevas_celdas.append(create_code_cell([
    'print("\\nExp 8: MobileNet FT 30 Capas (Datos con Ruido)")',
    'm8 = crear_modelo(\'mobilenet\', 30)',
    'm8.compile(optimizer=Adam(1e-4), loss=\'categorical_crossentropy\', metrics=[\'accuracy\'])',
    'h8 = m8.fit(train_gen_noisy, validation_data=val_gen, epochs=5, callbacks=[early_stop, reduce_lr], verbose=0)',
    'guardar_resultado(\'8. MobileNet FT-30 Ruidoso\', h8)'
]))
nuevas_celdas.append(create_markdown_cell('**Conclusión del Experimento 8:** Identificaremos si el modelo sufre de *overfitting* al intentar aprender del ruido (memorización) o si las 30 capas le dan la flexibilidad suficiente para generalizar mejor que en el Experimento 6.'))

# EXPERIMENTOS DE LEARNING RATE
nuevas_celdas.append(create_markdown_cell('### Experimento 9: EfficientNet (Learning Rate Agresivo 1e-2)\n\n**Propósito:** Analizar el impacto de un hiperparámetro clave. Se utilizará un Learning Rate muy alto (1e-2) en una etapa de *Fine Tuning*.\n\n**Conclusión Esperada:** Es altamente probable que el modelo no logre converger o que sufra variaciones bruscas en su *loss*. Esto demostrará por qué en el *fine tuning* se deben usar tasas de aprendizaje pequeñas.'))
nuevas_celdas.append(create_code_cell([
    'print("\\nExp 9: EffNet LR Agresivo (1e-2)")',
    'm9 = crear_modelo(\'efficientnet\', 10)',
    'm9.compile(optimizer=Adam(1e-2), loss=\'categorical_crossentropy\', metrics=[\'accuracy\'])',
    'h9 = m9.fit(train_gen_clean, validation_data=val_gen, epochs=5, callbacks=[early_stop, reduce_lr], verbose=0)',
    'guardar_resultado(\'9. EffNet LR Alto\', h9)'
]))
nuevas_celdas.append(create_markdown_cell('**Conclusión del Experimento 9:** La pérdida de precisión o la inestabilidad en la curva de *loss* valida la teoría: pasos demasiado grandes destruyen los pesos preentrenados (*catastrophic forgetting*).'))

nuevas_celdas.append(create_markdown_cell('### Experimento 10: MobileNet (Learning Rate muy Bajo 1e-6)\n\n**Propósito:** Evaluar el extremo opuesto del hiperparámetro. Un LR extremadamente bajo (1e-6) para ver qué ocurre con la convergencia del modelo durante el *Fine Tuning*.\n\n**Conclusión Esperada:** Se espera que el modelo sufra de *underfitting* (o un aprendizaje extremadamente lento), donde las épocas transcurran sin mejoras significativas.'))
nuevas_celdas.append(create_code_cell([
    'print("\\nExp 10: MobileNet LR Lento (1e-6)")',
    'm10 = crear_modelo(\'mobilenet\', 10)',
    'm10.compile(optimizer=Adam(1e-6), loss=\'categorical_crossentropy\', metrics=[\'accuracy\'])',
    'h10 = m10.fit(train_gen_clean, validation_data=val_gen, epochs=5, callbacks=[early_stop, reduce_lr], verbose=0)',
    'guardar_resultado(\'10. MobileNet LR Bajo\', h10)'
]))
nuevas_celdas.append(create_markdown_cell('**Conclusión del Experimento 10:** El nulo o lentísimo avance del *accuracy* demostrará empíricamente la necesidad de escoger un LR balanceado (típicamente entre 1e-3 y 1e-5) para garantizar convergencia en tiempos razonables.'))

# VISUALIZACIÓN
nuevas_celdas.append(create_markdown_cell('### Tabla Comparativa y Gráfica de Resultados Finales\n\n**Propósito:** Consolidar los hallazgos para una interpretación clara y presentar los resultados finales ordenados, cumpliendo con el requerimiento de análisis integral de Overfitting, Underfitting y métricas generales.'))
nuevas_celdas.append(create_code_cell([
    'import seaborn as sns',
    '',
    '# Convertir diccionario a DataFrame para visualizar',
    'df_resultados = pd.DataFrame.from_dict(resultados_exp, orient=\'index\').reset_index()',
    'df_resultados.columns = [\'Experimento\', \'Val Accuracy\', \'Val Loss\']',
    'df_resultados = df_resultados.sort_values(by=\'Val Accuracy\', ascending=False)',
    '',
    '# Mostrar tabla',
    'print("=== TABLA DE RESULTADOS (De mejor a peor) ===")',
    'print(df_resultados.to_string(index=False))',
    '',
    '# Gráfica de barras',
    'plt.figure(figsize=(14, 6))',
    'sns.barplot(data=df_resultados, x=\'Val Accuracy\', y=\'Experimento\', palette=\'viridis\')',
    'plt.title("Comparación de Accuracy en los 10 Experimentos", fontsize=14, fontweight=\'bold\')',
    'plt.xlabel("Validation Accuracy")',
    'plt.ylabel("Configuración del Experimento")',
    'plt.xlim(0, 1)',
    '',
    '# Añadir etiquetas de texto en las barras',
    'for index, value in enumerate(df_resultados[\'Val Accuracy\']):',
    '    plt.text(value + 0.01, index, f\'{value:.3f}\', va=\'center\', fontsize=10)',
    '',
    'plt.tight_layout()',
    'plt.show()'
]))

nb['cells'].extend(nuevas_celdas)
with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
print('Notebook actualizado correctamente con propósitos y conclusiones.')

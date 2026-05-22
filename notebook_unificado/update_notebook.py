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
        if 'PARTE 5: Análisis de Clustering' in source or 'PARTE 5' in source:
            start_idx = i
            break

if start_idx != -1:
    print(f'PARTE 5 encontrada en el índice {start_idx}')
    nb['cells'] = nb['cells'][:start_idx]
else:
    print('No se encontró la PARTE 5, se añadirá al final.')

nuevas_celdas = [
    {
        'cell_type': 'markdown',
        'metadata': {},
        'source': ['# =============================================\n', '# PARTE 5: 10 Experimentos de Modelado\n', '# =============================================']
    },
    {
        'cell_type': 'code',
        'execution_count': None,
        'metadata': {},
        'outputs': [],
        'source': [
            'import numpy as np\n',
            'import pandas as pd\n',
            'import matplotlib.pyplot as plt\n',
            'import tensorflow as tf\n',
            'from tensorflow.keras.callbacks import EarlyStopping\n',
            'from tensorflow.keras.optimizers import Adam\n',
            'from tensorflow.keras.applications import EfficientNetB0, MobileNetV2\n',
            'from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Lambda\n',
            'from tensorflow.keras.models import Model\n',
            'from tensorflow.keras.preprocessing.image import ImageDataGenerator\n',
            'from tensorflow.keras.applications.mobilenet_v2 import preprocess_input\n',
            '\n',
            '# 1. Generador SIN ruido (Solo reescalado básico/shifts leves)\n',
            'datagen_clean = ImageDataGenerator(validation_split=0.2) \n',
            'train_gen_clean = datagen_clean.flow_from_dataframe(\n',
            '    train_df, x_col=\'filepath\', y_col=\'label\', target_size=IMG_SIZE, \n',
            '    batch_size=BATCH_SIZE, class_mode=\'categorical\'\n',
            ')\n',
            '\n',
            '# 2. Generador CON ruido extremo (Augmentation pesado + brillo/rotaciones extremas)\n',
            'datagen_noisy = ImageDataGenerator(\n',
            '    rotation_range=45, width_shift_range=0.3, height_shift_range=0.3,\n',
            '    shear_range=0.3, zoom_range=0.3, horizontal_flip=True, brightness_range=[0.3, 1.7]\n',
            ')\n',
            'train_gen_noisy = datagen_noisy.flow_from_dataframe(\n',
            '    train_df, x_col=\'filepath\', y_col=\'label\', target_size=IMG_SIZE, \n',
            '    batch_size=BATCH_SIZE, class_mode=\'categorical\'\n',
            ')\n',
            '\n',
            '# Diccionario para guardar resultados de todos los experimentos\n',
            'resultados_exp = {}\n',
            '\n',
            'def guardar_resultado(nombre, history):\n',
            '    acc = history.history[\'val_accuracy\'][-1]\n',
            '    loss = history.history[\'val_loss\'][-1]\n',
            '    resultados_exp[nombre] = {\'Val Accuracy\': acc, \'Val Loss\': loss}\n',
            '    print(f"✅ {nombre} -> Acc: {acc:.4f} | Loss: {loss:.4f}")\n'
        ]
    },
    {
        'cell_type': 'code',
        'execution_count': None,
        'metadata': {},
        'outputs': [],
        'source': [
            'def crear_modelo(tipo=\'efficientnet\', congelar_hasta=None):\n',
            '    if tipo == \'efficientnet\':\n',
            '        base = EfficientNetB0(weights=\'imagenet\', include_top=False, input_shape=(224, 224, 3))\n',
            '        x = base.output\n',
            '        x = GlobalAveragePooling2D()(x)\n',
            '        x = Dropout(0.5)(x)\n',
            '        pred = Dense(len(selected_classes), activation=\'softmax\')(x)\n',
            '        model = Model(inputs=base.input, outputs=pred)\n',
            '    else: # mobilenet\n',
            '        base = MobileNetV2(weights=\'imagenet\', include_top=False, input_shape=(224, 224, 3))\n',
            '        inputs = tf.keras.Input(shape=(224, 224, 3))\n',
            '        x = Lambda(preprocess_input)(inputs)\n',
            '        x = base(x)\n',
            '        x = GlobalAveragePooling2D()(x)\n',
            '        x = Dropout(0.5)(x)\n',
            '        pred = Dense(len(selected_classes), activation=\'softmax\')(x)\n',
            '        model = Model(inputs=inputs, outputs=pred)\n',
            '    \n',
            '    # Lógica de congelamiento\n',
            '    if congelar_hasta == \'todo\':\n',
            '        base.trainable = False\n',
            '    elif isinstance(congelar_hasta, int):\n',
            '        base.trainable = True\n',
            '        for layer in base.layers[:-congelar_hasta]:\n',
            '            layer.trainable = False\n',
            '            \n',
            '    return model\n',
            '\n',
            'early_stop = EarlyStopping(monitor=\'val_loss\', patience=3, restore_best_weights=True)\n'
        ]
    },
    {
        'cell_type': 'code',
        'execution_count': None,
        'metadata': {},
        'outputs': [],
        'source': [
            'print("--- EXPERIMENTOS EFFICIENTNET ---")\n',
            '\n',
            '# EXP 1: EfficientNet (Congelado, Sin Ruido)\n',
            'print("\\nExp 1: EffNet Congelado (Datos Limpios)")\n',
            'm1 = crear_modelo(\'efficientnet\', \'todo\')\n',
            'm1.compile(optimizer=Adam(1e-3), loss=\'categorical_crossentropy\', metrics=[\'accuracy\'])\n',
            'h1 = m1.fit(train_gen_clean, validation_data=val_gen, epochs=5, callbacks=[early_stop], verbose=0)\n',
            'guardar_resultado(\'1. EffNet TL Limpio\', h1)\n',
            '\n',
            '# EXP 2: EfficientNet (Congelado, Con Ruido)\n',
            'print("\\nExp 2: EffNet Congelado (Datos con Ruido)")\n',
            'm2 = crear_modelo(\'efficientnet\', \'todo\')\n',
            'm2.compile(optimizer=Adam(1e-3), loss=\'categorical_crossentropy\', metrics=[\'accuracy\'])\n',
            'h2 = m2.fit(train_gen_noisy, validation_data=val_gen, epochs=5, callbacks=[early_stop], verbose=0)\n',
            'guardar_resultado(\'2. EffNet TL Ruidoso\', h2)\n',
            '\n',
            '# EXP 3: EfficientNet (Descongelar 10 capas, Sin Ruido)\n',
            'print("\\nExp 3: EffNet FT 10 Capas (Datos Limpios)")\n',
            'm3 = crear_modelo(\'efficientnet\', 10)\n',
            'm3.compile(optimizer=Adam(1e-4), loss=\'categorical_crossentropy\', metrics=[\'accuracy\'])\n',
            'h3 = m3.fit(train_gen_clean, validation_data=val_gen, epochs=5, callbacks=[early_stop], verbose=0)\n',
            'guardar_resultado(\'3. EffNet FT-10 Limpio\', h3)\n',
            '\n',
            '# EXP 4: EfficientNet (Descongelar 30 capas, Con Ruido)\n',
            'print("\\nExp 4: EffNet FT 30 Capas (Datos con Ruido)")\n',
            'm4 = crear_modelo(\'efficientnet\', 30)\n',
            'm4.compile(optimizer=Adam(1e-4), loss=\'categorical_crossentropy\', metrics=[\'accuracy\'])\n',
            'h4 = m4.fit(train_gen_noisy, validation_data=val_gen, epochs=5, callbacks=[early_stop], verbose=0)\n',
            'guardar_resultado(\'4. EffNet FT-30 Ruidoso\', h4)\n'
        ]
    },
    {
        'cell_type': 'code',
        'execution_count': None,
        'metadata': {},
        'outputs': [],
        'source': [
            'print("--- EXPERIMENTOS MOBILENET ---")\n',
            '\n',
            '# EXP 5: MobileNet (Congelado, Sin Ruido)\n',
            'print("\\nExp 5: MobileNet Congelado (Datos Limpios)")\n',
            'm5 = crear_modelo(\'mobilenet\', \'todo\')\n',
            'm5.compile(optimizer=Adam(1e-3), loss=\'categorical_crossentropy\', metrics=[\'accuracy\'])\n',
            'h5 = m5.fit(train_gen_clean, validation_data=val_gen, epochs=5, callbacks=[early_stop], verbose=0)\n',
            'guardar_resultado(\'5. MobileNet TL Limpio\', h5)\n',
            '\n',
            '# EXP 6: MobileNet (Congelado, Con Ruido)\n',
            'print("\\nExp 6: MobileNet Congelado (Datos con Ruido)")\n',
            'm6 = crear_modelo(\'mobilenet\', \'todo\')\n',
            'm6.compile(optimizer=Adam(1e-3), loss=\'categorical_crossentropy\', metrics=[\'accuracy\'])\n',
            'h6 = m6.fit(train_gen_noisy, validation_data=val_gen, epochs=5, callbacks=[early_stop], verbose=0)\n',
            'guardar_resultado(\'6. MobileNet TL Ruidoso\', h6)\n',
            '\n',
            '# EXP 7: MobileNet (Descongelar 10 capas, Sin Ruido)\n',
            'print("\\nExp 7: MobileNet FT 10 Capas (Datos Limpios)")\n',
            'm7 = crear_modelo(\'mobilenet\', 10)\n',
            'm7.compile(optimizer=Adam(1e-4), loss=\'categorical_crossentropy\', metrics=[\'accuracy\'])\n',
            'h7 = m7.fit(train_gen_clean, validation_data=val_gen, epochs=5, callbacks=[early_stop], verbose=0)\n',
            'guardar_resultado(\'7. MobileNet FT-10 Limpio\', h7)\n',
            '\n',
            '# EXP 8: MobileNet (Descongelar 30 capas, Con Ruido)\n',
            'print("\\nExp 8: MobileNet FT 30 Capas (Datos con Ruido)")\n',
            'm8 = crear_modelo(\'mobilenet\', 30)\n',
            'm8.compile(optimizer=Adam(1e-4), loss=\'categorical_crossentropy\', metrics=[\'accuracy\'])\n',
            'h8 = m8.fit(train_gen_noisy, validation_data=val_gen, epochs=5, callbacks=[early_stop], verbose=0)\n',
            'guardar_resultado(\'8. MobileNet FT-30 Ruidoso\', h8)\n'
        ]
    },
    {
        'cell_type': 'code',
        'execution_count': None,
        'metadata': {},
        'outputs': [],
        'source': [
            'print("--- EXPERIMENTOS DE LEARNING RATE ---")\n',
            '\n',
            '# EXP 9: EfficientNet LR muy alto (1e-2) -> Tiende a divergir\n',
            'print("\\nExp 9: EffNet LR Agresivo (1e-2)")\n',
            'm9 = crear_modelo(\'efficientnet\', 10)\n',
            'm9.compile(optimizer=Adam(1e-2), loss=\'categorical_crossentropy\', metrics=[\'accuracy\'])\n',
            'h9 = m9.fit(train_gen_clean, validation_data=val_gen, epochs=5, callbacks=[early_stop], verbose=0)\n',
            'guardar_resultado(\'9. EffNet LR Alto\', h9)\n',
            '\n',
            '# EXP 10: MobileNet LR muy bajo (1e-6) -> Aprendizaje muy lento\n',
            'print("\\nExp 10: MobileNet LR Lento (1e-6)")\n',
            'm10 = crear_modelo(\'mobilenet\', 10)\n',
            'm10.compile(optimizer=Adam(1e-6), loss=\'categorical_crossentropy\', metrics=[\'accuracy\'])\n',
            'h10 = m10.fit(train_gen_clean, validation_data=val_gen, epochs=5, callbacks=[early_stop], verbose=0)\n',
            'guardar_resultado(\'10. MobileNet LR Bajo\', h10)\n'
        ]
    },
    {
        'cell_type': 'code',
        'execution_count': None,
        'metadata': {},
        'outputs': [],
        'source': [
            'import seaborn as sns\n',
            '\n',
            '# Convertir diccionario a DataFrame para visualizar\n',
            'df_resultados = pd.DataFrame.from_dict(resultados_exp, orient=\'index\').reset_index()\n',
            'df_resultados.columns = [\'Experimento\', \'Val Accuracy\', \'Val Loss\']\n',
            'df_resultados = df_resultados.sort_values(by=\'Val Accuracy\', ascending=False)\n',
            '\n',
            '# Mostrar tabla\n',
            'print("=== TABLA DE RESULTADOS (De mejor a peor) ===")\n',
            'print(df_resultados.to_string(index=False))\n',
            '\n',
            '# Gráfica de barras\n',
            'plt.figure(figsize=(14, 6))\n',
            'sns.barplot(data=df_resultados, x=\'Val Accuracy\', y=\'Experimento\', palette=\'viridis\')\n',
            'plt.title("Comparación de Accuracy en los 10 Experimentos", fontsize=14, fontweight=\'bold\')\n',
            'plt.xlabel("Validation Accuracy")\n',
            'plt.ylabel("Configuración del Experimento")\n',
            'plt.xlim(0, 1)\n',
            '\n',
            '# Añadir etiquetas de texto en las barras\n',
            'for index, value in enumerate(df_resultados[\'Val Accuracy\']):\n',
            '    plt.text(value + 0.01, index, f\'{value:.3f}\', va=\'center\', fontsize=10)\n',
            '\n',
            'plt.tight_layout()\n',
            'plt.show()\n'
        ]
    }
]

nb['cells'].extend(nuevas_celdas)
with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
print('Notebook actualizado correctamente.')

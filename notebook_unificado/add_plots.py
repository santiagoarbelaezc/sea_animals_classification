import json

path = 'c:/Users/Santiago/OneDrive/Escritorio/Repositorios/sea_animals_classification/notebook_unificado/Proyecto_Final_6Clases_Colab.ipynb'
with open(path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

experiments = [
    (41, 'h1', '1. EffNet TL Limpio'),
    (44, 'h2', '2. EffNet TL Ruidoso'),
    (47, 'h3', '3. EffNet FT-10 Limpio'),
    (50, 'h4', '4. EffNet FT-30 Ruidoso'),
    (53, 'h5', '5. MobileNet TL Limpio'),
    (56, 'h6', '6. MobileNet TL Ruidoso'),
    (59, 'h7', '7. MobileNet FT-10 Limpio'),
    (62, 'h8', '8. MobileNet FT-30 Ruidoso'),
    (65, 'h9', '9. EffNet LR Alto'),
    (68, 'h10', '10. MobileNet LR Bajo'),
]

for idx, var, title in experiments:
    source = nb['cells'][idx]['source']
    # Check if the plot function is already appended
    has_plot = False
    for line in source:
        if f"plot_hist({var}" in line:
            has_plot = True
            break
            
    if not has_plot:
        if len(source) > 0 and not source[-1].endswith('\n'):
            source[-1] += '\n'
        source.append(f"plot_hist({var}, '{title}')\n")

with open(path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
    
print('Notebook updated successfully.')

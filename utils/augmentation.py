from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess
from .preprocessing import custom_preprocessing

def get_augmentation_generator(model_type='efficientnet', use_custom_low_quality=True):
    """
    Retorna un ImageDataGenerator configurado según el modelo y con aumentos.
    """
    
    if model_type.lower() == 'efficientnet':
        base_preprocess = efficientnet_preprocess
    elif model_type.lower() == 'mobilenet':
        base_preprocess = mobilenet_preprocess
    else:
        base_preprocess = None

    def full_preprocessing(image):
        if use_custom_low_quality:
            image = custom_preprocessing(image)
        if base_preprocess:
            image = base_preprocess(image)
        return image

    datagen = ImageDataGenerator(
        preprocessing_function=full_preprocessing,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        brightness_range=[0.8, 1.2]
    )
    
    # Para validación y test no aplicamos aumentos, solo preprocesamiento
    val_datagen = ImageDataGenerator(
        preprocessing_function=full_preprocessing
    )
    
    return datagen, val_datagen

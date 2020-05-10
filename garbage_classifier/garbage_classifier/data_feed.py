import numpy as np
import imgaug as ia
from tensorflow import keras

def data_feed(path, sample_path, input_size, batch_size=4, validation_split=0.1):
    def image_preprocessing(images):
        sometimes = lambda x: ia.augmenters.Sometimes(0.5, x)
        augmentation_sequence = ia.augmenters.Sequential([
            sometimes(ia.augmenters.AdditiveGaussianNoise(loc=(0.0, 0.1))),
            sometimes(ia.augmenters.Add(-0.1, 0.1))
        ])
        return augmentation_sequence(images=images)
    data_generator = keras.preprocessing.image.ImageDataGenerator(
        featurewise_center=True,
        featurewise_std_normalization=True,
        rotation_range=180,
        width_shift_range=0.2,
        height_shift_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        preprocessing_function=image_preprocessing,
        validation_split=validation_split
    )
    data_generator.fit(np.load(sample_path))
    flow_args = {
        'directory': path,
        'target_size': input_size,
        'color_mode': 'rgb',
        'batch_size': batch_size,
        'class_mode': 'categorical',
        'save_format': 'jpeg'
    }
    training_data_generator = data_generator.flow_from_directory(
        **flow_args,
        subset='training'
    )
    validation_data_generator = data_generator.flow_from_directory(
        **flow_args,
        subset='validation'
    )
    return training_data_generator, validation_data_generator

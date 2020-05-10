from tensorflow import keras

from . import utils

def _resnet(input_size, class_count, layer_filter_kernels):
    model_input = keras.layers.Input(shape=input_size)
    layer = utils._conv_batchnorm(
        filters=64,
        kernel_size=(7, 7),
        strides=(2, 2),
        kernel_regularizer=keras.regularizers.l2(1e-4),
        activation='relu'
    )(model_input)
    layer = keras.layers.MaxPool2D(
        pool_size=(3, 3),
        strides=(2, 2),
        padding='same'
    )(layer)
    for block_index, (layer_count, filters, kernel_size) in enumerate(layer_filter_kernels):
        for layer_index in range(layer_count):
            layer = utils._residual_block(
                filters=filters,
                kernel_size=kernel_size,
                kernel_regularizer=keras.regularizers.l2(1e-4),
                is_beginning_block=(layer_index == 0 and block_index != 0)
            )(layer)
    layer = keras.layers.GlobalAveragePooling2D()(layer)
    layer = keras.layers.Dense(
        units=class_count,
        activation='softmax',
        kernel_initializer='he_normal',
        kernel_regularizer=keras.regularizers.l2(1e-4)
    )(layer)
    model = keras.models.Model(inputs=model_input, outputs=layer)
    return model

def _resnet_pretrained(input_size, class_count, base_model):
    base_model = base_model(weights='imagenet', include_top=False)
    for layer in base_model:
        layer.trainable = False
    layer = base_model.output
    layer = keras.layers.GlobalAveragePooling2D()(layer)
    layer = keras.layers.Dense(
        layer.shape[-1], 
        activation='relu',
        kernel_regularizer=keras.regularizers.l2(1e-4)
    )(layer)
    layer = keras.layers.Dense(
        class_count,
        activation='softmax'
    )(layer)
    model = keras.models.Model(inputs=base_model.input, outputs=layer)
    return model

def resnet_34(input_size, class_count):
    return _resnet(
        input_size, 
        class_count, 
        layer_filter_kernels=[
            (3, [64, 64], [(3, 3), (3, 3)]), 
            (4, [128, 128], [(3, 3), (3, 3)]), 
            (6, [256, 256], [(3, 3), (3, 3)]), 
            (3, [512, 512], [(3, 3), (3, 3)])
        ]
    )

def resnet_50(input_size, class_count):
    return _resnet(
        input_size,
        class_count,
        layer_filter_kernels=[
            (3, [64, 64, 256], [(1, 1), (3, 3), (1, 1)]),
            (4, [128, 128, 512], [(1, 1), (3, 3), (1, 1)]),
            (6, [256, 256, 1024], [(1, 1), (3, 3), (1, 1)]),
            (3, [512, 512, 2048], [(1, 1), (3, 3), (1, 1)])
        ]
    )

def resnet_101(input_size, class_count):
    return _resnet(
        input_size,
        class_count,
        layer_filter_kernels=[
            (3, [64, 64, 256], [(1, 1), (3, 3), (1, 1)]),
            (4, [128, 128, 512], [(1, 1), (3, 3), (1, 1)]),
            (23, [256, 256, 1024], [(1, 1), (3, 3), (1, 1)]),
            (3, [512, 512, 2048], [(1, 1), (3, 3), (1, 1)])
        ]
    )

def resnet_152(input_size, class_count):
    return _resnet(
        input_size,
        class_count,
        layer_filter_kernels=[
            (3, [64, 64, 256], [(1, 1), (3, 3), (1, 1)]),
            (8, [128, 128, 512], [(1, 1), (3, 3), (1, 1)]),
            (36, [256, 256, 1024], [(1, 1), (3, 3), (1, 1)]),
            (3, [512, 512, 2048], [(1, 1), (3, 3), (1, 1)])
        ]
    )

def resnet_50_pretrained(input_size, class_count):
    return _resnet_pretrained(input_size, class_count, keras.applications.resnet50.ResNet50)

def resnet_101_pretrained(input_size, class_count):
    return _resnet_pretrained(input_size, class_count, keras.applications.resnet101.ResNet101)

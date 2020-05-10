from tensorflow import keras

def _conv_batchnorm(filters, kernel_size, strides=(1, 1), padding='valid', kernel_regularizer=None, activation='linear'):
    def structure(input_tensor):
        layer = keras.layers.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            kernel_initializer='he_normal',
            kernel_regularizer=kernel_regularizer
        )(input_tensor)
        layer = keras.layers.BatchNormalization()(layer)
        layer = keras.layers.Activation(activation)(layer)
        return layer
    return structure

def _residual_block(filters, kernel_size, kernel_regularizer=None, is_beginning_block=False):
    assert len(filters) == len(kernel_size)
    def structure(input_tensor):
        residual, layer = input_tensor, input_tensor
        for index, (f, k) in enumerate(zip(filters, kernel_size)):
            layer = _conv_batchnorm(
                filters=f,
                kernel_size=k,
                strides=(2, 2) if is_beginning_block and index == 0 else (1, 1),
                padding='same',
                kernel_regularizer=kernel_regularizer,
                activation='linear' if index == len(filters) - 1 else 'relu'
            )(layer)
        if tuple(residual.shape) != tuple(layer.shape):
            residual = _conv_batchnorm(
                filters=layer.shape[-1],
                kernel_size=(1, 1),
                strides=[*map(lambda x: int(round(x[0] / x[1])), [*zip(residual.shape, layer.shape)][1:3])],
                kernel_regularizer=kernel_regularizer
            )(residual)
        layer = keras.layers.Add()([residual, layer])
        layer = keras.layers.Activation('relu')(layer)
        return layer
    return structure

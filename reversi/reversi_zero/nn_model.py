import numpy as np
from tensorflow import keras


class NNModel(object):

    def __init__(self):
        self._create_model()
    
    def predict(self, state):
        result = self.model.predict(np.array([state]))
        return result[0][0], result[1][0]
    
    def fit(self, states, policies, values, batch_size=None, epochs=1):
        self.model.fit(
            x=states,
            y=[policies, values],
            batch_size=batch_size,
            epochs=epochs
        )
    
    def save(self, path):
        self.model.save(path)
    
    def load(self, path):
        self.model = keras.models.load_model(path)
    
    def clone(self):
        cloned_model = keras.models.clone_model(self.model)
        cloned_self = NNModel()
        cloned_self.model = cloned_model
        return cloned_self

    def _create_model(self):
        model_input = keras.layers.Input(
            shape=(8, 8, 2),
            dtype=np.float32
        )
        
        # Residual blocks
        layer = NNModel._conv_batchnorm(
            filters=128, 
            kernel_size=(3, 3), 
            padding='same', 
            kernel_regularizer=keras.regularizers.l2(1e-4), 
            activation='relu'
        )(model_input)
        for _ in range(6):
            layer = NNModel._residual_block(
                filters=128, 
                kernel_size=(3, 3), 
                padding='same', 
                kernel_regularizer=keras.regularizers.l2(1e-4)
            )(layer)
        
        # Value generation
        value = layer
        value = NNModel._conv_batchnorm(
            filters=1, 
            kernel_size=(1, 1), 
            kernel_regularizer=keras.regularizers.l2(1e-4), 
            activation='relu'
        )(value)
        value = keras.layers.Flatten()(value)
        value = keras.layers.Dense(
            units=64,
            activation='relu',
            kernel_regularizer=keras.regularizers.l2(1e-4)
        )(value)
        value = keras.layers.Dense(
            units=1,
            activation='tanh',
            name='value'
        )(value)

        # Policy generation
        policy = layer
        policy = NNModel._conv_batchnorm(
            filters=2, 
            kernel_size=(1, 1), 
            kernel_regularizer=keras.regularizers.l2(1e-4), 
            activation='relu'
        )(policy)
        policy = keras.layers.Flatten()(policy)
        policy = keras.layers.Dense(
            units=8 * 8,
            activation='softmax',
            name='policy'
        )(policy)

        # Model assemble
        model = keras.models.Model(inputs=[model_input], outputs=[policy, value])
        model.compile(
            optimizer=keras.optimizers.Adadelta(),
            loss=[keras.losses.categorical_crossentropy, keras.losses.mean_squared_error],
            loss_weights=[0.5, 0.5],
            metrics=['accuracy']
        )
        self.model = model
    
    @staticmethod
    def _conv_batchnorm(filters, kernel_size, padding='valid', kernel_regularizer=None, activation='linear'):
        def structure(input_tensor):
            layer = keras.layers.Conv2D(
                filters=filters,
                kernel_size=kernel_size,
                padding=padding,
                kernel_regularizer=kernel_regularizer
            )(input_tensor)
            layer = keras.layers.BatchNormalization()(layer)
            layer = keras.layers.Activation(activation)(layer)
            return layer
        return structure
    
    @staticmethod
    def _residual_block(filters, kernel_size, padding='valid', kernel_regularizer=None):
        def structure(input_tensor):
            residual = input_tensor
            layer = NNModel._conv_batchnorm(
                filters=128, 
                kernel_size=(3, 3), 
                padding='same', 
                kernel_regularizer=keras.regularizers.l2(1e-4),
                activation='relu'
            )(input_tensor)
            layer = NNModel._conv_batchnorm(
                filters=128, 
                kernel_size=(3, 3), 
                padding='same', 
                kernel_regularizer=keras.regularizers.l2(1e-4),
            )(layer)
            layer = keras.layers.Add()([residual, layer])
            layer = keras.layers.Activation('relu')(layer)
            return layer
        return structure

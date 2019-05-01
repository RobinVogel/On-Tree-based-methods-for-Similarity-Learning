"""Neural model."""
import json
import numpy as np
from keras.layers import Input, Dense
from keras.models import Model
from keras.optimizers import Adam
from keras import regularizers
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping

with open("treerank/params.json", "rt") as f:
    PARAMS = json.load(f)

# Basis from: https://keras.io/getting-started/functional-api-guide/
class NeuralModel:
    """Gets a tensor."""

    def __init__(self):
        self.name = "nn_sce"
        self.encoding_size = 128
        self.model = None
        self.encode_layer = None

        self.inputs = None
        self.encoding = None
        self.predictions = None

    def fit(self, X, y):
        """Fits the model."""

        input_shape = X.shape[1]
        classes = np.unique(y)
        num_classes = classes.shape[0]

        self.__define_net(input_shape, num_classes)

        self.model = Model(inputs=self.inputs, outputs=self.predictions)
        self.encode_layer = Model(inputs=self.inputs, outputs=self.encoding)

        # Possible optimizers:
        # sgd = optimizers.SGD(lr=0.01, clipnorm=1.)
        adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None,
                    decay=0.0, amsgrad=False)

        # Talk Stéphane:
        # There are many losses. A good first approximation is to use a simple
        # categorical cross-entropy.
        self.model.compile(optimizer=adam,
                           loss='categorical_crossentropy',
                           metrics=['accuracy'])

        new_labels = {v: k for k, v in enumerate(classes)}
        y_trans = np.array([new_labels[c] for c in y])
        y_one_hot = to_categorical(y_trans, num_classes=num_classes)

        overfitCallback = EarlyStopping(monitor='loss', min_delta=0, patience=20)
        # starts training
        self.model.fit(X, y_one_hot, epochs=PARAMS["nn_epoch_no"], 
                       validation_split=0.10, callbacks=[overfitCallback])

    def encode(self, X):
        """Encodes the vector X."""
        return self.encode_layer.predict(X)

    def __define_net(self, input_shape, num_classes):
        self.inputs = Input(shape=(input_shape,))
        
        l1_out_size = (3*input_shape + self.encoding_size)//4
        l2_out_size = (input_shape + self.encoding_size)//2
        l3_out_size = (input_shape + 3*self.encoding_size)//4
        x = Dense(l1_out_size, activation='relu',
                  kernel_regularizer=regularizers.l2(0.01))(self.inputs)
        x = Dense(l2_out_size, activation='relu',
                  kernel_regularizer=regularizers.l2(0.01))(x)
        out = Dense(l3_out_size, activation='relu',
                    kernel_regularizer=regularizers.l2(0.01))(x)

        self.encoding = Dense(self.encoding_size)(out)
        self.predictions = Dense(num_classes,
                                 activation='softmax')(
                                 self.encoding)

if __name__ == "__main__":
    pass

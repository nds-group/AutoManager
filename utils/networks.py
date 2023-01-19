import tensorflow as tf
from keras import Model
from keras.layers import LSTM, Dense, Dropout, Input, concatenate
from keras.models import Sequential


class MultiPred_NN:
    def __init__(self, nb_pred, nb_input, pred_arch):
        """
        Create the MultiPred_NN object
        Inputs:
            -nb_pred: Number of predictor of the model
            -pred_arch: Architecture of each predictor
        """
        super(MultiPred_NN, self).__init__()
        self.nb_pred = nb_pred
        self.nb_input = nb_input
        self.pred_l = []
        for i in range(nb_pred):
            self.pred_l.append(pred_arch(nb_input))

    def train_solo_all(self, datasets_l, epochs=20, batch_size=64, lr=0.001):
        for ind, model in enumerate(self.pred_l):
            print("Training NN nb", str(ind))
            (X, Y) = datasets_l[ind]
            opti = tf.keras.optimizers.Adam(learning_rate=lr)
            dataset = tf.data.Dataset.from_tensor_slices((X, Y))
            dataset = dataset.batch(batch_size, drop_remainder=False)
            for epo in range(epochs):
                for x, y in dataset:
                    with tf.GradientTape() as tape:
                        y_pred = model(x)
                        y_pred = tf.dtypes.cast(y_pred, tf.float64)
                        loss = tf.abs(tf.subtract(y, y_pred))
                    gradients = tape.gradient(loss, model.trainable_weights)
                    opti.apply_gradients(
                        zip(gradients, model.trainable_weights))
        print("All NNs trained individually")

    def combine_preds(self, activ):
        inputs_l = []
        nns_l = []
        for ind in range(self.nb_pred):
            input = Input(shape=(self.nb_input, 1))
            inputs_l.append(input)
            nns_l.append(self.pred_l[ind](input))
        merge = concatenate(nns_l)
        x = Dense(100, input_shape=(self.nb_pred, 1))(merge)
        x = Dense(80, activation=activ)(x)
        x = Dense(50, activation=activ)(x)
        outputs = Dense(units=self.nb_pred)(x)
        model = Model(inputs=inputs_l,
                      outputs=outputs)
        return model




def nnloss_model(activ, errorf, lr, input_size):
    """
    Create the Neural network model as Loss Function
    Inputs:
        - activ: Activation function to use
        - errorf: loss function to use
        - lr: Learning rate to use
        - input_size: Size of the input (must be 2*nbpredic + additional inputs)
    Outputs:
        - model: The loss neural network model
    """
    model = Sequential()
    model.add(Dense(200, activation=activ))
    model.add(Dense(180, activation=activ))
    model.add(Dense(150, activation=activ))
    model.add(Dense(120, activation=activ))
    model.add(Dense(100, activation=activ))
    model.add(Dense(80, activation=activ))
    model.add(Dense(60, activation=activ))
    model.add(Dense(30, activation=activ,
              kernel_regularizer=tf.keras.regularizers.l2(0.0001)))
    model.add(Dense(1))
    model.compile(optimizer=tf.keras.optimizers.Adam(
        learning_rate=lr), loss=errorf)
    return model


def main_nn_model(input):
    """
    Create the Main Neural network
    Inputs:
        -input: Size of the input
    Outputs:
        -model: The neural network model
    """
    inputs = Input(shape=(input, 1))
    x = LSTM(units=160, return_sequences=True,activation='relu')(inputs)
    x = LSTM(units=100, return_sequences=True,activation='relu')(x)
    x = LSTM(units=60, return_sequences=True,activation='relu')(x)
    x = LSTM(units=40)(x)
    x = Dense(30, activation='relu')(x)
    x = Dense(10, activation='relu')(x)
    outputs = Dense(units=1)(x)
    model = Model(inputs=inputs, outputs=outputs)
    return model



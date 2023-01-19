

import time

import numpy as np
import tensorflow as tf
from keras import backend as K


class LossLeaP:
    def __init__(self, nn_main, nn_loss, metric_fc,nb_predictors,isrnn=False, lr_main=0.001, batch_size_main=64, batch_size_loss=64, epochs=50, clr=None, coordinator=None, clr_coord=None):
        """
        Create The LossLearningPredictor object
        Inputs:
            -nn_main: Main neural network predictor (must be uncompiled)
            -nn_loss: Loss neural network
            -metric_fc: Metric used to train the nn_loss
            -isrnn: Boolean to know if nn_main is a Recurent Neural Network (RNN)
            -lr_main: Learning Rate for the nn_main
            -batch_size_main: batch size used by nn_main (>= batch_size_loss)
            -batch_size_loss: batch size used by nn_loss
            -epochs: Epochs for the training phase
            -clr : Cyclic Learning Rate object (if used)
        """
        super(LossLeaP, self).__init__()
        # Variables definition
        self.best_metric = 999
        self.nn_main = nn_main
        self.nn_loss = nn_loss
        self.metric_fc = metric_fc
        self.batch_size_main = batch_size_main
        self.batch_size_loss = batch_size_loss
        self.epochs = epochs
        self.clr = clr
        self.lr_main = lr_main
        self.nb_predictors = nb_predictors
        self.isrnn = isrnn
        self.opt = tf.keras.optimizers.Adam(
            learning_rate=self.lr_main)
        # Keep results during training
        self.history = {}


    def dual_train_multiples(self, datasets, metric_inputs=None, validation_data=None, noise=0, losstrain=True,path=None,path_loss=None):
        """
        Train neural networks models
        Inputs:
            -datasets: list of couples for each predictor [(X1,Y1),(X2,Y2),...]
            -metric: list of additionnal inputs for the loss neural network for each predictor 
            -validation_data: list of couples of Validation data [(X_val1, Y_val1),(X_val2, Y_val2),...]
            -noise: Noise added to the model
            -losstrain: Boolean to know if the loss neural network is already trained or not
            -path: path to save models
        """
        self.noise_initial = noise[0]
        self.tau=noise[1] 
        self.noise = noise[0]
        X, Y = datasets[0]
        Xsize = np.shape(X)[0]
        nbrun = Xsize * self.epochs / self.batch_size_main
        nb_steps_per_epoch = int(
            np.size(datasets[0][1]) / self.batch_size_main) + 1
        datasets_l = []
        for dataset in datasets:
            dataset = tf.data.Dataset.from_tensor_slices(dataset)
            dataset = dataset.batch(self.batch_size_main, drop_remainder=False)
            datasets_l.append(dataset)
        for epoch in range(self.epochs):
            start_time = time.time()
            iterators_l = []
            for dataset in datasets_l:
                iterators_l.append(dataset.__iter__())
            for step in range(nb_steps_per_epoch):
                input_full, met_value, loss_value = self.train_step(
                    iterators_l, metric_inputs)
                if self.clr != None:
                    self.clr.on_batch_end2(self.opt)
                if losstrain != False:
                    self.nn_loss.fit(input_full, met_value, epochs=1,
                                     batch_size=self.batch_size_loss, verbose=0)
                nbiter = ((epoch) * Xsize / self.batch_size_main) + (step + 1)
                self.noise = self.noise_initial*np.exp(-nbiter/self.tau)
                self.history.setdefault('train_loss', []).append(np.mean(loss_value))
                self.history.setdefault(
                    'train_metric', []).append(np.mean(met_value))
                # print training loss every step iterations
                if step % 100 == 1:
                    print(
                        "Training loss (for one batch) at step %d from epochs %d: %.4f  and   %.4f"
                        % (step, epoch, float(np.mean(met_value)), float(np.mean(loss_value)))
                    )
                    print("Seen so far: %s samples" %
                          ((step + 1) * self.batch_size_main))
                    print("Noise value is %f" %(self.noise))
            # Run a validation loop at the end of each epoch.
            if validation_data != None:
                met_value = self.validation_test(
                    validation_data, metric_inputs, noise)
                if path != None:
                    if met_value < self.best_metric:
                        self.nn_main.save(path)
                        self.nn_loss.save(path_loss)
                        self.best_metric = met_value
            print("Time taken: %.2fs" % (time.time() - start_time))

    def validation_test(self, validation_data, metric_inputs, noise):
        """
        Validation step
        Inputs:
            -validation_data: list of couples of Validation data [(X_val1, Y_val1),(X_val2, Y_val2),...]
            -metric_inputs: list of additionnal inputs for the loss neural network for each predictor 
            -noise: Noise added to the model
        """
        input_Xl = []
        full_y = None
        metric_l = []
        for xval, yval in validation_data:
            if self.isrnn == False:
                if noise != 0:
                    noiseadd = tf.zeros([np.shape(xval)[0], 1], tf.float64)
                    xval = tf.concat([xval, noiseadd], 1)
            else:
                if noise != 0:
                    noiseadd = tf.zeros(
                        [np.shape(xval)[0], 1, 1], tf.float64)
                    xval = tf.concat([xval, noiseadd], 1)
            input_Xl.append(xval)
            metric_l.append(yval)
        yvalpred = self.nn_main(inputs=input_Xl)
        yvalpred = tf.dtypes.cast(yvalpred, tf.float64)
        for ind in range(self.nb_predictors):
            met_y = tf.reshape(yvalpred[:, ind], [tf.shape(yvalpred)[0], 1])
            metric_l[ind] = tf.concat(
                [metric_l[ind], met_y], axis=1)
        if metric_inputs != None:
            met_value = np.mean(self.metric_fc(metric_l, metric_inputs))
        else:
            met_value = np.mean(self.metric_fc(metric_l))
        self.history.setdefault('val_metric', []).append(met_value)
        print("Validation metric: %.4f" % (float(met_value),))
        return met_value


    # Simple train of the main nn using tf function
    @tf.function
    def train_step(self, iterators_l, metric_inputs=None):
        metric_l = []
        samples_X = []
        input_ypred = None
        input_y = None
        noisefull = None
        input_Xl = []
        metric_l = []
        for ind, iterator in enumerate(iterators_l):
            X, y = iterator.get_next()
            if self.isrnn == False:
                noisenb = tf.random.normal([1], 0, self.noise, tf.float64)
                noiseadd = tf.zeros(
                    [tf.shape(X)[0], 1], tf.float64) + noisenb
            else:
                noisenb = tf.random.normal([1], 0, self.noise, tf.float64)
                noiseadd = tf.zeros(
                    [tf.shape(X)[0], 1, 1], tf.float64) + noisenb
            X = tf.concat([X, noiseadd], 1)
            if input_y is None:
                input_y = y
                noisefull = noiseadd
            else:
                input_y = tf.concat([input_y, y], axis=1)
                noisefull = tf.concat([noisefull, noiseadd], axis=1)
            input_Xl.append(X)
            metric_l.append(y)
        with tf.GradientTape(persistent=True) as tape:
            y_pred = self.nn_main(inputs=input_Xl)
            y_pred = tf.dtypes.cast(y_pred, tf.float64)
            if self.isrnn == True:
                noisefull = tf.reshape(noisefull, tf.shape(y_pred))
            y_pred = y_pred + noisefull

            for ind in range(self.nb_predictors):
                met_y = tf.reshape(y_pred[:, ind], [tf.shape(y_pred)[0], 1])
                metric_l[ind] = tf.concat(
                    [metric_l[ind], met_y], axis=1)
            if metric_inputs != None:
                met_value = self.metric_fc(metric_l, metric_inputs)
            else:
                met_value = self.metric_fc(metric_l)
            input_full = tf.concat([y_pred, input_y], axis=1)
            loss_value = self.nn_loss(input_full)
        gradients = tape.gradient(loss_value, self.nn_main.trainable_weights)
        self.opt.apply_gradients(
            zip(gradients, self.nn_main.trainable_weights))
        return input_full, met_value, loss_value

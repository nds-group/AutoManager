import utils.clr as clr
import keras
import utils.LLP as LLP
import matplotlib.pyplot as plt
import utils.networks as networks
import numpy as np
import pandas as pd
import argparse
import utils.plots as plots
import tensorflow as tf
from metric import metric_model
from default_config import args, DataPaths,ModelPaths,ResultsPath
import pickle

def add_nulnoise(datasets_l):
    datasetsBis_l = []
    for (X, Y) in datasets_l:
        noiseadd = tf.zeros(
            [np.shape(X)[0], 1, 1], tf.float64)
        datasetsBis_l.append((tf.concat([X, noiseadd], 1), Y))
    return datasetsBis_l


def train(cmd_args):
    # open a file, where you stored the pickled data
    datasetTrain_l_Path = open(cmd_args.TRAIN_DATASET_PATH, 'rb')
    datasets_l = pickle.load(datasetTrain_l_Path)
    datasetTest_l_Path = open(cmd_args.TEST_DATASET_PATH, 'rb')
    datasetsTest_l = pickle.load(datasetTest_l_Path)
    metric_inputs_Path = open(cmd_args.METRIC_INPUTS_PATH, 'rb')
    metric_inputs = pickle.load(metric_inputs_Path)
    metric_fc = metric_model



    X_train, Y_train = datasets_l[0]
    X_test, Y_test = datasetsTest_l[0]

    Multi_NN = networks.MultiPred_NN(
        cmd_args.nb_predictor, X_train.shape[1] + 1, networks.main_nn_model)
    datasetsTestBis_l = add_nulnoise(datasetsTest_l)
    # Train all predictors alone
    if cmd_args.pretrain == True:
        datasetsBis_l = add_nulnoise(datasets_l)
        Multi_NN.train_solo_all(datasetsBis_l, epochs=cmd_args.pretrain_epochs,
                                batch_size=cmd_args.pretrain_batch_size, lr=cmd_args.pretrain_learning_rate)
    # Combine the predictors
    model = Multi_NN.combine_preds('relu')
    # Create the Loss Neural Network
    nnloss = networks.nnloss_model(
        'relu', 'mean_squared_error', cmd_args.loss_learning_rate, cmd_args.loss_input_size)
    # tf.keras.utils.plot_model(model, cmd_args.RESULTS_DIR_PATH+'/NNdraw.png', show_shapes=True)

    # Create the CLR object
    clr_item = clr.CyclicLR(base_lr=cmd_args.CLR_base_lr, max_lr=cmd_args.CLR_max_lr, nb_cycles=cmd_args.CLR_nb_cycles,
                        full_size=np.shape(X_train)[0], epochs=cmd_args.epochs, batch_size=cmd_args.batch_size, mode=cmd_args.CLR_mode)
    # Create the LLP object
    LLPitem = LLP.LossLeaP(model, nnloss, metric_fc,cmd_args.nb_predictor ,isrnn=cmd_args.is_RNN,
                           batch_size_main=cmd_args.batch_size_loss, batch_size_loss=cmd_args.batch_size_loss, clr=clr_item, epochs=cmd_args.epochs)

    # Train the model
    tmax=cmd_args.epochs* X_train.shape[0]/cmd_args.batch_size
    tau= tmax/np.log(cmd_args.noise_initial/cmd_args.noise_final)
    LLPitem.dual_train_multiples(
        datasets_l, noise=[cmd_args.noise_initial,tau], validation_data=datasetsTest_l, metric_inputs=metric_inputs, path=cmd_args.PREDICTOR_PATH,path_loss=cmd_args.LOSS_MODEL_PATH)

    # Gather results
    LLPmet = LLPitem.history['val_metric']
    with open(cmd_args.RESULTS_DIR_PATH+'/validation_metric.npy', 'wb') as f:
        np.save(f, LLPitem.history['val_metric'])
    with open(cmd_args.RESULTS_DIR_PATH+'/train_metric.npy', 'wb') as f:
        np.save(f, LLPitem.history['train_metric'])


    ##################
    ###################
    #################
        ##################
    ###################
    #################
        ##################
    ###################
    #################
        ##################
    ###################
    #################



    # Plot the metric evolution
    # plots.metric_plot(LLPmet, cmd_args.RESULTS_DIR_PATH+'/metric_evolution.png')

    # noiseadd = tf.zeros([np.shape(X_test)[0], 1, 1], tf.float64)
    # X_preds = []
    # model = keras.models.load_model(path2)
    # for ind, (X, Y) in enumerate(datasetsTestBis_l):
    #     X_preds.append(X)
    # y_pred = model.predict(X_preds)
    # yPred_l = []
    # full_met_input = []
    # for ind in range(nb_serv):
    #     yPred_l.append(np.reshape(y_pred[:, ind], (np.shape(y_pred)[0], 1)))
    #     full_met_input.append(tf.concat((yTest_l[ind], yPred_l[-1]), axis=1))

    # metric_val = metric_fc(
    #     full_met_input, [maxi_l, full_max])
    # print("Final metric=")
    # print(np.mean(metric_val))
    # print(np.std(metric_val))
    # # Save the neural networks
    # path = "nn/" + folder + "/_nnloss" + str(M) + ".h5"
    # path2 = "nn/" + folder + "/_nnmain" + str(M) + ".h5"
    # nnloss.save(path)
    # model.save(path2)
    # # Plots the results of the model
    # PATH = "results/" + folder + "/traffic" + str(M) + ".html"
    # plots.plotly_traffic(yTest_l, yPred_l,
    #                      metric_val, PATH, show=True)
    # PATH = "results/" + folder + "/tottraffic" + str(M) + ".html"
    # plots.plotly_traffic_tot(yTest_l,  yPred_l, maxi_l,
    #                          full_max, PATH, show=True, scaled=scaled)

    ##################
    ###################
    #################    ##################
    ###################
    #################    ##################
    ###################
    #################    ##################
    ###################
    #################    ##################
    ###################
    #################    ##################
    ###################
    #################    ##################
    ###################
    #################
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description= "AutoManager model.",
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # General options
    general = parser.add_argument_group('General options')
    general.add_argument("-nb_predictor", "--nb_predictor", type=int, default=args.nb_predictor, help="Number of predictors.")
    general.add_argument("-is_RNN", "--is_RNN", type=bool, default=args.is_RNN, help="True if the predictors are RNNs.")
    general.add_argument("-learning_rate", "--learning_rate", type=float, default=args.learning_rate, help="Learning rate for the model. Useless if using CLR.")
    general.add_argument("-loss_input_size", "--loss_input_size", type=int, default=args.loss_input_size, help="Size of the input of the loss NN (Generally 2*nb_predictor + addition inputs).")
    general.add_argument("-loss_learning_rate","--loss_learning_rate", type=float, default=args.loss_learning_rate, help="Learning rate for the loss NN.")
    general.add_argument("-epochs", "--epochs", type=int, default=args.n_epochs, help="Number of epochs to train for.")
    general.add_argument("-batch_size", "--batch_size", type=int, default=args.batch_size, help="Batch size for training.")
    general.add_argument("-batch_size_loss", "--batch_size_loss", type=int, default=args.batch_size_loss, help="Batch size for training the loss NN.")


    #CLR options
    clrparser = parser.add_argument_group('CLR options')
    clrparser.add_argument("-CLR", "--CLR", type=bool, default=args.CLR_use, help="Use of CLR or not.")
    clrparser.add_argument("-CLR_base_lr", "--CLR_base_lr", type=float, default=args.CLR_base_lr, help="Base learning rate for CLR.")
    clrparser.add_argument("-CLR_max_lr", "--CLR_max_lr", type=float, default=args.CLR_max_lr, help="Max learning rate for CLR.")
    clrparser.add_argument("-CLR_nb_cycles", "--CLR_nb_cycles", type=int, default=args.CLR_nb_cycles, help="Number of cycles for CLR.")
    clrparser.add_argument("-CLR_mode", "--CLR_mode", type=str, default=args.CLR_mode, help="Full size for CLR.")

    # Pretraining options
    pretrain = parser.add_argument_group('Pretraining options options')
    pretrain.add_argument("-pretrain", "--pretrain", type=bool, default=args.pretrain, help="Pretrain the predictors or not.")
    pretrain.add_argument("-pretrain_epochs", "--pretrain_epochs", type=int, default=args.pretrain_epochs, help="Number of epochs to pretrain the predictors for.")
    pretrain.add_argument("-pretrain_batch_size", "--pretrain_batch_size", type=int, default=args.pretrain_batch_size, help="Batch size for pretraining the predictors.")
    pretrain.add_argument("-pretrain_learning_rate", "--pretrain_learning_rate", type=float, default=args.pretrain_learning_rate, help="Learning rate for pretraining the predictors.")

    # Noise config
    noise = parser.add_argument_group('Noise options')
    noise.add_argument("-noise_initial", "--noise_initial", type=bool, default=args.noise_initial, help="Initial noise value to add as input")
    noise.add_argument("-noise_final", "--noise_final", type=bool, default=args.noise_final, help="Final noise value to add as input")

    # Datasets Paths 
    data_args = parser.add_argument_group("Datasets Paths")
    data_args.add_argument("-TRAIN_DATASET_PATH", "--TRAIN_DATASET_PATH", type=str, default=DataPaths.TRAIN_DATASET_PATH, help="Path to the training dataset.")
    data_args.add_argument("-TEST_DATASET_PATH", "--TEST_DATASET_PATH", type=str, default=DataPaths.TEST_DATASET_PATH, help="Path to the test dataset.")
    data_args.add_argument("-METRIC_INPUTS_PATH", "--METRIC_INPUTS_PATH", type=str, default=DataPaths.METRIC_INPUTS_PATH, help="Path to the optional additional metric inputs.")

    #Model Paths
    model_args = parser.add_argument_group("Model Paths")
    model_args.add_argument("-PREDICTOR_PATH", "--PREDICTOR_PATH", type=str, default=ModelPaths.PREDICTOR_PATH, help="Path to the main nn model (predictor).")
    model_args.add_argument("-LOSS_MODEL_PATH", "--LOSS_MODEL_PATH", type=str, default=ModelPaths.LOSS_MODEL_PATH, help="Path to the loss nn model.")

    #Results Paths
    results_args = parser.add_argument_group("Results Paths")
    results_args.add_argument("-RESULTS_DIR_PATH", "--RESULTS_DIR_PATH", type=str, default=ResultsPath.RESULTS_DIR_PATH, help="Path to the results folder.")

    cmd_args = parser.parse_args()
    train(cmd_args)

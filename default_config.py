class DataPaths(object):
    #Train dataset and test dataset must be under the form [(DATASET1_X, DATASET1_Y), (DATASET2_X, DATASET2_Y), ...]
    #Metric input is a list of additionnal input for the metric function (must match the additional input of the metric function)
    TRAIN_DATASET_PATH = 'data/datasetsTrain_list.pickle'
    TEST_DATASET_PATH = 'data/datasetsTest_list.pickle'
    METRIC_INPUTS_PATH = 'data/metric_inputs.pickle'

class ModelPaths(object):
    PREDICTOR_PATH = 'NNs/predictor.h5'
    LOSS_MODEL_PATH = 'NNs/lossNN.h5'

class ResultsPath(object):
    RESULTS_DIR_PATH = 'results'

class args(object):
    #General config
    nb_predictor = 4
    n_epochs = 40
    batch_size = 16
    batch_size_loss = 16
    learning_rate = 0.0001
    is_RNN=True

    loss_learning_rate = 0.001
    loss_input_size= 2*nb_predictor #+ additional inputs

    #Cyclic Learning rate config
    CLR_use = True
    CLR_base_lr = 0.00001
    CLR_max_lr = 0.001
    CLR_nb_cycles = 3
    CLR_mode ='triangular'


    # Pretraining the predictors
    pretrain = False
    pretrain_epochs = 10
    pretrain_learning_rate = 0.001
    pretrain_batch_size = 64

    #Noise config
    noise_initial = 0.001
    noise_final = 0.00001


    
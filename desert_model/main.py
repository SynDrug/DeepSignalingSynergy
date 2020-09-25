import pickle
import numpy as np
import pandas as pd
from keras.models import load_model
from gen_matrix import GenMatrix
from parse_file import ParseFile
from load_data import LoadData
from keras_rand_nn import RandNN
from keras_rand_nn import RunRandNN
from analysis import Analyse


def build_rand_nn(matrixA, matrixB, num_gene, layer0, layer1, layer2, layer3):
    model = RandNN().keras_rand_nn(matrixA, matrixB, num_gene, layer0, layer1, layer2, layer3)
    return model

def run_rand_nn(model, dir_opt, RNA_seq_filename, matrixA, matrixB, input_num, epoch, batch_size, verbose):
    # AUTO UPDATE WEIGHT
    model, history, num_layer, path = RunRandNN(model, dir_opt, RNA_seq_filename).train(input_num, epoch, batch_size, verbose)
    return model, history, path

def continue_run_rand_nn(matrixA, matrixB, num_gene, num_pathway, layer1, layer2, layer3, path,
                dir_opt, RNA_seq_filename, input_num, epoch, batch_size, verbose):
    # RECONSTRCUT TO BE TRAINED MODEL
    model = RandNN().keras_rand_nn(matrixA, matrixB, num_gene, layer0, layer1, layer2, layer3)
    with open(path + '/layer_bias_list.txt', 'rb') as filebias:
        layer_bias_list = pickle.load(filebias)
    with open(path + '/layer_weight_list.txt', 'rb') as fileweight:
        layer_weight_list = pickle.load(fileweight)
    model.compile(loss='mean_squared_error',
                    optimizer='adam',
                    metrics=['mse', 'accuracy']) 
    xTmp, yTmp = LoadData(dir_opt, RNA_seq_filename).load_train(0, 1)
    model.fit(xTmp, yTmp, epochs = 1, validation_split = 1, verbose = 0)
    model_layer_list = []
    num_layer = len(model.layers)
    for i in range(num_layer):
        each_layer_list = [layer_weight_list[i], layer_bias_list[i]]
        model_layer_list.append(each_layer_list)
        model.layers[i].set_weights(each_layer_list)
    # AUTO UPDATE WEIGHT
    model, history, num_layer, path = RunRandNN(model, dir_opt, RNA_seq_filename).train(input_num, epoch, batch_size, verbose)
    return model, history, path

def auto_test_rand_nn(model, dir_opt, RNA_seq_filename, verbose, path):
    # GET MODEL IMMEDIATELY FROM TRAINED MODEL
    y_pred, score = RunRandNN(model, dir_opt, RNA_seq_filename).test(verbose, path)

def manual_test_rand_nn(matrixA, matrixB, num_gene, num_pathway, layer1, layer2, layer3, path, dir_opt, RNA_seq_filename):
    # RECONSTRCUT TEST MODEL
    model = RandNN().keras_rand_nn(matrixA, matrixB, num_gene, layer0, layer1, layer2, layer3)
    with open(path + '/layer_bias_list.txt', 'rb') as filebias:
        layer_bias_list = pickle.load(filebias)
    with open(path + '/layer_weight_list.txt', 'rb') as fileweight:
        layer_weight_list = pickle.load(fileweight)
    model.compile(loss='mean_squared_error',
                    optimizer='adam',
                    metrics=['mse', 'accuracy']) 
    xTmp, yTmp = LoadData(dir_opt, RNA_seq_filename).load_train(0, 1)
    model.fit(xTmp, yTmp, epochs = 1, validation_split = 1, verbose = 0)
    model_layer_list = []
    num_layer = len(model.layers)
    for i in range(num_layer):
        each_layer_list = [layer_weight_list[i], layer_bias_list[i]]
        model_layer_list.append(each_layer_list)
        model.layers[i].set_weights(each_layer_list)
    # PREDICT MODEL USING [xTe, yTe]
    verbose = 1
    y_pred, score = RunRandNN(model, dir_opt, RNA_seq_filename).test(verbose, path)


if __name__ == "__main__":
    # READ [NUM_FEATIRES/NUM_GENES/ NUM_PATHWAY] FROM DEEP_LEARNING_INPUT, RNA_SEQ, GENE_PATHWAY
    print('READING DIMS...')
    dir_opt = '/datainfo1'
    RNA_seq_filename = 'nci60-ccle_RNAseq_tpm1'
    pathway_filename = 'Selected_Kegg_Pathways1'
    zero_final_dl_input_df, input_num, num_feature, cellline_gene_df, num_gene, num_pathway = LoadData(dir_opt, RNA_seq_filename).pre_load_train()

    # BUILD NEURAL NETWORK
    print('BUILDING CUSTOMED NERUAL NETWORK...')
    layer0 = num_pathway
    layer1 = 2048
    layer2 = 256
    layer3 = 16
    matrixA = GenMatrix(dir_opt, RNA_seq_filename, pathway_filename).feature_gene_matrix(num_feature, num_gene)
    matrixB = GenMatrix(dir_opt, RNA_seq_filename, pathway_filename).gene_pathway_matrix(num_pathway)
    print(matrixA.shape)
    print(matrixB.shape)

    # # RUN DEEP NERUAL NETWORK
    print('RUNING DEEP NERUAL NETWORK...')
    model = build_rand_nn(matrixA, matrixB, num_gene, num_pathway, layer1, layer2, layer3)
    epoch = 20
    batch_size = 256
    verbose = 0
    model, history, path = run_rand_nn(model, dir_opt, RNA_seq_filename, matrixA, matrixB, input_num, epoch, batch_size, verbose)

    # # AUTO TEST DEEP NERUAL NETWORK MODEL
    print('TESTING DEEP NERUAL NETWORK...')
    auto_test_rand_nn(model, dir_opt, RNA_seq_filename, verbose, path)

    # MANUAL TEST DEEP NERUAL NETWORK MODEL
    print('TESTING DEEP NERUAL NETWORK...')
    path = '.' + dir_opt + '/result/epoch_20'
    manual_test_rand_nn(matrixA, matrixB, num_gene, num_pathway, layer1, layer2, layer3, path, dir_opt, RNA_seq_filename)

    # CONTINUE RUN TRAINED DEEP NEURAL NETWORK
    print('RUNING DEEP NERUAL NETWORK...')
    path = '.' + dir_opt + '/result/epoch_20'
    epoch = 20
    batch_size = 256
    verbose = 0
    model, history, path = continue_run_rand_nn(matrixA, matrixB, num_gene, num_pathway, layer1, layer2, layer3, path,
                dir_opt, RNA_seq_filename, input_num, epoch, batch_size, verbose)
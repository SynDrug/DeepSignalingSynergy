import os
import pickle
import innvestigate
import numpy as np
import pandas as pd
import tensorflow as tf

from keras import Model
from keras.models import load_model
from gen_matrix import GenMatrix
from parse_file import ParseFile
from load_data import LoadData
from analysis import Analyse
from keras_rand_nn import RandNN, RunRandNN

# BUILD DECOMPOSED MODEL
def build_rand_nn(matrixA, matrixB, num_gene, num_pathway, layer1, layer2, layer3):
    input_model, gene_model, pathway_model, model = RandNN().keras_rand_nn(matrixA, matrixB, num_gene, num_pathway, layer1, layer2, layer3)
    return input_model, gene_model, pathway_model, model

# RUN MODEL (AUTO UPDATE WEIGHT)
def run_rand_nn(model, dir_opt, RNA_seq_filename, matrixA, matrixB, input_num, epoch, batch_size, verbose):
    model, history, path = RunRandNN(model, dir_opt, RNA_seq_filename).train(input_num, epoch, batch_size, verbose)
    return model, history, path

# GET MODEL IMMEDIATELY FROM TRAINED MODEL
def auto_test_rand_nn(model, dir_opt, RNA_seq_filename, verbose, path):
    y_pred, score = RunRandNN(model, dir_opt, RNA_seq_filename).test(verbose, path)

def manual_test_rand_nn(matrixA, matrixB, num_gene, num_pathway, layer1, layer2, layer3, path, dir_opt, RNA_seq_filename):
    # MANUAL REBUILD THE MODEL
    input_model, gene_model, pathway_model, model = build_rand_nn(matrixA, matrixB, num_gene, num_pathway, layer1, layer2, layer3)
    with open(path + '/layer_list.txt', 'rb') as filelayer:
        layer_list = pickle.load(filelayer)
    model.compile(loss='mean_squared_error',
                    optimizer='adam',
                    metrics=['mse', 'accuracy']) 
    xTmp, yTmp = LoadData(dir_opt, RNA_seq_filename).load_train(0, 1)
    model.fit(xTmp, yTmp, epochs = 1, validation_split = 1, verbose = 0)
    num_layer = len(model.layers)
    for i in range(num_layer):
        model.get_layer(index = i).set_weights(layer_list[i])
    # PREDICT MODEL USING [xTe, yTe]
    verbose = 1
    y_pred, score = RunRandNN(model, dir_opt, RNA_seq_filename).test(verbose, path)

# CONTINUE RUN MODEL
def continue_run_rand_nn(matrixA, matrixB, num_gene, num_pathway, layer1, layer2, layer3, path,
                dir_opt, RNA_seq_filename, input_num, epoch, batch_size, verbose):
    # REBUILD DECOMPOSED MODEL FROM SAVED MODEL
    input_model, gene_model, pathway_model, model = build_rand_nn(matrixA, matrixB, num_gene, num_pathway, layer1, layer2, layer3)
    with open(path + '/layer_list.txt', 'rb') as filelayer:
        layer_list = pickle.load(filelayer)
    model.compile(loss='mean_squared_error',
                    optimizer='adam',
                    metrics=['mse', 'accuracy']) 
    xTmp, yTmp = LoadData(dir_opt, RNA_seq_filename).load_train(0, 1)
    model.fit(xTmp, yTmp, epochs = 1, validation_split = 1, verbose = 0)
    num_layer = len(model.layers)
    for i in range(num_layer):
        model.get_layer(index = i).set_weights(layer_list[i])
    # RUN MODEL (AUTO UPDATE WEIGHT)    
    model, history, path = run_rand_nn(model, dir_opt, RNA_seq_filename, matrixA, matrixB, input_num, epoch, batch_size, verbose)
    return model, history, path


if __name__ == "__main__":
    # READ [NUM_FEATIRES/NUM_GENES/ NUM_PATHWAY] FROM DEEP_LEARNING_INPUT, RNA_SEQ, GENE_PATHWAY
    print('READING DIMS...')
    dir_opt = '/datainfo2'
    RNA_seq_filename = 'nci60-ccle_RNAseq_tpm2'
    pathway_filename = 'Selected_Kegg_Pathways2'
    train_input_df, input_num, num_feature, cellline_gene_df, cpnum_df, num_gene, num_pathway = LoadData(dir_opt, RNA_seq_filename).pre_load_train()

    # BUILD NEURAL NETWORK
    print('BUILDING CUSTOMED NERUAL NETWORK...')
    layer0 = num_pathway
    layer1 = 256
    layer2 = 128
    layer3 = 32
    matrixA = GenMatrix(dir_opt, RNA_seq_filename, pathway_filename).feature_gene_matrix(num_feature, num_gene)
    matrixB = GenMatrix(dir_opt, RNA_seq_filename, pathway_filename).gene_pathway_matrix(num_pathway)
    print('-----MATRIX A - FEATURES OF GENE SHAPE-----')
    print(matrixA.shape)
    print('-----MATRIX B - GENE PATHWAY SHAPE-----')
    print(matrixB.shape)

    # RUN DEEP NERUAL NETWORKs
    print('RUNING DEEP NERUAL NETWORK...')
    epoch = 30
    batch_size = 256
    verbose = 0
    input_model, gene_model, pathway_model, model = build_rand_nn(matrixA, matrixB,
                num_gene, num_pathway, layer1, layer2, layer3)
    model, history, path = run_rand_nn(model, dir_opt, RNA_seq_filename, matrixA, matrixB, input_num, epoch, batch_size, verbose)

    # TEST DEEP NERUAL NETWORK MODEL
    # AUTO TEST NETWORK
    print('TESTING DEEP NERUAL NETWORK...')
    auto_test_rand_nn(model, dir_opt, RNA_seq_filename, verbose, path)


    # # MANUAL TEST NETWORK
    # path = '.' + dir_opt + '/result/epoch_60'
    # manual_test_rand_nn(matrixA, matrixB, num_gene, num_pathway, layer1, layer2, layer3, path, dir_opt, RNA_seq_filename)
    
    # # CONTINUE RUN TRAINED DEEP NEURAL NETWORK
    # print('CONTINUE RUNING DEEP NERUAL NETWORK...')
    # path = '.' + dir_opt + '/result/epoch_5'
    # epoch = 5
    # batch_size = 256
    # verbose = 0
    # model, history, path = continue_run_rand_nn(matrixA, matrixB, num_gene, num_pathway, layer1, layer2, layer3, path,
    #             dir_opt, RNA_seq_filename, input_num, epoch, batch_size, verbose)
    # auto_test_rand_nn(model, dir_opt, RNA_seq_filename, verbose, path)

    # # MANUAL TEST NETWORK
    # path = '.' + dir_opt + '/result/epoch_30'
    # manual_test_rand_nn(matrixA, matrixB, num_gene, num_pathway, layer1, layer2, layer3, path, dir_opt, RNA_seq_filename)
    


    
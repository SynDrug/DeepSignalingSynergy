import os
import shutil
import pickle
import simplejson
import numpy as np
import pandas as pd
import keras.backend as K
import matplotlib.pyplot as plt
from load_data import LoadData
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense

# BUILD A NOT FULLY CONNECTED NN
class CustomConnected(Dense):
    def __init__(self, units, connections, **kwargs):
        # THIS IS MATRIX_A
        self.connections = connections      
        # INITALIZE THE ORIGINAL DENSE WITH ALL THE USUAL ARGUMENTS   
        super(CustomConnected, self).__init__(units, **kwargs)  

    def call(self, inputs):
        output = K.dot(inputs, self.kernel * self.connections)
        if self.use_bias:
            output = K.bias_add(output, self.bias)
        if self.activation is not None:
            output = self.activation(output)
        return output


class RandNN():
    def __init__(self):
        pass

    # NEW RAND NETWORK
    def keras_rand_nn(self, matrixA, matrixB, num_gene, layer0, layer1, layer2, layer3):
        # INITIALIZE THE CONSTRUCTOR
        model = Sequential()
        # ADD AN INPUT LAYER 
        model.add(CustomConnected(num_gene, matrixA))
        # ADD AN INPUT LAYER 
        model.add(CustomConnected(layer0, matrixB))
        # ADD ONE HIDDEN LAYER
        model.add(Dense(layer1, activation='relu'))
        # ADD ANOTHER HIDDEN LAYER 
        model.add(Dense(layer2, activation='relu'))
        # ADD ANOTHER HIDDEN LAYER 
        model.add(Dense(layer3, activation='relu'))
        # ADD FINAL OUTPUT LAYER 
        # model.add(Dense(1, activation='sigmoid'))
        model.add(Dense(1, activation='linear'))
        return model


class RunRandNN():
    def __init__(self, model, dir_opt, RNA_seq_filename):
        self.model = model
        self.dir_opt = dir_opt
        self.RNA_seq_filename = RNA_seq_filename

    def train(self, input_num, epoch, batch_size, verbose):
        model = self.model
        dir_opt = self.dir_opt
        RNA_seq_filename = self.RNA_seq_filename
        model.compile(loss='mean_squared_error',
                    optimizer='adam',
                    metrics=['mse', 'accuracy']) 
        # CLEAN RESULT PREVIOUS EPOCH_I_PRED FILES
        folder_name = 'epoch_' + str(epoch)
        path = '.' + dir_opt + '/result/%s' % (folder_name)
        unit = 1
        while os.path.exists(path):
            path = '.' + dir_opt + '/result/%s_%d' % (folder_name, unit)
            unit += 1
        os.mkdir(path)
        # TRAIN MODEL IN EPOCH ITERATIONS
        epoch_mse_list = []
        epoch_pearson_list = []
        for i in range(epoch):
            print('--------------EPOCH: ' + str(i) + ' --------------') 
            epoch_train_pred = np.zeros((1, 1))
            upper_index = 0
            batch_mse_list = []
            for index in range(0, input_num, batch_size):
                if (index + batch_size) < input_num:
                    upper_index = index + batch_size
                else:
                    upper_index = input_num
                xTr_batch, yTr_batch = LoadData(dir_opt, RNA_seq_filename).load_train(index, upper_index)
                history = model.fit(xTr_batch, yTr_batch, epochs = 1, validation_split = 1, verbose = verbose)
                # PRESERVE MSE FOR EVERY BATCH
                print(history.history['mse'], history.history['accuracy'])
                batch_mse_list.append(history.history['mse'])
                # PRESERVE PREDICTION OF TRAINING MODEL IN EVERY BATCH
                train_batch_pred = np.array(model.predict(xTr_batch))
                epoch_train_pred = np.vstack((epoch_train_pred, train_batch_pred))
            # PRESERVE MSE FOR EVERY EPOCH
            print(np.mean(batch_mse_list)) # mse shuold use weight average, here mostly same, just ignore
            epoch_mse_list.append(np.mean(batch_mse_list))
            # SAVE RESULT FOR EVERY EPOCH PREDICTION
            epoch_train_pred = np.delete(epoch_train_pred, 0, axis = 0)
            np.save(path + '/epoch_' + str(i) + '_pred.npy', epoch_train_pred)
            # PRESERVE PEARSON CORR FOR EVERY EPOCH
            tmp_training_input_df = pd.read_csv('.' + dir_opt + '/filtered_data/TrainingInput.txt', delimiter = ',')
            final_row, final_col = tmp_training_input_df.shape
            epoch_train_pred_lists = list(epoch_train_pred)
            epoch_train_pred_list = [item for elem in epoch_train_pred_lists for item in elem]
            tmp_training_input_df.insert(final_col, 'Pred Score', epoch_train_pred_list, True)
            epoch_pearson = tmp_training_input_df.corr(method = 'pearson')
            epoch_pearson_list.append(epoch_pearson)
            print(epoch_pearson)
        # TRAINNING OUTPUT PRED
        final_train_input_df = pd.read_csv('.' + dir_opt + '/filtered_data/TrainingInput.txt', delimiter = ',')
        final_row, final_col = final_train_input_df.shape
        epoch_train_pred_lists = list(epoch_train_pred)
        epoch_train_pred_list = [item for elem in epoch_train_pred_lists for item in elem]
        final_train_input_df.insert(final_col, 'Pred Score', epoch_train_pred_list, True)
        final_train_input_df.to_csv(path + '/PredTrainingInput.txt', index = False, header = True)
        print(epoch_mse_list)
        print(epoch_pearson_list)
        # PRESERVE TRAINING MODEL RESULTS [MSE, PEARSON]
        fp = open(path + '/epoch_result_list.txt', 'w')
        simplejson.dump(str(epoch_mse_list), fp)
        simplejson.dump(str(epoch_pearson_list), fp)
        fp.close()
        # PRESERVE TRAINED MODEL EACH LAYER WEIGHT PARAMETERS
        layer_bias_list = []
        layer_weight_list = []
        num_layer = len(model.layers)
        for i in num_layer:
            layer_weight_list.append(model.get_layer(i).get_weights()[0])
            layer_bias_list.append(model.get_layer(i).get_weights()[1])
        with open(path + '/layer_bias_list.txt', 'wb') as filebias:
            pickle.dump(layer_bias_list, filebias)
        with open(path + '/layer_weight_list.txt', 'wb') as fileweight:
            pickle.dump(layer_weight_list, fileweight)
        return model, history, num_layer, path

    def test(self, verbose, path):
        print('TESTING DEEP NERUAL NETWORK...')
        model = self.model
        dir_opt = self.dir_opt
        RNA_seq_filename = self.RNA_seq_filename
        xTe, yTe = LoadData(dir_opt, RNA_seq_filename).load_test()
        # TEST OUTPUT PRED
        y_pred = model.predict(xTe)
        y_pred_list = [item for elem in y_pred for item in elem]
        score = model.evaluate(xTe, yTe, verbose = verbose)
        final_test_input_df = pd.read_csv('.' + dir_opt + '/filtered_data/TestInput.txt', delimiter = ',')
        final_row, final_col = final_test_input_df.shape
        final_test_input_df.insert(final_col, 'Pred Score', y_pred_list, True)
        final_test_input_df.to_csv(path + '/PredTestInput.txt', index = False, header = True)
        # ANALYSE PEARSON CORR
        test_pearson = final_test_input_df.corr(method = 'pearson')
        print(test_pearson)
        return y_pred, score


class RandNNPlot():
    def __init__(self, history):
        self.history = history

    def plot(self, epoch, show_plot):
        history = self.history
        # plot training & validation accuracy values
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('Model accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc = 'upper left')
        if os.path.isdir("./img") == False: 
                os.mkdir("./img")
        plt.savefig("./img/RandNN-Accuracy" + str(epoch) + ".png")
        if(show_plot == True):
            plt.show()
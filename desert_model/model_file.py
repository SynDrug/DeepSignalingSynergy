import keras.backend as K
from keras.layers import Input,Dense,BatchNormalization,Activation,Conv1D,\
    Flatten,Reshape,Dropout,concatenate,MaxPooling1D,add
from keras import Model

# build a not fully connected nn
class CustomConnected(Dense):
    def __init__(self, units, connections, **kwargs):
        # this is matrix A
        self.connections = connections      
        # initalize the original Dense with all the usual arguments   
        super(CustomConnected, self).__init__(units, **kwargs)  

    def call(self, inputs):
        output = K.dot(inputs, self.kernel * self.connections)
        if self.use_bias:
            output = K.bias_add(output, self.bias)
        if self.activation is not None:
            output = self.activation(output)
        return output

def cancer_pathway_dense(matrixA, matrixB, num_gene,num_feature, num_pathway,num_cli,layer_list,final_layer=16):
    #Gene information
    input_shape1=(num_gene*num_feature,)
    gene_input=Input(input_shape1)
    gene_mix=CustomConnected(num_gene,matrixA)(gene_input)
    inputModel=Model(gene_input,gene_mix)
    gene_info=inputModel(gene_input)
    gene_info_shape=K.int_shape(gene_info)[1:]
    gene_info_input=Input(gene_info_shape)
    pathway=CustomConnected(num_pathway,matrixB)(gene_info_input)
    pathway=Reshape((num_pathway,-1))(pathway)
    geneModel=Model(gene_info_input,pathway)
    pathway_info=geneModel(gene_info)
    #pathway information
    predict_shape=(num_pathway,)
    predict_input=Input(predict_shape)
    for i in range(len(layer_list)):
        if(i==0):
            dense=Dense(layer_list[i],activation="relu")(predict_input)
        else:
            dense = Dense(layer_list[i], activation="relu")(dense)
            dense=Dropout(0.2)(dense)
    pathwayModel=Model(predict_input,dense)
    genomic_info=pathwayModel(pathway_info)
    input_shape2=(num_cli,)
    cli_input=Input(input_shape2)
    genomic_info_input=Input((layer_list[-1],))
    concat=concatenate([cli_input,genomic_info_input])
    cli_dense=Dense(final_layer,activation="relu")(concat)
    output=Dense(1)(cli_dense)

    outputModel=Model([cli_input,genomic_info_input],output)
    output=outputModel([cli_input,genomic_info])
    model=Model([gene_input,cli_input],output)
    return model,inputModel,geneModel,pathwayModel,outputModel

def cancer_pathway_denBatch(matrixA, matrixB, num_gene,num_feature, num_pathway,num_cli,layer_list,final_layer=16):
    #Gene information
    input_shape1=(num_gene*num_feature,)
    input1=Input(input_shape1)
    gene_mix=CustomConnected(num_gene,matrixA)(input1)
    geneModel=Model(input1,gene_mix)
    gene_info=geneModel(input1)
    gene_info_shape=K.int_shape(gene_info)[1:]
    gene_info_input=Input(gene_info_shape)
    pathway=CustomConnected(num_pathway,matrixB)(gene_info_input)
    pathway=Reshape((num_pathway,-1))(pathway)
    costumizeModel=Model(gene_info_input,pathway)
    costumized=costumizeModel(gene_info)
    #pathway information
    predict_shape=(num_pathway,)
    predict_input=Input(predict_shape)
    for i in range(len(layer_list)):
        if(i==0):
            dense=Dense(layer_list[i],activation="relu")(predict_input)
        else:
            dense = Dense(layer_list[i], activation="relu")(dense)
            dense=Dropout(0.2)(dense)
        dense=BatchNormalization()(dense)
        dense=Activation("relu")(dense)
    #dense=Flatten()(conv)
    pathwayModel=Model(predict_input,dense)
    pathway_info=pathwayModel(costumized)
    input_shape2=(num_cli,)
    cli_input=Input(input_shape2)
    pathway_info_input=Input((layer_list[-1],))
    concat=concatenate([cli_input,pathway_info_input])
    cli_dense=Dense(final_layer,activation="relu")(concat)
    output=Dense(1)(cli_dense)

    predictModel=Model([cli_input,pathway_info_input],output)
    #costumized=Reshape((17,-1))(costumized)
    predicted=predictModel([cli_input,pathway_info])
    model=Model([input1,cli_input],predicted)
    return model,geneModel,costumizeModel,pathwayModel,predictModel


def cancer_pathway_convBatch(matrixA, matrixB, num_gene,num_feature, num_pathway,num_cli,kernel_list,filter_list,final_layer=16):
    #Gene information
    input_shape1=(num_gene*num_feature,)
    input1=Input(input_shape1)
    gene_mix=CustomConnected(num_gene,matrixA)(input1)
    geneModel=Model(input1,gene_mix)
    gene_info=geneModel(input1)
    gene_info_shape=K.int_shape(gene_info)[1:]
    gene_info_input=Input(gene_info_shape)
    pathway=CustomConnected(num_pathway,matrixB)(gene_info_input)
    pathway=Reshape((num_pathway,-1))(pathway)
    costumizeModel=Model(gene_info_input,pathway)
    costumized=costumizeModel(gene_info)
    #pathway information
    predict_shape=(num_pathway,1,)
    predict_input=Input(predict_shape)
    for i in range(len(kernel_list)):
        if(i==0):
            conv=Conv1D(filter_list[i],kernel_list[i],padding="same",activation="relu")(predict_input)
        else:
            conv = Conv1D(filter_list[i], kernel_list[i], padding="same",activation="relu")(conv)
        conv=MaxPooling1D()(conv)
        conv=BatchNormalization()(conv)
        conv=Activation("relu")(conv)
        conv=Dropout(0.2)(conv)
    dense=Flatten()(conv)
    pathwayModel=Model(predict_input,dense)
    pathway_info=pathwayModel(costumized)
    input_shape2=(num_cli,)
    cli_input=Input(input_shape2)
    pathway_info_input=Input(K.int_shape(dense)[1:])
    concat=concatenate([cli_input,pathway_info_input])
    cli_dense=Dense(final_layer,activation="relu")(concat)
    output=Dense(1)(cli_dense)

    predictModel=Model([cli_input,pathway_info_input],output)
    #costumized=Reshape((17,-1))(costumized)
    predicted=predictModel([cli_input,pathway_info])
    model=Model([input1,cli_input],predicted)
    return model,geneModel,costumizeModel,pathwayModel,predictModel


def inception(filter_size):

    def f(input_layer):
        inception1=Conv1D(filter_size,1,padding="same",activation="relu")(input_layer)

        inception2=Conv1D(filter_size,1,padding="same",activation="relu")(input_layer)
        inception2=Conv1D(filter_size,3,padding="same",activation="relu")(inception2)

        inception3=Conv1D(filter_size,1,padding="same",activation="relu")(input_layer)
        inception3=Conv1D(filter_size,3,padding="same",activation="relu")(inception3)
        inception3=Conv1D(filter_size,3,padding="same",activation="relu")(inception3)

        inception4=MaxPooling1D(padding="same",strides=1)(input_layer)
        inception4=Conv1D(filter_size,3,padding="same",activation="relu")(inception4)
        concat=concatenate([inception1,inception2,inception3,inception4],axis=-1)
        concat=Conv1D(filter_size,1,padding="same",activation="relu")(concat)

        output_layer=add([input_layer,concat])
        return output_layer
    return f

def cancer_pathway_inceptBatch(matrixA, matrixB, num_gene,num_feature, num_pathway,num_cli,filter_list,den1=256,den2=128,den3=64):
    #Gene information
    input_shape1=(num_gene*num_feature,)
    gene_input=Input(input_shape1)
    gene_mix=CustomConnected(num_gene,matrixA)(gene_input)
    inputModel=Model(gene_input,gene_mix)

    gene_info=inputModel(gene_input)
    gene_info_shape=K.int_shape(gene_info)[1:]
    gene_info_input=Input(gene_info_shape)
    pathway=CustomConnected(num_pathway,matrixB)(gene_info_input)
    geneModel=Model(gene_info_input,pathway)

    pathway_info=geneModel(gene_info)
    predict_shape=(num_pathway,)
    predict_input=Input(predict_shape)
    pathway=Reshape((num_pathway,-1))(predict_input)
    for i in range(len(filter_list)):
        if i==0:
            conv=Conv1D(filter_list[i],3,padding="same",activation="relu")(pathway)
        else:
            conv=Conv1D(filter_list[i],3,padding="same",activation="relu")(conv)
        conv=inception(filter_list[i])(conv)
        conv=BatchNormalization()(conv)
        conv=Activation("relu")(conv)
        conv=Dropout(0.2)(conv)

    dense=Flatten()(conv)
    dense=Dense(den1)(dense)
    input_shape2=(num_cli,)
    cli_input=Input(input_shape2)
    genomic_info_input=Input(K.int_shape(dense)[1:])
    concat=concatenate([cli_input,dense])
    #cli_conv=Reshape((K.int_shape(concat)[1],-1))(concat)
    #cli_conv=Conv1D(den2,3,padding="same",activation="relu")(cli_conv)
    #cli_conv=inception(den2)(cli_conv)
    #cli_conv = BatchNormalization()(cli_conv)
    #cli_conv = Activation("relu")(cli_conv)
    cli_conv=Dense(den2)(concat)
    cli_dense=Dropout(0.2)(cli_conv)
    #cli_dense=Flatten()(cli_conv)
    cli_dense=Dense(den3,activation="relu")(cli_dense)
    output=Dense(1)(cli_dense)

    pathwayModel=Model([predict_input,cli_input],output)
    output=pathwayModel([pathway_info,cli_input])
    model=Model([gene_input,cli_input],output)
    return model,inputModel,geneModel,pathwayModel



def cancer_pathway_inceptBatch_noCli(matrixA, matrixB, num_gene,num_feature, num_pathway,filter_list,den1=256,den2=128,den3=64):
    #Gene information
    input_shape1=(num_gene*num_feature,)
    gene_input=Input(input_shape1)
    gene_mix=CustomConnected(num_gene,matrixA)(gene_input)
    inputModel=Model(gene_input,gene_mix)

    gene_info=inputModel(gene_input)
    gene_info_shape=K.int_shape(gene_info)[1:]
    gene_info_input=Input(gene_info_shape)
    pathway=CustomConnected(num_pathway,matrixB)(gene_info_input)
    geneModel=Model(gene_info_input,pathway)

    pathway_info=geneModel(gene_info)
    predict_shape=(num_pathway,)
    predict_input=Input(predict_shape)
    pathway=Reshape((num_pathway,-1))(predict_input)
    for i in range(len(filter_list)):
        if i==0:
            conv=Conv1D(filter_list[i],3,padding="same",activation="relu")(pathway)
        else:
            conv=Conv1D(filter_list[i],3,padding="same",activation="relu")(conv)
        conv=inception(filter_list[i])(conv)
        conv=BatchNormalization()(conv)
        conv=Activation("relu")(conv)
        conv=Dropout(0.2)(conv)

    dense=Flatten()(conv)
    dense=Dense(den1)(dense)
    #cli_conv=Reshape((K.int_shape(concat)[1],-1))(concat)
    #cli_conv=Conv1D(den2,3,padding="same",activation="relu")(cli_conv)
    #cli_conv=inception(den2)(cli_conv)
    #cli_conv = BatchNormalization()(cli_conv)
    #cli_conv = Activation("relu")(cli_conv)
    cli_conv=Dense(den2)(dense)
    cli_dense=Dropout(0.2)(cli_conv)
    #cli_dense=Flatten()(cli_conv)
    cli_dense=Dense(den3,activation="relu")(cli_dense)
    output=Dense(1)(cli_dense)

    pathwayModel=Model(predict_input,output)
    output=pathwayModel(pathway_info)
    model=Model(gene_input,output)
    return model,inputModel,geneModel,pathwayModel
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from model_file import cancer_pathway_inceptBatch,cancer_pathway_inceptBatch_noCli
import innvestigate
from tensorflow import set_random_seed

set_random_seed(2)
np.random.seed(234)

pathway2=pd.read_table("data/Selected_Kegg_Pathways2.txt",sep="\t")
gene_cnv2=pd.read_table("data/BRCA-gene_cnv_2.txt",sep="\t")
gene_exp2=pd.read_table("data/BRCA-gene_exp_2.txt",sep="\t")
cli_info2=pd.read_table("data/BRCA-cli_info_2.txt",sep="\t")

geneSymbol=gene_cnv2["rownames(x2)"]
Y=np.array(cli_info2["x_survival_time"])
gene_exp2.drop(["rownames(x1)"],axis=1,inplace=True)
gene_cnv2.drop(["rownames(x2)"],axis=1,inplace=True)
pathway2.drop(["AllGenes"],axis=1,inplace=True)
gene_exp_arr=np.array(gene_exp2)
gene_cnv_arr=np.array(gene_cnv2)

X=np.vstack([gene_exp_arr,gene_cnv_arr])
X=X.T



clinical_variables=cli_info2[["x_age","x_sex","x_vital","x_stage"]]
clinical_dummy=pd.get_dummies(clinical_variables,columns=["x_sex","x_vital","x_stage"])
clinical_variables=np.array(clinical_dummy)



train_x,test_x,train_y,test_y,train_cli,test_cli=train_test_split(X,Y,clinical_variables,test_size=0.2,shuffle=False,random_state=234)
train_y_min=np.min(train_y)
train_y_max=np.max(train_y)
train_y_normal=(train_y-train_y_min)/(train_y_max-train_y_min)
test_y_normal=(test_y-train_y_min)/(train_y_max-train_y_min)
'''

train_y_mean=np.mean(train_y)
train_y_std=np.std(train_y)
train_y_normal=(train_y-train_y_mean)/train_y_std
test_y_normal=(test_y-train_y_mean)/train_y_std
'''

train_age_max=np.max(train_cli[:,0])
train_age_min=np.min(train_cli[:,0])
train_age_normal=(train_cli[:,0]-train_age_min)/(train_age_max-train_age_min)
test_age_normal=(test_cli[:,0]-train_age_min)/(train_age_max-train_age_min)
train_cli[:,0]=train_age_normal
test_cli[:,0]=test_age_normal

train_gene_quan=np.quantile(train_x[:,0:1967],[0.01,0.99],axis=0)
train_gene_min=train_gene_quan[0]
train_gene_max=train_gene_quan[1]
train_gene_range=train_gene_max-train_gene_min
drop_index=train_gene_range>0

train_gene_range=train_gene_range[drop_index]
train_gene_min=train_gene_min[drop_index]
geneSymbol=geneSymbol[drop_index]
geneSymbol.index=[i for i in range(1965)]
train_x=train_x[:,np.hstack([drop_index,drop_index])]
test_x=test_x[:,np.hstack([drop_index,drop_index])]

train_gene_normal=(train_x[:,0:1965]-train_gene_min)/train_gene_range*2-1
train_gene_normal[train_gene_normal>1]=1
train_gene_normal[train_gene_normal<-1]=-1
train_x=np.hstack([train_gene_normal,train_x[:,1965:3930]])

test_gene_normal=(test_x[:,0:1965]-train_gene_min)/train_gene_range*2-1
test_gene_normal[test_gene_normal>1]=1
test_gene_normal[test_gene_normal<-1]=-1
test_x=np.hstack([test_gene_normal,test_x[:,1965:3930]])


pathway_matrix=[]
for i in range(pathway2.shape[1]):
    p_index=[0 for k in range(1965)]
    p=pathway2.iloc[:,i]
    for j in range(len(p)):
        index=geneSymbol.loc[geneSymbol == p[j]].index.values
        if(len(index)!=0):
            p_index[index[0]]=1
    pathway_matrix.append(p_index)

pathway_matrix=np.array(pathway_matrix)
pathway_matrix=pathway_matrix.T


gene_matrix=[]
for i in range(1965):
    gene_index=[0 for j in range(3930)]
    gene_index[i]=1
    gene_index[i+1965]=1
    gene_matrix.append(gene_index)

gene_matrix=np.array(gene_matrix)
gene_matrix=gene_matrix.T


model,inputModel,geneModel,pathwayModel=cancer_pathway_inceptBatch(gene_matrix,pathway_matrix,1965,2,46,18,[16,8,4],den3=32)


model.compile(loss='mean_squared_error', # binary_crossentropy
                    optimizer='adadelta',
                    metrics=['mse'])

model.fit([train_x,train_cli],train_y_normal,batch_size=32,epochs=50,shuffle=False,validation_data=
           ([test_x,test_cli],test_y_normal))



predict_y=model.predict([test_x,test_cli])
predict_y=predict_y.reshape((97))
np.mean(np.square(predict_y-test_y_normal))

'''
# Creating an analyzer and set neuron_selection_mode to "index"
input_analyzer = innvestigate.create_analyzer("smoothgrad",inputModel,noise_scale=(np.max(new_X)-np.min(new_X))*0.1)
analysis=input_analyzer.analyze(new_X)
plot.imshow(new_X, cmap='seismic', interpolation='nearest',aspect="auto")
plot.ylabel("Sample Index")
plot.xlabel("input")
plot.show()
mean_analysis=np.mean(analysis,axis=0)
np.savetxt("BRCA_result/input_importance.txt",analysis)

'''
new_X=np.vstack([train_x,test_x])
import matplotlib.pyplot as plot
from matplotlib.backends.backend_pdf import PdfPages

predict_y_normal=model.predict(np.vstack([train_x,test_x]))
predict_y_normal=model.predict([np.vstack([train_x,test_x]),np.vstack([train_cli,test_cli])])
predict_y=predict_y_normal*(train_y_max-train_y_min)+train_y_min
pdf = PdfPages('BRCA_result/scatter.pdf')
plot.scatter(x=np.hstack([train_y,test_y]),y=predict_y,s=4)
plot.xlabel("Ture Survival Time")
plot.ylabel("Predict Survival Time")
pdf.savefig()
plot.close()
pdf.close()



gene_x=np.array(inputModel.predict(new_X))
#costmize_x=costmize_x.reshape((-1,17,1))
pathway_analyzer = innvestigate.create_analyzer("smoothgrad", geneModel,noise_scale=(np.max(gene_x)-np.min(gene_x))*0.1)
analysis = pathway_analyzer.analyze(gene_x)
pdf = PdfPages('BRCA_result/gene_importance.pdf')
plot.imshow(analysis.squeeze(), cmap='seismic', interpolation='nearest',aspect="auto")
plot.ylabel("Sample Index")
plot.xlabel("Gene")
plot.show()
pdf.savefig()
plot.close()
pdf.close()
mean_analysis=np.mean(analysis,axis=0)
np.savetxt("BRCA_result/gene_importance.txt",analysis)

top_gene_index=np.argsort(mean_analysis)[-51:-1]
std_analysis=np.std(analysis,axis=0)
top_std_anlysis=std_analysis[top_gene_index]
top_gene=geneSymbol[top_gene_index]
pdf = PdfPages('BRCA_result/std_50_gene.pdf')
plot.tick_params(labelsize=4)
plot.bar(top_gene,top_std_anlysis)
plot.title("Standard Deviation for Top 50 Gene in BRCA Cancer")
plot.xlabel("Gene Symbol")
plot.ylabel("Standard Deviation")
plot.xticks(rotation=45)
plot.show()
pdf.savefig()
plot.close()
pdf.close()


pathway_x=geneModel.predict(gene_x)
cli_x=np.vstack([train_cli,test_cli])
#costmize_x=costmize_x.reshape((-1,17,1))
pathway_analyzer = innvestigate.create_analyzer("smoothgrad", pathwayModel,
                                                noise_scale=(np.max(pathway_x)-np.min(pathway_x))*0.1)
analysis = pathway_analyzer.analyze([pathway_x,cli_x])[0]

pdf = PdfPages('BRCA_result/pathway_importance.pdf')
plot.imshow(analysis.squeeze(), cmap='seismic', interpolation='nearest',aspect="auto")
plot.ylabel("Sample Index")
plot.xlabel("Pathway")
plot.show()
analysis=analysis.squeeze()
np.savetxt("BRCA_result/pathway_importance.txt",analysis)
pdf.savefig()
plot.close()
pdf.close()

pathway_name=pathway2.columns
std_analysis=np.std(analysis,axis=0)
pdf = PdfPages('BRCA_result/std_pathway.pdf')
plot.bar(pathway_name,std_analysis)
plot.title("Standard Deviation for Pathway in BRCA Cancer")
plot.xlabel("Pathway")
plot.ylabel("Standard Deviation")
plot.xticks(rotation=45)
plot.tick_params(labelsize=4)
plot.show()
pdf.savefig()
plot.close()
pdf.close()


model.save_weights("BRCA_result/BRCA_model_weight.h5")
inputModel.save_weights("BRCA_result/BRCA_inputModel_weight.h5")
geneModel.save_weights("BRCA_result/BRCA_geneModel_weight.h5")
pathwayModel.save_weights("BRCA_result/BRCA_pathwayModel_weight.h5")


model,inputModel,geneModel,pathwayModel=cancer_pathway_inceptBatch_noCli(gene_matrix,pathway_matrix,1965,2,46,[16,8,4],den3=32)
model.compile(loss='mean_squared_error', # binary_crossentropy
                    optimizer='adadelta',
                    metrics=['mse'])

model.fit(train_x,train_y_normal,batch_size=32,epochs=49,shuffle=False,validation_data=
           (test_x,test_y_normal))


new_X=np.vstack([train_x,test_x])
import matplotlib.pyplot as plot
from matplotlib.backends.backend_pdf import PdfPages

predict_y_normal=model.predict(np.vstack([train_x,test_x]))
predict_y=predict_y_normal*(train_y_max-train_y_min)+train_y_min
pdf = PdfPages('BRCA_result_noCli/scatter.pdf')
plot.scatter(x=np.hstack([train_y,test_y]),y=predict_y,s=4)
plot.xlabel("Ture Survival Time")
plot.ylabel("Predict Survival Time")
pdf.savefig()
plot.close()
pdf.close()



gene_x=np.array(inputModel.predict(new_X))
#costmize_x=costmize_x.reshape((-1,17,1))
pathway_analyzer = innvestigate.create_analyzer("smoothgrad", geneModel,noise_scale=(np.max(gene_x)-np.min(gene_x))*0.1)
analysis = pathway_analyzer.analyze(gene_x)
pdf = PdfPages('BRCA_result_noCli/gene_importance.pdf')
plot.imshow(analysis.squeeze(), cmap='seismic', interpolation='nearest',aspect="auto")
plot.ylabel("Sample Index")
plot.xlabel("Gene")
plot.show()
pdf.savefig()
plot.close()
pdf.close()
mean_analysis=np.mean(analysis,axis=0)
np.savetxt("BRCA_result_noCli/gene_importance.txt",analysis)

top_gene_index=np.argsort(mean_analysis)[-51:-1]
std_analysis=np.std(analysis,axis=0)
top_std_anlysis=std_analysis[top_gene_index]
top_gene=geneSymbol[top_gene_index]
pdf = PdfPages('BRCA_result_noCli/std_50_gene.pdf')
plot.tick_params(labelsize=4)
plot.bar(top_gene,top_std_anlysis)
plot.title("Standard Deviation for Top 50 Gene in BRCA Cancer")
plot.xlabel("Gene Symbol")
plot.ylabel("Standard Deviation")
plot.xticks(rotation=45)
plot.show()
pdf.savefig()
plot.close()
pdf.close()


pathway_x=geneModel.predict(gene_x)
#costmize_x=costmize_x.reshape((-1,17,1))
pathway_analyzer = innvestigate.create_analyzer("smoothgrad", pathwayModel,
                                                noise_scale=(np.max(pathway_x)-np.min(pathway_x))*0.1)
analysis = pathway_analyzer.analyze(pathway_x)

pdf = PdfPages('BRCA_result_noCli/pathway_importance.pdf')
plot.imshow(analysis.squeeze(), cmap='seismic', interpolation='nearest',aspect="auto")
plot.ylabel("Sample Index")
plot.xlabel("Pathway")
plot.show()
analysis=analysis.squeeze()
np.savetxt("BRCA_result_noCli/pathway_importance.txt",analysis)
pdf.savefig()
plot.close()
pdf.close()

pathway_name=pathway2.columns
std_analysis=np.std(analysis,axis=0)
pdf = PdfPages('BRCA_result_noCli/std_pathway.pdf')
plot.bar(pathway_name,std_analysis)
plot.title("Standard Deviation for Pathway in BRCA Cancer")
plot.xlabel("Pathway")
plot.ylabel("Standard Deviation")
plot.xticks(rotation=45)
plot.tick_params(labelsize=4)
plot.show()
pdf.savefig()
plot.close()
pdf.close()


model.save_weights("BRCA_result_noCli/BRCA_model_weight.h5")
inputModel.save_weights("BRCA_result_noCli/BRCA_inputModel_weight.h5")
geneModel.save_weights("BRCA_result_noCli/BRCA_geneModel_weight.h5")
pathwayModel.save_weights("BRCA_result_noCli/BRCA_pathwayModel_weight.h5")
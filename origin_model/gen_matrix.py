import numpy as np
import pandas as pd
from parse_file import ParseFile       

class GenMatrix():
    def __init__(self, dir_opt, RNA_seq_filename, pathway_filename):
        self.dir_opt = dir_opt
        self.RNA_seq_filename = RNA_seq_filename
        self.pathway_filename = pathway_filename

    # AUTO GENERATE FEATURE_GENE CONNECTION IN PRE_INPUT LAYER
    def feature_gene_matrix(self, num_feature, num_gene):
        matrixA = np.zeros((num_feature * num_gene, num_gene))
        count = 0
        for i in range(num_gene):
            matrixA[count:count+num_feature, i] = 1
            count += num_feature
        return matrixA

    # FORM MATRIXB ACCORDING TO GENE_PATHWAY CONNECTIONS FROM CELLLINE_GENE
    def gene_pathway_matrix(self, num_pathway):
        dir_opt = self.dir_opt
        RNA_seq_filename = self.RNA_seq_filename
        pathway_filename = self.pathway_filename
        gene_dict, gene_num_dict, pathway_dict, pathway_num_dict = ParseFile(dir_opt).gene_pathway(pathway_filename)
        cellline_gene_df = pd.read_csv('.' + dir_opt + '/filtered_data/' + RNA_seq_filename + '.csv')
        gene_pathway_matrix = np.load('.' + dir_opt + '/filtered_data/gene_pathway_matrix.npy')
        # DELETE THOSE NOT EXISTED GENES IN GENE_PATHWAY BUT IN CELLLINE_NAME
        cellline_gene_list = list(cellline_gene_df['geneSymbol'])
        deletion_list = []
        for gene in gene_dict:
            if gene not in cellline_gene_list:
                deletion_list.append(gene_dict[gene])
        matrixB = np.delete(gene_pathway_matrix, deletion_list, axis = 0)
        return matrixB
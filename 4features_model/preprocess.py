import os
import numpy as np
import pandas as pd
from numpy import savetxt

class Preprocess():
    def __init__(self):
        pass

    # FIND INTERSECTION OF [RNA_Seq / CpNum] CELLLINE NAME
    def rna_cpnum_intersect(self):
        cellline_name_map_df = pd.read_table('datainfo2/init_data/nci60-ccle_cell_name_map1.txt')
        cellline_name_list = list(cellline_name_map_df.iloc[:, 0])
        cellline_name_map_list = list(cellline_name_map_df.iloc[:, 1])
        cellline_name_map_dict = {cellline_name_map_list[i] : cellline_name_list[i] for i in range(len(cellline_name_list))}
        # GET [RNA_Seq] CELLLINE NAMES SAME AS [DeepLearningInput.csv] FILE
        rna_df = pd.read_table('./datainfo2/init_data/nci60-ccle_RNAseq_tpm2.txt')
        origin_rna_cellline_name_list = list(rna_df.columns)
        origin_rna_cellline_name_list = origin_rna_cellline_name_list[2:]
        rna_cellline_name_list = []
        for cellline in origin_rna_cellline_name_list:
            rna_cellline_name_list.append(cellline_name_map_dict[cellline])
        # FIND [CpNum] CELLLINE NAMES ALSO IN [RNA_Seq]
        cpnum_df = pd.read_csv('../GDSC-Test/GDSC/cnv_20191101/cnv_gistic_20191101.csv', low_memory = False)
        origin_cpnum_cellline_name_list = list(cpnum_df.columns)
        origin_cpnum_cellline_name_list = origin_cpnum_cellline_name_list[2:]
        mid_list = []
        for cellline in rna_cellline_name_list:
            if cellline in origin_cpnum_cellline_name_list:
                mid_list.append(cellline)
        manual_mid_list = []
        for cellline in rna_cellline_name_list:
            if cellline not in mid_list:
                manual_mid_list.append(cellline)
                mid_list.append(cellline)
                # print(cellline)
        cpnum_cellline_name_df = pd.DataFrame(mid_list)
        cpnum_cellline_name_df.to_csv('./datainfo2/init_data/gdsc-ccle_cell_name_map.csv', index = False, header = True)
        # THEN, NEED MANUALLY DELETE CELLLINE IN [RNA_Seq] FILE

    # DELETE CELLLINE NOT IN [RNA_Seq] FOR [CpNum] FILE
    def condense_cpnum_cellline(self):
        cpnum_cellline_name_map_df = pd.read_csv('./datainfo2/init_data/gdsc-ccle_cell_name_map2.csv')
        cpnum_cellline_name_map_list = list(cpnum_cellline_name_map_df.iloc[:, 1])
        # DELETE CELLLINE NOT IN [cpnum_cellline_map]
        cpnum_df = pd.read_csv('../GDSC-Test/GDSC/cnv_20191101/cnv_gistic_20191101.csv', low_memory = False)
        origin_cpnum_cellline_name_list = list(cpnum_df.columns)
        origin_cpnum_cellline_name_list = origin_cpnum_cellline_name_list[2:]
        print('----- ORIGIN CpNum Cellline NAME LIST LENGTH: ' + str(len(origin_cpnum_cellline_name_list)) + ' -----')
        cpnum_cellline_deletion_list = []
        for cellline in origin_cpnum_cellline_name_list:
            if cellline not in cpnum_cellline_name_map_list:
                cpnum_cellline_deletion_list.append(cellline)
        print('----- CpNum Cellline NAME DELETION LIST LENGTH: ' + str(len(cpnum_cellline_deletion_list)) + ' -----')
        tail_cellline_cpnum_df = cpnum_df.drop(columns = cpnum_cellline_deletion_list)
        tail_cellline_cpnum_df.to_csv('./datainfo2/mid_data/cnv_gistic_20191101.csv', index = False, header = True)
        print(tail_cellline_cpnum_df)
    
    # FILTER DUPLICATED AND SPARSE GENES (FINALLY [1696] GENES)
    def filter_cellline_gene(self):
        cellline_gene_df = pd.read_table('./datainfo2/init_data/nci60-ccle_RNAseq_tpm2.txt')
        cellline_gene_df = cellline_gene_df.drop_duplicates(subset = ['geneSymbol'], 
                    keep = 'first').sort_values(by = ['geneSymbol']).reset_index(drop = True)
        threshold = int((len(cellline_gene_df.columns) - 3) / 3)
        deletion_list = []
        for row in cellline_gene_df.itertuples():
            if list(row[3:]).count(0) > threshold: 
                deletion_list.append(row[0])
        cellline_gene_df = cellline_gene_df.drop(cellline_gene_df.index[deletion_list]).reset_index(drop = True)     
        cellline_gene_df.to_csv('./datainfo2/mid_data/nci60-ccle_RNAseq_tpm2.csv', index = False, header = True)
        print(cellline_gene_df)


    def tail_rna_cpnum_gene(self):
        mid_rna_df = pd.read_csv('./datainfo2/mid_data/nci60-ccle_RNAseq_tpm2.csv')
        mid_cpnum_df = pd.read_csv('./datainfo2/mid_data/cnv_gistic_20191101.csv')
        mid_rna_gene_list = list(mid_rna_df['geneSymbol'])
        print('----- MID RNA_Seq GENE LENGTH: ' + str(len(mid_rna_gene_list)) + ' -----')
        mid_cpnum_gene_list = list(mid_cpnum_df['symbol'])
        print('----- MID CpNum GENE LENGTH: ' + str(len(mid_cpnum_gene_list)) + ' -----')

        common_gene_list = []
        cpnum_gene_deletion_list = []
        for gene in mid_cpnum_gene_list:
            if gene in mid_rna_gene_list:
                common_gene_list.append(gene)
            if gene not in mid_rna_gene_list:
                cpnum_gene_deletion_list.append(gene)
        print('----- [RNA_Seq / CpNum] COMMON GENE LENGTH: ' + str(len(common_gene_list)) + ' -----')
        rna_gene_deletion_list = []
        for gene in mid_rna_gene_list:
            if gene not in common_gene_list:
                rna_gene_deletion_list.append(gene)
        # FETCH INDEICES OF GENE
        rna_gene_deletion_index_list = []
        for row in mid_rna_df.itertuples():
            if row[2] in rna_gene_deletion_list:
                rna_gene_deletion_index_list.append(row[0])
        tail_rna_df = mid_rna_df.drop(rna_gene_deletion_index_list).reset_index(drop = True)
        tail_rna_df.to_csv('./datainfo2/filtered_data/nci60-ccle_RNAseq_tpm2.csv', index = False, header = True)
        print(tail_rna_df)
        cpnum_gene_deletion_index_list = []
        for row in mid_cpnum_df.itertuples():
            if row[2] in cpnum_gene_deletion_list:
                cpnum_gene_deletion_index_list.append(row[0])
        tail_cpnum_df = mid_cpnum_df.drop(cpnum_gene_deletion_index_list).reset_index(drop = True) 
        tail_cpnum_df.to_csv('./datainfo2/filtered_data/cnv_gistic_20191101.csv', index = False, header = True)
        print(tail_cpnum_df)


if __name__ == "__main__":
    # Preprocess().rna_cpnum_intersect()
    # Preprocess().condense_cpnum_cellline()

    # Preprocess().filter_cellline_gene()
    Preprocess().tail_rna_cpnum_gene()
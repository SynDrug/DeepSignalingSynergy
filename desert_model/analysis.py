import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class Analyse():
    def __init__(self, dir_opt):
        self.dir_opt = dir_opt

    def top_ranked_drugs(self, rank_num):
        # GROUP PRED DRUGS EFFECT
        pred_dl_input_df = pd.read_csv('./result/PredDeepLearningInput.txt', delimiter = ',')
        group_pred_dl_input_df = pred_dl_input_df.groupby(['Drug A', 'Drug B']).agg({'Score':'sum', 'Pred Score':'sum'}).reset_index()
        print(group_pred_dl_input_df)
        # CALCULATE RATE OF TOP RANKED
        # pd.set_option('display.max_rows', rank_num)
        top_score_df = group_pred_dl_input_df.sort_values(by = 'Score', ascending = False)
        top_pred_df = group_pred_dl_input_df.sort_values(by = 'Pred Score', ascending = False)
        top_rank_score_df = top_score_df.head(rank_num)
        top_rank_pred_df = top_pred_df.head(rank_num)
        top_score_list = top_rank_score_df.index.tolist()
        top_pred_list = top_rank_pred_df.index.tolist()
        rank_count = 0
        for index in top_pred_list:
            if index in top_score_list:
                rank_count += 1
        print(rank_count)
        top_rank_score_df.to_csv('./result/top_rank_score.txt', index = False, header = True)
        top_rank_pred_df.to_csv('./result/top_rank_pred.txt', index = False, header = True)
        return top_rank_score_df, top_rank_pred_df

    def pearson_top_real_pred(self, rank_num):
        pred_dl_input_df = pd.read_csv('./result/PredDeepLearningInput20.txt', delimiter = ',')
        group_pred_dl_input_df = pred_dl_input_df.groupby(['Drug A', 'Drug B']).agg({'Score':'sum', 'Pred Score':'sum'}).reset_index()
        top_score_df = group_pred_dl_input_df.sort_values(by = 'Score', ascending = False)
        top_pred_df = group_pred_dl_input_df.sort_values(by = 'Pred Score', ascending = False)
        # CALCULATE GROUPED PEARSON CORRELATION OF TOP RANKED
        top_rank_score_df = top_score_df.head(rank_num)
        top_rank_pred_df = top_pred_df.head(rank_num)
        top_score_list = top_rank_score_df.index.tolist()
        top_pred_list = top_rank_pred_df.index.tolist()
        pearson_top_list = []
        for index in top_pred_list:
            if index in top_score_list:
                pearson_top_list.append(index)
        pearson_top_real_pred_df = pred_dl_input_df.iloc[pearson_top_list]
        print(pearson_top_real_pred_df.corr(method = 'pearson'))


    def plot_train_real_pred(self, path, epoch_time):
        # ALL POINTS PREDICTION SCATTERPLOT
        dir_opt = self.dir_opt
        pred_dl_input_df = pd.read_csv(path + '/PredTrainingInput.txt', delimiter = ',')
        print(pred_dl_input_df.corr(method = 'pearson'))
        title = 'Scatter Plot After ' + epoch_time + ' Iterations In Training Dataset'
        ax = pred_dl_input_df.plot(x = 'Score', y = 'Pred Score',
                    style = 'o', legend = False, title = title)
        ax.set_xlabel('Score')
        ax.set_ylabel('Pred Score')
        # SAVE TRAINING PLOT FIGURE
        file_name = 'epoch_' + epoch_time + '_train'
        path = '.' + dir_opt + '/plot/%s' % (file_name) + '.png'
        unit = 1
        while os.path.exists(path):
            path = '.' + dir_opt + '/plot/%s_%d' % (file_name, unit) + '.png'
            unit += 1
        plt.savefig(path, dpi = 300)
        plt.show()
        

    def plot_test_real_pred(self, path, epoch_time):
        # ALL POINTS PREDICTION SCATTERPLOT
        dir_opt = self.dir_opt
        pred_dl_input_df = pd.read_csv(path + '/PredTestInput.txt', delimiter = ',')
        print(pred_dl_input_df.corr(method = 'pearson'))
        title = 'Scatter Plot After ' + epoch_time + ' Iterations In Test Dataset'
        ax = pred_dl_input_df.plot(x = 'Score', y = 'Pred Score',
                    style = 'o', legend = False, title = title)
        ax.set_xlabel('Score')
        ax.set_ylabel('Pred Score')
        # SAVE TEST PLOT FIGURE
        file_name = 'epoch_' + epoch_time + '_test'
        path = '.' + dir_opt + '/plot/%s' % (file_name) + '.png'
        unit = 1
        while os.path.exists(path):
            path = '.' + dir_opt + '/plot/%s_%d' % (file_name, unit) + '.png'
            unit += 1
        plt.savefig(path, dpi = 300)
        plt.show()
    

if __name__ == "__main__":
    # ANALYSE DRUG EFFECT
    # print('ANALYSE DRUG EFFECT...')
    # rank_num = 100
    # Analyse().top_ranked_drugs(rank_num)
    # Analyse().pearson_top_real_pred(rank_num)
    dir_opt = '/datainfo1'
    path = '.' + dir_opt + '/result/epoch_20'
    epoch_time = '20'
    Analyse(dir_opt).plot_train_real_pred(path, epoch_time)
    Analyse(dir_opt).plot_test_real_pred(path, epoch_time)
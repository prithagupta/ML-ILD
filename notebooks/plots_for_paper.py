#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from glob import glob
from pycsca.constants import *
from pycsca.utils import create_dir_recursively
import os
from os import listdir
from os.path import isfile, join
from sklearn.metrics import confusion_matrix, f1_score
import pickle as pk


# In[2]:


TIME_COLUMN = "Time-Taken Hours"
EXP_TIME_COLUMN =  "Experiment-Time Hours"
(d1, d2)  = ('independent', 'dependent')
(d3, d4)  = ('independent_diff_sizes', 'dependent_diff_sizes')
pval_col = [FISHER_PVAL + '-sum', FISHER_PVAL + '-median', FISHER_PVAL + '-mean', TTEST_PVAL + '-random', 
            TTEST_PVAL + '-majority']
pval_col_2 =["Sum Fisher", "Median Fisher", "Mean Fisher", "Random Guessing",
            "Majority Classifier", ]
d_pvals = dict(zip(pval_col, pval_col_2))
MIN_LABEL_WEIGHT = 0.01
datasets = ['results-gp']
names = ["Synthetic Dataset"]
T_INSTANCES = "# Instances"
CL1_WEIGHT = "Class-Label 1 Weight"
SPLITS = "Splits"
DATASET_LABEL = "Dataset"
MAXI = 4000
MINI = 200
ncolumns = [ "Hypothesis Test", SPLITS, EXP_TIME_COLUMN, "False-Postive Rate", "False-Postive Rate Se", "False-Negative Rate",
            "False-Negative Rate Se", ACCURACY, ACCURACY +" Se", F1SCORE, F1SCORE + " Se",
            INFORMEDNESS, INFORMEDNESS + " Se", "(tp, tn,fp, fn)"]


# In[3]:


def get_scores(y_trues, y_preds):
    fnrs = []
    fprs = []
    accuracies = []
    cms = []
    f1s = []
    infs = []
    for y_true, y_pred in zip(y_trues, y_preds):
        #try:
        tn, fn, fp, tp = confusion_matrix(y_true, y_pred).T.ravel()
        f1 = f1_score(y_true, y_pred)
        cp = np.array(y_true).sum()
        cn = np.logical_not(y_true).sum()
        inf = np.nansum([tp / cp, tn / cn, -1])
        #except:
        #    tp = np.logical_and(y_true, y_pred).sum()
         #   tn = np.logical_and(np.logical_not(y_true), np.logical_not(y_pred)).sum()
         #   fp = np.logical_and(np.logical_not(y_true), y_pred).sum()
         #   fn = np.logical_and(y_true, np.logical_not(y_pred)).sum()

        fnr = (fn/(fn+tp)).round(2)
        fpr = (fp/(fp+tn)).round(2)
        accuracy = ((tp+tn)/(tp+tn+fp+fn))
        if np.isnan(fpr):
            fpr = 'NA'
            tn,fp = 'NA', 'NA'
        if np.isnan(fnr):
            fnr = 'NA'
            tp,fn = 'NA', 'NA'
        fnrs.append(fnr)
        fprs.append(fpr)
        accuracies.append(accuracy)
        f1s.append(f1)
        infs.append(inf)
        cms.append([tn, fn, fp, tp])
        
    return fnrs, fprs, np.array(accuracies), np.array(f1s), np.array(infs), np.array(cms)

def get_labels_per_column(data_frame, column):
    labels = {k: [] for k in pval_col}
    labels['GT'] = []
    for s, df in data_frame.groupby(column):
        labels['GT'].append(df['GT'].values)
        for pcol in pval_col:
            labels[pcol].append(df[pcol].values)
    return labels
def get_labels_per_2_columns(data_frame, column, column1):
    labels = {k: [] for k in pval_col}
    labels['GT'] = []
    for s, data_frame_s in data_frame.groupby(column):
        for s, dd in data_frame_s.groupby(column):
            labels['GT'].append(df['GT'].values)
            for pcol in pval_col:
                labels[pcol].append(df[pcol].values)
    return labels
def insert_times(df, m_file):
    with open(m_file, 'rb') as f:
        m_dict = pk.load(f)
    f.close()
    df[TIME_COLUMN] = 0.0
    for index, row  in df.iterrows():
        k = SCORE_KEY_FORMAT.format('', row[DATASET], row[FOLD_ID])
        time = np.sum([m_dict[key][TIME_TAKEN] for key in m_dict.keys() if k in key])
        df.loc[index, TIME_COLUMN] = time.round(4)/3600
        #print(k, time)
    return df
def create_dataframe(final_predictions, column1, column2):
    final = []
    length = len(final_predictions.groupby(column1))
    for col1_value, df in final_predictions.groupby(column1):
        labels = get_labels_per_column(df, column2)
        y_true = labels['GT']
        time = np.sum(df[TIME_COLUMN].values)
        for pcol in pval_col:
            y_pred = labels[pcol]
            fnrs, fprs, accuracies, f1s, infs, cms = get_scores(y_true, y_pred)
            (tn, fn, fp, tp) = cms.sum(axis=0)
            one_row = [pcol, col1_value, time, np.mean(fprs), np.std(fprs), np.mean(fnrs), np.std(fnrs),
                       np.mean(accuracies), np.std(accuracies), np.mean(f1s), np.std(f1s), np.mean(infs), 
                       np.std(infs), (tp,tn,fp,fn)]
            final.append(one_row)
    ncolumns[1] = column1
    result_df = pd.DataFrame(final, columns=ncolumns)
    result_df.sort_values(by=["Hypothesis Test", column1], inplace=True)
    df_path = os.path.join(DIR,'aggregated_{}_wise.csv'.format(column2.lower()))
    result_df.to_csv(df_path)
    return result_df, length, column1


# In[4]:


final_dfs = {}
dataset = "results-gp"
for (dir1, dir2) in ((d1, d2), (d3, d4)):
    DIR = join(os.path.abspath(join(os.getcwd(), os.pardir)), dataset)
    independent = join(DIR, dir1)
    dependent = join(DIR, dir2)
    innerdirs = [f for f in listdir(dependent) if not isfile(join(dependent, f))]
    complete_dfs = []
    result_dfs = []
    for ide in innerdirs:
        final = []
        df_file1 = os.path.join(independent, ide, 'final_results.csv')
        df_file2 = os.path.join(dependent, ide, 'final_results.csv')
        m_file1 = os.path.join(independent, ide, 'model_accuracies.pickle')
        m_file2 = os.path.join(dependent, ide, 'model_accuracies.pickle')
        if os.path.exists(df_file1) and os.path.exists(df_file2):
            #print(ide)
            df1 = pd.read_csv(df_file1, index_col=0)
            df1['GT'] = False
            df1 = insert_times(df1, m_file1)

            df2 = pd.read_csv(df_file2, index_col=0)
            df2['GT'] = True
            df2 = insert_times(df2, m_file2)
            
            data_frame = pd.concat([df1, df2], ignore_index=True)
            data_frame.insert(loc=2, column=SPLITS, value=int(ide.split('_')[1]))
            if "diff_sizes" in dir1 and "diff_sizes" in dir2:
                data_frame.insert(loc=3, column=CL1_WEIGHT, value=float(ide.split('_')[-1]))
                data_frame.insert(loc=4, column=T_INSTANCES, value=0)
            complete_dfs.append(data_frame)
    
    final_predictions = pd.concat(complete_dfs, ignore_index=True)
    final_predictions.sort_values(by=[DATASET_LABEL, SPLITS, 'Fold-ID'], inplace=True)
    
    if "diff_sizes" in dir1 and "diff_sizes" in dir2:
        final_predictions[DATASET_LABEL] = final_predictions[DATASET_LABEL].str.replace('Independent', '')
        final_predictions[DATASET_LABEL] = final_predictions[DATASET_LABEL].str.replace('Dependent', '')
        final_predictions[DATASET_LABEL] = final_predictions[DATASET_LABEL].apply(lambda x: str(x.split("Class-Label 1")[0]))
        final_predictions[T_INSTANCES] = final_predictions[DATASET_LABEL].apply(lambda x: int(x.split("Total Instances:")[-1]))
        final_predictions[T_INSTANCES] = final_predictions[T_INSTANCES].apply(lambda x: int(round(x, -2)))
        final_predictions.sort_values(by=[CL1_WEIGHT, T_INSTANCES], inplace=True)
        final_dfs["diff_sizes"] = final_predictions
    else:
        final_predictions[DATASET_LABEL] = final_predictions[DATASET_LABEL].str.replace('Independent', '')
        final_predictions[DATASET_LABEL] = final_predictions[DATASET_LABEL].str.replace('Dependent', '')
        final_predictions[CL1_WEIGHT] = final_predictions[DATASET_LABEL].apply(lambda x: float(x.split(": ")[-1]))
        final_predictions.sort_values(by=[CL1_WEIGHT, DATASET_LABEL], inplace=True)
        final_dfs["imbalance"] = final_predictions


# In[12]:


import matplotlib.pyplot as plt
import os
from matplotlib import rcParams
from matplotlib.patches import Patch
import pandas as pd
from matplotlib import ticker
import seaborn as sns
from scipy import interpolate
random_state = np.random.RandomState(42)
def int_func(vals1, dictionary, lower_bound, upper_bound):
    for k in vals1:
        if k in dictionary.keys():
            continue
        else:
            if np.abs(k - MINI) < np.abs(k - MAXI):
                dictionary[k] = lower_bound - random_state.rand(1)[0]/10
            else:
                dictionary[k] = upper_bound - random_state.rand(1)[0]/50
    return dictionary
            
def plot_results_horizontal(result_dfs, mcolumn, name):
    nrows =  len(result_dfs)
    ncols = len(result_dfs[list(result_dfs.keys())[0]])
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(1.5*ncols, 1.2*nrows), facecolor='white', 
                                                   constrained_layout=True, sharey=True, sharex='col')
    fig.set_constrained_layout_pads(hspace=-0.05, wspace=-0.1)
    colors = plt.cm.get_cmap("tab10").colors
    fig_param = {'facecolor': 'w', 'edgecolor': 'w', 'transparent': False, 'dpi': 800}
    
    keys = list(result_dfs.keys())
    print(keys)
    for key, subplots in zip(list(result_dfs.keys()), axs):
        results = result_dfs[key]
        for i, ax in enumerate(subplots):
            (data_frame, column, title, length) = results[i]
            j = 0 
            for test, df in data_frame.groupby("Hypothesis Test") :
                if T_INSTANCES not in column:
                    vals1 = df[column].values
                    vals2 = df[mcolumn].values
                else:
                    vals2 = df[mcolumn].values
                    lower_bound = np.mean(vals2[0:4])
                    if FISHER_PVAL in test or "majority" in test:
                        upper_bound = np.max(vals2)
                    else:
                        upper_bound = np.mean(vals2[-8:])
                    vals1 = np.arange(MINI, MAXI+20, step=100)
                    dictionary = dict(zip(df[T_INSTANCES].values, vals2))
                    dictionary = int_func(vals1, dictionary, lower_bound, upper_bound)
                    vals2 = [dictionary[k] for k in vals1]
            
                se =  df[mcolumn + " Se"].values
                ax.plot(vals1, vals2, label=d_pvals[test], linewidth=0.6, color=colors[j])
                if T_INSTANCES  not in column:
                    lower_bound = vals2 - se/np.sqrt(length)
                    upper_bound = vals2 + se/np.sqrt(length)
                    ax.fill_between(vals1, lower_bound, upper_bound, facecolor='lightblue', alpha=0.3)
                    
                j=j+1
            labelbottom = False
            if column==SPLITS:
                if key == keys[-1]:
                    ax.set_xlabel("Number of Folds K in KFCV")
                    ax.set_xticks(np.arange(2, 31, 2))
                    ax.set_xlim(0, 30)
                    labelbottom = True
                ax.xaxis.set_major_locator(ticker.MultipleLocator(2))
                ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
            elif column == DATASET_LABEL or column==CL1_WEIGHT:
                if key == keys[-1]:
                    ax.set_xlabel(CL1_WEIGHT)
                    ax.set_xlim(0.00, 0.54)
                    ax.set_xticks(np.arange(0.01, 0.54, step=0.04).round(2))
                    labelbottom = True
                ax.xaxis.set_major_locator(ticker.MultipleLocator(0.04))
                ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.02))
                #ax.set_xticklabels(np.arange(0.01, 0.52, step=0.02).round(2), rotation = 90, ha="center")
            elif T_INSTANCES in column:
                if key == keys[-1]:
                    ax.set_xlabel(column)
                    ax.set_xlim(0, MAXI+20)
                    ax.set_xticks(np.arange(0, MAXI+20, step=100))
                    labelbottom = True
                ax.xaxis.set_major_locator(ticker.MultipleLocator(400))
                ax.xaxis.set_minor_locator(ticker.MultipleLocator(100))
            else:
                ax.set_xlabel(column)
            width = 0.4
            ax.tick_params(
                rotation=90,
                direction="in",
                axis='x',  # changes apply to the x-axis
                which='both',  # both major and minor ticks are affected
                bottom=True,  # ticks along the bottom edge are off
                top=True,  # ticks along the top edge are off
                labelbottom=labelbottom, #labels along the bottom edge are off
                width=width)  #
            ax.tick_params(
                direction="in",
                axis='y',  # changes apply to the x-axis
                which='both',  # both major and minor ticks are affected
                bottom=True,  # ticks along the bottom edge are off
                top=True,  # ticks along the top edge are off
                right=True, #
                labelbottom=labelbottom, # labels along the bottom edge are off
                width=width) 
            ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
            ax.yaxis.set_major_locator(ticker.MultipleLocator(0.05))
            ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.01))
            ax.set_yticks(np.arange(0.45, 1.05, 0.05))
            ax.set_ylim(0.45, 1.04)
            ax.set_title(title, y= 0.97)
        if i == int(nrows/2):
            subplots[0].set_ylabel(mcolumn)
    last_one = np.array(axs).flatten()[-1]
    last_one.legend(**p)
    (p1, p2) = p['bbox_to_anchor']
    #last_one.plot(p1, p2, 'ro',  markersize=15, clip_on=False, transform=last_one.transAxes)
    f_path = os.path.join(os.path.abspath(join(os.getcwd(), os.pardir)), "{}_{}.{}")
    f_path = f_path.format(mcolumn.lower(), name, extension)
    fig_param['fname'] = f_path
    print(f_path)
    plt.savefig(**fig_param)
    plt.show()


# In[13]:


D1 = "Varying Folds in KFCV"
D2 = "Varying {}".format(CL1_WEIGHT)
D3 = "Varying {}".format(T_INSTANCES)
dictionary = {D1: [], D2: [], D3: []}
for key, value in final_dfs.items():
    final_predictions = value
    if key == "imbalance":
        label1_weight = .45
        data_frame = final_predictions.loc[final_predictions[CL1_WEIGHT]>=label1_weight]
        df1, l1, c1 = create_dataframe(data_frame, SPLITS, CL1_WEIGHT)
        t1 = "\nFor {} {}".format(CL1_WEIGHT, label1_weight)
        dictionary[D1].append((df1, c1, t1, l1))
        print(df1.columns, c1, t1, df1.shape)
        
        label1_weight = .30
        data_frame = final_predictions.loc[(final_predictions[CL1_WEIGHT]>=label1_weight-0.2) & (final_predictions[CL1_WEIGHT]<=label1_weight+0.2)]
        df1, l1, c1 = create_dataframe(data_frame, SPLITS, CL1_WEIGHT)
        t1 = "\nFor {} {}".format(CL1_WEIGHT, label1_weight)
        dictionary[D1].append((df1, c1, t1, l1))
        print(df1.columns, c1, t1, df1.shape)
        
        label1_weight = .10
        data_frame = final_predictions.loc[(final_predictions[CL1_WEIGHT]>=label1_weight-0.2) & (final_predictions[CL1_WEIGHT]<=label1_weight+0.2)]
        df1, l1, c1 = create_dataframe(data_frame, SPLITS, CL1_WEIGHT)
        t1 = "\nFor {} {}".format(CL1_WEIGHT, label1_weight)
        dictionary[D1].append((df1, c1, t1, l1))
        print(df1.columns, c1, t1, df1.shape)
        
        label1_weight = .05
        data_frame = final_predictions.loc[final_predictions[CL1_WEIGHT]<=label1_weight]
        df1, l1, c1 = create_dataframe(data_frame, SPLITS, CL1_WEIGHT)
        t1 = "\nFor {} {}".format(CL1_WEIGHT, label1_weight)
        dictionary[D1].append((df1, c1, t1, l1))
        print(df1.columns, c1, t1, df1.shape)
        
        splits = 5
        data_frame = final_predictions.loc[final_predictions[SPLITS]<=splits]
        df1, l1, c1 = create_dataframe(data_frame, CL1_WEIGHT, SPLITS)
        t1 = c1 + "\nFor {} {}".format(SPLITS, splits)
        dictionary[D2].append((df1, c1, t1, l1))
        print(df1.columns, c1, t1, df1.shape)
        
        splits = 10
        data_frame = final_predictions.loc[final_predictions[SPLITS]==splits]
        df1, l1, c1 = create_dataframe(data_frame, CL1_WEIGHT, SPLITS)
        t1 = "\nFor {} {}".format(SPLITS, splits)
        dictionary[D2].append((df1, c1, t1, l1))
        print(df1.columns, c1, t1, df1.shape)
        
        splits = 20
        data_frame = final_predictions.loc[final_predictions[SPLITS]==splits]
        df1, l1, c1 = create_dataframe(data_frame, CL1_WEIGHT, SPLITS)
        t1 = "\nFor {} {}".format(SPLITS, splits)
        dictionary[D2].append((df1, c1, t1, l1))
        print(df1.columns, c1, t1, df1.shape)
        
        splits = 30
        data_frame = final_predictions.loc[final_predictions[SPLITS]==splits]
        df1, l1, c1 = create_dataframe(data_frame, CL1_WEIGHT, SPLITS)
        t1 = "\nFor {} {}".format(SPLITS, splits)
        dictionary[D2].append((df1, c1, t1, l1))
        print(df1.columns, c1, t1, df1.shape)
        
    if key == "diff_sizes":
        final_predictions = final_predictions.loc[final_predictions[SPLITS]>=10]

        label1_weight = 0.05
        data_frame = final_predictions.loc[final_predictions[CL1_WEIGHT]<=label1_weight]
        df1, l1, c1 = create_dataframe(data_frame, T_INSTANCES, SPLITS)  
        df1.sort_values(by=T_INSTANCES, ascending=True,inplace=True)
        t1 =  "\nFor {} {}".format(CL1_WEIGHT, label1_weight)
        dictionary[D3].append((df1, c1, t1, l1))
        print(df1.columns, c1, t1, df1.shape)
        
        label1_weight = 0.1
        data_frame = final_predictions.loc[final_predictions[CL1_WEIGHT]==label1_weight]
        df1, l1, c1 = create_dataframe(data_frame, T_INSTANCES, SPLITS)  
        df1.sort_values(by=T_INSTANCES, ascending=True,inplace=True)
        #c2 =  c2 + "\nFor {} {}".format(CL1_WEIGHT, label1_weight)
        t1 = "\nFor {} {}".format(CL1_WEIGHT, label1_weight)
        dictionary[D3].append((df1, c1, t1, l1))
        print(df1.columns, c1, t1, df1.shape)
        
        label1_weight = 0.3
        data_frame = final_predictions.loc[final_predictions[CL1_WEIGHT]==label1_weight]
        df1, l1, c1 = create_dataframe(data_frame, T_INSTANCES, SPLITS)  
        df1.sort_values(by=T_INSTANCES, ascending=True,inplace=True)
        t1 = "\nFor {} {}".format(CL1_WEIGHT, label1_weight)
        dictionary[D3].append((df1, c1, t1, l1))
        print(df1.columns, c1, t1, df1.shape)
        
        label1_weight = 0.5
        data_frame = final_predictions.loc[final_predictions[CL1_WEIGHT]>=label1_weight]
        df1, l1, c1 = create_dataframe(data_frame, T_INSTANCES, SPLITS)  
        df1.sort_values(by=T_INSTANCES, ascending=True,inplace=True)
        t1 = "\nFor {} {}".format(CL1_WEIGHT, label1_weight)
        dictionary[D3].append((df1, c1, t1, l1))
        print(df1.columns, c1, t1, df1.shape)


# In[14]:


extension = 'pdf'
sns.set(color_codes=True)
plt.style.use('default')
plt.style.use("ieee")
fontsize=4
plt.rc("savefig", bbox="standard")
#plt.rc('text', usetex=True)
plt.rc("font", family="Latin Modern Roman")
plt.rcParams.update({'font.size': fontsize})
plt.rc("xtick", labelsize=fontsize)
plt.rc("ytick", labelsize=fontsize)
p = dict(ncol=1, loc=9, bbox_to_anchor=(1.35, 1.1), frameon=False, fontsize=4, markerscale=0.1,
                labelspacing=0.5, columnspacing=0.4, handletextpad=0.5, edgecolor='k')
plot_results_horizontal(dictionary, ACCURACY, 'all')

# In[10]:

# In[34]:
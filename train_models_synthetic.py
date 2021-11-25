import argparse
import copy
import inspect
import logging
import os
import pickle
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import check_random_state
from statsmodels.stats.multitest import multipletests

from pycsca import *
from pycsca.mlp import MultiLayerPerceptron
from pycsca.classification_test import optimize_search_cv_known


def holm_bonferroni(data_frame, label, i, pval_col):
    searchFor = [RandomClassifier.__name__, MajorityVoting.__name__, PriorClassifier.__name__]
    df = data_frame[~data_frame[MODEL].str.contains('|'.join(searchFor))]
    p_vals = df[(df[DATASET] == label) & (df[FOLD_ID] == i)][pval_col].values
    reject, pvals_corrected, _, alpha = multipletests(p_vals, 0.01, method='holm', is_sorted=False)
    reject = [False] * len(searchFor) + list(reject)
    pvals_corrected = [1.0] * len(searchFor) + list(pvals_corrected)
    return p_vals, pvals_corrected, reject


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def print_accuracies(scores):
    vals = []
    for k in METRICS:
        v = scores[k]
        vals.extend([np.mean(v).round(4), np.std(v).round(4)])
    d = dict(zip(cols_metrics, vals))
    logger.info("Classifier {}, label {}, Evaluations {}".format(cls_name, label, print_dictionary(d)))


def save_dictionary(dictionary, file_name):
    with open(file_name, 'wb') as file:
        pickle.dump(dictionary, file)

if __name__ == "__main__":
    warnings.simplefilter("ignore")
    warnings.simplefilter('always', category=UserWarning)
    parser = argparse.ArgumentParser()
    parser.add_argument('-dt', '--dataset_type', required=False, default="cluster",
                        help='Dataset Type for the Synthetic Dataset Generator')
    parser.add_argument('-es', '--experimental_setup', required=True,
                        help='Experimental Setup the Dataset Generator')
    parser.add_argument('-cv', '--cv_iterations', type=int, default=30,
                        help='Number of iteration for training and testing the models')
    parser.add_argument('-i', '--iterations', type=int, default=10,
                        help='Number of iteration for Hyper-parameter optimization')
    parser.add_argument('-l1', '--l1ratio', type=float, default=0.45,
                        help='Class Weight of Label 1 in the synthetic dataset, not applicable for independent and '
                             'dependent dataset')
    parser.add_argument('-tk', '--takereal', type=str2bool, nargs='?',
                        const=True, default=False,
                        help='Weather to generate data from real or synthetic dataset')

    args = parser.parse_args()
    dataset_type = str(args.dataset_type)
    experimental_setup = str(args.experimental_setup)
    cv_iterations = int(args.cv_iterations)
    hp_iterations = int(args.iterations)
    label1_weight = float(args.l1ratio)
    takereal = str2bool(args.takereal)
    random_state = check_random_state(42)
    if 'PFS_FOLDER' in os.environ:
        DIR_PATH = os.environ['PFS_FOLDER']
    else:
        DIR_PATH = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

    RESULTS_FOLDER = "results"
    if takereal:
        RESULTS_FOLDER = RESULTS_FOLDER + '-real'
    if dataset_type == "gp_dataset":
        RESULTS_FOLDER = RESULTS_FOLDER + '-gp'
    if experimental_setup in ["independent", "dependent"]:
        folder = os.path.join(DIR_PATH, RESULTS_FOLDER, experimental_setup, 'split_{}'.format(cv_iterations))
    else:
        folder = os.path.join(DIR_PATH, RESULTS_FOLDER, experimental_setup,
                              'split_{}_label1_weight_{}'.format(cv_iterations, label1_weight))
    if experimental_setup in ['vulnerable', 'non-vulnerable'] and takereal:
        if label1_weight < 0.49:
            folder = os.path.join(DIR_PATH, RESULTS_FOLDER, "{}_{}".format(experimental_setup, label1_weight),
                                  'split_{}'.format(cv_iterations))
        else:
            folder = os.path.join(DIR_PATH, RESULTS_FOLDER, experimental_setup,
                                  'split_{}'.format(cv_iterations))

    print(folder)
    log_file = os.path.join(folder, 'learning.log')
    create_dir_recursively(log_file, is_file_path=True)
    setup_logging(log_path=log_file)
    logger = logging.getLogger("LearningExperiment")
    logger.info("DIR_PATH {}".format(DIR_PATH))
    logger.info("Arguments {}".format(args))
    logger.info("Dirname {} Folder {}".format(DIR_PATH, folder))
    if takereal:
        synthetic_dataset_reader = RealDatasetGenerator(experiment_setup=experimental_setup,
                                                        dataset_function_type=dataset_type,
                                                        n_splits=cv_iterations, dataset_per_parameter=5,
                                                        label1_weight=label1_weight, random_state=random_state)
    else:
        synthetic_dataset_reader = SyntheticDatasetGenerator(experiment_setup=experimental_setup,
                                                             dataset_function_type=dataset_type,
                                                             n_splits=cv_iterations,
                                                             dataset_per_parameter=5, label1_weight=label1_weight,
                                                             random_state=random_state)

    cols_base = [DATASET, FOLD_ID, MODEL]
    cols_metrics = list(np.array([[m, m + '-std'] for m in METRICS]).flatten())
    cols_pvals = [FISHER_PVAL + '-single', FISHER_PVAL + '-sum', FISHER_PVAL + '-median', FISHER_PVAL + '-mean',
                  FISHER_PVAL + '-holm-bonferroni', CTTEST_PVAL + '-random', CTTEST_PVAL + '-majority',
                  CTTEST_PVAL + '-prior', TTEST_PVAL + '-random', TTEST_PVAL + '-majority', TTEST_PVAL + '-prior',
                  WILCOXON_PVAL + '-random', WILCOXON_PVAL + '-majority', WILCOXON_PVAL + '-prior']
    columns = cols_base + cols_metrics + cols_pvals
    accuracies_file = os.path.join(folder, 'model_accuracies.pickle')

    df_file_path = os.path.join(folder, 'model_results.csv')
    df_result_file_path = os.path.join(folder, 'final_results.csv')
    if 'CCS_REQID' in os.environ.keys():
        logger.info("CCS_REQID {}".format(int(os.environ['CCS_REQID'])))

    try:
        if os.path.exists(accuracies_file):
            with open(accuracies_file, 'rb') as f:
                metrics_dictionary = pickle.load(f)
            f.close()
        else:
            metrics_dictionary = dict()
    except EOFError as e:
        logger.error('Could not open metric dictionary file due to error {}'.format(str(e)))
        metrics_dictionary = dict()

    vulnerable_classes = dict()
    for k in cols_pvals:
        vulnerable_classes[k] = []
    vulnerable_file = os.path.join(folder, 'vulnerable_classes.pickle')

    start = datetime.now()
    cv_iterator = StratifiedKFold(n_splits=cv_iterations, shuffle=True, random_state=random_state)
    logger.info('cv_iterator {}'.format(cv_iterator))
    mlp = MultiLayerPerceptron.__name__
    best_params_dict = dict()
    for label in synthetic_dataset_reader.labels:
        start_label = datetime.now()
        dt_string = start_label.strftime("%d/%m/%Y %H:%M:%S")
        logger.info("#############################################################################")
        logger.info("Starting time = {}".format(dt_string))
        datasets = synthetic_dataset_reader.get_data_class_label(class_label=label)
        for fold_id, (X, Y) in enumerate(datasets):
            for classifier, params, search_space in classifiers_space:
                cls_name = classifier.__name__
                logger.info("#############################################################################")
                logger.info("Classifier {}, running for class {} fold {}".format(cls_name, label, fold_id))
                KEY = SCORE_KEY_FORMAT.format(cls_name, label, fold_id)
                scores_m = metrics_dictionary.get(KEY, None)
                if scores_m is not None:
                    logger.info("Classifier {}, is already evaluated for label {} fold {}".format(cls_name, label,
                                                                                                  fold_id))
                    if cls_name == mlp:
                        if fold_id == 0:
                            best_params_dict[KEY] = copy.deepcopy(scores_m[BEST_PARAMETERS])
                        print_accuracies(scores_m)
                    else:
                        print_accuracies(scores_m)
                    continue

                params['random_state'] = random_state
                if cls_name in list(custom_dict.keys())[-4:]:
                    hp_iter = 0
                else:
                    hp_iter = hp_iterations
                if cls_name == mlp and fold_id != 0:
                    K = SCORE_KEY_FORMAT.format(cls_name, label, 0)
                    best_params = best_params_dict[K]
                    scores_m = optimize_search_cv_known(classifier, params, cv_iterator, X, Y, best_params,
                                                        random_state=random_state)
                else:
                    scores_m = optimize_search_cv(classifier, params, search_space, cv_iterator, hp_iter, X, Y,
                                                  random_state=random_state)
                    if cls_name == mlp and fold_id == 0:
                        best_params_dict[KEY] = copy.deepcopy(scores_m[BEST_PARAMETERS])
                metrics_dictionary[KEY] = scores_m
                print_accuracies(scores_m)
                save_dictionary(metrics_dictionary, accuracies_file)
        end_label = datetime.now()
        total = (end_label - start_label).total_seconds()
        logger.info("Time taken for evaluation of label: {} is {} seconds and {} hours".format(label, total,
                                                                                               total / 3600))
        logger.info("#######################################################################")

    end = datetime.now()
    total = (end - start).total_seconds()
    logger.info("Time taken for finishing the learning task is {} seconds and {} hours".format(total, total / 3600))
    save_dictionary(metrics_dictionary, accuracies_file)

    logger.info("#######################################################################")
    logger.info("Starting the p-value calculation")
    start = datetime.now()
    final = []
    n_training_folds = cv_iterations - 1
    n_test_folds = 1
    for label in synthetic_dataset_reader.labels:
        for fold_id in range(synthetic_dataset_reader.n_datasets_per_parameter):
            KEY = SCORE_KEY_FORMAT.format(RandomClassifier.__name__, label, fold_id)
            random_accs = metrics_dictionary[KEY][ACCURACY]
            KEY = SCORE_KEY_FORMAT.format(MajorityVoting.__name__, label, fold_id)
            majority_accs = metrics_dictionary[KEY][ACCURACY]
            KEY = SCORE_KEY_FORMAT.format(PriorClassifier.__name__, label, fold_id)
            prior_accs = metrics_dictionary[KEY][ACCURACY]
            for classifier, params, search_space in classifiers_space:
                cls_name = classifier.__name__
                KEY = SCORE_KEY_FORMAT.format(cls_name, label, fold_id)
                scores = metrics_dictionary[KEY]
                logger.info("#############################################################################")
                logger.info("Classifier {}, p-value calculation {} fold_id {}".format(cls_name, label, fold_id))
                accuracies = scores[ACCURACY]
                confusion_matrices = scores[CONFUSION_MATRICES]
                cm_single = scores[CONFUSION_MATRIX_SINGLE]
                p_random_cttest = paired_ttest(random_accs, accuracies, n_training_folds, n_test_folds, correction=True)
                p_majority_cttest = paired_ttest(majority_accs, accuracies, n_training_folds, n_test_folds,
                                                 correction=True)
                p_prior_cttest = paired_ttest(prior_accs, accuracies, n_training_folds, n_test_folds, correction=True)

                p_random_ttest = paired_ttest(random_accs, accuracies, n_training_folds, n_test_folds, correction=False)
                p_majority_ttest = paired_ttest(majority_accs, accuracies, n_training_folds, n_test_folds,
                                                correction=False)
                p_prior_ttest = paired_ttest(prior_accs, accuracies, n_training_folds, n_test_folds, correction=False)

                p_majority_wilcox = wilcoxon_signed_rank_test(majority_accs, accuracies)
                p_random_wilcox = wilcoxon_signed_rank_test(random_accs, accuracies)
                p_prior_wilcox = wilcoxon_signed_rank_test(prior_accs, accuracies)

                _, pvalue_single = fisher_exact(cm_single)
                confusion_matrix_sum = confusion_matrices.sum(axis=0)
                _, pvalue_sum = fisher_exact(confusion_matrix_sum)
                p_values = np.array([fisher_exact(cm)[1] for cm in confusion_matrices])
                pvalue_mean = np.mean(p_values)
                pvalue_median = np.median(p_values)
                logger.info("P-values {}".format(p_values))

                reject, pvals_corrected, _, alpha = multipletests(p_values, 0.01, method='holm', is_sorted=False)
                logger.info("Holm Bonnferroni Rejected Hypothesis: {} min: {} max: {}".format(np.sum(reject),
                                                                                              np.min(pvals_corrected),
                                                                                              np.max(pvals_corrected)))

                logger.info(" Corrected p_values {}".format(pvals_corrected))
                final_pvals = [pvalue_single, pvalue_sum, pvalue_median, pvalue_mean, np.median(pvals_corrected),
                               p_random_cttest, p_majority_cttest, p_prior_cttest, p_random_ttest, p_majority_ttest,
                               p_prior_ttest, p_random_wilcox, p_majority_wilcox, p_prior_wilcox]
                vals = []
                for k in METRICS:
                    v = scores[k]
                    vals.extend([np.mean(v).round(4), np.std(v).round(4)])
                d = dict(zip(cols_metrics + cols_pvals, vals + final_pvals))
                logger.info("Classifier {}, Metrics {}".format(cls_name, print_dictionary(d)))
                one_row = [label, fold_id, cls_name] + vals + final_pvals
                final.append(one_row)

    end = datetime.now()
    total = (end - start).total_seconds()
    logger.info("Time taken for finishing the p-value calculation is {} seconds and "
                "{} hours".format(total, total / 3600))

    data_frame = pd.DataFrame(final, columns=columns)
    data_frame['rank'] = data_frame[MODEL].map(custom_dict)
    data_frame.sort_values(by=[DATASET, 'rank'], ascending=[True, True], inplace=True)
    del data_frame['rank']
    data_frame = pd.DataFrame(final, columns=columns)
    data_frame.to_csv(df_file_path)

    for pval_col in cols_pvals:
        data_frame[pval_col + '-rejected'] = False

    final = []
    for label, j in synthetic_dataset_reader.label_mapping.items():
        for fold_id in range(synthetic_dataset_reader.n_datasets_per_parameter):
            one_row = [label, fold_id]
            for pval_col in cols_pvals:
                p_vals, pvals_corrected, reject = holm_bonferroni(data_frame, label, fold_id, pval_col=pval_col)
                data_frame.loc[(data_frame[DATASET] == label) & (data_frame[FOLD_ID] == fold_id), [
                    pval_col + '-rejected']] = reject
                # print(label, pval_col, reject)
                # print(data_frame[data_frame[DATASET] == label][[pval_col + '-corrected', pval_col + '-rejected']])
                # print('##############################################################################')
                one_row.extend([np.any(reject), np.sum(reject)])
                if np.any(reject):
                    vulnerable_classes[pval_col].append("dataset {}, fold {}".format(label, fold_id))
                    logger.info("Adding class {} for P-val {}".format(label, pval_col))
                    # print(print_dictionary(vulnerable_classes))
            df = data_frame[data_frame[MODEL].str.contains('|'.join(list(custom_dict.keys())[-5:-2]))]
            logger.info("Dataframe {}".format(df[(df[DATASET] == label) & (df[FOLD_ID] == fold_id)][[MODEL,
                                                  CTTEST_PVAL + '-random', CTTEST_PVAL + '-majority', CTTEST_PVAL + '-prior']]))
            p_vals_r = df[(df[DATASET] == label) & (df[FOLD_ID] == fold_id)][CTTEST_PVAL + '-random'].values
            p_vals_m = df[(df[DATASET] == label) & (df[FOLD_ID] == fold_id)][CTTEST_PVAL + '-majority'].values
            p_vals_p = df[(df[DATASET] == label) & (df[FOLD_ID] == fold_id)][CTTEST_PVAL + '-prior'].values
            logger.info("P-val Random {}, P-val Majority {}, P-val Prior {}".format(p_vals_r, p_vals_m, p_vals_p))
            one_row.extend([np.min(p_vals_r), np.any(p_vals_r < 0.01)])
            one_row.extend([np.min(p_vals_m), np.any(p_vals_m < 0.01)])
            one_row.extend([np.min(p_vals_p), np.any(p_vals_p < 0.01)])

            logger.info("Appending Row".format(one_row))
            final.append(one_row)

    logger.info(print_dictionary(vulnerable_classes))
    data_frame['rank'] = data_frame[MODEL].map(custom_dict)
    data_frame.sort_values(by=[DATASET, 'rank'], ascending=[True, True], inplace=True)
    del data_frame['rank']
    data_frame.to_csv(df_file_path)

    cols = list(np.array([[c, c + '-count'] for c in cols_pvals]).flatten())
    columns = [DATASET, FOLD_ID] + cols
    columns = columns + ["DL-random-pval", "DL-random", "DL-majority-pval", "DL-majority", "DL-prior-pval", "DL-prior"]
    data_frame = pd.DataFrame(final, columns=columns)
    data_frame.sort_values(by=[DATASET, FOLD_ID], ascending=[True, True], inplace=True)
    data_frame.to_csv(df_result_file_path)

    with open(vulnerable_file, "wb") as class_file:
        pickle.dump(vulnerable_classes, class_file)

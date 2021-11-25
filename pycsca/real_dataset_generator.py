import inspect
import logging
import os

import numpy as np
from sklearn.utils import check_random_state

from .csv_reader import CSVReader
from .synthetic_dataset_reader import SyntheticDatasetGenerator, WEIGHT_LABEL


class RealDatasetGenerator(SyntheticDatasetGenerator):
    def __init__(self, dataset_per_parameter=10, dataset_function_type='cluster', experiment_setup="independent",
                 random_state=None, **kwargs):
        DIR_PATH = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
        self.class_label = 1
        self.real_vulnerable = os.path.join(DIR_PATH, 'datasets', '2021-06-06-openssl0_9_7aserver')
        self.real_non_vulnerable = os.path.join(DIR_PATH, 'datasets', '2021-06-06-openssl0_9_7bserver')
        self.csv_reader_nv = CSVReader(folder=self.real_non_vulnerable)
        self.csv_reader_v = CSVReader(folder=self.real_vulnerable)
        self.logger = logging.getLogger(RealDatasetGenerator.__name__)
        self.logger.info('Vulnerable {} Non-Vulnerable {}'.format(self.csv_reader_v, self.csv_reader_nv))
        super(RealDatasetGenerator, self).__init__(dataset_per_parameter=dataset_per_parameter,
                                                   experiment_setup=experiment_setup,
                                                   dataset_function_type=dataset_function_type,
                                                   random_state=random_state,
                                                   **kwargs)

    def mk_dataset(self, dataset_params, random=False):
        if random:
            x, y = self.csv_reader_nv.get_data_class_label(class_label=self.class_label, missing_ccs_fin=False)
            self.logger.info("Working with the folder {}".format(self.csv_reader_nv.dataset_folder))
        else:
            x, y = self.csv_reader_v.get_data_class_label(class_label=self.class_label, missing_ccs_fin=False)
            self.logger.info("Working with the folder {}".format(self.csv_reader_v.dataset_folder))
        random_state = check_random_state(seed=dataset_params['random_state'])
        p = dataset_params['weights'][0]
        n_0 = int(p * self.total_instances)
        n_1 = self.total_instances - n_0
        self.logger.info("--------p {}, n_0 {}, n_1 {}-------".format(p, n_0, n_1))
        ind0 = random_state.choice(np.where(y == 0)[0], n_0)
        ind1 = random_state.choice(np.where(y == 1)[0], n_1)
        indx = np.concatenate((ind0, ind1))
        random_state.shuffle(indx)
        x, y = x[indx], y[indx]
        return x, y

    def make_gp_dataset(self, dataset_params, random=False):
        raise NotImplementedError

    def noise_variation_datasets(self, seed, label1_weight=0.45, n_splits=3, **kwargs):
        raise NotImplementedError

    def independent_dataset_diff_sizes(self, seed, label1_weight=0.45, n_splits=3, **kwargs):
        raise NotImplementedError

    def dependent_dataset_diff_sizes(self, seed, label1_weight=0.45, n_splits=3, **kwargs):
        raise NotImplementedError

    def get_vulnerable_folds(self, seed, label1_weight=0.45, fold_id=1, **kwargs):
        x, y = self.csv_reader_v.get_data_class_label(class_label=self.class_label, missing_ccs_fin=False)
        random_state = check_random_state(seed=seed)
        _, test_index = list(self.cv_iterator.split(x, y))[fold_id]
        X_test, y_test = x[test_index], y[test_index]
        if label1_weight > 0.49:
            p = (y.sum() / y.shape[0])
            start = 100
            end = 6000
        else:
            p = label1_weight
            start = np.round((2 * 20 / label1_weight) + 100, -2)
            end = 6000
        datasets = {}
        v_label = "Vulnerable"
        self.logger.info("Sampling Vulnerable real dataset {} {}".format(WEIGHT_LABEL, label1_weight))
        for total_instances in list(np.arange(start, end + 10, step=100)):
            label = "{} Total Instances: {}".format(v_label, int(total_instances))
            self.labels.add(label)
            x, y = self.get_indices(p, random_state, total_instances, X_test, y_test)
            datasets[label] = (x, y)

        self.logger.info("For Complete Dataset")
        if label1_weight < 0.49:
            ind0 = np.where(y_test == 0)[0]
            n_0 = len(ind0)
            n_1 = int(p / (1 - p) * n_0)
            self.logger.info("--------p {}, n_0 {}, n_1 {}-------".format(p, n_0, n_1))
            ind1 = random_state.choice(np.where(y_test == 1)[0], n_1)
            indx = np.concatenate((ind0, ind1))
            random_state.shuffle(indx)
            total_instances = X_test[indx].shape[0]
            label = "{} Total Instances: {}".format(v_label, np.round(total_instances, -3))
            datasets[label] = X_test[indx], y_test[indx]
            self.labels.add(label)
        else:
            total_instances = X_test.shape[0]
            ind0 = np.where(y_test == 0)[0]
            n_0 = len(ind0)
            n_1 = total_instances - n_0
            self.logger.info("--------p {}, n_0 {}, n_1 {}-------".format(p, n_0, n_1))
            self.logger.info("--------Total Instances {}-------".format(total_instances))
            label = "{} Total Instances: {}".format(v_label, np.round(total_instances, -3))
            datasets[label] = X_test, y_test
            self.labels.add(label)
        return datasets

    def get_non_vulnerable_folds(self, seed, label1_weight=0.45, fold_id=1, **kwargs):
        x, y = self.csv_reader_nv.get_data_class_label(class_label=self.class_label, missing_ccs_fin=False)
        random_state = check_random_state(seed=seed)
        _, test_index = list(self.cv_iterator.split(x, y))[fold_id]
        X_test, y_test = x[test_index], y[test_index]
        if label1_weight > 0.49:
            p = (y.sum() / y.shape[0])
            start = 100
            end = 6000
        else:
            p = label1_weight
            start = np.round((2 * 20 / label1_weight) + 100, -2)
            end = 6000
        datasets = {}
        self.logger.info("Sampling Non-Vulnerable real dataset {} {}".format(WEIGHT_LABEL, label1_weight))
        v_label = "Non-Vulnerable"
        for total_instances in list(np.arange(start, end + 10, step=100)):
            label = "{} Total Instances: {}".format(v_label, int(total_instances))
            self.labels.add(label)
            x, y = self.get_indices(p, random_state, total_instances, X_test, y_test)
            datasets[label] = (x, y)

        self.logger.info("For Complete Dataset")
        if label1_weight < 0.49:
            ind0 = np.where(y_test == 0)[0]
            n_0 = len(ind0)
            n_1 = int(p / (1 - p) * n_0)
            self.logger.info("--------p {}, n_0 {}, n_1 {}-------".format(p, n_0, n_1))
            ind1 = random_state.choice(np.where(y_test == 1)[0], n_1)
            indx = np.concatenate((ind0, ind1))
            random_state.shuffle(indx)
            total_instances = X_test[indx].shape[0]
            self.logger.info("--------Total Instances {}-------".format(total_instances))
            label = "{} Total Instances: {}".format(v_label, np.round(total_instances, -3))
            datasets[label] = X_test[indx], y_test[indx]
            self.labels.add(label)
        else:
            total_instances = X_test.shape[0]
            ind0 = np.where(y_test == 0)[0]
            n_0 = len(ind0)
            n_1 = total_instances - n_0
            self.logger.info("--------p {}, n_0 {}, n_1 {}-------".format(p, n_0, n_1))
            self.logger.info("--------Total Instances {}-------".format(total_instances))
            label = "{} Total Instances: {}".format(v_label, np.round(total_instances, -3))
            datasets[label] = X_test, y_test
            self.labels.add(label)
        return datasets

    def get_indices(self, p, random_state, total_instances, X_test, y_test):
        n_0 = int((1 - p) * total_instances)
        n_1 = int(total_instances - n_0)
        self.logger.info("--------p {}, n_0 {}, n_1 {}-------".format(p, n_0, n_1))
        ind0 = random_state.choice(np.where(y_test == 0)[0], n_0)
        ind1 = random_state.choice(np.where(y_test == 1)[0], n_1)
        indx = np.concatenate((ind0, ind1))
        random_state.shuffle(indx)
        x, y = X_test[indx], y_test[indx]
        return x, y

import logging
import numpy as np
from abc import ABCMeta
from sklearn.datasets import make_classification
from sklearn.gaussian_process.kernels import Matern
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import check_random_state

from .utils import print_dictionary

WEIGHT_LABEL = "Class-Label 1 Weight"

MIN_LABEL_WEIGHT = 0.01


class SyntheticDatasetGenerator(metaclass=ABCMeta):
    def __init__(self, dataset_per_parameter=5, experiment_setup="independent", dataset_function_type='cluster',
                 random_state=None, **kwargs):
        self.random_state = check_random_state(random_state)
        self.logger = logging.getLogger(SyntheticDatasetGenerator.__name__)
        self.n_datasets_per_parameter = dataset_per_parameter
        self.labels = set()
        self.datasets = []
        self.es_function_options = {"independent": self.independent_datasets, "dependent": self.dependent_datasets,
                                    "noise_variation": self.noise_variation_datasets,
                                    "independent_diff_sizes": self.independent_dataset_diff_sizes,
                                    "dependent_diff_sizes": self.dependent_dataset_diff_sizes,
                                    'vulnerable': self.get_vulnerable_folds,
                                    'non-vulnerable': self.get_non_vulnerable_folds}
        if experiment_setup not in self.es_function_options.keys():
            raise ValueError
        self.experimental_setup_function = self.es_function_options[experiment_setup]

        self.dataset_function_options = {"cluster": self.mk_dataset, "gp_dataset": self.make_gp_dataset}
        if experiment_setup not in self.es_function_options.keys():
            raise ValueError
        if dataset_function_type not in self.dataset_function_options.keys():
            raise ValueError
        self.experimental_setup_function = self.es_function_options[experiment_setup]
        self.dataset_function = self.dataset_function_options[dataset_function_type]
        self.logger.info("Dataset Function {}".format(self.dataset_function))
        self.logger.info("Experimental Setup {}".format(self.experimental_setup_function))

        self.logger.info("Key word arguments {}".format(kwargs))
        self.logger.info('############################################################')
        self.logger.info(experiment_setup.title())
        self.total_instances = None
        self.cv_iterator = StratifiedKFold(n_splits=self.n_datasets_per_parameter, random_state=42)
        self.__load__dataset(**kwargs)

    def __load__dataset(self, **kwargs):
        for i in range(self.n_datasets_per_parameter):
            self.logger.info('#####################  Fold ID {} #####################'.format(i))
            seed = self.random_state.randint(2 ** 32, dtype="uint32") + i
            datasets = self.experimental_setup_function(seed=seed, fold_id=i, **kwargs)
            self.datasets.append(datasets)
        label_encoder = LabelEncoder()
        label_encoder.fit_transform(list(self.labels))
        self.label_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
        self.labels = list(self.label_mapping.keys())
        self.n_labels = len(self.label_mapping)
        self.logger.info("Labels {}".format(self.labels))
        self.logger.info("Label Mapping {}".format(print_dictionary(self.label_mapping)))

    def get_n_instances(self, n_splits, weights):
        return np.max([int(2 * n_splits / np.min(weights)), 200])

    def mk_dataset(self, dataset_params, random=False):
        dataset_params['n_samples'] = int(2 * self.total_instances)
        X, y = make_classification(**dataset_params)
        p = dataset_params['weights'][0]
        n_0 = int(p * self.total_instances)
        n_1 = self.total_instances - n_0
        self.logger.info("--------p {}, n_0 {}, n_1 {}, flip_y {}-------".format(p, n_0, n_1, dataset_params['flip_y']))
        if random:
            indx = self.random_state.choice(np.where(y == 1)[0], n_0 + n_1)
            X, y = X[indx], y[indx]
            y = self.random_state.choice([0, 1], p=[p, 1 - p], size=self.total_instances)
        else:
            ind0 = self.random_state.choice(np.where(y == 0)[0], n_0)
            ind1 = self.random_state.choice(np.where(y == 1)[0], n_1)
            indx = np.concatenate((ind0, ind1))
            self.random_state.shuffle(indx)
            X, y = X[indx], y[indx]
        return X, y

    def make_gp_dataset(self, dataset_params, random=False):
        """Creates a nonlinear object ranking problem by sampling from a
        Gaussian process as the latent utility function.
        Note that this function needs to compute a kernel matrix of size
        (n_instances * n_objects) ** 2, which could allocate a large chunk of the
        memory."""

        n_features = 10
        random_state = dataset_params['random_state']
        p = dataset_params['weights'][0]
        n_0 = int(p * self.total_instances)
        n_1 = self.total_instances - n_0
        kernel_params = dict()
        noise = dataset_params['flip_y']
        n_instances = int(2 * self.total_instances)
        X = random_state.rand(n_instances, n_features)
        L = np.linalg.cholesky(Matern(**kernel_params)(X))
        f = L.dot(random_state.randn(n_instances)) + random_state.normal(scale=noise, size=n_instances)
        X = X.reshape(self.total_instances, n_features)
        if random:
            Y = self.random_state.choice([0, 1], p=[p, 1 - p], size=n_instances)
            ind0 = self.random_state.choice(np.where(Y == 0)[0], n_0)
            ind1 = self.random_state.choice(np.where(Y == 1)[0], n_1)
            indx = np.concatenate((ind0, ind1))
            self.random_state.shuffle(indx)
            X, Y = X[indx], Y[indx]
        else:
            Y = np.array((np.exp(f) / (1 + np.exp(f))) > 1 - p, dtype=int)
            ind0 = self.random_state.choice(np.where(Y == 0)[0], n_0)
            ind1 = self.random_state.choice(np.where(Y == 1)[0], n_1)
            indx = np.concatenate((ind0, ind1))
            self.random_state.shuffle(indx)
            X, y = X[indx], Y[indx]
        return X, Y

    def independent_datasets(self, seed, n_splits=3, **kwargs):
        dataset_params = dict(n_features=10, n_informative=2, n_redundant=0, weights=[0.5, 0.5],
                              n_clusters_per_class=2, flip_y=1.0, random_state=seed)
        datasets = {}
        weights = [1 - MIN_LABEL_WEIGHT, MIN_LABEL_WEIGHT]
        self.total_instances = self.get_n_instances(n_splits, weights)
        self.logger.info("Total Instances {}".format(self.total_instances))
        for p in np.arange(MIN_LABEL_WEIGHT, 0.52, step=0.02):
            label = "Independent {}: {}".format(WEIGHT_LABEL, p.round(2))
            self.logger.info("**************** Generating data for label {} ****************".format(label))
            dataset_params['weights'] = [1 - p, p]
            X, Y = self.mk_dataset(dataset_params, random=True)
            self.labels.add(label)
            datasets[label] = (X, Y)
        return datasets

    def dependent_datasets(self, seed, n_splits=3, **kwargs):
        dataset_params = dict(n_features=10, n_informative=10, n_redundant=0, weights=[0.5, 0.5],
                              n_clusters_per_class=2, flip_y=0.01, random_state=seed)
        datasets = {}
        weights = [1 - MIN_LABEL_WEIGHT, MIN_LABEL_WEIGHT]
        self.total_instances = self.get_n_instances(n_splits, weights)
        self.logger.info("Total Instances {}".format(self.total_instances))
        for p in np.arange(MIN_LABEL_WEIGHT, 0.52, step=0.02):
            label = "Dependent {}: {}".format(WEIGHT_LABEL, p.round(2))
            self.logger.info("**************** Generating data for label {} ****************".format(label))
            p0 = 1 - p
            p1 = p
            dataset_params['weights'] = [p0, p1]
            X, Y = self.mk_dataset(dataset_params, random=False)
            self.labels.add(label)
            datasets[label] = (X, Y)
        return datasets

    def noise_variation_datasets(self, seed, label1_weight=0.45, n_splits=3, **kwargs):
        weights = [1 - label1_weight, label1_weight]
        self.total_instances = self.get_n_instances(n_splits, weights)
        self.logger.info("Total Instances {}".format(self.total_instances))
        dataset_params = dict(n_features=10, n_informative=10, n_redundant=0,
                              weights=[1 - label1_weight, label1_weight],
                              n_clusters_per_class=2, flip_y=0.01, random_state=seed)
        datasets = {}
        for flip in np.linspace(0.01, 1.0, num=30):
            label = 'Noise {}: {}: {}'.format(flip.round(4), WEIGHT_LABEL, label1_weight)
            self.logger.info("**************** Generating data for label {} ****************".format(label))
            dataset_params['flip_y'] = flip
            X, Y = self.mk_dataset(dataset_params, random=False)
            self.labels.add(label)
            datasets[label] = (X, Y)
        return datasets

    def independent_dataset_diff_sizes(self, seed, label1_weight=0.45, n_splits=3, **kwargs):
        weights = [1 - label1_weight, label1_weight]
        dataset_params = dict(n_features=10, n_informative=2, n_redundant=0, weights=weights,
                              n_clusters_per_class=2, flip_y=1.0, random_state=seed)
        start = self.get_n_instances(n_splits, weights)
        datasets = {}
        end = 5000
        for total_instances in list(np.arange(start, end + 10, step=100)):
            self.total_instances = total_instances
            label = "Independent Total Instances: {} {}: {}".format(self.total_instances, WEIGHT_LABEL, label1_weight)
            self.logger.info("**************** Generating data for label {} ****************".format(label))
            X, Y = self.mk_dataset(dataset_params, random=True)
            self.labels.add(label)
            datasets[label] = (X, Y)
        return datasets

    def dependent_dataset_diff_sizes(self, seed, label1_weight=0.45, n_splits=3, **kwargs):
        weights = [1 - label1_weight, label1_weight]
        dataset_params = dict(n_features=10, n_informative=10, n_redundant=0, weights=weights,
                              n_clusters_per_class=2, flip_y=0.01, random_state=seed)
        start = self.get_n_instances(n_splits, weights)
        datasets = {}
        end = 5000
        for total_instances in list(np.arange(start, end + 10, step=100)):
            self.total_instances = total_instances
            label = "Dependent Total Instances: {} {}: {}".format(self.total_instances, WEIGHT_LABEL, label1_weight)
            self.logger.info("**************** Generating data for label {} ****************".format(label))
            X, Y = self.mk_dataset(dataset_params, random=False)
            self.labels.add(label)
            datasets[label] = (X, Y)
        return datasets

    def get_vulnerable_folds(self, seed, fold_id=1, **kwargs):
        raise NotImplementedError

    def get_non_vulnerable_folds(self, seed, fold_id=1, **kwargs):
        raise NotImplementedError

    def get_data_class_label(self, class_label):
        datasets = []
        for i in range(self.n_datasets_per_parameter):
            datasets.append(self.datasets[i][class_label])
        return datasets

    def create_permutations_data(self, Y, fold_id, n_samples=100000):
        Ys = {}
        Yi = np.copy(Y)
        for i in range(n_samples):
            seed = self.random_state.randint(2 ** 30, dtype="uint32") + fold_id + i
            random_state = check_random_state(seed)
            random_state.shuffle(Yi)
            Ys[i] = Yi
        return Ys

    def create_bernoulli_data(self, Y, fold_id, n_samples=100000):
        Ys = {}
        n_instances = Y.shape[0]
        classes_ = np.unique(Y)
        n_classes = len(classes_)
        class_probabilities = np.zeros(n_classes) + 1 / n_classes
        for i in np.unique(classes_):
            class_probabilities[i] = len(Y[Y == i]) / len(Y)
        for i in range(n_samples):
            seed = self.random_state.randint(2 ** 30, dtype="uint32") + fold_id + i
            random_state = check_random_state(seed)
            Ys[i] = random_state.choice(classes_, p=class_probabilities, size=n_instances)
        return Ys

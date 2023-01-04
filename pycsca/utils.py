import inspect
import logging
import os
import sys
import warnings

import numpy as np
import pandas as pd
import scipy.stats as st
from sklearn.preprocessing import StandardScaler

from tensorflow.keras.optimizers import Adam, SGD

warnings.filterwarnings('ignore')

__all__ = ['create_dir_recursively', 'setup_logging', 'progress_bar', 'print_dictionary', 'str2bool',
           'standardize_features', 'standardize_features']



def progress_bar(count, total, status=''):
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))

    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    sys.stdout.write('[%s] %s/%s ...%s\r' % (bar, count, total, status))
    sys.stdout.flush()


def print_dictionary(dictionary, sep='\n'):
    output = "\n"
    for key, value in dictionary.items():
        output = output + str(key) + " => " + str(value) + sep
    return output


def create_dir_recursively(path, is_file_path=False):
    if is_file_path:
        path = os.path.dirname(path)
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def standardize_features(x_train, x_test):
    standardize = Standardize()
    x_train = standardize.fit_transform(x_train)
    x_test = standardize.transform(x_test)
    return x_train, x_test


class Standardize(object):
    def __init__(self, scalar=StandardScaler):
        self.scalar = scalar
        self.n_features = None
        self.scalars = dict()

    def fit(self, X):
        if isinstance(X, dict):
            self.n_features = list(X.keys())
            for k, x in X.items():
                scalar = self.scalar()
                self.scalars[k] = scalar.fit(x)
        if isinstance(X, (np.ndarray, np.generic)):
            self.scalar = self.scalar()
            self.scalar.fit(X)
            self.n_features = X.shape[-1]

    def transform(self, X):
        if isinstance(X, dict):
            for n in self.n_features:
                X[n] = self.scalars[n].transform(X[n])
        if isinstance(X, (np.ndarray, np.generic)):
            X = self.scalar.transform(X)
        return X

    def fit_transform(self, X):
        self.fit(X)
        X = self.transform(X)
        return X


def setup_logging(log_path=None, level=logging.DEBUG):
    """Function setup as many logging for the experiments"""
    if log_path is None:
        dirname = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
        dirname = os.path.dirname(dirname)
        log_path = os.path.join(dirname, "experiments", "logs", "logs.log")
        create_dir_recursively(log_path, True)
    logging.basicConfig(filename=log_path, level=level,
                        format='%(asctime)s %(name)s %(levelname)-8s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')
    logger = logging.getLogger("SetupLogger")
    logger.info("log file path: {}".format(log_path))
    logging.getLogger("matplotlib").setLevel(logging.ERROR)

    # logging.captureWarnings(True)


def str2bool(v):
    if int(v) > 0:
        v = 'true'
    return str(v).lower() in ("yes", "true", "t", "1")



def best_fit_distribution(data, logger, bins=200):
    """Model data by finding best fit distribution to data"""
    # Get histogram of original data
    y, x = np.histogram(data, bins=bins, density=True)
    x = (x + np.roll(x, -1))[:-1] / 2.0

    # Distributions to check
    DISTRIBUTIONS = [
        st.alpha, st.arcsine, st.beta, st.cauchy, st.chi, st.chi2,
        st.expon, st.exponnorm, st.exponweib, st.exponpow,
        st.genlogistic, st.genpareto, st.gennorm, st.genexpon,
        st.genextreme, st.gamma, st.gengamma, st.gumbel_r,
        st.gumbel_l, st.halfcauchy, st.halflogistic, st.halfnorm, st.halfgennorm, st.invgamma,
        st.invgauss, st.invweibull, st.johnsonsb, st.johnsonsu, st.laplace,
        st.logistic, st.loggamma, st.loglaplace, st.lognorm, st.lomax, st.maxwell,
        st.norm, st.pareto, st.pearson3, st.powerlognorm, st.powernorm,
        st.recipinvgauss, st.t,  st.truncnorm,
        st.uniform, st.vonmises, st.wald, st.weibull_min, st.weibull_max]

    # Best holders
    best_distribution = st.norm
    best_params = (0.0, 1.0)
    best_sse = np.inf
    final = []
    # Estimate distribution parameters from data
    for distribution in DISTRIBUTIONS:
        #logger.info("######################################################")
        # Try to fit the distribution
        #logger.info("{}".format(distribution.name))
        try:
            # fit dist to data
            params = distribution.fit(data)

            # Separate parts of parameters
            arg = params[:-2]
            loc = params[-2]
            scale = params[-1]

            # Calculate fitted PDF and error with fit in distribution
            pdf = distribution.pdf(x, loc=loc, scale=scale, *arg)
            sse = np.sum(np.power(y - pdf, 2.0))
            p = st.kstest(data, distribution.name, args=params)[1]

            # identify if this distribution is better
            if best_sse > sse > 0:
                best_distribution = distribution
                best_params = params
                best_sse = sse
            onerow = [distribution.name, params, p, sse]
            final.append(onerow)
            #logger.info("sse for {}: {}: params: {}".format(distribution.name, sse, params))
        except Exception:
            pass
    df = pd.DataFrame(final, columns=["Distribution", "Parameters", 'p', "sse"])
    logger.info("Best fitting distribution: " + str(best_distribution.name))
    logger.info("Best best_sse: " + str(best_sse))
    logger.info("Parameters for the best fit: " + str(best_params))
    return df, best_distribution.name, best_params


def check_file_exists(file_path):
    file_path = os.path.normpath(file_path)
    if not os.path.exists(file_path):
        print("Error: provided file path '%s' does not exist!" % file_path)
        sys.exit(-1)
    return


def get_optimizer(solver, learning_rate, beta_1, beta_2, epsilon, momentum, nesterovs_momentum):
    if solver.lower() == 'adam':
        optimizer = Adam(learning_rate=learning_rate, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon)
    elif solver.lower() == 'sgd':
        optimizer = SGD(learning_rate=learning_rate, momentum=momentum, nesterov=nesterovs_momentum)
    else:
        optimizer = None
        #logger.error('No suitable solver found for ' + solver)
        sys.exit(-1)

    return optimizer

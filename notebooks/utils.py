import inspect
import logging
import os
import sys

import numpy as np
import pandas as pd
import scipy.stats as st
from sklearn.preprocessing import MinMaxScaler

_all_ = ['get_best_distribution', 'get_data_from_files', 'setup_logging', 'create_dir_recursively', 'progress_bar',
           'normalize']


def create_dir_recursively(path, is_file_path=False):
    if is_file_path:
        path = os.path.dirname(path)
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def get_best_distribution(data, logger):
    dist_names = np.array([d for d in dir(st) if isinstance(getattr(st, d), st.rv_continuous)])
    dist_names = [d for d in dist_names if 'levy' not in d]
    # dist_names = ["norm", "exponweib", "weibull_max", "weibull_min", "pareto", "genextreme", 'beta', 'betaprime']
    # logger.info("dist_names {}".format(dist_names))
    dist_results = []
    params = {}
    for dist_name in dist_names:
        try:
            dist = getattr(st, dist_name)
            param = dist.fit(data)

            params[dist_name] = param

            # Applying the Kolmogorov-Smirnov test
            D, p = st.kstest(data, dist_name, args=param)
            logger.info("p value for {} = {}".format(dist_name, str(p)))
            dist_results.append((dist_name, p))
        except ValueError:
            params[dist_name] = ()
            dist_results.append((dist_name, 0.0))
    # select the best fitted distribution
    best_dist, best_p = (max(dist_results, key=lambda item: item[1]))
    # store the name of the best fit and its p value

    logger.info("Best fitting distribution: {}".format(best_dist))
    logger.info("Best p value: {}".format(str(best_p)))
    logger.info("Parameters for the best fit: {}".format(params[best_dist]))
    return best_dist, best_p, params[best_dist]


def get_data_from_files(files):
    data_frame = None
    for f in files:
        if data_frame is None:
            data_frame = pd.read_csv(f)
        else:
            df = pd.read_csv(f)
            data_frame = pd.concat([data_frame, df])
    data_frame.sort_values(by=['id'], inplace=True)
    return data_frame


def setup_logging(log_path=None, level=logging.DEBUG):
    """Function setup as many logging for the experiments"""
    if log_path is None:
        dirname = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
        dirname = os.path.dirname(dirname)
        log_path = os.path.join(dirname, "logs", "logs.log")
    create_dir_recursively(log_path, True)
    logging.basicConfig(filename=log_path, level=level,
                        format='%(asctime)s %(name)s %(levelname)-8s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')
    logger = logging.getLogger("SetupLogger")
    logger.info("log file path: {}".format(log_path))


def progress_bar(count, total, status=''):
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))

    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    sys.stdout.write('[%s] %s/%s ...%s\r' % (bar, count, total, status))
    sys.stdout.flush()


def normalize(data):
    scalar = MinMaxScaler()
    d = scalar.fit_transform(data[:, None]).flatten()
    if not np.any(np.isnan(d)):
        data = scalar.fit_transform(data[:, None]).flatten()
    return data, scalar
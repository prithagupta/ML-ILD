import logging
import os

from keras import Sequential
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Dense

from pycsca.utils import get_optimizer, check_file_exists

from sklearn.base import BaseEstimator, ClassifierMixin


class MultiLayerPerceptronTF(BaseEstimator, ClassifierMixin):
    def __init__(self, input_dim, n_classes, n_hidden=100, n_units=10, activation='relu',
                 loss='categorical_crossentropy', solver='adam', metrics=['accuracy'],
                 learning_rate=0.001, momentum=0.9, nesterovs_momentum=True,
                 epochs=75, batch_size=100, validation_fraction=0.1, early_stopping=False,
                 verbose=1, model_save_path='', beta_1=0.9, beta_2=0.999, epsilon=1e-08):
        self.input_dim = input_dim
        self.n_classes = n_classes
        self.n_units = n_units
        self.n_hidden = n_hidden
        self.activation = activation
        self.loss = loss
        self.solver = solver
        self.metrics = metrics
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.nesterovs_momentum = nesterovs_momentum
        self.epochs = epochs
        self.batch_size = batch_size
        self.validation_fraction = validation_fraction
        self.early_stopping = early_stopping
        self.verbose = verbose
        self.model_save_path = model_save_path
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.logger = logging.getLogger(name=MultiLayerPerceptronTF.__name__)
        self.model = self._construct_model()

    def fit(self, X, y):
        check_file_exists(os.path.dirname(self.model_save_path))

        save_model = ModelCheckpoint(self.model_save_path)
        callbacks = [save_model]

        if self.early_stopping:
            if self.validation_fraction == 0:
                self.validation_fraction = 0.1
            callbacks.append(EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True))

        self.model.fit(x=X, y=y, batch_size=self.batch_size, verbose=self.verbose,
                       validation_split=self.validation_fraction, epochs=self.epochs, callbacks=callbacks)

        return self

    def predict(self, X):
        predictions = self.model.predict(x=X, batch_size=self.batch_size, verbose=self.verbose)
        return predictions

    def score(self, X, y, sample_weight=None):
        model_metrics = self.model.evaluate(x=X, y=y, batch_size=self.batch_size, verbose=self.verbose)
        return model_metrics

    def predict_proba(self, X):
        prob_predictions = self.model.predict_proba(x=X, batch_size=self.batch_size, verbose=self.verbose)
        return prob_predictions

    def decision_function(self, X):
        prob_predictions = self.model.predict_proba(x=X, batch_size=self.batch_size, verbose=self.verbose)
        return prob_predictions

    def _construct_model(self):
        model = Sequential()

        # Input layer
        model.add(Dense(self.input_dim, activation=self.activation))

        # Hidden layers
        for i in range(self.n_hidden):
            model.add(Dense(self.n_units, activation=self.activation))

        # Output layer
        model.add(Dense(self.n_classes, activation='softmax'))

        # Initialize optimizer
        optimizer = get_optimizer(solver=self.solver, learning_rate=self.learning_rate, beta_1=self.beta_1,
                                  beta_2=self.beta_2, epsilon=self.epsilon, momentum=self.momentum,
                                  nesterovs_momentum=self.nesterovs_momentum)

        # Compile the model
        model.compile(loss=self.loss, optimizer=optimizer, metrics=self.metrics)

        return model

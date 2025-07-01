#!/usr/bin/env python
# coding: utf-8

# In[8]:


import tensorflow.keras as keras
import tensorflow as tf
import numpy as np
import time
import logging
from sklearn import metrics

from time_series_classifiers.deep_learning_classifiers.utils_functions import save_logs
from time_series_classifiers.deep_learning_classifiers.utils_functions import calculate_metrics
from utils.program_stats import timeit
from utils.util_functions import is_gpu_available

# Configure logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# In[18]:


import argparse
import configparser
import os
from pathlib import Path
import logging
import numpy as np
import getpass

from configobj import ConfigObj
from sklearn.preprocessing import OneHotEncoder
import tensorflow as tf

from data_processing.create_segments import TRAIN_DATASET_Y, TRAIN_DATASET_X, TEST_DATASET_X, TEST_DATASET_Y
from utils.math_funtions import get_combinations
from utils.sklearn_utils import report_average, plot_confusion_matrix
from utils.util_functions import create_directory_if_not_exists


# In[19]:


os.environ["CUDA_VISIBLE_DEVICES"]="0"

FILE_NAME_X = '{}_{}_X.npy'
FILE_NAME_Y = '{}_{}_Y.npy'
FILE_NAME_PID = '{}_{}_pid.npy'


# In[39]:


class ValAccuracyLogger(keras.callbacks.Callback):
    def __init__(self, x_val, y_val):
        super().__init__()
        self.x_val = x_val
        self.y_val = y_val

    def on_epoch_end(self, epoch, logs=None):
        val_loss, val_acc = self.model.evaluate(self.x_val, self.y_val, verbose=1)
        logger.info("Epoch %d: val_loss=%.4f, val_accuracy=%.4f", epoch + 1, val_loss, val_acc)


# In[11]:


class Classifier_FCN:

    def __init__(self, output_directory, exercise, input_shape, nb_classes,
                 learning_rate=0.001, patience=50, min_lr=1e-4,
                 batch_size=16, hidden_units=[128, 256, 128], num_layers=3,
                 verbose=True, build=True):

        self.output_directory = output_directory
        self.model_name = "fcn"
        self.exercise = exercise
        self.verbose = verbose
        self.learning_rate = learning_rate
        self.patience = patience
        self.min_lr = min_lr
        self.batch_size = batch_size
        self.hidden_units = hidden_units
        self.num_layers = num_layers

        if build:
            self.model = self.build_model(input_shape, nb_classes)
            if verbose:
                self.model.summary()
            self.model.save_weights(self.output_directory + 'model_init.weights.h5')


    def build_model(self, input_shape, nb_classes):
        input_layer = keras.layers.Input(shape=input_shape)
        x = input_layer

        for i in range(min(self.num_layers, len(self.hidden_units))):
            x = keras.layers.Conv1D(self.hidden_units[i], kernel_size=8 if i == 0 else 3, padding='same')(x)
            x = keras.layers.BatchNormalization()(x)
            x = keras.layers.ReLU()(x)

        x = keras.layers.GlobalAveragePooling1D()(x)
        output_layer = keras.layers.Dense(nb_classes, activation='softmax')(x)

        model = keras.models.Model(inputs=input_layer, outputs=output_layer)
        optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate)
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        return model

    @timeit
    def fit(self, x_train, y_train, x_val, y_val, nb_epochs):
        if not is_gpu_available():
            logger.error('GPU not available')
            exit()

        mini_batch_size = int(min(x_train.shape[0] / 10, self.batch_size))

        reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                                      patience=self.patience, min_lr=self.min_lr)
        checkpoint_path = self.output_directory + 'best_model.keras'
        model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                           monitor='val_loss', save_best_only=True)
        val_logger = ValAccuracyLogger(x_val, y_val)

        callbacks = [reduce_lr, model_checkpoint, val_logger]

        logger.info("Starting training for %d epochs", nb_epochs)
        history = self.model.fit(x_train, y_train,
                                 batch_size=mini_batch_size,
                                 epochs=nb_epochs,
                                 verbose=self.verbose,
                                 validation_data=(x_val, y_val),
                                 callbacks=callbacks)

        logger.info("Training complete. Saving last model.")
        self.model.save(self.output_directory + 'last_model.keras')
        keras.backend.clear_session()
        return history

    @timeit
    def predict(self, x_test, y_true, enc):
        model_path = self.output_directory + 'best_model.keras'
        model = keras.models.load_model(model_path)
        y_pred = model.predict(x_test)
        y_pred_idx = np.argmax(y_pred, axis=1)
        y_pred_label = np.array([enc.categories_[0][i] for i in y_pred_idx])
        y_true_label = np.array([enc.categories_[0][i] for i in y_true])
        conf_matrix = metrics.confusion_matrix(y_true_label, y_pred_label)
        class_report = metrics.classification_report(y_true_label, y_pred_label)

        logger.info("Confusion Matrix:\n%s", conf_matrix)
        logger.info("Classification Report:\n%s", class_report)

        df_metrics = calculate_metrics(y_true, y_pred_idx, 0.0)
        return conf_matrix, class_report, df_metrics


# In[14]:





# In[16]:





# In[23]:


def read_dataset(data_path, data_type, dataset_name):
    """
    Function to read the data in numpy format to be used for classification
    :param data_path:
    :param dataset_name:
    :return:
    """
    datasets_dict = {}
    x_train = np.load(os.path.join(data_path, FILE_NAME_X.format("TRAIN", data_type)), allow_pickle=True)
    y_train = np.load(os.path.join(data_path, FILE_NAME_Y.format("TRAIN", data_type)), allow_pickle=True)

    x_val = np.load(os.path.join(data_path, FILE_NAME_X.format("VAL", data_type)), allow_pickle=True)
    y_val = np.load(os.path.join(data_path, FILE_NAME_Y.format("VAL", data_type)), allow_pickle=True)

    x_test = np.load(os.path.join(data_path, FILE_NAME_X.format("TEST", data_type)), allow_pickle=True)
    y_test = np.load(os.path.join(data_path, FILE_NAME_Y.format("TEST", data_type)), allow_pickle=True)
    datasets_dict[dataset_name] = (x_train.copy(), y_train.copy(), x_test.copy(), y_test.copy(), x_val.copy(), y_val.copy())
    logger.info("Data shape is: ")
    logger.info("Train data shape: {} {} | Val data shape: {} {} | Test data shape:"
                " {} {}".format(x_train.shape, y_train.shape, x_val.shape, y_val.shape,
                                x_test.shape, y_test.shape))
    return datasets_dict


# In[30]:


data_type = "default_type"
home_path = str(Path.home())
exercise = "MP"
input_data_path = "TrainTestData"
base_path = "Research/Data/HumanPoseEstimation"
seed_value = "103007"
combination = "MulticlassSplit"
dataset_name = "MulticlassSplit"
classifier_type = "fcn"
output_path = os.path.join(home_path, "Results/Output")
base_path = os.path.join(home_path, base_path, exercise)
input_data_path = os.path.join(base_path, input_data_path)

data_path = os.path.join(input_data_path, seed_value, combination)
datasets_dict = read_dataset(data_path, data_type, combination)


# In[31]:


weights_directory = os.path.join(output_path, 'dl_weights', exercise, classifier_type, combination,
                                         seed_value)
create_directory_if_not_exists(weights_directory)


# In[28]:


x_train, y_train = datasets_dict[dataset_name][0], datasets_dict[dataset_name][1]

x_val, y_val = datasets_dict[dataset_name][4], datasets_dict[dataset_name][5]

x_test, y_test = datasets_dict[dataset_name][2], datasets_dict[dataset_name][3]

all_labels = np.concatenate((y_train, y_test), axis=0)
nb_classes = len(np.unique(np.concatenate((y_train, y_test), axis=0)))

# transform the labels from integers to one hot vectors
# Encode labels if not already one-hot
if y_train.ndim == 1 or y_train.shape[1] == 1:
    enc = OneHotEncoder(categories='auto')
    enc.fit(all_labels.reshape(-1, 1))
    y_train = enc.transform(y_train.reshape(-1, 1)).toarray()
    y_test = enc.transform(y_test.reshape(-1, 1)).toarray()
    y_val = enc.transform(y_val.reshape(-1, 1)).toarray()
else:
    enc = None  # Assume already one-hot

# save orignal y because later we will use binary
y_true = np.argmax(y_test, axis=1)

if len(x_train.shape) == 2:  # if univariate
    # add a dimension to make it multivariate with one dimension
    x_train = x_train[..., np.newaxis]
    x_test = x_test[..., np.newaxis]
    x_val = x_val[..., np.newaxis]

input_shape = x_train.shape[1:]


# In[32]:


def run_grid_search(config_list, x_train, y_train, x_val, y_val, nb_epochs):
    results = []
    for config in config_list:
        logger.info("Running config: %s", config)

        model = create_fcn_from_config(config)
        history = model.fit(x_train, y_train, x_val, y_val, nb_epochs)
        results.append({"config": config, "history": history.history})
    return results


# In[35]:


config_list = [
    {
        "output_directory": output_path,
        "exercise": "MP",
        "input_shape": (161, 16),
        "nb_classes": 4,
        "learning_rate": lr,
        "batch_size": bs,
        "hidden_units": hu,
        "num_layers": nl,
        "patience": 30,
        "min_lr": 1e-5,
        "verbose": True
    }
    for lr in [0.001, 0.0005, 0.0001]
    for bs in [16, 32]
    for hu in [[64, 128, 64], [128, 256, 128]]
    for nl in [2, 3]
]


# In[36]:


def create_fcn_from_config(config):
    return Classifier_FCN(
        output_directory=config.get("output_directory", "./"),
        exercise=config.get("exercise", "default"),
        input_shape=config["input_shape"],
        nb_classes=config["nb_classes"],
        learning_rate=config.get("learning_rate", 0.001),
        patience=config.get("patience", 50),
        min_lr=config.get("min_lr", 1e-4),
        batch_size=config.get("batch_size", 16),
        hidden_units=config.get("hidden_units", [128, 256, 128]),
        num_layers=config.get("num_layers", 3),
        verbose=config.get("verbose", True),
        build=True
    )


# In[37]:


nb_epochs = 600
results = run_grid_search(config_list, x_train, y_train, x_val, y_val, nb_epochs)


# In[38]:


results


# In[ ]:





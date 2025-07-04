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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
DL_METHODS = ["fcn", "cnn", "resnet"]
os.environ["CUDA_VISIBLE_DEVICES"]="0"

FILE_NAME_X = '{}_{}_X.npy'
FILE_NAME_Y = '{}_{}_Y.npy'
FILE_NAME_PID = '{}_{}_pid.npy'


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


def fit_classifier(classifier_name, datasets_dict, dataset_name, exercise):
    """
    Function to train the model on the data
    :param classifier_name:
    :param datasets_dict:
    :param dataset_name:
    :param exercise:
    :return:
    """
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
    # Create classifier instance
    classifier = create_classifier(classifier_name, input_shape, nb_classes, weights_directory, exercise)

    # Train the model
    classifier.fit(x_train, y_train, x_val, y_val, nb_epochs=nb_epochs)

    # Predict and evaluate
    confusion_matrix, class_report, df_metrics = classifier.predict(x_test, y_true, enc)

    classification_report_list.append(class_report)
    labels = list(enc.categories_[0]) if enc else list(range(nb_classes))
    plot_confusion_matrix(weights_directory, seed_value, confusion_matrix, labels)

    logger.info("Metrics summary: %s", df_metrics)


def create_classifier(classifier_name, input_shape, nb_classes, output_directory, exercise, verbose=True):
    """
    Function to create the classifier and store it in a dictionary
    :param classifier_name:
    :param input_shape:
    :param nb_classes:
    :param output_directory:
    :param exercise:
    :param verbose:
    :return:
    """
    if classifier_name == 'fcn':
        from time_series_classifiers.deep_learning_classifiers.classifiers import fcn
        return fcn.Classifier_FCN(output_directory, exercise, input_shape, nb_classes)
    if classifier_name == 'resnet':
        from time_series_classifiers.deep_learning_classifiers.classifiers import resnet
        return resnet.Classifier_RESNET(output_directory, exercise, input_shape, nb_classes)
    if classifier_name == 'cnn':
        from time_series_classifiers.deep_learning_classifiers.classifiers import cnn
        return cnn.Classifier_CNN(output_directory, exercise, input_shape, nb_classes)
    # if classifier_name == 'inception':
    #     from deep_learning_classifiers.classifiers import inception
    #     return inception.Classifier_INCEPTION(output_directory, input_shape, nb_classes, verbose)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dl_config", required=True, help="path of the config file")
    parser.add_argument("--exercise_config", required=True, help="path of the config file")
    args = parser.parse_args()
    dl_config = ConfigObj(args.dl_config)

    home_path = str(Path.home())

    exercise = dl_config["EXERCISE"]
    classifier_type = dl_config["CLASSIFIER_TYPE"]
    nb_epochs = dl_config.as_int("EPOCHS")
    multiclass_classification = dl_config.as_bool("MULTICLASS_CLASSIFICATION")
    multiclass_dir = dl_config["MULTICLASS_DIR"]
    output_path = os.path.join(home_path, dl_config["OUTPUT_PATH"])
    data_type = dl_config["DATA_TYPE"]
    seed_values = dl_config.as_list("SEED_VALUES")
    assert type(seed_values) == list

    base_path = os.path.join(home_path, dl_config["BASE_PATH"], exercise)
    input_data_path = os.path.join(base_path, dl_config["INPUT_DATA_PATH"])

    logger.info("Final paths after substitution: {} {}".format(input_data_path, output_path))

    config_parser = configparser.RawConfigParser()
    config_parser.read(args.exercise_config)
    valid_classes = config_parser.get(exercise, "valid_classes").split(",")
    class_combination = get_combinations(valid_classes, 2)
    if multiclass_classification:
        class_combination = [multiclass_dir]
    label_index_mapping = {i + 1: value for i, value in enumerate(valid_classes)}
    index_label_mapping = {value: i + 1 for i, value in enumerate(valid_classes)}

    combination = multiclass_dir
    classification_report_list = []
    logger.info("Fitting {} on {}: ".format(classifier_type, combination))
    for seed_value in seed_values:
        logger.info("Seed value: {}".format(seed_value))
        data_path = os.path.join(input_data_path, seed_value, combination)
        classes_list = combination.split("vs")
        weights_directory = os.path.join(output_path, 'dl_weights', exercise, classifier_type, combination,
                                         seed_value)
        create_directory_if_not_exists(weights_directory)
        datasets_dict = read_dataset(data_path, data_type, combination)
        fit_classifier(classifier_type, datasets_dict, combination, exercise)
        tf.keras.backend.clear_session()
        logger.info('DONE')

    logger.info("Average classification report")
    logger.info(report_average(*classification_report_list))

"""
export PYTHONPATH=/home/ashisig/Research/Projects/FinalBodyMTS/
"""
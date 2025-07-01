# FCN model
# when tuning start with learning rate->mini_batch_size -> 
# momentum-> #hidden_units -> # learning_rate_decay -> #layers 
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

class ValAccuracyLogger(keras.callbacks.Callback):
    def __init__(self, x_val, y_val):
        super().__init__()
        self.x_val = x_val
        self.y_val = y_val

    def on_epoch_end(self, epoch, logs=None):
        val_loss, val_acc = self.model.evaluate(self.x_val, self.y_val, verbose=0)
        logger.info("Epoch %d: val_loss=%.4f, val_accuracy=%.4f", epoch + 1, val_loss, val_acc)


class Classifier_FCN:

    def __init__(self, output_directory, exercise, input_shape, nb_classes,
                 learning_rate=0.001, patience=50, min_lr=1e-4,
                 verbose=True, build=True):

        self.output_directory = output_directory
        self.model_name = "fcn"
        self.exercise = exercise
        self.verbose = verbose
        self.learning_rate = learning_rate
        self.patience = patience
        self.min_lr = min_lr

        if build:
            self.model = self.build_model(input_shape, nb_classes)
            if verbose:
                self.model.summary()
            self.model.save_weights(self.output_directory + 'model_init.weights.h5')


    def build_model(self, input_shape, nb_classes):
        input_layer = keras.layers.Input(shape=input_shape)

        x = keras.layers.Conv1D(128, 8, padding='same')(input_layer)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.ReLU()(x)

        x = keras.layers.Conv1D(256, 5, padding='same')(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.ReLU()(x)

        x = keras.layers.Conv1D(128, 3, padding='same')(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.ReLU()(x)

        x = keras.layers.GlobalAveragePooling1D()(x)
        output_layer = keras.layers.Dense(nb_classes, activation='softmax')(x)

        model = keras.models.Model(inputs=input_layer, outputs=output_layer)
        optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate)
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        return model

    @timeit
    def fit(self, x_train, y_train, x_val, y_val, nb_epochs, batch_size=16):
        if not is_gpu_available():
            logger.error('GPU not available')
            exit()

        mini_batch_size = int(min(x_train.shape[0] / 10, batch_size))

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

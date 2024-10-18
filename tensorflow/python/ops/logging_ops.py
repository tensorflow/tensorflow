import logging
import sys
from typing import Tuple
import tensorflow as tf
import numpy as np

def create_keras_sequential_model() -> tf.keras.Model:
    """Creates a simple Keras sequential model."""
    return tf.keras.Sequential([
        tf.keras.layers.Input(shape=(10,)),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(1),
    ])

def generate_random_data(num_samples: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
    """Generates random input and output data."""
    x = np.random.random((num_samples, 10))
    y = np.random.random((num_samples, 1))
    return x, y

def compile_model(model: tf.keras.Model) -> None:
    """Compiles the Keras model with specified optimizer and loss function."""
    model.compile(optimizer="sgd", loss="mean_absolute_error", metrics=["mae"])

def fit_model(model: tf.keras.Model, x: np.ndarray, y: np.ndarray, epochs: int = 10) -> None:
    """Fits the model to the provided data."""
    model.fit(x, y, epochs=epochs, verbose=0)

def create_compile_return_model() -> tf.keras.Model:
    """Creates, compiles, and fits the Keras model, returning the trained model."""
    model = create_keras_sequential_model()
    compile_model(model)
    x, y = generate_random_data()
    fit_model(model, x, y)
    return model

def get_custom_formatter_logger() -> logging.Logger:
    """Sets up a custom logger with a specific format."""
    log_format_with_time = "%(asctime)s - %(levelname)s - %(message)s"
    log_handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(log_format_with_time)
    log_handler.setFormatter(formatter)
    
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(log_handler)
    
    return logger

if __name__ == "__main__":
    logger = get_custom_formatter_logger()
    logger.info("TF version: %s", tf.__version__)
    logger.info("Started")
    
    keras_model = create_compile_return_model()
    logger.info("Model created, compiled, and trained. Now saving the model!")
    
    tf.saved_model.save(keras_model, "my_fake_model")
    logger.info("Model saved!")
    logger.info("Finished")

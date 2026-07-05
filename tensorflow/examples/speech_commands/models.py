# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Model definitions for simple speech recognition."""

import math

import tensorflow as tf


def _next_power_of_two(x):
    """Calculates the smallest enclosing power of two for an input."""
    return 1 if x == 0 else 2 ** (int(x) - 1).bit_length()


def prepare_model_settings(
    label_count,
    sample_rate,
    clip_duration_ms,
    window_size_ms,
    window_stride_ms,
    feature_bin_count,
    preprocess,
):
    """Calculates common settings needed for all models. (Unchanged)"""
    desired_samples = int(sample_rate * clip_duration_ms / 1000)
    window_size_samples = int(sample_rate * window_size_ms / 1000)
    window_stride_samples = int(sample_rate * window_stride_ms / 1000)
    length_minus_window = desired_samples - window_size_samples
    if length_minus_window < 0:
        spectrogram_length = 0
    else:
        spectrogram_length = 1 + int(length_minus_window / window_stride_samples)

    if preprocess == "average":
        fft_bin_count = 1 + (_next_power_of_two(window_size_samples) / 2)
        average_window_width = int(math.floor(fft_bin_count / feature_bin_count))
        fingerprint_width = int(math.ceil(fft_bin_count / average_window_width))
    elif preprocess in ["mfcc", "micro"]:
        average_window_width = -1
        fingerprint_width = feature_bin_count
    else:
        raise ValueError('Unknown preprocess mode "%s"' % (preprocess))

    fingerprint_size = fingerprint_width * spectrogram_length
    return {
        "desired_samples": desired_samples,
        "window_size_samples": window_size_samples,
        "window_stride_samples": window_stride_samples,
        "spectrogram_length": spectrogram_length,
        "fingerprint_width": fingerprint_width,
        "fingerprint_size": fingerprint_size,
        "label_count": label_count,
        "sample_rate": sample_rate,
        "preprocess": preprocess,
        "average_window_width": average_window_width,
    }


def create_model(model_settings, model_architecture):
    """Builds a Keras model of the requested architecture."""
    if model_architecture == "single_fc":
        return create_single_fc_model(model_settings)
    elif model_architecture == "conv":
        return create_conv_model(model_settings)
    elif model_architecture == "low_latency_conv":
        return create_low_latency_conv_model(model_settings)
    elif model_architecture == "low_latency_svdf":
        return create_low_latency_svdf_model(model_settings)
    elif model_architecture == "tiny_conv":
        return create_tiny_conv_model(model_settings)
    elif model_architecture == "tiny_embedding_conv":
        return create_tiny_embedding_conv_model(model_settings)
    else:
        raise Exception(f'model_architecture "{model_architecture}" not recognized.')


# ------------------------------------------------------------------
# Model Architectures
# ------------------------------------------------------------------


def create_single_fc_model(model_settings):
    """Builds a model with a single hidden fully-connected layer."""
    return tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(model_settings["fingerprint_size"],)),
            tf.keras.layers.Dense(
                units=model_settings["label_count"],
                kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.001),
            ),
        ]
    )


def create_conv_model(model_settings):
    """Builds a standard convolutional model."""
    fingerprint_size = model_settings["fingerprint_size"]
    input_frequency_size = model_settings["fingerprint_width"]
    input_time_size = model_settings["spectrogram_length"]
    label_count = model_settings["label_count"]

    return tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(fingerprint_size,)),
            tf.keras.layers.Reshape((input_time_size, input_frequency_size, 1)),
            # First Conv Block
            tf.keras.layers.Conv2D(
                filters=64,
                kernel_size=(20, 8),
                strides=(1, 1),
                padding="same",
                activation="relu",
                kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.01),
            ),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.MaxPooling2D(
                pool_size=(2, 2), strides=(2, 2), padding="same"
            ),
            # Second Conv Block
            tf.keras.layers.Conv2D(
                filters=64,
                kernel_size=(10, 4),
                strides=(1, 1),
                padding="same",
                activation="relu",
                kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.01),
            ),
            tf.keras.layers.Dropout(0.5),
            # Classification Head
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(
                units=label_count,
                kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.01),
            ),
        ]
    )


def create_low_latency_conv_model(model_settings):
    """Builds a convolutional model with low compute requirements."""
    fingerprint_size = model_settings["fingerprint_size"]
    input_frequency_size = model_settings["fingerprint_width"]
    input_time_size = model_settings["spectrogram_length"]
    label_count = model_settings["label_count"]

    return tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(fingerprint_size,)),
            tf.keras.layers.Reshape((input_time_size, input_frequency_size, 1)),
            tf.keras.layers.Conv2D(
                filters=186,
                kernel_size=(input_time_size, 8),
                strides=(1, 1),
                padding="valid",
                activation="relu",
                kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.01),
            ),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(
                units=128,
                activation="relu",  # Assuming missing relu in original TF1 code before dropout
                kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.01),
            ),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(
                units=128,
                activation="relu",
                kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.01),
            ),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(
                units=label_count,
                kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.01),
            ),
        ]
    )


# --- Custom Layer for SVDF Math ---
class SVDFLayer(tf.keras.layers.Layer):
    """Stateless Keras Layer encapsulating the SVDF matrix math for training."""

    def __init__(
        self, num_units, rank, input_time_size, input_frequency_size, **kwargs
    ):
        super(SVDFLayer, self).__init__(**kwargs)
        self.num_units = num_units
        self.rank = rank
        self.num_filters = rank * num_units
        self.input_time_size = input_time_size
        self.input_frequency_size = input_frequency_size

    def build(self, input_shape):
        self.weights_frequency = self.add_weight(
            name="weights_frequency",
            shape=(self.input_frequency_size, 1, self.num_filters),
            initializer=tf.keras.initializers.TruncatedNormal(stddev=0.01),
            trainable=True,
        )

        self.weights_time = self.add_weight(
            name="weights_time",
            shape=(self.num_filters, self.input_time_size, 1),
            initializer=tf.keras.initializers.TruncatedNormal(stddev=0.01),
            trainable=True,
        )

        self.bias_weight = self.add_weight(
            name="bias", shape=(self.num_units,), initializer="zeros", trainable=True
        )

    def call(self, inputs):
        # 1. Expand and Convolve Frequency
        x = tf.expand_dims(inputs, 2)
        x = tf.nn.conv1d(
            x, self.weights_frequency, stride=self.input_frequency_size, padding="VALID"
        )

        # 2. Transpose and Multiply Time
        x = tf.transpose(x, [2, 0, 1])
        x = tf.matmul(x, self.weights_time)

        # 3. Reshape and Reduce Sum
        x = tf.reshape(x, [self.num_units, self.rank, -1])
        x = tf.reduce_sum(x, axis=1)
        x = tf.transpose(x)

        return tf.nn.relu(tf.nn.bias_add(x, self.bias_weight))


def create_low_latency_svdf_model(model_settings):
    """Builds an SVDF model with low compute requirements."""
    fingerprint_size = model_settings["fingerprint_size"]
    input_frequency_size = model_settings["fingerprint_width"]
    input_time_size = model_settings["spectrogram_length"]
    label_count = model_settings["label_count"]

    # Using the Functional API to integrate the custom SVDF layer
    inputs = tf.keras.Input(shape=(fingerprint_size,))

    x = SVDFLayer(
        num_units=1280,
        rank=2,
        input_time_size=input_time_size,
        input_frequency_size=input_frequency_size,
    )(inputs)
    x = tf.keras.layers.Dropout(0.5)(x)

    x = tf.keras.layers.Dense(
        units=256, kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.01)
    )(x)
    x = tf.keras.layers.Dropout(0.5)(x)

    x = tf.keras.layers.Dense(
        units=256, kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.01)
    )(x)
    x = tf.keras.layers.Dropout(0.5)(x)

    outputs = tf.keras.layers.Dense(
        units=label_count,
        kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.01),
    )(x)

    return tf.keras.Model(inputs=inputs, outputs=outputs)


def create_tiny_conv_model(model_settings):
    """Builds a convolutional model aimed at microcontrollers."""
    fingerprint_size = model_settings["fingerprint_size"]
    input_frequency_size = model_settings["fingerprint_width"]
    input_time_size = model_settings["spectrogram_length"]
    label_count = model_settings["label_count"]

    return tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(fingerprint_size,)),
            tf.keras.layers.Reshape((input_time_size, input_frequency_size, 1)),
            tf.keras.layers.Conv2D(
                filters=8,
                kernel_size=(10, 8),
                strides=(2, 2),
                padding="same",
                activation="relu",
                kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.01),
            ),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(
                units=label_count,
                kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.01),
            ),
        ]
    )


def create_tiny_embedding_conv_model(model_settings):
    """Builds an embedded convolutional model aimed at microcontrollers."""
    fingerprint_size = model_settings["fingerprint_size"]
    input_frequency_size = model_settings["fingerprint_width"]
    input_time_size = model_settings["spectrogram_length"]
    label_count = model_settings["label_count"]

    return tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(fingerprint_size,)),
            tf.keras.layers.Reshape((input_time_size, input_frequency_size, 1)),
            tf.keras.layers.Conv2D(
                filters=8,
                kernel_size=(10, 8),
                strides=(2, 2),
                padding="same",
                activation="relu",
                kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.01),
            ),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Conv2D(
                filters=8,
                kernel_size=(10, 8),
                strides=(8, 8),
                padding="same",
                activation="relu",
                kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.01),
            ),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(
                units=label_count,
                kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.01),
            ),
        ]
    )

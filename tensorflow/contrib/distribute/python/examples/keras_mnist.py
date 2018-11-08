# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""An example training a Keras Model using MirroredStrategy and native APIs."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


NUM_CLASSES = 10


def get_input_datasets(use_bfloat16=False):
  """Downloads the MNIST dataset and creates train and eval dataset objects.

  Args:
    use_bfloat16: Boolean to determine if input should be cast to bfloat16

  Returns:
    Train dataset, eval dataset and input shape.

  """
  # input image dimensions
  img_rows, img_cols = 28, 28
  cast_dtype = tf.bfloat16 if use_bfloat16 else tf.float32

  # the data, split between train and test sets
  (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

  if tf.keras.backend.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
  else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

  x_train = x_train.astype('float32')
  x_test = x_test.astype('float32')
  x_train /= 255
  x_test /= 255

  # convert class vectors to binary class matrices
  y_train = tf.keras.utils.to_categorical(y_train, NUM_CLASSES)
  y_test = tf.keras.utils.to_categorical(y_test, NUM_CLASSES)

  # train dataset
  train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
  train_ds = train_ds.repeat()
  train_ds = train_ds.map(lambda x, y: (tf.cast(x, cast_dtype), y))
  train_ds = train_ds.batch(64, drop_remainder=True)

  # eval dataset
  eval_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))
  eval_ds = eval_ds.repeat()
  eval_ds = eval_ds.map(lambda x, y: (tf.cast(x, cast_dtype), y))
  eval_ds = eval_ds.batch(64, drop_remainder=True)

  return train_ds, eval_ds, input_shape


def get_model(input_shape):
  """Builds a Sequential CNN model to recognize MNIST digits.

  Args:
    input_shape: Shape of the input depending on the `image_data_format`.

  Returns:
    a Keras model

  """
  # Define a CNN model to recognize MNIST digits.
  model = tf.keras.models.Sequential()
  model.add(tf.keras.layers.Conv2D(32, kernel_size=(3, 3),
                                   activation='relu',
                                   input_shape=input_shape))
  model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
  model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
  model.add(tf.keras.layers.Dropout(0.25))
  model.add(tf.keras.layers.Flatten())
  model.add(tf.keras.layers.Dense(128, activation='relu'))
  model.add(tf.keras.layers.Dropout(0.5))
  model.add(tf.keras.layers.Dense(NUM_CLASSES, activation='softmax'))
  return model


def main(_):
  # Build the train and eval datasets from the MNIST data. Also return the
  # input shape which is constructed based on the `image_data_format`
  # i.e channels_first or channels_last.
  train_ds, eval_ds, input_shape = get_input_datasets()
  model = get_model(input_shape)

  # Instantiate the MirroredStrategy object. If we don't specify `num_gpus` or
  # the `devices` argument then all the GPUs available on the machine are used.
  strategy = tf.contrib.distribute.MirroredStrategy()

  # Compile the model by passing the distribution strategy object to the
  # `distribute` argument. `fit`, `evaluate` and `predict` will be distributed
  # based on the strategy instantiated.
  model.compile(loss=tf.keras.losses.categorical_crossentropy,
                optimizer=tf.train.RMSPropOptimizer(learning_rate=0.001),
                metrics=['accuracy'],
                distribute=strategy)

  # Train the model with the train dataset.
  model.fit(x=train_ds, epochs=20, steps_per_epoch=468)

  # Evaluate the model with the eval dataset.
  score = model.evaluate(eval_ds, steps=10, verbose=0)
  print('Test loss:', score[0])
  print('Test accuracy:', score[1])


if __name__ == '__main__':
  tf.app.run()

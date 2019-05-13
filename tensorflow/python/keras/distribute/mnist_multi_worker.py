# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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

from absl import flags

# pylint: disable=g-direct-tensorflow-import
from tensorflow.python import keras
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.distribute import collective_all_reduce_strategy as collective_strategy
from tensorflow.python.distribute import multi_worker_util
from tensorflow.python.distribute.cluster_resolver import TFConfigClusterResolver
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.keras import backend
from tensorflow.python.keras import utils
from tensorflow.python.keras.datasets import mnist
from tensorflow.python.keras.optimizer_v2 import rmsprop
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import app
from tensorflow.python.platform import tf_logging as logging

NUM_CLASSES = 10

flags.DEFINE_boolean(name='enable_eager', default=False, help='Enable eager?')
flags.DEFINE_enum('distribution_strategy', None, ['multi_worker_mirrored'],
                  'The Distribution Strategy to use.')
flags.DEFINE_string('model_dir', None, 'Directory for TensorBoard/Checkpoint.')


# TODO(rchao): Use multi_worker_util.maybe_shard_dataset() once that is provided
# there.
def maybe_shard_dataset(dataset):
  """Shard the dataset if running in multi-node environment."""
  cluster_resolver = TFConfigClusterResolver()
  cluster_spec = cluster_resolver.cluster_spec().as_dict()
  if cluster_spec:
    dataset = dataset.shard(
        multi_worker_util.worker_count(cluster_spec,
                                       cluster_resolver.task_type),
        multi_worker_util.id_in_cluster(
            cluster_spec, cluster_resolver.task_type, cluster_resolver.task_id))
  return dataset


def get_data_shape():
  # input image dimensions
  img_rows, img_cols = 28, 28
  if backend.image_data_format() == 'channels_first':
    return 1, img_rows, img_cols
  else:
    return img_rows, img_cols, 1


def get_input_datasets(use_bfloat16=False):
  """Downloads the MNIST dataset and creates train and eval dataset objects.

  Args:
    use_bfloat16: Boolean to determine if input should be cast to bfloat16

  Returns:
    Train dataset and eval dataset. The dataset doesn't include batch dim.

  """
  cast_dtype = dtypes.bfloat16 if use_bfloat16 else dtypes.float32

  # the data, split between train and test sets
  (x_train, y_train), (x_test, y_test) = mnist.load_data()

  train_data_shape = (x_train.shape[0],) + get_data_shape()
  test_data_shape = (x_test.shape[0],) + get_data_shape()
  if backend.image_data_format() == 'channels_first':
    x_train = x_train.reshape(train_data_shape)
    x_test = x_test.reshape(test_data_shape)
  else:
    x_train = x_train.reshape(train_data_shape)
    x_test = x_test.reshape(test_data_shape)

  x_train = x_train.astype('float32')
  x_test = x_test.astype('float32')
  x_train /= 255
  x_test /= 255

  # convert class vectors to binary class matrices
  y_train = utils.to_categorical(y_train, NUM_CLASSES)
  y_test = utils.to_categorical(y_test, NUM_CLASSES)

  # train dataset
  train_ds = dataset_ops.Dataset.from_tensor_slices((x_train, y_train))
  # TODO(rchao): Remove maybe_shard_dataset() once auto-sharding is done.
  train_ds = maybe_shard_dataset(train_ds)
  train_ds = train_ds.repeat()
  train_ds = train_ds.map(lambda x, y: (math_ops.cast(x, cast_dtype), y))
  train_ds = train_ds.batch(64, drop_remainder=True)

  # eval dataset
  eval_ds = dataset_ops.Dataset.from_tensor_slices((x_test, y_test))
  # TODO(rchao): Remove maybe_shard_dataset() once auto-sharding is done.
  eval_ds = maybe_shard_dataset(eval_ds)
  eval_ds = eval_ds.repeat()
  eval_ds = eval_ds.map(lambda x, y: (math_ops.cast(x, cast_dtype), y))
  eval_ds = eval_ds.batch(64, drop_remainder=True)

  return train_ds, eval_ds


def get_model(index=0):
  """Builds a Sequential CNN model to recognize MNIST digits.

  Args:
    index: The worker index. Defaults to 0.

  Returns:
    a CNN Keras model used for MNIST

  """

  # Define a CNN model to recognize MNIST digits.
  model = keras.models.Sequential()
  model.add(
      keras.layers.Conv2D(
          32,
          kernel_size=(3, 3),
          activation='relu',
          input_shape=get_data_shape()))
  model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
  model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
  model.add(keras.layers.Dropout(0.25, name='dropout_worker%s_first' % index))
  model.add(keras.layers.Flatten())
  model.add(keras.layers.Dense(128, activation='relu'))
  model.add(keras.layers.Dropout(0.5, name='dropout_worker%s_second' % index))
  model.add(keras.layers.Dense(NUM_CLASSES, activation='softmax'))
  return model


def main(_):
  if flags.FLAGS.enable_eager:
    ops.enable_eager_execution()
    logging.info('Eager execution enabled for MNIST Multi-Worker.')
  else:
    logging.info('Eager execution not enabled for MNIST Multi-Worker.')

  # Build the train and eval datasets from the MNIST data.
  train_ds, eval_ds = get_input_datasets()

  if flags.FLAGS.distribution_strategy == 'multi_worker_mirrored':
    # MultiWorkerMirroredStrategy for multi-worker distributed MNIST training.
    strategy = collective_strategy.CollectiveAllReduceStrategy()
  else:
    raise ValueError('Only `multi_worker_mirrored` is supported strategy '
                     'in Keras MNIST example at this time. Strategy passed '
                     'in is %s' % flags.FLAGS.distribution_strategy)

  # Create and compile the model under Distribution strategy scope.
  # `fit`, `evaluate` and `predict` will be distributed based on the strategy
  # model was compiled with.
  with strategy.scope():
    model = get_model()
    optimizer = rmsprop.RMSProp(learning_rate=0.001)
    model.compile(
        loss=keras.losses.categorical_crossentropy,
        optimizer=optimizer,
        metrics=['accuracy'])

  # Train the model with the train dataset.
  tensorboard_callback = keras.callbacks.TensorBoard(
      log_dir=flags.FLAGS.model_dir)
  model.fit(
      x=train_ds,
      epochs=20,
      steps_per_epoch=468,
      callbacks=[tensorboard_callback])

  # Evaluate the model with the eval dataset.
  score = model.evaluate(eval_ds, steps=10, verbose=0)
  logging.info('Test loss:{}'.format(score[0]))
  logging.info('Test accuracy:{}'.format(score[1]))


if __name__ == '__main__':
  logging.set_verbosity(logging.INFO)
  app.run()

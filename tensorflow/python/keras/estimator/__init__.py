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
"""Keras estimator API."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.eager import monitoring
from tensorflow.python.util.tf_export import keras_export

# Keras has undeclared dependency on tensorflow/estimator:estimator_py.
# As long as you depend //third_party/py/tensorflow:tensorflow target
# everything will work as normal.

_model_to_estimator_usage_gauge = monitoring.BoolGauge(
    '/tensorflow/api/keras/model_to_estimator',
    'Whether tf.keras.estimator.model_to_estimator() is called.', 'version')


# LINT.IfChange
@keras_export(v1=['keras.estimator.model_to_estimator'])
def model_to_estimator(
    keras_model=None,
    keras_model_path=None,
    custom_objects=None,
    model_dir=None,
    config=None,
    checkpoint_format='saver'):
  """Constructs an `Estimator` instance from given keras model.

  If you use infrastructure or other tooling that relies on Estimators, you can
  still build a Keras model and use model_to_estimator to convert the Keras
  model to an Estimator for use with downstream systems.

  For usage example, please see:
  [Creating estimators from Keras
  Models](https://www.tensorflow.org/guide/estimators#creating_estimators_from_keras_models).

  __Sample Weights__
  Estimators returned by `model_to_estimator` are configured to handle sample
  weights (similar to `keras_model.fit(x, y, sample_weights)`). To pass sample
  weights when training or evaluating the Estimator, the first item returned by
  the input function should be a dictionary with keys `features` and
  `sample_weights`. Example below:

  ```
  keras_model = tf.keras.Model(...)
  keras_model.compile(...)

  estimator = tf.keras.estimator.model_to_estimator(keras_model)

  def input_fn():
    return dataset_ops.Dataset.from_tensors(
        ({'features': features, 'sample_weights': sample_weights},
         targets))

  estimator.train(input_fn, steps=1)
  ```

  Args:
    keras_model: A compiled Keras model object. This argument is mutually
      exclusive with `keras_model_path`.
    keras_model_path: Path to a compiled Keras model saved on disk, in HDF5
      format, which can be generated with the `save()` method of a Keras model.
      This argument is mutually exclusive with `keras_model`.
    custom_objects: Dictionary for custom objects.
    model_dir: Directory to save `Estimator` model parameters, graph, summary
      files for TensorBoard, etc.
    config: `RunConfig` to config `Estimator`.
    checkpoint_format: Sets the format of the checkpoint saved by the estimator
      when training. May be `saver` or `checkpoint`, depending on whether to
      save checkpoints from `tf.train.Saver` or `tf.train.Checkpoint`. This
      argument currently defaults to `saver`. When 2.0 is released, the default
      will be `checkpoint`. Estimators use name-based `tf.train.Saver`
      checkpoints, while Keras models use object-based checkpoints from
      `tf.train.Checkpoint`. Currently, saving object-based checkpoints from
      `model_to_estimator` is only supported by Functional and Sequential
      models.

  Returns:
    An Estimator from given keras model.

  Raises:
    ValueError: if neither keras_model nor keras_model_path was given.
    ValueError: if both keras_model and keras_model_path was given.
    ValueError: if the keras_model_path is a GCS URI.
    ValueError: if keras_model has not been compiled.
    ValueError: if an invalid checkpoint_format was given.
  """
  try:
    from tensorflow_estimator.python.estimator import keras as keras_lib  # pylint: disable=g-import-not-at-top
  except ImportError:
    raise NotImplementedError(
        'tf.keras.estimator.model_to_estimator function not available in your '
        'installation.')
  _model_to_estimator_usage_gauge.get_cell('v1').set(True)
  return keras_lib.model_to_estimator(  # pylint:disable=unexpected-keyword-arg
      keras_model=keras_model,
      keras_model_path=keras_model_path,
      custom_objects=custom_objects,
      model_dir=model_dir,
      config=config,
      checkpoint_format=checkpoint_format,
      use_v2_estimator=False)


@keras_export('keras.estimator.model_to_estimator', v1=[])
def model_to_estimator_v2(
    keras_model=None,
    keras_model_path=None,
    custom_objects=None,
    model_dir=None,
    config=None,
    checkpoint_format='checkpoint'):
  """Constructs an `Estimator` instance from given keras model.

  If you use infrastructure or other tooling that relies on Estimators, you can
  still build a Keras model and use model_to_estimator to convert the Keras
  model to an Estimator for use with downstream systems.

  For usage example, please see:
  [Creating estimators from Keras
  Models](https://www.tensorflow.org/guide/estimators#creating_estimators_from_keras_models).

  __Sample Weights__
  Estimators returned by `model_to_estimator` are configured to handle sample
  weights (similar to `keras_model.fit(x, y, sample_weights)`). To pass sample
  weights when training or evaluating the Estimator, the first item returned by
  the input function should be a dictionary with keys `features` and
  `sample_weights`. Example below:

  ```
  keras_model = tf.keras.Model(...)
  keras_model.compile(...)

  estimator = tf.keras.estimator.model_to_estimator(keras_model)

  def input_fn():
    return dataset_ops.Dataset.from_tensors(
        ({'features': features, 'sample_weights': sample_weights},
         targets))

  estimator.train(input_fn, steps=1)
  ```

  Args:
    keras_model: A compiled Keras model object. This argument is mutually
      exclusive with `keras_model_path`.
    keras_model_path: Path to a compiled Keras model saved on disk, in HDF5
      format, which can be generated with the `save()` method of a Keras model.
      This argument is mutually exclusive with `keras_model`.
    custom_objects: Dictionary for custom objects.
    model_dir: Directory to save `Estimator` model parameters, graph, summary
      files for TensorBoard, etc.
    config: `RunConfig` to config `Estimator`.
    checkpoint_format: Sets the format of the checkpoint saved by the estimator
      when training. May be `saver` or `checkpoint`, depending on whether to
      save checkpoints from `tf.compat.v1.train.Saver` or `tf.train.Checkpoint`.
      The default is `checkpoint`. Estimators use name-based `tf.train.Saver`
      checkpoints, while Keras models use object-based checkpoints from
      `tf.train.Checkpoint`. Currently, saving object-based checkpoints from
      `model_to_estimator` is only supported by Functional and Sequential
      models.

  Returns:
    An Estimator from given keras model.

  Raises:
    ValueError: if neither keras_model nor keras_model_path was given.
    ValueError: if both keras_model and keras_model_path was given.
    ValueError: if the keras_model_path is a GCS URI.
    ValueError: if keras_model has not been compiled.
    ValueError: if an invalid checkpoint_format was given.
  """
  try:
    from tensorflow_estimator.python.estimator import keras as keras_lib  # pylint: disable=g-import-not-at-top
  except ImportError:
    raise NotImplementedError(
        'tf.keras.estimator.model_to_estimator function not available in your '
        'installation.')
  _model_to_estimator_usage_gauge.get_cell('v2').set(True)
  return keras_lib.model_to_estimator(  # pylint:disable=unexpected-keyword-arg
      keras_model=keras_model,
      keras_model_path=keras_model_path,
      custom_objects=custom_objects,
      model_dir=model_dir,
      config=config,
      checkpoint_format=checkpoint_format,
      use_v2_estimator=True)
# LINT.ThenChange(//tensorflow_estimator/python/estimator/keras.py)

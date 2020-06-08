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
  [Creating estimators from Keras Models](
    https://www.tensorflow.org/guide/estimators#creating_estimators_from_keras_models).

  Sample Weights:
  Estimators returned by `model_to_estimator` are configured so that they can
  handle sample weights (similar to `keras_model.fit(x, y, sample_weights)`).

  To pass sample weights when training or evaluating the Estimator, the first
  item returned by the input function should be a dictionary with keys
  `features` and `sample_weights`. Example below:

  ```python
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
      exclusive with `keras_model_path`. Estimator's `model_fn` uses the
      structure of the model to clone the model. Defaults to `None`.
    keras_model_path: Path to a compiled Keras model saved on disk, in HDF5
      format, which can be generated with the `save()` method of a Keras model.
      This argument is mutually exclusive with `keras_model`.
      Defaults to `None`.
    custom_objects: Dictionary for cloning customized objects. This is
      used with classes that is not part of this pip package. For example, if
      user maintains a `relu6` class that inherits from `tf.keras.layers.Layer`,
      then pass `custom_objects={'relu6': relu6}`. Defaults to `None`.
    model_dir: Directory to save `Estimator` model parameters, graph, summary
      files for TensorBoard, etc. If unset a directory will be created with
      `tempfile.mkdtemp`
    config: `RunConfig` to config `Estimator`. Allows setting up things in
      `model_fn` based on configuration such as `num_ps_replicas`, or
      `model_dir`. Defaults to `None`. If both `config.model_dir` and the
      `model_dir` argument (above) are specified the `model_dir` **argument**
      takes precedence.
    checkpoint_format: Sets the format of the checkpoint saved by the estimator
      when training. May be `saver` or `checkpoint`, depending on whether to
      save checkpoints from `tf.train.Saver` or `tf.train.Checkpoint`. This
      argument currently defaults to `saver`. When 2.0 is released, the default
      will be `checkpoint`. Estimators use name-based `tf.train.Saver`
      checkpoints, while Keras models use object-based checkpoints from
      `tf.train.Checkpoint`. Currently, saving object-based checkpoints from
      `model_to_estimator` is only supported by Functional and Sequential
      models. Defaults to 'saver'.

  Returns:
    An Estimator from given keras model.

  Raises:
    ValueError: If neither keras_model nor keras_model_path was given.
    ValueError: If both keras_model and keras_model_path was given.
    ValueError: If the keras_model_path is a GCS URI.
    ValueError: If keras_model has not been compiled.
    ValueError: If an invalid checkpoint_format was given.
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
def model_to_estimator_v2(keras_model=None,
                          keras_model_path=None,
                          custom_objects=None,
                          model_dir=None,
                          config=None,
                          checkpoint_format='checkpoint',
                          metric_names_map=None):
  """Constructs an `Estimator` instance from given keras model.

  If you use infrastructure or other tooling that relies on Estimators, you can
  still build a Keras model and use model_to_estimator to convert the Keras
  model to an Estimator for use with downstream systems.

  For usage example, please see:
  [Creating estimators from Keras Models](
    https://www.tensorflow.org/guide/estimators#creating_estimators_from_keras_models).

  Sample Weights:
  Estimators returned by `model_to_estimator` are configured so that they can
  handle sample weights (similar to `keras_model.fit(x, y, sample_weights)`).

  To pass sample weights when training or evaluating the Estimator, the first
  item returned by the input function should be a dictionary with keys
  `features` and `sample_weights`. Example below:

  ```python
  keras_model = tf.keras.Model(...)
  keras_model.compile(...)

  estimator = tf.keras.estimator.model_to_estimator(keras_model)

  def input_fn():
    return dataset_ops.Dataset.from_tensors(
        ({'features': features, 'sample_weights': sample_weights},
         targets))

  estimator.train(input_fn, steps=1)
  ```

  Note: We do not support creating weighted metrics in Keras and converting them
  to weighted metrics in the Estimator API using `model_to_estimator`.
  You will have to create these metrics directly on the estimator spec using the
  `add_metrics` function.

  To customize the estimator `eval_metric_ops` names, you can pass in the
  `metric_names_map` dictionary mapping the keras model output metric names
  to the custom names as follows:

  ```python
    input_a = tf.keras.layers.Input(shape=(16,), name='input_a')
    input_b = tf.keras.layers.Input(shape=(16,), name='input_b')
    dense = tf.keras.layers.Dense(8, name='dense_1')
    interm_a = dense(input_a)
    interm_b = dense(input_b)
    merged = tf.keras.layers.concatenate([interm_a, interm_b], name='merge')
    output_a = tf.keras.layers.Dense(3, activation='softmax', name='dense_2')(
            merged)
    output_b = tf.keras.layers.Dense(2, activation='softmax', name='dense_3')(
            merged)
    keras_model = tf.keras.models.Model(
        inputs=[input_a, input_b], outputs=[output_a, output_b])
    keras_model.compile(
        loss='categorical_crossentropy',
        optimizer='rmsprop',
        metrics={
            'dense_2': 'categorical_accuracy',
            'dense_3': 'categorical_accuracy'
        })

    metric_names_map = {
        'dense_2_categorical_accuracy': 'acc_1',
        'dense_3_categorical_accuracy': 'acc_2',
    }
    keras_est = tf.keras.estimator.model_to_estimator(
        keras_model=keras_model,
        config=config,
        metric_names_map=metric_names_map)
  ```

  Args:
    keras_model: A compiled Keras model object. This argument is mutually
      exclusive with `keras_model_path`. Estimator's `model_fn` uses the
      structure of the model to clone the model. Defaults to `None`.
    keras_model_path: Path to a compiled Keras model saved on disk, in HDF5
      format, which can be generated with the `save()` method of a Keras model.
      This argument is mutually exclusive with `keras_model`.
      Defaults to `None`.
    custom_objects: Dictionary for cloning customized objects. This is
      used with classes that is not part of this pip package. For example, if
      user maintains a `relu6` class that inherits from `tf.keras.layers.Layer`,
      then pass `custom_objects={'relu6': relu6}`. Defaults to `None`.
    model_dir: Directory to save `Estimator` model parameters, graph, summary
      files for TensorBoard, etc. If unset a directory will be created with
      `tempfile.mkdtemp`
    config: `RunConfig` to config `Estimator`. Allows setting up things in
      `model_fn` based on configuration such as `num_ps_replicas`, or
      `model_dir`. Defaults to `None`. If both `config.model_dir` and the
      `model_dir` argument (above) are specified the `model_dir` **argument**
      takes precedence.
    checkpoint_format: Sets the format of the checkpoint saved by the estimator
      when training. May be `saver` or `checkpoint`, depending on whether to
      save checkpoints from `tf.compat.v1.train.Saver` or `tf.train.Checkpoint`.
      The default is `checkpoint`. Estimators use name-based `tf.train.Saver`
      checkpoints, while Keras models use object-based checkpoints from
      `tf.train.Checkpoint`. Currently, saving object-based checkpoints from
      `model_to_estimator` is only supported by Functional and Sequential
      models. Defaults to 'checkpoint'.
    metric_names_map: Optional dictionary mapping Keras model output metric
      names to custom names. This can be used to override the default Keras
      model output metrics names in a multi IO model use case and provide custom
      names for the `eval_metric_ops` in Estimator.
      The Keras model metric names can be obtained using `model.metrics_names`
      excluding any loss metrics such as total loss and output losses.
      For example, if your Keras model has two outputs `out_1` and `out_2`,
      with `mse` loss and `acc` metric, then `model.metrics_names` will be
      `['loss', 'out_1_loss', 'out_2_loss', 'out_1_acc', 'out_2_acc']`.
      The model metric names excluding the loss metrics will be
      `['out_1_acc', 'out_2_acc']`.

  Returns:
    An Estimator from given keras model.

  Raises:
    ValueError: If neither keras_model nor keras_model_path was given.
    ValueError: If both keras_model and keras_model_path was given.
    ValueError: If the keras_model_path is a GCS URI.
    ValueError: If keras_model has not been compiled.
    ValueError: If an invalid checkpoint_format was given.
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
      use_v2_estimator=True,
      metric_names_map=metric_names_map)
# LINT.ThenChange(//tensorflow_estimator/python/estimator/keras.py)

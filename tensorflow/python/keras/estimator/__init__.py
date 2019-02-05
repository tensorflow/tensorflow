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

from tensorflow.python.util.tf_export import keras_export

# Keras has undeclared dependency on tensorflow/estimator:estimator_py.
# As long as you depend //third_party/py/tensorflow:tensorflow target
# everything will work as normal.


# LINT.IfChange
@keras_export('keras.estimator.model_to_estimator')
def model_to_estimator(
    keras_model=None,
    keras_model_path=None,
    custom_objects=None,
    model_dir=None,
    config=None):
  """Constructs an `Estimator` instance from given keras model.

  For usage example, please see:
  [Creating estimators from Keras
  Models](https://tensorflow.org/guide/estimators#model_to_estimator).

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

  Returns:
    An Estimator from given keras model.

  Raises:
    ValueError: if neither keras_model nor keras_model_path was given.
    ValueError: if both keras_model and keras_model_path was given.
    ValueError: if the keras_model_path is a GCS URI.
    ValueError: if keras_model has not been compiled.
  """
  try:
    from tensorflow_estimator.python.estimator import keras as keras_lib  # pylint: disable=g-import-not-at-top
  except ImportError:
    raise NotImplementedError(
        'tf.keras.estimator.model_to_estimator function not available in your '
        'installation.')
  return keras_lib.model_to_estimator(
      keras_model=keras_model,
      keras_model_path=keras_model_path,
      custom_objects=custom_objects,
      model_dir=model_dir,
      config=config)

# LINT.ThenChange(//tensorflow_estimator/python/estimator/keras.py)



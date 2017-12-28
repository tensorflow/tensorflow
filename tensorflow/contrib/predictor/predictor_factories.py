# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""Factory functions for `Predictor`s."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.predictor import contrib_estimator_predictor
from tensorflow.contrib.predictor import core_estimator_predictor
from tensorflow.contrib.predictor import saved_model_predictor

from tensorflow.contrib.learn.python.learn.estimators import estimator as contrib_estimator
from tensorflow.python.estimator import estimator as core_estimator


def from_contrib_estimator(estimator,
                           prediction_input_fn,
                           input_alternative_key=None,
                           output_alternative_key=None,
                           graph=None):
  """Constructs a `Predictor` from a `tf.contrib.learn.Estimator`.

  Args:
    estimator: an instance of `tf.contrib.learn.Estimator`.
    prediction_input_fn: a function that takes no arguments and returns an
      instance of `InputFnOps`.
    input_alternative_key: Optional. Specify the input alternative used for
      prediction.
    output_alternative_key: Specify the output alternative used for
      prediction. Not needed for single-headed models but required for
      multi-headed models.
    graph: Optional. The Tensorflow `graph` in which prediction should be
      done.

  Returns:
    An initialized `Predictor`.

  Raises:
    TypeError: if `estimator` is a core `Estimator` instead of a contrib
      `Estimator`.
  """
  if isinstance(estimator, core_estimator.Estimator):
    raise TypeError('Espected estimator to be of type '
                    'tf.contrib.learn.Estimator, but got type '
                    'tf.python.estimator.Estimator. You likely want to call '
                    'from_estimator.')
  return contrib_estimator_predictor.ContribEstimatorPredictor(
      estimator,
      prediction_input_fn,
      input_alternative_key=input_alternative_key,
      output_alternative_key=output_alternative_key,
      graph=graph)


def from_estimator(estimator,
                   serving_input_receiver_fn,
                   output_key=None,
                   graph=None):
  """Constructs a `Predictor` from a `tf.python.estimator.Estimator`.

  Args:
    estimator: an instance of `learn.python.estimator.Estimator`.
    serving_input_receiver_fn: a function that takes no arguments and returns
      an instance of `ServingInputReceiver` compatible with `estimator`.
    output_key: Optional string specifying the export output to use. If
      `None`, then `DEFAULT_SERVING_SIGNATURE_DEF_KEY` is used.
    graph: Optional. The Tensorflow `graph` in which prediction should be
      done.

  Returns:
    An initialized `Predictor`.

  Raises:
    TypeError: if `estimator` is a contrib `Estimator` instead of a core
      `Estimator`.
  """
  if isinstance(estimator, contrib_estimator.Estimator):
    raise TypeError('Espected estimator to be of type '
                    'tf.python.estimator.Estimator, but got type '
                    'tf.contrib.learn.Estimator. You likely want to call '
                    'from_contrib_estimator.')
  return core_estimator_predictor.CoreEstimatorPredictor(
      estimator, serving_input_receiver_fn, output_key=output_key, graph=graph)


def from_saved_model(export_dir,
                     signature_def_key=None,
                     signature_def=None,
                     tags=None,
                     graph=None):
  """Constructs a `Predictor` from a `SavedModel` on disk.

  Args:
    export_dir: a path to a directory containing a `SavedModel`.
    signature_def_key: Optional string specifying the signature to use. If
      `None`, then `DEFAULT_SERVING_SIGNATURE_DEF_KEY` is used. Only one of
    `signature_def_key` and `signature_def`
    signature_def: A `SignatureDef` proto specifying the inputs and outputs
      for prediction. Only one of `signature_def_key` and `signature_def`
      should be specified.
    tags: Optional. Tags that will be used to retrieve the correct
      `SignatureDef`. Defaults to `DEFAULT_TAGS`.
    graph: Optional. The Tensorflow `graph` in which prediction should be
      done.

  Returns:
    An initialized `Predictor`.

  Raises:
    ValueError: More than one of `signature_def_key` and `signature_def` is
      specified.
  """
  return saved_model_predictor.SavedModelPredictor(
      export_dir,
      signature_def_key=signature_def_key,
      signature_def=signature_def,
      tags=tags,
      graph=graph)

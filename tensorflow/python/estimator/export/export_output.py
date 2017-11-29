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
"""Classes for different types of export output."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc

import six


from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.saved_model import signature_def_utils


class ExportOutput(object):
  """Represents an output of a model that can be served.

  These typically correspond to model heads.
  """

  __metaclass__ = abc.ABCMeta

  @abc.abstractmethod
  def as_signature_def(self, receiver_tensors):
    """Generate a SignatureDef proto for inclusion in a MetaGraphDef.

    The SignatureDef will specify outputs as described in this ExportOutput,
    and will use the provided receiver_tensors as inputs.

    Args:
      receiver_tensors: a `Tensor`, or a dict of string to `Tensor`, specifying
        input nodes that will be fed.
    """
    pass


class ClassificationOutput(ExportOutput):
  """Represents the output of a classification head.

  Either classes or scores or both must be set.

  The classes `Tensor` must provide string labels, not integer class IDs.

  If only classes is set, it is interpreted as providing top-k results in
  descending order.

  If only scores is set, it is interpreted as providing a score for every class
  in order of class ID.

  If both classes and scores are set, they are interpreted as zipped, so each
  score corresponds to the class at the same index.  Clients should not depend
  on the order of the entries.
  """

  def __init__(self, scores=None, classes=None):
    """Constructor for `ClassificationOutput`.

    Args:
      scores: A float `Tensor` giving scores (sometimes but not always
          interpretable as probabilities) for each class.  May be `None`, but
          only if `classes` is set.  Interpretation varies-- see class doc.
      classes: A string `Tensor` giving predicted class labels.  May be `None`,
          but only if `scores` is set.  Interpretation varies-- see class doc.

    Raises:
      ValueError: if neither classes nor scores is set, or one of them is not a
          `Tensor` with the correct dtype.
    """
    if (scores is not None
        and not (isinstance(scores, ops.Tensor)
                 and scores.dtype.is_floating)):
      raise ValueError('Classification scores must be a float32 Tensor; '
                       'got {}'.format(scores))
    if (classes is not None
        and not (isinstance(classes, ops.Tensor)
                 and dtypes.as_dtype(classes.dtype) == dtypes.string)):
      raise ValueError('Classification classes must be a string Tensor; '
                       'got {}'.format(classes))
    if scores is None and classes is None:
      raise ValueError('At least one of scores and classes must be set.')

    self._scores = scores
    self._classes = classes

  @property
  def scores(self):
    return self._scores

  @property
  def classes(self):
    return self._classes

  def as_signature_def(self, receiver_tensors):
    if len(receiver_tensors) != 1:
      raise ValueError('Classification input must be a single string Tensor; '
                       'got {}'.format(receiver_tensors))
    (_, examples), = receiver_tensors.items()
    if dtypes.as_dtype(examples.dtype) != dtypes.string:
      raise ValueError('Classification input must be a single string Tensor; '
                       'got {}'.format(receiver_tensors))
    return signature_def_utils.classification_signature_def(
        examples, self.classes, self.scores)


class RegressionOutput(ExportOutput):
  """Represents the output of a regression head."""

  def __init__(self, value):
    """Constructor for `RegressionOutput`.

    Args:
      value: a float `Tensor` giving the predicted values.  Required.

    Raises:
      ValueError: if the value is not a `Tensor` with dtype tf.float32.
    """
    if not (isinstance(value, ops.Tensor) and value.dtype.is_floating):
      raise ValueError('Regression output value must be a float32 Tensor; '
                       'got {}'.format(value))
    self._value = value

  @property
  def value(self):
    return self._value

  def as_signature_def(self, receiver_tensors):
    if len(receiver_tensors) != 1:
      raise ValueError('Regression input must be a single string Tensor; '
                       'got {}'.format(receiver_tensors))
    (_, examples), = receiver_tensors.items()
    if dtypes.as_dtype(examples.dtype) != dtypes.string:
      raise ValueError('Regression input must be a single string Tensor; '
                       'got {}'.format(receiver_tensors))
    return signature_def_utils.regression_signature_def(examples, self.value)


class PredictOutput(ExportOutput):
  """Represents the output of a generic prediction head.

  A generic prediction need not be either a classification or a regression.

  Named outputs must be provided as a dict from string to `Tensor`,
  """

  def __init__(self, outputs):
    """Constructor for PredictOutput.

    Args:
      outputs: A dict of string to `Tensor` representing the predictions.

    Raises:
      ValueError: if the outputs is not dict, or any of its keys are not
          strings, or any of its values are not `Tensor`s.
    """
    if not isinstance(outputs, dict):
      raise ValueError(
          'Prediction outputs must be given as a dict of string to Tensor; '
          'got {}'.format(outputs))
    for key, value in outputs.items():
      if not isinstance(key, six.string_types):
        raise ValueError(
            'Prediction output key must be a string; got {}.'.format(key))
      if not isinstance(value, ops.Tensor):
        raise ValueError(
            'Prediction output value must be a Tensor; got {}.'.format(value))
    self._outputs = outputs

  @property
  def outputs(self):
    return self._outputs

  def as_signature_def(self, receiver_tensors):
    return signature_def_utils.predict_signature_def(receiver_tensors,
                                                     self.outputs)

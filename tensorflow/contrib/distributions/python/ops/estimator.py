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
"""Functions to bridge `Distribution`s and `tf.contrib.learn.estimator` APIs."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.learn.python.learn.estimators.head import _compute_weighted_loss
from tensorflow.contrib.learn.python.learn.estimators.head import _RegressionHead
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.util import deprecation


__all__ = [
    "estimator_head_distribution_regression",
]


@deprecation.deprecated(
    "2018-10-01",
    "The TensorFlow Distributions library has moved to "
    "TensorFlow Probability "
    "(https://github.com/tensorflow/probability). You "
    "should update all references to use `tfp.distributions` "
    "instead of `tf.contrib.distributions`.",
    warn_once=True)
def estimator_head_distribution_regression(make_distribution_fn,
                                           label_dimension=1,
                                           logits_dimension=None,
                                           label_name=None,
                                           weight_column_name=None,
                                           enable_centered_bias=False,
                                           head_name=None):
  """Creates a `Head` for regression under a generic distribution.

  Args:
    make_distribution_fn: Python `callable` which returns a `tf.Distribution`
      instance created using only logits.
    label_dimension: Number of regression labels per example. This is the size
      of the last dimension of the labels `Tensor` (typically, this has shape
      `[batch_size, label_dimension]`).
    logits_dimension: Number of logits per example. This is the size of the last
      dimension of the logits `Tensor` (typically, this has shape
      `[batch_size, logits_dimension]`).
      Default value: `label_dimension`.
    label_name: Python `str`, name of the key in label `dict`. Can be `None` if
      label is a `Tensor` (single headed models).
    weight_column_name: Python `str` defining feature column name representing
      weights. It is used to down weight or boost examples during training. It
      will be multiplied by the loss of the example.
    enable_centered_bias: Python `bool`. If `True`, estimator will learn a
      centered bias variable for each class. Rest of the model structure learns
      the residual after centered bias.
    head_name: Python `str`, name of the head. Predictions, summary and metrics
      keys are suffixed by `"/" + head_name` and the default variable scope is
      `head_name`.

  Returns:
    An instance of `Head` for generic regression.
  """
  return _DistributionRegressionHead(
      make_distribution_fn=make_distribution_fn,
      label_dimension=label_dimension,
      logits_dimension=logits_dimension,
      label_name=label_name,
      weight_column_name=weight_column_name,
      enable_centered_bias=enable_centered_bias,
      head_name=head_name)


class _DistributionRegressionHead(_RegressionHead):
  """Creates a _RegressionHead instance from an arbitrary `Distribution`."""

  @deprecation.deprecated(
      "2018-10-01",
      "The TensorFlow Distributions library has moved to "
      "TensorFlow Probability "
      "(https://github.com/tensorflow/probability). You "
      "should update all references to use `tfp.distributions` "
      "instead of `tf.contrib.distributions`.",
      warn_once=True)
  def __init__(self,
               make_distribution_fn,
               label_dimension,
               logits_dimension=None,
               label_name=None,
               weight_column_name=None,
               enable_centered_bias=False,
               head_name=None):
    """`Head` for regression.

    Args:
      make_distribution_fn: Python `callable` which returns a `tf.Distribution`
        instance created using only logits.
      label_dimension: Number of regression labels per example. This is the
        size of the last dimension of the labels `Tensor` (typically, this has
        shape `[batch_size, label_dimension]`).
      logits_dimension: Number of logits per example. This is the size of the
        last dimension of the logits `Tensor` (typically, this has shape
        `[batch_size, logits_dimension]`).
        Default value: `label_dimension`.
      label_name: Python `str`, name of the key in label `dict`. Can be `None`
        if label is a tensor (single headed models).
      weight_column_name: Python `str` defining feature column name representing
        weights. It is used to down weight or boost examples during training. It
        will be multiplied by the loss of the example.
      enable_centered_bias: Python `bool`. If `True`, estimator will learn a
        centered bias variable for each class. Rest of the model structure
        learns the residual after centered bias.
      head_name: Python `str`, name of the head. Predictions, summary and
        metrics keys are suffixed by `"/" + head_name` and the default variable
        scope is `head_name`.

    Raises:
      TypeError: if `make_distribution_fn` is not `callable`.
    """
    if not callable(make_distribution_fn):
      raise TypeError("`make_distribution_fn` must be a callable function.")

    self._distributions = {}
    self._make_distribution_fn = make_distribution_fn

    def static_value(x):
      """Returns the static value of a `Tensor` or `None`."""
      return tensor_util.constant_value(ops.convert_to_tensor(x))

    def concat_vectors(*args):
      """Concatenates input vectors, statically if possible."""
      args_ = [static_value(x) for x in args]
      if any(vec is None for vec in args_):
        return array_ops.concat(args, axis=0)
      return [val for vec in args_ for val in vec]

    def loss_fn(labels, logits, weights=None):
      """Returns the loss of using `logits` to predict `labels`."""
      d = self.distribution(logits)
      labels_batch_shape = labels.shape.with_rank_at_least(1)[:-1]
      labels_batch_shape = (
          labels_batch_shape.as_list() if labels_batch_shape.is_fully_defined()
          else array_ops.shape(labels)[:-1])
      labels = array_ops.reshape(
          labels,
          shape=concat_vectors(labels_batch_shape, d.event_shape_tensor()))
      return _compute_weighted_loss(
          loss_unweighted=-d.log_prob(labels),
          weight=weights)

    def link_fn(logits):
      """Returns the inverse link function at `logits`."""
      # Note: What the API calls a "link function" is really the inverse-link
      # function, i.e., the "mean".
      d = self.distribution(logits)
      return d.mean()

    super(_DistributionRegressionHead, self).__init__(
        label_dimension=label_dimension,
        loss_fn=loss_fn,
        link_fn=link_fn,
        logits_dimension=logits_dimension,
        label_name=label_name,
        weight_column_name=weight_column_name,
        enable_centered_bias=enable_centered_bias,
        head_name=head_name)

  @property
  def distributions(self):
    """Returns all distributions created by `DistributionRegressionHead`."""
    return self._distributions

  def distribution(self, logits, name=None):
    """Retrieves a distribution instance, parameterized by `logits`.

    Args:
      logits: `float`-like `Tensor` representing the parameters of the
        underlying distribution.
      name: The Python `str` name to given to this op.
        Default value: "distribution".

    Returns:
      distribution: `tf.Distribution` instance parameterized by `logits`.
    """
    with ops.name_scope(name, "distribution", [logits]):
      d = self._distributions.get(logits, None)
      if d is None:
        d = self._make_distribution_fn(logits)
        self._distributions[logits] = d
      return d

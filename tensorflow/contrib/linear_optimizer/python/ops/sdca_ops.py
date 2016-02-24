# Copyright 2016 Google Inc. All Rights Reserved.
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
"""Proximal stochastic dual coordinate ascent optimizer for linear models."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# pylint: disable=wildcard-import
from tensorflow.contrib.linear_optimizer.gen_sdca_ops import *
from tensorflow.python.framework import ops
from tensorflow.python.framework.ops import convert_to_tensor
from tensorflow.python.framework.ops import name_scope
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.nn import sigmoid_cross_entropy_with_logits

__all__ = ['SdcaModel']


class SdcaModel(object):
  """Stochastic dual coordinate ascent solver for linear models.

    This class currently only supports a single machine (multi-threaded)
    implementation. We expect the data, and weights to fit in a single machine.

    Loss functions supported:
     * Binary logistic loss

    This class defines an optimizer API to train a linear model.

    ### Usage

    ```python
    # Create a solver with the desired parameters.
    lr = tf.contrib.linear_optimizer.SdcaModel(examples, variables, options)
    opt_op = lr.minimize()

    predictions = lr.predictions(examples)
    # Primal loss + L1 loss + L2 loss.
    regularized_loss = lr.regularized_loss(examples)
    # Primal loss only
    unregularized_loss = lr.unregularized_loss(examples)

    examples: {
      sparse_features: list of SparseTensors of value type float32.
      dense_features: list of dense tensors of type float32.
      example_labels: a tensor of of shape [Num examples]
      example_weights: a tensor of shape [Num examples]
    }
    variables: {
      sparse_features_weights: list of tensors of shape [vocab size]
      dense_features_weights: list of tensors of shape [1]
      dual: tensor of shape [Num examples]
    }
    options: {
      symmetric_l1_regularization: 0.0
      symmetric_l2_regularization: 1.0
      loss_type: "logistic_loss"
    }
    ```

    In the training program you will just have to run the returned Op from
    minimize().

    ```python
    # Execute opt_op once to perform training, which continues until
    convergence.
      The op makes use of duality gap as a certificate for termination. Duality
      gap is set to 0.01 as default.
    opt_op.run()
    ```
    """

  def __init__(self, examples, variables, options):
    """Create a new sdca optimizer."""

    if not examples or not variables or not options:
      raise ValueError('All arguments must be specified.')

    if options['loss_type'] != 'logistic_loss':
      raise ValueError('Optimizer only supports logistic regression (for now).')

    self._assertSpecified(
        ['example_labels', 'example_weights', 'sparse_features',
         'dense_features'], examples)
    self._assertList(['sparse_features', 'dense_features'], examples)

    self._assertSpecified(
        ['sparse_features_weights', 'dense_features_weights',
         'training_log_loss', 'dual'], variables)
    self._assertList(
        ['sparse_features_weights', 'dense_features_weights'], variables)

    self._assertSpecified(
        ['loss_type', 'symmetric_l2_regularization',
         'symmetric_l1_regularization'], options)

    self._examples = examples
    self._variables = variables
    self._options = options
    self._training_log_loss = convert_to_tensor(
        self._variables['training_log_loss'],
        as_ref=True)

  def _assertSpecified(self, items, check_in):
    for x in items:
      if check_in[x] is None:
        raise ValueError(check_in[x] + ' must be specified.')

  def _assertList(self, items, check_in):
    for x in items:
      if not isinstance(check_in[x], list):
        raise ValueError(x + ' must be a list.')

  def _l1_loss(self):
    """"Computes the l1 loss of the model."""
    with name_scope('l1_loss'):
      sparse_weights = self._convert_n_to_tensor(self._variables[
          'sparse_features_weights'])
      dense_weights = self._convert_n_to_tensor(self._variables[
          'dense_features_weights'])
      l1 = self._options['symmetric_l1_regularization']
      loss = 0
      for w in sparse_weights:
        loss += l1 * math_ops.reduce_sum(abs(w))
      for w in dense_weights:
        loss += l1 * math_ops.reduce_sum(abs(w))
      return loss

  def _l2_loss(self):
    """"Computes the l1 loss of the model."""
    with name_scope('l2_loss'):
      sparse_weights = self._convert_n_to_tensor(self._variables[
          'sparse_features_weights'])
      dense_weights = self._convert_n_to_tensor(self._variables[
          'dense_features_weights'])
      l2 = self._options['symmetric_l2_regularization']
      loss = 0
      for w in sparse_weights:
        loss += l2 * math_ops.reduce_sum(math_ops.square(w))
      for w in dense_weights:
        loss += l2 * math_ops.reduce_sum(math_ops.square(w))
      return loss

  def _logits(self, examples):
    """Compute logits for each example."""
    with name_scope('logits'):
      sparse_variables = self._convert_n_to_tensor(self._variables[
          'sparse_features_weights'])
      logits = 0
      for st_i, sv in zip(examples['sparse_features'], sparse_variables):
        ei, fi = array_ops.split(1, 2, st_i.indices)
        ei = array_ops.reshape(ei, [-1])
        fi = array_ops.reshape(fi, [-1])
        fv = array_ops.reshape(st_i.values, [-1])
        # TODO(rohananil): This does not work if examples have empty
        # features.
        logits += math_ops.segment_sum(
            math_ops.mul(
                array_ops.gather(sv, fi), fv), array_ops.reshape(ei, [-1]))
      dense_features = self._convert_n_to_tensor(examples['dense_features'])
      dense_variables = self._convert_n_to_tensor(self._variables[
          'dense_features_weights'])
      for i in xrange(len(dense_variables)):
        logits += dense_features[i] * dense_variables[i]
      return logits

  def _convert_n_to_tensor(self, input_list, as_ref=False):
    """Converts input list to a set of tensors."""
    return [convert_to_tensor(x, as_ref=as_ref) for x in input_list]

  def predictions(self, examples):
    """Add operations to compute predictions by the model.

        Args:
          examples: Examples to compute prediction on.

        Returns:
          An Operation that computes the predictions for examples. For logistic
          loss
          output is a tensor with sigmoid output.
        Raises:
          ValueError: if examples are not well defined.
        """
    self._assertSpecified(
        ['example_weights', 'sparse_features', 'dense_features'], examples)
    self._assertList(['sparse_features', 'dense_features'], examples)
    with name_scope('sdca/prediction'):
      logits = self._logits(examples)
      # TODO(rohananil): Change prediction when supporting linear
      # regression.
      return math_ops.sigmoid(logits)

  def minimize(self):
    """Add operations to train a linear model by minimizing the loss function.

        Returns:
          An Operation that updates the variables passed in the constructor.
        """
    with name_scope('sdca/minimize'):
      sparse_features_indices = []
      sparse_features_weights = []
      for sf in self._examples['sparse_features']:
        sparse_features_indices.append(ops.convert_to_tensor(sf.indices))
        sparse_features_weights.append(ops.convert_to_tensor(sf.values))

      return sdca_solver(
          sparse_features_indices,
          sparse_features_weights,
          self._convert_n_to_tensor(self._examples['dense_features']),
          convert_to_tensor(self._examples['example_weights']),
          convert_to_tensor(self._examples['example_labels']),
          convert_to_tensor(self._variables['dual'],
                            as_ref=True),
          self._convert_n_to_tensor(self._variables[
              'sparse_features_weights'],
                                    as_ref=True),
          self._convert_n_to_tensor(self._variables['dense_features_weights'],
                                    as_ref=True),
          self._training_log_loss,
          L1=self._options['symmetric_l1_regularization'],
          L2=self._options['symmetric_l2_regularization'],
          LossType=self._options['loss_type'])

  def unregularized_loss(self, examples):
    """Add operations to compute the loss (without the regularization loss).

        Args:
          examples: Examples to compute unregularized loss on.

        Returns:
          An Operation that computes mean (unregularized) loss for given set of
          examples.
        Raises:
          ValueError: if examples are not well defined.
        """
    self._assertSpecified(
        ['example_labels', 'example_weights', 'sparse_features',
         'dense_features'], examples)
    self._assertList(['sparse_features', 'dense_features'], examples)
    with name_scope('sdca/unregularized_loss'):
      logits = self._logits(examples)
      # TODO(rohananil): Change loss when supporting linear regression.
      return math_ops.reduce_sum(math_ops.mul(
          sigmoid_cross_entropy_with_logits(logits, convert_to_tensor(examples[
              'example_labels'])), convert_to_tensor(examples[
                  'example_weights']))) / math_ops.reduce_sum(
                      ops.convert_to_tensor(examples['example_weights']))

  def regularized_loss(self, examples):
    """Add operations to compute the loss with regularization loss included.

        Args:
          examples: Examples to compute loss on.

        Returns:
          An Operation that computes mean (regularized) loss for given set of
          examples.
        Raises:
          ValueError: if examples are not well defined.
        """
    self._assertSpecified(
        ['example_labels', 'example_weights', 'sparse_features',
         'dense_features'], examples)
    self._assertList(['sparse_features', 'dense_features'], examples)
    with name_scope('sdca/regularized_loss'):
      logits = self._logits(examples)
      # TODO(rohananil): Change loss when supporting linear regression.
      return self._l1_loss() + self._l2_loss() + self.unregularized_loss(
          examples)

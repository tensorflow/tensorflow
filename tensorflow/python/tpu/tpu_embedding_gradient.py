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
# ===================================================================
"""Optional helper for gradient handling."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.tpu.ops import tpu_ops


def get_gradients_through_compute_gradients(optimizer, loss, activations):
  """Compute gradients to send to TPU embedding.

  Args:
    optimizer: a subclass of optimizer.Optimizer, usually CrossShardOptimizer.
      Used to call compute_gradients().
    loss: a Tensor to call optimizer.compute_gradients() on.
    activations: an OrderedDict mapping feature_name to Tensors of activations.

  Returns:
    An OrderedDict mapping from feature name Strings to Tensors of gradients of
      the loss wrt the activations of the features.
  """
  activation_list = activations.values()
  grads_and_vars = optimizer.compute_gradients(loss, activation_list)
  grads = [grad for grad, _ in grads_and_vars]
  feature_to_gradient_dict = collections.OrderedDict(
      zip(activations.keys(), grads))
  return feature_to_gradient_dict


def create_dummy_table_variables(tpu_embedding):
  """Create dummy embedding table variables.

  The sole purpose of these dummy variables are to trigger gradient
  calculation wrt them so that the gradients wrt activation can be captured
  and later sent to TPU embedding.

  Args:
    tpu_embedding: TPUEmbedding, dummy table variables will be created for use
      with tpu_embedding.

  Returns:
    A tuple of dummy variables and their initializer.

  Raises:
    RuntimeError: if collection to store gradients already exists and is not
    empty.
  """
  dummy_table_variables = collections.OrderedDict()
  for table_id, table in enumerate(tpu_embedding.table_to_features_dict):
    dummy_table_variables[table] = (
        # Explicitly specifying collections prevents this variable from
        # being added to the GLOBAL_VARIABLES collection, so that Saver()
        # ignores it.
        # But Tensorflow optimizer creates slot variable for these dummy
        # variable, e.g. tpu_embedding_dummy_table_variable_mlp_user/Adam{_1},
        # which will be in GLOBAL_VARIABLES collection,
        variable_scope.get_variable(
            'tpu_embedding_dummy_table_variable_{}'.format(table),
            dtype=dtypes.float32,
            shape=[1],
            use_resource=True,
            trainable=True,
            collections=['tpu_embedding_dummy_table_variables']))

    g = ops.get_default_graph()
    table_gradients = g.get_collection_ref(
        'tpu_embedding_gradients_table_{}'.format(table_id))
    if table_gradients:
      raise RuntimeError(
          'tpu_embedding_gradients_table_{} is not empty.'.format(table_id))
    num_features = len(tpu_embedding.table_to_features_dict[table])
    table_gradients.extend([None for _ in range(num_features)])

  return (dummy_table_variables,
          variables.variables_initializer(
              dummy_table_variables.values(),
              name='tpu_embedding_dummy_table_variables_init'))


def hook_dummy_table_variables_to_activations(tpu_embedding, activations,
                                              dummy_table_variables):
  """Have activations depend on dummy table variables for gradient intercept.

  Args:
    tpu_embedding: TPUEmbedding, activations and dummy_table_variables are from
      tpu_embedding.
    activations: An OrderedDict of feature name String to activation tensors.
    dummy_table_variables: An OrderedDict of table name String to dummy table
      variables.

  Returns:
    An OrderedDict of feature name String to activation tensors, which can be
      used just as the activations input.
  """
  new_activations = collections.OrderedDict()
  for feature in activations:
    table = tpu_embedding.feature_to_config_dict[feature].table_id
    new_activations[feature] = tpu_ops.tpu_embedding_activations(
        dummy_table_variables[table],
        activations[feature],
        table_id=list(tpu_embedding.table_to_config_dict).index(table),
        lookup_id=tpu_embedding.table_to_features_dict[table].index(feature))
  return new_activations


def get_gradients_through_dummy_table_variables(tpu_embedding):
  """Get gradients wrt the activations of each feature.

  Args:
    tpu_embedding: TPUEmbedding, create dummy table variable to be used with
      tpu_embedding.

  Returns:
    An OrderedDict mapping feature name to gradient.

  Raises:
    ValueError: if some gradients are not defined.
  """
  g = ops.get_default_graph()
  gradients_found = False
  for table_id, table in enumerate(tpu_embedding.table_to_config_dict):
    table_gradients = g.get_collection(
        'tpu_embedding_gradients_table_{}'.format(table_id))
    if any(gradient is None for gradient in table_gradients):
      # TODO(bfontain): create a white-list for optimizers which are compatible
      # with `tf.stop_gradient`.
      logging.warn(
          'Table {} with id {} has undefined gradients: this is probably '
          'because the model asked TPUEmbedding to compute activations that '
          'were not used, or tf.stop_gradient() is applied. Gradients of zeros '
          'are sent back to TPUEmbedding instead. Gradients of zeros and no '
          'gradients are equivalent for SGD, AdaGrad, FTRL, etc, but '
          'might differ for other optimizers due to implementation of TPU '
          'embedding optimizers.'.format(table, table_id))
    gradients_found = gradients_found or any(
        gradient is not None for gradient in table_gradients)

  if not gradients_found:
    logging.warn(
        'All tables have undefined gradients: this is probably because the '
        'model asked TPUEmbedding to compute activations that were not used. '
        'If all TPUEmbedding features have stop_gradients, consider using the '
        'INFERENCE mode instead.')

  feature_to_gradient_dict = collections.OrderedDict()
  for table_id, table in enumerate(tpu_embedding.table_to_config_dict):
    table_gradients = g.get_collection(
        'tpu_embedding_gradients_table_{}'.format(table_id))
    for feature, gradient in zip(tpu_embedding.table_to_features_dict[table],
                                 table_gradients):
      if gradient is not None:
        feature_to_gradient_dict[feature] = gradient
      else:
        dimension = tpu_embedding.table_to_config_dict[table].dimension
        batch_size = tpu_embedding.batch_size_per_core
        max_sequence_length = (
            tpu_embedding.feature_to_config_dict[feature].max_sequence_length)
        if max_sequence_length:
          feature_to_gradient_dict[feature] = array_ops.zeros(
              [batch_size, max_sequence_length, dimension])
        else:
          feature_to_gradient_dict[feature] = array_ops.zeros(
              [batch_size, dimension])

  return feature_to_gradient_dict

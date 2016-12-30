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
"""Treats a decision tree as a representation transformation layer.

A decision tree transformer takes features as input and returns the probability
of reaching each leaf as output.  The routing throughout the tree is learnable
via backpropagation.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.tensor_forest.hybrid.python import hybrid_layer
from tensorflow.contrib.tensor_forest.hybrid.python.ops import training_ops
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import variable_scope


class DecisionsToDataLayer(hybrid_layer.HybridLayer):
  """A layer that treats soft decisions as data."""

  def _define_vars(self, params, **kwargs):
    with ops.device(self.device_assigner.get_device(self.layer_num)):

      self.tree_parameters = variable_scope.get_variable(
          name='tree_parameters_%d' % self.layer_num,
          shape=[params.num_nodes, params.num_features],
          initializer=init_ops.truncated_normal_initializer(
              mean=params.weight_init_mean, stddev=params.weight_init_std))

      self.tree_thresholds = variable_scope.get_variable(
          name='tree_thresholds_%d' % self.layer_num,
          shape=[params.num_nodes],
          initializer=init_ops.truncated_normal_initializer(
              mean=params.weight_init_mean, stddev=params.weight_init_std))

  def __init__(self, params, layer_num, device_assigner,
               *args, **kwargs):
    super(DecisionsToDataLayer, self).__init__(
        params, layer_num, device_assigner, *args, **kwargs)

    self.training_ops = training_ops.Load()

  def inference_graph(self, data):
    with ops.device(self.device_assigner.get_device(self.layer_num)):
      routing_probabilities = self.training_ops.routing_function(
          data,
          self.tree_parameters,
          self.tree_thresholds,
          max_nodes=self.params.num_nodes)

      output = array_ops.slice(
          routing_probabilities,
          [0, self.params.num_nodes - self.params.num_leaves - 1],
          [-1, self.params.num_leaves])

      return output


class KFeatureDecisionsToDataLayer(hybrid_layer.HybridLayer):
  """A layer that treats soft decisions made on single features as data."""

  def _define_vars(self, params, **kwargs):
    with ops.device(self.device_assigner.get_device(self.layer_num)):

      self.tree_parameters = variable_scope.get_variable(
          name='tree_parameters_%d' % self.layer_num,
          shape=[params.num_nodes, params.num_features_per_node],
          initializer=init_ops.truncated_normal_initializer(
              mean=params.weight_init_mean, stddev=params.weight_init_std))

      self.tree_thresholds = variable_scope.get_variable(
          name='tree_thresholds_%d' % self.layer_num,
          shape=[params.num_nodes],
          initializer=init_ops.truncated_normal_initializer(
              mean=params.weight_init_mean, stddev=params.weight_init_std))

  def __init__(self, params, layer_num, device_assigner,
               *args, **kwargs):
    super(KFeatureDecisionsToDataLayer, self).__init__(
        params, layer_num, device_assigner, *args, **kwargs)

    self.training_ops = training_ops.Load()

  # pylint: disable=unused-argument
  def inference_graph(self, data):
    with ops.device(self.device_assigner.get_device(self.layer_num)):
      routing_probabilities = self.training_ops.k_feature_routing_function(
          data,
          self.tree_parameters,
          self.tree_thresholds,
          max_nodes=self.params.num_nodes,
          num_features_per_node=self.params.num_features_per_node,
          layer_num=0,
          random_seed=self.params.base_random_seed)

      output = array_ops.slice(
          routing_probabilities,
          [0, self.params.num_nodes - self.params.num_leaves - 1],
          [-1, self.params.num_leaves])

      return output


class HardDecisionsToDataLayer(DecisionsToDataLayer):
  """A layer that learns a soft decision tree but treats it as hard at test."""

  def _define_vars(self, params, **kwargs):
    with ops.device(self.device_assigner.get_device(self.layer_num)):

      self.tree_parameters = variable_scope.get_variable(
          name='hard_tree_parameters_%d' % self.layer_num,
          shape=[params.num_nodes, params.num_features],
          initializer=variable_scope.truncated_normal_initializer(
              mean=params.weight_init_mean, stddev=params.weight_init_std))

      self.tree_thresholds = variable_scope.get_variable(
          name='hard_tree_thresholds_%d' % self.layer_num,
          shape=[params.num_nodes],
          initializer=variable_scope.truncated_normal_initializer(
              mean=params.weight_init_mean, stddev=params.weight_init_std))

  def soft_inference_graph(self, data):
    return super(HardDecisionsToDataLayer, self).inference_graph(data)

  def inference_graph(self, data):
    with ops.device(self.device_assigner.get_device(self.layer_num)):
      path_probability, path = self.training_ops.hard_routing_function(
          data,
          self.tree_parameters,
          self.tree_thresholds,
          max_nodes=self.params.num_nodes,
          tree_depth=self.params.hybrid_tree_depth)

      output = array_ops.slice(
          self.training_ops.unpack_path(path, path_probability),
          [0, self.params.num_nodes - self.params.num_leaves - 1],
          [-1, self.params.num_leaves])

      return output


class StochasticHardDecisionsToDataLayer(HardDecisionsToDataLayer):
  """A layer that learns a soft decision tree by sampling paths."""

  def _define_vars(self, params, **kwargs):
    with ops.device(self.device_assigner.get_device(self.layer_num)):

      self.tree_parameters = variable_scope.get_variable(
          name='stochastic_hard_tree_parameters_%d' % self.layer_num,
          shape=[params.num_nodes, params.num_features],
          initializer=init_ops.truncated_normal_initializer(
              mean=params.weight_init_mean, stddev=params.weight_init_std))

      self.tree_thresholds = variable_scope.get_variable(
          name='stochastic_hard_tree_thresholds_%d' % self.layer_num,
          shape=[params.num_nodes],
          initializer=init_ops.truncated_normal_initializer(
              mean=params.weight_init_mean, stddev=params.weight_init_std))

  def soft_inference_graph(self, data):
    with ops.device(self.device_assigner.get_device(self.layer_num)):
      path_probability, path = (
          self.training_ops.stochastic_hard_routing_function(
              data,
              self.tree_parameters,
              self.tree_thresholds,
              tree_depth=self.params.hybrid_tree_depth,
              random_seed=self.params.base_random_seed))

      output = array_ops.slice(
          self.training_ops.unpack_path(path, path_probability),
          [0, self.params.num_nodes - self.params.num_leaves - 1],
          [-1, self.params.num_leaves])

      return output

  def inference_graph(self, data):
    with ops.device(self.device_assigner.get_device(self.layer_num)):
      path_probability, path = self.training_ops.hard_routing_function(
          data,
          self.tree_parameters,
          self.tree_thresholds,
          max_nodes=self.params.num_nodes,
          tree_depth=self.params.hybrid_tree_depth)

      output = array_ops.slice(
          self.training_ops.unpack_path(path, path_probability),
          [0, self.params.num_nodes - self.params.num_leaves - 1],
          [-1, self.params.num_leaves])

      return output


class StochasticSoftDecisionsToDataLayer(StochasticHardDecisionsToDataLayer):
  """A layer that learns a soft decision tree by sampling paths."""

  def _define_vars(self, params, **kwargs):
    with ops.device(self.device_assigner.get_device(self.layer_num)):

      self.tree_parameters = variable_scope.get_variable(
          name='stochastic_soft_tree_parameters_%d' % self.layer_num,
          shape=[params.num_nodes, params.num_features],
          initializer=init_ops.truncated_normal_initializer(
              mean=params.weight_init_mean, stddev=params.weight_init_std))

      self.tree_thresholds = variable_scope.get_variable(
          name='stochastic_soft_tree_thresholds_%d' % self.layer_num,
          shape=[params.num_nodes],
          initializer=init_ops.truncated_normal_initializer(
              mean=params.weight_init_mean, stddev=params.weight_init_std))

  def inference_graph(self, data):
    with ops.device(self.device_assigner.get_device(self.layer_num)):
      routes = self.training_ops.routing_function(
          data,
          self.tree_parameters,
          self.tree_thresholds,
          max_nodes=self.params.num_nodes)

      leaf_routes = array_ops.slice(
          routes, [0, self.params.num_nodes - self.params.num_leaves - 1],
          [-1, self.params.num_leaves])

      return leaf_routes

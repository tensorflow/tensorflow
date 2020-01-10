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
"""HierarchicalController Class.

The HierarchicalController encompasses the entire lifecycle of training the
device placement policy, including generating op embeddings, getting groups for
each op, placing those groups and running the predicted placements.

Different assignment models can inherit from this class.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import numpy as np
import six
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops as tf_ops
from tensorflow.python.grappler.controller import Controller
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.summary import summary
from tensorflow.python.training import adam
from tensorflow.python.training import gradient_descent
from tensorflow.python.training import learning_rate_decay
from tensorflow.python.training import training_util


class PlacerParams(object):
  """Class to hold a set of placement parameters as name-value pairs.

  A typical usage is as follows:

  ```python
  # Create a PlacerParams object specifying names and values of the model
  # parameters:
  params = PlacerParams(hidden_size=128, decay_steps=50)

  # The parameters are available as attributes of the PlacerParams object:
  hparams.hidden_size ==> 128
  hparams.decay_steps ==> 50
  ```

  """

  def __init__(self, **kwargs):
    """Create an instance of `PlacerParams` from keyword arguments.

    The keyword arguments specify name-values pairs for the parameters.
    The parameter types are inferred from the type of the values passed.

    The parameter names are added as attributes of `PlacerParams` object,
    and they can be accessed directly with the dot notation `params._name_`.

    Example:

    ```python
    # Define 1 parameter: 'hidden_size'
    params = PlacerParams(hidden_size=128)
    params.hidden_size ==> 128
    ```

    Args:
      **kwargs: Key-value pairs where the key is the parameter name and
        the value is the value for the parameter.
    """
    for name, value in six.iteritems(kwargs):
      self.add_param(name, value)

  def add_param(self, name, value):
    """Adds {name, value} pair to hyperparameters.

    Args:
      name: Name of the hyperparameter.
      value: Value of the hyperparameter. Can be one of the following types:
        int, float, string, int list, float list, or string list.

    Raises:
      ValueError: if one of the arguments is invalid.
    """
    # Keys in kwargs are unique, but 'name' could be the name of a pre-existing
    # attribute of this object.  In that case we refuse to use it as a
    # parameter name.
    if getattr(self, name, None) is not None:
      raise ValueError("Parameter name is reserved: %s" % name)
    setattr(self, name, value)


def hierarchical_controller_hparams():
  """Hyperparameters for hierarchical planner."""
  return PlacerParams(
      hidden_size=512,
      forget_bias_init=1.0,
      temperature=1.0,
      logits_std_noise=0.5,
      stop_noise_step=750,
      decay_steps=50,
      max_num_outputs=5,
      max_output_size=5,
      tanh_constant=1.0,
      adj_embed_dim=20,
      grouping_hidden_size=64,
      num_groups=None,
      bi_lstm=True,
      failing_signal=100,
      stop_sampling=500,
      start_with_failing_signal=True,
      always_update_baseline=False,
      bl_dec=0.9,
      grad_bound=1.0,
      lr=0.1,
      lr_dec=0.95,
      start_decay_step=400,
      optimizer_type="adam",
      stop_updating_after_steps=1000,
      name="hierarchical_controller",
      keep_prob=1.0,
      reward_function="sqrt",
      seed=1234,
      # distributed training params
      num_children=1)


class HierarchicalController(Controller):
  """HierarchicalController class."""

  def __init__(self, hparams, item, cluster, controller_id=0):
    """HierarchicalController class initializer.

    Args:
      hparams: All hyper-parameters.
      item: The metagraph to place.
      cluster: The cluster of hardware devices to optimize for.
      controller_id: the id of the controller in a multi-controller setup.
    """
    super(HierarchicalController, self).__init__(item, cluster)
    self.ctrl_id = controller_id
    self.hparams = hparams

    if self.hparams.num_groups is None:
      self.num_groups = min(256, 20 * self.num_devices)
    else:
      self.num_groups = self.hparams.num_groups

    # creates self.op_embeddings and self.type_dict
    self.create_op_embeddings(verbose=False)
    # TODO(azalia) clean up embedding/group_embedding_size names
    self.group_emb_size = (
        2 * self.num_groups + len(self.type_dict) +
        self.hparams.max_num_outputs * self.hparams.max_output_size)
    self.embedding_size = self.group_emb_size
    self.initializer = init_ops.glorot_uniform_initializer(
        seed=self.hparams.seed)

    with variable_scope.variable_scope(
        self.hparams.name,
        initializer=self.initializer,
        reuse=variable_scope.AUTO_REUSE):
      # define parameters of feedforward
      variable_scope.get_variable("w_grouping_ff", [
          1 + self.hparams.max_num_outputs * self.hparams.max_output_size +
          self.hparams.adj_embed_dim, self.hparams.grouping_hidden_size
      ])
      variable_scope.get_variable(
          "w_grouping_softmax",
          [self.hparams.grouping_hidden_size, self.num_groups])
      if self.hparams.bi_lstm:
        variable_scope.get_variable("encoder_lstm_forward", [
            self.embedding_size + self.hparams.hidden_size / 2,
            2 * self.hparams.hidden_size
        ])
        variable_scope.get_variable("encoder_lstm_backward", [
            self.embedding_size + self.hparams.hidden_size / 2,
            2 * self.hparams.hidden_size
        ])
        variable_scope.get_variable(
            "device_embeddings", [self.num_devices, self.hparams.hidden_size])
        variable_scope.get_variable(
            "decoder_lstm",
            [2 * self.hparams.hidden_size, 4 * self.hparams.hidden_size])
        variable_scope.get_variable(
            "device_softmax", [2 * self.hparams.hidden_size, self.num_devices])
        variable_scope.get_variable("device_go_embedding",
                                    [1, self.hparams.hidden_size])
        variable_scope.get_variable(
            "encoder_forget_bias",
            shape=1,
            dtype=dtypes.float32,
            initializer=init_ops.constant_initializer(
                self.hparams.forget_bias_init))
        variable_scope.get_variable(
            "decoder_forget_bias",
            shape=1,
            dtype=dtypes.float32,
            initializer=init_ops.constant_initializer(
                self.hparams.forget_bias_init))
        variable_scope.get_variable(
            "attn_w_1", [self.hparams.hidden_size, self.hparams.hidden_size])
        variable_scope.get_variable(
            "attn_w_2", [self.hparams.hidden_size, self.hparams.hidden_size])
        variable_scope.get_variable("attn_v", [self.hparams.hidden_size, 1])

      else:
        variable_scope.get_variable("encoder_lstm", [
            self.embedding_size + self.hparams.hidden_size,
            4 * self.hparams.hidden_size
        ])
        variable_scope.get_variable(
            "device_embeddings", [self.num_devices, self.hparams.hidden_size])
        variable_scope.get_variable(
            "decoder_lstm",
            [2 * self.hparams.hidden_size, 4 * self.hparams.hidden_size])
        variable_scope.get_variable(
            "device_softmax", [2 * self.hparams.hidden_size, self.num_devices])
        variable_scope.get_variable("device_go_embedding",
                                    [1, self.hparams.hidden_size])
        variable_scope.get_variable(
            "encoder_forget_bias",
            shape=1,
            dtype=dtypes.float32,
            initializer=init_ops.constant_initializer(
                self.hparams.forget_bias_init))
        variable_scope.get_variable(
            "decoder_forget_bias",
            shape=1,
            dtype=dtypes.float32,
            initializer=init_ops.constant_initializer(
                self.hparams.forget_bias_init))
        variable_scope.get_variable(
            "attn_w_1", [self.hparams.hidden_size, self.hparams.hidden_size])
        variable_scope.get_variable(
            "attn_w_2", [self.hparams.hidden_size, self.hparams.hidden_size])
        variable_scope.get_variable("attn_v", [self.hparams.hidden_size, 1])
    seq2seq_input_layer = array_ops.placeholder_with_default(
        array_ops.zeros([self.hparams.num_children,
                         self.num_groups,
                         self.group_emb_size],
                        dtypes.float32),
        shape=(self.hparams.num_children, self.num_groups, self.group_emb_size))
    self.seq2seq_input_layer = seq2seq_input_layer

  def compute_reward(self, run_time):
    if self.hparams.reward_function == "id":
      reward = run_time
    elif self.hparams.reward_function == "sqrt":
      reward = math.sqrt(run_time)
    elif self.hparams.reward_function == "log":
      reward = math.log1p(run_time)
    else:
      raise NotImplementedError(
          "Unrecognized reward function '%s', consider your "
          "--reward_function flag value." % self.hparams.reward_function)
    return reward

  def build_controller(self):
    """RL optimization interface.

    Returns:
      ops: A dictionary holding handles of the model used for training.
    """

    self._global_step = training_util.get_or_create_global_step()
    ops = {}
    ops["loss"] = 0

    failing_signal = self.compute_reward(self.hparams.failing_signal)

    ctr = {}

    with tf_ops.name_scope("controller_{}".format(self.ctrl_id)):
      with variable_scope.variable_scope("controller_{}".format(self.ctrl_id)):
        ctr["reward"] = {"value": [], "ph": [], "update": []}
        ctr["ready"] = {"value": [], "ph": [], "update": []}
        ctr["best_reward"] = {"value": [], "update": []}
        for i in range(self.hparams.num_children):
          reward_value = variable_scope.get_local_variable(
              "reward_{}".format(i),
              initializer=0.0,
              dtype=dtypes.float32,
              trainable=False)
          reward_ph = array_ops.placeholder(
              dtypes.float32, shape=(), name="reward_ph_{}".format(i))
          reward_update = state_ops.assign(
              reward_value, reward_ph, use_locking=True)
          ctr["reward"]["value"].append(reward_value)
          ctr["reward"]["ph"].append(reward_ph)
          ctr["reward"]["update"].append(reward_update)
          best_reward = variable_scope.get_local_variable(
              "best_reward_{}".format(i),
              initializer=failing_signal,
              dtype=dtypes.float32,
              trainable=False)
          ctr["best_reward"]["value"].append(best_reward)
          ctr["best_reward"]["update"].append(
              state_ops.assign(best_reward,
                               math_ops.minimum(best_reward, reward_update)))

          ready_value = variable_scope.get_local_variable(
              "ready_{}".format(i),
              initializer=True,
              dtype=dtypes.bool,
              trainable=False)
          ready_ph = array_ops.placeholder(
              dtypes.bool, shape=(), name="ready_ph_{}".format(i))
          ready_update = state_ops.assign(
              ready_value, ready_ph, use_locking=True)
          ctr["ready"]["value"].append(ready_value)
          ctr["ready"]["ph"].append(ready_ph)
          ctr["ready"]["update"].append(ready_update)

      ctr["grouping_y_preds"], ctr["grouping_log_probs"] = self.get_groupings()
      summary.histogram(
          "grouping_actions",
          array_ops.slice(ctr["grouping_y_preds"]["sample"], [0, 0],
                          [1, array_ops.shape(self.op_embeddings)[0]]))

      with variable_scope.variable_scope("controller_{}".format(self.ctrl_id)):
        ctr["baseline"] = variable_scope.get_local_variable(
            "baseline",
            initializer=failing_signal
            if self.hparams.start_with_failing_signal else 0.0,
            dtype=dtypes.float32,
            trainable=False)

      new_baseline = self.hparams.bl_dec * ctr["baseline"] + (
          1 - self.hparams.bl_dec) * math_ops.reduce_mean(
              ctr["reward"]["value"])
      if not self.hparams.always_update_baseline:
        baseline_mask = math_ops.less(ctr["reward"]["value"], failing_signal)
        selected_reward = array_ops.boolean_mask(ctr["reward"]["value"],
                                                 baseline_mask)
        selected_baseline = control_flow_ops.cond(
            math_ops.reduce_any(baseline_mask),
            lambda: math_ops.reduce_mean(selected_reward),
            lambda: constant_op.constant(0, dtype=dtypes.float32))
        ctr["pos_reward"] = selected_baseline
        pos_ = math_ops.less(
            constant_op.constant(0, dtype=dtypes.float32), selected_baseline)
        selected_baseline = self.hparams.bl_dec * ctr["baseline"] + (
            1 - self.hparams.bl_dec) * selected_baseline
        selected_baseline = control_flow_ops.cond(
            pos_, lambda: selected_baseline, lambda: ctr["baseline"])
        new_baseline = control_flow_ops.cond(
            math_ops.less(self.global_step,
                          self.hparams.stop_updating_after_steps),
            lambda: new_baseline, lambda: selected_baseline)
      ctr["baseline_update"] = state_ops.assign(
          ctr["baseline"], new_baseline, use_locking=True)

      ctr["y_preds"], ctr["log_probs"] = self.get_placements()
      summary.histogram("actions", ctr["y_preds"]["sample"])
      mask = math_ops.less(ctr["reward"]["value"], failing_signal)
      ctr["loss"] = ctr["reward"]["value"] - ctr["baseline"]
      ctr["loss"] *= (
          ctr["log_probs"]["sample"] + ctr["grouping_log_probs"]["sample"])

      selected_loss = array_ops.boolean_mask(ctr["loss"], mask)
      selected_loss = control_flow_ops.cond(
          math_ops.reduce_any(mask),
          lambda: math_ops.reduce_mean(-selected_loss),
          lambda: constant_op.constant(0, dtype=dtypes.float32))

      ctr["loss"] = control_flow_ops.cond(
          math_ops.less(self.global_step,
                        self.hparams.stop_updating_after_steps),
          lambda: math_ops.reduce_mean(-ctr["loss"]), lambda: selected_loss)

      ctr["reward_s"] = math_ops.reduce_mean(ctr["reward"]["value"])
      summary.scalar("loss", ctr["loss"])
      summary.scalar("avg_reward", ctr["reward_s"])
      summary.scalar("best_reward_so_far", best_reward)
      summary.scalar(
          "advantage",
          math_ops.reduce_mean(ctr["reward"]["value"] - ctr["baseline"]))

    with variable_scope.variable_scope(
        "optimizer", reuse=variable_scope.AUTO_REUSE):
      (ctr["train_op"], ctr["lr"], ctr["grad_norm"],
       ctr["grad_norms"]) = self._get_train_ops(
           ctr["loss"],
           tf_ops.get_collection(tf_ops.GraphKeys.TRAINABLE_VARIABLES),
           self.global_step,
           grad_bound=self.hparams.grad_bound,
           lr_init=self.hparams.lr,
           lr_dec=self.hparams.lr_dec,
           start_decay_step=self.hparams.start_decay_step,
           decay_steps=self.hparams.decay_steps,
           optimizer_type=self.hparams.optimizer_type)

    summary.scalar("gradnorm", ctr["grad_norm"])
    summary.scalar("lr", ctr["lr"])
    ctr["summary"] = summary.merge_all()
    ops["controller"] = ctr

    self.ops = ops
    return ops

  @property
  def global_step(self):
    return self._global_step

  def create_op_embeddings(self, verbose=False):
    if verbose:
      print("process input graph for op embeddings")
    self.num_ops = len(self.important_ops)
    # topological sort of important nodes
    topo_order = [op.name for op in self.important_ops]

    # create index to name for topologicaly sorted important nodes
    name_to_topo_order_index = {}
    for idx, x in enumerate(topo_order):
      name_to_topo_order_index[x] = idx
    self.name_to_topo_order_index = name_to_topo_order_index

    # create adj matrix
    adj_dict = {}
    for idx, op in enumerate(self.important_ops):
      for output_op in self.get_node_fanout(op):
        output_op_name = output_op.name
        if output_op_name in self.important_op_names:
          if name_to_topo_order_index[op.name] not in adj_dict:
            adj_dict[name_to_topo_order_index[op.name]] = []
          adj_dict[name_to_topo_order_index[op.name]].extend(
              [name_to_topo_order_index[output_op_name], 1])
          if output_op_name not in adj_dict:
            adj_dict[name_to_topo_order_index[output_op_name]] = []
          adj_dict[name_to_topo_order_index[output_op_name]].extend(
              [name_to_topo_order_index[op.name], -1])

    # get op_type op_output_shape, and adj info
    output_embed_dim = (self.hparams.max_num_outputs *
                        self.hparams.max_output_size)

    # TODO(bsteiner): don't filter based on used ops so that we can generalize
    # to models that use other types of ops.
    used_ops = set()
    for node in self.important_ops:
      op_type = str(node.op)
      used_ops.add(op_type)

    self.type_dict = {}
    for op_type in self.cluster.ListAvailableOps():
      if op_type in used_ops:
        self.type_dict[op_type] = len(self.type_dict)

    op_types = np.zeros([self.num_ops], dtype=np.int32)
    op_output_shapes = np.full(
        [self.num_ops, output_embed_dim], -1.0, dtype=np.float32)
    for idx, node in enumerate(self.important_ops):
      op_types[idx] = self.type_dict[node.op]
      # output shape
      op_name = node.name
      for i, output_prop in enumerate(self.node_properties[op_name]):
        if output_prop.shape.__str__() == "<unknown>":
          continue
        shape = output_prop.shape
        for j, dim in enumerate(shape.dim):
          if dim.size >= 0:
            if i * self.hparams.max_output_size + j >= output_embed_dim:
              break
            op_output_shapes[idx,
                             i * self.hparams.max_output_size + j] = dim.size
    # adj for padding
    op_adj = np.full(
        [self.num_ops, self.hparams.adj_embed_dim], 0, dtype=np.float32)
    for idx in adj_dict:
      neighbors = adj_dict[int(idx)]
      min_dim = min(self.hparams.adj_embed_dim, len(neighbors))
      padding_size = self.hparams.adj_embed_dim - min_dim
      neighbors = neighbors[:min_dim] + [0] * padding_size
      op_adj[int(idx)] = neighbors

    # op_embedding   starts here
    op_embeddings = np.zeros(
        [
            self.num_ops,
            1 + self.hparams.max_num_outputs * self.hparams.max_output_size +
            self.hparams.adj_embed_dim
        ],
        dtype=np.float32)
    for idx, op_name in enumerate(topo_order):
      op_embeddings[idx] = np.concatenate(
          (np.array([op_types[idx]]), op_output_shapes[idx], op_adj[int(idx)]))
    self.op_embeddings = constant_op.constant(
        op_embeddings, dtype=dtypes.float32)
    if verbose:
      print("num_ops = {}".format(self.num_ops))
      print("num_types = {}".format(len(self.type_dict)))

  def get_groupings(self, *args, **kwargs):
    num_children = self.hparams.num_children
    with variable_scope.variable_scope("controller_{}".format(self.ctrl_id)):
      grouping_actions_cache = variable_scope.get_local_variable(
          "grouping_actions_cache",
          initializer=init_ops.zeros_initializer,
          dtype=dtypes.int32,
          shape=[num_children, self.num_ops],
          trainable=False)
    input_layer = self.op_embeddings
    input_layer = array_ops.expand_dims(input_layer, 0)
    feed_ff_input_layer = array_ops.tile(input_layer, [num_children, 1, 1])
    grouping_actions, grouping_log_probs = {}, {}
    grouping_actions["sample"], grouping_log_probs[
        "sample"] = self.make_grouping_predictions(feed_ff_input_layer)

    grouping_actions["sample"] = state_ops.assign(grouping_actions_cache,
                                                  grouping_actions["sample"])
    self.grouping_actions_cache = grouping_actions_cache

    return grouping_actions, grouping_log_probs

  def make_grouping_predictions(self, input_layer, reuse=None):
    """model that predicts grouping (grouping_actions).

    Args:
      input_layer: group_input_layer
      reuse: reuse

    Returns:
       grouping_actions: actions
       grouping_log_probs: log probabilities corresponding to actions
    """
    with variable_scope.variable_scope(self.hparams.name, reuse=True):
      # input_layer: tensor of size [1, num_ops, hidden_size]
      w_grouping_ff = variable_scope.get_variable("w_grouping_ff")
      w_grouping_softmax = variable_scope.get_variable("w_grouping_softmax")

    batch_size = array_ops.shape(input_layer)[0]
    embedding_dim = array_ops.shape(input_layer)[2]

    reshaped = array_ops.reshape(input_layer,
                                 [batch_size * self.num_ops, embedding_dim])
    ff_output = math_ops.matmul(reshaped, w_grouping_ff)
    logits = math_ops.matmul(ff_output, w_grouping_softmax)
    if self.hparams.logits_std_noise > 0:
      num_in_logits = math_ops.cast(
          array_ops.size(logits), dtype=dtypes.float32)
      avg_norm = math_ops.divide(
          linalg_ops.norm(logits), math_ops.sqrt(num_in_logits))
      logits_noise = random_ops.random_normal(
          array_ops.shape(logits),
          stddev=self.hparams.logits_std_noise * avg_norm)
      logits = control_flow_ops.cond(
          self.global_step > self.hparams.stop_noise_step, lambda: logits,
          lambda: logits + logits_noise)
    logits = array_ops.reshape(logits,
                               [batch_size * self.num_ops, self.num_groups])
    actions = random_ops.multinomial(logits, 1, seed=self.hparams.seed)
    actions = math_ops.cast(actions, dtypes.int32)
    actions = array_ops.reshape(actions, [batch_size, self.num_ops])
    action_label = array_ops.reshape(actions, [-1])
    log_probs = nn_ops.sparse_softmax_cross_entropy_with_logits(
        logits=logits, labels=action_label)
    log_probs = array_ops.reshape(log_probs, [batch_size, -1])
    log_probs = math_ops.reduce_sum(log_probs, 1)
    grouping_actions = actions
    grouping_log_probs = log_probs
    return grouping_actions, grouping_log_probs

  def create_group_embeddings(self, grouping_actions, verbose=False):
    """Approximating the blocks of a TF graph from a graph_def.

    Args:
      grouping_actions: grouping predictions.
      verbose: print stuffs.

    Returns:
      groups: list of groups.
    """
    groups = [
        self._create_group_embeddings(grouping_actions, i, verbose) for
        i in range(self.hparams.num_children)
    ]
    return np.stack(groups, axis=0)

  def _create_group_embeddings(self, grouping_actions, child_id, verbose=False):
    """Approximating the blocks of a TF graph from a graph_def for each child.

    Args:
      grouping_actions: grouping predictions.
      child_id: child_id for the group.
      verbose: print stuffs.

    Returns:
      groups: group embedding for the child_id.
    """
    if verbose:
      print("Processing input_graph")

    # TODO(azalia): Build inter-adjacencies dag matrix.
    # record dag_matrix
    dag_matrix = np.zeros([self.num_groups, self.num_groups], dtype=np.float32)
    for op in self.important_ops:
      topo_op_index = self.name_to_topo_order_index[op.name]
      group_index = grouping_actions[child_id][topo_op_index]
      for output_op in self.get_node_fanout(op):
        if output_op.name not in self.important_op_names:
          continue
        output_group_index = (
            grouping_actions[child_id][self.name_to_topo_order_index[
                output_op.name]])
        dag_matrix[group_index, output_group_index] += 1.0
    num_connections = np.sum(dag_matrix)
    num_intra_group_connections = dag_matrix.trace()
    num_inter_group_connections = num_connections - num_intra_group_connections
    if verbose:
      print("grouping evaluation metric")
      print(("num_connections={} num_intra_group_connections={} "
             "num_inter_group_connections={}").format(
                 num_connections, num_intra_group_connections,
                 num_inter_group_connections))
    self.dag_matrix = dag_matrix

    # output_shape
    op_output_shapes = np.zeros(
        [
            len(self.important_ops),
            self.hparams.max_num_outputs * self.hparams.max_output_size
        ],
        dtype=np.float32)

    for idx, op in enumerate(self.important_ops):
      for i, output_properties in enumerate(self.node_properties[op.name]):
        if output_properties.shape.__str__() == "<unknown>":
          continue
        if i > self.hparams.max_num_outputs:
          break
        shape = output_properties.shape
        for j, dim in enumerate(shape.dim):
          if dim.size > 0:
            k = i * self.hparams.max_output_size + j
            if k >= self.hparams.max_num_outputs * self.hparams.max_output_size:
              break
            op_output_shapes[idx, k] = dim.size

    # group_embedding
    group_embedding = np.zeros(
        [
            self.num_groups, len(self.type_dict) +
            self.hparams.max_num_outputs * self.hparams.max_output_size
        ],
        dtype=np.float32)
    for op_index, op in enumerate(self.important_ops):
      group_index = grouping_actions[child_id][
          self.name_to_topo_order_index[op.name]]
      type_name = str(op.op)
      type_index = self.type_dict[type_name]
      group_embedding[group_index, type_index] += 1
      group_embedding[group_index, :self.hparams.max_num_outputs * self.hparams.
                      max_output_size] += (
                          op_output_shapes[op_index])
    grouping_adjacencies = np.concatenate(
        [dag_matrix, np.transpose(dag_matrix)], axis=1)
    group_embedding = np.concatenate(
        [grouping_adjacencies, group_embedding], axis=1)
    group_normalizer = np.amax(group_embedding, axis=1, keepdims=True)
    group_embedding /= (group_normalizer + 1.0)
    if verbose:
      print("Finished Processing Input Graph")
    return group_embedding

  def get_placements(self, *args, **kwargs):
    num_children = self.hparams.num_children
    with variable_scope.variable_scope("controller_{}".format(self.ctrl_id)):
      actions_cache = variable_scope.get_local_variable(
          "actions_cache",
          initializer=init_ops.zeros_initializer,
          dtype=dtypes.int32,
          shape=[num_children, self.num_groups],
          trainable=False)

    x = self.seq2seq_input_layer
    last_c, last_h, attn_mem = self.encode(x)
    actions, log_probs = {}, {}
    actions["sample"], log_probs["sample"] = (
        self.decode(
            x, last_c, last_h, attn_mem, mode="sample"))
    actions["target"], log_probs["target"] = (
        self.decode(
            x,
            last_c,
            last_h,
            attn_mem,
            mode="target",
            y=actions_cache))
    actions["greedy"], log_probs["greedy"] = (
        self.decode(
            x, last_c, last_h, attn_mem, mode="greedy"))
    actions["sample"] = control_flow_ops.cond(
        self.global_step < self.hparams.stop_sampling,
        lambda: state_ops.assign(actions_cache, actions["sample"]),
        lambda: state_ops.assign(actions_cache, actions["target"]))
    self.actions_cache = actions_cache

    return actions, log_probs

  def encode(self, x):
    """Encoder using LSTM.

    Args:
      x: tensor of size [num_children, num_groups, embedding_size]

    Returns:
      last_c, last_h: tensors of size [num_children, hidden_size], the final
        LSTM states
      attn_mem: tensor of size [num_children, num_groups, hidden_size], the
      attention
        memory, i.e. concatenation of all hidden states, linearly transformed by
        an attention matrix attn_w_1
    """
    if self.hparams.bi_lstm:
      with variable_scope.variable_scope(self.hparams.name, reuse=True):
        w_lstm_forward = variable_scope.get_variable("encoder_lstm_forward")
        w_lstm_backward = variable_scope.get_variable("encoder_lstm_backward")
        forget_bias = variable_scope.get_variable("encoder_forget_bias")
        attn_w_1 = variable_scope.get_variable("attn_w_1")
    else:
      with variable_scope.variable_scope(self.hparams.name, reuse=True):
        w_lstm = variable_scope.get_variable("encoder_lstm")
        forget_bias = variable_scope.get_variable("encoder_forget_bias")
        attn_w_1 = variable_scope.get_variable("attn_w_1")

    embedding_size = array_ops.shape(x)[2]

    signals = array_ops.split(x, self.num_groups, axis=1)
    for i in range(len(signals)):
      signals[i] = array_ops.reshape(
          signals[i], [self.hparams.num_children, embedding_size])

    if self.hparams.bi_lstm:

      def body(i, prev_c_forward, prev_h_forward, prev_c_backward,
               prev_h_backward):
        """while loop for LSTM."""
        signal_forward = signals[i]
        next_c_forward, next_h_forward = lstm(signal_forward, prev_c_forward,
                                              prev_h_forward, w_lstm_forward,
                                              forget_bias)

        signal_backward = signals[self.num_groups - 1 - i]
        next_c_backward, next_h_backward = lstm(
            signal_backward, prev_c_backward, prev_h_backward, w_lstm_backward,
            forget_bias)

        next_h = array_ops.concat([next_h_forward, next_h_backward], axis=1)
        all_h.append(next_h)

        return (next_c_forward, next_h_forward, next_c_backward,
                next_h_backward)

      c_forward = array_ops.zeros(
          [self.hparams.num_children, self.hparams.hidden_size / 2],
          dtype=dtypes.float32)
      h_forward = array_ops.zeros(
          [self.hparams.num_children, self.hparams.hidden_size / 2],
          dtype=dtypes.float32)

      c_backward = array_ops.zeros(
          [self.hparams.num_children, self.hparams.hidden_size / 2],
          dtype=dtypes.float32)
      h_backward = array_ops.zeros(
          [self.hparams.num_children, self.hparams.hidden_size / 2],
          dtype=dtypes.float32)
      all_h = []

      for i in range(0, self.num_groups):
        c_forward, h_forward, c_backward, h_backward = body(
            i, c_forward, h_forward, c_backward, h_backward)

      last_c = array_ops.concat([c_forward, c_backward], axis=1)
      last_h = array_ops.concat([h_forward, h_backward], axis=1)
      attn_mem = array_ops.stack(all_h)

    else:

      def body(i, prev_c, prev_h):
        signal = signals[i]
        next_c, next_h = lstm(signal, prev_c, prev_h, w_lstm, forget_bias)
        all_h.append(next_h)
        return next_c, next_h

      c = array_ops.zeros(
          [self.hparams.num_children, self.hparams.hidden_size],
          dtype=dtypes.float32)
      h = array_ops.zeros(
          [self.hparams.num_children, self.hparams.hidden_size],
          dtype=dtypes.float32)
      all_h = []

      for i in range(0, self.num_groups):
        c, h = body(i, c, h)

      last_c = c
      last_h = h
      attn_mem = array_ops.stack(all_h)

    attn_mem = array_ops.transpose(attn_mem, [1, 0, 2])
    attn_mem = array_ops.reshape(
        attn_mem,
        [self.hparams.num_children * self.num_groups, self.hparams.hidden_size])
    attn_mem = math_ops.matmul(attn_mem, attn_w_1)
    attn_mem = array_ops.reshape(
        attn_mem,
        [self.hparams.num_children, self.num_groups, self.hparams.hidden_size])

    return last_c, last_h, attn_mem

  def decode(self,
             x,
             last_c,
             last_h,
             attn_mem,
             mode="target",
             y=None):
    """Decoder using LSTM.

    Args:
      x: tensor of size [num_children, num_groups, embedding_size].
      last_c: tensor of size [num_children, hidden_size], the final LSTM states
          computed by self.encoder.
      last_h: same as last_c.
      attn_mem: tensor of size [num_children, num_groups, hidden_size].
      mode: "target" or "sample".
      y: tensor of size [num_children, num_groups], the device placements.

    Returns:
      actions: tensor of size [num_children, num_groups], the placements of
          devices
    """
    with variable_scope.variable_scope(self.hparams.name, reuse=True):
      w_lstm = variable_scope.get_variable("decoder_lstm")
      forget_bias = variable_scope.get_variable("decoder_forget_bias")
      device_embeddings = variable_scope.get_variable("device_embeddings")
      device_softmax = variable_scope.get_variable("device_softmax")
      device_go_embedding = variable_scope.get_variable("device_go_embedding")
      attn_w_2 = variable_scope.get_variable("attn_w_2")
      attn_v = variable_scope.get_variable("attn_v")

    actions = tensor_array_ops.TensorArray(
        dtypes.int32,
        size=self.num_groups,
        infer_shape=False,
        clear_after_read=False)

    # pylint: disable=unused-argument
    def condition(i, *args):
      return math_ops.less(i, self.num_groups)

    # pylint: disable=missing-docstring
    def body(i, prev_c, prev_h, actions, log_probs):
      # pylint: disable=g-long-lambda
      signal = control_flow_ops.cond(
          math_ops.equal(i, 0),
          lambda: array_ops.tile(device_go_embedding,
                                 [self.hparams.num_children, 1]),
          lambda: embedding_ops.embedding_lookup(device_embeddings,
                                                 actions.read(i - 1))
      )
      if self.hparams.keep_prob is not None:
        signal = nn_ops.dropout(signal, rate=(1 - self.hparams.keep_prob))
      next_c, next_h = lstm(signal, prev_c, prev_h, w_lstm, forget_bias)
      query = math_ops.matmul(next_h, attn_w_2)
      query = array_ops.reshape(
          query, [self.hparams.num_children, 1, self.hparams.hidden_size])
      query = math_ops.tanh(query + attn_mem)
      query = array_ops.reshape(query, [
          self.hparams.num_children * self.num_groups, self.hparams.hidden_size
      ])
      query = math_ops.matmul(query, attn_v)
      query = array_ops.reshape(query,
                                [self.hparams.num_children, self.num_groups])
      query = nn_ops.softmax(query)
      query = array_ops.reshape(query,
                                [self.hparams.num_children, self.num_groups, 1])
      query = math_ops.reduce_sum(attn_mem * query, axis=1)
      query = array_ops.concat([next_h, query], axis=1)
      logits = math_ops.matmul(query, device_softmax)
      logits /= self.hparams.temperature
      if self.hparams.tanh_constant > 0:
        logits = math_ops.tanh(logits) * self.hparams.tanh_constant
      if self.hparams.logits_std_noise > 0:
        num_in_logits = math_ops.cast(
            array_ops.size(logits), dtype=dtypes.float32)
        avg_norm = math_ops.divide(
            linalg_ops.norm(logits), math_ops.sqrt(num_in_logits))
        logits_noise = random_ops.random_normal(
            array_ops.shape(logits),
            stddev=self.hparams.logits_std_noise * avg_norm)
        logits = control_flow_ops.cond(
            self.global_step > self.hparams.stop_noise_step, lambda: logits,
            lambda: logits + logits_noise)

      if mode == "sample":
        next_y = random_ops.multinomial(logits, 1, seed=self.hparams.seed)
      elif mode == "greedy":
        next_y = math_ops.argmax(logits, 1)
      elif mode == "target":
        next_y = array_ops.slice(y, [0, i], [-1, 1])
      else:
        raise NotImplementedError
      next_y = math_ops.cast(next_y, dtypes.int32)
      next_y = array_ops.reshape(next_y, [self.hparams.num_children])
      actions = actions.write(i, next_y)
      log_probs += nn_ops.sparse_softmax_cross_entropy_with_logits(
          logits=logits, labels=next_y)
      return i + 1, next_c, next_h, actions, log_probs

    loop_vars = [
        constant_op.constant(0, dtype=dtypes.int32), last_c, last_h, actions,
        array_ops.zeros([self.hparams.num_children], dtype=dtypes.float32)
    ]
    loop_outputs = control_flow_ops.while_loop(condition, body, loop_vars)

    last_c = loop_outputs[-4]
    last_h = loop_outputs[-3]
    actions = loop_outputs[-2].stack()
    actions = array_ops.transpose(actions, [1, 0])
    log_probs = loop_outputs[-1]
    return actions, log_probs

  def eval_placement(self,
                     sess,
                     child_id=0,
                     verbose=False):
    grouping_actions, actions = sess.run([
        self.grouping_actions_cache,
        self.actions_cache
    ])
    grouping_actions = grouping_actions[child_id]
    actions = actions[child_id]
    if verbose:
      global_step = sess.run(self.global_step)
      if global_step % 100 == 0:
        log_string = "op group assignments: "
        for a in grouping_actions:
          log_string += "{} ".format(a)
        print(log_string[:-1])
        log_string = "group device assignments: "
        for a in actions:
          log_string += "{} ".format(a)
        print(log_string[:-1])

    for op in self.important_ops:
      topo_order_index = self.name_to_topo_order_index[op.name]
      group_index = grouping_actions[topo_order_index]
      op.device = self.devices[actions[group_index]].name
    try:
      _, run_time, _ = self.cluster.MeasureCosts(self.item)
    except errors.ResourceExhaustedError:
      run_time = self.hparams.failing_signal
    return run_time

  def update_reward(self,
                    sess,
                    run_time,
                    child_id=0,
                    verbose=False):
    reward = self.compute_reward(run_time)
    controller_ops = self.ops["controller"]
    _, best_reward = sess.run(
        [
            controller_ops["reward"]["update"][child_id],
            controller_ops["best_reward"]["update"][child_id]
        ],
        feed_dict={
            controller_ops["reward"]["ph"][child_id]: reward,
        })
    if verbose:
      print(("run_time={:<.5f} reward={:<.5f} "
             "best_reward={:<.5f}").format(run_time, reward, best_reward))

    # Reward is a double, best_reward a float: allow for some slack in the
    # comparison.
    updated = abs(best_reward - reward) < 1e-6
    return updated

  def generate_grouping(self, sess):
    controller_ops = self.ops["controller"]
    grouping_actions = sess.run(controller_ops["grouping_y_preds"]["sample"])
    return grouping_actions

  def generate_placement(self, grouping, sess):
    controller_ops = self.ops["controller"]
    feed_seq2seq_input_dict = {}
    feed_seq2seq_input_dict[self.seq2seq_input_layer] = grouping
    sess.run(
        controller_ops["y_preds"]["sample"], feed_dict=feed_seq2seq_input_dict)

  def process_reward(self, sess):
    controller_ops = self.ops["controller"]
    run_ops = [
        controller_ops["loss"], controller_ops["lr"],
        controller_ops["grad_norm"], controller_ops["grad_norms"],
        controller_ops["train_op"]
    ]
    sess.run(run_ops)
    sess.run(controller_ops["baseline_update"])

  def _get_train_ops(self,
                     loss,
                     tf_variables,
                     global_step,
                     grad_bound=1.25,
                     lr_init=1e-3,
                     lr_dec=0.9,
                     start_decay_step=10000,
                     decay_steps=100,
                     optimizer_type="adam"):
    """Loss optimizer.

    Args:
      loss: scalar tf tensor
      tf_variables: list of training variables, typically
        tf.compat.v1.trainable_variables()
      global_step: global_step
      grad_bound: max gradient norm
      lr_init: initial learning rate
      lr_dec: leaning rate decay coefficient
      start_decay_step: start decaying learning rate after this many steps
      decay_steps: apply decay rate factor at this step intervals
      optimizer_type: optimizer type should be either adam or sgd

    Returns:
      train_op: training op
      learning_rate: scalar learning rate tensor
      grad_norm: l2 norm of the gradient vector
      all_grad_norms: l2 norm of each component
    """
    lr_gstep = global_step - start_decay_step

    def f1():
      return constant_op.constant(lr_init)

    def f2():
      return learning_rate_decay.exponential_decay(lr_init, lr_gstep,
                                                   decay_steps, lr_dec, True)

    learning_rate = control_flow_ops.cond(
        math_ops.less(global_step, start_decay_step),
        f1,
        f2,
        name="learning_rate")

    if optimizer_type == "adam":
      opt = adam.AdamOptimizer(learning_rate)
    elif optimizer_type == "sgd":
      opt = gradient_descent.GradientDescentOptimizer(learning_rate)
    grads_and_vars = opt.compute_gradients(loss, tf_variables)
    grad_norm = clip_ops.global_norm([g for g, v in grads_and_vars])
    all_grad_norms = {}
    clipped_grads = []
    clipped_rate = math_ops.maximum(grad_norm / grad_bound, 1.0)
    for g, v in grads_and_vars:
      if g is not None:
        if isinstance(g, tf_ops.IndexedSlices):
          clipped = g.values / clipped_rate
          norm_square = math_ops.reduce_sum(clipped * clipped)
          clipped = tf_ops.IndexedSlices(clipped, g.indices)
        else:
          clipped = g / clipped_rate
          norm_square = math_ops.reduce_sum(clipped * clipped)
        all_grad_norms[v.name] = math_ops.sqrt(norm_square)
        clipped_grads.append((clipped, v))

    train_op = opt.apply_gradients(clipped_grads, global_step)
    return train_op, learning_rate, grad_norm, all_grad_norms


def lstm(x, prev_c, prev_h, w_lstm, forget_bias):
  """LSTM cell.

  Args:
    x: tensors of size [num_children, hidden_size].
    prev_c: tensors of size [num_children, hidden_size].
    prev_h: same as prev_c.
    w_lstm: .
    forget_bias: .

  Returns:
    next_c:
    next_h:
  """
  ifog = math_ops.matmul(array_ops.concat([x, prev_h], axis=1), w_lstm)
  i, f, o, g = array_ops.split(ifog, 4, axis=1)
  i = math_ops.sigmoid(i)
  f = math_ops.sigmoid(f + forget_bias)
  o = math_ops.sigmoid(o)
  g = math_ops.tanh(g)
  next_c = i * g + f * prev_c
  next_h = o * math_ops.tanh(next_c)
  return next_c, next_h

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
"""Graph Placer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
from tensorflow.core.protobuf import meta_graph_pb2
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops as tf_ops
from tensorflow.python.grappler import cluster as gcluster
from tensorflow.python.grappler import hierarchical_controller
from tensorflow.python.grappler import item as gitem
from tensorflow.python.grappler import tf_optimizer
from tensorflow.python.training import training


def PlaceGraph(metagraph,
               cluster=None,
               allotted_time=3600,
               hparams=None,
               verbose=False):
  """Place the provided metagraph.

  Args:
    metagraph: the metagraph to place.
    cluster: an optional set of hardware resource to optimize the placement for.
      If none is specified, we'll optimize the placement for the hardware
      available on the local machine.
    allotted_time: the maximum amount to time in seconds to spend optimizing
      the placement.
    hparams: hyperparameters used to fine tune the placer.
    verbose: prints debug information if True.

  Returns:
    The placed metagraph.
  """
  if cluster is None:
    cluster = gcluster.Cluster()

  # Optimize the metagraph to speedup the placement
  rewriter_config = rewriter_config_pb2.RewriterConfig()
  rewriter_config.optimizers.append("pruning")
  rewriter_config.optimizers.append("constfold")
  rewriter_config.optimizers.append("arithmetic")
  rewriter_config.optimizers.append("dependency")
  rewriter_config.optimizers.append("pruning")
  optimized_graph = tf_optimizer.OptimizeGraph(
      rewriter_config, metagraph, verbose=verbose, cluster=cluster)
  optimized_metagraph = meta_graph_pb2.MetaGraphDef()
  optimized_metagraph.CopyFrom(metagraph)
  optimized_metagraph.graph_def.CopyFrom(optimized_graph)

  item = gitem.Item(optimized_metagraph)

  # Measure the runtime achievable with the original placement.
  try:
    _, original_run_time, _ = cluster.MeasureCosts(item)
    if verbose:
      print("Runtime for original placement: " + str(original_run_time))
  except errors.OpError as e:
    if verbose:
      print("Original placement isn't feasible: " + str(e))
    original_run_time = hparams.failing_signal

  if hparams is None:
    hparams = hierarchical_controller.hierarchical_controller_hparams()
  # We run with a single child
  hparams.num_children = 1

  with tf_ops.Graph().as_default():
    # Place all the nodes of the controller on the CPU. We don't want them to
    # fight for accelerator memory with the model to optimize.
    with tf_ops.device("/device:CPU:0"):
      model = hierarchical_controller.HierarchicalController(
          hparams, item, cluster)
      ops = model.build_controller()
      session_creator = training.ChiefSessionCreator()
      with training.MonitoredSession(session_creator=session_creator) as sess:
        start_time = time.time()
        current_time = start_time
        while current_time - start_time < allotted_time:
          grouping_actions = model.generate_grouping(sess)
          input_to_seq2seq = model.create_group_embeddings(
              grouping_actions, verbose=verbose)
          model.generate_placement(input_to_seq2seq, sess)
          try:
            run_time = model.eval_placement(
                sess,
                verbose=verbose)
          except errors.OpError as e:
            if verbose:
              print("Failed to run graph:" + str(e))
            run_time = hparams.failing_signal
          updated = model.update_reward(sess, run_time, verbose=verbose)
          if updated and run_time < original_run_time:
            if verbose:
              print("Found better placement, with runtime " + str(run_time))
            model.export_placement(metagraph)

          model.process_reward(sess)

          current_time = time.time()

  return metagraph

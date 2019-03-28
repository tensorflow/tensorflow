# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
# =============================================================================


def automatic_sharding(num_shards,
                       input_ts,
                       loss_ts,
                       train_ops=None,
                       edge_filter=None):
  """Automatically set shards for all connected nodes in graph.

  Args:
    :param num_shards: number of shards to split graph over
    :param input_ts: tensor closest to the datafeed in graph
    :param loss_ts: tensor closest to the loss in graph
    :param train_ops: an operation or list of operations which are returned by
                      Optimizer.minimize()
    :param edge_filter: a callable predicate, with the signature fn(edge), where
                        edge is a tuple with the name of the source op, and the
                        name of the destination op.
  """
  from tensorflow.contrib.ipu.python import autoshard_cnn
  autoshard_cnn.automatic_sharding(num_shards, input_ts, loss_ts, train_ops,
                                   edge_filter)

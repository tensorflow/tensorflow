# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Test utilities."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import io
import itertools
import threading

from absl import app

from tensorflow.python.compat import v2_compat
from tensorflow.python.distribute import collective_all_reduce_strategy
from tensorflow.python.distribute import multi_process_runner
from tensorflow.python.distribute import tpu_strategy
from tensorflow.python.distribute import values
from tensorflow.python.eager import context
from tensorflow.python.framework import config
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.util import nest

try:
  import objgraph  # pylint:disable=g-import-not-at-top
except ImportError:
  objgraph = None


def gather(strategy, value):
  """Gathers value from all workers.

  This is intended for tests before we implement an official all-gather API.

  Args:
    strategy: a `tf.distribute.Strategy`.
    value: a nested structure of n-dim `tf.distribute.DistributedValue` of
      `tf.Tensor`, or of a `tf.Tensor` if the strategy only has one replica.
      Cannot contain tf.sparse.SparseTensor.

  Returns:
    a (n+1)-dim `tf.Tensor`.
  """
  return nest.map_structure(functools.partial(_gather, strategy), value)


def _gather(strategy, value):
  """Gathers a single value."""
  # pylint: disable=protected-access
  if not isinstance(value, values.DistributedValues):
    value = values.PerReplica([ops.convert_to_tensor(value)])
  if not isinstance(strategy.extended,
                    collective_all_reduce_strategy.CollectiveAllReduceExtended):
    return array_ops.stack(value._values)
  assert len(strategy.extended.worker_devices) == len(value._values)
  inputs = [array_ops.expand_dims_v2(v, axis=0) for v in value._values]
  return strategy.gather(values.PerReplica(inputs), axis=0)
  # pylint: enable=protected-access


def set_logical_devices_to_at_least(device, num):
  """Create logical devices of at least a given number."""
  if num < 1:
    raise ValueError("`num` must be at least 1 not %r" % (num,))
  physical_devices = config.list_physical_devices(device)
  if not physical_devices:
    raise RuntimeError("No {} found".format(device))
  if len(physical_devices) >= num:
    return
  # By default each physical device corresponds to one logical device. We create
  # multiple logical devices for the last physical device so that we have `num`
  # logical devices.
  num = num - len(physical_devices) + 1
  logical_devices = []
  for _ in range(num):
    if device.upper() == "GPU":
      logical_devices.append(
          context.LogicalDeviceConfiguration(memory_limit=2048))
    else:
      logical_devices.append(context.LogicalDeviceConfiguration())
  # Create logical devices from the last device since sometimes the first GPU
  # is the primary graphic card and may have less memory available.
  config.set_logical_device_configuration(physical_devices[-1], logical_devices)


def _set_logical_devices():
  if config.list_physical_devices("GPU"):
    set_logical_devices_to_at_least("GPU", 2)
  if config.list_physical_devices("CPU"):
    set_logical_devices_to_at_least("CPU", 2)


def main(enable_v2_behavior=True, config_logical_devices=True):
  """All-in-one main function for tf.distribute tests."""
  if config_logical_devices:
    app.call_after_init(_set_logical_devices)
  if enable_v2_behavior:
    v2_compat.enable_v2_behavior()
  else:
    v2_compat.disable_v2_behavior()
  multi_process_runner.test_main()


def _op_dependencies(op):
  """Returns the data and control dependencies of a tf.Operation combined."""
  deps = []
  for node in itertools.chain(op.inputs, op.control_inputs):
    if isinstance(node, ops.Tensor):
      node = node.op
    assert isinstance(node, ops.Operation)
    deps.append(node)
  return deps


def topological_sort_operations(operations):
  """Topological sorts a list of operations.

  This does a topological sort of the operations in a graph. The edges include
  both data dependencies and control dependencies. Note that the edge goes from
  an operation to its dependencies.

  Args:
    operations: a list of tf.Operation in the same graph.

  Returns:
    A map from a tf.Operation to its topological order.
  """
  in_degrees = {}
  for op in operations:
    if op not in in_degrees:
      in_degrees[op] = 0
    for next_op in _op_dependencies(op):
      in_degrees[next_op] = in_degrees.get(next_op, 0) + 1
  nexts = []
  for op, in_degree in in_degrees.items():
    if in_degree == 0:
      nexts.append(op)
  order = {}
  next_order = 0
  while nexts:
    op, nexts = nexts[0], nexts[1:]
    order[op] = next_order
    next_order += 1
    for next_op in _op_dependencies(op):
      in_degrees[next_op] -= 1
      if in_degrees[next_op] == 0:
        nexts.append(next_op)
  assert len(order) == len(operations)
  return order


def _exists_dependency(start, end):
  """Returns whether there exists a dependency chain from start to end."""
  nexts = [start]
  while nexts:
    op, nexts = nexts[0], nexts[1:]
    for next_op in _op_dependencies(op):
      if next_op == end:
        return True
      nexts.append(next_op)
  return False


def assert_sequential_execution(order, operations):
  """Asserts there's a deterministic execution order between the operations.

  Args:
    order: a map from a tf.Operation to its topological order.
    operations: a list of operations that should be executed sequentially. It
      can be given in any order.
  """
  # Topological ordering guarantees that, if there's a dependency from N_a to
  # N_b, then order[N_a] < order[N_b]. If there do exist a path of dependencies
  # among the operations, it always goes from a operation with a smaller
  # topological order to one with a larger topological order. Therefore, we only
  # need to sort the operations by their topological orders, and verify that
  # there's a path of dependency between adjacent pairs.
  operations = sorted(operations, key=lambda op: order[op])
  for i in range(len(operations) - 1):
    if not _exists_dependency(operations[i], operations[i + 1]):
      print(operations[i].graph.as_graph_def())
      raise AssertionError(
          "No dependency between {} and {}. Graph is dumped to stdout.".format(
              operations[i].name, operations[i + 1].name))


def get_running_threads():
  """Returns a set of all running thread names."""
  running_threads = set()
  for thread in threading.enumerate():
    if thread.name is not None:
      running_threads.add(thread.name)
  return running_threads


def has_thread(prefix, running_threads):
  """Returns whether any 'running_threads' is prefixed with 'prefix'.

  Args:
    prefix: The prefix of the expected thread name.
    running_threads: A collection of the running thread names.
  """
  for thread in running_threads:
    if thread.startswith(prefix):
      return True
  return False


def show_backref(target, max_depth=3):
  """Returns a dot graph of all the objects that are referencing the target.

  A object referencing graph is useful to debug memory leak like circular
  reference. objgraph provides a good visualization of the memory graph than
  most python built-in utilities like gc.get_referrers(), which are not
  human-readable sometimes.

  The dot graph will be written to a string IO object, and can be rendered with
  graphviz in operating system.
  E.g. dot -Tpng {$dot_graph} -o output.png
  Args:
    target: The target object for the memory graph.
    max_depth: The maximum depth of the graph. By default 3 layers of references
    are used. Increases this a lot may result in the graph growing too big.

  Returns:
    A string that contains the object reference graph.
  Raises:
    NotImplementedError: if objgraph is not installed.
  """
  if objgraph is None:
    raise NotImplementedError("objgraph is not installed.")
  string_io = io.StringIO()
  objgraph.show_backrefs(target, max_depth=max_depth, output=string_io)
  graph = string_io.getvalue()
  string_io.close()
  return graph


def create_per_replica(strategy, value_list):
  """Creates a PerReplica of Tensors from the value_list."""
  if len(strategy.extended.worker_devices) != len(value_list):
    raise ValueError(
        "the length of values must be the same as the number of worker devices")
  tensors = []
  for device, value in zip(strategy.extended.worker_devices, value_list):
    with ops.device(device):
      tensors.append(ops.convert_to_tensor(value))
  return values.PerReplica(tensors)


def is_tpu_strategy(strategy):
  """Returns whether the strategy is a TPU strategy."""
  return isinstance(strategy,
                    (tpu_strategy.TPUStrategy, tpu_strategy.TPUStrategyV1,
                     tpu_strategy.TPUStrategyV2))

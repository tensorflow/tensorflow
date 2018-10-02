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
"""TensorFlow Debugger (tfdbg) Utilities."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re

from six.moves import xrange  # pylint: disable=redefined-builtin


def add_debug_tensor_watch(run_options,
                           node_name,
                           output_slot=0,
                           debug_ops="DebugIdentity",
                           debug_urls=None,
                           tolerate_debug_op_creation_failures=False,
                           global_step=-1):
  """Add watch on a `Tensor` to `RunOptions`.

  N.B.:
    1. Under certain circumstances, the `Tensor` may not get actually watched
      (e.g., if the node of the `Tensor` is constant-folded during runtime).
    2. For debugging purposes, the `parallel_iteration` attribute of all
      `tf.while_loop`s in the graph are set to 1 to prevent any node from
      being executed multiple times concurrently. This change does not affect
      subsequent non-debugged runs of the same `tf.while_loop`s.

  Args:
    run_options: An instance of `config_pb2.RunOptions` to be modified.
    node_name: (`str`) name of the node to watch.
    output_slot: (`int`) output slot index of the tensor from the watched node.
    debug_ops: (`str` or `list` of `str`) name(s) of the debug op(s). Can be a
      `list` of `str` or a single `str`. The latter case is equivalent to a
      `list` of `str` with only one element.
      For debug op types with customizable attributes, each debug op string can
      optionally contain a list of attribute names, in the syntax of:
        debug_op_name(attr_name_1=attr_value_1;attr_name_2=attr_value_2;...)
    debug_urls: (`str` or `list` of `str`) URL(s) to send debug values to,
      e.g., `file:///tmp/tfdbg_dump_1`, `grpc://localhost:12345`.
    tolerate_debug_op_creation_failures: (`bool`) Whether to tolerate debug op
      creation failures by not throwing exceptions.
    global_step: (`int`) Optional global_step count for this debug tensor
      watch.
  """

  watch_opts = run_options.debug_options.debug_tensor_watch_opts
  run_options.debug_options.global_step = global_step

  watch = watch_opts.add()
  watch.tolerate_debug_op_creation_failures = (
      tolerate_debug_op_creation_failures)
  watch.node_name = node_name
  watch.output_slot = output_slot

  if isinstance(debug_ops, str):
    debug_ops = [debug_ops]

  watch.debug_ops.extend(debug_ops)

  if debug_urls:
    if isinstance(debug_urls, str):
      debug_urls = [debug_urls]

    watch.debug_urls.extend(debug_urls)


def watch_graph(run_options,
                graph,
                debug_ops="DebugIdentity",
                debug_urls=None,
                node_name_regex_whitelist=None,
                op_type_regex_whitelist=None,
                tensor_dtype_regex_whitelist=None,
                tolerate_debug_op_creation_failures=False,
                global_step=-1,
                reset_disk_byte_usage=False):
  """Add debug watches to `RunOptions` for a TensorFlow graph.

  To watch all `Tensor`s on the graph, let both `node_name_regex_whitelist`
  and `op_type_regex_whitelist` be the default (`None`).

  N.B.:
    1. Under certain circumstances, the `Tensor` may not get actually watched
      (e.g., if the node of the `Tensor` is constant-folded during runtime).
    2. For debugging purposes, the `parallel_iteration` attribute of all
      `tf.while_loop`s in the graph are set to 1 to prevent any node from
      being executed multiple times concurrently. This change does not affect
      subsequent non-debugged runs of the same `tf.while_loop`s.


  Args:
    run_options: An instance of `config_pb2.RunOptions` to be modified.
    graph: An instance of `ops.Graph`.
    debug_ops: (`str` or `list` of `str`) name(s) of the debug op(s) to use.
    debug_urls: URLs to send debug values to. Can be a list of strings,
      a single string, or None. The case of a single string is equivalent to
      a list consisting of a single string, e.g., `file:///tmp/tfdbg_dump_1`,
      `grpc://localhost:12345`.
      For debug op types with customizable attributes, each debug op name string
      can optionally contain a list of attribute names, in the syntax of:
        debug_op_name(attr_name_1=attr_value_1;attr_name_2=attr_value_2;...)
    node_name_regex_whitelist: Regular-expression whitelist for node_name,
      e.g., `"(weight_[0-9]+|bias_.*)"`
    op_type_regex_whitelist: Regular-expression whitelist for the op type of
      nodes, e.g., `"(Variable|Add)"`.
      If both `node_name_regex_whitelist` and `op_type_regex_whitelist`
      are set, the two filtering operations will occur in a logical `AND`
      relation. In other words, a node will be included if and only if it
      hits both whitelists.
    tensor_dtype_regex_whitelist: Regular-expression whitelist for Tensor
      data type, e.g., `"^int.*"`.
      This whitelist operates in logical `AND` relations to the two whitelists
      above.
    tolerate_debug_op_creation_failures: (`bool`) whether debug op creation
      failures (e.g., due to dtype incompatibility) are to be tolerated by not
      throwing exceptions.
    global_step: (`int`) Optional global_step count for this debug tensor
      watch.
    reset_disk_byte_usage: (`bool`) whether to reset the tracked disk byte
      usage to zero (default: `False`).
  """

  if isinstance(debug_ops, str):
    debug_ops = [debug_ops]

  node_name_pattern = (re.compile(node_name_regex_whitelist)
                       if node_name_regex_whitelist else None)
  op_type_pattern = (re.compile(op_type_regex_whitelist)
                     if op_type_regex_whitelist else None)
  tensor_dtype_pattern = (re.compile(tensor_dtype_regex_whitelist)
                          if tensor_dtype_regex_whitelist else None)

  ops = graph.get_operations()
  for op in ops:
    # Skip nodes without any output tensors.
    if not op.outputs:
      continue

    node_name = op.name
    op_type = op.type

    if node_name_pattern and not node_name_pattern.match(node_name):
      continue
    if op_type_pattern and not op_type_pattern.match(op_type):
      continue

    for slot in xrange(len(op.outputs)):
      if (tensor_dtype_pattern and
          not tensor_dtype_pattern.match(op.outputs[slot].dtype.name)):
        continue

      add_debug_tensor_watch(
          run_options,
          node_name,
          output_slot=slot,
          debug_ops=debug_ops,
          debug_urls=debug_urls,
          tolerate_debug_op_creation_failures=(
              tolerate_debug_op_creation_failures),
          global_step=global_step)
  run_options.debug_options.reset_disk_byte_usage = reset_disk_byte_usage


def watch_graph_with_blacklists(run_options,
                                graph,
                                debug_ops="DebugIdentity",
                                debug_urls=None,
                                node_name_regex_blacklist=None,
                                op_type_regex_blacklist=None,
                                tensor_dtype_regex_blacklist=None,
                                tolerate_debug_op_creation_failures=False,
                                global_step=-1,
                                reset_disk_byte_usage=False):
  """Add debug tensor watches, blacklisting nodes and op types.

  This is similar to `watch_graph()`, but the node names and op types are
  blacklisted, instead of whitelisted.

  N.B.:
    1. Under certain circumstances, the `Tensor` may not get actually watched
      (e.g., if the node of the `Tensor` is constant-folded during runtime).
    2. For debugging purposes, the `parallel_iteration` attribute of all
      `tf.while_loop`s in the graph are set to 1 to prevent any node from
      being executed multiple times concurrently. This change does not affect
      subsequent non-debugged runs of the same `tf.while_loop`s.

  Args:
    run_options: An instance of `config_pb2.RunOptions` to be modified.
    graph: An instance of `ops.Graph`.
    debug_ops: (`str` or `list` of `str`) name(s) of the debug op(s) to use.
      See the documentation of `watch_graph` for more details.
    debug_urls: URL(s) to send debug values to, e.g.,
      `file:///tmp/tfdbg_dump_1`, `grpc://localhost:12345`.
    node_name_regex_blacklist: Regular-expression blacklist for node_name.
      This should be a string, e.g., `"(weight_[0-9]+|bias_.*)"`.
    op_type_regex_blacklist: Regular-expression blacklist for the op type of
      nodes, e.g., `"(Variable|Add)"`.
      If both node_name_regex_blacklist and op_type_regex_blacklist
      are set, the two filtering operations will occur in a logical `OR`
      relation. In other words, a node will be excluded if it hits either of
      the two blacklists; a node will be included if and only if it hits
      neither of the blacklists.
    tensor_dtype_regex_blacklist: Regular-expression blacklist for Tensor
      data type, e.g., `"^int.*"`.
      This blacklist operates in logical `OR` relations to the two whitelists
      above.
    tolerate_debug_op_creation_failures: (`bool`) whether debug op creation
      failures (e.g., due to dtype incompatibility) are to be tolerated by not
      throwing exceptions.
    global_step: (`int`) Optional global_step count for this debug tensor
      watch.
    reset_disk_byte_usage: (`bool`) whether to reset the tracked disk byte
      usage to zero (default: `False`).
  """

  if isinstance(debug_ops, str):
    debug_ops = [debug_ops]

  node_name_pattern = (re.compile(node_name_regex_blacklist) if
                       node_name_regex_blacklist else None)
  op_type_pattern = (re.compile(op_type_regex_blacklist) if
                     op_type_regex_blacklist else None)
  tensor_dtype_pattern = (re.compile(tensor_dtype_regex_blacklist) if
                          tensor_dtype_regex_blacklist else None)

  ops = graph.get_operations()
  for op in ops:
    # Skip nodes without any output tensors.
    if not op.outputs:
      continue

    node_name = op.name
    op_type = op.type

    if node_name_pattern and node_name_pattern.match(node_name):
      continue
    if op_type_pattern and op_type_pattern.match(op_type):
      continue

    for slot in xrange(len(op.outputs)):
      if (tensor_dtype_pattern and
          tensor_dtype_pattern.match(op.outputs[slot].dtype.name)):
        continue

      add_debug_tensor_watch(
          run_options,
          node_name,
          output_slot=slot,
          debug_ops=debug_ops,
          debug_urls=debug_urls,
          tolerate_debug_op_creation_failures=(
              tolerate_debug_op_creation_failures),
          global_step=global_step)
    run_options.debug_options.reset_disk_byte_usage = reset_disk_byte_usage

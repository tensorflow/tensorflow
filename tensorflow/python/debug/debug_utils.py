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
                           debug_urls=None):
  """Add debug tensor watch option to RunOptions.

  Args:
    run_options: An instance of tensorflow.core.protobuf.config_pb2.RunOptions
    node_name: Name of the node to watch.
    output_slot: Output slot index of the tensor from the watched node.
    debug_ops: Name(s) of the debug op(s). Default: "DebugIdentity".
        Can be a list of strings or a single string. The latter case is
        equivalent to a list of string with only one element.
    debug_urls: URLs to send debug signals to: a non-empty list of strings or
        a string, or None. The case of a string is equivalent to a list of
        string with only one element.
  """

  watch_opts = run_options.debug_tensor_watch_opts

  watch = watch_opts.add()
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
                op_type_regex_whitelist=None):
  """Add debug tensor watch options to RunOptions based on a TensorFlow graph.

  To watch all tensors on the graph, set both node_name_regex_whitelist
  and op_type_regex_whitelist to None.

  Args:
    run_options: An instance of tensorflow.core.protobuf.config_pb2.RunOptions
    graph: An instance of tensorflow.python.framework.ops.Graph
    debug_ops: Name of the debug op to use. Default: "DebugIdentity".
        Can be a list of strings of a single string. The latter case is
        equivalent to a list of a single string.
    debug_urls: Debug urls. Can be a list of strings, a single string, or
        None. The case of a single string is equivalen to a list consisting
        of a single string.
    node_name_regex_whitelist: Regular-expression whitelist for node_name.
        This should be a string, e.g., "(weight_[0-9]+|bias_.*)"
    op_type_regex_whitelist: Regular-expression whitelist for the op type of
        nodes. If both node_name_regex_whitelist and op_type_regex_whitelist
        are none, the two filtering operations will occur in an "AND"
        relation. In other words, a node will be included if and only if it
        hits both whitelists. This should be a string, e.g., "(Variable|Add)".
  """

  if isinstance(debug_ops, str):
    debug_ops = [debug_ops]

  if node_name_regex_whitelist:
    node_name_pattern = re.compile(node_name_regex_whitelist)
  else:
    node_name_pattern = None

  if op_type_regex_whitelist:
    op_type_pattern = re.compile(op_type_regex_whitelist)
  else:
    op_type_pattern = None

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
      add_debug_tensor_watch(
          run_options,
          node_name,
          output_slot=slot,
          debug_ops=debug_ops,
          debug_urls=debug_urls)


def watch_graph_with_blacklists(run_options,
                                graph,
                                debug_ops="DebugIdentity",
                                debug_urls=None,
                                node_name_regex_blacklist=None,
                                op_type_regex_blacklist=None):
  """Add debug tensor watch options, blacklisting nodes and op types.

  This is similar to watch_graph(), but the node names and op types can be
  blacklisted, instead of whitelisted.

  Args:
    run_options: An instance of tensorflow.core.protobuf.config_pb2.RunOptions
    graph: An instance of tensorflow.python.framework.ops.Graph
    debug_ops: Name of the debug op to use. Default: "DebugIdentity".
        Can be a list of strings of a single string. The latter case is
        equivalent to a list of a single string.
    debug_urls: Debug urls. Can be a list of strings, a single string, or
        None. The case of a single string is equivalen to a list consisting
        of a single string.
    node_name_regex_blacklist: Regular-expression blacklist for node_name.
        This should be a string, e.g., "(weight_[0-9]+|bias_.*)"
    op_type_regex_blacklist: Regular-expression blacklist for the op type of
        nodes. If both node_name_regex_blacklist and op_type_regex_blacklist
        are none, the two filtering operations will occur in an "OR"
        relation. In other words, a node will be excluded if it hits either of
        the two blacklists; a node will be included if and only if it hits
        none of the blacklists. This should be a string, e.g.,
        "(Variable|Add)".
  """

  if isinstance(debug_ops, str):
    debug_ops = [debug_ops]

  if node_name_regex_blacklist:
    node_name_pattern = re.compile(node_name_regex_blacklist)
  else:
    node_name_pattern = None

  if op_type_regex_blacklist:
    op_type_pattern = re.compile(op_type_regex_blacklist)
  else:
    op_type_pattern = None

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
      add_debug_tensor_watch(
          run_options,
          node_name,
          output_slot=slot,
          debug_ops=debug_ops,
          debug_urls=debug_urls)

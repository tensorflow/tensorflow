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
# =============================================================================
"""Exposes the scope for NVIDIA optimizers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import contextlib

from tensorflow.core.framework import attr_value_pb2
from tensorflow.python.framework import ops

# follows xla_scope as example

_NV_SCOPE_KEY = ("__nv_scope",)


class NvidiaScope(object):
  """Keeps track of scopes"""

  def __init__(self, depth):
    self.depth = depth


def ConditionalWrapper(selector, value):
  """ Simple wrapper that returns a callable to disable attributes for the excluded nodes."""

  def ConditionalSetter(node_def):
    if callable(selector):
      accept = selector(node_def).b
    else:
      accept = selector.b
    if accept:
      return value
    return None

  return ConditionalSetter


@contextlib.contextmanager
def nvidia_scope(
    select=True,
    scope_name=None,
    optimizer=None,
    precision=None,
    engine_mode=None,
    maximum_cached_engines=None,
    maximum_workspace_size=None,
):
  """Include or exclude nodes in optimized segment with given parameters.

  NOTE: this is a hint and will be supported on a best-effort basis.
  Args:
    include      : Whether to include the node or not.
      Can be a callable. (Default= True)
    scope_name   : Name of the scope. Same named scopes will attempted to be
      merged, if they have identical parameters. If None, will be generated
      during conversion (Default=None)
    optimizer    : Which NVIDIA optimizer to use. Only "TRT" for the time
      being. (Default= "TRT")
    precision    : Engine precision, must be one of
      [ "FP32", "FP16", "INT8" ]. (Default= "FP32")
    engine_mode  : Whether to create engine offline or dynamically on the
      fly during running. Valid modes are "STATIC" or "DYNAMIC".
      Can be extended in future. Dynamic engines can adapt to different input
      shapes at the cost of extra engine creation and when shapes can not be
      inferred offline, while static engines construct the engines at
      initialization and better for fixed size networks where shapes can be
      inferred offline. (Default= "DYNAMIC")
    maximum_cached_engines: Number of cached engines(int). Dynamic engines
      can keep some number of engines with different input shapes in memory
      in order to optimize execution. (Default= 1)
    maximum_workspace_size: Maximum workspace size assigned to this
      segment (int). This will override the optimizers same named setting.
      If -1, optimizer level setting is shared between all engines
      proportional to their size. (Default= -1)

  Example usage:
    with tf.contrib.tensorrt.nvidia_scope():
      c = tf.matmul(a, b)  # included in segment
    with tf.contrib.tensorrt.nvidia_scope(include=False):
      d = tf.matmul(a, c)  # excluded in segment
    with tf.contrib.tensorrt.nvidia_scope(
        select=lambda node_def: 'matmul' in node_def.op.lower()):
      e = tf.matmul(a, b) + d  # matmul is included, the addition is excluded.

  Yields:
    The current scope, enabling or disabling inclusion.

  """
  # enum would have been better here but we need attrvalue to have these enums
  # which is not possible. So we stick with strings.
  optimizer_ = "TRT"
  precision_ = "FP32"
  engine_mode_ = "DYNAMIC"
  maximum_cached_engines_ = 1
  maximum_workspace_size_ = -1

  if callable(select):
    def segment_include(node_def):
      return attr_value_pb2.AttrValue(b=select(node_def))
  else:
    segment_include = attr_value_pb2.AttrValue(b=select)

  attrs = {"_NvidiaSegmentInclude": segment_include}

  if scope_name:
    attrs["_NvidiaSegmentName"] = ConditionalWrapper(
        segment_include, attr_value_pb2.AttrValue(s=scope_name.encode()))

  if optimizer:
    if optimizer.upper() not in ["TRT"]:
      raise RuntimeError("Nvidia Scope: optimizer must be 'TRT'")
    else:
      optimizer_ = optimizer.upper()
      attrs["_NvidiaOptimizer"] = ConditionalWrapper(
          segment_include, attr_value_pb2.AttrValue(s=optimizer_.encode()))

  if precision:
    if precision.upper() not in ["FP32", "FP16", "INT8"]:
      raise RuntimeError(
          "Nvidia Scope: precision must be one of ['FP32', 'FP16', 'INT8']")
    else:
      precision_ = precision.upper()
      attrs["_NvidiaPrecision"] = ConditionalWrapper(
          segment_include, attr_value_pb2.AttrValue(s=precision_.encode()))
  if engine_mode:
    if engine_mode.upper() not in ["STATIC", "DYNAMIC"]:
      raise RuntimeError(
          "Nvidia Scope: engine_mode must be either 'STATIC' or 'DYNAMIC'")
    else:
      engine_mode_ = engine_mode.upper()
      attrs["_NvidiaEngineMode"] = ConditionalWrapper(
          segment_include, attr_value_pb2.AttrValue(s=engine_mode_.encode()))

  if maximum_cached_engines:
    if not isinstance(maximum_cached_engines,
                      int) or maximum_cached_engines < 1:
      raise RuntimeError(
          "Nvidia Scope: maximum_cached_engines must be an int >=1")
    else:
      maximum_cached_engines_ = maximum_cached_engines
      attrs["_NvidiaMaxEngines"] = ConditionalWrapper(
          segment_include, attr_value_pb2.AttrValue(i=maximum_cached_engines_))
  if maximum_workspace_size:
    if not isinstance(maximum_workspace_size, int):
      raise RuntimeError(
          "Nvidia Scope: maximum_workspace_size must be an integer")
    else:
      maximum_workspace_size_ = maximum_workspace_size
      attrs["_NvidiaMaxWorkspace"] = ConditionalWrapper(
          segment_include, attr_value_pb2.AttrValue(i=maximum_workspace_size_))

  # Find the singleton counter for the current scoped graph.  If it
  # doesn't exist, create one.
  nvidia_scope_counter = ops.get_collection(_NV_SCOPE_KEY)
  if not nvidia_scope_counter:
    nvidia_scope_counter = NvidiaScope(0)
    ops.add_to_collection(_NV_SCOPE_KEY, nvidia_scope_counter)
  else:
    nvidia_scope_counter = nvidia_scope_counter[0]

  if nvidia_scope_counter.depth == 0:
    # If we're at the root scope, set defaults for non-user set values.
    # scope name is not set if user doesn't specify it. In that case,
    # segmenter decides on names
    # For any sub-scopes, parent attributes are inherited if user doesn't
    # override them. Optimizer will try to merge different scopes
    # with same attributes
    attrs["_NvidiaOptimizer"] = ConditionalWrapper(
        segment_include, attr_value_pb2.AttrValue(s=optimizer_.encode()))
    attrs["_NvidiaPrecision"] = ConditionalWrapper(
        segment_include, attr_value_pb2.AttrValue(s=precision_.encode()))
    attrs["_NvidiaEngineMode"] = ConditionalWrapper(
        segment_include, attr_value_pb2.AttrValue(s=engine_mode_.encode()))
    attrs["_NvidiaMaxEngines"] = ConditionalWrapper(
        segment_include, attr_value_pb2.AttrValue(i=maximum_cached_engines_))
    attrs["_NvidiaMaxWorkspace"] = ConditionalWrapper(
        segment_include, attr_value_pb2.AttrValue(i=maximum_workspace_size_))
  nvidia_scope_counter.depth += 1

  # pylint: disable=protected-access
  with ops.get_default_graph()._attr_scope(attrs):
    yield
  # pylint: enable=protected-access

  nvidia_scope_counter.depth -= 1

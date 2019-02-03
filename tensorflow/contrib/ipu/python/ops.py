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
# =============================================================================

"""Ops related to the Graphcore IPU."""

from tensorflow.compiler.plugin.poplar.ops import gen_ipu_ops
from tensorflow.compiler.xla import xla_data_pb2
from tensorflow.core.framework import attr_value_pb2
from tensorflow.core.framework import summary_pb2
from tensorflow.python.framework import ops
from tensorflow.python.ops.variable_scope import variable_scope
from tensorflow.python.summary.summary import tensor_summary
from tensorflow.python.util import tf_contextlib

def ipu_compile_summary(name, op, collections=None):
  """Create an IPU compiler summary operation.

  Args:
    name: A name for the summary
    op: An operation to make this summary dependent upon
    collections: Optional collections to add the summary into

  Returns:
    The new summary operation.
  """

  with ops.device("cpu"):
    with ops.control_dependencies([op]):

      reports = gen_ipu_ops.ipu_event_trace()

      summary_metadata = summary_pb2.SummaryMetadata(
        plugin_data=summary_pb2.SummaryMetadata.PluginData(
          plugin_name="ipu"))

      t_summary = tensor_summary(name='ipu_trace', tensor=reports,
                                 summary_metadata=summary_metadata,
                                 collections=collections, display_name=name)

  return t_summary


@tf_contextlib.contextmanager
def ipu_jit_scope(ipu_scope):
  scope = "jit_scope_ipu_" + str(ipu_scope)
  attrs = {
    "_XlaCompile": attr_value_pb2.AttrValue(b=True),
    "_XlaSeparateCompiledGradients": attr_value_pb2.AttrValue(b=False),
    "_XlaScope": attr_value_pb2.AttrValue(s=scope.encode())
  }

  # pylint: disable=protected-access
  with ops.get_default_graph()._attr_scope(attrs):
    yield
  # pylint: enable=protected-access


@tf_contextlib.contextmanager
def ipu_scope(device):
  with variable_scope('', use_resource=True):
    with ops.device(device):
      with ipu_jit_scope(0) as scope:
        yield scope

@tf_contextlib.contextmanager
def ipu_shard(index):

  ipus = []
  if hasattr(index, '__iter__'):
    ipus = index
  else:
    ipus = [index]

  proto = xla_data_pb2.OpSharding(
    type=xla_data_pb2.OpSharding.MAXIMAL, tile_assignment_devices=ipus)

  attr_value = attr_value_pb2.AttrValue(s=proto.SerializeToString())
  attrs = {"_XlaSharding": attr_value}

  # pylint: disable=protected-access
  with ops.get_default_graph()._attr_scope(attrs):
    yield
  # pylint: enable=protected-access

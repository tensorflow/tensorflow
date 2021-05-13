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
# ==============================================================================

"""API for enabling v2 control flow."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_util
from tensorflow.python.ops import control_flow_util_v2
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util.tf_export import tf_export


@tf_export(v1=["enable_control_flow_v2"])
def enable_control_flow_v2():  # pylint: disable=invalid-name
  """Use control flow v2.

  control flow v2 (cfv2) is an improved version of control flow in TensorFlow
  with support for higher order derivatives. Enabling cfv2 will change the
  graph/function representation of control flow, e.g., `tf.while_loop` and
  `tf.cond` will generate functional `While` and `If` ops instead of low-level
  `Switch`, `Merge` etc. ops. Note: Importing and running graphs exported
  with old control flow will still be supported.

  Calling tf.enable_control_flow_v2() lets you opt-in to this TensorFlow 2.0
  feature.

  Note: v2 control flow is always enabled inside of tf.function. Calling this
  function is not required.
  """
  # pylint: disable=protected-access
  logging.vlog(1, "Enabling control flow v2")
  ops._control_flow_api_gauge.get_cell().set(True)
  control_flow_util.ENABLE_CONTROL_FLOW_V2 = True


@tf_export(v1=["disable_control_flow_v2"])
def disable_control_flow_v2():  # pylint: disable=invalid-name
  """Opts out of control flow v2.

  Note: v2 control flow is always enabled inside of tf.function. Calling this
  function has no effect in that case.

  If your code needs tf.disable_control_flow_v2() to be called to work
  properly please file a bug.
  """
  # pylint: disable=protected-access
  logging.vlog(1, "Disabling control flow v2")
  ops._control_flow_api_gauge.get_cell().set(False)
  control_flow_util.ENABLE_CONTROL_FLOW_V2 = False


@tf_export(v1=["control_flow_v2_enabled"])
def control_flow_v2_enabled():  # pylint: disable=invalid-name
  """Returns `True` if v2 control flow is enabled.

  Note: v2 control flow is always enabled inside of tf.function.
  """
  return control_flow_util.EnableControlFlowV2(ops.get_default_graph())


@tf_export(v1=["experimental.output_all_intermediates"])
def output_all_intermediates(state):  # pylint: disable=invalid-name
  """Whether to output all intermediates from functional control flow ops.

  The "default" behavior to is to output all intermediates when using v2 control
  flow inside Keras models in graph mode (possibly inside Estimators). This is
  needed to support taking gradients of v2 control flow. In graph mode, Keras
  can sometimes freeze the forward graph before the gradient computation which
  does not work for v2 control flow since it requires updating the forward ops
  to output the needed intermediates. We work around this by proactively
  outputting the needed intermediates when building the forward pass itself.
  Ideally any such extra tensors should be pruned out at runtime. However, if
  for any reason this doesn't work for you or if you have an inference-only
  model you can turn this behavior off using
  `tf.compat.v1.experimental.output_all_intermediates(False)`.

  If with the default behavior you are still seeing errors of the form
  "Connecting to invalid output X of source node Y which has Z outputs" try
  setting `tf.compat.v1.experimental.output_all_intermediates(True)` and
  please file an issue at https://github.com/tensorflow/tensorflow/issues.

  Args:
    state: True, False or None. None restores the default behavior.
  """
  control_flow_util_v2._EXPERIMENTAL_OUTPUT_ALL_INTERMEDIATES_OVERRIDE = state  # pylint: disable=protected-access

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
"""Switching v2 features on and off."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python import tf2
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import control_flow_v2_toggles
from tensorflow.python.ops import variable_scope

from tensorflow.python.util.tf_export import tf_export


@tf_export(v1=["enable_v2_behavior"])
def enable_v2_behavior():
  """Enables TensorFlow 2.x behaviors.

  This function can be called at the beginning of the program (before `Tensors`,
  `Graphs` or other structures have been created, and before devices have been
  initialized. It switches all global behaviors that are different between
  TensorFlow 1.x and 2.x to behave as intended for 2.x.

  This function is called in the main TensorFlow `__init__.py` file, user should
  not need to call it, except during complex migrations.
  """
  # TF2 behavior is enabled if either 1) enable_v2_behavior() is called or
  # 2) the TF2_BEHAVIOR=1 environment variable is set.  In the latter case,
  # the modules below independently check if tf2.enabled().
  tf2.enable()
  ops.enable_eager_execution()
  tensor_shape.enable_v2_tensorshape()  # Also switched by tf2
  variable_scope.enable_resource_variables()
  ops.enable_tensor_equality()
  # Enables TensorArrayV2 and control flow V2.
  control_flow_v2_toggles.enable_control_flow_v2()


@tf_export(v1=["disable_v2_behavior"])
def disable_v2_behavior():
  """Disables TensorFlow 2.x behaviors.

  This function can be called at the beginning of the program (before `Tensors`,
  `Graphs` or other structures have been created, and before devices have been
  initialized. It switches all global behaviors that are different between
  TensorFlow 1.x and 2.x to behave as intended for 1.x.

  User can call this function to disable 2.x behavior during complex migrations.
  """
  tf2.disable()
  ops.disable_eager_execution()
  tensor_shape.disable_v2_tensorshape()  # Also switched by tf2
  variable_scope.disable_resource_variables()
  ops.disable_tensor_equality()
  # Disables TensorArrayV2 and control flow V2.
  control_flow_v2_toggles.disable_control_flow_v2()

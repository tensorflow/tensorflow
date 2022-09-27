# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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
"""Saves a TensorFlow model containing ReadVariableOp nodes.

   The saved model is loaded and executed by tests to check that TF-TRT can
   successfully convert and execute models with variables without freezing.
"""

import numpy as np
from tensorflow.python.eager import def_function
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_spec
from tensorflow.python.module import module
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variables
from tensorflow.python.saved_model import save


class MyModel(module.Module):
  """Simple model with two variables."""

  def __init__(self):
    self.var1 = variables.Variable(
        np.array([[[13.]]], dtype=np.float32), name="var1")
    self.var2 = variables.Variable(
        np.array([[[37.]]], dtype=np.float32), name="var2")

  @def_function.function
  def __call__(self, input1, input2):
    mul1 = input1 * self.var1
    mul2 = input2 * self.var2
    add = mul1 + mul2
    sub = add - 45.
    return array_ops.identity(sub, name="output")


def GenerateModelWithReadVariableOp(tf_saved_model_dir):
  """Generate a model with ReadVariableOp nodes."""
  my_model = MyModel()
  cfunc = my_model.__call__.get_concrete_function(
      tensor_spec.TensorSpec([None, 1, 1], dtypes.float32),
      tensor_spec.TensorSpec([None, 1, 1], dtypes.float32))
  # pylint: disable=not-callable
  save(my_model, tf_saved_model_dir, signatures=cfunc)


if __name__ == "__main__":
  GenerateModelWithReadVariableOp(
      tf_saved_model_dir="tf_readvariableop_saved_model")

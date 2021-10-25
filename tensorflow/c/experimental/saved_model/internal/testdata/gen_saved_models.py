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
# Lint as: python3
"""Creates saved models used for testing.

This executable should be run with an argument pointing to the testdata/ folder
in this directory. It will re-generate the saved models that are used for
testing.
"""

import os
from absl import app

from tensorflow.python.compat import v2_compat

from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_spec
from tensorflow.python.module import module
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variables
from tensorflow.python.saved_model import saved_model


def _gen_uninitialized_variable(base_dir):
  """Generates a saved model with an uninitialized variable."""

  class SubModule(module.Module):
    """A module with an UninitializedVariable."""

    def __init__(self):
      self.uninitialized_variable = resource_variable_ops.UninitializedVariable(
          name="uninitialized_variable", dtype=dtypes.int64)

  class Module(module.Module):
    """A module with an UninitializedVariable."""

    def __init__(self):
      super(Module, self).__init__()
      self.sub_module = SubModule()
      self.initialized_variable = variables.Variable(
          1.0, name="initialized_variable")
      # An UninitializedVariable with the same name as the variable in the
      # SubModule, but with a different type.
      self.uninitialized_variable = resource_variable_ops.UninitializedVariable(
          name="uninitialized_variable", dtype=dtypes.float32)

    @def_function.function(
        input_signature=[tensor_spec.TensorSpec((), dtypes.float32)])
    def compute(self, value):
      return self.initialized_variable + value

  to_save = Module()
  saved_model.save(
      to_save, export_dir=os.path.join(base_dir, "UninitializedVariable"))


def _gen_simple_while_loop(base_dir):
  """Generates a saved model with a while loop."""

  class Module(module.Module):
    """A module with a while loop."""

    @def_function.function(
        input_signature=[tensor_spec.TensorSpec((), dtypes.float32)])
    def compute(self, value):
      acc, _ = control_flow_ops.while_loop(
          cond=lambda acc, i: i > 0,
          body=lambda acc, i: (acc + i, i - 1),
          loop_vars=(constant_op.constant(0.0), value))
      return acc

  to_save = Module()
  saved_model.save(
      to_save, export_dir=os.path.join(base_dir, "SimpleWhileLoop"))


def main(args):
  if len(args) != 2:
    raise app.UsageError("Expected one argument (base_dir).")
  _, base_dir = args
  _gen_uninitialized_variable(base_dir)
  _gen_simple_while_loop(base_dir)


if __name__ == "__main__":
  v2_compat.enable_v2_behavior()
  app.run(main)

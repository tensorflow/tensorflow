# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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

from tensorflow.python.eager import context
from tensorflow.python.eager.polymorphic_function import atomic_function
from tensorflow.python.eager.polymorphic_function import polymorphic_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.platform import test
from tensorflow.python.util import compat


def get_function_def_and_type(foo, inputs):
  """Traces `foo` generate the FunctionDef and FunctionType."""
  concrete = polymorphic_function.function(foo).get_concrete_function(*inputs)
  atomic = concrete._inference_function
  return atomic.definition, atomic.function_type


def make_atomic_function(definition, func_type):
  bound_context = context.context()
  if bound_context.has_function(compat.as_bytes(definition.signature.name)):
    raise ValueError("Function already registered in context.")

  bound_context.add_function_def(definition)
  return atomic_function.AtomicFunction(
      definition.signature.name, bound_context, func_type
  )


class AtomicFunctionTest(test.TestCase):

  def test_call_eager(self):
    definition, func_type = get_function_def_and_type(
        lambda x, y: x + y, (constant_op.constant(1), constant_op.constant(2))
    )

    atomic = make_atomic_function(definition, func_type)

    self.assertEqual(
        atomic(constant_op.constant(3), constant_op.constant(4))[0].numpy(),
        7,
    )

  def test_call_graph(self):
    definition, func_type = get_function_def_and_type(
        lambda x, y: x + y, (constant_op.constant(1), constant_op.constant(2))
    )

    atomic = make_atomic_function(definition, func_type)

    @polymorphic_function.function
    def foo(a, b):
      return atomic(a, b)[0]

    self.assertEqual(
        foo(constant_op.constant(3), constant_op.constant(4)).numpy(),
        7,
    )

  def test_variable_input_eager(self):
    definition, func_type = get_function_def_and_type(
        lambda x, y: x + y,
        (resource_variable_ops.ResourceVariable(1), constant_op.constant(2)),
    )

    atomic = make_atomic_function(definition, func_type)

    self.assertEqual(
        atomic(
            resource_variable_ops.ResourceVariable(3)._handle,
            constant_op.constant(4),
        )[0].numpy(),
        7,
    )

  def test_variable_input_graph(self):
    definition, func_type = get_function_def_and_type(
        lambda x, y: x + y,
        (resource_variable_ops.ResourceVariable(1), constant_op.constant(2)),
    )

    atomic = make_atomic_function(definition, func_type)

    @polymorphic_function.function
    def foo(a, b):
      return atomic(a, b)[0]

    self.assertEqual(
        foo(
            resource_variable_ops.ResourceVariable(3)._handle,
            constant_op.constant(4),
        ).numpy(),
        7,
    )


if __name__ == "__main__":
  ops.enable_eager_execution()
  test.main()

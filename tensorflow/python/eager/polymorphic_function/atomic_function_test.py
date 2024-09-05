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

from tensorflow.python.eager.polymorphic_function import atomic_function
from tensorflow.python.eager.polymorphic_function import polymorphic_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.platform import test


def get_function_def_and_type(foo, inputs):
  """Traces `foo` generate the FunctionDef and FunctionType."""
  concrete = polymorphic_function.function(foo).get_concrete_function(*inputs)
  atomic = concrete._inference_function
  return atomic.definition, atomic.function_type


class AtomicFunctionTest(test.TestCase):

  def test_call_eager(self):
    definition, func_type = get_function_def_and_type(
        lambda x, y: x + y, (constant_op.constant(1), constant_op.constant(2))
    )

    atomic = atomic_function.from_function_def(definition, func_type)

    self.assertRegex(
        str(atomic),
        r"<AtomicFunction> .*(x: TensorSpec.*, y: TensorSpec.*) ->"
        r" TensorSpec.*",
    )
    self.assertRegex(
        repr(atomic).replace("\n", " "),
        r"AtomicFunction.*name.*bound_context.*function_type.*"
        r"children.*call_options.*cached_graph.*",
    )

    self.assertEqual(
        atomic.call_flat(constant_op.constant(3), constant_op.constant(4))[
            0
        ].numpy(),
        7,
    )

  def test_call_graph(self):
    definition, func_type = get_function_def_and_type(
        lambda x, y: x + y, (constant_op.constant(1), constant_op.constant(2))
    )

    atomic = atomic_function.from_function_def(definition, func_type)

    @polymorphic_function.function
    def foo(a, b):
      return atomic.call_flat(a, b)[0]

    self.assertEqual(
        foo(constant_op.constant(3), constant_op.constant(4)).numpy(),
        7,
    )

  def test_variable_input_eager(self):
    definition, func_type = get_function_def_and_type(
        lambda x, y: x + y,
        (resource_variable_ops.ResourceVariable(1), constant_op.constant(2)),
    )

    atomic = atomic_function.from_function_def(definition, func_type)

    self.assertEqual(
        atomic.call_flat(
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

    atomic = atomic_function.from_function_def(definition, func_type)

    @polymorphic_function.function
    def foo(a, b):
      return atomic.call_flat(a, b)[0]

    self.assertEqual(
        foo(
            resource_variable_ops.ResourceVariable(3)._handle,
            constant_op.constant(4),
        ).numpy(),
        7,
    )

  def test_call_with_captures(self):
    my_capture = constant_op.constant(2)

    @polymorphic_function.function
    def foo(x):
      my_dict = {}
      my_dict["my_tensor"] = x["my_tensor"]
      my_dict["my_resource"] = x["my_variable"].handle
      my_dict["my_capture"] = my_capture
      my_dict["my_ints"] = x["my_ints"]
      return my_dict

    structured_inputs = {
        "my_tensor": constant_op.constant(1),
        "my_variable": resource_variable_ops.ResourceVariable(1),
        "my_ints": [1, 2, 3],
    }

    function_def, function_type = get_function_def_and_type(
        foo, (structured_inputs,)
    )

    atomic = atomic_function.from_function_def(function_def, function_type)

    with self.assertRaisesRegex(ValueError, "Use call_with_captures instead."):
      atomic(structured_inputs)

    result = atomic.call_with_captures((structured_inputs,), {}, [my_capture])
    self.assertEqual(
        result["my_tensor"].numpy(), structured_inputs["my_tensor"].numpy()
    )
    self.assertEqual(result["my_resource"].dtype, dtypes.resource)
    self.assertEqual(result["my_capture"].numpy(), my_capture.numpy())
    self.assertEqual(result["my_ints"][0].numpy(), 1)
    self.assertEqual(result["my_ints"][1].numpy(), 2)
    self.assertEqual(result["my_ints"][2].numpy(), 3)

  def test_call(self):
    @polymorphic_function.function
    def foo(x):
      my_dict = {}
      my_dict["my_tensor"] = x["my_tensor"]
      my_dict["my_resource"] = x["my_variable"].handle
      my_dict["my_ints"] = x["my_ints"]
      return my_dict

    structured_inputs = {
        "my_tensor": constant_op.constant(1),
        "my_variable": resource_variable_ops.ResourceVariable(1),
        "my_ints": [1, 2, 3],
    }

    function_def, function_type = get_function_def_and_type(
        foo, (structured_inputs,)
    )

    atomic = atomic_function.from_function_def(function_def, function_type)

    result = atomic(structured_inputs)
    self.assertEqual(
        result["my_tensor"].numpy(), structured_inputs["my_tensor"].numpy()
    )
    self.assertEqual(result["my_resource"].dtype, dtypes.resource)
    self.assertEqual(result["my_ints"][0].numpy(), 1)
    self.assertEqual(result["my_ints"][1].numpy(), 2)
    self.assertEqual(result["my_ints"][2].numpy(), 3)

if __name__ == "__main__":
  ops.enable_eager_execution()
  test.main()

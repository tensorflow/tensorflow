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
# ==============================================================================
"""Tests for high-level function transformation API."""

import collections

from absl.testing import parameterized

from tensorflow.core.function.testing import test_pass
from tensorflow.core.function.transform import transform
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import test_util
from tensorflow.python.module import module as module_lib
from tensorflow.python.ops import custom_gradient
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import while_loop
from tensorflow.python.platform import test
from tensorflow.python.saved_model import load as load_lib
from tensorflow.python.saved_model import save as save_lib


def add_to_multiply(fndef):
  for node in fndef.node_def:
    if node.name.endswith("x_plus_y"):
      node.name = node.name.replace("x_plus_y", "x_times_y")
      node.op = "Mul"
    for idx, inp in enumerate(node.input):
      if inp.endswith("x_plus_y:z:0"):
        node.input[idx] = inp.replace("x_plus_y", "x_times_y")


class Model(module_lib.Module):

  @def_function.function
  def f(self, x, y, add_2):
    r = math_ops.add(x, y, name="x_plus_y")
    if add_2:
      return r + 2
    else:
      return r


def apply_transform(f, transform_fn):
  """Wrapper to apply a transformation on every traced tf.function."""

  @def_function.function
  def wrapped(*args):
    updated_cf = transform.transform_function(
        f, inputs=args, transform_fn=transform_fn)
    return updated_cf(*args)

  return wrapped


class TransformTest(test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name="transform",
          transform_fn=add_to_multiply,
          mlir_pipeline=None),
      dict(
          testcase_name="mlir_pipeline",
          transform_fn=None,
          mlir_pipeline="test-pass"),
      dict(
          testcase_name="transform_and_mlir_pipeline",
          transform_fn=add_to_multiply,
          mlir_pipeline="test-pass"),
      dict(
          testcase_name="transform_list",
          transform_fn=[add_to_multiply],
          mlir_pipeline=None),
      dict(
          testcase_name="mlir_pipeline_list",
          transform_fn=None,
          mlir_pipeline=["test-pass"]),
      dict(
          testcase_name="transform_list_and_mlir_pipeline_list",
          transform_fn=[add_to_multiply],
          mlir_pipeline=["test-pass"]),
  )
  @test_util.run_v2_only
  def test_concrete_function_with(self, transform_fn, mlir_pipeline):

    @def_function.function(input_signature=[
        tensor_spec.TensorSpec((), dtype=dtypes.float32),
        tensor_spec.TensorSpec((), dtype=dtypes.float32)
    ])
    def f(x, y):
      return math_ops.add(x, y, name="x_plus_y")

    # transfrom f(x, y): x + y -> f(x, y): x * y
    f = transform.transform_function(
        f, transform_fn=transform_fn, mlir_pipeline=mlir_pipeline)
    one = constant_op.constant(1.0)
    self.assertEqual(f(one, one), 1.0)

    @def_function.function
    def f2(x, y):
      z = f(x, y)
      return math_ops.add(z, 10.0)

    self.assertEqual(f2(one, one), 11.0)

    @def_function.function
    def f_g(x, y):
      z = f(x, y)
      dz_dx, dz_dy = gradients_impl.gradients(z, [x, y])
      return math_ops.add(dz_dx, dz_dy)

    self.assertEqual(
        f_g(constant_op.constant(2.0), constant_op.constant(3.0)), 5.0)

  @test_util.run_v2_only
  def test_transform_with_keywords(self):

    @def_function.function
    def f(x, y):
      return math_ops.add(x, y, name="x_plus_y")

    one = constant_op.constant(1.0)

    # transfrom f(x, y): x + y -> f(x, y): x * y
    g = transform.transform_function(
        f,
        inputs=[one],
        kw_inputs={"y": one},
        transform_fn=add_to_multiply,
        mlir_pipeline="test-pass")

    self.assertEqual(g(one, one), 1.0)
    self.assertEqual(g(one, y=one), 1.0)
    self.assertEqual(g(x=one, y=one), 1.0)

  @test_util.run_v2_only
  def test_transform_with_keywords_only(self):

    @def_function.function
    def f(x, y):
      return math_ops.add(x, y, name="x_plus_y")

    one = constant_op.constant(1.0)

    # transfrom f(x, y): x + y -> f(x, y): x * y
    g = transform.transform_function(
        f,
        inputs=None,
        kw_inputs={"x": one, "y": one},
        transform_fn=add_to_multiply,
        mlir_pipeline="test-pass")

    self.assertEqual(g(one, one), 1.0)
    self.assertEqual(g(one, y=one), 1.0)
    self.assertEqual(g(x=one, y=one), 1.0)

  @test_util.run_v2_only
  def test_function_spec(self):

    @def_function.function
    def f(x, y):
      return math_ops.add(x, y, name="x_plus_y")

    args = [1, 1]
    self.assertEqual(f(*args), 2)

    updated_f = transform.transform_function(
        f, inputs=args, transform_fn=add_to_multiply)
    self.assertEqual(updated_f(*args), 1)

    self.assertSequenceAlmostEqual(
        f.get_concrete_function(
            *args).pretty_printed_signature().split("\n")[1:],
        updated_f.pretty_printed_signature().split("\n")[1:])

  @test_util.run_v2_only
  def test_transform_with_custom_gradients(self):

    @custom_gradient.custom_gradient
    def add(x, y):
      e = math_ops.add(x, y, name="x_plus_y")

      # custom gradient that returns gradients of x * y instead of x + y
      def grad(upstream):
        dz_dx = y
        dz_dy = x
        return upstream * dz_dx, upstream * dz_dy

      return e, grad

    @def_function.function
    def f(x, y):
      return add(x, y)

    one = constant_op.constant(1.0)
    f = transform.transform_function(
        f, inputs=[one, one], transform_fn=add_to_multiply)
    self.assertEqual(f(one, one), 1.0)

    @def_function.function
    def f_g(x, y):
      z = f(x, y)
      dz_dx, dz_dy = gradients_impl.gradients(z, [x, y])
      return math_ops.add(dz_dx, dz_dy)

    self.assertEqual(
        f_g(constant_op.constant(2.0), constant_op.constant(3.0)), 5.0)

  @test_util.run_v2_only
  def test_transform_with_nested_function(self):

    @def_function.function
    def f(x, z):

      @def_function.function
      def add():
        i = constant_op.constant(1.0)
        c = lambda i: math_ops.less(i, 3.0)
        b = lambda i: (math_ops.add(i, z, name="x_plus_y"))
        i = while_loop.while_loop_v2(c, b, [i])
        return i

      y = add()
      return math_ops.add(x, y, name="x_plus_y")

    one = constant_op.constant(1.0)
    two = constant_op.constant(2.0)
    inputs = [one, two]

    # By default only `f` is transformed.
    updated_f = transform.transform_function(
        f, inputs=inputs, transform_fn=add_to_multiply)
    self.assertEqual(updated_f(*inputs), 3.0)  # 1 x (1 + 2) = 3

    # Extract all the functions in `f`'s library that we want to transform.
    nested_transforms = {}
    gdef = f.get_concrete_function(*inputs).graph.as_graph_def()
    for fdef in gdef.library.function:
      nested_transforms[fdef.signature.name] = add_to_multiply

    # Transform `f` and all of its library functions.
    updated_f = transform.transform_function(
        f,
        inputs=inputs,
        transform_fn=add_to_multiply,
        nested_fn_transforms=nested_transforms)
    self.assertEqual(updated_f(*inputs), 4.0)  # 1 x (1 x 2 x 2) = 4

  @parameterized.named_parameters(
      dict(
          testcase_name="fn",
          transform_fn=add_to_multiply),
      dict(
          testcase_name="mlir",
          mlir_pipeline="test-pass"),
      dict(
          testcase_name="fn_and_mlir",
          transform_fn=add_to_multiply,
          mlir_pipeline="test-pass"),
  )
  @test_util.run_v2_only
  def test_nested_python_function_transform_with(self,
                                                 transform_fn=None,
                                                 mlir_pipeline=None):

    @def_function.function
    def f(x, y):

      def inner_add():
        return math_ops.add(x, y, name="x_plus_y")

      return inner_add()

    inputs = [1.0, 2.0]
    self.assertEqual(f(*inputs), 3.0)  # 1 + 2 = 3

    # Transform `f`.
    updated_f = transform.transform_function(
        f,
        inputs=inputs,
        transform_fn=transform_fn,
        mlir_pipeline=mlir_pipeline)
    # Nested Python functions should be transformed.
    self.assertEqual(updated_f(*inputs), 2.0)  # 1 x 2 = 2

  @parameterized.named_parameters(
      dict(
          testcase_name="fn_and_nested_fn",
          transform_fn=add_to_multiply,
          nested_fn=True),
      dict(
          testcase_name="fn_and_nested_mlir",
          transform_fn=add_to_multiply,
          nested_mlir=True),
      dict(
          testcase_name="mlir_and_nested_fn",
          mlir_pipeline="test-pass",
          nested_fn=True),
      dict(
          testcase_name="mlir_and_nested_mlir",
          mlir_pipeline="test-pass",
          nested_mlir=True),
      dict(
          testcase_name="mlir_fn_and_nested_mlir_fn",
          transform_fn=add_to_multiply,
          mlir_pipeline="test-pass",
          nested_fn=True,
          nested_mlir=True),
  )
  @test_util.run_v2_only
  def test_nested_transform_with(self,
                                 transform_fn=None,
                                 mlir_pipeline=None,
                                 nested_fn=False,
                                 nested_mlir=False):

    @def_function.function
    def f(x, y, z):

      @def_function.function
      def inner_add():
        return math_ops.add(y, z, name="x_plus_y")

      return math_ops.add(x, inner_add(), name="x_plus_y")

    # 1, 2, 4 are picked so the following combinations create different results.
    # 1 + (2 + 4) = 7
    # 1 + (2 * 4) = 9
    # 1 * (2 + 4) = 6
    # 1 * (2 * 4) = 8
    inputs = [1.0, 2.0, 4.0]
    self.assertEqual(f(*inputs), 7.0)  # 1 + (2 + 4) = 7

    # Extract all the functions in `f`'s library that we want to transform.
    nested_fn_transforms = {}
    nested_mlir_transforms = {}

    cf = f.get_concrete_function(*inputs)
    gdef = cf.graph.as_graph_def()
    for fdef in gdef.library.function:
      fdef_name = fdef.signature.name
      if nested_fn:
        nested_fn_transforms[fdef_name] = add_to_multiply
      if nested_mlir:
        nested_mlir_transforms[fdef_name] = "test-pass"

    # Transform `f` and all of its library functions.
    updated_f = transform.transform_function(
        f,
        inputs=inputs,
        transform_fn=transform_fn,
        mlir_pipeline=mlir_pipeline,
        nested_fn_transforms=nested_fn_transforms,
        nested_mlir_transforms=nested_mlir_transforms)
    self.assertEqual(updated_f(*inputs), 8.0)  # 1 x (2 x 4) = 8

  @parameterized.named_parameters(
      dict(
          testcase_name="fn_and_nested_fn",
          transform_fn=add_to_multiply,
          nested_fn=True),
      dict(
          testcase_name="fn_and_nested_mlir",
          transform_fn=add_to_multiply,
          nested_mlir=True),
      dict(
          testcase_name="mlir_and_nested_fn",
          mlir_pipeline="test-pass",
          nested_fn=True),
      dict(
          testcase_name="mlir_and_nested_mlir",
          mlir_pipeline="test-pass",
          nested_mlir=True),
      dict(
          testcase_name="mlir_fn_and_nested_mlir_fn",
          transform_fn=add_to_multiply,
          mlir_pipeline="test-pass",
          nested_fn=True,
          nested_mlir=True),
  )
  @test_util.run_v2_only
  def test_nested_transform_in_tf_function_with(self,
                                                transform_fn=None,
                                                mlir_pipeline=None,
                                                nested_fn=False,
                                                nested_mlir=False):

    @def_function.function
    def g(x, y, z):

      @def_function.function
      def f(x, y, z):

        @def_function.function
        def inner_add():
          return math_ops.add(y, z, name="x_plus_y")

        return math_ops.add(x, inner_add(), name="x_plus_y")

      nested_fn_transforms = {}
      nested_mlir_transforms = {}

      cf = f.get_concrete_function(*inputs)
      gdef = cf.graph.as_graph_def()
      for fdef in gdef.library.function:
        fdef_name = fdef.signature.name
        if nested_fn:
          nested_fn_transforms[fdef_name] = add_to_multiply
        if nested_mlir:
          nested_mlir_transforms[fdef_name] = "test-pass"

      updated_f = transform.transform_function(
          f,
          inputs=inputs,
          transform_fn=transform_fn,
          mlir_pipeline=mlir_pipeline,
          nested_fn_transforms=nested_fn_transforms,
          nested_mlir_transforms=nested_mlir_transforms)

      return updated_f(x, y, z)

    inputs = [1.0, 2.0, 4.0]
    graph_def = g.get_concrete_function(*inputs).graph.as_graph_def()
    # Confirm all "AddV2" nodes in the library functions of graph_def are
    # transformed to "Mul".
    ops = collections.Counter()
    for fdef in graph_def.library.function:
      for node in fdef.node_def:
        ops[node.op] += 1

    self.assertNotIn("AddV2", ops)
    self.assertEqual(ops["Mul"], 2)

  @test_util.run_v2_only
  def test_save_transform_for_all_signatures(self):
    m = Model()
    # Originally f does addition
    self.assertEqual(
        m.f(
            constant_op.constant(1, dtypes.int32),
            constant_op.constant(2, dtypes.int32),
            constant_op.constant(False, dtypes.bool)), 3)

    self.assertEqual(
        m.f(
            constant_op.constant(1, dtypes.int32),
            constant_op.constant(2, dtypes.int32),
            constant_op.constant(True, dtypes.bool)), 5)

    # Transform every input signature of f to a multiply
    m.f = apply_transform(m.f, add_to_multiply)

    # Validate arbitrary signatures.
    self.assertEqual(
        m.f(
            constant_op.constant(1, dtypes.int32),
            constant_op.constant(2, dtypes.int32),
            constant_op.constant(False, dtypes.bool)), 2)
    self.assertEqual(
        m.f(
            constant_op.constant(1, dtypes.int32),
            constant_op.constant(2, dtypes.int32),
            constant_op.constant(True, dtypes.bool)), 4)
    self.assertEqual(
        m.f(
            constant_op.constant(1.0, dtypes.float32),
            constant_op.constant(2.0, dtypes.float32),
            constant_op.constant(False, dtypes.bool)), (2.0))

    # Save and restore the model.
    save_lib.save(m, "/tmp/testing_model")
    m_loaded = load_lib.load("/tmp/testing_model")

    # Validate the restored model.
    self.assertEqual(
        m_loaded.f(
            constant_op.constant(1, dtypes.int32),
            constant_op.constant(2, dtypes.int32),
            constant_op.constant(False, dtypes.bool)), 2)
    self.assertEqual(
        m_loaded.f(
            constant_op.constant(1.1, dtypes.float32),
            constant_op.constant(2.0, dtypes.float32),
            constant_op.constant(True, dtypes.bool)), (4.2))


if __name__ == "__main__":
  test_pass.RegisterTestPass()
  test.main()

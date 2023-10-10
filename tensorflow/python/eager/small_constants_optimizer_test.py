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

import contextlib
import timeit

from tensorflow.python.eager import context
from tensorflow.python.eager.polymorphic_function import polymorphic_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.platform import test


@contextlib.contextmanager
def options(optimizer_options):
  old_opts = context.context().get_optimizer_experimental_options()
  context.context().set_optimizer_experimental_options(optimizer_options)
  try:
    yield
  finally:
    context.context().set_optimizer_experimental_options(old_opts)


class FunctionTest(test.TestCase):

  @test_util.run_v2_only
  def test_grappler_optimization(self):
    @polymorphic_function.function
    def brancher(inp):
      x = constant_op.constant(1)
      for _ in range(1000):
        if inp:
          x = x + constant_op.constant(1)
        else:
          x = x + constant_op.constant(2)
      return x

    @polymorphic_function.function
    def brancher_true():
      left = constant_op.constant(True)
      x = constant_op.constant(1)
      for _ in range(1000):
        if left:
          x = x + constant_op.constant(1)
        else:
          x = x + constant_op.constant(2)
      return x

    x = constant_op.constant(True)
    self.assertEqual(brancher(x), brancher_true())  # Trace each function once.

    benchmark = min(timeit.repeat(lambda: brancher(x), repeat=5, number=100))
    opt_benchmark = min(timeit.repeat(brancher_true, repeat=5, number=100))

    # Constant folded execution is usually 15 - 20 times faster. Here we check
    # for a 3x speedup to account for various machines the test might run on.
    self.assertLess(opt_benchmark * 3, benchmark)

  @test_util.run_v2_only
  def test_small_constants_optimization_with_grappler(self):
    def func(inp):
      x = constant_op.constant(1)
      for _ in range(1000):
        if inp:
          x = x + constant_op.constant(1)
        else:
          x = x + constant_op.constant(2)
      return x

    brancher = polymorphic_function.function(func)
    brancher_opt = polymorphic_function.function(
        func, experimental_attributes={'runtime_constant_optimization': True}
    )

    # Trace each function once.
    with ops.device_v2('CPU'):
      x = constant_op.constant(True)
    self.assertEqual(brancher(x), brancher_opt(x))

    benchmark = min(timeit.repeat(lambda: brancher(x), repeat=5, number=100))
    opt_benchmark = min(
        timeit.repeat(lambda: brancher_opt(x), repeat=5, number=100)
    )

    # Constant folded execution is usually 15 - 20 times faster. Here we check
    # for a 2x speedup to account for various machines the test might run on.
    # Specially the kokoro machines seems to run much slower.
    self.assertLess(opt_benchmark * 2, benchmark)

  @test_util.run_v2_only
  @test_util.run_gpu_only
  def test_small_constants_optimization_disabled(self):
    @polymorphic_function.function(
        experimental_attributes={'runtime_constant_optimization': True}
    )
    def func(inp):
      return inp

    x = constant_op.constant(True)
    with self.assertRaisesRegex(
        errors.InvalidArgumentError,
        (
            'Expecting boolean tensor to be on host when'
            ' small_constants_optimizer is enabled.'
        ),
    ):
      func(x)

  @test_util.run_v2_only
  def test_small_constants_optimization_invalid_input(self):
    @polymorphic_function.function(
        experimental_attributes={'runtime_constant_optimization': True}
    )
    def func(inp):
      return inp

    with ops.device_v2('CPU'):
      x = constant_op.constant([True, True])
    # runtime_constant_optimization should not crash when the tf.function
    # is passed in a boolean tensor having > 1 element.
    self.assertAllEqual(func(x), x)

  @test_util.run_v2_only
  def test_small_constants_optimization_without_grappler(self):
    def func(inp):
      x = constant_op.constant(1)
      for _ in range(1000):
        if inp:
          x = x + constant_op.constant(1)
        else:
          x = x + constant_op.constant(2)
      return x

    brancher = polymorphic_function.function(func)
    brancher_opt = polymorphic_function.function(
        func, experimental_attributes={'runtime_constant_optimization': True}
    )

    # Trace each function once.
    with ops.device_v2('CPU'):
      x = constant_op.constant(True)
    self.assertEqual(brancher(x), brancher_opt(x))

    # Disable grappler and check that performance is still good with
    # small_constants_optimizer.
    with options({'disable_meta_optimizer': True}):
      benchmark = min(timeit.repeat(lambda: brancher(x), repeat=5, number=100))
      opt_benchmark = min(
          timeit.repeat(lambda: brancher_opt(x), repeat=5, number=100)
      )

    # Constant folded execution is usually 150x times faster (against a base
    # that has no grappler optimization). Here we check
    # for a 5x speedup to account for various machines the test might run on.
    # Specially the kokoro machines seems to run much slower.
    self.assertLess(opt_benchmark * 5, benchmark)


if __name__ == '__main__':
  ops.enable_eager_execution()
  test.main()

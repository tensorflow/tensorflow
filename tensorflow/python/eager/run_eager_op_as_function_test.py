# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for wrapping an eager op in a call op at runtime."""
import time

from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.eager import benchmarks_test_base
from tensorflow.python.eager import context
from tensorflow.python.eager import test
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.util import tf_inspect


def run_benchmark(func, num_iters, unused_execution_mode):
  # warm up
  func()
  start = time.time()
  for _ in range(num_iters):
    func()
  end = time.time()
  return end - start


CPU = "/device:CPU:0"
GPU = "/device:GPU:0"


# TODO(srbs): Why can't we use absl parameterized here?
class MicroBenchmarks(benchmarks_test_base.MicroBenchmarksBase):

  def __init__(self):
    super().__init__()
    self._m_2_by_2 = random_ops.random_uniform((2, 2))
    self._m_100_by_100 = random_ops.random_uniform((100, 100))
    self._m_1000_by_1000 = random_ops.random_uniform((1000, 1000))

  def _get_benchmark_name(self):
    """Copied from benchmarks_test.py."""
    stack = tf_inspect.stack()
    name = None
    for frame in stack[::-1]:
      f_locals = frame[0].f_locals
      f_self = f_locals.get("self", None)
      if isinstance(f_self, test.Benchmark):
        name = frame[3]  # Get the method name
        # This is a hack to get around the fact that some methods might have a
        # disable_tfrt decorator around them. In that case a function called
        # 'decorated' wraps the real called function underneath and so we
        # peek one deeper into the stack to get the real name.
        if name == "decorated":
          continue
        else:
          break
    if name is None:
      raise ValueError("Unable to determine calling Benchmark function.")
    if context.is_tfrt_enabled():
      name = name + "_tfrt"
    return name

  def _run(self, func, num_iters):
    self.run_report(run_benchmark, func, num_iters)

  def _benchmark_matmul(self, mat, device):
    if device == GPU and not context.num_gpus():
      return
    with context.device(device):
      if device == GPU:
        mat = mat.gpu()
      func = lambda: math_ops.matmul(mat, mat)
      self._run(func, num_iters=1000)

  def benchmark_tf_matmul_2_by_2_CPU(self):
    self._benchmark_matmul(self._m_2_by_2, CPU)

  def benchmark_tf_matmul_2_by_2_GPU(self):
    self._benchmark_matmul(self._m_2_by_2, GPU)

  def benchmark_tf_matmul_100_by_100_CPU(self):
    self._benchmark_matmul(self._m_100_by_100, CPU)

  def benchmark_tf_matmul_100_by_100_GPU(self):
    self._benchmark_matmul(self._m_100_by_100, GPU)

  def benchmark_tf_matmul_1000_by_1000_CPU(self):
    self._benchmark_matmul(self._m_1000_by_1000, CPU)

  def benchmark_tf_matmul_1000_by_1000_GPU(self):
    self._benchmark_matmul(self._m_1000_by_1000, GPU)


class RunEagerOpAsFunctionTest(test.TestCase):

  def setUp(self):
    super().setUp()
    self._m_2_by_2 = random_ops.random_uniform((2, 2))

  def testArrayFill(self):
    array_ops.fill(
        constant_op.constant([2], dtype=dtypes.int64), constant_op.constant(1))

  def testDatasetMap(self):
    # On GPU, this test triggers a different error: InternalError:
    #   {{function_node __wrapped__RangeDataset}} No unary variant device copy
    #   function found ...... [[{{node RangeDataset/_8}}]] [Op:RangeDataset]
    with ops.device("CPU"):
      dataset_ops.Dataset.range(2).map(math_ops.square)

  def testMatmul(self):
    math_ops.matmul(self._m_2_by_2, self._m_2_by_2)

  def testMixedTypeListInputFastPath(self):
    array_ops.identity_n([self._m_2_by_2, self._m_2_by_2])

  def testMixedTypeListInputEagerFallback(self):
    array_ops.identity_n([1, 1])

  def testMixedTypeListInputFastPathDifferentArity(self):
    # This tests that the FunctionDef cache key contains the number of args.
    array_ops.identity_n([self._m_2_by_2, self._m_2_by_2])
    array_ops.identity_n([self._m_2_by_2, self._m_2_by_2, self._m_2_by_2])

  def testMixedTypeListInputEagerFallbackDifferentArity(self):
    array_ops.identity_n([1, 1])
    array_ops.identity_n([1, 1, 1])

  def testSingleTypeListFastPath(self):
    array_ops.concat([self._m_2_by_2, self._m_2_by_2], axis=-1)

  def testSingleTypeListEagerFallback(self):
    array_ops.concat([[1], [2]], axis=-1)

  def testSingleTypeListFastPathDifferentArity(self):
    array_ops.concat([self._m_2_by_2, self._m_2_by_2], axis=-1)
    array_ops.concat([self._m_2_by_2, self._m_2_by_2, self._m_2_by_2], axis=-1)

  def testSingleTypeListEagerFallbackDifferentArity(self):
    array_ops.concat([[1], [2]], axis=-1)
    array_ops.concat([[1], [2], [3]], axis=-1)


if __name__ == "__main__":
  context.enable_run_eager_op_as_function()
  test.main()

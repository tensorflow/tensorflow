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
"""Tests for tf.data placement within tf.functions."""
from absl.testing import parameterized

from tensorflow.python.data.experimental.ops import prefetching_ops
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.eager import def_function
from tensorflow.python.framework import combinations
from tensorflow.python.framework import config
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import cond
from tensorflow.python.ops import gen_dataset_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test


class PlacementTest(test_base.DatasetTestBase, parameterized.TestCase):
  """Tests for tf.data placement within tf.functions.

  Specifically, tf.data dataset tensors cannot be copied between devices. These
  tests verify the ops are placed in a way that avoids this.
  """

  def setUp(self):
    super(PlacementTest, self).setUp()
    # Grappler optimizations can affect whether the placement issues occur,
    # since they may inadvertently rewrite nodes and edges in a way that removes
    # cross-device copies.
    config.set_optimizer_experimental_options({"disable_meta_optimizer": True})

  @combinations.generate(test_base.eager_only_combinations())
  def testWhileWithCapturedDataset(self):
    dataset = dataset_ops.Dataset.range(10)

    @def_function.function
    def f():
      total = constant_op.constant(0, dtypes.int64)
      for _ in math_ops.range(1):
        for elem in dataset:
          total += elem
      return total

    self.assertEqual(f().numpy(), 45)

  @combinations.generate(test_base.eager_only_combinations())
  def testWhile(self):

    @def_function.function
    def f():
      dataset = dataset_ops.Dataset.range(10)
      total = constant_op.constant(0, dtypes.int64)
      for _ in math_ops.range(1):
        for elem in dataset:
          total += elem
      return total

    self.assertEqual(f().numpy(), 45)

  @combinations.generate(test_base.eager_only_combinations())
  def testCondWithPlacement(self):
    # When the cond op is explicitly placed, there shouldn't be cross-device
    # copies.
    @def_function.function
    def f():
      dataset = dataset_ops.Dataset.range(10)

      def fn():
        return dataset.map(lambda x: x+1)

      c = constant_op.constant(2)
      with ops.device("/cpu:0"):
        a = cond.cond(math_ops.equal(c, 2), fn, fn)
        iterator = iter(a)
        nxt = next(iterator)
      return nxt

    self.assertEqual(f().numpy(), 1)

  @combinations.generate(test_base.eager_only_combinations())
  def testCondWithColocation(self):
    # When the cond op is colocated with the dataset, there shouldn't be
    # cross-device copies.
    @def_function.function
    def f():
      dataset = dataset_ops.Dataset.range(8)

      def fn():
        return dataset.map(lambda x: x+1)

      c = constant_op.constant(2)
      with ops.colocate_with(dataset._variant_tensor):  # pylint:disable=protected-access
        a = cond.cond(math_ops.equal(c, 2), fn, fn)
        iterator = iter(a)
        nxt = next(iterator)
      return nxt

    self.assertEqual(f().numpy(), 1)

  @combinations.generate(test_base.eager_only_combinations())
  def testCond(self):

    @def_function.function
    def f():
      dataset = dataset_ops.Dataset.range(8)
      c = constant_op.constant(2)
      a = cond.cond(
          math_ops.equal(c, 2),
          lambda: dataset.map(lambda x: x + 1),
          lambda: dataset.map(lambda x: x + 2),
      )
      return next(iter(a))

    self.assertEqual(f().numpy(), 1)

  @combinations.generate(test_base.eager_only_combinations())
  def testId(self):
    # Ideally, placer should know that Identity(dataset) should be on the same
    # device as the dataset.
    @def_function.function
    def f():
      dataset = dataset_ops.Dataset.range(10)
      dataset = array_ops.identity(dataset)
      return dataset
    f()

  @combinations.generate(test_base.default_test_combinations())
  @test_util.run_gpu_only
  def testFunctionCall(self):
    # Ideally, placer should know that Call(dataset) should be on the same
    # device as the dataset. Create a function that could be place don the GPU,
    # but a Dataset that cannot.
    @def_function.function
    def test_call(dataset):
      return dataset.reduce(0, lambda s, _: s + 1)

    @def_function.function
    def f():
      dataset = dataset_ops.Dataset.range(10)
      return test_call(dataset)

    self.assertEqual(self.evaluate(f()), 10)

  @combinations.generate(test_base.eager_only_combinations())
  @test_util.run_gpu_only
  def testIteratorOnDeviceEagerMode(self):
    dataset = dataset_ops.Dataset.range(10)
    dataset = dataset.apply(prefetching_ops.prefetch_to_device("/gpu:0"))
    iterator = iter(dataset)
    data = next(iterator)
    optional_data = iterator.get_next_as_optional()

    self.assertIn("gpu:0", dataset._variant_tensor.device.lower())
    self.assertIn("gpu:0", iterator._iterator_resource.device.lower())
    self.assertIn("gpu:0", data.device.lower())
    self.assertIn("gpu:0", optional_data.get_value().device.lower())
    self.assertIn("gpu:0", optional_data.has_value().device.lower())

  # There are HostMemory constraints on AnonymousIteratorV2 and
  # DeleteIterator kernels on TPU but not on GPU. This is intentional because
  # when running AnonymousIteratorV2 in a function
  #
  # - If the op is placed on GPU, the variant _Retval is placed on GPU.
  # - However, if the op is placed on TPU, the variant _Retval is placed on
  #   CPU.
  #
  # So if were to add HostMemory constraints to the GPU kernels it would lead
  # to variant device copy errors.
  #
  # TODO(b/204231062): Unify behavior across GPU and TPU.
  @combinations.generate(test_base.eager_only_combinations())
  @test_util.run_gpu_only
  def testCreateIteratorInFuncOnGpu(self):

    @def_function.function
    def create_iter():
      return gen_dataset_ops.anonymous_iterator_v2(
          output_types=[dtypes.float32], output_shapes=[[]])

    create_iter()

  @combinations.generate(test_base.graph_only_combinations())
  @test_util.run_gpu_only
  def testIteratorOnDeviceGraphModeOneShotIterator(self):
    self.skipTest("TODO(b/169429285): tf.data.Dataset.make_one_shot_iterator "
                  "does not support GPU placement.")

    dataset = dataset_ops.Dataset.range(10)
    dataset = dataset.apply(prefetching_ops.prefetch_to_device("/gpu:0"))
    iterator = dataset_ops.make_one_shot_iterator(dataset)
    data = iterator.get_next()
    optional_data = iterator.get_next_as_optional()

    with ops.colocate_with(dataset._variant_tensor):
      dataset_device = test_ops.device_placement_op()
    self.assertIn(b"GPU:0", self.evaluate(dataset_device))

    with ops.colocate_with(iterator._iterator_resource):
      iterator_device = test_ops.device_placement_op()
    self.assertIn(b"GPU:0", self.evaluate(iterator_device))

    with ops.colocate_with(data):
      data_device = test_ops.device_placement_op()
    self.assertIn(b"GPU:0", self.evaluate(data_device))

    with ops.colocate_with(optional_data.get_value()):
      get_value_device = test_ops.device_placement_op()
    self.assertIn(b"GPU:0", self.evaluate(get_value_device))

    with ops.colocate_with(optional_data.has_value()):
      has_value_device = test_ops.device_placement_op()
    self.assertIn(b"GPU:0", self.evaluate(has_value_device))

  @combinations.generate(test_base.graph_only_combinations())
  @test_util.run_gpu_only
  def testIteratorOnDeviceGraphModeInitializableIterator(self):
    dataset = dataset_ops.Dataset.range(10)
    dataset = dataset.apply(prefetching_ops.prefetch_to_device("/gpu:0"))
    iterator = dataset_ops.make_initializable_iterator(dataset)
    data = iterator.get_next()
    optional_data = iterator.get_next_as_optional()

    with ops.colocate_with(dataset._variant_tensor):
      dataset_device = test_ops.device_placement_op()
    self.assertIn(b"GPU:0", self.evaluate(dataset_device))

    with ops.colocate_with(iterator._iterator_resource):
      iterator_device = test_ops.device_placement_op()
    self.assertIn(b"GPU:0", self.evaluate(iterator_device))

    with ops.colocate_with(data):
      data_device = test_ops.device_placement_op()
    self.assertIn(b"GPU:0", self.evaluate(data_device))

    with ops.colocate_with(optional_data.get_value()):
      get_value_device = test_ops.device_placement_op()
    self.assertIn(b"GPU:0", self.evaluate(get_value_device))

    with ops.colocate_with(optional_data.has_value()):
      has_value_device = test_ops.device_placement_op()
    self.assertIn(b"GPU:0", self.evaluate(has_value_device))

  @combinations.generate(test_base.eager_only_combinations())
  @test_util.run_gpu_only
  def testIterDatasetEagerModeWithExplicitDevice(self):

    @def_function.function
    def comp():
      value = constant_op.constant(0, dtype=dtypes.int64)
      for d in iter(dataset_ops.Dataset.range(10)):
        value += d
      return value

    with ops.device("/gpu:0"):
      result = comp()
    self.assertEqual(result.numpy(), 45)

  @combinations.generate(test_base.eager_only_combinations())
  @test_util.run_gpu_only
  def testFunctionInliningColocation(self):

    @def_function.function
    def f(ds):
      return next(iter(ds))

    @def_function.function
    def g():
      dataset = dataset_ops.Dataset.range(10)
      return f(dataset)

    with ops.device("/gpu:0"):
      self.assertEqual(self.evaluate(g()), 0)


if __name__ == "__main__":
  test.main()

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
"""Tests for the `MapVectorization` optimization."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools

from absl.testing import parameterized
import numpy as np

from tensorflow.core.example import example_pb2
from tensorflow.core.example import feature_pb2
from tensorflow.python.data.experimental.ops import batching
from tensorflow.python.data.experimental.ops import testing
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import combinations
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import bitwise_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import parsing_ops
from tensorflow.python.platform import test


def _generate_test_combinations(cases):

  def reduce_fn(x, y):
    name, fn = y
    return x + combinations.combine(map_fn=combinations.NamedObject(name, fn))

  return functools.reduce(reduce_fn, cases, [])


def _unary_bitwise_test_combinations():
  cases = [("Invert", bitwise_ops.invert)]
  return _generate_test_combinations(cases)


def _unary_logical_test_combinations():
  cases = [("LogicalNot", math_ops.logical_not)]
  return _generate_test_combinations(cases)


def _unary_complex_test_combinations():
  cases = [
      ("Angle", math_ops.angle),
      ("ComplexAbs", math_ops.abs),
      ("Conj", math_ops.conj),
      ("Imag", math_ops.imag),
      ("Real", math_ops.real),
  ]
  return _generate_test_combinations(cases)


def _unary_real_test_combinations():
  # acosh requires values x >= 1
  def safe_acosh(x):
    return math_ops.acosh(1 + math_ops.square(x))

  cases = [
      ("Abs", math_ops.abs),
      ("Acos", math_ops.acos),
      ("Acosh", safe_acosh),
      ("Asin", math_ops.asin),
      ("Asinh", math_ops.asinh),
      ("Atan", math_ops.atan),
      ("Atanh", math_ops.atanh),
      ("BesselI0e", math_ops.bessel_i0e),
      ("BesselI1e", math_ops.bessel_i1e),
      ("Ceil", math_ops.ceil),
      ("Cos", math_ops.cos),
      ("Cosh", math_ops.cosh),
      ("Digamma", math_ops.digamma),
      ("Elu", nn.elu),
      ("Erf", math_ops.erf),
      ("Erfc", math_ops.erfc),
      ("Exp", math_ops.exp),
      ("Expm1", math_ops.expm1),
      ("Floor", math_ops.floor),
      ("Inv", math_ops.inv),
      ("IsFinite", math_ops.is_finite),
      ("IsInf", math_ops.is_inf),
      ("Lgamma", math_ops.lgamma),
      ("Log", math_ops.log),
      ("Log1p", math_ops.log1p),
      ("Neg", math_ops.negative),
      ("Reciprocal", math_ops.reciprocal),
      ("Relu", nn.relu),
      ("Relu6", nn.relu6),
      ("Rint", math_ops.rint),
      ("Round", math_ops.round),
      ("Rsqrt", math_ops.rsqrt),
      ("Selu", nn.selu),
      ("Sigmoid", math_ops.sigmoid),
      ("Sign", math_ops.sign),
      ("Sin", math_ops.sin),
      ("Sinh", math_ops.sinh),
      ("Softplus", nn.softplus),
      ("Softsign", nn.softsign),
      ("Sqrt", math_ops.sqrt),
      ("Square", math_ops.square),
      ("Tan", math_ops.tan),
      ("Tanh", math_ops.tanh),
  ]
  return _generate_test_combinations(cases)


def _binary_bitwise_test_combinations():
  cases = [("BitwiseAnd", bitwise_ops.bitwise_and),
           ("BitwiseOr", bitwise_ops.bitwise_or),
           ("BitwiseXor", bitwise_ops.bitwise_xor),
           ("LeftShift", bitwise_ops.left_shift),
           ("RightShift", bitwise_ops.right_shift)]
  return _generate_test_combinations(cases)


def _binary_logical_test_combinations():
  cases = [("LogicalAnd", math_ops.logical_and),
           ("LogicalOr", math_ops.logical_or)]
  return _generate_test_combinations(cases)


def _binary_real_test_combinations():

  def safe_polygamma(x, y):
    return math_ops.polygamma(
        math_ops.round(clip_ops.clip_by_value(y, 1, 10)), x * x + 1)

  def safe_zeta(x, y):
    return math_ops.zeta(x * x + 1, y * y)

  cases = [
      ("Add", math_ops.add),
      ("AddV2", math_ops.add_v2),
      ("Atan2", math_ops.atan2),
      ("Complex", math_ops.complex),
      ("DivNoNan", math_ops.div_no_nan),
      ("Equal", math_ops.equal),
      ("FloorDiv", math_ops.floor_div),
      ("FloorMod", math_ops.floor_mod),
      ("Greater", math_ops.greater),
      ("GreaterEqual", math_ops.greater_equal),
      ("Igamma", math_ops.igamma),
      ("Igammac", math_ops.igammac),
      ("IgammaGradA", math_ops.igamma_grad_a),
      ("Less", math_ops.less),
      ("LessEqual", math_ops.less_equal),
      ("Maximum", math_ops.maximum),
      ("Minimum", math_ops.minimum),
      ("Mod", math_ops.mod),
      ("Mul", math_ops.multiply),
      ("NotEqual", math_ops.not_equal),
      ("Polygamma", safe_polygamma),
      ("Pow", math_ops.pow),
      ("RealDiv", math_ops.divide),
      ("SquareDifference", math_ops.squared_difference),
      ("Sub", math_ops.subtract),
      ("TruncateMod", math_ops.truncate_mod),
      ("Zeta", safe_zeta),
  ]
  return _generate_test_combinations(cases)


# TODO(rachelim): Consolidate tests with pfor when APIs are somewhat shared.
class MapVectorizationTest(test_base.DatasetTestBase, parameterized.TestCase):

  def _enable_map_vectorization(self, dataset, use_choose=True):
    options = dataset_ops.Options()
    opt_options = options.experimental_optimization
    opt_options.map_vectorization.enabled = True
    opt_options.map_vectorization.use_choose_fastest = use_choose
    return dataset.with_options(options)

  def _get_test_datasets(self,
                         base_dataset,
                         map_fn,
                         num_parallel_calls=None,
                         expect_optimized=True):
    """Given base dataset and map fn, creates test datasets.

    Returns a tuple of (unoptimized dataset, optimized dataset). The
    unoptimized dataset has the assertion that Batch follows Map. The optimized
    dataset has the assertion that Map follows Batch, and has the
    "map_vectorization" optimization applied.

    Args:
      base_dataset: Input dataset to map->batch
      map_fn: Map function to use
      num_parallel_calls: (Optional.) num_parallel_calls argument for map
      expect_optimized: (Optional.) Whether we expect the optimization to take
        place, in which case we will assert that Batch is followed by Map,
        otherwise Map followed by Batch. Defaults to True.

    Returns:
      Tuple of (unoptimized dataset, optimized dataset).
    """
    map_node_name = "Map" if num_parallel_calls is None else "ParallelMap"

    def _make_dataset(node_names):
      dataset = base_dataset.apply(testing.assert_next(node_names))
      dataset = dataset.map(map_fn, num_parallel_calls)
      dataset = dataset.batch(100)
      options = dataset_ops.Options()
      options.experimental_optimization.apply_default_optimizations = False
      options.experimental_optimization.map_and_batch_fusion = False
      dataset = dataset.with_options(options)
      return dataset

    unoptimized = _make_dataset([map_node_name, "BatchV2"])
    # Note that because of the `ChooseDataset` fork, we can't use `assert_next`
    # to verify the optimization result.
    optimized = _make_dataset(["ChooseFastestBranch"] if expect_optimized else
                              [map_node_name, "BatchV2"])
    optimized = self._enable_map_vectorization(optimized)
    return unoptimized, optimized

  def _testOptimization(self, map_fn, dataset_factory, num_parallel_calls):
    dataset = dataset_factory()
    unoptimized, optimized = self._get_test_datasets(dataset, map_fn,
                                                     num_parallel_calls)
    self.assertDatasetsEqual(unoptimized, optimized)

  @combinations.generate(
      combinations.times(test_base.default_test_combinations(),
                         combinations.combine(num_parallel_calls=[None, 12])))
  def testBasic(self, num_parallel_calls):
    data = np.random.rand(10, 3)
    dataset_factory = lambda: dataset_ops.Dataset.from_tensors(data).repeat(5)
    map_fn = lambda x: (x, x + 1)
    self._testOptimization(map_fn, dataset_factory, num_parallel_calls)

  @combinations.generate(
      combinations.times(test_base.default_test_combinations(),
                         combinations.combine(num_parallel_calls=[None, 12])))
  def testBroadcast(self, num_parallel_calls):
    data = np.random.rand(10, 3)
    dataset_factory = lambda: dataset_ops.Dataset.from_tensors(data).repeat(5)
    value = np.random.rand(1, 1, 1, 1, 1, 1)
    map_fn = lambda x: x + value
    self._testOptimization(map_fn, dataset_factory, num_parallel_calls)

  @combinations.generate(
      combinations.times(test_base.default_test_combinations(),
                         combinations.combine(num_parallel_calls=[None, 12])))
  def testCast(self, num_parallel_calls):
    data = np.random.rand(10, 3)
    dataset_factory = lambda: dataset_ops.Dataset.from_tensors(data).repeat(5)
    map_fn = lambda x: math_ops.cast(x, dtypes.float64)
    self._testOptimization(map_fn, dataset_factory, num_parallel_calls)

  @combinations.generate(
      combinations.times(test_base.default_test_combinations(),
                         combinations.combine(num_parallel_calls=[None, 12])))
  def testConst(self, num_parallel_calls):
    data = np.random.rand(10, 3)
    dataset_factory = lambda: dataset_ops.Dataset.from_tensors(data).repeat(5)
    map_fn = lambda x: 2
    self._testOptimization(map_fn, dataset_factory, num_parallel_calls)

  @combinations.generate(
      combinations.times(test_base.default_test_combinations(),
                         combinations.combine(num_parallel_calls=[None, 12])))
  def testCycle(self, num_parallel_calls):
    dataset_factory = lambda: dataset_ops.Dataset.from_tensors(1)

    def map_fn(x):
      c = lambda i: math_ops.less(i, 10)
      b = lambda i: math_ops.add(i, 1)
      return control_flow_ops.while_loop(c, b, [x])

    self._testOptimization(map_fn, dataset_factory, num_parallel_calls)

  @combinations.generate(
      combinations.times(test_base.default_test_combinations(),
                         combinations.combine(num_parallel_calls=[None, 12])))
  def testReshape(self, num_parallel_calls):
    data = np.random.rand(10, 3)
    dataset_factory = lambda: dataset_ops.Dataset.from_tensors(data).repeat(5)
    map_fn = lambda x: array_ops.reshape(x, (-1, 30))
    self._testOptimization(map_fn, dataset_factory, num_parallel_calls)

  @combinations.generate(
      combinations.times(test_base.default_test_combinations(),
                         combinations.combine(num_parallel_calls=[None, 12])))
  def testTranspose(self, num_parallel_calls):
    data = np.random.rand(10, 3)
    dataset_factory = lambda: dataset_ops.Dataset.from_tensors(data).repeat(5)
    map_fn = array_ops.transpose
    self._testOptimization(map_fn, dataset_factory, num_parallel_calls)

  @combinations.generate(
      combinations.times(test_base.default_test_combinations(),
                         combinations.combine(num_parallel_calls=[None, 12])))
  def testUnstack(self, num_parallel_calls):
    data = np.random.rand(10, 3)
    dataset_factory = lambda: dataset_ops.Dataset.from_tensors(data).repeat(5)
    map_fns = [array_ops.unstack, lambda x: array_ops.unstack(x, axis=-1)]
    for map_fn in map_fns:
      self._testOptimization(map_fn, dataset_factory, num_parallel_calls)

  @combinations.generate(
      combinations.times(test_base.default_test_combinations(),
                         _unary_bitwise_test_combinations(),
                         combinations.combine(num_parallel_calls=[None, 12])))
  def testUnaryBitwiseOperations(self, map_fn, num_parallel_calls):
    x = np.random.randint(0, 10, (7, 3, 5))
    dataset_factory = lambda: dataset_ops.Dataset.from_tensor_slices(x)
    self._testOptimization(map_fn, dataset_factory, num_parallel_calls)

  @combinations.generate(
      combinations.times(test_base.default_test_combinations(),
                         _unary_logical_test_combinations(),
                         combinations.combine(num_parallel_calls=[None, 12])))
  def testUnaryLogicalOperations(self, map_fn, num_parallel_calls):
    x = np.random.rand(3, 5)
    dataset_factory = lambda: dataset_ops.Dataset.from_tensor_slices(x > 0)
    self._testOptimization(map_fn, dataset_factory, num_parallel_calls)

  @combinations.generate(
      combinations.times(test_base.default_test_combinations(),
                         _unary_complex_test_combinations(),
                         combinations.combine(num_parallel_calls=[None, 12])))
  def testUnaryComplexOperations(self, map_fn, num_parallel_calls):
    x = math_ops.complex(np.random.rand(3, 5), np.random.rand(3, 5))
    dataset_factory = lambda: dataset_ops.Dataset.from_tensor_slices(x)
    self._testOptimization(map_fn, dataset_factory, num_parallel_calls)

  @combinations.generate(
      combinations.times(test_base.default_test_combinations(),
                         _unary_real_test_combinations(),
                         combinations.combine(num_parallel_calls=[None, 12])))
  def testUnaryRealOperations(self, map_fn, num_parallel_calls):
    x = np.random.rand(3, 5)
    dataset_factory = lambda: dataset_ops.Dataset.from_tensor_slices(x)
    self._testOptimization(map_fn, dataset_factory, num_parallel_calls)

  @combinations.generate(
      combinations.times(test_base.default_test_combinations(),
                         _binary_bitwise_test_combinations(),
                         combinations.combine(num_parallel_calls=[None, 12])))
  def testBinaryBitwiseOperations(self, map_fn, num_parallel_calls):
    x = np.random.randint(0, 10, (7, 3, 5))
    y = np.random.randint(0, 10, (3, 5))
    dataset_factory = lambda: dataset_ops.Dataset.from_tensors((x, y))
    self._testOptimization(map_fn, dataset_factory, num_parallel_calls)

  @combinations.generate(
      combinations.times(test_base.default_test_combinations(),
                         _binary_logical_test_combinations(),
                         combinations.combine(num_parallel_calls=[None, 12])))
  def testBinaryLogicalOperations(self, map_fn, num_parallel_calls):
    x = np.random.rand(7, 3, 5)
    y = np.random.rand(3, 5)
    dataset_factory = lambda: dataset_ops.Dataset.from_tensors((x > 0, y > 0))
    self._testOptimization(map_fn, dataset_factory, num_parallel_calls)

  @combinations.generate(
      combinations.times(test_base.default_test_combinations(),
                         _binary_real_test_combinations(),
                         combinations.combine(num_parallel_calls=[None, 12])))
  def testBinaryRealOperations(self, map_fn, num_parallel_calls):
    x = np.random.rand(7, 3, 5)
    y = np.random.rand(3, 5)
    dataset_factory = lambda: dataset_ops.Dataset.from_tensors((x, y))
    self._testOptimization(map_fn, dataset_factory, num_parallel_calls)

  @combinations.generate(
      combinations.times(test_base.default_test_combinations(),
                         combinations.combine(num_parallel_calls=[None, 12])))
  def testDecodeCsv(self, num_parallel_calls):

    def dataset_factory():
      return dataset_ops.Dataset.from_tensor_slices(["1.0:2:a",
                                                     "2.4:5:c"]).repeat(5)

    def decode_csv_fn(x):
      return parsing_ops.decode_csv(
          x,
          record_defaults=[
              constant_op.constant([], dtypes.float32),
              constant_op.constant([], dtypes.int32),
              constant_op.constant([], dtypes.string)
          ],
          field_delim=":")

    self._testOptimization(decode_csv_fn, dataset_factory, num_parallel_calls)

  @combinations.generate(
      combinations.times(test_base.default_test_combinations(),
                         combinations.combine(num_parallel_calls=[None, 12])))
  def testParseSingleExample(self, num_parallel_calls):

    def dataset_factory():

      def _int64_feature(*values):
        return feature_pb2.Feature(
            int64_list=feature_pb2.Int64List(value=values))

      def _bytes_feature(*values):
        return feature_pb2.Feature(
            bytes_list=feature_pb2.BytesList(
                value=[v.encode("utf-8") for v in values]))

      # pylint:disable=g-complex-comprehension
      return dataset_ops.Dataset.from_tensor_slices(
          constant_op.constant([
              example_pb2.Example(
                  features=feature_pb2.Features(
                      feature={
                          "dense_int": _int64_feature(i),
                          "dense_str": _bytes_feature(str(i)),
                      })).SerializeToString() for i in range(10)
          ]))

    def parse_fn(x):
      features = {
          "dense_int": parsing_ops.FixedLenFeature((), dtypes.int64, 0),
          "dense_str": parsing_ops.FixedLenFeature((), dtypes.string, ""),
      }
      return parsing_ops.parse_single_example(x, features)

    def dense_only_parse_fn(x):
      return [
          y for y in parse_fn(x)
          if not isinstance(y, sparse_tensor.SparseTensor)
      ]

    map_fns = [parse_fn, dense_only_parse_fn]

    for map_fn in map_fns:
      self._testOptimization(map_fn, dataset_factory, num_parallel_calls)

  @combinations.generate(test_base.default_test_combinations())
  def testOptimizationBadMapFn(self):
    # Test map functions that give an error
    def map_fn(x):
      # x has leading dimension 5, this will raise an error
      return array_ops.gather(x, 10)

    with self.assertRaisesRegexp(errors.InvalidArgumentError,
                                 r"indices = 10 is not in \[0, 5\)"):
      base_dataset = dataset_ops.Dataset.range(5).repeat(5).batch(
          5, drop_remainder=True)
      _, optimized = self._get_test_datasets(base_dataset, map_fn)
      nxt = dataset_ops.make_one_shot_iterator(optimized).get_next()
      self.evaluate(nxt)

  @combinations.generate(test_base.default_test_combinations())
  def testOptimizationWithCapturedInputs(self):
    # Tests that vectorization works with captured inputs.
    y = constant_op.constant(1, shape=(2,))
    z = constant_op.constant(2, shape=(2,))

    def map_fn(x):
      return x, y, z

    base_dataset = dataset_ops.Dataset.from_tensor_slices([[1, 2],
                                                           [3, 4]]).repeat(5)
    unoptimized, optimized = self._get_test_datasets(
        base_dataset, map_fn, expect_optimized=True)
    self.assertDatasetsEqual(optimized, unoptimized)

  @combinations.generate(test_base.default_test_combinations())
  def testOptimizationWithMapAndBatchFusion(self):
    # Tests that vectorization works on fused map and batch.
    def map_fn(x):
      return x**2

    base_dataset = dataset_ops.Dataset.range(1000)
    options = dataset_ops.Options()
    options.experimental_optimization.apply_default_optimizations = False
    base_dataset = base_dataset.with_options(options)

    def _make_dataset(node_names):
      dataset = base_dataset.apply(testing.assert_next(node_names))
      dataset = dataset.apply(batching.map_and_batch(map_fn, 100))
      return dataset

    unoptimized = _make_dataset(["MapAndBatch"])
    optimized = _make_dataset(["ChooseFastestBranch"])
    optimized = self._enable_map_vectorization(optimized)
    self.assertDatasetsEqual(optimized, unoptimized)

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          combinations.combine(
              fuse_first=[True, False], fuse_second=[True, False])))
  def testOptimizationWithChainedMapAndBatch(self, fuse_first, fuse_second):
    # Tests that vectorization works on chained map and batch functions.
    def map_fn(x):
      return x * 2

    unoptimized_seq = []

    def make_apply_fn(is_fused):
      if is_fused:
        unoptimized_seq.append("MapAndBatch")

        def apply_fn(dataset):
          return dataset.apply(
              batching.map_and_batch(map_fn, 2, 12, drop_remainder=True))

        return apply_fn
      else:
        unoptimized_seq.extend(["ParallelMap", "BatchV2"])

        def apply_fn(dataset):
          return dataset.map(map_fn, 12).batch(2, drop_remainder=True)

        return apply_fn

    base_dataset = dataset_ops.Dataset.range(1000)
    options = dataset_ops.Options()
    options.experimental_optimization.apply_default_optimizations = False
    base_dataset = base_dataset.with_options(options)

    apply_fn_1 = make_apply_fn(fuse_first)
    apply_fn_2 = make_apply_fn(fuse_second)

    def make_dataset(node_names):
      dataset = base_dataset.apply(testing.assert_next(node_names))
      dataset = apply_fn_1(dataset)
      dataset = apply_fn_2(dataset)
      return dataset

    unoptimized = make_dataset(unoptimized_seq)
    optimized = make_dataset(["ChooseFastestBranch", "ChooseFastestBranch"])
    optimized = self._enable_map_vectorization(optimized)
    self.assertDatasetsEqual(optimized, unoptimized)

  @combinations.generate(test_base.default_test_combinations())
  def testOptimizationIgnoreStateful(self):

    def map_fn(x):
      with ops.control_dependencies([check_ops.assert_equal(x, np.int64(0))]):
        return array_ops.identity(x)

    dataset = dataset_ops.Dataset.range(10)
    dataset = dataset.map(map_fn)
    dataset = dataset.batch(10)
    dataset = self._enable_map_vectorization(dataset, use_choose=False)
    with self.assertRaises(errors.InvalidArgumentError):
      get_next = self.getNext(dataset)
      self.evaluate(get_next())

  @combinations.generate(test_base.default_test_combinations())
  def testOptimizationIgnoreRagged(self):
    # Make sure we ignore inputs that might not be uniformly sized
    def map_fn(x):
      return array_ops.gather(x, np.int64(0))

    # output_shape = (?,)
    base_dataset = dataset_ops.Dataset.range(20).batch(3, drop_remainder=False)
    unoptimized, optimized = self._get_test_datasets(
        base_dataset, map_fn, expect_optimized=False)
    self.assertDatasetsEqual(unoptimized, optimized)

  @combinations.generate(test_base.default_test_combinations())
  def testOptimizationIgnoreRaggedMap(self):
    # Don't optimize when the output of the map fn shapes are unknown.
    def map_fn(x):
      return array_ops.tile(x, x)

    dataset = dataset_ops.Dataset.range(10).batch(1)
    dataset = dataset.map(map_fn)
    dataset = dataset.batch(10)
    dataset = self._enable_map_vectorization(dataset, use_choose=False)
    with self.assertRaises(errors.InvalidArgumentError):
      get_next = self.getNext(dataset)
      self.evaluate(get_next())

  @combinations.generate(test_base.default_test_combinations())
  def testOptimizationWithUnknownBatchShape(self):
    tensor = sparse_tensor.SparseTensor(
        indices=[[0, 0], [1, 2]], values=[1, 2], dense_shape=[3, 4])

    # Datasets with sparse tensors have unknown output shapes.
    base_dataset = dataset_ops.Dataset.from_tensors(tensor)
    unoptimized = base_dataset.apply(batching.map_and_batch(lambda x: x, 2))
    options = dataset_ops.Options()
    options.experimental_optimization.apply_default_optimizations = False
    unoptimized = unoptimized.with_options(options)

    optimized = self._enable_map_vectorization(unoptimized)
    self.assertDatasetsEqual(unoptimized, optimized)

  @combinations.generate(test_base.default_test_combinations())
  def testOptimizationWithSparseTensor(self):
    base_dataset = dataset_ops.Dataset.from_tensors(0)

    def map_fn(x):
      del x
      return sparse_tensor.SparseTensor(
          indices=[[0, 0], [1, 2]], values=[1, 2], dense_shape=[3, 4])

    # Datasets with sparse tensors have unknown output shapes.
    unoptimized = base_dataset.apply(batching.map_and_batch(map_fn, 2))
    options = dataset_ops.Options()
    options.experimental_optimization.apply_default_optimizations = False
    unoptimized = unoptimized.with_options(options)
    optimized = self._enable_map_vectorization(unoptimized)
    self.assertDatasetsEqual(unoptimized, optimized)

  @combinations.generate(test_base.default_test_combinations())
  def testOptimizationWithPrefetch(self):
    dataset = dataset_ops.Dataset.range(10)
    dataset = dataset.map(lambda x: x)
    dataset = dataset.prefetch(1)
    dataset = dataset.batch(10)
    dataset = self._enable_map_vectorization(dataset)
    self.assertDatasetProduces(dataset, [list(range(10))])

  @combinations.generate(test_base.default_test_combinations())
  def testOptimizationWithoutChooseFastest(self):
    dataset = dataset_ops.Dataset.range(10)
    dataset = dataset.map(lambda x: x**2)
    dataset = dataset.batch(10)
    dataset = self._enable_map_vectorization(dataset, use_choose=False)
    self.assertDatasetProduces(dataset, [[x**2 for x in range(10)]])


if __name__ == "__main__":
  test.main()

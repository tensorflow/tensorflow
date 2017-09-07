# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for layers.feature_column."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import itertools
import os
import tempfile

import numpy as np

from tensorflow.contrib.layers.python.layers import feature_column as fc
from tensorflow.contrib.layers.python.layers import feature_column_ops
from tensorflow.python.feature_column import feature_column as fc_core
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import sparse_tensor as sparse_tensor_lib
from tensorflow.python.ops import parsing_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.training import saver


def _sparse_id_tensor(shape, vocab_size, seed=112123):
  # Returns a arbitrary `SparseTensor` with given shape and vocab size.
  np.random.seed(seed)
  indices = np.array(list(itertools.product(*[range(s) for s in shape])))

  # In order to create some sparsity, we include a value outside the vocab.
  values = np.random.randint(0, vocab_size + 1, size=np.prod(shape))

  # Remove entries outside the vocabulary.
  keep = values < vocab_size
  indices = indices[keep]
  values = values[keep]

  return sparse_tensor_lib.SparseTensor(
      indices=indices, values=values, dense_shape=shape)


class FeatureColumnTest(test.TestCase):

  def testImmutability(self):
    a = fc.sparse_column_with_hash_bucket("aaa", hash_bucket_size=100)
    with self.assertRaises(AttributeError):
      a.column_name = "bbb"

  def testSparseColumnWithHashBucket(self):
    a = fc.sparse_column_with_hash_bucket("aaa", hash_bucket_size=100)
    self.assertEqual(a.name, "aaa")
    self.assertEqual(a.dtype, dtypes.string)

    a = fc.sparse_column_with_hash_bucket(
        "aaa", hash_bucket_size=100, dtype=dtypes.int64)
    self.assertEqual(a.name, "aaa")
    self.assertEqual(a.dtype, dtypes.int64)

    with self.assertRaisesRegexp(ValueError, "dtype must be string or integer"):
      a = fc.sparse_column_with_hash_bucket(
          "aaa", hash_bucket_size=100, dtype=dtypes.float32)

  def testSparseColumnWithVocabularyFile(self):
    b = fc.sparse_column_with_vocabulary_file(
        "bbb", vocabulary_file="a_file", vocab_size=454)
    self.assertEqual(b.dtype, dtypes.string)
    self.assertEqual(b.lookup_config.vocab_size, 454)
    self.assertEqual(b.lookup_config.vocabulary_file, "a_file")

    with self.assertRaises(ValueError):
      # Vocabulary size should be defined if vocabulary_file is used.
      fc.sparse_column_with_vocabulary_file("bbb", vocabulary_file="somefile")

    b = fc.sparse_column_with_vocabulary_file(
        "bbb", vocabulary_file="a_file", vocab_size=454, dtype=dtypes.int64)
    self.assertEqual(b.dtype, dtypes.int64)

    with self.assertRaisesRegexp(ValueError, "dtype must be string or integer"):
      b = fc.sparse_column_with_vocabulary_file(
          "bbb", vocabulary_file="a_file", vocab_size=454, dtype=dtypes.float32)

  def testWeightedSparseColumn(self):
    ids = fc.sparse_column_with_keys("ids", ["marlo", "omar", "stringer"])
    weighted_ids = fc.weighted_sparse_column(ids, "weights")
    self.assertEqual(weighted_ids.name, "ids_weighted_by_weights")

  def testWeightedSparseColumnDeepCopy(self):
    ids = fc.sparse_column_with_keys("ids", ["marlo", "omar", "stringer"])
    weighted = fc.weighted_sparse_column(ids, "weights")
    weighted_copy = copy.deepcopy(weighted)
    self.assertEqual(weighted_copy.sparse_id_column.name, "ids")
    self.assertEqual(weighted_copy.weight_column_name, "weights")
    self.assertEqual(weighted_copy.name, "ids_weighted_by_weights")

  def testEmbeddingColumn(self):
    a = fc.sparse_column_with_hash_bucket(
        "aaa", hash_bucket_size=100, combiner="sum")
    b = fc.embedding_column(a, dimension=4, combiner="mean")
    self.assertEqual(b.sparse_id_column.name, "aaa")
    self.assertEqual(b.dimension, 4)
    self.assertEqual(b.combiner, "mean")

  def testEmbeddingColumnDeepCopy(self):
    a = fc.sparse_column_with_hash_bucket(
        "aaa", hash_bucket_size=100, combiner="sum")
    column = fc.embedding_column(a, dimension=4, combiner="mean")
    column_copy = copy.deepcopy(column)
    self.assertEqual(column_copy.name, "aaa_embedding")
    self.assertEqual(column_copy.sparse_id_column.name, "aaa")
    self.assertEqual(column_copy.dimension, 4)
    self.assertEqual(column_copy.combiner, "mean")

  def testScatteredEmbeddingColumn(self):
    column = fc.scattered_embedding_column(
        "aaa", size=100, dimension=10, hash_key=1)
    self.assertEqual(column.column_name, "aaa")
    self.assertEqual(column.size, 100)
    self.assertEqual(column.dimension, 10)
    self.assertEqual(column.hash_key, 1)
    self.assertEqual(column.name, "aaa_scattered_embedding")

  def testScatteredEmbeddingColumnDeepCopy(self):
    column = fc.scattered_embedding_column(
        "aaa", size=100, dimension=10, hash_key=1)
    column_copy = copy.deepcopy(column)
    self.assertEqual(column_copy.column_name, "aaa")
    self.assertEqual(column_copy.size, 100)
    self.assertEqual(column_copy.dimension, 10)
    self.assertEqual(column_copy.hash_key, 1)
    self.assertEqual(column_copy.name, "aaa_scattered_embedding")

  def testSharedEmbeddingColumn(self):
    a1 = fc.sparse_column_with_keys("a1", ["marlo", "omar", "stringer"])
    a2 = fc.sparse_column_with_keys("a2", ["marlo", "omar", "stringer"])
    b = fc.shared_embedding_columns([a1, a2], dimension=4, combiner="mean")
    self.assertEqual(len(b), 2)
    self.assertEqual(b[0].shared_embedding_name, "a1_a2_shared_embedding")
    self.assertEqual(b[1].shared_embedding_name, "a1_a2_shared_embedding")

    # Create a sparse id tensor for a1.
    input_tensor_c1 = sparse_tensor_lib.SparseTensor(
        indices=[[0, 0], [1, 1], [2, 2]], values=[0, 1, 2], dense_shape=[3, 3])
    # Create a sparse id tensor for a2.
    input_tensor_c2 = sparse_tensor_lib.SparseTensor(
        indices=[[0, 0], [1, 1], [2, 2]], values=[0, 1, 2], dense_shape=[3, 3])
    with variable_scope.variable_scope("run_1"):
      b1 = feature_column_ops.input_from_feature_columns({
          b[0]: input_tensor_c1
      }, [b[0]])
      b2 = feature_column_ops.input_from_feature_columns({
          b[1]: input_tensor_c2
      }, [b[1]])
    with self.test_session() as sess:
      sess.run(variables.global_variables_initializer())
      b1_value = b1.eval()
      b2_value = b2.eval()
    for i in range(len(b1_value)):
      self.assertAllClose(b1_value[i], b2_value[i])

    # Test the case when a shared_embedding_name is explicitly specified.
    d = fc.shared_embedding_columns(
        [a1, a2],
        dimension=4,
        combiner="mean",
        shared_embedding_name="my_shared_embedding")
    # a3 is a completely different sparse column with a1 and a2, but since the
    # same shared_embedding_name is passed in, a3 will have the same embedding
    # as a1 and a2
    a3 = fc.sparse_column_with_keys("a3", [42, 1, -1000], dtype=dtypes.int32)
    e = fc.shared_embedding_columns(
        [a3],
        dimension=4,
        combiner="mean",
        shared_embedding_name="my_shared_embedding")
    with variable_scope.variable_scope("run_2"):
      d1 = feature_column_ops.input_from_feature_columns({
          d[0]: input_tensor_c1
      }, [d[0]])
      e1 = feature_column_ops.input_from_feature_columns({
          e[0]: input_tensor_c1
      }, [e[0]])
    with self.test_session() as sess:
      sess.run(variables.global_variables_initializer())
      d1_value = d1.eval()
      e1_value = e1.eval()
    for i in range(len(d1_value)):
      self.assertAllClose(d1_value[i], e1_value[i])

  def testSharedEmbeddingColumnWithWeightedSparseColumn(self):
    # Tests creation of shared embeddings containing weighted sparse columns.
    sparse_col = fc.sparse_column_with_keys("a1", ["marlo", "omar", "stringer"])
    ids = fc.sparse_column_with_keys("ids", ["marlo", "omar", "stringer"])
    weighted_sparse_col = fc.weighted_sparse_column(ids, "weights")
    self.assertEqual(weighted_sparse_col.name, "ids_weighted_by_weights")

    b = fc.shared_embedding_columns([sparse_col, weighted_sparse_col],
                                    dimension=4, combiner="mean")
    self.assertEqual(len(b), 2)
    self.assertEqual(b[0].shared_embedding_name,
                     "a1_ids_weighted_by_weights_shared_embedding")
    self.assertEqual(b[1].shared_embedding_name,
                     "a1_ids_weighted_by_weights_shared_embedding")

    # Tries reversing order to check compatibility condition.
    b = fc.shared_embedding_columns([weighted_sparse_col, sparse_col],
                                    dimension=4, combiner="mean")
    self.assertEqual(len(b), 2)
    self.assertEqual(b[0].shared_embedding_name,
                     "a1_ids_weighted_by_weights_shared_embedding")
    self.assertEqual(b[1].shared_embedding_name,
                     "a1_ids_weighted_by_weights_shared_embedding")

    # Tries adding two weighted columns to check compatibility between them.
    weighted_sparse_col_2 = fc.weighted_sparse_column(ids, "weights_2")
    b = fc.shared_embedding_columns([weighted_sparse_col,
                                     weighted_sparse_col_2],
                                    dimension=4, combiner="mean")
    self.assertEqual(len(b), 2)
    self.assertEqual(
        b[0].shared_embedding_name,
        "ids_weighted_by_weights_ids_weighted_by_weights_2_shared_embedding"
    )
    self.assertEqual(
        b[1].shared_embedding_name,
        "ids_weighted_by_weights_ids_weighted_by_weights_2_shared_embedding"
    )

  def testSharedEmbeddingColumnDeterminism(self):
    # Tests determinism in auto-generated shared_embedding_name.
    sparse_id_columns = tuple([
        fc.sparse_column_with_keys(k, ["foo", "bar"])
        for k in ["07", "02", "00", "03", "05", "01", "09", "06", "04", "08"]
    ])
    output = fc.shared_embedding_columns(
        sparse_id_columns, dimension=2, combiner="mean")
    self.assertEqual(len(output), 10)
    for x in output:
      self.assertEqual(x.shared_embedding_name,
                       "00_01_02_plus_7_others_shared_embedding")

  def testSharedEmbeddingColumnErrors(self):
    # Tries passing in a string.
    with self.assertRaises(TypeError):
      invalid_string = "Invalid string."
      fc.shared_embedding_columns(invalid_string, dimension=2, combiner="mean")

    # Tries passing in a set of sparse columns.
    with self.assertRaises(TypeError):
      invalid_set = set([
          fc.sparse_column_with_keys("a", ["foo", "bar"]),
          fc.sparse_column_with_keys("b", ["foo", "bar"]),
      ])
      fc.shared_embedding_columns(invalid_set, dimension=2, combiner="mean")

  def testSharedEmbeddingColumnDeepCopy(self):
    a1 = fc.sparse_column_with_keys("a1", ["marlo", "omar", "stringer"])
    a2 = fc.sparse_column_with_keys("a2", ["marlo", "omar", "stringer"])
    columns = fc.shared_embedding_columns(
        [a1, a2], dimension=4, combiner="mean")
    columns_copy = copy.deepcopy(columns)
    self.assertEqual(
        columns_copy[0].shared_embedding_name, "a1_a2_shared_embedding")
    self.assertEqual(
        columns_copy[1].shared_embedding_name, "a1_a2_shared_embedding")

  def testOneHotColumn(self):
    a = fc.sparse_column_with_keys("a", ["a", "b", "c", "d"])
    onehot_a = fc.one_hot_column(a)
    self.assertEqual(onehot_a.sparse_id_column.name, "a")
    self.assertEqual(onehot_a.length, 4)

    b = fc.sparse_column_with_hash_bucket(
        "b", hash_bucket_size=100, combiner="sum")
    onehot_b = fc.one_hot_column(b)
    self.assertEqual(onehot_b.sparse_id_column.name, "b")
    self.assertEqual(onehot_b.length, 100)

  def testOneHotReshaping(self):
    """Tests reshaping behavior of `OneHotColumn`."""
    id_tensor_shape = [3, 2, 4, 5]

    sparse_column = fc.sparse_column_with_keys(
        "animals", ["squirrel", "moose", "dragon", "octopus"])
    one_hot = fc.one_hot_column(sparse_column)

    vocab_size = len(sparse_column.lookup_config.keys)
    id_tensor = _sparse_id_tensor(id_tensor_shape, vocab_size)

    for output_rank in range(1, len(id_tensor_shape) + 1):
      with variable_scope.variable_scope("output_rank_{}".format(output_rank)):
        one_hot_output = one_hot._to_dnn_input_layer(
            id_tensor, output_rank=output_rank)
      with self.test_session() as sess:
        one_hot_value = sess.run(one_hot_output)
        expected_shape = (id_tensor_shape[:output_rank - 1] + [vocab_size])
        self.assertEquals(expected_shape, list(one_hot_value.shape))

  def testOneHotColumnForWeightedSparseColumn(self):
    ids = fc.sparse_column_with_keys("ids", ["marlo", "omar", "stringer"])
    weighted_ids = fc.weighted_sparse_column(ids, "weights")
    one_hot = fc.one_hot_column(weighted_ids)
    self.assertEqual(one_hot.sparse_id_column.name, "ids_weighted_by_weights")
    self.assertEqual(one_hot.length, 3)

  def testOneHotColumnDeepCopy(self):
    a = fc.sparse_column_with_keys("a", ["a", "b", "c", "d"])
    column = fc.one_hot_column(a)
    column_copy = copy.deepcopy(column)
    self.assertEqual(column_copy.sparse_id_column.name, "a")
    self.assertEqual(column.name, "a_one_hot")
    self.assertEqual(column.length, 4)

  def testRealValuedVarLenColumn(self):
    c = fc._real_valued_var_len_column("ccc", is_sparse=True)
    self.assertTrue(c.is_sparse)
    self.assertTrue(c.default_value is None)
    # default_value is an integer.
    c5 = fc._real_valued_var_len_column("c5", default_value=2)
    self.assertEqual(c5.default_value, 2)
    # default_value is a float.
    d4 = fc._real_valued_var_len_column("d4", is_sparse=True)
    self.assertEqual(d4.default_value, None)
    self.assertEqual(d4.is_sparse, True)
    # Default value is a list but dimension is None.
    with self.assertRaisesRegexp(ValueError,
                                 "Only scalar default value.*"):
      fc._real_valued_var_len_column("g5", default_value=[2., 3.])

  def testRealValuedVarLenColumnDtypes(self):
    rvc = fc._real_valued_var_len_column("rvc", is_sparse=True)
    self.assertDictEqual(
        {
            "rvc": parsing_ops.VarLenFeature(dtype=dtypes.float32)
        }, rvc.config)

    rvc = fc._real_valued_var_len_column("rvc", default_value=0,
                                         is_sparse=False)
    self.assertDictEqual(
        {
            "rvc": parsing_ops.FixedLenSequenceFeature(shape=[],
                                                       dtype=dtypes.float32,
                                                       allow_missing=True,
                                                       default_value=0.0)
        }, rvc.config)

    rvc = fc._real_valued_var_len_column("rvc", dtype=dtypes.int32,
                                         default_value=0, is_sparse=True)
    self.assertDictEqual(
        {
            "rvc": parsing_ops.VarLenFeature(dtype=dtypes.int32)
        }, rvc.config)

    with self.assertRaisesRegexp(TypeError,
                                 "dtype must be convertible to float"):
      fc._real_valued_var_len_column("rvc", dtype=dtypes.string,
                                     default_value="", is_sparse=True)

  def testRealValuedColumn(self):
    a = fc.real_valued_column("aaa")
    self.assertEqual(a.name, "aaa")
    self.assertEqual(a.dimension, 1)
    b = fc.real_valued_column("bbb", 10)
    self.assertEqual(b.dimension, 10)
    self.assertTrue(b.default_value is None)

    with self.assertRaisesRegexp(TypeError, "dimension must be an integer"):
      fc.real_valued_column("d3", dimension=1.0)

    with self.assertRaisesRegexp(ValueError,
                                 "dimension must be greater than 0"):
      fc.real_valued_column("d3", dimension=0)

    with self.assertRaisesRegexp(ValueError,
                                 "dtype must be convertible to float"):
      fc.real_valued_column("d3", dtype=dtypes.string)

    # default_value is an integer.
    c1 = fc.real_valued_column("c1", default_value=2)
    self.assertListEqual(list(c1.default_value), [2.])
    c2 = fc.real_valued_column("c2", default_value=2, dtype=dtypes.int32)
    self.assertListEqual(list(c2.default_value), [2])
    c3 = fc.real_valued_column("c3", dimension=4, default_value=2)
    self.assertListEqual(list(c3.default_value), [2, 2, 2, 2])
    c4 = fc.real_valued_column(
        "c4", dimension=4, default_value=2, dtype=dtypes.int32)
    self.assertListEqual(list(c4.default_value), [2, 2, 2, 2])

    # default_value is a float.
    d1 = fc.real_valued_column("d1", default_value=2.)
    self.assertListEqual(list(d1.default_value), [2.])
    d2 = fc.real_valued_column("d2", dimension=4, default_value=2.)
    self.assertListEqual(list(d2.default_value), [2., 2., 2., 2.])
    with self.assertRaisesRegexp(TypeError,
                                 "default_value must be compatible with dtype"):
      fc.real_valued_column("d3", default_value=2., dtype=dtypes.int32)

    # default_value is neither integer nor float.
    with self.assertRaisesRegexp(TypeError,
                                 "default_value must be compatible with dtype"):
      fc.real_valued_column("e1", default_value="string")
    with self.assertRaisesRegexp(TypeError,
                                 "default_value must be compatible with dtype"):
      fc.real_valued_column("e1", dimension=3, default_value=[1, 3., "string"])

    # default_value is a list of integers.
    f1 = fc.real_valued_column("f1", default_value=[2])
    self.assertListEqual(list(f1.default_value), [2])
    f2 = fc.real_valued_column("f2", dimension=3, default_value=[2, 2, 2])
    self.assertListEqual(list(f2.default_value), [2., 2., 2.])
    f3 = fc.real_valued_column(
        "f3", dimension=3, default_value=[2, 2, 2], dtype=dtypes.int32)
    self.assertListEqual(list(f3.default_value), [2, 2, 2])

    # default_value is a list of floats.
    g1 = fc.real_valued_column("g1", default_value=[2.])
    self.assertListEqual(list(g1.default_value), [2.])
    g2 = fc.real_valued_column("g2", dimension=3, default_value=[2., 2, 2])
    self.assertListEqual(list(g2.default_value), [2., 2., 2.])
    with self.assertRaisesRegexp(TypeError,
                                 "default_value must be compatible with dtype"):
      fc.real_valued_column("g3", default_value=[2.], dtype=dtypes.int32)
    with self.assertRaisesRegexp(
        ValueError, "The length of default_value must be equal to dimension"):
      fc.real_valued_column("g4", dimension=3, default_value=[2.])

    # Test that the normalizer_fn gets stored for a real_valued_column
    normalizer = lambda x: x - 1
    h1 = fc.real_valued_column("h1", normalizer=normalizer)
    self.assertEqual(normalizer(10), h1.normalizer_fn(10))

    # Test that normalizer is not stored within key
    self.assertFalse("normalizer" in g1.key)
    self.assertFalse("normalizer" in g2.key)
    self.assertFalse("normalizer" in h1.key)

  def testRealValuedColumnReshaping(self):
    """Tests reshaping behavior of `RealValuedColumn`."""
    batch_size = 4
    sequence_length = 8
    dimensions = [3, 4, 5]

    np.random.seed(2222)
    input_shape = [batch_size, sequence_length] + dimensions
    real_valued_input = np.random.rand(*input_shape)
    real_valued_column = fc.real_valued_column("values")

    for output_rank in range(1, 3 + len(dimensions)):
      with variable_scope.variable_scope("output_rank_{}".format(output_rank)):
        real_valued_output = real_valued_column._to_dnn_input_layer(
            constant_op.constant(
                real_valued_input, dtype=dtypes.float32),
            output_rank=output_rank)
      with self.test_session() as sess:
        real_valued_eval = sess.run(real_valued_output)
      expected_shape = (input_shape[:output_rank - 1] +
                        [np.prod(input_shape[output_rank - 1:])])
      self.assertEquals(expected_shape, list(real_valued_eval.shape))

  def testRealValuedColumnDensification(self):
    """Tests densification behavior of `RealValuedColumn`."""
    # No default value, dimension 1 float.
    real_valued_column = fc._real_valued_var_len_column(
        "sparse_real_valued1", is_sparse=True)
    sparse_tensor = sparse_tensor_lib.SparseTensor(
        values=[2.0, 5.0], indices=[[0, 0], [2, 0]], dense_shape=[3, 1])
    with self.assertRaisesRegexp(
        ValueError, "Set is_sparse to False"):
      real_valued_column._to_dnn_input_layer(sparse_tensor)

  def testRealValuedColumnDeepCopy(self):
    column = fc.real_valued_column(
        "aaa", dimension=3, default_value=[1, 2, 3], dtype=dtypes.int32)
    column_copy = copy.deepcopy(column)
    self.assertEqual(column_copy.name, "aaa")
    self.assertEqual(column_copy.dimension, 3)
    self.assertEqual(column_copy.default_value, (1, 2, 3))

  def testBucketizedColumnNameEndsWithUnderscoreBucketized(self):
    a = fc.bucketized_column(fc.real_valued_column("aaa"), [0, 4])
    self.assertEqual(a.name, "aaa_bucketized")

  def testBucketizedColumnRequiresRealValuedColumn(self):
    with self.assertRaisesRegexp(
        TypeError, "source_column must be an instance of _RealValuedColumn"):
      fc.bucketized_column("bbb", [0])
    with self.assertRaisesRegexp(
        TypeError, "source_column must be an instance of _RealValuedColumn"):
      fc.bucketized_column(
          fc.sparse_column_with_integerized_feature(
              column_name="bbb", bucket_size=10), [0])

  def testBucketizedColumnRequiresRealValuedColumnDimension(self):
    with self.assertRaisesRegexp(
        TypeError, "source_column must be an instance of _RealValuedColumn.*"):
      fc.bucketized_column(fc._real_valued_var_len_column("bbb",
                                                          is_sparse=True),
                           [0])

  def testBucketizedColumnRequiresSortedBuckets(self):
    with self.assertRaisesRegexp(ValueError,
                                 "boundaries must be a sorted list"):
      fc.bucketized_column(fc.real_valued_column("ccc"), [5, 0, 4])

  def testBucketizedColumnWithSameBucketBoundaries(self):
    a_bucketized = fc.bucketized_column(
        fc.real_valued_column("a"), [1., 2., 2., 3., 3.])
    self.assertEqual(a_bucketized.name, "a_bucketized")
    self.assertTupleEqual(a_bucketized.boundaries, (1., 2., 3.))

  def testBucketizedColumnDeepCopy(self):
    """Tests that we can do a deepcopy of a bucketized column.

    This test requires that the bucketized column also accept boundaries
    as tuples.
    """
    bucketized = fc.bucketized_column(
        fc.real_valued_column("a"), [1., 2., 2., 3., 3.])
    self.assertEqual(bucketized.name, "a_bucketized")
    self.assertTupleEqual(bucketized.boundaries, (1., 2., 3.))
    bucketized_copy = copy.deepcopy(bucketized)
    self.assertEqual(bucketized_copy.name, "a_bucketized")
    self.assertTupleEqual(bucketized_copy.boundaries, (1., 2., 3.))

  def testCrossedColumnNameCreatesSortedNames(self):
    a = fc.sparse_column_with_hash_bucket("aaa", hash_bucket_size=100)
    b = fc.sparse_column_with_hash_bucket("bbb", hash_bucket_size=100)
    bucket = fc.bucketized_column(fc.real_valued_column("cost"), [0, 4])
    crossed = fc.crossed_column(set([b, bucket, a]), hash_bucket_size=10000)

    self.assertEqual("aaa_X_bbb_X_cost_bucketized", crossed.name,
                     "name should be generated by sorted column names")
    self.assertEqual("aaa", crossed.columns[0].name)
    self.assertEqual("bbb", crossed.columns[1].name)
    self.assertEqual("cost_bucketized", crossed.columns[2].name)

  def testCrossedColumnNotSupportRealValuedColumn(self):
    b = fc.sparse_column_with_hash_bucket("bbb", hash_bucket_size=100)
    with self.assertRaisesRegexp(
        TypeError, "columns must be a set of _SparseColumn, _CrossedColumn, "
        "or _BucketizedColumn instances"):
      fc.crossed_column(
          set([b, fc.real_valued_column("real")]), hash_bucket_size=10000)

  def testCrossedColumnDeepCopy(self):
    a = fc.sparse_column_with_hash_bucket("aaa", hash_bucket_size=100)
    b = fc.sparse_column_with_hash_bucket("bbb", hash_bucket_size=100)
    bucket = fc.bucketized_column(fc.real_valued_column("cost"), [0, 4])
    crossed = fc.crossed_column(set([b, bucket, a]), hash_bucket_size=10000)
    crossed_copy = copy.deepcopy(crossed)
    self.assertEqual("aaa_X_bbb_X_cost_bucketized", crossed_copy.name,
                     "name should be generated by sorted column names")
    self.assertEqual("aaa", crossed_copy.columns[0].name)
    self.assertEqual("bbb", crossed_copy.columns[1].name)
    self.assertEqual("cost_bucketized", crossed_copy.columns[2].name)

  def testFloat32WeightedSparseInt32ColumnDtypes(self):
    ids = fc.sparse_column_with_keys("ids", [42, 1, -1000], dtype=dtypes.int32)
    weighted_ids = fc.weighted_sparse_column(ids, "weights")
    self.assertDictEqual({
        "ids": parsing_ops.VarLenFeature(dtypes.int32),
        "weights": parsing_ops.VarLenFeature(dtypes.float32)
    }, weighted_ids.config)

  def testFloat32WeightedSparseStringColumnDtypes(self):
    ids = fc.sparse_column_with_keys("ids", ["marlo", "omar", "stringer"])
    weighted_ids = fc.weighted_sparse_column(ids, "weights")
    self.assertDictEqual({
        "ids": parsing_ops.VarLenFeature(dtypes.string),
        "weights": parsing_ops.VarLenFeature(dtypes.float32)
    }, weighted_ids.config)

  def testInt32WeightedSparseStringColumnDtypes(self):
    ids = fc.sparse_column_with_keys("ids", ["marlo", "omar", "stringer"])
    weighted_ids = fc.weighted_sparse_column(ids, "weights", dtype=dtypes.int32)
    self.assertDictEqual({
        "ids": parsing_ops.VarLenFeature(dtypes.string),
        "weights": parsing_ops.VarLenFeature(dtypes.int32)
    }, weighted_ids.config)

    with self.assertRaisesRegexp(ValueError,
                                 "dtype is not convertible to float"):
      weighted_ids = fc.weighted_sparse_column(
          ids, "weights", dtype=dtypes.string)

  def testInt32WeightedSparseInt64ColumnDtypes(self):
    ids = fc.sparse_column_with_keys("ids", [42, 1, -1000], dtype=dtypes.int64)
    weighted_ids = fc.weighted_sparse_column(ids, "weights", dtype=dtypes.int32)
    self.assertDictEqual({
        "ids": parsing_ops.VarLenFeature(dtypes.int64),
        "weights": parsing_ops.VarLenFeature(dtypes.int32)
    }, weighted_ids.config)

    with self.assertRaisesRegexp(ValueError,
                                 "dtype is not convertible to float"):
      weighted_ids = fc.weighted_sparse_column(
          ids, "weights", dtype=dtypes.string)

  def testRealValuedColumnDtypes(self):
    rvc = fc.real_valued_column("rvc")
    self.assertDictEqual(
        {
            "rvc": parsing_ops.FixedLenFeature(
                [1], dtype=dtypes.float32)
        },
        rvc.config)

    rvc = fc.real_valued_column("rvc", dtype=dtypes.int32)
    self.assertDictEqual(
        {
            "rvc": parsing_ops.FixedLenFeature(
                [1], dtype=dtypes.int32)
        },
        rvc.config)

    with self.assertRaisesRegexp(ValueError,
                                 "dtype must be convertible to float"):
      fc.real_valued_column("rvc", dtype=dtypes.string)

  def testSparseColumnDtypes(self):
    sc = fc.sparse_column_with_integerized_feature("sc", 10)
    self.assertDictEqual(
        {
            "sc": parsing_ops.VarLenFeature(dtype=dtypes.int64)
        }, sc.config)

    sc = fc.sparse_column_with_integerized_feature("sc", 10, dtype=dtypes.int32)
    self.assertDictEqual(
        {
            "sc": parsing_ops.VarLenFeature(dtype=dtypes.int32)
        }, sc.config)

    with self.assertRaisesRegexp(ValueError, "dtype must be an integer"):
      fc.sparse_column_with_integerized_feature("sc", 10, dtype=dtypes.float32)

  def testSparseColumnSingleBucket(self):
    sc = fc.sparse_column_with_integerized_feature("sc", 1)
    self.assertDictEqual(
        {
            "sc": parsing_ops.VarLenFeature(dtype=dtypes.int64)
        }, sc.config)
    self.assertEqual(1, sc._wide_embedding_lookup_arguments(None).vocab_size)

  def testSparseColumnAcceptsDenseScalar(self):
    """Tests that `SparseColumn`s accept dense scalar inputs."""
    batch_size = 4
    dense_scalar_input = [1, 2, 3, 4]
    sparse_column = fc.sparse_column_with_integerized_feature("values", 10)
    features = {"values":
                constant_op.constant(dense_scalar_input, dtype=dtypes.int64)}
    sparse_column.insert_transformed_feature(features)
    sparse_output = features[sparse_column]
    expected_shape = [batch_size, 1]
    with self.test_session() as sess:
      sparse_result = sess.run(sparse_output)
    self.assertEquals(expected_shape, list(sparse_result.dense_shape))

  def testSparseColumnIntegerizedDeepCopy(self):
    """Tests deepcopy of sparse_column_with_integerized_feature."""
    column = fc.sparse_column_with_integerized_feature("a", 10)
    self.assertEqual("a", column.name)
    column_copy = copy.deepcopy(column)
    self.assertEqual("a", column_copy.name)
    self.assertEqual(10, column_copy.bucket_size)
    self.assertTrue(column_copy.is_integerized)

  def testSparseColumnHashBucketDeepCopy(self):
    """Tests deepcopy of sparse_column_with_hash_bucket."""
    column = fc.sparse_column_with_hash_bucket("a", 10)
    self.assertEqual("a", column.name)
    column_copy = copy.deepcopy(column)
    self.assertEqual("a", column_copy.name)
    self.assertEqual(10, column_copy.bucket_size)
    self.assertFalse(column_copy.is_integerized)

  def testSparseColumnKeysDeepCopy(self):
    """Tests deepcopy of sparse_column_with_keys."""
    column = fc.sparse_column_with_keys(
        "a", keys=["key0", "key1", "key2"])
    self.assertEqual("a", column.name)
    column_copy = copy.deepcopy(column)
    self.assertEqual("a", column_copy.name)
    self.assertEqual(
        fc._SparseIdLookupConfig(  # pylint: disable=protected-access
            keys=("key0", "key1", "key2"),
            vocab_size=3,
            default_value=-1),
        column_copy.lookup_config)
    self.assertFalse(column_copy.is_integerized)

  def testSparseColumnVocabularyDeepCopy(self):
    """Tests deepcopy of sparse_column_with_vocabulary_file."""
    column = fc.sparse_column_with_vocabulary_file(
        "a", vocabulary_file="path_to_file", vocab_size=3)
    self.assertEqual("a", column.name)
    column_copy = copy.deepcopy(column)
    self.assertEqual("a", column_copy.name)
    self.assertEqual(
        fc._SparseIdLookupConfig(  # pylint: disable=protected-access
            vocabulary_file="path_to_file",
            num_oov_buckets=0,
            vocab_size=3,
            default_value=-1),
        column_copy.lookup_config)
    self.assertFalse(column_copy.is_integerized)

  def testCreateFeatureSpec(self):
    sparse_col = fc.sparse_column_with_hash_bucket(
        "sparse_column", hash_bucket_size=100)
    embedding_col = fc.embedding_column(
        fc.sparse_column_with_hash_bucket(
            "sparse_column_for_embedding", hash_bucket_size=10),
        dimension=4)
    str_sparse_id_col = fc.sparse_column_with_keys(
        "str_id_column", ["marlo", "omar", "stringer"])
    int32_sparse_id_col = fc.sparse_column_with_keys(
        "int32_id_column", [42, 1, -1000], dtype=dtypes.int32)
    int64_sparse_id_col = fc.sparse_column_with_keys(
        "int64_id_column", [42, 1, -1000], dtype=dtypes.int64)
    weighted_id_col = fc.weighted_sparse_column(str_sparse_id_col,
                                                "str_id_weights_column")
    real_valued_col1 = fc.real_valued_column("real_valued_column1")
    real_valued_col2 = fc.real_valued_column("real_valued_column2", 5)
    bucketized_col1 = fc.bucketized_column(
        fc.real_valued_column("real_valued_column_for_bucketization1"), [0, 4])
    bucketized_col2 = fc.bucketized_column(
        fc.real_valued_column("real_valued_column_for_bucketization2", 4),
        [0, 4])
    a = fc.sparse_column_with_hash_bucket("cross_aaa", hash_bucket_size=100)
    b = fc.sparse_column_with_hash_bucket("cross_bbb", hash_bucket_size=100)
    cross_col = fc.crossed_column(set([a, b]), hash_bucket_size=10000)
    one_hot_col = fc.one_hot_column(fc.sparse_column_with_hash_bucket(
        "sparse_column_for_one_hot", hash_bucket_size=100))
    scattered_embedding_col = fc.scattered_embedding_column(
        "scattered_embedding_column", size=100, dimension=10, hash_key=1)
    feature_columns = set([
        sparse_col, embedding_col, weighted_id_col, int32_sparse_id_col,
        int64_sparse_id_col, real_valued_col1, real_valued_col2,
        bucketized_col1, bucketized_col2, cross_col, one_hot_col,
        scattered_embedding_col
    ])
    expected_config = {
        "sparse_column":
            parsing_ops.VarLenFeature(dtypes.string),
        "sparse_column_for_embedding":
            parsing_ops.VarLenFeature(dtypes.string),
        "str_id_column":
            parsing_ops.VarLenFeature(dtypes.string),
        "int32_id_column":
            parsing_ops.VarLenFeature(dtypes.int32),
        "int64_id_column":
            parsing_ops.VarLenFeature(dtypes.int64),
        "str_id_weights_column":
            parsing_ops.VarLenFeature(dtypes.float32),
        "real_valued_column1":
            parsing_ops.FixedLenFeature(
                [1], dtype=dtypes.float32),
        "real_valued_column2":
            parsing_ops.FixedLenFeature(
                [5], dtype=dtypes.float32),
        "real_valued_column_for_bucketization1":
            parsing_ops.FixedLenFeature(
                [1], dtype=dtypes.float32),
        "real_valued_column_for_bucketization2":
            parsing_ops.FixedLenFeature(
                [4], dtype=dtypes.float32),
        "cross_aaa":
            parsing_ops.VarLenFeature(dtypes.string),
        "cross_bbb":
            parsing_ops.VarLenFeature(dtypes.string),
        "sparse_column_for_one_hot":
            parsing_ops.VarLenFeature(dtypes.string),
        "scattered_embedding_column":
            parsing_ops.VarLenFeature(dtypes.string),
    }

    config = fc.create_feature_spec_for_parsing(feature_columns)
    self.assertDictEqual(expected_config, config)

    # Tests that contrib feature columns work with core library:
    config_core = fc_core.make_parse_example_spec(feature_columns)
    self.assertDictEqual(expected_config, config_core)

    # Test that the same config is parsed out if we pass a dictionary.
    feature_columns_dict = {
        str(i): val
        for i, val in enumerate(feature_columns)
    }
    config = fc.create_feature_spec_for_parsing(feature_columns_dict)
    self.assertDictEqual(expected_config, config)

  def testCreateFeatureSpec_ExperimentalColumns(self):
    real_valued_col0 = fc._real_valued_var_len_column(
        "real_valued_column0", is_sparse=True)
    real_valued_col1 = fc._real_valued_var_len_column(
        "real_valued_column1", dtype=dtypes.int64, default_value=0,
        is_sparse=False)
    feature_columns = set([real_valued_col0, real_valued_col1])
    expected_config = {
        "real_valued_column0": parsing_ops.VarLenFeature(dtype=dtypes.float32),
        "real_valued_column1":
            parsing_ops.FixedLenSequenceFeature(
                [], dtype=dtypes.int64, allow_missing=True, default_value=0),
    }

    config = fc.create_feature_spec_for_parsing(feature_columns)
    self.assertDictEqual(expected_config, config)

  def testCreateFeatureSpec_RealValuedColumnWithDefaultValue(self):
    real_valued_col1 = fc.real_valued_column(
        "real_valued_column1", default_value=2)
    real_valued_col2 = fc.real_valued_column(
        "real_valued_column2", 5, default_value=4)
    real_valued_col3 = fc.real_valued_column(
        "real_valued_column3", default_value=[8])
    real_valued_col4 = fc.real_valued_column(
        "real_valued_column4", 3, default_value=[1, 0, 6])
    real_valued_col5 = fc._real_valued_var_len_column(
        "real_valued_column5", default_value=2, is_sparse=True)
    real_valued_col6 = fc._real_valued_var_len_column(
        "real_valued_column6", dtype=dtypes.int64, default_value=1,
        is_sparse=False)
    feature_columns = [
        real_valued_col1, real_valued_col2, real_valued_col3, real_valued_col4,
        real_valued_col5, real_valued_col6
    ]
    config = fc.create_feature_spec_for_parsing(feature_columns)
    self.assertEqual(6, len(config))
    self.assertDictEqual(
        {
            "real_valued_column1":
                parsing_ops.FixedLenFeature(
                    [1], dtype=dtypes.float32, default_value=[2.]),
            "real_valued_column2":
                parsing_ops.FixedLenFeature(
                    [5],
                    dtype=dtypes.float32,
                    default_value=[4., 4., 4., 4., 4.]),
            "real_valued_column3":
                parsing_ops.FixedLenFeature(
                    [1], dtype=dtypes.float32, default_value=[8.]),
            "real_valued_column4":
                parsing_ops.FixedLenFeature(
                    [3], dtype=dtypes.float32, default_value=[1., 0., 6.]),
            "real_valued_column5":
                parsing_ops.VarLenFeature(dtype=dtypes.float32),
            "real_valued_column6":
                parsing_ops.FixedLenSequenceFeature(
                    [], dtype=dtypes.int64, allow_missing=True,
                    default_value=1)
        },
        config)

  def testCreateSequenceFeatureSpec(self):
    sparse_col = fc.sparse_column_with_hash_bucket(
        "sparse_column", hash_bucket_size=100)
    embedding_col = fc.embedding_column(
        fc.sparse_column_with_hash_bucket(
            "sparse_column_for_embedding", hash_bucket_size=10),
        dimension=4)
    sparse_id_col = fc.sparse_column_with_keys("id_column",
                                               ["marlo", "omar", "stringer"])
    weighted_id_col = fc.weighted_sparse_column(sparse_id_col,
                                                "id_weights_column")
    real_valued_col1 = fc.real_valued_column("real_valued_column", dimension=2)
    real_valued_col2 = fc.real_valued_column(
        "real_valued_default_column", dimension=5, default_value=3.0)
    real_valued_col3 = fc._real_valued_var_len_column(
        "real_valued_var_len_column", default_value=3.0, is_sparse=True)
    real_valued_col4 = fc._real_valued_var_len_column(
        "real_valued_var_len_dense_column", default_value=4.0, is_sparse=False)

    feature_columns = set([
        sparse_col, embedding_col, weighted_id_col, real_valued_col1,
        real_valued_col2, real_valued_col3, real_valued_col4
    ])

    feature_spec = fc._create_sequence_feature_spec_for_parsing(feature_columns)

    expected_feature_spec = {
        "sparse_column":
            parsing_ops.VarLenFeature(dtypes.string),
        "sparse_column_for_embedding":
            parsing_ops.VarLenFeature(dtypes.string),
        "id_column":
            parsing_ops.VarLenFeature(dtypes.string),
        "id_weights_column":
            parsing_ops.VarLenFeature(dtypes.float32),
        "real_valued_column":
            parsing_ops.FixedLenSequenceFeature(
                shape=[2], dtype=dtypes.float32, allow_missing=False),
        "real_valued_default_column":
            parsing_ops.FixedLenSequenceFeature(
                shape=[5], dtype=dtypes.float32, allow_missing=True),
        "real_valued_var_len_column":
            parsing_ops.VarLenFeature(dtype=dtypes.float32),
        "real_valued_var_len_dense_column":
            parsing_ops.FixedLenSequenceFeature(
                shape=[], dtype=dtypes.float32, allow_missing=True),
    }

    self.assertDictEqual(expected_feature_spec, feature_spec)

  def testMakePlaceHolderTensorsForBaseFeatures(self):
    sparse_col = fc.sparse_column_with_hash_bucket(
        "sparse_column", hash_bucket_size=100)
    real_valued_col = fc.real_valued_column("real_valued_column", 5)
    vlen_real_valued_col = fc._real_valued_var_len_column(
        "vlen_real_valued_column", is_sparse=True)

    bucketized_col = fc.bucketized_column(
        fc.real_valued_column("real_valued_column_for_bucketization"), [0, 4])
    feature_columns = set(
        [sparse_col, real_valued_col, vlen_real_valued_col, bucketized_col])
    placeholders = (
        fc.make_place_holder_tensors_for_base_features(feature_columns))

    self.assertEqual(4, len(placeholders))
    self.assertTrue(
        isinstance(placeholders["sparse_column"],
                   sparse_tensor_lib.SparseTensor))
    self.assertTrue(
        isinstance(placeholders["vlen_real_valued_column"],
                   sparse_tensor_lib.SparseTensor))
    placeholder = placeholders["real_valued_column"]
    self.assertGreaterEqual(
        placeholder.name.find(u"Placeholder_real_valued_column"), 0)
    self.assertEqual(dtypes.float32, placeholder.dtype)
    self.assertEqual([None, 5], placeholder.get_shape().as_list())
    placeholder = placeholders["real_valued_column_for_bucketization"]
    self.assertGreaterEqual(
        placeholder.name.find(
            u"Placeholder_real_valued_column_for_bucketization"), 0)
    self.assertEqual(dtypes.float32, placeholder.dtype)
    self.assertEqual([None, 1], placeholder.get_shape().as_list())

  def testInitEmbeddingColumnWeightsFromCkpt(self):
    sparse_col = fc.sparse_column_with_hash_bucket(
        column_name="object_in_image", hash_bucket_size=4)
    # Create _EmbeddingColumn which randomly initializes embedding of size
    # [4, 16].
    embedding_col = fc.embedding_column(sparse_col, dimension=16)

    # Creating a SparseTensor which has all the ids possible for the given
    # vocab.
    input_tensor = sparse_tensor_lib.SparseTensor(
        indices=[[0, 0], [1, 1], [2, 2], [3, 3]],
        values=[0, 1, 2, 3],
        dense_shape=[4, 4])

    # Invoking 'layers.input_from_feature_columns' will create the embedding
    # variable. Creating under scope 'run_1' so as to prevent name conflicts
    # when creating embedding variable for 'embedding_column_pretrained'.
    with variable_scope.variable_scope("run_1"):
      with variable_scope.variable_scope(embedding_col.name):
        # This will return a [4, 16] tensor which is same as embedding variable.
        embeddings = feature_column_ops.input_from_feature_columns({
            embedding_col: input_tensor
        }, [embedding_col])

    save = saver.Saver()
    ckpt_dir_prefix = os.path.join(self.get_temp_dir(),
                                   "init_embedding_col_w_from_ckpt")
    ckpt_dir = tempfile.mkdtemp(prefix=ckpt_dir_prefix)
    checkpoint_path = os.path.join(ckpt_dir, "model.ckpt")

    with self.test_session() as sess:
      sess.run(variables.global_variables_initializer())
      saved_embedding = embeddings.eval()
      save.save(sess, checkpoint_path)

    embedding_col_initialized = fc.embedding_column(
        sparse_id_column=sparse_col,
        dimension=16,
        ckpt_to_load_from=checkpoint_path,
        tensor_name_in_ckpt=("run_1/object_in_image_embedding/"
                             "input_from_feature_columns/object"
                             "_in_image_embedding/weights"))

    with variable_scope.variable_scope("run_2"):
      # This will initialize the embedding from provided checkpoint and return a
      # [4, 16] tensor which is same as embedding variable. Since we didn't
      # modify embeddings, this should be same as 'saved_embedding'.
      pretrained_embeddings = feature_column_ops.input_from_feature_columns({
          embedding_col_initialized: input_tensor
      }, [embedding_col_initialized])

    with self.test_session() as sess:
      sess.run(variables.global_variables_initializer())
      loaded_embedding = pretrained_embeddings.eval()

    self.assertAllClose(saved_embedding, loaded_embedding)

  def testInitCrossedColumnWeightsFromCkpt(self):
    sparse_col_1 = fc.sparse_column_with_hash_bucket(
        column_name="col_1", hash_bucket_size=4)
    sparse_col_2 = fc.sparse_column_with_keys(
        column_name="col_2", keys=("foo", "bar", "baz"))
    sparse_col_3 = fc.sparse_column_with_keys(
        column_name="col_3", keys=(42, 1, -1000), dtype=dtypes.int64)

    crossed_col = fc.crossed_column(
        columns=[sparse_col_1, sparse_col_2, sparse_col_3], hash_bucket_size=4)

    input_tensor = sparse_tensor_lib.SparseTensor(
        indices=[[0, 0], [1, 1], [2, 2], [3, 3]],
        values=[0, 1, 2, 3],
        dense_shape=[4, 4])

    # Invoking 'weighted_sum_from_feature_columns' will create the crossed
    # column weights variable.
    with variable_scope.variable_scope("run_1"):
      with variable_scope.variable_scope(crossed_col.name):
        # Returns looked up column weights which is same as crossed column
        # weights as well as actual references to weights variables.
        _, col_weights, _ = (
            feature_column_ops.weighted_sum_from_feature_columns({
                sparse_col_1.name: input_tensor,
                sparse_col_2.name: input_tensor,
                sparse_col_3.name: input_tensor
            }, [crossed_col], 1))
        # Update the weights since default initializer initializes all weights
        # to 0.0.
        for weight in col_weights.values():
          assign_op = state_ops.assign(weight[0], weight[0] + 0.5)

    save = saver.Saver()
    ckpt_dir_prefix = os.path.join(self.get_temp_dir(),
                                   "init_crossed_col_w_from_ckpt")
    ckpt_dir = tempfile.mkdtemp(prefix=ckpt_dir_prefix)
    checkpoint_path = os.path.join(ckpt_dir, "model.ckpt")

    with self.test_session() as sess:
      sess.run(variables.global_variables_initializer())
      sess.run(assign_op)
      saved_col_weights = col_weights[crossed_col][0].eval()
      save.save(sess, checkpoint_path)

    crossed_col_initialized = fc.crossed_column(
        columns=[sparse_col_1, sparse_col_2],
        hash_bucket_size=4,
        ckpt_to_load_from=checkpoint_path,
        tensor_name_in_ckpt=("run_1/col_1_X_col_2_X_col_3/"
                             "weighted_sum_from_feature_columns/"
                             "col_1_X_col_2_X_col_3/weights"))

    with variable_scope.variable_scope("run_2"):
      # This will initialize the crossed column weights from provided checkpoint
      # and return a [4, 1] tensor which is same as weights variable. Since we
      # won't modify weights, this should be same as 'saved_col_weights'.
      _, col_weights, _ = (feature_column_ops.weighted_sum_from_feature_columns(
          {
              sparse_col_1.name: input_tensor,
              sparse_col_2.name: input_tensor
          }, [crossed_col_initialized], 1))
      col_weights_from_ckpt = col_weights[crossed_col_initialized][0]

    with self.test_session() as sess:
      sess.run(variables.global_variables_initializer())
      loaded_col_weights = col_weights_from_ckpt.eval()

    self.assertAllClose(saved_col_weights, loaded_col_weights)


if __name__ == "__main__":
  test.main()

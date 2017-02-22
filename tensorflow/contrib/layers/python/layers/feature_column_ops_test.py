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
"""Tests for layers.feature_column_ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

# TODO: #6568 Remove this hack that makes dlopen() not crash.
if hasattr(sys, "getdlopenflags") and hasattr(sys, "setdlopenflags"):
  import ctypes
  sys.setdlopenflags(sys.getdlopenflags() | ctypes.RTLD_GLOBAL)

import numpy as np

from tensorflow.contrib import layers
from tensorflow.contrib.layers.python.layers import feature_column
from tensorflow.contrib.layers.python.layers import feature_column_ops
from tensorflow.core.example import example_pb2
from tensorflow.core.example import feature_pb2
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import partitioned_variables
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables as variables_lib
from tensorflow.python.platform import test


class TransformerTest(test.TestCase):

  def testRealValuedColumnIsIdentityTransformation(self):
    real_valued = feature_column.real_valued_column("price")
    features = {"price": constant_op.constant([[20.], [110], [-3]])}
    output = feature_column_ops._Transformer(features).transform(real_valued)
    with self.test_session():
      self.assertAllEqual(output.eval(), [[20.], [110], [-3]])

  def testSparseRealValuedColumnIdentityTransformation(self):
    sparse_real_valued = feature_column.real_valued_column(
        "rating", dimension=None)
    rating_tensor = sparse_tensor.SparseTensor(
        values=[2.0, 5.0], indices=[[0, 0], [2, 0]], dense_shape=[3, 1])
    features = {"rating": rating_tensor}
    output = feature_column_ops._Transformer(features).transform(
        sparse_real_valued)
    with self.test_session():
      self.assertAllEqual(output.values.eval(), rating_tensor.values.eval())
      self.assertAllEqual(output.indices.eval(), rating_tensor.indices.eval())
      self.assertAllEqual(output.dense_shape.eval(),
                          rating_tensor.dense_shape.eval())

  def testSparseRealValuedColumnWithTransformation(self):

    def square_fn(x):
      return sparse_tensor.SparseTensor(
          values=x.values**2, indices=x.indices, dense_shape=x.dense_shape)

    sparse_real_valued = feature_column.real_valued_column(
        "rating", dimension=None, normalizer=square_fn)
    rating_tensor = sparse_tensor.SparseTensor(
        values=[2.0, 5.0], indices=[[0, 0], [2, 0]], dense_shape=[3, 1])
    features = {"rating": rating_tensor}
    output_dict = feature_column_ops.transform_features(features,
                                                        [sparse_real_valued])
    self.assertTrue(sparse_real_valued in output_dict)
    output = output_dict[sparse_real_valued]
    with self.test_session():
      self.assertArrayNear(output.values.eval(), [4.0, 25.0], 1e-5)
      self.assertAllEqual(output.indices.eval(), rating_tensor.indices.eval())
      self.assertAllEqual(output.dense_shape.eval(),
                          rating_tensor.dense_shape.eval())

  def testBucketizedColumn(self):
    bucket = feature_column.bucketized_column(
        feature_column.real_valued_column("price"), boundaries=[0., 10., 100.])
    # buckets 2, 3, 0
    features = {"price": constant_op.constant([[20.], [110], [-3]])}

    # Test transform features.
    output = feature_column_ops.transform_features(
        features=features, feature_columns=[bucket])
    self.assertEqual(len(output), 1)
    self.assertIn(bucket, output)
    with self.test_session():
      self.assertAllEqual(output[bucket].eval(), [[2], [3], [0]])

  def testBucketizedColumnWithMultiDimensions(self):
    bucket = feature_column.bucketized_column(
        feature_column.real_valued_column("price", 2),
        boundaries=[0., 10., 100.])
    # buckets 2, 3, 0
    features = {
        "price": constant_op.constant([[20., 110], [110., 20], [-3, -3]])
    }
    output = feature_column_ops._Transformer(features).transform(bucket)
    with self.test_session():
      self.assertAllEqual(output.eval(), [[2, 3], [3, 2], [0, 0]])

  def testCachedTransformation(self):
    bucket = feature_column.bucketized_column(
        feature_column.real_valued_column("price"), boundaries=[0., 10., 100.])
    # buckets 2, 3, 0
    features = {"price": constant_op.constant([[20.], [110], [-3]])}
    transformer = feature_column_ops._Transformer(features)
    with self.test_session() as sess:
      transformer.transform(bucket)
      num_of_ops = len(sess.graph.get_operations())
      # Verify that the second call to transform the same feature
      # doesn't increase the number of ops.
      transformer.transform(bucket)
      self.assertEqual(num_of_ops, len(sess.graph.get_operations()))

  def testSparseColumnWithHashBucket(self):
    hashed_sparse = feature_column.sparse_column_with_hash_bucket("wire", 10)
    wire_tensor = sparse_tensor.SparseTensor(
        values=["omar", "stringer", "marlo"],
        indices=[[0, 0], [1, 0], [1, 1]],
        dense_shape=[2, 2])
    features = {"wire": wire_tensor}
    # Test transform features.
    output = feature_column_ops.transform_features(
        features=features, feature_columns=[hashed_sparse])
    self.assertEqual(len(output), 1)
    self.assertIn(hashed_sparse, output)
    with self.test_session():
      self.assertEqual(output[hashed_sparse].values.dtype, dtypes.int64)
      self.assertTrue(
          all(x < 10 and x >= 0 for x in output[hashed_sparse].values.eval()))
      self.assertAllEqual(output[hashed_sparse].indices.eval(),
                          wire_tensor.indices.eval())
      self.assertAllEqual(output[hashed_sparse].dense_shape.eval(),
                          wire_tensor.dense_shape.eval())

  def testSparseIntColumnWithHashBucket(self):
    """Tests a sparse column with int values."""
    hashed_sparse = feature_column.sparse_column_with_hash_bucket(
        "wire", 10, dtype=dtypes.int64)
    wire_tensor = sparse_tensor.SparseTensor(
        values=[101, 201, 301],
        indices=[[0, 0], [1, 0], [1, 1]],
        dense_shape=[2, 2])
    features = {"wire": wire_tensor}
    # Test transform features.
    output = feature_column_ops.transform_features(
        features=features, feature_columns=[hashed_sparse])
    self.assertEqual(len(output), 1)
    self.assertIn(hashed_sparse, output)
    with self.test_session():
      self.assertEqual(output[hashed_sparse].values.dtype, dtypes.int64)
      self.assertTrue(
          all(x < 10 and x >= 0 for x in output[hashed_sparse].values.eval()))
      self.assertAllEqual(output[hashed_sparse].indices.eval(),
                          wire_tensor.indices.eval())
      self.assertAllEqual(output[hashed_sparse].dense_shape.eval(),
                          wire_tensor.dense_shape.eval())

  def testSparseColumnWithHashBucketWithDenseInputTensor(self):
    hashed_sparse = feature_column.sparse_column_with_hash_bucket("wire", 10)
    wire_tensor = constant_op.constant(
        [["omar", "stringer"], ["marlo", "rick"]])
    features = {"wire": wire_tensor}
    output = feature_column_ops._Transformer(features).transform(hashed_sparse)

    with self.test_session():
      # While the input is a dense Tensor, the output should be a SparseTensor.
      self.assertIsInstance(output, sparse_tensor.SparseTensor)
      self.assertEqual(output.values.dtype, dtypes.int64)
      self.assertTrue(all(x < 10 and x >= 0 for x in output.values.eval()))
      self.assertAllEqual(output.indices.eval(),
                          [[0, 0], [0, 1], [1, 0], [1, 1]])
      self.assertAllEqual(output.dense_shape.eval(), [2, 2])

  def testEmbeddingColumn(self):
    hashed_sparse = feature_column.sparse_column_with_hash_bucket("wire", 10)
    wire_tensor = sparse_tensor.SparseTensor(
        values=["omar", "stringer", "marlo"],
        indices=[[0, 0], [1, 0], [1, 1]],
        dense_shape=[2, 2])
    features = {"wire": wire_tensor}
    output = feature_column_ops._Transformer(features).transform(
        feature_column.embedding_column(hashed_sparse, 10))
    expected = feature_column_ops._Transformer(features).transform(
        hashed_sparse)
    with self.test_session():
      self.assertAllEqual(output.values.eval(), expected.values.eval())
      self.assertAllEqual(output.indices.eval(), expected.indices.eval())
      self.assertAllEqual(output.dense_shape.eval(),
                          expected.dense_shape.eval())

    # Test transform features.
    output = feature_column_ops.transform_features(
        features=features, feature_columns=[hashed_sparse])
    self.assertEqual(len(output), 1)
    self.assertIn(hashed_sparse, output)

  def testSparseColumnWithKeys(self):
    keys_sparse = feature_column.sparse_column_with_keys(
        "wire", ["marlo", "omar", "stringer"])
    wire_tensor = sparse_tensor.SparseTensor(
        values=["omar", "stringer", "marlo"],
        indices=[[0, 0], [1, 0], [1, 1]],
        dense_shape=[2, 2])
    features = {"wire": wire_tensor}
    # Test transform features.
    output = feature_column_ops.transform_features(
        features=features, feature_columns=[keys_sparse])
    self.assertEqual(len(output), 1)
    self.assertIn(keys_sparse, output)
    with self.test_session():
      data_flow_ops.tables_initializer().run()
      self.assertEqual(output[keys_sparse].values.dtype, dtypes.int64)
      self.assertAllEqual(output[keys_sparse].values.eval(), [1, 2, 0])
      self.assertAllEqual(output[keys_sparse].indices.eval(),
                          wire_tensor.indices.eval())
      self.assertAllEqual(output[keys_sparse].dense_shape.eval(),
                          wire_tensor.dense_shape.eval())

  def testSparseColumnWithKeysWithDenseInputTensor(self):
    keys_sparse = feature_column.sparse_column_with_keys(
        "wire", ["marlo", "omar", "stringer", "rick"])
    wire_tensor = constant_op.constant(
        [["omar", "stringer"], ["marlo", "rick"]])

    features = {"wire": wire_tensor}
    output = feature_column_ops._Transformer(features).transform(keys_sparse)

    with self.test_session():
      data_flow_ops.tables_initializer().run()
      # While the input is a dense Tensor, the output should be a SparseTensor.
      self.assertIsInstance(output, sparse_tensor.SparseTensor)
      self.assertEqual(output.dtype, dtypes.int64)
      self.assertAllEqual(output.values.eval(), [1, 2, 0, 3])
      self.assertAllEqual(output.indices.eval(),
                          [[0, 0], [0, 1], [1, 0], [1, 1]])
      self.assertAllEqual(output.dense_shape.eval(), [2, 2])

  def testSparseColumnWithHashBucket_IsIntegerized(self):
    hashed_sparse = feature_column.sparse_column_with_integerized_feature(
        "wire", 10)
    wire_tensor = sparse_tensor.SparseTensor(
        values=[100, 1, 25],
        indices=[[0, 0], [1, 0], [1, 1]],
        dense_shape=[2, 2])
    features = {"wire": wire_tensor}
    # Test transform features.
    output = feature_column_ops.transform_features(
        features=features, feature_columns=[hashed_sparse])
    self.assertEqual(len(output), 1)
    self.assertIn(hashed_sparse, output)
    with self.test_session():
      self.assertEqual(output[hashed_sparse].values.dtype, dtypes.int32)
      self.assertTrue(
          all(x < 10 and x >= 0 for x in output[hashed_sparse].values.eval()))
      self.assertAllEqual(output[hashed_sparse].indices.eval(),
                          wire_tensor.indices.eval())
      self.assertAllEqual(output[hashed_sparse].dense_shape.eval(),
                          wire_tensor.dense_shape.eval())

  def testSparseColumnWithHashBucketWithDenseInputTensor_IsIntegerized(self):
    hashed_sparse = feature_column.sparse_column_with_integerized_feature(
        "wire", 10)
    # wire_tensor = tf.SparseTensor(values=[100, 1, 25],
    #                               indices=[[0, 0], [1, 0], [1, 1]],
    #                               dense_shape=[2, 2])
    wire_tensor = constant_op.constant([[100, 0], [1, 25]])
    features = {"wire": wire_tensor}
    output = feature_column_ops._Transformer(features).transform(hashed_sparse)
    with self.test_session():
      # While the input is a dense Tensor, the output should be a SparseTensor.
      self.assertIsInstance(output, sparse_tensor.SparseTensor)
      self.assertEqual(output.values.dtype, dtypes.int32)
      self.assertTrue(all(x < 10 and x >= 0 for x in output.values.eval()))
      self.assertAllEqual(output.indices.eval(),
                          [[0, 0], [0, 1], [1, 0], [1, 1]])
      self.assertAllEqual(output.dense_shape.eval(), [2, 2])

  def testWeightedSparseColumn(self):
    ids = feature_column.sparse_column_with_keys("ids",
                                                 ["marlo", "omar", "stringer"])
    ids_tensor = sparse_tensor.SparseTensor(
        values=["stringer", "stringer", "marlo"],
        indices=[[0, 0], [1, 0], [1, 1]],
        dense_shape=[2, 2])
    weighted_ids = feature_column.weighted_sparse_column(ids, "weights")
    weights_tensor = sparse_tensor.SparseTensor(
        values=[10.0, 20.0, 30.0],
        indices=[[0, 0], [1, 0], [1, 1]],
        dense_shape=[2, 2])
    features = {"ids": ids_tensor, "weights": weights_tensor}
    # Test transform features.
    output = feature_column_ops.transform_features(
        features=features, feature_columns=[weighted_ids])
    self.assertEqual(len(output), 1)
    self.assertIn(weighted_ids, output)

    with self.test_session():
      data_flow_ops.tables_initializer().run()
      self.assertAllEqual(output[weighted_ids][0].dense_shape.eval(),
                          ids_tensor.dense_shape.eval())
      self.assertAllEqual(output[weighted_ids][0].indices.eval(),
                          ids_tensor.indices.eval())
      self.assertAllEqual(output[weighted_ids][0].values.eval(), [2, 2, 0])
      self.assertAllEqual(output[weighted_ids][1].dense_shape.eval(),
                          weights_tensor.dense_shape.eval())
      self.assertAllEqual(output[weighted_ids][1].indices.eval(),
                          weights_tensor.indices.eval())
      self.assertEqual(output[weighted_ids][1].values.dtype, dtypes.float32)
      self.assertAllEqual(output[weighted_ids][1].values.eval(),
                          weights_tensor.values.eval())

  def testSparseColumnWithVocabulary(self):
    vocabulary_file = os.path.join(self.get_temp_dir(), "movies.txt")
    with open(vocabulary_file, "w") as f:
      f.write("\n".join(["marlo", "omar", "stringer"]) + "\n")
    vocab_sparse = feature_column.sparse_column_with_vocabulary_file(
        "wire", vocabulary_file, vocab_size=3)
    wire_tensor = sparse_tensor.SparseTensor(
        values=["omar", "stringer", "marlo"],
        indices=[[0, 0], [1, 0], [1, 1]],
        dense_shape=[2, 2])
    features = {"wire": wire_tensor}
    output = feature_column_ops.transform_features(
        features=features, feature_columns=[vocab_sparse])
    self.assertEqual(len(output), 1)
    self.assertIn(vocab_sparse, output)
    with self.test_session():
      data_flow_ops.tables_initializer().run()
      self.assertEqual(output[vocab_sparse].values.dtype, dtypes.int64)
      self.assertAllEqual(output[vocab_sparse].values.eval(), [1, 2, 0])
      self.assertAllEqual(output[vocab_sparse].indices.eval(),
                          wire_tensor.indices.eval())
      self.assertAllEqual(output[vocab_sparse].dense_shape.eval(),
                          wire_tensor.dense_shape.eval())

  def testSparseColumnWithVocabularyWithDenseInputTensor(self):
    vocabulary_file = os.path.join(self.get_temp_dir(), "movies.txt")
    with open(vocabulary_file, "w") as f:
      f.write("\n".join(["marlo", "omar", "stringer"]) + "\n")
    vocab_sparse = feature_column.sparse_column_with_vocabulary_file(
        "wire", vocabulary_file, vocab_size=3)
    wire_tensor = constant_op.constant(
        [["omar", "stringer"], ["marlo", "omar"]])
    features = {"wire": wire_tensor}
    output = feature_column_ops.transform_features(
        features=features, feature_columns=[vocab_sparse])
    self.assertEqual(len(output), 1)
    self.assertIn(vocab_sparse, output)
    with self.test_session():
      data_flow_ops.tables_initializer().run()
      self.assertEqual(output[vocab_sparse].values.dtype, dtypes.int64)
      self.assertAllEqual(output[vocab_sparse].values.eval(), [1, 2, 0, 1])
      self.assertAllEqual(output[vocab_sparse].indices.eval(),
                          [[0, 0], [0, 1], [1, 0], [1, 1]])
      self.assertAllEqual(output[vocab_sparse].dense_shape.eval(), [2, 2])

  def testSparseIntColumnWithVocabulary(self):
    """Tests a sparse integer column with vocabulary."""
    vocabulary_file = os.path.join(self.get_temp_dir(), "courses.txt")
    with open(vocabulary_file, "w") as f:
      f.write("\n".join(["101", "201", "301"]) + "\n")
    vocab_sparse = feature_column.sparse_column_with_vocabulary_file(
        "wire", vocabulary_file, vocab_size=3, dtype=dtypes.int64)
    wire_tensor = sparse_tensor.SparseTensor(
        values=[201, 301, 101],
        indices=[[0, 0], [1, 0], [1, 1]],
        dense_shape=[2, 2])
    features = {"wire": wire_tensor}
    output = feature_column_ops.transform_features(
        features=features, feature_columns=[vocab_sparse])
    self.assertEqual(len(output), 1)
    self.assertIn(vocab_sparse, output)
    with self.test_session():
      data_flow_ops.tables_initializer().run()
      self.assertEqual(output[vocab_sparse].values.dtype, dtypes.int64)
      self.assertAllEqual(output[vocab_sparse].values.eval(), [1, 2, 0])
      self.assertAllEqual(output[vocab_sparse].indices.eval(),
                          wire_tensor.indices.eval())
      self.assertAllEqual(output[vocab_sparse].dense_shape.eval(),
                          wire_tensor.dense_shape.eval())

  def testSparseIntColumnWithVocabularyWithDenseInputTensor(self):
    """Tests a sparse integer column with vocabulary."""
    vocabulary_file = os.path.join(self.get_temp_dir(), "courses.txt")
    with open(vocabulary_file, "w") as f:
      f.write("\n".join(["101", "201", "301"]) + "\n")
    vocab_sparse = feature_column.sparse_column_with_vocabulary_file(
        "wire", vocabulary_file, vocab_size=3, dtype=dtypes.int64)
    wire_tensor = constant_op.constant([[201, 301], [101, 201]])
    features = {"wire": wire_tensor}
    output = feature_column_ops.transform_features(
        features=features, feature_columns=[vocab_sparse])
    self.assertEqual(len(output), 1)
    self.assertIn(vocab_sparse, output)
    with self.test_session():
      data_flow_ops.tables_initializer().run()
      self.assertEqual(output[vocab_sparse].values.dtype, dtypes.int64)
      self.assertAllEqual(output[vocab_sparse].values.eval(), [1, 2, 0, 1])
      self.assertAllEqual(output[vocab_sparse].indices.eval(),
                          [[0, 0], [0, 1], [1, 0], [1, 1]])
      self.assertAllEqual(output[vocab_sparse].dense_shape.eval(), [2, 2])

  def testCrossColumn(self):
    language = feature_column.sparse_column_with_hash_bucket(
        "language", hash_bucket_size=3)
    country = feature_column.sparse_column_with_hash_bucket(
        "country", hash_bucket_size=5)
    country_language = feature_column.crossed_column(
        [language, country], hash_bucket_size=15)
    features = {
        "language":
            sparse_tensor.SparseTensor(
                values=["english", "spanish"],
                indices=[[0, 0], [1, 0]],
                dense_shape=[2, 1]),
        "country":
            sparse_tensor.SparseTensor(
                values=["US", "SV"],
                indices=[[0, 0], [1, 0]],
                dense_shape=[2, 1])
    }
    # Test transform features.
    output = feature_column_ops.transform_features(
        features=features, feature_columns=[country_language])
    self.assertEqual(len(output), 1)
    self.assertIn(country_language, output)
    with self.test_session():
      self.assertEqual(output[country_language].values.dtype, dtypes.int64)
      self.assertTrue(
          all(x < 15 and x >= 0 for x in output[country_language].values.eval(
          )))

  def testCrossWithBucketizedColumn(self):
    price_bucket = feature_column.bucketized_column(
        feature_column.real_valued_column("price"), boundaries=[0., 10., 100.])
    country = feature_column.sparse_column_with_hash_bucket(
        "country", hash_bucket_size=5)
    country_price = feature_column.crossed_column(
        [country, price_bucket], hash_bucket_size=15)
    features = {
        "price":
            constant_op.constant([[20.]]),
        "country":
            sparse_tensor.SparseTensor(
                values=["US", "SV"],
                indices=[[0, 0], [0, 1]],
                dense_shape=[1, 2])
    }
    # Test transform features.
    output = feature_column_ops.transform_features(
        features=features, feature_columns=[country_price])
    self.assertEqual(len(output), 1)
    self.assertIn(country_price, output)
    with self.test_session():
      self.assertEqual(output[country_price].values.dtype, dtypes.int64)
      self.assertTrue(
          all(x < 15 and x >= 0 for x in output[country_price].values.eval()))

  def testCrossWithMultiDimensionBucketizedColumn(self):
    country = feature_column.sparse_column_with_hash_bucket(
        "country", hash_bucket_size=5)
    price_bucket = feature_column.bucketized_column(
        feature_column.real_valued_column("price", 2),
        boundaries=[0., 10., 100.])
    country_price = feature_column.crossed_column(
        [country, price_bucket], hash_bucket_size=1000)

    with ops.Graph().as_default():
      features = {
          "price":
              constant_op.constant([[20., 210.], [110., 50.], [-3., -30.]]),
          "country":
              sparse_tensor.SparseTensor(
                  values=["US", "SV", "US"],
                  indices=[[0, 0], [1, 0], [2, 0]],
                  dense_shape=[3, 2])
      }
      output, column_to_variable, _ = (
          feature_column_ops.weighted_sum_from_feature_columns(
              features, [country_price], num_outputs=1))

      weights = column_to_variable[country_price][0]
      grad = array_ops.squeeze(
          gradients_impl.gradients(output, weights)[0].values)
      with self.test_session():
        variables_lib.global_variables_initializer().run()
        self.assertEqual(len(grad.eval()), 6)

      # Test transform features.
      output = feature_column_ops.transform_features(
          features=features, feature_columns=[country_price])
      self.assertEqual(len(output), 1)
      self.assertIn(country_price, output)

  def testCrossWithCrossedColumn(self):
    price_bucket = feature_column.bucketized_column(
        feature_column.real_valued_column("price"), boundaries=[0., 10., 100.])
    country = feature_column.sparse_column_with_hash_bucket(
        "country", hash_bucket_size=5)
    country_price = feature_column.crossed_column(
        [country, price_bucket], hash_bucket_size=15)
    wire = feature_column.sparse_column_with_hash_bucket("wire", 10)
    wire_country_price = feature_column.crossed_column(
        [wire, country_price], hash_bucket_size=15)
    features = {
        "price":
            constant_op.constant([[20.]]),
        "country":
            sparse_tensor.SparseTensor(
                values=["US", "SV"],
                indices=[[0, 0], [0, 1]],
                dense_shape=[1, 2]),
        "wire":
            sparse_tensor.SparseTensor(
                values=["omar", "stringer", "marlo"],
                indices=[[0, 0], [0, 1], [0, 2]],
                dense_shape=[1, 3])
    }
    # Test transform features.
    output = feature_column_ops.transform_features(
        features=features, feature_columns=[wire_country_price])
    self.assertEqual(len(output), 1)
    self.assertIn(wire_country_price, output)
    with self.test_session():
      self.assertEqual(output[wire_country_price].values.dtype, dtypes.int64)
      self.assertTrue(
          all(x < 15 and x >= 0 for x in output[wire_country_price].values.eval(
          )))

  def testIfFeatureTableContainsTransformationReturnIt(self):
    any_column = feature_column.sparse_column_with_hash_bucket("sparse", 10)
    features = {any_column: "any-thing-even-not-a-tensor"}
    output = feature_column_ops._Transformer(features).transform(any_column)
    self.assertEqual(output, "any-thing-even-not-a-tensor")


class CreateInputLayersForDNNsTest(test.TestCase):

  def testFeatureColumnDictFails(self):
    real_valued = feature_column.real_valued_column("price")
    features = {"price": constant_op.constant([[20.], [110], [-3]])}
    with self.assertRaisesRegexp(
        ValueError,
        "Expected feature_columns to be iterable, found dict"):
      feature_column_ops.input_from_feature_columns(
          features, {"feature": real_valued})

  def testAllDNNColumns(self):
    sparse_column = feature_column.sparse_column_with_keys(
        "ids", ["a", "b", "c", "unseen"])

    real_valued_column = feature_column.real_valued_column("income", 2)
    sparse_real_valued_column = feature_column.real_valued_column(
        "rating", dimension=None)
    one_hot_column = feature_column.one_hot_column(sparse_column)
    embedding_column = feature_column.embedding_column(sparse_column, 10)
    features = {
        "ids":
            sparse_tensor.SparseTensor(
                values=["c", "b", "a"],
                indices=[[0, 0], [1, 0], [2, 0]],
                dense_shape=[3, 1]),
        "income":
            constant_op.constant([[20.3, 10], [110.3, 0.4], [-3.0, 30.4]]),
        "rating":
            sparse_tensor.SparseTensor(
                values=[3.5, 5.0], indices=[[0, 0], [2, 0]], dense_shape=[3, 1])
    }
    output = feature_column_ops.input_from_feature_columns(features, [
        one_hot_column, embedding_column, real_valued_column,
        sparse_real_valued_column
    ])
    with self.test_session():
      variables_lib.global_variables_initializer().run()
      data_flow_ops.tables_initializer().run()
      self.assertAllEqual(output.eval().shape, [3, 3 + 4 + 10])

  def testRealValuedColumn(self):
    real_valued = feature_column.real_valued_column("price")
    features = {"price": constant_op.constant([[20.], [110], [-3]])}
    output = feature_column_ops.input_from_feature_columns(features,
                                                           [real_valued])
    with self.test_session():
      self.assertAllClose(output.eval(), features["price"].eval())

  def testRealValuedColumnWithMultiDimensions(self):
    real_valued = feature_column.real_valued_column("price", 2)
    features = {
        "price": constant_op.constant([[20., 10.], [110, 0.], [-3, 30]])
    }
    output = feature_column_ops.input_from_feature_columns(features,
                                                           [real_valued])
    with self.test_session():
      self.assertAllClose(output.eval(), features["price"].eval())

  def testRealValuedColumnSparse(self):
    sparse_real_valued = feature_column.real_valued_column(
        "rating", dimension=None, default_value=-1)
    rating_tensor = sparse_tensor.SparseTensor(
        values=[2.0, 5.0], indices=[[0, 0], [2, 0]], dense_shape=[3, 1])
    features = {"rating": rating_tensor}
    output = feature_column_ops.input_from_feature_columns(features,
                                                           [sparse_real_valued])
    with self.test_session():
      self.assertAllClose(output.eval(), [[2.0], [-1.0], [5.0]])

  def testRealValuedColumnWithNormalizer(self):
    real_valued = feature_column.real_valued_column(
        "price", normalizer=lambda x: x - 2)
    features = {"price": constant_op.constant([[20.], [110], [-3]])}
    output = feature_column_ops.input_from_feature_columns(features,
                                                           [real_valued])
    with self.test_session():
      self.assertAllClose(output.eval(), features["price"].eval() - 2)

  def testRealValuedColumnWithMultiDimensionsAndNormalizer(self):
    real_valued = feature_column.real_valued_column(
        "price", 2, normalizer=lambda x: x - 2)
    features = {
        "price": constant_op.constant([[20., 10.], [110, 0.], [-3, 30]])
    }
    output = feature_column_ops.input_from_feature_columns(features,
                                                           [real_valued])
    with self.test_session():
      self.assertAllClose(output.eval(), features["price"].eval() - 2)

  def testBucketizedColumnWithNormalizerSucceedsForDNN(self):
    bucket = feature_column.bucketized_column(
        feature_column.real_valued_column(
            "price", normalizer=lambda x: x - 15),
        boundaries=[0., 10., 100.])
    # buckets 2, 3, 0
    features = {"price": constant_op.constant([[20.], [110], [-3]])}
    output = feature_column_ops.input_from_feature_columns(features, [bucket])
    expected = [[0, 1, 0, 0], [0, 0, 1, 0], [1, 0, 0, 0]]
    with self.test_session():
      self.assertAllClose(output.eval(), expected)

  def testBucketizedColumnWithMultiDimensionsSucceedsForDNN(self):
    bucket = feature_column.bucketized_column(
        feature_column.real_valued_column("price", 2),
        boundaries=[0., 10., 100.])
    # buckets [2, 3], [3, 2], [0, 0]. dimension = 2
    features = {
        "price": constant_op.constant([[20., 200], [110, 50], [-3, -3]])
    }
    output = feature_column_ops.input_from_feature_columns(features, [bucket])
    expected = [[0, 0, 1, 0, 0, 0, 0, 1], [0, 0, 0, 1, 0, 0, 1, 0],
                [1, 0, 0, 0, 1, 0, 0, 0]]
    with self.test_session():
      self.assertAllClose(output.eval(), expected)

  def testOneHotColumnFromWeightedSparseColumnSucceedsForDNN(self):
    ids_column = feature_column.sparse_column_with_keys(
        "ids", ["a", "b", "c", "unseen"])
    ids_tensor = sparse_tensor.SparseTensor(
        values=["c", "b", "a", "c"],
        indices=[[0, 0], [1, 0], [2, 0], [2, 1]],
        dense_shape=[3, 2])
    weighted_ids_column = feature_column.weighted_sparse_column(ids_column,
                                                                "weights")
    weights_tensor = sparse_tensor.SparseTensor(
        values=[10.0, 20.0, 30.0, 40.0],
        indices=[[0, 0], [1, 0], [2, 0], [2, 1]],
        dense_shape=[3, 2])
    features = {"ids": ids_tensor, "weights": weights_tensor}
    one_hot_column = feature_column.one_hot_column(weighted_ids_column)
    output = feature_column_ops.input_from_feature_columns(features,
                                                           [one_hot_column])
    with self.test_session():
      variables_lib.global_variables_initializer().run()
      data_flow_ops.tables_initializer().run()
      self.assertAllEqual([[0, 0, 10., 0], [0, 20., 0, 0], [30., 0, 40., 0]],
                          output.eval())

  def testOneHotColumnFromSparseColumnWithKeysSucceedsForDNN(self):
    ids_column = feature_column.sparse_column_with_keys(
        "ids", ["a", "b", "c", "unseen"])
    ids_tensor = sparse_tensor.SparseTensor(
        values=["c", "b", "a"],
        indices=[[0, 0], [1, 0], [2, 0]],
        dense_shape=[3, 1])
    one_hot_sparse = feature_column.one_hot_column(ids_column)
    features = {"ids": ids_tensor}
    output = feature_column_ops.input_from_feature_columns(features,
                                                           [one_hot_sparse])

    with self.test_session():
      variables_lib.global_variables_initializer().run()
      data_flow_ops.tables_initializer().run()
      self.assertAllEqual([[0, 0, 1, 0], [0, 1, 0, 0], [1, 0, 0, 0]],
                          output.eval())

  def testOneHotColumnFromMultivalentSparseColumnWithKeysSucceedsForDNN(self):
    ids_column = feature_column.sparse_column_with_keys(
        "ids", ["a", "b", "c", "unseen"])
    ids_tensor = sparse_tensor.SparseTensor(
        values=["c", "b", "a", "c"],
        indices=[[0, 0], [1, 0], [2, 0], [2, 1]],
        dense_shape=[3, 2])
    one_hot_sparse = feature_column.one_hot_column(ids_column)
    features = {"ids": ids_tensor}
    output = feature_column_ops.input_from_feature_columns(features,
                                                           [one_hot_sparse])

    with self.test_session():
      variables_lib.global_variables_initializer().run()
      data_flow_ops.tables_initializer().run()
      self.assertAllEqual([[0, 0, 1, 0], [0, 1, 0, 0], [1, 0, 1, 0]],
                          output.eval())

  def testOneHotColumnFromSparseColumnWithIntegerizedFeaturePassesForDNN(self):
    ids_column = feature_column.sparse_column_with_integerized_feature(
        "ids", bucket_size=4)
    one_hot_sparse = feature_column.one_hot_column(ids_column)
    features = {
        "ids":
            sparse_tensor.SparseTensor(
                values=[2, 1, 0, 2],
                indices=[[0, 0], [1, 0], [2, 0], [2, 1]],
                dense_shape=[3, 2])
    }
    output = feature_column_ops.input_from_feature_columns(features,
                                                           [one_hot_sparse])
    with self.test_session():
      variables_lib.global_variables_initializer().run()
      self.assertAllEqual([[0, 0, 1, 0], [0, 1, 0, 0], [1, 0, 1, 0]],
                          output.eval())

  def testOneHotColumnFromSparseColumnWithHashBucketSucceedsForDNN(self):
    hashed_sparse = feature_column.sparse_column_with_hash_bucket("feat", 10)
    wire_tensor = sparse_tensor.SparseTensor(
        values=["a", "b", "c1", "c2"],
        indices=[[0, 0], [1, 0], [2, 0], [2, 1]],
        dense_shape=[3, 2])
    features = {"feat": wire_tensor}
    one_hot_sparse = feature_column.one_hot_column(hashed_sparse)
    output = feature_column_ops.input_from_feature_columns(features,
                                                           [one_hot_sparse])
    with self.test_session():
      variables_lib.global_variables_initializer().run()
      data_flow_ops.tables_initializer().run()
      self.assertAllEqual([3, 10], output.eval().shape)

  def testEmbeddingColumnSucceedsForDNN(self):
    hashed_sparse = feature_column.sparse_column_with_hash_bucket("wire", 10)
    wire_tensor = sparse_tensor.SparseTensor(
        values=["omar", "stringer", "marlo", "xx", "yy"],
        indices=[[0, 0], [1, 0], [1, 1], [2, 0], [3, 0]],
        dense_shape=[4, 2])
    features = {"wire": wire_tensor}
    embeded_sparse = feature_column.embedding_column(hashed_sparse, 10)
    output = feature_column_ops.input_from_feature_columns(features,
                                                           [embeded_sparse])
    with self.test_session():
      variables_lib.global_variables_initializer().run()
      self.assertAllEqual(output.eval().shape, [4, 10])

  def testScatteredEmbeddingColumnSucceedsForDNN(self):
    wire_tensor = sparse_tensor.SparseTensor(
        values=["omar", "stringer", "marlo", "omar"],
        indices=[[0, 0], [1, 0], [1, 1], [2, 0]],
        dense_shape=[3, 2])

    features = {"wire": wire_tensor}
    # Big enough hash space so that hopefully there is no collision
    embedded_sparse = feature_column.scattered_embedding_column(
        "wire", 1000, 3, layers.SPARSE_FEATURE_CROSS_DEFAULT_HASH_KEY)
    output = feature_column_ops.input_from_feature_columns(
        features, [embedded_sparse], weight_collections=["my_collection"])
    weights = ops.get_collection("my_collection")
    grad = gradients_impl.gradients(output, weights)
    with self.test_session():
      variables_lib.global_variables_initializer().run()
      gradient_values = []
      # Collect the gradient from the different partitions (one in this test)
      for p in range(len(grad)):
        gradient_values.extend(grad[p].values.eval())
      gradient_values.sort()
      self.assertAllEqual(gradient_values, [0.5] * 6 + [2] * 3)

  def testEmbeddingColumnWithInitializerSucceedsForDNN(self):
    hashed_sparse = feature_column.sparse_column_with_hash_bucket("wire", 10)
    wire_tensor = sparse_tensor.SparseTensor(
        values=["omar", "stringer", "marlo"],
        indices=[[0, 0], [1, 0], [1, 1]],
        dense_shape=[2, 2])
    features = {"wire": wire_tensor}
    init_value = 133.7
    embeded_sparse = feature_column.embedding_column(
        hashed_sparse,
        10,
        initializer=init_ops.constant_initializer(init_value))
    output = feature_column_ops.input_from_feature_columns(features,
                                                           [embeded_sparse])

    with self.test_session():
      variables_lib.global_variables_initializer().run()
      output_eval = output.eval()
      self.assertAllEqual(output_eval.shape, [2, 10])
      self.assertAllClose(output_eval, np.tile(init_value, [2, 10]))

  def testEmbeddingColumnWithMultipleInitializersFails(self):
    hashed_sparse = feature_column.sparse_column_with_hash_bucket("wire", 10)
    wire_tensor = sparse_tensor.SparseTensor(
        values=["omar", "stringer", "marlo"],
        indices=[[0, 0], [1, 0], [1, 1]],
        dense_shape=[2, 2])
    features = {"wire": wire_tensor}
    embedded_sparse = feature_column.embedding_column(
        hashed_sparse,
        10,
        initializer=init_ops.truncated_normal_initializer(
            mean=42, stddev=1337))
    embedded_sparse_alternate = feature_column.embedding_column(
        hashed_sparse,
        10,
        initializer=init_ops.truncated_normal_initializer(
            mean=1337, stddev=42))

    # Makes sure that trying to use different initializers with the same
    # embedding column explicitly fails.
    with self.test_session():
      with self.assertRaisesRegexp(
          ValueError,
          "Duplicate feature column key found for column: wire_embedding"):
        feature_column_ops.input_from_feature_columns(
            features, [embedded_sparse, embedded_sparse_alternate])

  def testEmbeddingColumnWithWeightedSparseColumnSucceedsForDNN(self):
    """Tests DNN input with embedded weighted sparse column."""
    ids = feature_column.sparse_column_with_keys("ids",
                                                 ["marlo", "omar", "stringer"])
    ids_tensor = sparse_tensor.SparseTensor(
        values=["stringer", "stringer", "marlo"],
        indices=[[0, 0], [1, 0], [1, 1]],
        dense_shape=[2, 2])
    weighted_ids = feature_column.weighted_sparse_column(ids, "weights")
    weights_tensor = sparse_tensor.SparseTensor(
        values=[10.0, 20.0, 30.0],
        indices=[[0, 0], [1, 0], [1, 1]],
        dense_shape=[2, 2])
    features = {"ids": ids_tensor, "weights": weights_tensor}
    embeded_sparse = feature_column.embedding_column(weighted_ids, 10)
    output = feature_column_ops.input_from_feature_columns(features,
                                                           [embeded_sparse])
    with self.test_session():
      variables_lib.global_variables_initializer().run()
      data_flow_ops.tables_initializer().run()
      self.assertAllEqual(output.eval().shape, [2, 10])

  def testEmbeddingColumnWithIntegerWeightedSparseColumnSucceedsForDNN(self):
    """Same as the previous test, but with integer weights."""
    ids = feature_column.sparse_column_with_keys("ids",
                                                 ["marlo", "omar", "stringer"])
    ids_tensor = sparse_tensor.SparseTensor(
        values=["stringer", "stringer", "marlo"],
        indices=[[0, 0], [1, 0], [1, 1]],
        dense_shape=[2, 2])
    weighted_ids = feature_column.weighted_sparse_column(
        ids, "weights", dtype=dtypes.int32)
    weights_tensor = sparse_tensor.SparseTensor(
        values=constant_op.constant([10, 20, 30], dtype=dtypes.int32),
        indices=[[0, 0], [1, 0], [1, 1]],
        dense_shape=[2, 2])
    features = {"ids": ids_tensor, "weights": weights_tensor}
    embeded_sparse = feature_column.embedding_column(weighted_ids, 10)
    output = feature_column_ops.input_from_feature_columns(features,
                                                           [embeded_sparse])
    with self.test_session():
      variables_lib.global_variables_initializer().run()
      data_flow_ops.tables_initializer().run()
      self.assertAllEqual(output.eval().shape, [2, 10])

  def testEmbeddingColumnWithCrossedColumnSucceedsForDNN(self):
    a = feature_column.sparse_column_with_hash_bucket(
        "aaa", hash_bucket_size=100)
    b = feature_column.sparse_column_with_hash_bucket(
        "bbb", hash_bucket_size=100)
    crossed = feature_column.crossed_column(set([a, b]), hash_bucket_size=10000)
    wire_tensor = sparse_tensor.SparseTensor(
        values=["omar", "stringer", "marlo"],
        indices=[[0, 0], [1, 0], [1, 1]],
        dense_shape=[2, 2])
    features = {"aaa": wire_tensor, "bbb": wire_tensor}
    embeded_sparse = feature_column.embedding_column(crossed, 10)
    output = feature_column_ops.input_from_feature_columns(features,
                                                           [embeded_sparse])
    with self.test_session():
      variables_lib.global_variables_initializer().run()
      self.assertAllEqual(output.eval().shape, [2, 10])

  def testSparseColumnFailsForDNN(self):
    hashed_sparse = feature_column.sparse_column_with_hash_bucket("wire", 10)
    wire_tensor = sparse_tensor.SparseTensor(
        values=["omar", "stringer", "marlo"],
        indices=[[0, 0], [1, 0], [1, 1]],
        dense_shape=[2, 2])
    features = {"wire": wire_tensor}
    with self.test_session():
      with self.assertRaisesRegexp(
          ValueError, "Error creating input layer for column: wire"):
        variables_lib.global_variables_initializer().run()
        feature_column_ops.input_from_feature_columns(features, [hashed_sparse])

  def testWeightedSparseColumnFailsForDNN(self):
    ids = feature_column.sparse_column_with_keys("ids",
                                                 ["marlo", "omar", "stringer"])
    ids_tensor = sparse_tensor.SparseTensor(
        values=["stringer", "stringer", "marlo"],
        indices=[[0, 0], [1, 0], [1, 1]],
        dense_shape=[2, 2])
    weighted_ids = feature_column.weighted_sparse_column(ids, "weights")
    weights_tensor = sparse_tensor.SparseTensor(
        values=[10.0, 20.0, 30.0],
        indices=[[0, 0], [1, 0], [1, 1]],
        dense_shape=[2, 2])
    features = {"ids": ids_tensor, "weights": weights_tensor}
    with self.test_session():
      with self.assertRaisesRegexp(
          ValueError,
          "Error creating input layer for column: ids_weighted_by_weights"):
        data_flow_ops.tables_initializer().run()
        feature_column_ops.input_from_feature_columns(features, [weighted_ids])

  def testCrossedColumnFailsForDNN(self):
    a = feature_column.sparse_column_with_hash_bucket(
        "aaa", hash_bucket_size=100)
    b = feature_column.sparse_column_with_hash_bucket(
        "bbb", hash_bucket_size=100)
    crossed = feature_column.crossed_column(set([a, b]), hash_bucket_size=10000)
    wire_tensor = sparse_tensor.SparseTensor(
        values=["omar", "stringer", "marlo"],
        indices=[[0, 0], [1, 0], [1, 1]],
        dense_shape=[2, 2])
    features = {"aaa": wire_tensor, "bbb": wire_tensor}
    with self.test_session():
      with self.assertRaisesRegexp(
          ValueError, "Error creating input layer for column: aaa_X_bbb"):
        variables_lib.global_variables_initializer().run()
        feature_column_ops.input_from_feature_columns(features, [crossed])

  def testDeepColumnsSucceedForDNN(self):
    real_valued = feature_column.real_valued_column("income", 3)
    bucket = feature_column.bucketized_column(
        feature_column.real_valued_column("price", 2),
        boundaries=[0., 10., 100.])
    hashed_sparse = feature_column.sparse_column_with_hash_bucket("wire", 10)
    features = {
        "income":
            constant_op.constant([[20., 10, -5], [110, 0, -7], [-3, 30, 50]]),
        "price":
            constant_op.constant([[20., 200], [110, 2], [-20, -30]]),
        "wire":
            sparse_tensor.SparseTensor(
                values=["omar", "stringer", "marlo"],
                indices=[[0, 0], [1, 0], [2, 0]],
                dense_shape=[3, 1])
    }
    embeded_sparse = feature_column.embedding_column(
        hashed_sparse, 10, initializer=init_ops.constant_initializer(133.7))
    output = feature_column_ops.input_from_feature_columns(
        features, [real_valued, bucket, embeded_sparse])
    with self.test_session():
      variables_lib.global_variables_initializer().run()
      # size of output = 3 (real_valued) + 2 * 4 (bucket) + 10 (embedding) = 21
      self.assertAllEqual(output.eval().shape, [3, 21])

  def testEmbeddingColumnForDNN(self):
    hashed_sparse = feature_column.sparse_column_with_hash_bucket("wire", 10)
    wire_tensor = sparse_tensor.SparseTensor(
        values=["omar", "stringer", "marlo"],
        indices=[[0, 0], [1, 0], [1, 1]],
        dense_shape=[3, 2])
    features = {"wire": wire_tensor}
    embeded_sparse = feature_column.embedding_column(
        hashed_sparse,
        1,
        combiner="sum",
        initializer=init_ops.ones_initializer())
    output = feature_column_ops.input_from_feature_columns(features,
                                                           [embeded_sparse])
    with self.test_session():
      variables_lib.global_variables_initializer().run()
      # score: (number of values)
      self.assertAllEqual(output.eval(), [[1.], [2.], [0.]])

  def testEmbeddingColumnWithMaxNormForDNN(self):
    hashed_sparse = feature_column.sparse_column_with_hash_bucket("wire", 10)
    wire_tensor = sparse_tensor.SparseTensor(
        values=["omar", "stringer", "marlo"],
        indices=[[0, 0], [1, 0], [1, 1]],
        dense_shape=[3, 2])
    features = {"wire": wire_tensor}
    embedded_sparse = feature_column.embedding_column(
        hashed_sparse,
        1,
        combiner="sum",
        initializer=init_ops.ones_initializer(),
        max_norm=0.5)
    output = feature_column_ops.input_from_feature_columns(features,
                                                           [embedded_sparse])
    with self.test_session():
      variables_lib.global_variables_initializer().run()
      # score: (number of values * 0.5)
      self.assertAllClose(output.eval(), [[0.5], [1.], [0.]])

  def testEmbeddingColumnWithWeightedSparseColumnForDNN(self):
    ids = feature_column.sparse_column_with_keys("ids",
                                                 ["marlo", "omar", "stringer"])
    ids_tensor = sparse_tensor.SparseTensor(
        values=["stringer", "stringer", "marlo"],
        indices=[[0, 0], [1, 0], [1, 1]],
        dense_shape=[3, 2])
    weighted_ids = feature_column.weighted_sparse_column(ids, "weights")
    weights_tensor = sparse_tensor.SparseTensor(
        values=[10.0, 20.0, 30.0],
        indices=[[0, 0], [1, 0], [1, 1]],
        dense_shape=[3, 2])
    features = {"ids": ids_tensor, "weights": weights_tensor}
    embeded_sparse = feature_column.embedding_column(
        weighted_ids,
        1,
        combiner="sum",
        initializer=init_ops.ones_initializer())
    output = feature_column_ops.input_from_feature_columns(features,
                                                           [embeded_sparse])
    with self.test_session():
      variables_lib.global_variables_initializer().run()
      data_flow_ops.tables_initializer().run()
      # score: (sum of weights)
      self.assertAllEqual(output.eval(), [[10.], [50.], [0.]])

  def testInputLayerWithCollectionsForDNN(self):
    real_valued = feature_column.real_valued_column("price")
    bucket = feature_column.bucketized_column(
        real_valued, boundaries=[0., 10., 100.])
    hashed_sparse = feature_column.sparse_column_with_hash_bucket("wire", 10)
    features = {
        "price":
            constant_op.constant([[20.], [110], [-3]]),
        "wire":
            sparse_tensor.SparseTensor(
                values=["omar", "stringer", "marlo"],
                indices=[[0, 0], [1, 0], [2, 0]],
                dense_shape=[3, 1])
    }
    embeded_sparse = feature_column.embedding_column(hashed_sparse, 10)
    feature_column_ops.input_from_feature_columns(
        features, [real_valued, bucket, embeded_sparse],
        weight_collections=["my_collection"])
    weights = ops.get_collection("my_collection")
    # one variable for embeded sparse
    self.assertEqual(1, len(weights))

  def testInputLayerWithTrainableArgForDNN(self):
    real_valued = feature_column.real_valued_column("price")
    bucket = feature_column.bucketized_column(
        real_valued, boundaries=[0., 10., 100.])
    hashed_sparse = feature_column.sparse_column_with_hash_bucket("wire", 10)
    features = {
        "price":
            constant_op.constant([[20.], [110], [-3]]),
        "wire":
            sparse_tensor.SparseTensor(
                values=["omar", "stringer", "marlo"],
                indices=[[0, 0], [1, 0], [2, 0]],
                dense_shape=[3, 1])
    }
    embeded_sparse = feature_column.embedding_column(hashed_sparse, 10)
    feature_column_ops.input_from_feature_columns(
        features, [real_valued, bucket, embeded_sparse],
        weight_collections=["my_collection"],
        trainable=False)
    # There should not be any trainable variables
    self.assertEqual(0, len(variables_lib.trainable_variables()))

    feature_column_ops.input_from_feature_columns(
        features, [real_valued, bucket, embeded_sparse],
        weight_collections=["my_collection"],
        trainable=True)
    # There should  one trainable variable for embeded sparse
    self.assertEqual(1, len(variables_lib.trainable_variables()))

  def testInputLayerWithNonTrainableEmbeddingForDNN(self):
    sparse_1 = feature_column.sparse_column_with_hash_bucket("wire_1", 10)
    sparse_2 = feature_column.sparse_column_with_hash_bucket("wire_2", 10)
    features = {
        "wire_1":
            sparse_tensor.SparseTensor(
                values=["omar", "stringer", "marlo"],
                indices=[[0, 0], [1, 0], [2, 0]],
                dense_shape=[3, 1]),
        "wire_2":
            sparse_tensor.SparseTensor(
                values=["jack", "jill"],
                indices=[[0, 0], [1, 0]],
                dense_shape=[4, 1])
    }
    dims_1 = 10
    init_1 = 3.14
    embeded_1 = feature_column.embedding_column(
        sparse_1, dims_1, initializer=init_ops.constant_initializer(init_1),
        trainable=False)
    output_1 = feature_column_ops.input_from_feature_columns(
        features, [embeded_1])
    # There should be no trainable variables for sparse_1
    self.assertEqual(0, len(variables_lib.trainable_variables()))

    dims_2 = 7
    init_2 = 6.14
    embeded_2 = feature_column.embedding_column(
        sparse_2, dims_2, initializer=init_ops.constant_initializer(init_2),
        trainable=True)
    output_2 = feature_column_ops.input_from_feature_columns(
        features, [embeded_2])
    # There should be one trainable variables for sparse_2
    self.assertEqual(1, len(variables_lib.trainable_variables()))

    with self.test_session():
      variables_lib.global_variables_initializer().run()
      output_1_eval = output_1.eval()
      output_2_eval = output_2.eval()
      self.assertAllEqual(output_1_eval.shape, [3, dims_1])
      self.assertAllClose(output_1_eval, np.tile(init_1, [3, dims_1]))
      self.assertAllEqual(output_2_eval.shape, [4, dims_2])
      self.assertAllClose(output_2_eval, np.concatenate(
          (np.tile(init_2, [2, dims_2]), np.tile(0, [2, dims_2]))))


class SequenceInputFromFeatureColumnTest(test.TestCase):

  def testSupportedColumns(self):
    measurement = feature_column.real_valued_column("measurements")
    country = feature_column.sparse_column_with_hash_bucket("country", 100)
    pets = feature_column.sparse_column_with_hash_bucket("pets", 100)
    ids = feature_column.sparse_column_with_integerized_feature("id", 100)

    country_x_pets = feature_column.crossed_column([country, pets], 100)
    country_x_pets_onehot = feature_column.one_hot_column(country_x_pets)
    bucketized_measurement = feature_column.bucketized_column(measurement,
                                                              [.25, .5, .75])
    embedded_id = feature_column.embedding_column(ids, 100)

    # `_BucketizedColumn` is not supported.
    self.assertRaisesRegexp(
        ValueError,
        "FeatureColumn type _BucketizedColumn is not currently supported",
        feature_column_ops.sequence_input_from_feature_columns, {},
        [measurement, bucketized_measurement])

    # `_CrossedColumn` is not supported.
    self.assertRaisesRegexp(
        ValueError,
        "FeatureColumn type _CrossedColumn is not currently supported",
        feature_column_ops.sequence_input_from_feature_columns, {},
        [embedded_id, country_x_pets])

    # `country_x_pets_onehot` depends on a `_CrossedColumn` which is forbidden.
    self.assertRaisesRegexp(
        ValueError, "Column country_X_pets .* _CrossedColumn",
        feature_column_ops.sequence_input_from_feature_columns, {},
        [embedded_id, country_x_pets_onehot])

  def testRealValuedColumn(self):
    batch_size = 4
    sequence_length = 8
    dimension = 3

    np.random.seed(1111)
    measurement_input = np.random.rand(batch_size, sequence_length, dimension)
    measurement_column = feature_column.real_valued_column("measurements")
    columns_to_tensors = {
        "measurements": constant_op.constant(measurement_input)
    }
    model_input_tensor = feature_column_ops.sequence_input_from_feature_columns(
        columns_to_tensors, [measurement_column])

    with self.test_session() as sess:
      model_inputs = sess.run(model_input_tensor)
    self.assertAllClose(measurement_input, model_inputs)

  def testRealValuedColumnWithExtraDimensions(self):
    batch_size = 4
    sequence_length = 8
    dimensions = [3, 4, 5]

    np.random.seed(2222)
    measurement_input = np.random.rand(batch_size, sequence_length, *dimensions)
    measurement_column = feature_column.real_valued_column("measurements")
    columns_to_tensors = {
        "measurements": constant_op.constant(measurement_input)
    }
    model_input_tensor = feature_column_ops.sequence_input_from_feature_columns(
        columns_to_tensors, [measurement_column])

    expected_shape = [batch_size, sequence_length, np.prod(dimensions)]
    reshaped_measurements = np.reshape(measurement_input, expected_shape)

    with self.test_session() as sess:
      model_inputs = sess.run(model_input_tensor)

    self.assertAllClose(reshaped_measurements, model_inputs)

  def testRealValuedColumnWithNormalizer(self):
    batch_size = 4
    sequence_length = 8
    dimension = 3
    normalizer = lambda x: x - 2

    np.random.seed(3333)
    measurement_input = np.random.rand(batch_size, sequence_length, dimension)
    measurement_column = feature_column.real_valued_column(
        "measurements", normalizer=normalizer)
    columns_to_tensors = {
        "measurements": constant_op.constant(measurement_input)
    }
    model_input_tensor = feature_column_ops.sequence_input_from_feature_columns(
        columns_to_tensors, [measurement_column])

    with self.test_session() as sess:
      model_inputs = sess.run(model_input_tensor)
    self.assertAllClose(normalizer(measurement_input), model_inputs)

  def testRealValuedColumnWithMultiDimensionsAndNormalizer(self):
    batch_size = 4
    sequence_length = 8
    dimensions = [3, 4, 5]
    normalizer = lambda x: x / 2.0

    np.random.seed(1234)
    measurement_input = np.random.rand(batch_size, sequence_length, *dimensions)
    measurement_column = feature_column.real_valued_column(
        "measurements", normalizer=normalizer)
    columns_to_tensors = {
        "measurements": constant_op.constant(measurement_input)
    }
    model_input_tensor = feature_column_ops.sequence_input_from_feature_columns(
        columns_to_tensors, [measurement_column])

    expected_shape = [batch_size, sequence_length, np.prod(dimensions)]
    reshaped_measurements = np.reshape(measurement_input, expected_shape)

    with self.test_session() as sess:
      model_inputs = sess.run(model_input_tensor)

    self.assertAllClose(normalizer(reshaped_measurements), model_inputs)

  def testOneHotColumnFromSparseColumnWithKeys(self):
    ids_tensor = sparse_tensor.SparseTensor(
        values=["c", "b",
                "a", "c", "b",
                "b"],
        indices=[[0, 0, 0], [0, 1, 0],
                 [1, 0, 0], [1, 0, 1], [1, 1, 0],
                 [3, 2, 0]],
        dense_shape=[4, 3, 2])

    ids_column = feature_column.sparse_column_with_keys(
        "ids", ["a", "b", "c", "unseen"])
    one_hot_column = feature_column.one_hot_column(ids_column)
    columns_to_tensors = {"ids": ids_tensor}
    model_input_tensor = feature_column_ops.sequence_input_from_feature_columns(
        columns_to_tensors, [one_hot_column])

    with self.test_session() as sess:
      variables_lib.global_variables_initializer().run()
      data_flow_ops.tables_initializer().run()
      model_input = sess.run(model_input_tensor)

    expected_input_shape = np.array([4, 3, 4])
    expected_model_input = np.array(
        [[[0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 0]],
         [[1, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 0]],
         [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
         [[0, 0, 0, 0], [0, 0, 0, 0], [0, 1, 0, 0]]],
        dtype=np.float32)

    self.assertAllEqual(expected_input_shape, model_input.shape)
    self.assertAllClose(expected_model_input, model_input)

  def testOneHotColumnFromSparseColumnWithHashBucket(self):
    hash_buckets = 10
    ids_tensor = sparse_tensor.SparseTensor(
        values=["c", "b",
                "a", "c", "b",
                "b"],
        indices=[[0, 0, 0], [0, 1, 0],
                 [1, 0, 0], [1, 0, 1], [1, 1, 0],
                 [3, 2, 0]],
        dense_shape=[4, 3, 2])

    hashed_ids_column = feature_column.sparse_column_with_hash_bucket(
        "ids", hash_buckets)
    one_hot_column = feature_column.one_hot_column(hashed_ids_column)
    columns_to_tensors = {"ids": ids_tensor}
    model_input_tensor = feature_column_ops.sequence_input_from_feature_columns(
        columns_to_tensors, [one_hot_column])

    with self.test_session() as sess:
      variables_lib.global_variables_initializer().run()
      data_flow_ops.tables_initializer().run()
      model_input = sess.run(model_input_tensor)

    expected_input_shape = np.array([4, 3, hash_buckets])
    self.assertAllEqual(expected_input_shape, model_input.shape)

  def testEmbeddingColumn(self):
    hash_buckets = 10
    embedding_dimension = 5
    ids_tensor = sparse_tensor.SparseTensor(
        values=["c", "b",
                "a", "c", "b",
                "b"],
        indices=[[0, 0, 0], [0, 1, 0],
                 [1, 0, 0], [1, 0, 1], [1, 1, 0],
                 [3, 2, 0]],
        dense_shape=[4, 3, 2])

    expected_input_shape = np.array([4, 3, embedding_dimension])

    hashed_ids_column = feature_column.sparse_column_with_hash_bucket(
        "ids", hash_buckets)
    embedded_column = feature_column.embedding_column(hashed_ids_column,
                                                      embedding_dimension)
    columns_to_tensors = {"ids": ids_tensor}
    model_input_tensor = feature_column_ops.sequence_input_from_feature_columns(
        columns_to_tensors, [embedded_column])

    with self.test_session() as sess:
      variables_lib.global_variables_initializer().run()
      data_flow_ops.tables_initializer().run()
      model_input = sess.run(model_input_tensor)

    self.assertAllEqual(expected_input_shape, model_input.shape)

  def testEmbeddingColumnGradient(self):
    hash_buckets = 1000
    embedding_dimension = 3
    ids_tensor = sparse_tensor.SparseTensor(
        values=["c", "b",
                "a", "c", "b",
                "b"],
        indices=[[0, 0, 0], [0, 1, 0],
                 [1, 0, 0], [1, 0, 1], [1, 1, 0],
                 [3, 2, 0]],
        dense_shape=[4, 3, 2])

    hashed_ids_column = feature_column.sparse_column_with_hash_bucket(
        "ids", hash_buckets)
    embedded_column = feature_column.embedding_column(
        hashed_ids_column, embedding_dimension, combiner="sum")
    columns_to_tensors = {"ids": ids_tensor}
    model_input_tensor = feature_column_ops.sequence_input_from_feature_columns(
        columns_to_tensors, [embedded_column],
        weight_collections=["my_collection"])
    embedding_weights = ops.get_collection("my_collection")
    gradient_tensor = gradients_impl.gradients(model_input_tensor,
                                               embedding_weights)
    with self.test_session() as sess:
      variables_lib.global_variables_initializer().run()
      data_flow_ops.tables_initializer().run()
      model_input, gradients = sess.run([model_input_tensor, gradient_tensor])

    expected_input_shape = [4, 3, embedding_dimension]
    self.assertAllEqual(expected_input_shape, model_input.shape)

    # `ids_tensor` consists of 7 instances of <empty>, 3 occurences of "b",
    # 2 occurences of "c" and 1 instance of "a".
    expected_gradient_values = sorted([0., 3., 2., 1.] * embedding_dimension)
    actual_gradient_values = np.sort(gradients[0].values, axis=None)
    self.assertAllClose(expected_gradient_values, actual_gradient_values)

  def testMultipleColumns(self):
    batch_size = 4
    sequence_length = 3
    measurement_dimension = 5
    country_hash_size = 10
    max_id = 200
    id_embedding_dimension = 11
    normalizer = lambda x: x / 10.0

    measurement_tensor = random_ops.random_uniform(
        [batch_size, sequence_length, measurement_dimension])
    country_tensor = sparse_tensor.SparseTensor(
        values=["us", "ca",
                "ru", "fr", "ca",
                "mx"],
        indices=[[0, 0, 0], [0, 1, 0],
                 [1, 0, 0], [1, 0, 1], [1, 1, 0],
                 [3, 2, 0]],
        dense_shape=[4, 3, 2])
    id_tensor = sparse_tensor.SparseTensor(
        values=[2, 5,
                26, 123, 1,
                0],
        indices=[[0, 0, 0], [0, 0, 1],
                 [0, 1, 1], [1, 0, 0], [1, 1, 0],
                 [3, 2, 0]],
        dense_shape=[4, 3, 2])

    columns_to_tensors = {
        "measurements": measurement_tensor,
        "country": country_tensor,
        "id": id_tensor
    }

    measurement_column = feature_column.real_valued_column(
        "measurements", normalizer=normalizer)
    country_column = feature_column.sparse_column_with_hash_bucket(
        "country", country_hash_size)
    id_column = feature_column.sparse_column_with_integerized_feature("id",
                                                                      max_id)

    onehot_country_column = feature_column.one_hot_column(country_column)
    embedded_id_column = feature_column.embedding_column(id_column,
                                                         id_embedding_dimension)

    model_input_columns = [
        measurement_column, onehot_country_column, embedded_id_column
    ]

    model_input_tensor = feature_column_ops.sequence_input_from_feature_columns(
        columns_to_tensors, model_input_columns)
    self.assertEqual(dtypes.float32, model_input_tensor.dtype)

    with self.test_session() as sess:
      variables_lib.global_variables_initializer().run()
      data_flow_ops.tables_initializer().run()
      model_input = sess.run(model_input_tensor)

    expected_input_shape = [
        batch_size, sequence_length,
        measurement_dimension + country_hash_size + id_embedding_dimension
    ]
    self.assertAllEqual(expected_input_shape, model_input.shape)


class WeightedSumTest(test.TestCase):

  def testFeatureColumnDictFails(self):
    hashed_sparse = feature_column.sparse_column_with_hash_bucket("wire", 10)
    wire_tensor = sparse_tensor.SparseTensor(
        values=["omar", "stringer", "marlo"],
        indices=[[0, 0], [1, 0], [1, 1]],
        dense_shape=[2, 2])
    features = {"wire": wire_tensor}
    with self.assertRaisesRegexp(
        ValueError,
        "Expected feature_columns to be iterable, found dict"):
      feature_column_ops.weighted_sum_from_feature_columns(
          features, {"feature": hashed_sparse}, num_outputs=5)

  def testSparseColumn(self):
    hashed_sparse = feature_column.sparse_column_with_hash_bucket("wire", 10)
    wire_tensor = sparse_tensor.SparseTensor(
        values=["omar", "stringer", "marlo"],
        indices=[[0, 0], [1, 0], [1, 1]],
        dense_shape=[2, 2])
    features = {"wire": wire_tensor}
    logits, _, _ = feature_column_ops.weighted_sum_from_feature_columns(
        features, [hashed_sparse], num_outputs=5)
    with self.test_session():
      variables_lib.global_variables_initializer().run()
      self.assertAllEqual(logits.eval().shape, [2, 5])

  def testSparseIntColumn(self):
    """Tests a sparse column with int values."""
    hashed_sparse = feature_column.sparse_column_with_hash_bucket(
        "wire", 10, dtype=dtypes.int64)
    wire_tensor = sparse_tensor.SparseTensor(
        values=[101, 201, 301],
        indices=[[0, 0], [1, 0], [1, 1]],
        dense_shape=[2, 2])
    features = {"wire": wire_tensor}
    logits, _, _ = feature_column_ops.weighted_sum_from_feature_columns(
        features, [hashed_sparse], num_outputs=5)
    with self.test_session():
      variables_lib.global_variables_initializer().run()
      self.assertAllEqual(logits.eval().shape, [2, 5])

  def testSparseColumnWithDenseInputTensor(self):
    hashed_sparse = feature_column.sparse_column_with_hash_bucket("wire", 10)
    wire_tensor = constant_op.constant(
        [["omar", "stringer"], ["marlo", "rick"]])
    features = {"wire": wire_tensor}
    logits, _, _ = feature_column_ops.weighted_sum_from_feature_columns(
        features, [hashed_sparse], num_outputs=5)
    with self.test_session():
      variables_lib.global_variables_initializer().run()
      self.assertAllEqual(logits.eval().shape, [2, 5])

  def testWeightedSparseColumn(self):
    ids = feature_column.sparse_column_with_keys("ids",
                                                 ["marlo", "omar", "stringer"])
    ids_tensor = sparse_tensor.SparseTensor(
        values=["stringer", "stringer", "marlo"],
        indices=[[0, 0], [1, 0], [1, 1]],
        dense_shape=[2, 2])
    weighted_ids = feature_column.weighted_sparse_column(ids, "weights")
    weights_tensor = sparse_tensor.SparseTensor(
        values=[10.0, 20.0, 30.0],
        indices=[[0, 0], [1, 0], [1, 1]],
        dense_shape=[2, 2])
    features = {"ids": ids_tensor, "weights": weights_tensor}
    logits, _, _ = feature_column_ops.weighted_sum_from_feature_columns(
        features, [weighted_ids], num_outputs=5)
    with self.test_session():
      variables_lib.global_variables_initializer().run()
      data_flow_ops.tables_initializer().run()
      self.assertAllEqual(logits.eval().shape, [2, 5])

  def testWeightedSparseColumnWithDenseInputTensor(self):
    ids = feature_column.sparse_column_with_keys(
        "ids", ["marlo", "omar", "stringer", "rick"])
    ids_tensor = constant_op.constant([["omar", "stringer"], ["marlo", "rick"]])
    weighted_ids = feature_column.weighted_sparse_column(ids, "weights")
    weights_tensor = constant_op.constant([[10.0, 20.0], [30.0, 40.0]])

    features = {"ids": ids_tensor, "weights": weights_tensor}
    logits, _, _ = feature_column_ops.weighted_sum_from_feature_columns(
        features, [weighted_ids], num_outputs=5)

    with self.test_session():
      variables_lib.global_variables_initializer().run()
      data_flow_ops.tables_initializer().run()
      self.assertAllEqual(logits.eval().shape, [2, 5])

  def testCrossedColumn(self):
    a = feature_column.sparse_column_with_hash_bucket(
        "aaa", hash_bucket_size=100)
    b = feature_column.sparse_column_with_hash_bucket(
        "bbb", hash_bucket_size=100)
    crossed = feature_column.crossed_column(set([a, b]), hash_bucket_size=10000)
    wire_tensor = sparse_tensor.SparseTensor(
        values=["omar", "stringer", "marlo"],
        indices=[[0, 0], [1, 0], [1, 1]],
        dense_shape=[2, 2])
    features = {"aaa": wire_tensor, "bbb": wire_tensor}
    logits, _, _ = feature_column_ops.weighted_sum_from_feature_columns(
        features, [crossed], num_outputs=5)
    with self.test_session():
      variables_lib.global_variables_initializer().run()
      self.assertAllEqual(logits.eval().shape, [2, 5])

  def testEmbeddingColumn(self):
    hashed_sparse = feature_column.sparse_column_with_hash_bucket("wire", 10)
    wire_tensor = sparse_tensor.SparseTensor(
        values=["omar", "stringer", "marlo"],
        indices=[[0, 0], [1, 0], [1, 1]],
        dense_shape=[2, 2])
    features = {"wire": wire_tensor}
    embeded_sparse = feature_column.embedding_column(hashed_sparse, 10)
    with self.test_session():
      with self.assertRaisesRegexp(
          ValueError, "Error creating weighted sum for column: wire_embedding"):
        variables_lib.global_variables_initializer().run()
        feature_column_ops.weighted_sum_from_feature_columns(
            features, [embeded_sparse], num_outputs=5)

  def testSparseFeatureColumnWithVocabularyFile(self):
    vocabulary_file = os.path.join(self.get_temp_dir(), "movies.txt")
    with open(vocabulary_file, "w") as f:
      f.write("\n".join(["head-on", "matrix", "winter sleep"]) + "\n")
    movies = feature_column.sparse_column_with_vocabulary_file(
        column_name="movies", vocabulary_file=vocabulary_file, vocab_size=3)
    with ops.Graph().as_default():
      features = {
          "movies":
              sparse_tensor.SparseTensor(
                  values=["matrix", "head-on", "winter sleep"],
                  indices=[[0, 0], [0, 1], [1, 0]],
                  dense_shape=[2, 2])
      }
      output, column_to_variable, _ = (
          feature_column_ops.weighted_sum_from_feature_columns(
              features, [movies], num_outputs=1))
      with self.test_session() as sess:
        variables_lib.initialize_all_variables().run()
        data_flow_ops.tables_initializer().run()

        weights = column_to_variable[movies][0]
        self.assertEqual(weights.get_shape(), (3, 1))
        sess.run(weights.assign([[0.1], [0.3], [0.5]]))
        # score for first example = 0.3 (matrix) + 0.1 (head-on) = 0.4
        # score for second example = 0.5 (winter sleep)
        self.assertAllClose(output.eval(), [[0.4], [0.5]])

  def testRealValuedColumnWithMultiDimensions(self):
    real_valued = feature_column.real_valued_column("price", 2)
    features = {
        "price": constant_op.constant([[20., 10.], [110, 0.], [-3, 30]])
    }
    logits, _, _ = feature_column_ops.weighted_sum_from_feature_columns(
        features, [real_valued], num_outputs=5)
    with self.test_session():
      variables_lib.global_variables_initializer().run()
      self.assertAllEqual(logits.eval().shape, [3, 5])

  def testBucketizedColumnWithMultiDimensions(self):
    bucket = feature_column.bucketized_column(
        feature_column.real_valued_column("price", 2),
        boundaries=[0., 10., 100.])
    features = {
        "price": constant_op.constant([[20., 10.], [110, 0.], [-3, 30]])
    }
    logits, _, _ = feature_column_ops.weighted_sum_from_feature_columns(
        features, [bucket], num_outputs=5)
    with self.test_session():
      variables_lib.global_variables_initializer().run()
      self.assertAllEqual(logits.eval().shape, [3, 5])

  def testAllWideColumns(self):
    real_valued = feature_column.real_valued_column("income", 2)
    bucket = feature_column.bucketized_column(
        feature_column.real_valued_column("price"), boundaries=[0., 10., 100.])
    hashed_sparse = feature_column.sparse_column_with_hash_bucket("wire", 10)
    crossed = feature_column.crossed_column([bucket, hashed_sparse], 100)
    features = {
        "income":
            constant_op.constant([[20., 10], [110, 0], [-3, 30]]),
        "price":
            constant_op.constant([[20.], [110], [-3]]),
        "wire":
            sparse_tensor.SparseTensor(
                values=["omar", "stringer", "marlo"],
                indices=[[0, 0], [1, 0], [2, 0]],
                dense_shape=[3, 1])
    }
    output, _, _ = feature_column_ops.weighted_sum_from_feature_columns(
        features, [real_valued, bucket, hashed_sparse, crossed], num_outputs=5)
    with self.test_session():
      variables_lib.global_variables_initializer().run()
      self.assertAllEqual(output.eval().shape, [3, 5])

  def testPredictions(self):
    language = feature_column.sparse_column_with_keys(
        column_name="language", keys=["english", "finnish", "hindi"])
    age = feature_column.real_valued_column("age")
    with ops.Graph().as_default():
      features = {
          "age":
              constant_op.constant([[1], [2]]),
          "language":
              sparse_tensor.SparseTensor(
                  values=["hindi", "english"],
                  indices=[[0, 0], [1, 0]],
                  dense_shape=[2, 1]),
      }
      output, column_to_variable, bias = (
          feature_column_ops.weighted_sum_from_feature_columns(
              features, [age, language], num_outputs=1))
      with self.test_session() as sess:
        variables_lib.global_variables_initializer().run()
        data_flow_ops.tables_initializer().run()

        self.assertAllClose(output.eval(), [[0.], [0.]])

        sess.run(bias.assign([0.1]))
        self.assertAllClose(output.eval(), [[0.1], [0.1]])

        # score: 0.1 + age*0.1
        sess.run(column_to_variable[age][0].assign([[0.2]]))
        self.assertAllClose(output.eval(), [[0.3], [0.5]])

        # score: 0.1 + age*0.1 + language_weight[language_index]
        sess.run(column_to_variable[language][0].assign([[0.1], [0.3], [0.2]]))
        self.assertAllClose(output.eval(), [[0.5], [0.6]])

  def testJointPredictions(self):
    country = feature_column.sparse_column_with_keys(
        column_name="country", keys=["us", "finland"])
    language = feature_column.sparse_column_with_keys(
        column_name="language", keys=["english", "finnish", "hindi"])
    with ops.Graph().as_default():
      features = {
          "country":
              sparse_tensor.SparseTensor(
                  values=["finland", "us"],
                  indices=[[0, 0], [1, 0]],
                  dense_shape=[2, 1]),
          "language":
              sparse_tensor.SparseTensor(
                  values=["hindi", "english"],
                  indices=[[0, 0], [1, 0]],
                  dense_shape=[2, 1]),
      }
      output, variables, bias = (
          feature_column_ops.joint_weighted_sum_from_feature_columns(
              features, [country, language], num_outputs=1))
      # Assert that only a single weight is created.
      self.assertEqual(len(variables), 1)
      with self.test_session() as sess:
        variables_lib.global_variables_initializer().run()
        data_flow_ops.tables_initializer().run()

        self.assertAllClose(output.eval(), [[0.], [0.]])

        sess.run(bias.assign([0.1]))
        self.assertAllClose(output.eval(), [[0.1], [0.1]])

        # shape is [5,1] because 1 class and 2 + 3 features.
        self.assertEquals(variables[0].get_shape().as_list(), [5, 1])

        # score: bias + country_weight + language_weight
        sess.run(variables[0].assign([[0.1], [0.2], [0.3], [0.4], [0.5]]))
        self.assertAllClose(output.eval(), [[0.8], [0.5]])

  def testJointPredictionsWeightedFails(self):
    language = feature_column.weighted_sparse_column(
        feature_column.sparse_column_with_keys(
            column_name="language", keys=["english", "finnish", "hindi"]),
        "weight")
    with ops.Graph().as_default():
      features = {
          "weight":
              constant_op.constant([[1], [2]]),
          "language":
              sparse_tensor.SparseTensor(
                  values=["hindi", "english"],
                  indices=[[0, 0], [1, 0]],
                  dense_shape=[2, 1]),
      }
      with self.assertRaises(AssertionError):
        feature_column_ops.joint_weighted_sum_from_feature_columns(
            features, [language], num_outputs=1)

  def testJointPredictionsRealFails(self):
    age = feature_column.real_valued_column("age")
    with ops.Graph().as_default():
      features = {"age": constant_op.constant([[1], [2]]),}
      with self.assertRaises(NotImplementedError):
        feature_column_ops.joint_weighted_sum_from_feature_columns(
            features, [age], num_outputs=1)

  def testPredictionsWithWeightedSparseColumn(self):
    language = feature_column.sparse_column_with_keys(
        column_name="language", keys=["english", "finnish", "hindi"])
    weighted_language = feature_column.weighted_sparse_column(
        sparse_id_column=language, weight_column_name="age")
    with ops.Graph().as_default():
      features = {
          "language":
              sparse_tensor.SparseTensor(
                  values=["hindi", "english"],
                  indices=[[0, 0], [1, 0]],
                  dense_shape=[2, 1]),
          "age":
              sparse_tensor.SparseTensor(
                  values=[10.0, 20.0],
                  indices=[[0, 0], [1, 0]],
                  dense_shape=[2, 1])
      }
      output, column_to_variable, bias = (
          feature_column_ops.weighted_sum_from_feature_columns(
              features, [weighted_language], num_outputs=1))
      with self.test_session() as sess:
        variables_lib.global_variables_initializer().run()
        data_flow_ops.tables_initializer().run()

        self.assertAllClose(output.eval(), [[0.], [0.]])

        sess.run(bias.assign([0.1]))
        self.assertAllClose(output.eval(), [[0.1], [0.1]])

        # score: bias + age*language_weight[index]
        sess.run(column_to_variable[weighted_language][0].assign([[0.1], [0.2],
                                                                  [0.3]]))
        self.assertAllClose(output.eval(), [[3.1], [2.1]])

  def testPredictionsWithMultivalentColumnButNoCross(self):
    language = feature_column.sparse_column_with_keys(
        column_name="language", keys=["english", "turkish", "hindi"])
    with ops.Graph().as_default():
      features = {
          "language":
              sparse_tensor.SparseTensor(
                  values=["hindi", "english"],
                  indices=[[0, 0], [0, 1]],
                  dense_shape=[1, 2])
      }
      output, column_to_variable, bias = (
          feature_column_ops.weighted_sum_from_feature_columns(
              features, [language], num_outputs=1))
      with self.test_session() as sess:
        variables_lib.global_variables_initializer().run()
        data_flow_ops.tables_initializer().run()

        # score: 0.1 + language_weight['hindi'] + language_weight['english']
        sess.run(bias.assign([0.1]))
        sess.run(column_to_variable[language][0].assign([[0.1], [0.3], [0.2]]))
        self.assertAllClose(output.eval(), [[0.4]])

  def testSparseFeatureColumnWithHashedBucketSize(self):
    movies = feature_column.sparse_column_with_hash_bucket(
        column_name="movies", hash_bucket_size=15)
    with ops.Graph().as_default():
      features = {
          "movies":
              sparse_tensor.SparseTensor(
                  values=["matrix", "head-on", "winter sleep"],
                  indices=[[0, 0], [0, 1], [1, 0]],
                  dense_shape=[2, 2])
      }
      output, column_to_variable, _ = (
          feature_column_ops.weighted_sum_from_feature_columns(
              features, [movies], num_outputs=1))
      with self.test_session() as sess:
        variables_lib.global_variables_initializer().run()
        data_flow_ops.tables_initializer().run()

        weights = column_to_variable[movies][0]
        self.assertEqual(weights.get_shape(), (15, 1))
        sess.run(weights.assign(weights + 0.4))
        # score for first example = 0.4 (matrix) + 0.4 (head-on) = 0.8
        # score for second example = 0.4 (winter sleep)
        self.assertAllClose(output.eval(), [[0.8], [0.4]])

  def testCrossUsageInPredictions(self):
    language = feature_column.sparse_column_with_hash_bucket(
        "language", hash_bucket_size=3)
    country = feature_column.sparse_column_with_hash_bucket(
        "country", hash_bucket_size=5)
    country_language = feature_column.crossed_column(
        [language, country], hash_bucket_size=10)
    with ops.Graph().as_default():
      features = {
          "language":
              sparse_tensor.SparseTensor(
                  values=["english", "spanish"],
                  indices=[[0, 0], [1, 0]],
                  dense_shape=[2, 1]),
          "country":
              sparse_tensor.SparseTensor(
                  values=["US", "SV"],
                  indices=[[0, 0], [1, 0]],
                  dense_shape=[2, 1])
      }
      output, column_to_variable, _ = (
          feature_column_ops.weighted_sum_from_feature_columns(
              features, [country_language], num_outputs=1))
      with self.test_session() as sess:
        variables_lib.global_variables_initializer().run()
        data_flow_ops.tables_initializer().run()

        weights = column_to_variable[country_language][0]
        sess.run(weights.assign(weights + 0.4))
        self.assertAllClose(output.eval(), [[0.4], [0.4]])

  def testCrossColumnByItself(self):
    language = feature_column.sparse_column_with_hash_bucket(
        "language", hash_bucket_size=3)
    language_language = feature_column.crossed_column(
        [language, language], hash_bucket_size=10)
    with ops.Graph().as_default():
      features = {
          "language":
              sparse_tensor.SparseTensor(
                  values=["english", "spanish"],
                  indices=[[0, 0], [0, 1]],
                  dense_shape=[1, 2]),
      }
      output, column_to_variable, _ = (
          feature_column_ops.weighted_sum_from_feature_columns(
              features, [language_language], num_outputs=1))
      with self.test_session() as sess:
        variables_lib.global_variables_initializer().run()
        data_flow_ops.tables_initializer().run()

        weights = column_to_variable[language_language][0]
        sess.run(weights.assign(weights + 0.4))
        # There are two features inside language. If we cross it by itself we'll
        # have four crossed features.
        self.assertAllClose(output.eval(), [[1.6]])

  def testMultivalentCrossUsageInPredictions(self):
    language = feature_column.sparse_column_with_hash_bucket(
        "language", hash_bucket_size=3)
    country = feature_column.sparse_column_with_hash_bucket(
        "country", hash_bucket_size=5)
    country_language = feature_column.crossed_column(
        [language, country], hash_bucket_size=10)
    with ops.Graph().as_default():
      features = {
          "language":
              sparse_tensor.SparseTensor(
                  values=["english", "spanish"],
                  indices=[[0, 0], [0, 1]],
                  dense_shape=[1, 2]),
          "country":
              sparse_tensor.SparseTensor(
                  values=["US", "SV"],
                  indices=[[0, 0], [0, 1]],
                  dense_shape=[1, 2])
      }
      output, column_to_variable, _ = (
          feature_column_ops.weighted_sum_from_feature_columns(
              features, [country_language], num_outputs=1))
      with self.test_session() as sess:
        variables_lib.global_variables_initializer().run()
        data_flow_ops.tables_initializer().run()

        weights = column_to_variable[country_language][0]
        sess.run(weights.assign(weights + 0.4))
        # There are four crosses each with 0.4 weight.
        # score = 0.4 + 0.4 + 0.4 + 0.4
        self.assertAllClose(output.eval(), [[1.6]])

  def testMultivalentCrossUsageInPredictionsWithPartition(self):
    # bucket size has to be big enough to allow sharding.
    language = feature_column.sparse_column_with_hash_bucket(
        "language", hash_bucket_size=64 << 19)
    country = feature_column.sparse_column_with_hash_bucket(
        "country", hash_bucket_size=64 << 18)
    country_language = feature_column.crossed_column(
        [language, country], hash_bucket_size=64 << 18)
    with ops.Graph().as_default():
      features = {
          "language":
              sparse_tensor.SparseTensor(
                  values=["english", "spanish"],
                  indices=[[0, 0], [0, 1]],
                  dense_shape=[1, 2]),
          "country":
              sparse_tensor.SparseTensor(
                  values=["US", "SV"],
                  indices=[[0, 0], [0, 1]],
                  dense_shape=[1, 2])
      }
      with variable_scope.variable_scope(
          "weighted_sum_from_feature_columns",
          features.values(),
          partitioner=partitioned_variables.min_max_variable_partitioner(
              max_partitions=10, min_slice_size=((64 << 20) - 1))) as scope:
        output, column_to_variable, _ = (
            feature_column_ops.weighted_sum_from_feature_columns(
                features, [country, language, country_language],
                num_outputs=1,
                scope=scope))
      with self.test_session() as sess:
        variables_lib.global_variables_initializer().run()
        data_flow_ops.tables_initializer().run()

        self.assertEqual(2, len(column_to_variable[country]))
        self.assertEqual(3, len(column_to_variable[language]))
        self.assertEqual(2, len(column_to_variable[country_language]))

        weights = column_to_variable[country_language]
        for partition_variable in weights:
          sess.run(partition_variable.assign(partition_variable + 0.4))
        # There are four crosses each with 0.4 weight.
        # score = 0.4 + 0.4 + 0.4 + 0.4
        self.assertAllClose(output.eval(), [[1.6]])

  def testRealValuedColumnHavingMultiDimensions(self):
    country = feature_column.sparse_column_with_hash_bucket(
        "country", hash_bucket_size=5)
    age = feature_column.real_valued_column("age")
    # The following RealValuedColumn has 3 dimensions.
    incomes = feature_column.real_valued_column("incomes", 3)

    with ops.Graph().as_default():
      features = {
          "age":
              constant_op.constant([[1], [1]]),
          "incomes":
              constant_op.constant([[100., 200., 300.], [10., 20., 30.]]),
          "country":
              sparse_tensor.SparseTensor(
                  values=["US", "SV"],
                  indices=[[0, 0], [1, 0]],
                  dense_shape=[2, 2])
      }
      output, column_to_variable, _ = (
          feature_column_ops.weighted_sum_from_feature_columns(
              features, [country, age, incomes], num_outputs=1))
      with self.test_session() as sess:
        variables_lib.global_variables_initializer().run()
        data_flow_ops.tables_initializer().run()

        incomes_weights = column_to_variable[incomes][0]
        sess.run(incomes_weights.assign([[0.1], [0.2], [0.3]]))
        self.assertAllClose(output.eval(), [[140.], [14.]])

  def testMulticlassWithRealValuedColumnHavingMultiDimensionsAndSparse(self):
    country = feature_column.sparse_column_with_hash_bucket(
        "country", hash_bucket_size=5)
    age = feature_column.real_valued_column("age")
    # The following RealValuedColumn has no predefined dimension so it
    # can be missing.
    height = feature_column.real_valued_column("height", dimension=None)
    # The following RealValuedColumn has 3 dimensions.
    incomes = feature_column.real_valued_column("incomes", 3)
    with ops.Graph().as_default():
      features = {
          "age":
              constant_op.constant([[1], [1]]),
          "incomes":
              constant_op.constant([[100., 200., 300.], [10., 20., 30.]]),
          "height":
              sparse_tensor.SparseTensor(
                  values=[5.0, 4.0, 6.0],
                  indices=[[0, 0], [0, 1], [1, 1]],
                  dense_shape=[2, 2]),
          "country":
              sparse_tensor.SparseTensor(
                  values=["US", "SV"],
                  indices=[[0, 0], [1, 0]],
                  dense_shape=[2, 2])
      }
      output, column_to_variable, _ = (
          feature_column_ops.weighted_sum_from_feature_columns(
              features, [country, age, height, incomes], num_outputs=5))
      with self.test_session() as sess:
        variables_lib.global_variables_initializer().run()
        data_flow_ops.tables_initializer().run()

        height_weights = column_to_variable[height][0]
        sess.run(
            height_weights.assign(
                [[1., 2., 3., 5., 10.], [1., 2., 3., 5., 10.]]))
        self.assertAllClose(output.eval(), [[9., 18., 27., 45., 90.],
                                            [6., 12., 18., 30., 60.]])

        incomes_weights = column_to_variable[incomes][0]
        sess.run(
            incomes_weights.assign([[0.01, 0.1, 1., 10., 100.],
                                    [0.02, 0.2, 2., 20., 200.],
                                    [0.03, 0.3, 3., 30., 300.]]))
        self.assertAllClose(
            output.eval(),
            [[14. + 9., 140. + 18., 1400. + 27., 14000. + 45., 140000. + 90.],
             [1.4 + 6., 14. + 12., 140. + 18., 1400. + 30., 14000. + 60.]])

  def testBucketizedColumn(self):
    bucket = feature_column.bucketized_column(
        feature_column.real_valued_column("price"), boundaries=[0., 10., 100.])
    with ops.Graph().as_default():
      # buckets 2, 3, 0
      features = {"price": constant_op.constant([[20.], [110], [-3]])}
      output, column_to_variable, _ = (
          feature_column_ops.weighted_sum_from_feature_columns(
              features, [bucket], num_outputs=1))
      with self.test_session() as sess:
        variables_lib.global_variables_initializer().run()
        data_flow_ops.tables_initializer().run()

        sess.run(column_to_variable[bucket][0].assign([[0.1], [0.2], [0.3],
                                                       [0.4]]))
        self.assertAllClose(output.eval(), [[0.3], [0.4], [0.1]])

  def testBucketizedColumnHavingMultiDimensions(self):
    country = feature_column.sparse_column_with_hash_bucket(
        "country", hash_bucket_size=5)
    bucket = feature_column.bucketized_column(
        feature_column.real_valued_column("price", 2),
        boundaries=[0., 10., 100.])
    with ops.Graph().as_default():
      # buckets 2, 3, 0
      features = {
          "price":
              constant_op.constant([[20., 210], [110, 50], [-3, -30]]),
          "country":
              sparse_tensor.SparseTensor(
                  values=["US", "SV"],
                  indices=[[0, 0], [1, 0]],
                  dense_shape=[3, 2])
      }
      output, column_to_variable, _ = (
          feature_column_ops.weighted_sum_from_feature_columns(
              features, [bucket, country], num_outputs=1))
      with self.test_session() as sess:
        variables_lib.global_variables_initializer().run()
        data_flow_ops.tables_initializer().run()

        # dimension = 2, bucket_size = 4, num_classes = 1
        sess.run(column_to_variable[bucket][0].assign(
            [[0.1], [0.2], [0.3], [0.4], [1], [2], [3], [4]]))
        self.assertAllClose(output.eval(), [[0.3 + 4], [0.4 + 3], [0.1 + 1]])

  def testMulticlassWithBucketizedColumnHavingMultiDimensions(self):
    country = feature_column.sparse_column_with_hash_bucket(
        "country", hash_bucket_size=5)
    bucket = feature_column.bucketized_column(
        feature_column.real_valued_column("price", 2),
        boundaries=[0., 10., 100.])
    with ops.Graph().as_default():
      # buckets 2, 3, 0
      features = {
          "price":
              constant_op.constant([[20., 210], [110, 50], [-3, -30]]),
          "country":
              sparse_tensor.SparseTensor(
                  values=["US", "SV"],
                  indices=[[0, 0], [1, 0]],
                  dense_shape=[3, 2])
      }
      output, column_to_variable, _ = (
          feature_column_ops.weighted_sum_from_feature_columns(
              features, [bucket, country], num_outputs=5))
      with self.test_session() as sess:
        variables_lib.global_variables_initializer().run()
        data_flow_ops.tables_initializer().run()

        # dimension = 2, bucket_size = 4, num_classes = 5
        sess.run(column_to_variable[bucket][0].assign(
            [[0.1, 1, 10, 100, 1000], [0.2, 2, 20, 200, 2000],
             [0.3, 3, 30, 300, 3000], [0.4, 4, 40, 400, 4000],
             [5, 50, 500, 5000, 50000], [6, 60, 600, 6000, 60000],
             [7, 70, 700, 7000, 70000], [8, 80, 800, 8000, 80000]]))
        self.assertAllClose(
            output.eval(),
            [[0.3 + 8, 3 + 80, 30 + 800, 300 + 8000, 3000 + 80000],
             [0.4 + 7, 4 + 70, 40 + 700, 400 + 7000, 4000 + 70000],
             [0.1 + 5, 1 + 50, 10 + 500, 100 + 5000, 1000 + 50000]])

  def testCrossWithBucketizedColumn(self):
    price_bucket = feature_column.bucketized_column(
        feature_column.real_valued_column("price"), boundaries=[0., 10., 100.])
    country = feature_column.sparse_column_with_hash_bucket(
        "country", hash_bucket_size=5)
    country_price = feature_column.crossed_column(
        [country, price_bucket], hash_bucket_size=10)
    with ops.Graph().as_default():
      features = {
          "price":
              constant_op.constant([[20.]]),
          "country":
              sparse_tensor.SparseTensor(
                  values=["US", "SV"],
                  indices=[[0, 0], [0, 1]],
                  dense_shape=[1, 2])
      }
      output, column_to_variable, _ = (
          feature_column_ops.weighted_sum_from_feature_columns(
              features, [country_price], num_outputs=1))
      with self.test_session() as sess:
        variables_lib.global_variables_initializer().run()
        data_flow_ops.tables_initializer().run()

        weights = column_to_variable[country_price][0]
        sess.run(weights.assign(weights + 0.4))
        # There are two crosses each with 0.4 weight.
        # score = 0.4 + 0.4
        self.assertAllClose(output.eval(), [[0.8]])

  def testCrossWithCrossedColumn(self):
    price_bucket = feature_column.bucketized_column(
        feature_column.real_valued_column("price"), boundaries=[0., 10., 100.])
    language = feature_column.sparse_column_with_hash_bucket(
        "language", hash_bucket_size=3)
    country = feature_column.sparse_column_with_hash_bucket(
        "country", hash_bucket_size=5)
    country_language = feature_column.crossed_column(
        [language, country], hash_bucket_size=10)
    country_language_price = feature_column.crossed_column(
        set([country_language, price_bucket]), hash_bucket_size=15)
    with ops.Graph().as_default():
      features = {
          "price":
              constant_op.constant([[20.]]),
          "country":
              sparse_tensor.SparseTensor(
                  values=["US", "SV"],
                  indices=[[0, 0], [0, 1]],
                  dense_shape=[1, 2]),
          "language":
              sparse_tensor.SparseTensor(
                  values=["english", "spanish"],
                  indices=[[0, 0], [0, 1]],
                  dense_shape=[1, 2])
      }
      output, column_to_variable, _ = (
          feature_column_ops.weighted_sum_from_feature_columns(
              features, [country_language_price], num_outputs=1))
      with self.test_session() as sess:
        variables_lib.global_variables_initializer().run()
        data_flow_ops.tables_initializer().run()

        weights = column_to_variable[country_language_price][0]
        sess.run(weights.assign(weights + 0.4))
        # There are two crosses each with 0.4 weight.
        # score = 0.4 + 0.4 + 0.4 + 0.4
        self.assertAllClose(output.eval(), [[1.6]])

  def testIntegerizedColumn(self):
    product = feature_column.sparse_column_with_integerized_feature(
        "product", bucket_size=5)
    with ops.Graph().as_default():
      features = {
          "product":
              sparse_tensor.SparseTensor(
                  values=[0, 4, 2],
                  indices=[[0, 0], [1, 0], [2, 0]],
                  dense_shape=[3, 1])
      }
      output, column_to_variable, _ = (
          feature_column_ops.weighted_sum_from_feature_columns(
              features, [product], num_outputs=1))
      with self.test_session() as sess:
        variables_lib.global_variables_initializer().run()
        data_flow_ops.tables_initializer().run()
        product_weights = column_to_variable[product][0]
        sess.run(product_weights.assign([[0.1], [0.2], [0.3], [0.4], [0.5]]))
        self.assertAllClose(output.eval(), [[0.1], [0.5], [0.3]])

  def testIntegerizedColumnWithDenseInputTensor(self):
    product = feature_column.sparse_column_with_integerized_feature(
        "product", bucket_size=5)
    with ops.Graph().as_default():
      features = {"product": constant_op.constant([[0], [4], [2]])}
      output, column_to_variable, _ = (
          feature_column_ops.weighted_sum_from_feature_columns(
              features, [product], num_outputs=1))
      with self.test_session() as sess:
        variables_lib.global_variables_initializer().run()
        data_flow_ops.tables_initializer().run()
        product_weights = column_to_variable[product][0]
        sess.run(product_weights.assign([[0.1], [0.2], [0.3], [0.4], [0.5]]))
        self.assertAllClose(output.eval(), [[0.1], [0.5], [0.3]])

  def testIntegerizedColumnWithDenseInputTensor2(self):
    product = feature_column.sparse_column_with_integerized_feature(
        "product", bucket_size=5)
    with ops.Graph().as_default():
      features = {"product": constant_op.constant([[0, 4], [2, 3]])}
      output, column_to_variable, _ = (
          feature_column_ops.weighted_sum_from_feature_columns(
              features, [product], num_outputs=1))
      with self.test_session() as sess:
        variables_lib.global_variables_initializer().run()
        data_flow_ops.tables_initializer().run()
        product_weights = column_to_variable[product][0]
        sess.run(product_weights.assign([[0.1], [0.2], [0.3], [0.4], [0.5]]))
        self.assertAllClose(output.eval(), [[0.6], [0.7]])

  def testIntegerizedColumnWithInvalidId(self):
    product = feature_column.sparse_column_with_integerized_feature(
        "product", bucket_size=5)
    with ops.Graph().as_default():
      features = {
          "product":
              sparse_tensor.SparseTensor(
                  values=[5, 4, 7],
                  indices=[[0, 0], [1, 0], [2, 0]],
                  dense_shape=[3, 1])
      }
      output, column_to_variable, _ = (
          feature_column_ops.weighted_sum_from_feature_columns(
              features, [product], num_outputs=1))
      with self.test_session() as sess:
        variables_lib.global_variables_initializer().run()
        data_flow_ops.tables_initializer().run()
        product_weights = column_to_variable[product][0]
        sess.run(product_weights.assign([[0.1], [0.2], [0.3], [0.4], [0.5]]))
        self.assertAllClose(output.eval(), [[0.1], [0.5], [0.3]])

  def testMulticlassWithOnlyBias(self):
    with ops.Graph().as_default():
      features = {"age": constant_op.constant([[10.], [20.], [30.], [40.]])}
      output, _, bias = feature_column_ops.weighted_sum_from_feature_columns(
          features, [feature_column.real_valued_column("age")], num_outputs=3)
      with self.test_session() as sess:
        variables_lib.global_variables_initializer().run()
        data_flow_ops.tables_initializer().run()
        sess.run(bias.assign([0.1, 0.2, 0.3]))
        self.assertAllClose(output.eval(), [[0.1, 0.2, 0.3], [0.1, 0.2, 0.3],
                                            [0.1, 0.2, 0.3], [0.1, 0.2, 0.3]])

  def testMulticlassWithRealValuedColumn(self):
    with ops.Graph().as_default():
      column = feature_column.real_valued_column("age")
      features = {"age": constant_op.constant([[10.], [20.], [30.], [40.]])}
      output, column_to_variable, _ = (
          feature_column_ops.weighted_sum_from_feature_columns(
              features, [column], num_outputs=3))
      with self.test_session() as sess:
        variables_lib.global_variables_initializer().run()
        data_flow_ops.tables_initializer().run()
        weights = column_to_variable[column][0]
        self.assertEqual(weights.get_shape(), (1, 3))
        sess.run(weights.assign([[0.01, 0.03, 0.05]]))
        self.assertAllClose(output.eval(), [[0.1, 0.3, 0.5], [0.2, 0.6, 1.0],
                                            [0.3, 0.9, 1.5], [0.4, 1.2, 2.0]])

  def testMulticlassWithSparseColumn(self):
    with ops.Graph().as_default():
      column = feature_column.sparse_column_with_keys(
          column_name="language",
          keys=["english", "arabic", "hindi", "russian", "swahili"])
      features = {
          "language":
              sparse_tensor.SparseTensor(
                  values=["hindi", "english", "arabic", "russian"],
                  indices=[[0, 0], [1, 0], [2, 0], [3, 0]],
                  dense_shape=[4, 1])
      }
      output, column_to_variable, _ = (
          feature_column_ops.weighted_sum_from_feature_columns(
              features, [column], num_outputs=3))
      with self.test_session() as sess:
        variables_lib.global_variables_initializer().run()
        data_flow_ops.tables_initializer().run()
        weights = column_to_variable[column][0]
        self.assertEqual(weights.get_shape(), (5, 3))
        sess.run(
            weights.assign([[0.1, 0.4, 0.7],
                            [0.2, 0.5, 0.8],
                            [0.3, 0.6, 0.9],
                            [0.4, 0.7, 1.0],
                            [0.5, 0.8, 1.1]]))
        self.assertAllClose(output.eval(), [[0.3, 0.6, 0.9],
                                            [0.1, 0.4, 0.7],
                                            [0.2, 0.5, 0.8],
                                            [0.4, 0.7, 1.0]])

  def testMulticlassWithBucketizedColumn(self):
    column = feature_column.bucketized_column(
        feature_column.real_valued_column("price"),
        boundaries=[0., 100., 500., 1000.])
    with ops.Graph().as_default():
      # buckets 0, 2, 1, 2
      features = {"price": constant_op.constant([[-3], [110], [20.], [210]])}
      output, column_to_variable, _ = (
          feature_column_ops.weighted_sum_from_feature_columns(
              features, [column], num_outputs=3))
      with self.test_session() as sess:
        variables_lib.global_variables_initializer().run()
        data_flow_ops.tables_initializer().run()

        weights = column_to_variable[column][0]
        self.assertEqual(weights.get_shape(), (5, 3))
        sess.run(
            weights.assign([[0.1, 0.4, 0.7],
                            [0.2, 0.5, 0.8],
                            [0.3, 0.6, 0.9],
                            [0.4, 0.7, 1.0],
                            [0.5, 0.8, 1.1]]))
        self.assertAllClose(output.eval(), [[0.1, 0.4, 0.7],
                                            [0.3, 0.6, 0.9],
                                            [0.2, 0.5, 0.8],
                                            [0.3, 0.6, 0.9]])

  def testMulticlassWithCrossedColumn(self):
    language = feature_column.sparse_column_with_hash_bucket(
        "language", hash_bucket_size=3)
    country = feature_column.sparse_column_with_hash_bucket(
        "country", hash_bucket_size=2)
    column = feature_column.crossed_column(
        {language, country}, hash_bucket_size=5)
    with ops.Graph().as_default():
      features = {
          "language":
              sparse_tensor.SparseTensor(
                  values=["english", "spanish", "russian", "swahili"],
                  indices=[[0, 0], [1, 0], [2, 0], [3, 0]],
                  dense_shape=[4, 1]),
          "country":
              sparse_tensor.SparseTensor(
                  values=["US", "SV", "RU", "KE"],
                  indices=[[0, 0], [1, 0], [2, 0], [3, 0]],
                  dense_shape=[4, 1])
      }
      output, column_to_variable, _ = (
          feature_column_ops.weighted_sum_from_feature_columns(
              features, [column], num_outputs=3))
      with self.test_session() as sess:
        variables_lib.global_variables_initializer().run()
        data_flow_ops.tables_initializer().run()

        weights = column_to_variable[column][0]
        self.assertEqual(weights.get_shape(), (5, 3))
        sess.run(
            weights.assign([[0.1, 0.4, 0.7],
                            [0.2, 0.5, 0.8],
                            [0.3, 0.6, 0.9],
                            [0.4, 0.7, 1.0],
                            [0.5, 0.8, 1.1]]))
        self.assertAllClose(array_ops.shape(output).eval(), [4, 3])

  def testMulticlassWithMultivalentColumn(self):
    column = feature_column.sparse_column_with_keys(
        column_name="language",
        keys=["english", "turkish", "hindi", "russian", "swahili"])
    with ops.Graph().as_default():
      features = {
          "language":
              sparse_tensor.SparseTensor(
                  values=["hindi", "english", "turkish", "turkish", "english"],
                  indices=[[0, 0], [0, 1], [1, 0], [2, 0], [3, 0]],
                  dense_shape=[4, 2])
      }
      output, column_to_variable, _ = (
          feature_column_ops.weighted_sum_from_feature_columns(
              features, [column], num_outputs=3))
      with self.test_session() as sess:
        variables_lib.global_variables_initializer().run()
        data_flow_ops.tables_initializer().run()

        weights = column_to_variable[column][0]
        self.assertEqual(weights.get_shape(), (5, 3))
        sess.run(
            weights.assign([[0.1, 0.4, 0.7],
                            [0.2, 0.5, 0.8],
                            [0.3, 0.6, 0.9],
                            [0.4, 0.7, 1.0],
                            [0.5, 0.8, 1.1]]))
        self.assertAllClose(output.eval(), [[0.4, 1.0, 1.6],
                                            [0.2, 0.5, 0.8],
                                            [0.2, 0.5, 0.8],
                                            [0.1, 0.4, 0.7]])

  def testVariablesAddedToCollection(self):
    price_bucket = feature_column.bucketized_column(
        feature_column.real_valued_column("price"), boundaries=[0., 10., 100.])
    country = feature_column.sparse_column_with_hash_bucket(
        "country", hash_bucket_size=5)
    country_price = feature_column.crossed_column(
        [country, price_bucket], hash_bucket_size=10)
    with ops.Graph().as_default():
      features = {
          "price":
              constant_op.constant([[20.]]),
          "country":
              sparse_tensor.SparseTensor(
                  values=["US", "SV"],
                  indices=[[0, 0], [0, 1]],
                  dense_shape=[1, 2])
      }
      feature_column_ops.weighted_sum_from_feature_columns(
          features, [country_price, price_bucket],
          num_outputs=1,
          weight_collections=["my_collection"])
      weights = ops.get_collection("my_collection")
      # 3 = bias + price_bucket + country_price
      self.assertEqual(3, len(weights))


class ParseExampleTest(test.TestCase):

  def testParseExample(self):
    bucket = feature_column.bucketized_column(
        feature_column.real_valued_column(
            "price", dimension=3),
        boundaries=[0., 10., 100.])
    wire_cast = feature_column.sparse_column_with_keys(
        "wire_cast", ["marlo", "omar", "stringer"])
    # buckets 2, 3, 0
    data = example_pb2.Example(features=feature_pb2.Features(feature={
        "price":
            feature_pb2.Feature(float_list=feature_pb2.FloatList(
                value=[20., 110, -3])),
        "wire_cast":
            feature_pb2.Feature(bytes_list=feature_pb2.BytesList(
                value=[b"stringer", b"marlo"])),
    }))
    output = feature_column_ops.parse_feature_columns_from_examples(
        serialized=[data.SerializeToString()],
        feature_columns=[bucket, wire_cast])
    self.assertIn(bucket, output)
    self.assertIn(wire_cast, output)
    with self.test_session():
      data_flow_ops.tables_initializer().run()
      self.assertAllEqual(output[bucket].eval(), [[2, 3, 0]])
      self.assertAllEqual(output[wire_cast].indices.eval(), [[0, 0], [0, 1]])
      self.assertAllEqual(output[wire_cast].values.eval(), [2, 0])

  def testParseSequenceExample(self):
    location_keys = ["east_side", "west_side", "nyc"]
    embedding_dimension = 10

    location = feature_column.sparse_column_with_keys(
        "location", keys=location_keys)
    location_onehot = feature_column.one_hot_column(location)
    wire_cast = feature_column.sparse_column_with_keys(
        "wire_cast", ["marlo", "omar", "stringer"])
    wire_cast_embedded = feature_column.embedding_column(
        wire_cast, dimension=embedding_dimension)
    measurements = feature_column.real_valued_column(
        "measurements", dimension=2)

    context_feature_columns = [location_onehot]
    sequence_feature_columns = [wire_cast_embedded, measurements]

    sequence_example = example_pb2.SequenceExample(
        context=feature_pb2.Features(feature={
            "location":
                feature_pb2.Feature(bytes_list=feature_pb2.BytesList(
                    value=[b"west_side"])),
        }),
        feature_lists=feature_pb2.FeatureLists(feature_list={
            "wire_cast":
                feature_pb2.FeatureList(feature=[
                    feature_pb2.Feature(bytes_list=feature_pb2.BytesList(
                        value=[b"marlo", b"stringer"])),
                    feature_pb2.Feature(bytes_list=feature_pb2.BytesList(
                        value=[b"omar", b"stringer", b"marlo"])),
                    feature_pb2.Feature(bytes_list=feature_pb2.BytesList(
                        value=[b"marlo"])),
                ]),
            "measurements":
                feature_pb2.FeatureList(feature=[
                    feature_pb2.Feature(float_list=feature_pb2.FloatList(
                        value=[0.2, 0.3])),
                    feature_pb2.Feature(float_list=feature_pb2.FloatList(
                        value=[0.1, 0.8])),
                    feature_pb2.Feature(float_list=feature_pb2.FloatList(
                        value=[0.5, 0.0])),
                ])
        }))

    ctx, seq = feature_column_ops.parse_feature_columns_from_sequence_examples(
        serialized=sequence_example.SerializeToString(),
        context_feature_columns=context_feature_columns,
        sequence_feature_columns=sequence_feature_columns)

    self.assertIn("location", ctx)
    self.assertIsInstance(ctx["location"], sparse_tensor.SparseTensor)
    self.assertIn("wire_cast", seq)
    self.assertIsInstance(seq["wire_cast"], sparse_tensor.SparseTensor)
    self.assertIn("measurements", seq)
    self.assertIsInstance(seq["measurements"], ops.Tensor)

    with self.test_session() as sess:
      location_val, wire_cast_val, measurement_val = sess.run(
          [ctx["location"], seq["wire_cast"], seq["measurements"]])

    self.assertAllEqual(location_val.indices, np.array([[0]]))
    self.assertAllEqual(location_val.values, np.array([b"west_side"]))
    self.assertAllEqual(location_val.dense_shape, np.array([1]))

    self.assertAllEqual(wire_cast_val.indices,
                        np.array(
                            [[0, 0], [0, 1], [1, 0], [1, 1], [1, 2], [2, 0]]))
    self.assertAllEqual(
        wire_cast_val.values,
        np.array(
            [b"marlo", b"stringer", b"omar", b"stringer", b"marlo", b"marlo"]))
    self.assertAllEqual(wire_cast_val.dense_shape, np.array([3, 3]))

    self.assertAllClose(measurement_val,
                        np.array([[0.2, 0.3], [0.1, 0.8], [0.5, 0.0]]))


class InferRealValuedColumnTest(test.TestCase):

  def testTensorInt32(self):
    self.assertEqual(
        feature_column_ops.infer_real_valued_columns(
            array_ops.zeros(
                shape=[33, 4], dtype=dtypes.int32)), [
                    feature_column.real_valued_column(
                        "", dimension=4, dtype=dtypes.int32)
                ])

  def testTensorInt64(self):
    self.assertEqual(
        feature_column_ops.infer_real_valued_columns(
            array_ops.zeros(
                shape=[33, 4], dtype=dtypes.int64)), [
                    feature_column.real_valued_column(
                        "", dimension=4, dtype=dtypes.int64)
                ])

  def testTensorFloat32(self):
    self.assertEqual(
        feature_column_ops.infer_real_valued_columns(
            array_ops.zeros(
                shape=[33, 4], dtype=dtypes.float32)), [
                    feature_column.real_valued_column(
                        "", dimension=4, dtype=dtypes.float32)
                ])

  def testTensorFloat64(self):
    self.assertEqual(
        feature_column_ops.infer_real_valued_columns(
            array_ops.zeros(
                shape=[33, 4], dtype=dtypes.float64)), [
                    feature_column.real_valued_column(
                        "", dimension=4, dtype=dtypes.float64)
                ])

  def testDictionary(self):
    self.assertItemsEqual(
        feature_column_ops.infer_real_valued_columns({
            "a": array_ops.zeros(
                shape=[33, 4], dtype=dtypes.int32),
            "b": array_ops.zeros(
                shape=[3, 2], dtype=dtypes.float32)
        }), [
            feature_column.real_valued_column(
                "a", dimension=4, dtype=dtypes.int32),
            feature_column.real_valued_column(
                "b", dimension=2, dtype=dtypes.float32)
        ])

  def testNotGoodDtype(self):
    with self.assertRaises(ValueError):
      feature_column_ops.infer_real_valued_columns(
          constant_op.constant(
              [["a"]], dtype=dtypes.string))

  def testSparseTensor(self):
    with self.assertRaises(ValueError):
      feature_column_ops.infer_real_valued_columns(
          sparse_tensor.SparseTensor(
              indices=[[0, 0]], values=["a"], dense_shape=[1, 1]))


if __name__ == "__main__":
  test.main()

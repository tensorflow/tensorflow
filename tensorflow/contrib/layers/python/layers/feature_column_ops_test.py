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

import collections
import numpy as np
import tensorflow as tf

from tensorflow.contrib.layers.python.layers import feature_column_ops
from tensorflow.python.ops import init_ops


class TransformerTest(tf.test.TestCase):

  def testRealValuedColumnIsIdentityTransformation(self):
    real_valued = tf.contrib.layers.real_valued_column("price")
    features = {"price": tf.constant([[20.], [110], [-3]])}
    output = feature_column_ops._Transformer(features).transform(real_valued)
    with self.test_session():
      self.assertAllEqual(output.eval(), [[20.], [110], [-3]])

  def testBucketizedColumn(self):
    bucket = tf.contrib.layers.bucketized_column(
        tf.contrib.layers.real_valued_column("price"),
        boundaries=[0., 10., 100.])
    # buckets 2, 3, 0
    features = {"price": tf.constant([[20.], [110], [-3]])}

    # Test transform features.
    output = tf.contrib.layers.transform_features(
        features=features, feature_columns=[bucket])
    self.assertEqual(len(output), 1)
    self.assertIn(bucket, output)
    with self.test_session():
      self.assertAllEqual(output[bucket].eval(), [[2], [3], [0]])


  def testBucketizedColumnWithMultiDimensions(self):
    bucket = tf.contrib.layers.bucketized_column(
        tf.contrib.layers.real_valued_column("price", 2),
        boundaries=[0., 10., 100.])
    # buckets 2, 3, 0
    features = {"price": tf.constant([[20., 110], [110., 20], [-3, -3]])}
    output = feature_column_ops._Transformer(features).transform(bucket)
    with self.test_session():
      self.assertAllEqual(output.eval(), [[2, 3], [3, 2], [0, 0]])

  def testCachedTransformation(self):
    bucket = tf.contrib.layers.bucketized_column(
        tf.contrib.layers.real_valued_column("price"),
        boundaries=[0., 10., 100.])
    # buckets 2, 3, 0
    features = {"price": tf.constant([[20.], [110], [-3]])}
    transformer = feature_column_ops._Transformer(features)
    with self.test_session() as sess:
      transformer.transform(bucket)
      num_of_ops = len(sess.graph.get_operations())
      # Verify that the second call to transform the same feature
      # doesn't increase the number of ops.
      transformer.transform(bucket)
      self.assertEqual(num_of_ops, len(sess.graph.get_operations()))

  def testSparseColumnWithHashBucket(self):
    hashed_sparse = tf.contrib.layers.sparse_column_with_hash_bucket("wire", 10)
    wire_tensor = tf.SparseTensor(values=["omar", "stringer", "marlo"],
                                  indices=[[0, 0], [1, 0], [1, 1]],
                                  shape=[2, 2])
    features = {"wire": wire_tensor}
    # Test transform features.
    output = tf.contrib.layers.transform_features(
        features=features, feature_columns=[hashed_sparse])
    self.assertEqual(len(output), 1)
    self.assertIn(hashed_sparse, output)
    with self.test_session():
      self.assertEqual(output[hashed_sparse].values.dtype, tf.int64)
      self.assertTrue(
          all(x < 10 and x >= 0 for x in output[hashed_sparse].values.eval()))
      self.assertAllEqual(output[hashed_sparse].indices.eval(),
                          wire_tensor.indices.eval())
      self.assertAllEqual(output[hashed_sparse].shape.eval(),
                          wire_tensor.shape.eval())

  def testSparseIntColumnWithHashBucket(self):
    """Tests a sparse column with int values."""
    hashed_sparse = tf.contrib.layers.sparse_column_with_hash_bucket(
        "wire", 10, dtype=tf.int64)
    wire_tensor = tf.SparseTensor(values=[101, 201, 301],
                                  indices=[[0, 0], [1, 0], [1, 1]],
                                  shape=[2, 2])
    features = {"wire": wire_tensor}
    # Test transform features.
    output = tf.contrib.layers.transform_features(
        features=features, feature_columns=[hashed_sparse])
    self.assertEqual(len(output), 1)
    self.assertIn(hashed_sparse, output)
    with self.test_session():
      self.assertEqual(output[hashed_sparse].values.dtype, tf.int64)
      self.assertTrue(
          all(x < 10 and x >= 0 for x in output[hashed_sparse].values.eval()))
      self.assertAllEqual(output[hashed_sparse].indices.eval(),
                          wire_tensor.indices.eval())
      self.assertAllEqual(output[hashed_sparse].shape.eval(),
                          wire_tensor.shape.eval())

  def testSparseColumnWithHashBucketWithDenseInputTensor(self):
    hashed_sparse = tf.contrib.layers.sparse_column_with_hash_bucket("wire", 10)
    wire_tensor = tf.constant([["omar", "stringer"], ["marlo", "rick"]])
    features = {"wire": wire_tensor}
    output = feature_column_ops._Transformer(features).transform(hashed_sparse)

    with self.test_session():
      # While the input is a dense Tensor, the output should be a SparseTensor.
      self.assertIsInstance(output, tf.SparseTensor)
      self.assertEqual(output.values.dtype, tf.int64)
      self.assertTrue(all(x < 10 and x >= 0 for x in output.values.eval()))
      self.assertAllEqual(output.indices.eval(),
                          [[0, 0], [0, 1], [1, 0], [1, 1]])
      self.assertAllEqual(output.shape.eval(), [2, 2])

  def testEmbeddingColumn(self):
    hashed_sparse = tf.contrib.layers.sparse_column_with_hash_bucket("wire", 10)
    wire_tensor = tf.SparseTensor(values=["omar", "stringer", "marlo"],
                                  indices=[[0, 0], [1, 0], [1, 1]],
                                  shape=[2, 2])
    features = {"wire": wire_tensor}
    output = feature_column_ops._Transformer(features).transform(
        tf.contrib.layers.embedding_column(hashed_sparse, 10))
    expected = feature_column_ops._Transformer(features).transform(
        hashed_sparse)
    with self.test_session():
      self.assertAllEqual(output.values.eval(), expected.values.eval())
      self.assertAllEqual(output.indices.eval(), expected.indices.eval())
      self.assertAllEqual(output.shape.eval(), expected.shape.eval())

    # Test transform features.
    output = tf.contrib.layers.transform_features(
        features=features, feature_columns=[hashed_sparse])
    self.assertEqual(len(output), 1)
    self.assertIn(hashed_sparse, output)

  def testSparseColumnWithKeys(self):
    keys_sparse = tf.contrib.layers.sparse_column_with_keys(
        "wire", ["marlo", "omar", "stringer"])
    wire_tensor = tf.SparseTensor(values=["omar", "stringer", "marlo"],
                                  indices=[[0, 0], [1, 0], [1, 1]],
                                  shape=[2, 2])
    features = {"wire": wire_tensor}
    # Test transform features.
    output = tf.contrib.layers.transform_features(
        features=features, feature_columns=[keys_sparse])
    self.assertEqual(len(output), 1)
    self.assertIn(keys_sparse, output)
    with self.test_session():
      tf.initialize_all_tables().run()
      self.assertEqual(output[keys_sparse].values.dtype, tf.int64)
      self.assertAllEqual(output[keys_sparse].values.eval(), [1, 2, 0])
      self.assertAllEqual(output[keys_sparse].indices.eval(),
                          wire_tensor.indices.eval())
      self.assertAllEqual(output[keys_sparse].shape.eval(),
                          wire_tensor.shape.eval())

  def testSparseColumnWithKeysWithDenseInputTensor(self):
    keys_sparse = tf.contrib.layers.sparse_column_with_keys(
        "wire", ["marlo", "omar", "stringer", "rick"])
    wire_tensor = tf.constant([["omar", "stringer"], ["marlo", "rick"]])

    features = {"wire": wire_tensor}
    output = feature_column_ops._Transformer(features).transform(keys_sparse)

    with self.test_session():
      tf.initialize_all_tables().run()
      # While the input is a dense Tensor, the output should be a SparseTensor.
      self.assertIsInstance(output, tf.SparseTensor)
      self.assertEqual(output.dtype, tf.int64)
      self.assertAllEqual(output.values.eval(), [1, 2, 0, 3])
      self.assertAllEqual(output.indices.eval(),
                          [[0, 0], [0, 1], [1, 0], [1, 1]])
      self.assertAllEqual(output.shape.eval(), [2, 2])

  def testSparseColumnWithHashBucket_IsIntegerized(self):
    hashed_sparse = tf.contrib.layers.sparse_column_with_integerized_feature(
        "wire", 10)
    wire_tensor = tf.SparseTensor(values=[100, 1, 25],
                                  indices=[[0, 0], [1, 0], [1, 1]],
                                  shape=[2, 2])
    features = {"wire": wire_tensor}
    # Test transform features.
    output = tf.contrib.layers.transform_features(
        features=features, feature_columns=[hashed_sparse])
    self.assertEqual(len(output), 1)
    self.assertIn(hashed_sparse, output)
    with self.test_session():
      self.assertEqual(output[hashed_sparse].values.dtype, tf.int32)
      self.assertTrue(
          all(x < 10 and x >= 0 for x in output[hashed_sparse].values.eval()))
      self.assertAllEqual(output[hashed_sparse].indices.eval(),
                          wire_tensor.indices.eval())
      self.assertAllEqual(output[hashed_sparse].shape.eval(),
                          wire_tensor.shape.eval())

  def testSparseColumnWithHashBucketWithDenseInputTensor_IsIntegerized(self):
    hashed_sparse = tf.contrib.layers.sparse_column_with_integerized_feature(
        "wire", 10)
    # wire_tensor = tf.SparseTensor(values=[100, 1, 25],
    #                               indices=[[0, 0], [1, 0], [1, 1]],
    #                               shape=[2, 2])
    wire_tensor = tf.constant([[100, 0], [1, 25]])
    features = {"wire": wire_tensor}
    output = feature_column_ops._Transformer(features).transform(hashed_sparse)
    with self.test_session():
      # While the input is a dense Tensor, the output should be a SparseTensor.
      self.assertIsInstance(output, tf.SparseTensor)
      self.assertEqual(output.values.dtype, tf.int32)
      self.assertTrue(all(x < 10 and x >= 0 for x in output.values.eval()))
      self.assertAllEqual(output.indices.eval(),
                          [[0, 0], [0, 1], [1, 0], [1, 1]])
      self.assertAllEqual(output.shape.eval(), [2, 2])

  def testWeightedSparseColumn(self):
    ids = tf.contrib.layers.sparse_column_with_keys(
        "ids", ["marlo", "omar", "stringer"])
    ids_tensor = tf.SparseTensor(values=["stringer", "stringer", "marlo"],
                                 indices=[[0, 0], [1, 0], [1, 1]],
                                 shape=[2, 2])
    weighted_ids = tf.contrib.layers.weighted_sparse_column(ids, "weights")
    weights_tensor = tf.SparseTensor(values=[10.0, 20.0, 30.0],
                                     indices=[[0, 0], [1, 0], [1, 1]],
                                     shape=[2, 2])
    features = {"ids": ids_tensor,
                "weights": weights_tensor}
    # Test transform features.
    output = tf.contrib.layers.transform_features(
        features=features, feature_columns=[weighted_ids])
    self.assertEqual(len(output), 1)
    self.assertIn(weighted_ids, output)
    print(output)
    with self.test_session():
      tf.initialize_all_tables().run()
      self.assertAllEqual(output[weighted_ids][0].shape.eval(),
                          ids_tensor.shape.eval())
      self.assertAllEqual(output[weighted_ids][0].indices.eval(),
                          ids_tensor.indices.eval())
      self.assertAllEqual(output[weighted_ids][0].values.eval(), [2, 2, 0])
      self.assertAllEqual(output[weighted_ids][1].shape.eval(),
                          weights_tensor.shape.eval())
      self.assertAllEqual(output[weighted_ids][1].indices.eval(),
                          weights_tensor.indices.eval())
      self.assertEqual(output[weighted_ids][1].values.dtype, tf.float32)
      self.assertAllEqual(output[weighted_ids][1].values.eval(),
                          weights_tensor.values.eval())

  def testCrossColumn(self):
    language = tf.contrib.layers.sparse_column_with_hash_bucket(
        "language", hash_bucket_size=3)
    country = tf.contrib.layers.sparse_column_with_hash_bucket(
        "country", hash_bucket_size=5)
    country_language = tf.contrib.layers.crossed_column(
        [language, country], hash_bucket_size=15)
    features = {
        "language": tf.SparseTensor(values=["english", "spanish"],
                                    indices=[[0, 0], [1, 0]],
                                    shape=[2, 1]),
        "country": tf.SparseTensor(values=["US", "SV"],
                                   indices=[[0, 0], [1, 0]],
                                   shape=[2, 1])
    }
    # Test transform features.
    output = tf.contrib.layers.transform_features(
        features=features, feature_columns=[country_language])
    self.assertEqual(len(output), 1)
    self.assertIn(country_language, output)
    with self.test_session():
      self.assertEqual(output[country_language].values.dtype, tf.int64)
      self.assertTrue(
          all(x < 15 and x >= 0 for x in output[country_language].values.eval(
          )))

  def testCrossWithBucketizedColumn(self):
    price_bucket = tf.contrib.layers.bucketized_column(
        tf.contrib.layers.real_valued_column("price"),
        boundaries=[0., 10., 100.])
    country = tf.contrib.layers.sparse_column_with_hash_bucket(
        "country", hash_bucket_size=5)
    country_price = tf.contrib.layers.crossed_column(
        [country, price_bucket], hash_bucket_size=15)
    features = {
        "price": tf.constant([[20.]]),
        "country": tf.SparseTensor(values=["US", "SV"],
                                   indices=[[0, 0], [0, 1]],
                                   shape=[1, 2])
    }
    # Test transform features.
    output = tf.contrib.layers.transform_features(
        features=features, feature_columns=[country_price])
    self.assertEqual(len(output), 1)
    self.assertIn(country_price, output)
    with self.test_session():
      self.assertEqual(output[country_price].values.dtype, tf.int64)
      self.assertTrue(
          all(x < 15 and x >= 0 for x in output[country_price].values.eval()))

  def testCrossWithMultiDimensionBucketizedColumn(self):
    country = tf.contrib.layers.sparse_column_with_hash_bucket(
        "country", hash_bucket_size=5)
    price_bucket = tf.contrib.layers.bucketized_column(
        tf.contrib.layers.real_valued_column("price", 2),
        boundaries=[0., 10., 100.])
    country_price = tf.contrib.layers.crossed_column(
        [country, price_bucket], hash_bucket_size=1000)

    with tf.Graph().as_default():
      features = {"price": tf.constant([[20., 210.], [110., 50.], [-3., -30.]]),
                  "country": tf.SparseTensor(values=["US", "SV", "US"],
                                             indices=[[0, 0], [1, 0], [2, 0]],
                                             shape=[3, 2])}
      output, column_to_variable, _ = (
          tf.contrib.layers.weighted_sum_from_feature_columns(features,
                                                              [country_price],
                                                              num_outputs=1))

      weights = column_to_variable[country_price][0]
      grad = tf.squeeze(tf.gradients(output, weights)[0].values)
      with self.test_session():
        tf.global_variables_initializer().run()
        self.assertEqual(len(grad.eval()), 6)

      # Test transform features.
      output = tf.contrib.layers.transform_features(
          features=features, feature_columns=[country_price])
      self.assertEqual(len(output), 1)
      self.assertIn(country_price, output)

  def testCrossWithCrossedColumn(self):
    price_bucket = tf.contrib.layers.bucketized_column(
        tf.contrib.layers.real_valued_column("price"),
        boundaries=[0., 10., 100.])
    country = tf.contrib.layers.sparse_column_with_hash_bucket(
        "country", hash_bucket_size=5)
    country_price = tf.contrib.layers.crossed_column(
        [country, price_bucket], hash_bucket_size=15)
    wire = tf.contrib.layers.sparse_column_with_hash_bucket("wire", 10)
    wire_country_price = tf.contrib.layers.crossed_column(
        [wire, country_price], hash_bucket_size=15)
    features = {
        "price": tf.constant([[20.]]),
        "country": tf.SparseTensor(values=["US", "SV"],
                                   indices=[[0, 0], [0, 1]],
                                   shape=[1, 2]),
        "wire": tf.SparseTensor(values=["omar", "stringer", "marlo"],
                                indices=[[0, 0], [0, 1], [0, 2]],
                                shape=[1, 3])
    }
    # Test transform features.
    output = tf.contrib.layers.transform_features(
        features=features, feature_columns=[wire_country_price])
    self.assertEqual(len(output), 1)
    self.assertIn(wire_country_price, output)
    with self.test_session():
      self.assertEqual(output[wire_country_price].values.dtype, tf.int64)
      self.assertTrue(
          all(x < 15 and x >= 0 for x in output[wire_country_price].values.eval(
          )))

  def testIfFeatureTableContainsTransformationReturnIt(self):
    any_column = tf.contrib.layers.sparse_column_with_hash_bucket("sparse", 10)
    features = {any_column: "any-thing-even-not-a-tensor"}
    output = feature_column_ops._Transformer(features).transform(any_column)
    self.assertEqual(output, "any-thing-even-not-a-tensor")


class CreateInputLayersForDNNsTest(tf.test.TestCase):

  def testAllDNNColumns(self):
    sparse_column = tf.contrib.layers.sparse_column_with_keys(
        "ids", ["a", "b", "c", "unseen"])

    real_valued_column = tf.contrib.layers.real_valued_column("income", 2)
    one_hot_column = tf.contrib.layers.one_hot_column(sparse_column)
    embedding_column = tf.contrib.layers.embedding_column(sparse_column, 10)
    features = {
        "ids": tf.SparseTensor(
            values=["c", "b", "a"],
            indices=[[0, 0], [1, 0], [2, 0]],
            shape=[3, 1]),
        "income": tf.constant([[20.3, 10], [110.3, 0.4], [-3.0, 30.4]])
    }
    output = tf.contrib.layers.input_from_feature_columns(features,
                                                          [one_hot_column,
                                                           embedding_column,
                                                           real_valued_column])
    with self.test_session():
      tf.global_variables_initializer().run()
      tf.initialize_all_tables().run()
      self.assertAllEqual(output.eval().shape, [3, 2 + 4 + 10])

  def testRealValuedColumn(self):
    real_valued = tf.contrib.layers.real_valued_column("price")
    features = {"price": tf.constant([[20.], [110], [-3]])}
    output = tf.contrib.layers.input_from_feature_columns(features,
                                                          [real_valued])
    with self.test_session():
      self.assertAllClose(output.eval(), features["price"].eval())

  def testRealValuedColumnWithMultiDimensions(self):
    real_valued = tf.contrib.layers.real_valued_column("price", 2)
    features = {"price": tf.constant([[20., 10.],
                                      [110, 0.],
                                      [-3, 30]])}
    output = tf.contrib.layers.input_from_feature_columns(features,
                                                          [real_valued])
    with self.test_session():
      self.assertAllClose(output.eval(), features["price"].eval())

  def testRealValuedColumnWithNormalizer(self):
    real_valued = tf.contrib.layers.real_valued_column(
        "price", normalizer=lambda x: x - 2)
    features = {"price": tf.constant([[20.], [110], [-3]])}
    output = tf.contrib.layers.input_from_feature_columns(features,
                                                          [real_valued])
    with self.test_session():
      self.assertAllClose(output.eval(), features["price"].eval() - 2)

  def testRealValuedColumnWithMultiDimensionsAndNormalizer(self):
    real_valued = tf.contrib.layers.real_valued_column(
        "price", 2, normalizer=lambda x: x - 2)
    features = {"price": tf.constant([[20., 10.], [110, 0.], [-3, 30]])}
    output = tf.contrib.layers.input_from_feature_columns(features,
                                                          [real_valued])
    with self.test_session():
      self.assertAllClose(output.eval(), features["price"].eval() - 2)

  def testBucketizedColumnSucceedsForDNN(self):
    bucket = tf.contrib.layers.bucketized_column(
        tf.contrib.layers.real_valued_column("price"),
        boundaries=[0., 10., 100.])
    # buckets 2, 3, 0
    features = {"price": tf.constant([[20.], [110], [-3]])}
    output = tf.contrib.layers.input_from_feature_columns(features, [bucket])
    expected = [[0, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0]]
    with self.test_session():
      self.assertAllClose(output.eval(), expected)

  def testBucketizedColumnWithNormalizerSucceedsForDNN(self):
    bucket = tf.contrib.layers.bucketized_column(
        tf.contrib.layers.real_valued_column(
            "price", normalizer=lambda x: x - 15),
        boundaries=[0., 10., 100.])
    # buckets 2, 3, 0
    features = {"price": tf.constant([[20.], [110], [-3]])}
    output = tf.contrib.layers.input_from_feature_columns(features, [bucket])
    expected = [[0, 1, 0, 0], [0, 0, 1, 0], [1, 0, 0, 0]]
    with self.test_session():
      self.assertAllClose(output.eval(), expected)

  def testBucketizedColumnWithMultiDimensionsSucceedsForDNN(self):
    bucket = tf.contrib.layers.bucketized_column(
        tf.contrib.layers.real_valued_column("price", 2),
        boundaries=[0., 10., 100.])
    # buckets [2, 3], [3, 2], [0, 0]. dimension = 2
    features = {"price": tf.constant([[20., 200],
                                      [110, 50],
                                      [-3, -3]])}
    output = tf.contrib.layers.input_from_feature_columns(features, [bucket])
    expected = [[0, 0, 1, 0, 0, 0, 0, 1],
                [0, 0, 0, 1, 0, 0, 1, 0],
                [1, 0, 0, 0, 1, 0, 0, 0]]
    with self.test_session():
      self.assertAllClose(output.eval(), expected)

  def testOneHotColumnFromWeightedSparseColumnFails(self):
    ids_column = tf.contrib.layers.sparse_column_with_keys(
        "ids", ["a", "b", "c", "unseen"])
    ids_tensor = tf.SparseTensor(
        values=["c", "b", "a", "c"],
        indices=[[0, 0], [1, 0], [2, 0], [2, 1]],
        shape=[3, 2])
    weighted_ids_column = tf.contrib.layers.weighted_sparse_column(ids_column,
                                                                   "weights")
    weights_tensor = tf.SparseTensor(
        values=[10.0, 20.0, 30.0, 40.0],
        indices=[[0, 0], [1, 0], [2, 0], [2, 1]],
        shape=[3, 2])
    features = {"ids": ids_tensor, "weights": weights_tensor}
    one_hot_column = tf.contrib.layers.one_hot_column(weighted_ids_column)
    with self.test_session():
      tf.global_variables_initializer().run()
      tf.initialize_all_tables().run()
      with self.assertRaisesRegexp(
          ValueError,
          "one_hot_column does not yet support weighted_sparse_column"):
        _ = tf.contrib.layers.input_from_feature_columns(features,
                                                         [one_hot_column])

  def testOneHotColumnFromSparseColumnWithKeysSucceedsForDNN(self):
    ids_column = tf.contrib.layers.sparse_column_with_keys(
        "ids", ["a", "b", "c", "unseen"])
    ids_tensor = tf.SparseTensor(
        values=["c", "b", "a"], indices=[[0, 0], [1, 0], [2, 0]], shape=[3, 1])
    one_hot_sparse = tf.contrib.layers.one_hot_column(ids_column)
    features = {"ids": ids_tensor}
    output = tf.contrib.layers.input_from_feature_columns(features,
                                                          [one_hot_sparse])

    with self.test_session():
      tf.global_variables_initializer().run()
      tf.initialize_all_tables().run()
      self.assertAllEqual([[0, 0, 1, 0], [0, 1, 0, 0], [1, 0, 0, 0]],
                          output.eval())

  def testOneHotColumnFromMultivalentSparseColumnWithKeysSucceedsForDNN(self):
    ids_column = tf.contrib.layers.sparse_column_with_keys(
        "ids", ["a", "b", "c", "unseen"])
    ids_tensor = tf.SparseTensor(
        values=["c", "b", "a", "c"],
        indices=[[0, 0], [1, 0], [2, 0], [2, 1]],
        shape=[3, 2])
    one_hot_sparse = tf.contrib.layers.one_hot_column(ids_column)
    features = {"ids": ids_tensor}
    output = tf.contrib.layers.input_from_feature_columns(features,
                                                          [one_hot_sparse])

    with self.test_session():
      tf.global_variables_initializer().run()
      tf.initialize_all_tables().run()
      self.assertAllEqual([[0, 0, 1, 0], [0, 1, 0, 0], [1, 0, 1, 0]],
                          output.eval())

  def testOneHotColumnFromSparseColumnWithIntegerizedFeaturePassesForDNN(self):
    ids_column = tf.contrib.layers.sparse_column_with_integerized_feature(
        "ids", bucket_size=4)
    one_hot_sparse = tf.contrib.layers.one_hot_column(ids_column)
    features = {"ids": tf.SparseTensor(
        values=[2, 1, 0, 2],
        indices=[[0, 0], [1, 0], [2, 0], [2, 1]],
        shape=[3, 2])}
    output = tf.contrib.layers.input_from_feature_columns(features,
                                                          [one_hot_sparse])
    with self.test_session():
      tf.global_variables_initializer().run()
      self.assertAllEqual([[0, 0, 1, 0], [0, 1, 0, 0], [1, 0, 1, 0]],
                          output.eval())

  def testOneHotColumnFromSparseColumnWithHashBucketSucceedsForDNN(self):
    hashed_sparse = tf.contrib.layers.sparse_column_with_hash_bucket("feat", 10)
    wire_tensor = tf.SparseTensor(
        values=["a", "b", "c1", "c2"],
        indices=[[0, 0], [1, 0], [2, 0], [2, 1]],
        shape=[3, 2])
    features = {"feat": wire_tensor}
    one_hot_sparse = tf.contrib.layers.one_hot_column(hashed_sparse)
    output = tf.contrib.layers.input_from_feature_columns(features,
                                                          [one_hot_sparse])
    with self.test_session():
      tf.global_variables_initializer().run()
      tf.initialize_all_tables().run()
      self.assertAllEqual([3, 10], output.eval().shape)

  def testEmbeddingColumnSucceedsForDNN(self):
    hashed_sparse = tf.contrib.layers.sparse_column_with_hash_bucket("wire", 10)
    wire_tensor = tf.SparseTensor(
        values=["omar", "stringer", "marlo", "xx", "yy"],
        indices=[[0, 0], [1, 0], [1, 1], [2, 0], [3, 0]],
        shape=[4, 2])
    features = {"wire": wire_tensor}
    embeded_sparse = tf.contrib.layers.embedding_column(hashed_sparse, 10)
    output = tf.contrib.layers.input_from_feature_columns(features,
                                                          [embeded_sparse])
    with self.test_session():
      tf.global_variables_initializer().run()
      self.assertAllEqual(output.eval().shape, [4, 10])

  def testHashedEmbeddingColumnSucceedsForDNN(self):
    wire_tensor = tf.SparseTensor(values=["omar", "stringer", "marlo", "omar"],
                                  indices=[[0, 0], [1, 0], [1, 1], [2, 0]],
                                  shape=[3, 2])

    features = {"wire": wire_tensor}
    # Big enough hash space so that hopefully there is no collision
    embedded_sparse = tf.contrib.layers.hashed_embedding_column("wire", 1000, 3)
    output = tf.contrib.layers.input_from_feature_columns(
        features, [embedded_sparse], weight_collections=["my_collection"])
    weights = tf.get_collection("my_collection")
    grad = tf.gradients(output, weights)
    with self.test_session():
      tf.global_variables_initializer().run()
      gradient_values = []
      # Collect the gradient from the different partitions (one in this test)
      for p in range(len(grad)):
        gradient_values.extend(grad[p].values.eval())
      gradient_values.sort()
      self.assertAllEqual(gradient_values, [0.5]*6 + [2]*3)

  def testEmbeddingColumnWithInitializerSucceedsForDNN(self):
    hashed_sparse = tf.contrib.layers.sparse_column_with_hash_bucket("wire", 10)
    wire_tensor = tf.SparseTensor(values=["omar", "stringer", "marlo"],
                                  indices=[[0, 0], [1, 0], [1, 1]],
                                  shape=[2, 2])
    features = {"wire": wire_tensor}
    init_value = 133.7
    embeded_sparse = tf.contrib.layers.embedding_column(
        hashed_sparse,
        10, initializer=tf.constant_initializer(init_value))
    output = tf.contrib.layers.input_from_feature_columns(features,
                                                          [embeded_sparse])

    with self.test_session():
      tf.global_variables_initializer().run()
      output_eval = output.eval()
      self.assertAllEqual(output_eval.shape, [2, 10])
      self.assertAllClose(output_eval, np.tile(init_value, [2, 10]))

  def testEmbeddingColumnWithMultipleInitializersFails(self):
    hashed_sparse = tf.contrib.layers.sparse_column_with_hash_bucket("wire", 10)
    wire_tensor = tf.SparseTensor(values=["omar", "stringer", "marlo"],
                                  indices=[[0, 0], [1, 0], [1, 1]],
                                  shape=[2, 2])
    features = {"wire": wire_tensor}
    embedded_sparse = tf.contrib.layers.embedding_column(
        hashed_sparse,
        10,
        initializer=tf.truncated_normal_initializer(mean=42,
                                                    stddev=1337))
    embedded_sparse_alternate = tf.contrib.layers.embedding_column(
        hashed_sparse,
        10,
        initializer=tf.truncated_normal_initializer(mean=1337,
                                                    stddev=42))

    # Makes sure that trying to use different initializers with the same
    # embedding column explicitly fails.
    with self.test_session():
      with self.assertRaisesRegexp(
          ValueError,
          "Duplicate feature column key found for column: wire_embedding"):
        tf.contrib.layers.input_from_feature_columns(
            features, [embedded_sparse, embedded_sparse_alternate])

  def testEmbeddingColumnWithWeightedSparseColumnSucceedsForDNN(self):
    ids = tf.contrib.layers.sparse_column_with_keys(
        "ids", ["marlo", "omar", "stringer"])
    ids_tensor = tf.SparseTensor(values=["stringer", "stringer", "marlo"],
                                 indices=[[0, 0], [1, 0], [1, 1]],
                                 shape=[2, 2])
    weighted_ids = tf.contrib.layers.weighted_sparse_column(ids, "weights")
    weights_tensor = tf.SparseTensor(values=[10.0, 20.0, 30.0],
                                     indices=[[0, 0], [1, 0], [1, 1]],
                                     shape=[2, 2])
    features = {"ids": ids_tensor,
                "weights": weights_tensor}
    embeded_sparse = tf.contrib.layers.embedding_column(weighted_ids, 10)
    output = tf.contrib.layers.input_from_feature_columns(features,
                                                          [embeded_sparse])
    with self.test_session():
      tf.global_variables_initializer().run()
      tf.initialize_all_tables().run()
      self.assertAllEqual(output.eval().shape, [2, 10])

  def testEmbeddingColumnWithCrossedColumnSucceedsForDNN(self):
    a = tf.contrib.layers.sparse_column_with_hash_bucket("aaa",
                                                         hash_bucket_size=100)
    b = tf.contrib.layers.sparse_column_with_hash_bucket("bbb",
                                                         hash_bucket_size=100)
    crossed = tf.contrib.layers.crossed_column(
        set([a, b]), hash_bucket_size=10000)
    wire_tensor = tf.SparseTensor(values=["omar", "stringer", "marlo"],
                                  indices=[[0, 0], [1, 0], [1, 1]],
                                  shape=[2, 2])
    features = {"aaa": wire_tensor, "bbb": wire_tensor}
    embeded_sparse = tf.contrib.layers.embedding_column(crossed, 10)
    output = tf.contrib.layers.input_from_feature_columns(features,
                                                          [embeded_sparse])
    with self.test_session():
      tf.global_variables_initializer().run()
      self.assertAllEqual(output.eval().shape, [2, 10])

  def testSparseColumnFailsForDNN(self):
    hashed_sparse = tf.contrib.layers.sparse_column_with_hash_bucket("wire", 10)
    wire_tensor = tf.SparseTensor(values=["omar", "stringer", "marlo"],
                                  indices=[[0, 0], [1, 0], [1, 1]],
                                  shape=[2, 2])
    features = {"wire": wire_tensor}
    with self.test_session():
      with self.assertRaisesRegexp(
          ValueError, "Error creating input layer for column: wire"):
        tf.global_variables_initializer().run()
        tf.contrib.layers.input_from_feature_columns(features, [hashed_sparse])

  def testWeightedSparseColumnFailsForDNN(self):
    ids = tf.contrib.layers.sparse_column_with_keys(
        "ids", ["marlo", "omar", "stringer"])
    ids_tensor = tf.SparseTensor(values=["stringer", "stringer", "marlo"],
                                 indices=[[0, 0], [1, 0], [1, 1]],
                                 shape=[2, 2])
    weighted_ids = tf.contrib.layers.weighted_sparse_column(ids, "weights")
    weights_tensor = tf.SparseTensor(values=[10.0, 20.0, 30.0],
                                     indices=[[0, 0], [1, 0], [1, 1]],
                                     shape=[2, 2])
    features = {"ids": ids_tensor,
                "weights": weights_tensor}
    with self.test_session():
      with self.assertRaisesRegexp(
          ValueError,
          "Error creating input layer for column: ids_weighted_by_weights"):
        tf.initialize_all_tables().run()
        tf.contrib.layers.input_from_feature_columns(features, [weighted_ids])

  def testCrossedColumnFailsForDNN(self):
    a = tf.contrib.layers.sparse_column_with_hash_bucket("aaa",
                                                         hash_bucket_size=100)
    b = tf.contrib.layers.sparse_column_with_hash_bucket("bbb",
                                                         hash_bucket_size=100)
    crossed = tf.contrib.layers.crossed_column(
        set([a, b]), hash_bucket_size=10000)
    wire_tensor = tf.SparseTensor(values=["omar", "stringer", "marlo"],
                                  indices=[[0, 0], [1, 0], [1, 1]],
                                  shape=[2, 2])
    features = {"aaa": wire_tensor, "bbb": wire_tensor}
    with self.test_session():
      with self.assertRaisesRegexp(
          ValueError, "Error creating input layer for column: aaa_X_bbb"):
        tf.global_variables_initializer().run()
        tf.contrib.layers.input_from_feature_columns(features, [crossed])

  def testDeepColumnsSucceedForDNN(self):
    real_valued = tf.contrib.layers.real_valued_column("income", 3)
    bucket = tf.contrib.layers.bucketized_column(
        tf.contrib.layers.real_valued_column("price", 2),
        boundaries=[0., 10., 100.])
    hashed_sparse = tf.contrib.layers.sparse_column_with_hash_bucket("wire", 10)
    features = {
        "income": tf.constant([[20., 10, -5], [110, 0, -7], [-3, 30, 50]]),
        "price": tf.constant([[20., 200], [110, 2], [-20, -30]]),
        "wire": tf.SparseTensor(values=["omar", "stringer", "marlo"],
                                indices=[[0, 0], [1, 0], [2, 0]],
                                shape=[3, 1])
    }
    embeded_sparse = tf.contrib.layers.embedding_column(
        hashed_sparse,
        10, initializer=tf.constant_initializer(133.7))
    output = tf.contrib.layers.input_from_feature_columns(
        features, [real_valued, bucket, embeded_sparse])
    with self.test_session():
      tf.global_variables_initializer().run()
      # size of output = 3 (real_valued) + 2 * 4 (bucket) + 10 (embedding) = 21
      self.assertAllEqual(output.eval().shape, [3, 21])

  def testEmbeddingColumnForDNN(self):
    hashed_sparse = tf.contrib.layers.sparse_column_with_hash_bucket("wire", 10)
    wire_tensor = tf.SparseTensor(values=["omar", "stringer", "marlo"],
                                  indices=[[0, 0], [1, 0], [1, 1]],
                                  shape=[3, 2])
    features = {"wire": wire_tensor}
    embeded_sparse = tf.contrib.layers.embedding_column(
        hashed_sparse,
        1,
        combiner="sum",
        initializer=init_ops.ones_initializer())
    output = tf.contrib.layers.input_from_feature_columns(features,
                                                          [embeded_sparse])
    with self.test_session():
      tf.global_variables_initializer().run()
      # score: (number of values)
      self.assertAllEqual(output.eval(), [[1.], [2.], [0.]])

  def testEmbeddingColumnWithWeightedSparseColumnForDNN(self):
    ids = tf.contrib.layers.sparse_column_with_keys(
        "ids", ["marlo", "omar", "stringer"])
    ids_tensor = tf.SparseTensor(values=["stringer", "stringer", "marlo"],
                                 indices=[[0, 0], [1, 0], [1, 1]],
                                 shape=[3, 2])
    weighted_ids = tf.contrib.layers.weighted_sparse_column(ids, "weights")
    weights_tensor = tf.SparseTensor(values=[10.0, 20.0, 30.0],
                                     indices=[[0, 0], [1, 0], [1, 1]],
                                     shape=[3, 2])
    features = {"ids": ids_tensor,
                "weights": weights_tensor}
    embeded_sparse = tf.contrib.layers.embedding_column(
        weighted_ids,
        1,
        combiner="sum",
        initializer=init_ops.ones_initializer())
    output = tf.contrib.layers.input_from_feature_columns(features,
                                                          [embeded_sparse])
    with self.test_session():
      tf.global_variables_initializer().run()
      tf.initialize_all_tables().run()
      # score: (sum of weights)
      self.assertAllEqual(output.eval(), [[10.], [50.], [0.]])

  def testInputLayerWithCollectionsForDNN(self):
    real_valued = tf.contrib.layers.real_valued_column("price")
    bucket = tf.contrib.layers.bucketized_column(real_valued,
                                                 boundaries=[0., 10., 100.])
    hashed_sparse = tf.contrib.layers.sparse_column_with_hash_bucket("wire", 10)
    features = {
        "price": tf.constant([[20.], [110], [-3]]),
        "wire": tf.SparseTensor(values=["omar", "stringer", "marlo"],
                                indices=[[0, 0], [1, 0], [2, 0]],
                                shape=[3, 1])
    }
    embeded_sparse = tf.contrib.layers.embedding_column(hashed_sparse, 10)
    tf.contrib.layers.input_from_feature_columns(
        features, [real_valued, bucket, embeded_sparse],
        weight_collections=["my_collection"])
    weights = tf.get_collection("my_collection")
    # one variable for embeded sparse
    self.assertEqual(1, len(weights))

  def testInputLayerWithTrainableArgForDNN(self):
    real_valued = tf.contrib.layers.real_valued_column("price")
    bucket = tf.contrib.layers.bucketized_column(real_valued,
                                                 boundaries=[0., 10., 100.])
    hashed_sparse = tf.contrib.layers.sparse_column_with_hash_bucket("wire", 10)
    features = {
        "price": tf.constant([[20.], [110], [-3]]),
        "wire": tf.SparseTensor(values=["omar", "stringer", "marlo"],
                                indices=[[0, 0], [1, 0], [2, 0]],
                                shape=[3, 1])
    }
    embeded_sparse = tf.contrib.layers.embedding_column(hashed_sparse, 10)
    tf.contrib.layers.input_from_feature_columns(
        features, [real_valued, bucket, embeded_sparse],
        weight_collections=["my_collection"],
        trainable=False)
    # There should not be any trainable variables
    self.assertEqual(0, len(tf.trainable_variables()))

    tf.contrib.layers.input_from_feature_columns(
        features, [real_valued, bucket, embeded_sparse],
        weight_collections=["my_collection"],
        trainable=True)
    # There should  one trainable variable for embeded sparse
    self.assertEqual(1, len(tf.trainable_variables()))


class SequenceInputFromFeatureColumnTest(tf.test.TestCase):

  def testSupportedColumns(self):
    measurement = tf.contrib.layers.real_valued_column("measurements")
    country = tf.contrib.layers.sparse_column_with_hash_bucket(
        "country", 100)
    pets = tf.contrib.layers.sparse_column_with_hash_bucket(
        "pets", 100)
    ids = tf.contrib.layers.sparse_column_with_integerized_feature(
        "id", 100)

    country_x_pets = tf.contrib.layers.crossed_column(
        [country, pets], 100)
    country_x_pets_onehot = tf.contrib.layers.one_hot_column(
        country_x_pets)
    bucketized_measurement = tf.contrib.layers.bucketized_column(
        measurement, [.25, .5, .75])
    embedded_id = tf.contrib.layers.embedding_column(
        ids, 100)

    # `_BucketizedColumn` is not supported.
    self.assertRaisesRegexp(
        ValueError,
        "FeatureColumn type _BucketizedColumn is not currently supported",
        tf.contrib.layers.sequence_input_from_feature_columns,
        {}, [measurement, bucketized_measurement])

    # `_CrossedColumn` is not supported.
    self.assertRaisesRegexp(
        ValueError,
        "FeatureColumn type _CrossedColumn is not currently supported",
        tf.contrib.layers.sequence_input_from_feature_columns,
        {}, [embedded_id, country_x_pets])

    # `country_x_pets_onehot` depends on a `_CrossedColumn` which is forbidden.
    self.assertRaisesRegexp(
        ValueError,
        "Column country_X_pets .* _CrossedColumn",
        tf.contrib.layers.sequence_input_from_feature_columns,
        {}, [embedded_id, country_x_pets_onehot])

  def testRealValuedColumn(self):
    batch_size = 4
    sequence_length = 8
    dimension = 3

    np.random.seed(1111)
    measurement_input = np.random.rand(batch_size, sequence_length, dimension)
    measurement_column = tf.contrib.layers.real_valued_column("measurements")
    columns_to_tensors = {"measurements": tf.constant(measurement_input)}
    model_input_tensor = tf.contrib.layers.sequence_input_from_feature_columns(
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
    measurement_column = tf.contrib.layers.real_valued_column("measurements")
    columns_to_tensors = {"measurements": tf.constant(measurement_input)}
    model_input_tensor = tf.contrib.layers.sequence_input_from_feature_columns(
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
    measurement_column = tf.contrib.layers.real_valued_column(
        "measurements", normalizer=normalizer)
    columns_to_tensors = {"measurements": tf.constant(measurement_input)}
    model_input_tensor = tf.contrib.layers.sequence_input_from_feature_columns(
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
    measurement_column = tf.contrib.layers.real_valued_column(
        "measurements", normalizer=normalizer)
    columns_to_tensors = {"measurements": tf.constant(measurement_input)}
    model_input_tensor = tf.contrib.layers.sequence_input_from_feature_columns(
        columns_to_tensors, [measurement_column])

    expected_shape = [batch_size, sequence_length, np.prod(dimensions)]
    reshaped_measurements = np.reshape(measurement_input, expected_shape)

    with self.test_session() as sess:
      model_inputs = sess.run(model_input_tensor)

    self.assertAllClose(normalizer(reshaped_measurements), model_inputs)

  def testOneHotColumnFromSparseColumnWithKeys(self):
    ids_tensor = tf.SparseTensor(
        values=["c", "b",
                "a", "c", "b",
                "b"],
        indices=[[0, 0, 0], [0, 1, 0],
                 [1, 0, 0], [1, 0, 1], [1, 1, 0],
                 [3, 2, 0]],
        shape=[4, 3, 2])

    ids_column = tf.contrib.layers.sparse_column_with_keys(
        "ids", ["a", "b", "c", "unseen"])
    one_hot_column = tf.contrib.layers.one_hot_column(ids_column)
    columns_to_tensors = {"ids": ids_tensor}
    model_input_tensor = tf.contrib.layers.sequence_input_from_feature_columns(
        columns_to_tensors, [one_hot_column])

    with self.test_session() as sess:
      tf.global_variables_initializer().run()
      tf.initialize_all_tables().run()
      model_input = sess.run(model_input_tensor)

    expected_input_shape = np.array([4, 3, 4])
    expected_model_input = np.array(
        [[[0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 0]],
         [[1, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 0]],
         [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
         [[0, 0, 0, 0], [0, 0, 0, 0], [0, 1, 0, 0]]], dtype=np.float32)

    self.assertAllEqual(expected_input_shape, model_input.shape)
    self.assertAllClose(expected_model_input, model_input)

  def testOneHotColumnFromSparseColumnWithHashBucket(self):
    hash_buckets = 10
    ids_tensor = tf.SparseTensor(
        values=["c", "b",
                "a", "c", "b",
                "b"],
        indices=[[0, 0, 0], [0, 1, 0],
                 [1, 0, 0], [1, 0, 1], [1, 1, 0],
                 [3, 2, 0]],
        shape=[4, 3, 2])

    hashed_ids_column = tf.contrib.layers.sparse_column_with_hash_bucket(
        "ids", hash_buckets)
    one_hot_column = tf.contrib.layers.one_hot_column(hashed_ids_column)
    columns_to_tensors = {"ids": ids_tensor}
    model_input_tensor = tf.contrib.layers.sequence_input_from_feature_columns(
        columns_to_tensors, [one_hot_column])

    with self.test_session() as sess:
      tf.global_variables_initializer().run()
      tf.initialize_all_tables().run()
      model_input = sess.run(model_input_tensor)

    expected_input_shape = np.array([4, 3, hash_buckets])
    self.assertAllEqual(expected_input_shape, model_input.shape)

  def testEmbeddingColumn(self):
    hash_buckets = 10
    embedding_dimension = 5
    ids_tensor = tf.SparseTensor(
        values=["c", "b",
                "a", "c", "b",
                "b"],
        indices=[[0, 0, 0], [0, 1, 0],
                 [1, 0, 0], [1, 0, 1], [1, 1, 0],
                 [3, 2, 0]],
        shape=[4, 3, 2])

    expected_input_shape = np.array([4, 3, embedding_dimension])

    hashed_ids_column = tf.contrib.layers.sparse_column_with_hash_bucket(
        "ids", hash_buckets)
    embedded_column = tf.contrib.layers.embedding_column(
        hashed_ids_column, embedding_dimension)
    columns_to_tensors = {"ids": ids_tensor}
    model_input_tensor = tf.contrib.layers.sequence_input_from_feature_columns(
        columns_to_tensors, [embedded_column])

    with self.test_session() as sess:
      tf.global_variables_initializer().run()
      tf.initialize_all_tables().run()
      model_input = sess.run(model_input_tensor)

    self.assertAllEqual(expected_input_shape, model_input.shape)

  def testEmbeddingColumnGradient(self):
    hash_buckets = 1000
    embedding_dimension = 3
    ids_tensor = tf.SparseTensor(
        values=["c", "b",
                "a", "c", "b",
                "b"],
        indices=[[0, 0, 0], [0, 1, 0],
                 [1, 0, 0], [1, 0, 1], [1, 1, 0],
                 [3, 2, 0]],
        shape=[4, 3, 2])

    hashed_ids_column = tf.contrib.layers.sparse_column_with_hash_bucket(
        "ids", hash_buckets)
    embedded_column = tf.contrib.layers.embedding_column(
        hashed_ids_column, embedding_dimension, combiner="sum")
    columns_to_tensors = {"ids": ids_tensor}
    model_input_tensor = tf.contrib.layers.sequence_input_from_feature_columns(
        columns_to_tensors,
        [embedded_column],
        weight_collections=["my_collection"])
    embedding_weights = tf.get_collection("my_collection")
    gradient_tensor = tf.gradients(model_input_tensor, embedding_weights)
    with self.test_session() as sess:
      tf.global_variables_initializer().run()
      tf.initialize_all_tables().run()
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

    measurement_tensor = tf.random_uniform(
        [batch_size, sequence_length, measurement_dimension])
    country_tensor = tf.SparseTensor(
        values=["us", "ca",
                "ru", "fr", "ca",
                "mx"],
        indices=[[0, 0, 0], [0, 1, 0],
                 [1, 0, 0], [1, 0, 1], [1, 1, 0],
                 [3, 2, 0]],
        shape=[4, 3, 2])
    id_tensor = tf.SparseTensor(
        values=[2, 5,
                26, 123, 1,
                0],
        indices=[[0, 0, 0], [0, 0, 1], [0, 1, 1],
                 [1, 0, 0], [1, 1, 0],
                 [3, 2, 0]],
        shape=[4, 3, 2])

    columns_to_tensors = {"measurements": measurement_tensor,
                          "country": country_tensor,
                          "id": id_tensor}

    measurement_column = tf.contrib.layers.real_valued_column(
        "measurements", normalizer=normalizer)
    country_column = tf.contrib.layers.sparse_column_with_hash_bucket(
        "country", country_hash_size)
    id_column = tf.contrib.layers.sparse_column_with_integerized_feature(
        "id", max_id)

    onehot_country_column = tf.contrib.layers.one_hot_column(country_column)
    embedded_id_column = tf.contrib.layers.embedding_column(
        id_column, id_embedding_dimension)

    model_input_columns = [measurement_column,
                           onehot_country_column,
                           embedded_id_column]

    model_input_tensor = tf.contrib.layers.sequence_input_from_feature_columns(
        columns_to_tensors, model_input_columns)
    self.assertEqual(tf.float32, model_input_tensor.dtype)

    with self.test_session() as sess:
      tf.global_variables_initializer().run()
      tf.initialize_all_tables().run()
      model_input = sess.run(model_input_tensor)

    expected_input_shape = [
        batch_size,
        sequence_length,
        measurement_dimension + country_hash_size + id_embedding_dimension]
    self.assertAllEqual(expected_input_shape, model_input.shape)


class WeightedSumTest(tf.test.TestCase):

  def testSparseColumn(self):
    hashed_sparse = tf.contrib.layers.sparse_column_with_hash_bucket("wire", 10)
    wire_tensor = tf.SparseTensor(values=["omar", "stringer", "marlo"],
                                  indices=[[0, 0], [1, 0], [1, 1]],
                                  shape=[2, 2])
    features = {"wire": wire_tensor}
    logits, _, _ = tf.contrib.layers.weighted_sum_from_feature_columns(
        features, [hashed_sparse], num_outputs=5)
    with self.test_session():
      tf.global_variables_initializer().run()
      self.assertAllEqual(logits.eval().shape, [2, 5])

  def testSparseIntColumn(self):
    """Tests a sparse column with int values."""
    hashed_sparse = tf.contrib.layers.sparse_column_with_hash_bucket(
        "wire", 10, dtype=tf.int64)
    wire_tensor = tf.SparseTensor(values=[101, 201, 301],
                                  indices=[[0, 0], [1, 0], [1, 1]],
                                  shape=[2, 2])
    features = {"wire": wire_tensor}
    logits, _, _ = tf.contrib.layers.weighted_sum_from_feature_columns(
        features, [hashed_sparse], num_outputs=5)
    with self.test_session():
      tf.global_variables_initializer().run()
      self.assertAllEqual(logits.eval().shape, [2, 5])

  def testSparseColumnWithDenseInputTensor(self):
    hashed_sparse = tf.contrib.layers.sparse_column_with_hash_bucket("wire", 10)
    wire_tensor = tf.constant([["omar", "stringer"], ["marlo", "rick"]])
    features = {"wire": wire_tensor}
    logits, _, _ = tf.contrib.layers.weighted_sum_from_feature_columns(
        features, [hashed_sparse], num_outputs=5)
    with self.test_session():
      tf.initialize_all_variables().run()
      self.assertAllEqual(logits.eval().shape, [2, 5])

  def testWeightedSparseColumn(self):
    ids = tf.contrib.layers.sparse_column_with_keys(
        "ids", ["marlo", "omar", "stringer"])
    ids_tensor = tf.SparseTensor(values=["stringer", "stringer", "marlo"],
                                 indices=[[0, 0], [1, 0], [1, 1]],
                                 shape=[2, 2])
    weighted_ids = tf.contrib.layers.weighted_sparse_column(ids, "weights")
    weights_tensor = tf.SparseTensor(values=[10.0, 20.0, 30.0],
                                     indices=[[0, 0], [1, 0], [1, 1]],
                                     shape=[2, 2])
    features = {"ids": ids_tensor,
                "weights": weights_tensor}
    logits, _, _ = tf.contrib.layers.weighted_sum_from_feature_columns(
        features, [weighted_ids], num_outputs=5)
    with self.test_session():
      tf.global_variables_initializer().run()
      tf.initialize_all_tables().run()
      self.assertAllEqual(logits.eval().shape, [2, 5])

  def testWeightedSparseColumnWithDenseInputTensor(self):
    ids = tf.contrib.layers.sparse_column_with_keys(
        "ids", ["marlo", "omar", "stringer", "rick"])
    ids_tensor = tf.constant([["omar", "stringer"], ["marlo", "rick"]])
    weighted_ids = tf.contrib.layers.weighted_sparse_column(ids, "weights")
    weights_tensor = tf.constant([[10.0, 20.0], [30.0, 40.0]])

    features = {"ids": ids_tensor,
                "weights": weights_tensor}
    logits, _, _ = tf.contrib.layers.weighted_sum_from_feature_columns(
        features, [weighted_ids], num_outputs=5)

    with self.test_session():
      tf.initialize_all_variables().run()
      tf.initialize_all_tables().run()
      self.assertAllEqual(logits.eval().shape, [2, 5])

  def testCrossedColumn(self):
    a = tf.contrib.layers.sparse_column_with_hash_bucket("aaa",
                                                         hash_bucket_size=100)
    b = tf.contrib.layers.sparse_column_with_hash_bucket("bbb",
                                                         hash_bucket_size=100)
    crossed = tf.contrib.layers.crossed_column(
        set([a, b]), hash_bucket_size=10000)
    wire_tensor = tf.SparseTensor(values=["omar", "stringer", "marlo"],
                                  indices=[[0, 0], [1, 0], [1, 1]],
                                  shape=[2, 2])
    features = {"aaa": wire_tensor, "bbb": wire_tensor}
    logits, _, _ = tf.contrib.layers.weighted_sum_from_feature_columns(
        features, [crossed], num_outputs=5)
    with self.test_session():
      tf.global_variables_initializer().run()
      self.assertAllEqual(logits.eval().shape, [2, 5])

  def testEmbeddingColumn(self):
    hashed_sparse = tf.contrib.layers.sparse_column_with_hash_bucket("wire", 10)
    wire_tensor = tf.SparseTensor(values=["omar", "stringer", "marlo"],
                                  indices=[[0, 0], [1, 0], [1, 1]],
                                  shape=[2, 2])
    features = {"wire": wire_tensor}
    embeded_sparse = tf.contrib.layers.embedding_column(hashed_sparse, 10)
    with self.test_session():
      with self.assertRaisesRegexp(
          ValueError, "Error creating weighted sum for column: wire_embedding"):
        tf.global_variables_initializer().run()
        tf.contrib.layers.weighted_sum_from_feature_columns(features,
                                                            [embeded_sparse],
                                                            num_outputs=5)

  def testRealValuedColumnWithMultiDimensions(self):
    real_valued = tf.contrib.layers.real_valued_column("price", 2)
    features = {"price": tf.constant([[20., 10.], [110, 0.], [-3, 30]])}
    logits, _, _ = tf.contrib.layers.weighted_sum_from_feature_columns(
        features, [real_valued], num_outputs=5)
    with self.test_session():
      tf.global_variables_initializer().run()
      self.assertAllEqual(logits.eval().shape, [3, 5])

  def testBucketizedColumnWithMultiDimensions(self):
    bucket = tf.contrib.layers.bucketized_column(
        tf.contrib.layers.real_valued_column("price", 2),
        boundaries=[0., 10., 100.])
    features = {"price": tf.constant([[20., 10.], [110, 0.], [-3, 30]])}
    logits, _, _ = tf.contrib.layers.weighted_sum_from_feature_columns(
        features, [bucket], num_outputs=5)
    with self.test_session():
      tf.global_variables_initializer().run()
      self.assertAllEqual(logits.eval().shape, [3, 5])

  def testAllWideColumns(self):
    real_valued = tf.contrib.layers.real_valued_column("income", 2)
    bucket = tf.contrib.layers.bucketized_column(
        tf.contrib.layers.real_valued_column("price"),
        boundaries=[0., 10., 100.])
    hashed_sparse = tf.contrib.layers.sparse_column_with_hash_bucket("wire", 10)
    crossed = tf.contrib.layers.crossed_column([bucket, hashed_sparse], 100)
    features = {
        "income": tf.constant([[20., 10], [110, 0], [-3, 30]]),
        "price": tf.constant([[20.], [110], [-3]]),
        "wire": tf.SparseTensor(values=["omar", "stringer", "marlo"],
                                indices=[[0, 0], [1, 0], [2, 0]],
                                shape=[3, 1])
    }
    output, _, _ = tf.contrib.layers.weighted_sum_from_feature_columns(
        features, [real_valued, bucket, hashed_sparse, crossed],
        num_outputs=5)
    with self.test_session():
      tf.global_variables_initializer().run()
      self.assertAllEqual(output.eval().shape, [3, 5])

  def testPredictions(self):
    language = tf.contrib.layers.sparse_column_with_keys(
        column_name="language",
        keys=["english", "finnish", "hindi"])
    age = tf.contrib.layers.real_valued_column("age")
    with tf.Graph().as_default():
      features = {
          "age": tf.constant([[1], [2]]),
          "language": tf.SparseTensor(values=["hindi", "english"],
                                      indices=[[0, 0], [1, 0]],
                                      shape=[2, 1]),
      }
      output, column_to_variable, bias = (
          tf.contrib.layers.weighted_sum_from_feature_columns(features,
                                                              [age, language],
                                                              num_outputs=1))
      with self.test_session() as sess:
        tf.global_variables_initializer().run()
        tf.initialize_all_tables().run()

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
    country = tf.contrib.layers.sparse_column_with_keys(
        column_name="country",
        keys=["us", "finland"])
    language = tf.contrib.layers.sparse_column_with_keys(
        column_name="language",
        keys=["english", "finnish", "hindi"])
    with tf.Graph().as_default():
      features = {
          "country": tf.SparseTensor(values=["finland", "us"],
                                     indices=[[0, 0], [1, 0]],
                                     shape=[2, 1]),
          "language": tf.SparseTensor(values=["hindi", "english"],
                                      indices=[[0, 0], [1, 0]],
                                      shape=[2, 1]),
      }
      output, variables, bias = (
          tf.contrib.layers.joint_weighted_sum_from_feature_columns(
              features, [country, language], num_outputs=1))
      # Assert that only a single weight is created.
      self.assertEqual(len(variables), 1)
      with self.test_session() as sess:
        tf.global_variables_initializer().run()
        tf.initialize_all_tables().run()

        self.assertAllClose(output.eval(), [[0.], [0.]])

        sess.run(bias.assign([0.1]))
        self.assertAllClose(output.eval(), [[0.1], [0.1]])

        # shape is [5,1] because 1 class and 2 + 3 features.
        self.assertEquals(variables[0].get_shape().as_list(), [5, 1])

        # score: bias + country_weight + language_weight
        sess.run(variables[0].assign([[0.1], [0.2], [0.3], [0.4], [0.5]]))
        self.assertAllClose(output.eval(), [[0.8], [0.5]])

  def testJointPredictionsWeightedFails(self):
    language = tf.contrib.layers.weighted_sparse_column(
        tf.contrib.layers.sparse_column_with_keys(
            column_name="language",
            keys=["english", "finnish", "hindi"]),
        "weight")
    with tf.Graph().as_default():
      features = {
          "weight": tf.constant([[1], [2]]),
          "language": tf.SparseTensor(values=["hindi", "english"],
                                      indices=[[0, 0], [1, 0]],
                                      shape=[2, 1]),
      }
      with self.assertRaises(AssertionError):
        tf.contrib.layers.joint_weighted_sum_from_feature_columns(
            features, [language], num_outputs=1)

  def testJointPredictionsRealFails(self):
    age = tf.contrib.layers.real_valued_column("age")
    with tf.Graph().as_default():
      features = {
          "age": tf.constant([[1], [2]]),
      }
      with self.assertRaises(NotImplementedError):
        tf.contrib.layers.joint_weighted_sum_from_feature_columns(
            features, [age], num_outputs=1)

  def testPredictionsWithWeightedSparseColumn(self):
    language = tf.contrib.layers.sparse_column_with_keys(
        column_name="language",
        keys=["english", "finnish", "hindi"])
    weighted_language = tf.contrib.layers.weighted_sparse_column(
        sparse_id_column=language,
        weight_column_name="age")
    with tf.Graph().as_default():
      features = {
          "language": tf.SparseTensor(values=["hindi", "english"],
                                      indices=[[0, 0], [1, 0]],
                                      shape=[2, 1]),
          "age": tf.SparseTensor(values=[10.0, 20.0],
                                 indices=[[0, 0], [1, 0]],
                                 shape=[2, 1])
      }
      output, column_to_variable, bias = (
          tf.contrib.layers.weighted_sum_from_feature_columns(
              features, [weighted_language], num_outputs=1))
      with self.test_session() as sess:
        tf.global_variables_initializer().run()
        tf.initialize_all_tables().run()

        self.assertAllClose(output.eval(), [[0.], [0.]])

        sess.run(bias.assign([0.1]))
        self.assertAllClose(output.eval(), [[0.1], [0.1]])

        # score: bias + age*language_weight[index]
        sess.run(column_to_variable[weighted_language][0].assign(
            [[0.1], [0.2], [0.3]]))
        self.assertAllClose(output.eval(), [[3.1], [2.1]])

  def testPredictionsWithMultivalentColumnButNoCross(self):
    language = tf.contrib.layers.sparse_column_with_keys(
        column_name="language",
        keys=["english", "turkish", "hindi"])
    with tf.Graph().as_default():
      features = {
          "language": tf.SparseTensor(values=["hindi", "english"],
                                      indices=[[0, 0], [0, 1]],
                                      shape=[1, 2])
      }
      output, column_to_variable, bias = (
          tf.contrib.layers.weighted_sum_from_feature_columns(features,
                                                              [language],
                                                              num_outputs=1))
      with self.test_session() as sess:
        tf.global_variables_initializer().run()
        tf.initialize_all_tables().run()

        # score: 0.1 + language_weight['hindi'] + language_weight['english']
        sess.run(bias.assign([0.1]))
        sess.run(column_to_variable[language][0].assign([[0.1], [0.3], [0.2]]))
        self.assertAllClose(output.eval(), [[0.4]])

  def testSparseFeatureColumnWithHashedBucketSize(self):
    movies = tf.contrib.layers.sparse_column_with_hash_bucket(
        column_name="movies", hash_bucket_size=15)
    with tf.Graph().as_default():
      features = {
          "movies": tf.SparseTensor(
              values=["matrix", "head-on", "winter sleep"],
              indices=[[0, 0], [0, 1], [1, 0]],
              shape=[2, 2])
      }
      output, column_to_variable, _ = (
          tf.contrib.layers.weighted_sum_from_feature_columns(features,
                                                              [movies],
                                                              num_outputs=1))
      with self.test_session() as sess:
        tf.global_variables_initializer().run()
        tf.initialize_all_tables().run()

        weights = column_to_variable[movies][0]
        self.assertEqual(weights.get_shape(), (15, 1))
        sess.run(weights.assign(weights + 0.4))
        # score for first example = 0.4 (matrix) + 0.4 (head-on) = 0.8
        # score for second example = 0.4 (winter sleep)
        self.assertAllClose(output.eval(), [[0.8], [0.4]])

  def testCrossUsageInPredictions(self):
    language = tf.contrib.layers.sparse_column_with_hash_bucket(
        "language", hash_bucket_size=3)
    country = tf.contrib.layers.sparse_column_with_hash_bucket(
        "country", hash_bucket_size=5)
    country_language = tf.contrib.layers.crossed_column(
        [language, country], hash_bucket_size=10)
    with tf.Graph().as_default():
      features = {
          "language": tf.SparseTensor(values=["english", "spanish"],
                                      indices=[[0, 0], [1, 0]],
                                      shape=[2, 1]),
          "country": tf.SparseTensor(values=["US", "SV"],
                                     indices=[[0, 0], [1, 0]],
                                     shape=[2, 1])
      }
      output, column_to_variable, _ = (
          tf.contrib.layers.weighted_sum_from_feature_columns(
              features, [country_language],
              num_outputs=1))
      with self.test_session() as sess:
        tf.global_variables_initializer().run()
        tf.initialize_all_tables().run()

        weights = column_to_variable[country_language][0]
        sess.run(weights.assign(weights + 0.4))
        self.assertAllClose(output.eval(), [[0.4], [0.4]])

  def testCrossColumnByItself(self):
    language = tf.contrib.layers.sparse_column_with_hash_bucket(
        "language", hash_bucket_size=3)
    language_language = tf.contrib.layers.crossed_column(
        [language, language], hash_bucket_size=10)
    with tf.Graph().as_default():
      features = {
          "language": tf.SparseTensor(values=["english", "spanish"],
                                      indices=[[0, 0], [0, 1]],
                                      shape=[1, 2]),
      }
      output, column_to_variable, _ = (
          tf.contrib.layers.weighted_sum_from_feature_columns(
              features, [language_language],
              num_outputs=1))
      with self.test_session() as sess:
        tf.global_variables_initializer().run()
        tf.initialize_all_tables().run()

        weights = column_to_variable[language_language][0]
        sess.run(weights.assign(weights + 0.4))
        # There are two features inside language. If we cross it by itself we'll
        # have four crossed features.
        self.assertAllClose(output.eval(), [[1.6]])

  def testMultivalentCrossUsageInPredictions(self):
    language = tf.contrib.layers.sparse_column_with_hash_bucket(
        "language", hash_bucket_size=3)
    country = tf.contrib.layers.sparse_column_with_hash_bucket(
        "country", hash_bucket_size=5)
    country_language = tf.contrib.layers.crossed_column(
        [language, country], hash_bucket_size=10)
    with tf.Graph().as_default():
      features = {
          "language": tf.SparseTensor(values=["english", "spanish"],
                                      indices=[[0, 0], [0, 1]],
                                      shape=[1, 2]),
          "country": tf.SparseTensor(values=["US", "SV"],
                                     indices=[[0, 0], [0, 1]],
                                     shape=[1, 2])
      }
      output, column_to_variable, _ = (
          tf.contrib.layers.weighted_sum_from_feature_columns(
              features, [country_language],
              num_outputs=1))
      with self.test_session() as sess:
        tf.global_variables_initializer().run()
        tf.initialize_all_tables().run()

        weights = column_to_variable[country_language][0]
        sess.run(weights.assign(weights + 0.4))
        # There are four crosses each with 0.4 weight.
        # score = 0.4 + 0.4 + 0.4 + 0.4
        self.assertAllClose(output.eval(), [[1.6]])

  def testMultivalentCrossUsageInPredictionsWithPartition(self):
    # bucket size has to be big enough to allow sharding.
    language = tf.contrib.layers.sparse_column_with_hash_bucket(
        "language", hash_bucket_size=64 << 19)
    country = tf.contrib.layers.sparse_column_with_hash_bucket(
        "country", hash_bucket_size=64 << 18)
    country_language = tf.contrib.layers.crossed_column(
        [language, country], hash_bucket_size=64 << 18)
    with tf.Graph().as_default():
      features = {
          "language": tf.SparseTensor(values=["english", "spanish"],
                                      indices=[[0, 0], [0, 1]],
                                      shape=[1, 2]),
          "country": tf.SparseTensor(values=["US", "SV"],
                                     indices=[[0, 0], [0, 1]],
                                     shape=[1, 2])
      }
      with tf.variable_scope(
          "weighted_sum_from_feature_columns",
          features.values(),
          partitioner=tf.min_max_variable_partitioner(
              max_partitions=10, min_slice_size=((64 << 20) - 1))) as scope:
        output, column_to_variable, _ = (
            tf.contrib.layers.weighted_sum_from_feature_columns(
                features, [country, language, country_language],
                num_outputs=1,
                scope=scope))
      with self.test_session() as sess:
        tf.global_variables_initializer().run()
        tf.initialize_all_tables().run()

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
    country = tf.contrib.layers.sparse_column_with_hash_bucket(
        "country", hash_bucket_size=5)
    age = tf.contrib.layers.real_valued_column("age")
    # The following RealValuedColumn has 3 dimensions.
    incomes = tf.contrib.layers.real_valued_column("incomes", 3)

    with tf.Graph().as_default():
      features = {"age": tf.constant([[1], [1]]),
                  "incomes": tf.constant([[100., 200., 300.], [10., 20., 30.]]),
                  "country": tf.SparseTensor(values=["US", "SV"],
                                             indices=[[0, 0], [1, 0]],
                                             shape=[2, 2])}
      output, column_to_variable, _ = (
          tf.contrib.layers.weighted_sum_from_feature_columns(
              features, [country, age, incomes],
              num_outputs=1))
      with self.test_session() as sess:
        tf.global_variables_initializer().run()
        tf.initialize_all_tables().run()

        incomes_weights = column_to_variable[incomes][0]
        sess.run(incomes_weights.assign([[0.1], [0.2], [0.3]]))
        self.assertAllClose(output.eval(), [[140.], [14.]])

  def testMulticlassWithRealValuedColumnHavingMultiDimensions(self):
    country = tf.contrib.layers.sparse_column_with_hash_bucket(
        "country", hash_bucket_size=5)
    age = tf.contrib.layers.real_valued_column("age")
    # The following RealValuedColumn has 3 dimensions.
    incomes = tf.contrib.layers.real_valued_column("incomes", 3)
    with tf.Graph().as_default():
      features = {"age": tf.constant([[1], [1]]),
                  "incomes": tf.constant([[100., 200., 300.], [10., 20., 30.]]),
                  "country": tf.SparseTensor(values=["US", "SV"],
                                             indices=[[0, 0], [1, 0]],
                                             shape=[2, 2])}
      output, column_to_variable, _ = (
          tf.contrib.layers.weighted_sum_from_feature_columns(
              features, [country, age, incomes],
              num_outputs=5))
      with self.test_session() as sess:
        tf.global_variables_initializer().run()
        tf.initialize_all_tables().run()

        incomes_weights = column_to_variable[incomes][0]
        sess.run(incomes_weights.assign([[0.01, 0.1, 1., 10., 100.],
                                         [0.02, 0.2, 2., 20., 200.],
                                         [0.03, 0.3, 3., 30., 300.]]))
        self.assertAllClose(output.eval(), [[14., 140., 1400., 14000., 140000.],
                                            [1.4, 14., 140., 1400., 14000.]])

  def testBucketizedColumn(self):
    bucket = tf.contrib.layers.bucketized_column(
        tf.contrib.layers.real_valued_column("price"),
        boundaries=[0., 10., 100.])
    with tf.Graph().as_default():
      # buckets 2, 3, 0
      features = {"price": tf.constant([[20.], [110], [-3]])}
      output, column_to_variable, _ = (
          tf.contrib.layers.weighted_sum_from_feature_columns(features,
                                                              [bucket],
                                                              num_outputs=1))
      with self.test_session() as sess:
        tf.global_variables_initializer().run()
        tf.initialize_all_tables().run()

        sess.run(column_to_variable[bucket][0].assign([[0.1], [0.2], [0.3], [0.4
                                                                            ]]))
        self.assertAllClose(output.eval(), [[0.3], [0.4], [0.1]])

  def testBucketizedColumnHavingMultiDimensions(self):
    country = tf.contrib.layers.sparse_column_with_hash_bucket(
        "country", hash_bucket_size=5)
    bucket = tf.contrib.layers.bucketized_column(
        tf.contrib.layers.real_valued_column("price", 2),
        boundaries=[0., 10., 100.])
    with tf.Graph().as_default():
      # buckets 2, 3, 0
      features = {"price": tf.constant([[20., 210], [110, 50], [-3, -30]]),
                  "country": tf.SparseTensor(values=["US", "SV"],
                                             indices=[[0, 0], [1, 0]],
                                             shape=[3, 2])}
      output, column_to_variable, _ = (
          tf.contrib.layers.weighted_sum_from_feature_columns(features,
                                                              [bucket, country],
                                                              num_outputs=1))
      with self.test_session() as sess:
        tf.global_variables_initializer().run()
        tf.initialize_all_tables().run()

        # dimension = 2, bucket_size = 4, num_classes = 1
        sess.run(column_to_variable[bucket][0].assign(
            [[0.1], [0.2], [0.3], [0.4], [1], [2], [3], [4]]))
        self.assertAllClose(output.eval(), [[0.3 + 4], [0.4 + 3], [0.1 + 1]])

  def testMulticlassWithBucketizedColumnHavingMultiDimensions(self):
    country = tf.contrib.layers.sparse_column_with_hash_bucket(
        "country", hash_bucket_size=5)
    bucket = tf.contrib.layers.bucketized_column(
        tf.contrib.layers.real_valued_column("price", 2),
        boundaries=[0., 10., 100.])
    with tf.Graph().as_default():
      # buckets 2, 3, 0
      features = {"price": tf.constant([[20., 210], [110, 50], [-3, -30]]),
                  "country": tf.SparseTensor(values=["US", "SV"],
                                             indices=[[0, 0], [1, 0]],
                                             shape=[3, 2])}
      output, column_to_variable, _ = (
          tf.contrib.layers.weighted_sum_from_feature_columns(features,
                                                              [bucket, country],
                                                              num_outputs=5))
      with self.test_session() as sess:
        tf.global_variables_initializer().run()
        tf.initialize_all_tables().run()

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
    price_bucket = tf.contrib.layers.bucketized_column(
        tf.contrib.layers.real_valued_column("price"),
        boundaries=[0., 10., 100.])
    country = tf.contrib.layers.sparse_column_with_hash_bucket(
        "country", hash_bucket_size=5)
    country_price = tf.contrib.layers.crossed_column(
        [country, price_bucket], hash_bucket_size=10)
    with tf.Graph().as_default():
      features = {
          "price": tf.constant([[20.]]),
          "country": tf.SparseTensor(values=["US", "SV"],
                                     indices=[[0, 0], [0, 1]],
                                     shape=[1, 2])
      }
      output, column_to_variable, _ = (
          tf.contrib.layers.weighted_sum_from_feature_columns(features,
                                                              [country_price],
                                                              num_outputs=1))
      with self.test_session() as sess:
        tf.global_variables_initializer().run()
        tf.initialize_all_tables().run()

        weights = column_to_variable[country_price][0]
        sess.run(weights.assign(weights + 0.4))
        # There are two crosses each with 0.4 weight.
        # score = 0.4 + 0.4
        self.assertAllClose(output.eval(), [[0.8]])

  def testCrossWithCrossedColumn(self):
    price_bucket = tf.contrib.layers.bucketized_column(
        tf.contrib.layers.real_valued_column("price"),
        boundaries=[0., 10., 100.])
    language = tf.contrib.layers.sparse_column_with_hash_bucket(
        "language", hash_bucket_size=3)
    country = tf.contrib.layers.sparse_column_with_hash_bucket(
        "country", hash_bucket_size=5)
    country_language = tf.contrib.layers.crossed_column(
        [language, country], hash_bucket_size=10)
    country_language_price = tf.contrib.layers.crossed_column(
        set([country_language, price_bucket]),
        hash_bucket_size=15)
    with tf.Graph().as_default():
      features = {
          "price": tf.constant([[20.]]),
          "country": tf.SparseTensor(values=["US", "SV"],
                                     indices=[[0, 0], [0, 1]],
                                     shape=[1, 2]),
          "language": tf.SparseTensor(values=["english", "spanish"],
                                      indices=[[0, 0], [0, 1]],
                                      shape=[1, 2])
      }
      output, column_to_variable, _ = (
          tf.contrib.layers.weighted_sum_from_feature_columns(
              features, [country_language_price],
              num_outputs=1))
      with self.test_session() as sess:
        tf.global_variables_initializer().run()
        tf.initialize_all_tables().run()

        weights = column_to_variable[country_language_price][0]
        sess.run(weights.assign(weights + 0.4))
        # There are two crosses each with 0.4 weight.
        # score = 0.4 + 0.4 + 0.4 + 0.4
        self.assertAllClose(output.eval(), [[1.6]])

  def testIntegerizedColumn(self):
    product = tf.contrib.layers.sparse_column_with_integerized_feature(
        "product", bucket_size=5)
    with tf.Graph().as_default():
      features = {"product": tf.SparseTensor(values=[0, 4, 2],
                                             indices=[[0, 0], [1, 0], [2, 0]],
                                             shape=[3, 1])}
      output, column_to_variable, _ = (
          tf.contrib.layers.weighted_sum_from_feature_columns(features,
                                                              [product],
                                                              num_outputs=1))
      with self.test_session() as sess:
        tf.global_variables_initializer().run()
        tf.initialize_all_tables().run()
        product_weights = column_to_variable[product][0]
        sess.run(product_weights.assign([[0.1], [0.2], [0.3], [0.4], [0.5]]))
        self.assertAllClose(output.eval(), [[0.1], [0.5], [0.3]])

  def testIntegerizedColumnWithDenseInputTensor(self):
    product = tf.contrib.layers.sparse_column_with_integerized_feature(
        "product", bucket_size=5)
    with tf.Graph().as_default():
      features = {"product": tf.constant([[0], [4], [2]])}
      output, column_to_variable, _ = (
          tf.contrib.layers.weighted_sum_from_feature_columns(features,
                                                              [product],
                                                              num_outputs=1))
      with self.test_session() as sess:
        tf.initialize_all_variables().run()
        tf.initialize_all_tables().run()
        product_weights = column_to_variable[product][0]
        sess.run(product_weights.assign([[0.1], [0.2], [0.3], [0.4], [0.5]]))
        self.assertAllClose(output.eval(), [[0.1], [0.5], [0.3]])

  def testIntegerizedColumnWithDenseInputTensor2(self):
    product = tf.contrib.layers.sparse_column_with_integerized_feature(
        "product", bucket_size=5)
    with tf.Graph().as_default():
      features = {"product": tf.constant([[0, 4], [2, 3]])}
      output, column_to_variable, _ = (
          tf.contrib.layers.weighted_sum_from_feature_columns(features,
                                                              [product],
                                                              num_outputs=1))
      with self.test_session() as sess:
        tf.initialize_all_variables().run()
        tf.initialize_all_tables().run()
        product_weights = column_to_variable[product][0]
        sess.run(product_weights.assign([[0.1], [0.2], [0.3], [0.4], [0.5]]))
        self.assertAllClose(output.eval(), [[0.6], [0.7]])

  def testIntegerizedColumnWithInvalidId(self):
    product = tf.contrib.layers.sparse_column_with_integerized_feature(
        "product", bucket_size=5)
    with tf.Graph().as_default():
      features = {"product": tf.SparseTensor(values=[5, 4, 7],
                                             indices=[[0, 0], [1, 0], [2, 0]],
                                             shape=[3, 1])}
      output, column_to_variable, _ = (
          tf.contrib.layers.weighted_sum_from_feature_columns(features,
                                                              [product],
                                                              num_outputs=1))
      with self.test_session() as sess:
        tf.global_variables_initializer().run()
        tf.initialize_all_tables().run()
        product_weights = column_to_variable[product][0]
        sess.run(product_weights.assign([[0.1], [0.2], [0.3], [0.4], [0.5]]))
        self.assertAllClose(output.eval(), [[0.1], [0.5], [0.3]])

  def testMulticlassWithOnlyBias(self):
    with tf.Graph().as_default():
      features = {"age": tf.constant([[10.], [20.], [30.], [40.]])}
      output, _, bias = tf.contrib.layers.weighted_sum_from_feature_columns(
          features, [tf.contrib.layers.real_valued_column("age")],
          num_outputs=3)
      with self.test_session() as sess:
        tf.global_variables_initializer().run()
        tf.initialize_all_tables().run()
        sess.run(bias.assign([0.1, 0.2, 0.3]))
        self.assertAllClose(output.eval(), [[0.1, 0.2, 0.3], [0.1, 0.2, 0.3],
                                            [0.1, 0.2, 0.3], [0.1, 0.2, 0.3]])

  def testMulticlassWithRealValuedColumn(self):
    with tf.Graph().as_default():
      column = tf.contrib.layers.real_valued_column("age")
      features = {"age": tf.constant([[10.], [20.], [30.], [40.]])}
      output, column_to_variable, _ = (
          tf.contrib.layers.weighted_sum_from_feature_columns(features,
                                                              [column],
                                                              num_outputs=3))
      with self.test_session() as sess:
        tf.global_variables_initializer().run()
        tf.initialize_all_tables().run()
        weights = column_to_variable[column][0]
        self.assertEqual(weights.get_shape(), (1, 3))
        sess.run(weights.assign([[0.01, 0.03, 0.05]]))
        self.assertAllClose(output.eval(), [[0.1, 0.3, 0.5], [0.2, 0.6, 1.0],
                                            [0.3, 0.9, 1.5], [0.4, 1.2, 2.0]])

  def testMulticlassWithSparseColumn(self):
    with tf.Graph().as_default():
      column = tf.contrib.layers.sparse_column_with_keys(
          column_name="language",
          keys=["english", "arabic", "hindi", "russian", "swahili"])
      features = {
          "language": tf.SparseTensor(
              values=["hindi", "english", "arabic", "russian"],
              indices=[[0, 0], [1, 0], [2, 0], [3, 0]],
              shape=[4, 1])
      }
      output, column_to_variable, _ = (
          tf.contrib.layers.weighted_sum_from_feature_columns(features,
                                                              [column],
                                                              num_outputs=3))
      with self.test_session() as sess:
        tf.global_variables_initializer().run()
        tf.initialize_all_tables().run()
        weights = column_to_variable[column][0]
        self.assertEqual(weights.get_shape(), (5, 3))
        sess.run(weights.assign([[0.1, 0.4, 0.7], [0.2, 0.5, 0.8],
                                 [0.3, 0.6, 0.9], [0.4, 0.7, 1.0], [0.5, 0.8,
                                                                    1.1]]))
        self.assertAllClose(output.eval(), [[0.3, 0.6, 0.9], [0.1, 0.4, 0.7],
                                            [0.2, 0.5, 0.8], [0.4, 0.7, 1.0]])

  def testMulticlassWithBucketizedColumn(self):
    column = tf.contrib.layers.bucketized_column(
        tf.contrib.layers.real_valued_column("price"),
        boundaries=[0., 100., 500., 1000.])
    with tf.Graph().as_default():
      # buckets 0, 2, 1, 2
      features = {"price": tf.constant([[-3], [110], [20.], [210]])}
      output, column_to_variable, _ = (
          tf.contrib.layers.weighted_sum_from_feature_columns(features,
                                                              [column],
                                                              num_outputs=3))
      with self.test_session() as sess:
        tf.global_variables_initializer().run()
        tf.initialize_all_tables().run()

        weights = column_to_variable[column][0]
        self.assertEqual(weights.get_shape(), (5, 3))
        sess.run(weights.assign([[0.1, 0.4, 0.7], [0.2, 0.5, 0.8],
                                 [0.3, 0.6, 0.9], [0.4, 0.7, 1.0], [0.5, 0.8,
                                                                    1.1]]))
        self.assertAllClose(output.eval(), [[0.1, 0.4, 0.7], [0.3, 0.6, 0.9],
                                            [0.2, 0.5, 0.8], [0.3, 0.6, 0.9]])

  def testMulticlassWithCrossedColumn(self):
    language = tf.contrib.layers.sparse_column_with_hash_bucket(
        "language", hash_bucket_size=3)
    country = tf.contrib.layers.sparse_column_with_hash_bucket(
        "country", hash_bucket_size=2)
    column = tf.contrib.layers.crossed_column(
        {language, country}, hash_bucket_size=5)
    with tf.Graph().as_default():
      features = {
          "language": tf.SparseTensor(
              values=["english", "spanish", "russian", "swahili"],
              indices=[[0, 0], [1, 0], [2, 0], [3, 0]],
              shape=[4, 1]),
          "country": tf.SparseTensor(values=["US", "SV", "RU", "KE"],
                                     indices=[[0, 0], [1, 0], [2, 0], [3, 0]],
                                     shape=[4, 1])
      }
      output, column_to_variable, _ = (
          tf.contrib.layers.weighted_sum_from_feature_columns(features,
                                                              [column],
                                                              num_outputs=3))
      with self.test_session() as sess:
        tf.global_variables_initializer().run()
        tf.initialize_all_tables().run()

        weights = column_to_variable[column][0]
        self.assertEqual(weights.get_shape(), (5, 3))
        sess.run(weights.assign([[0.1, 0.4, 0.7], [0.2, 0.5, 0.8],
                                 [0.3, 0.6, 0.9], [0.4, 0.7, 1.0], [0.5, 0.8,
                                                                    1.1]]))
        self.assertAllClose(tf.shape(output).eval(), [4, 3])

  def testMulticlassWithMultivalentColumn(self):
    column = tf.contrib.layers.sparse_column_with_keys(
        column_name="language",
        keys=["english", "turkish", "hindi", "russian", "swahili"])
    with tf.Graph().as_default():
      features = {
          "language": tf.SparseTensor(
              values=["hindi", "english", "turkish", "turkish", "english"],
              indices=[[0, 0], [0, 1], [1, 0], [2, 0], [3, 0]],
              shape=[4, 2])
      }
      output, column_to_variable, _ = (
          tf.contrib.layers.weighted_sum_from_feature_columns(features,
                                                              [column],
                                                              num_outputs=3))
      with self.test_session() as sess:
        tf.global_variables_initializer().run()
        tf.initialize_all_tables().run()

        weights = column_to_variable[column][0]
        self.assertEqual(weights.get_shape(), (5, 3))
        sess.run(weights.assign([[0.1, 0.4, 0.7], [0.2, 0.5, 0.8],
                                 [0.3, 0.6, 0.9], [0.4, 0.7, 1.0], [0.5, 0.8,
                                                                    1.1]]))
        self.assertAllClose(output.eval(), [[0.4, 1.0, 1.6], [0.2, 0.5, 0.8],
                                            [0.2, 0.5, 0.8], [0.1, 0.4, 0.7]])

  def testVariablesAddedToCollection(self):
    price_bucket = tf.contrib.layers.bucketized_column(
        tf.contrib.layers.real_valued_column("price"),
        boundaries=[0., 10., 100.])
    country = tf.contrib.layers.sparse_column_with_hash_bucket(
        "country", hash_bucket_size=5)
    country_price = tf.contrib.layers.crossed_column(
        [country, price_bucket], hash_bucket_size=10)
    with tf.Graph().as_default():
      features = {
          "price": tf.constant([[20.]]),
          "country": tf.SparseTensor(values=["US", "SV"],
                                     indices=[[0, 0], [0, 1]],
                                     shape=[1, 2])
      }
      tf.contrib.layers.weighted_sum_from_feature_columns(
          features, [country_price, price_bucket],
          num_outputs=1,
          weight_collections=["my_collection"])
      weights = tf.get_collection("my_collection")
      # 3 = bias + price_bucket + country_price
      self.assertEqual(3, len(weights))


class ParseExampleTest(tf.test.TestCase):

  def testParseExample(self):
    bucket = tf.contrib.layers.bucketized_column(
        tf.contrib.layers.real_valued_column("price", dimension=3),
        boundaries=[0., 10., 100.])
    wire_cast = tf.contrib.layers.sparse_column_with_keys(
        "wire_cast", ["marlo", "omar", "stringer"])
    # buckets 2, 3, 0
    data = tf.train.Example(features=tf.train.Features(feature={
        "price": tf.train.Feature(float_list=tf.train.FloatList(value=[20., 110,
                                                                       -3])),
        "wire_cast": tf.train.Feature(bytes_list=tf.train.BytesList(value=[
            b"stringer", b"marlo"
        ])),
    }))
    output = tf.contrib.layers.parse_feature_columns_from_examples(
        serialized=[data.SerializeToString()],
        feature_columns=[bucket, wire_cast])
    self.assertIn(bucket, output)
    self.assertIn(wire_cast, output)
    with self.test_session():
      tf.initialize_all_tables().run()
      self.assertAllEqual(output[bucket].eval(), [[2, 3, 0]])
      self.assertAllEqual(output[wire_cast].indices.eval(), [[0, 0], [0, 1]])
      self.assertAllEqual(output[wire_cast].values.eval(), [2, 0])


  def testParseSequenceExample(self):
    location_keys = ["east_side", "west_side", "nyc"]
    embedding_dimension = 10


    location = tf.contrib.layers.sparse_column_with_keys(
        "location", keys=location_keys)
    location_onehot = tf.contrib.layers.one_hot_column(location)
    wire_cast = tf.contrib.layers.sparse_column_with_keys(
        "wire_cast", ["marlo", "omar", "stringer"])
    wire_cast_embedded = tf.contrib.layers.embedding_column(
        wire_cast, dimension=embedding_dimension)
    measurements = tf.contrib.layers.real_valued_column("measurements", dimension=2)

    context_feature_columns = [location_onehot]
    sequence_feature_columns = [wire_cast_embedded, measurements]

    sequence_example = tf.train.SequenceExample(
        context=tf.train.Features(feature={
            "location": tf.train.Feature(
                bytes_list=tf.train.BytesList(
                    value=[b"west_side"])),
        }),
        feature_lists=tf.train.FeatureLists(feature_list={
            "wire_cast": tf.train.FeatureList(feature=[
                tf.train.Feature(bytes_list=tf.train.BytesList(
                    value=[b"marlo", b"stringer"])),
                tf.train.Feature(bytes_list=tf.train.BytesList(
                    value=[b"omar", b"stringer", b"marlo"])),
                tf.train.Feature(bytes_list=tf.train.BytesList(
                    value=[b"marlo"])),

            ]),
            "measurements": tf.train.FeatureList(feature=[
                tf.train.Feature(float_list=tf.train.FloatList(
                    value=[0.2, 0.3])),
                tf.train.Feature(float_list=tf.train.FloatList(
                    value=[0.1, 0.8])),
                tf.train.Feature(float_list=tf.train.FloatList(
                    value=[0.5, 0.0])),
            ])
        }))


    ctx, seq = tf.contrib.layers.parse_feature_columns_from_sequence_examples(
         serialized=sequence_example.SerializeToString(),
         context_feature_columns=context_feature_columns,
         sequence_feature_columns=sequence_feature_columns)

    self.assertIn("location", ctx)
    self.assertIsInstance(ctx["location"], tf.SparseTensor)
    self.assertIn("wire_cast", seq)
    self.assertIsInstance(seq["wire_cast"], tf.SparseTensor)
    self.assertIn("measurements", seq)
    self.assertIsInstance(seq["measurements"], tf.Tensor)

    with self.test_session() as sess:
      location_val, wire_cast_val, measurement_val = sess.run([
          ctx["location"], seq["wire_cast"], seq["measurements"]])

    self.assertAllEqual(location_val.indices, np.array([[0]]))
    self.assertAllEqual(location_val.values, np.array([b"west_side"]))
    self.assertAllEqual(location_val.shape, np.array([1]))

    self.assertAllEqual(wire_cast_val.indices, np.array(
        [[0, 0], [0, 1], [1, 0], [1, 1], [1, 2], [2, 0]]))
    self.assertAllEqual(wire_cast_val.values, np.array(
        [b"marlo", b"stringer", b"omar", b"stringer", b"marlo", b"marlo"]))
    self.assertAllEqual(wire_cast_val.shape, np.array([3, 3]))

    self.assertAllClose(
        measurement_val, np.array([[0.2, 0.3], [0.1, 0.8], [0.5, 0.0]]))

class InferRealValuedColumnTest(tf.test.TestCase):

  def testTensorInt32(self):
    self.assertEqual(
        tf.contrib.layers.infer_real_valued_columns(
            tf.zeros(shape=[33, 4], dtype=tf.int32)),
        [tf.contrib.layers.real_valued_column("", dimension=4, dtype=tf.int32)])

  def testTensorInt64(self):
    self.assertEqual(
        tf.contrib.layers.infer_real_valued_columns(
            tf.zeros(shape=[33, 4], dtype=tf.int64)),
        [tf.contrib.layers.real_valued_column("", dimension=4, dtype=tf.int64)])

  def testTensorFloat32(self):
    self.assertEqual(
        tf.contrib.layers.infer_real_valued_columns(
            tf.zeros(shape=[33, 4], dtype=tf.float32)),
        [tf.contrib.layers.real_valued_column(
            "", dimension=4, dtype=tf.float32)])

  def testTensorFloat64(self):
    self.assertEqual(
        tf.contrib.layers.infer_real_valued_columns(
            tf.zeros(shape=[33, 4], dtype=tf.float64)),
        [tf.contrib.layers.real_valued_column(
            "", dimension=4, dtype=tf.float64)])

  def testDictionary(self):
    self.assertItemsEqual(
        tf.contrib.layers.infer_real_valued_columns({
            "a": tf.zeros(shape=[33, 4], dtype=tf.int32),
            "b": tf.zeros(shape=[3, 2], dtype=tf.float32)
        }),
        [tf.contrib.layers.real_valued_column(
            "a", dimension=4, dtype=tf.int32),
         tf.contrib.layers.real_valued_column(
             "b", dimension=2, dtype=tf.float32)])

  def testNotGoodDtype(self):
    with self.assertRaises(ValueError):
      tf.contrib.layers.infer_real_valued_columns(
          tf.constant([["a"]], dtype=tf.string))

  def testSparseTensor(self):
    with self.assertRaises(ValueError):
      tf.contrib.layers.infer_real_valued_columns(
          tf.SparseTensor(indices=[[0, 0]], values=["a"], shape=[1, 1]))


if __name__ == "__main__":
  tf.test.main()

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

import tensorflow as tf

from tensorflow.contrib.layers.python.layers import feature_column_ops


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
    output = feature_column_ops._Transformer(features).transform(bucket)
    with self.test_session():
      self.assertAllEqual(output.eval(), [[2], [3], [0]])

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
    output = feature_column_ops._Transformer(features).transform(hashed_sparse)
    with self.test_session():
      self.assertEqual(output.values.dtype, tf.int64)
      self.assertTrue(all(x < 10 and x >= 0 for x in output.values.eval()))
      self.assertAllEqual(output.indices.eval(), wire_tensor.indices.eval())
      self.assertAllEqual(output.shape.eval(), wire_tensor.shape.eval())

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

  def testSparseColumnWithKeys(self):
    keys_sparse = tf.contrib.layers.sparse_column_with_keys(
        "wire", ["marlo", "omar", "stringer"])
    wire_tensor = tf.SparseTensor(values=["omar", "stringer", "marlo"],
                                  indices=[[0, 0], [1, 0], [1, 1]],
                                  shape=[2, 2])
    features = {"wire": wire_tensor}
    output = feature_column_ops._Transformer(features).transform(keys_sparse)
    with self.test_session():
      tf.initialize_all_tables().run()
      self.assertEqual(output.values.dtype, tf.int64)
      self.assertAllEqual(output.values.eval(), [1, 2, 0])
      self.assertAllEqual(output.indices.eval(), wire_tensor.indices.eval())
      self.assertAllEqual(output.shape.eval(), wire_tensor.shape.eval())

  def testSparseColumnWithHashBucket_IsIntegerized(self):
    hashed_sparse = tf.contrib.layers.sparse_column_with_integerized_feature(
        "wire", 10)
    wire_tensor = tf.SparseTensor(values=[100, 1, 25],
                                  indices=[[0, 0], [1, 0], [1, 1]],
                                  shape=[2, 2])
    features = {"wire": wire_tensor}
    output = feature_column_ops._Transformer(features).transform(hashed_sparse)
    with self.test_session():
      self.assertEqual(output.values.dtype, tf.int32)
      self.assertTrue(all(x < 10 and x >= 0 for x in output.values.eval()))
      self.assertAllEqual(output.indices.eval(), wire_tensor.indices.eval())
      self.assertAllEqual(output.shape.eval(), wire_tensor.shape.eval())

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
    output = feature_column_ops._Transformer(features).transform(
        country_language)
    with self.test_session():
      self.assertEqual(output.values.dtype, tf.int64)
      self.assertTrue(all(x < 15 and x >= 0 for x in output.values.eval()))

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
    output = feature_column_ops._Transformer(features).transform(country_price)
    with self.test_session():
      self.assertEqual(output.values.dtype, tf.int64)
      self.assertTrue(all(x < 15 and x >= 0 for x in output.values.eval()))

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
    output = feature_column_ops._Transformer(features).transform(
        wire_country_price)
    with self.test_session():
      self.assertEqual(output.values.dtype, tf.int64)
      self.assertTrue(all(x < 15 and x >= 0 for x in output.values.eval()))

  def testIfFeatureTableContainsTransfromationReturnIt(self):
    any_column = tf.contrib.layers.sparse_column_with_hash_bucket("sparse", 10)
    features = {any_column: "any-thing-even-not-a-tensor"}
    output = feature_column_ops._Transformer(features).transform(any_column)
    self.assertEqual(output, "any-thing-even-not-a-tensor")


class InputLayerTest(tf.test.TestCase):

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

  def testBucketizedColumn(self):
    bucket = tf.contrib.layers.bucketized_column(
        tf.contrib.layers.real_valued_column("price"),
        boundaries=[0., 10., 100.])
    # buckets 2, 3, 0
    features = {"price": tf.constant([[20.], [110], [-3]])}
    output = tf.contrib.layers.input_from_feature_columns(features, [bucket])
    expected = [[0, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0]]
    with self.test_session():
      self.assertAllClose(output.eval(), expected)

  def testBucketizedColumnWithMultiDimensions(self):
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

  def testEmbeddingColumn(self):
    hashed_sparse = tf.contrib.layers.sparse_column_with_hash_bucket("wire", 10)
    wire_tensor = tf.SparseTensor(values=["omar", "stringer", "marlo"],
                                  indices=[[0, 0], [1, 0], [1, 1]],
                                  shape=[2, 2])
    features = {"wire": wire_tensor}
    embeded_sparse = tf.contrib.layers.embedding_column(hashed_sparse, 10)
    output = tf.contrib.layers.input_from_feature_columns(features,
                                                          [embeded_sparse])
    with self.test_session():
      tf.initialize_all_variables().run()
      self.assertAllEqual(output.eval().shape, [2, 10])

  def testSparseColumn(self):
    hashed_sparse = tf.contrib.layers.sparse_column_with_hash_bucket("wire", 10)
    wire_tensor = tf.SparseTensor(values=["omar", "stringer", "marlo"],
                                  indices=[[0, 0], [1, 0], [1, 1]],
                                  shape=[2, 2])
    features = {"wire": wire_tensor}
    with self.assertRaises(ValueError):
      tf.initialize_all_variables().run()
      tf.contrib.layers.input_layer(features, [hashed_sparse])

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
    with self.assertRaises(ValueError):
      tf.initialize_all_variables().run()
      tf.contrib.layers.input_layer(features, [crossed])

  def testAllColumns(self):
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
    embeded_sparse = tf.contrib.layers.embedding_column(hashed_sparse, 10)
    output = tf.contrib.layers.input_from_feature_columns(
        features, [real_valued, bucket, embeded_sparse])
    with self.test_session():
      tf.initialize_all_variables().run()
      # size of output = 3 (real_valued) + 2 * 4 (bucket) + 10 (embedding) = 21
      self.assertAllEqual(output.eval().shape, [3, 21])

  def testInputLayerWithCollections(self):
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

  def testInputLayerWithTrainableArg(self):
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
      tf.initialize_all_variables().run()
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
      tf.initialize_all_variables().run()
      self.assertAllEqual(logits.eval().shape, [2, 5])

  def testEmbeddingColumn(self):
    hashed_sparse = tf.contrib.layers.sparse_column_with_hash_bucket("wire", 10)
    wire_tensor = tf.SparseTensor(values=["omar", "stringer", "marlo"],
                                  indices=[[0, 0], [1, 0], [1, 1]],
                                  shape=[2, 2])
    features = {"wire": wire_tensor}
    embeded_sparse = tf.contrib.layers.embedding_column(hashed_sparse, 10)
    with self.assertRaises(ValueError):
      tf.initialize_all_variables().run()
      tf.contrib.layers.weighted_sum_from_feature_columns(features,
                                                          [embeded_sparse])

  def testRealValuedColumnWithMultiDimensions(self):
    real_valued = tf.contrib.layers.real_valued_column("price", 2)
    features = {"price": tf.constant([[20., 10.], [110, 0.], [-3, 30]])}
    logits, _, _ = tf.contrib.layers.weighted_sum_from_feature_columns(
        features, [real_valued], num_outputs=5)
    with self.test_session():
      tf.initialize_all_variables().run()
      self.assertAllEqual(logits.eval().shape, [3, 5])

  def testBucketizedColumnWithMultiDimensions(self):
    bucket = tf.contrib.layers.bucketized_column(
        tf.contrib.layers.real_valued_column("price", 2),
        boundaries=[0., 10., 100.])
    features = {"price": tf.constant([[20., 10.], [110, 0.], [-3, 30]])}
    logits, _, _ = tf.contrib.layers.weighted_sum_from_feature_columns(
        features, [bucket], num_outputs=5)
    with self.test_session():
      tf.initialize_all_variables().run()
      self.assertAllEqual(logits.eval().shape, [3, 5])

  def testAllColumns(self):
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
      tf.initialize_all_variables().run()
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
        tf.initialize_all_variables().run()
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
        tf.initialize_all_variables().run()
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
        tf.initialize_all_variables().run()
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
        tf.initialize_all_variables().run()
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
        tf.initialize_all_variables().run()
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
        tf.initialize_all_variables().run()
        tf.initialize_all_tables().run()

        weights = column_to_variable[country_language][0]
        sess.run(weights.assign(weights + 0.4))
        # There are four crosses each with 0.4 weight.
        # score = 0.4 + 0.4 + 0.4 + 0.4
        self.assertAllClose(output.eval(), [[1.6]])

  def testMultivalentCrossUsageInPredictionsWithPartition(self):
    # bucket size has to be big enough to allwo sharding.
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
      output, column_to_variable, _ = (
          tf.contrib.layers.weighted_sum_from_feature_columns(
              features, [country, language, country_language],
              num_outputs=1))
      with self.test_session() as sess:
        tf.initialize_all_variables().run()
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
        tf.initialize_all_variables().run()
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
        tf.initialize_all_variables().run()
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
        tf.initialize_all_variables().run()
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
        tf.initialize_all_variables().run()
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
        tf.initialize_all_variables().run()
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
        tf.initialize_all_variables().run()
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
        tf.initialize_all_variables().run()
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
        tf.initialize_all_variables().run()
        tf.initialize_all_tables().run()
        product_weights = column_to_variable[product][0]
        sess.run(product_weights.assign([[0.1], [0.2], [0.3], [0.4], [0.5]]))
        self.assertAllClose(output.eval(), [[0.1], [0.5], [0.3]])

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
        tf.initialize_all_variables().run()
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
        tf.initialize_all_variables().run()
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
        tf.initialize_all_variables().run()
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
        tf.initialize_all_variables().run()
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
        tf.initialize_all_variables().run()
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
        tf.initialize_all_variables().run()
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
        tf.initialize_all_variables().run()
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


class InferRealValuedColumnTest(tf.test.TestCase):

  def testTensor(self):
    self.assertEqual(
        tf.contrib.layers.infer_real_valued_columns(
            tf.zeros(shape=[33, 4], dtype=tf.int32)),
        [tf.contrib.layers.real_valued_column("", dimension=4, dtype=tf.int32)])

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

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

import tensorflow as tf


class FeatureColumnTest(tf.test.TestCase):

  def testImmutability(self):
    a = tf.contrib.layers.sparse_column_with_hash_bucket("aaa",
                                                         hash_bucket_size=100)
    with self.assertRaises(AttributeError):
      a.column_name = "bbb"

  def testSparseColumn(self):
    a = tf.contrib.layers.sparse_column_with_hash_bucket("aaa",
                                                         hash_bucket_size=100)
    self.assertEqual(a.name, "aaa")

  def testEmbeddingColumn(self):
    a = tf.contrib.layers.sparse_column_with_hash_bucket("aaa",
                                                         hash_bucket_size=100,
                                                         combiner="sum")
    b = tf.contrib.layers.embedding_column(a, dimension=4, combiner="mean")
    self.assertEqual(b.sparse_id_column.name, "aaa")
    self.assertEqual(b.dimension, 4)
    self.assertEqual(b.combiner, "mean")

  def testRealValuedColumn(self):
    a = tf.contrib.layers.real_valued_column("aaa")
    self.assertEqual(a.name, "aaa")
    self.assertEqual(a.dimension, 1)
    b = tf.contrib.layers.real_valued_column("bbb", 10)
    self.assertEqual(b.dimension, 10)
    self.assertTrue(b.default_value is None)

    # default_value is an integer.
    c1 = tf.contrib.layers.real_valued_column("c1", default_value=2)
    self.assertListEqual(list(c1.default_value), [2.])
    c2 = tf.contrib.layers.real_valued_column("c2",
                                              default_value=2,
                                              dtype=tf.int32)
    self.assertListEqual(list(c2.default_value), [2])
    c3 = tf.contrib.layers.real_valued_column("c3",
                                              dimension=4,
                                              default_value=2)
    self.assertListEqual(list(c3.default_value), [2, 2, 2, 2])
    c4 = tf.contrib.layers.real_valued_column("c4",
                                              dimension=4,
                                              default_value=2,
                                              dtype=tf.int32)
    self.assertListEqual(list(c4.default_value), [2, 2, 2, 2])

    # default_value is a float.
    d1 = tf.contrib.layers.real_valued_column("d1", default_value=2.)
    self.assertListEqual(list(d1.default_value), [2.])
    d2 = tf.contrib.layers.real_valued_column("d2",
                                              dimension=4,
                                              default_value=2.)
    self.assertListEqual(list(d2.default_value), [2., 2., 2., 2.])
    with self.assertRaises(TypeError):
      tf.contrib.layers.real_valued_column("d3",
                                           default_value=2.,
                                           dtype=tf.int32)

    # default_value is neither interger nor float.
    with self.assertRaises(TypeError):
      tf.contrib.layers.real_valued_column("e1", default_value="string")
    with self.assertRaises(TypeError):
      tf.contrib.layers.real_valued_column("e1",
                                           dimension=3,
                                           default_value=[1, 3., "string"])

    # default_value is a list of integers.
    f1 = tf.contrib.layers.real_valued_column("f1", default_value=[2])
    self.assertListEqual(list(f1.default_value), [2])
    f2 = tf.contrib.layers.real_valued_column("f2",
                                              dimension=3,
                                              default_value=[2, 2, 2])
    self.assertListEqual(list(f2.default_value), [2., 2., 2.])
    f3 = tf.contrib.layers.real_valued_column("f3",
                                              dimension=3,
                                              default_value=[2, 2, 2],
                                              dtype=tf.int32)
    self.assertListEqual(list(f3.default_value), [2, 2, 2])

    # default_value is a list of floats.
    g1 = tf.contrib.layers.real_valued_column("g1", default_value=[2.])
    self.assertListEqual(list(g1.default_value), [2.])
    g2 = tf.contrib.layers.real_valued_column("g2",
                                              dimension=3,
                                              default_value=[2., 2, 2])
    self.assertListEqual(list(g2.default_value), [2., 2., 2.])
    with self.assertRaises(TypeError):
      tf.contrib.layers.real_valued_column("g3",
                                           default_value=[2.],
                                           dtype=tf.int32)
    with self.assertRaises(ValueError):
      tf.contrib.layers.real_valued_column("g4",
                                           dimension=3,
                                           default_value=[2.])

  def testBucketizedColumnNameEndsWithUnderscoreBucketized(self):
    a = tf.contrib.layers.bucketized_column(
        tf.contrib.layers.real_valued_column("aaa"), [0, 4])
    self.assertEqual(a.name, "aaa_BUCKETIZED")

  def testBucketizedColumnRequiresRealValuedColumn(self):
    with self.assertRaises(TypeError):
      tf.contrib.layers.bucketized_column("bbb", [0])

  def testBucketizedColumnRequiresSortedBuckets(self):
    with self.assertRaises(ValueError):
      tf.contrib.layers.bucketized_column(
          tf.contrib.layers.real_valued_column("ccc"), [5, 0, 4])

  def testBucketizedColumnWithSameBucketBoundaries(self):
    a_bucketized = tf.contrib.layers.bucketized_column(
        tf.contrib.layers.real_valued_column("a"), [1., 2., 2., 3., 3.])
    self.assertEqual(a_bucketized.name, "a_BUCKETIZED")
    self.assertTupleEqual(a_bucketized.boundaries, (1., 2., 3.))

  def testCrossedColumnNameCreatesSortedNames(self):
    a = tf.contrib.layers.sparse_column_with_hash_bucket("aaa",
                                                         hash_bucket_size=100)
    b = tf.contrib.layers.sparse_column_with_hash_bucket("bbb",
                                                         hash_bucket_size=100)
    bucket = tf.contrib.layers.bucketized_column(
        tf.contrib.layers.real_valued_column("cost"), [0, 4])
    crossed = tf.contrib.layers.crossed_column(
        set([b, bucket, a]), hash_bucket_size=10000)

    self.assertEqual("aaa_X_bbb_X_cost_BUCKETIZED", crossed.name,
                     "name should be generated by sorted column names")
    self.assertEqual("aaa", crossed.columns[0].name)
    self.assertEqual("bbb", crossed.columns[1].name)
    self.assertEqual("cost_BUCKETIZED", crossed.columns[2].name)

  def testCrossedColumnNotSupportRealValuedColumn(self):
    b = tf.contrib.layers.sparse_column_with_hash_bucket("bbb",
                                                         hash_bucket_size=100)
    with self.assertRaises(TypeError):
      tf.contrib.layers.crossed_column(
          set([b, tf.contrib.layers.real_valued_column("real")]),
          hash_bucket_size=10000)

  def testRealValuedColumnDtypes(self):
    rvc = tf.contrib.layers.real_valued_column("rvc")
    self.assertDictEqual(
        {"rvc": tf.FixedLenFeature(
            [1], dtype=tf.float32)},
        rvc.config)

    rvc = tf.contrib.layers.real_valued_column("rvc", dtype=tf.int32)
    self.assertDictEqual(
        {"rvc": tf.FixedLenFeature(
            [1], dtype=tf.int32)},
        rvc.config)

    with self.assertRaises(ValueError):
      tf.contrib.layers.real_valued_column("rvc", dtype=tf.string)

  def testSparseColumnDtypes(self):
    sc = tf.contrib.layers.sparse_column_with_integerized_feature("sc", 10)
    self.assertDictEqual({"sc": tf.VarLenFeature(dtype=tf.int64)}, sc.config)

    sc = tf.contrib.layers.sparse_column_with_integerized_feature(
        "sc", 10, dtype=tf.int32)
    self.assertDictEqual({"sc": tf.VarLenFeature(dtype=tf.int32)}, sc.config)

    with self.assertRaises(ValueError):
      tf.contrib.layers.sparse_column_with_integerized_feature("sc",
                                                               10,
                                                               dtype=tf.float32)

  def testCreateFeatureSpec(self):
    sparse_col = tf.contrib.layers.sparse_column_with_hash_bucket(
        "sparse_column", hash_bucket_size=100)
    embedding_col = tf.contrib.layers.embedding_column(
        tf.contrib.layers.sparse_column_with_hash_bucket(
            "sparse_column_for_embedding",
            hash_bucket_size=10),
        dimension=4)
    real_valued_col1 = tf.contrib.layers.real_valued_column(
        "real_valued_column1")
    real_valued_col2 = tf.contrib.layers.real_valued_column(
        "real_valued_column2", 5)
    bucketized_col1 = tf.contrib.layers.bucketized_column(
        tf.contrib.layers.real_valued_column(
            "real_valued_column_for_bucketization1"), [0, 4])
    bucketized_col2 = tf.contrib.layers.bucketized_column(
        tf.contrib.layers.real_valued_column(
            "real_valued_column_for_bucketization2", 4), [0, 4])
    a = tf.contrib.layers.sparse_column_with_hash_bucket("cross_aaa",
                                                         hash_bucket_size=100)
    b = tf.contrib.layers.sparse_column_with_hash_bucket("cross_bbb",
                                                         hash_bucket_size=100)
    cross_col = tf.contrib.layers.crossed_column(
        set([a, b]), hash_bucket_size=10000)
    feature_columns = set([sparse_col, embedding_col,
                           real_valued_col1, real_valued_col2,
                           bucketized_col1, bucketized_col2,
                           cross_col])
    config = tf.contrib.layers.create_feature_spec_for_parsing(feature_columns)
    self.assertDictEqual({
        "sparse_column": tf.VarLenFeature(tf.string),
        "sparse_column_for_embedding": tf.VarLenFeature(tf.string),
        "real_valued_column1": tf.FixedLenFeature([1], dtype=tf.float32),
        "real_valued_column2": tf.FixedLenFeature([5], dtype=tf.float32),
        "real_valued_column_for_bucketization1":
            tf.FixedLenFeature([1], dtype=tf.float32),
        "real_valued_column_for_bucketization2":
            tf.FixedLenFeature([4], dtype=tf.float32),
        "cross_aaa": tf.VarLenFeature(tf.string),
        "cross_bbb": tf.VarLenFeature(tf.string)}, config)

  def testCreateFeatureSpec_RealValuedColumnWithDefaultValue(self):
    real_valued_col1 = tf.contrib.layers.real_valued_column(
        "real_valued_column1", default_value=2)
    real_valued_col2 = tf.contrib.layers.real_valued_column(
        "real_valued_column2", 5, default_value=4)
    real_valued_col3 = tf.contrib.layers.real_valued_column(
        "real_valued_column3", default_value=[8])
    real_valued_col4 = tf.contrib.layers.real_valued_column(
        "real_valued_column4", 3,
        default_value=[1, 0, 6])
    feature_columns = [real_valued_col1, real_valued_col2,
                       real_valued_col3, real_valued_col4]
    config = tf.contrib.layers.create_feature_spec_for_parsing(feature_columns)
    self.assertEqual(4, len(config))
    self.assertDictEqual({
        "real_valued_column1":
            tf.FixedLenFeature([1], dtype=tf.float32, default_value=[2.]),
        "real_valued_column2":
            tf.FixedLenFeature([5], dtype=tf.float32,
                               default_value=[4., 4., 4., 4., 4.]),
        "real_valued_column3":
            tf.FixedLenFeature([1], dtype=tf.float32, default_value=[8.]),
        "real_valued_column4":
            tf.FixedLenFeature([3], dtype=tf.float32,
                               default_value=[1., 0., 6.])}, config)

  def testMakePlaceHolderTensorsForBaseFeatures(self):
    sparse_col = tf.contrib.layers.sparse_column_with_hash_bucket(
        "sparse_column", hash_bucket_size=100)
    real_valued_col = tf.contrib.layers.real_valued_column("real_valued_column",
                                                           5)
    bucketized_col = tf.contrib.layers.bucketized_column(
        tf.contrib.layers.real_valued_column(
            "real_valued_column_for_bucketization"), [0, 4])
    feature_columns = set([sparse_col, real_valued_col, bucketized_col])
    placeholders = (
        tf.contrib.layers.make_place_holder_tensors_for_base_features(
            feature_columns))

    self.assertEqual(3, len(placeholders))
    self.assertTrue(isinstance(placeholders["sparse_column"],
                               tf.SparseTensor))
    placeholder = placeholders["real_valued_column"]
    self.assertTrue(placeholder.name.startswith(u"Placeholder"))
    self.assertEqual(tf.float32, placeholder.dtype)
    self.assertEqual([None, 5], placeholder.get_shape().as_list())
    placeholder = placeholders["real_valued_column_for_bucketization"]
    self.assertTrue(placeholder.name.startswith(u"Placeholder"))
    self.assertEqual(tf.float32, placeholder.dtype)
    self.assertEqual([None, 1], placeholder.get_shape().as_list())


if __name__ == "__main__":
  tf.test.main()

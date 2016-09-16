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

import os

import tensorflow as tf


class FeatureColumnTest(tf.test.TestCase):

  def testImmutability(self):
    a = tf.contrib.layers.sparse_column_with_hash_bucket("aaa",
                                                         hash_bucket_size=100)
    with self.assertRaises(AttributeError):
      a.column_name = "bbb"

  def testSparseColumnWithHashBucket(self):
    a = tf.contrib.layers.sparse_column_with_hash_bucket("aaa",
                                                         hash_bucket_size=100)
    self.assertEqual(a.name, "aaa")
    self.assertEqual(a.dtype, tf.string)

    a = tf.contrib.layers.sparse_column_with_hash_bucket("aaa",
                                                         hash_bucket_size=100,
                                                         dtype=tf.int64)
    self.assertEqual(a.name, "aaa")
    self.assertEqual(a.dtype, tf.int64)

    with self.assertRaisesRegexp(ValueError,
                                 "dtype must be string or integer"):
      a = tf.contrib.layers.sparse_column_with_hash_bucket("aaa",
                                                           hash_bucket_size=100,
                                                           dtype=tf.float32)

  def testWeightedSparseColumn(self):
    ids = tf.contrib.layers.sparse_column_with_keys(
        "ids", ["marlo", "omar", "stringer"])
    weighted_ids = tf.contrib.layers.weighted_sparse_column(ids, "weights")
    self.assertEqual(weighted_ids.name, "ids_weighted_by_weights")

  def testEmbeddingColumn(self):
    a = tf.contrib.layers.sparse_column_with_hash_bucket("aaa",
                                                         hash_bucket_size=100,
                                                         combiner="sum")
    b = tf.contrib.layers.embedding_column(a, dimension=4, combiner="mean")
    self.assertEqual(b.sparse_id_column.name, "aaa")
    self.assertEqual(b.dimension, 4)
    self.assertEqual(b.combiner, "mean")

  def testSharedEmbeddingColumn(self):
    a1 = tf.contrib.layers.sparse_column_with_keys(
        "a1", ["marlo", "omar", "stringer"])
    a2 = tf.contrib.layers.sparse_column_with_keys(
        "a2", ["marlo", "omar", "stringer"])
    b = tf.contrib.layers.shared_embedding_columns(
        [a1, a2], dimension=4, combiner="mean")
    self.assertEqual(len(b), 2)
    self.assertEqual(b[0].shared_embedding_name, "a1_a2_shared_embedding")
    self.assertEqual(b[1].shared_embedding_name, "a1_a2_shared_embedding")

    # Create a sparse id tensor for a1.
    input_tensor_c1 = tf.SparseTensor(indices=[[0, 0], [1, 1], [2, 2]],
                                      values=[0, 1, 2], shape=[3, 3])
    # Create a sparse id tensor for a2.
    input_tensor_c2 = tf.SparseTensor(indices=[[0, 0], [1, 1], [2, 2]],
                                      values=[0, 1, 2], shape=[3, 3])
    with tf.variable_scope("run_1"):
      b1 = b[0].to_dnn_input_layer(input_tensor_c1)
      b2 = b[1].to_dnn_input_layer(input_tensor_c2)
    with self.test_session() as sess:
      sess.run(tf.initialize_all_variables())
      b1_value = b1.eval()
      b2_value = b2.eval()
    for i in range(len(b1_value)):
      self.assertAllClose(b1_value[i], b2_value[i])

    # Test the case when a shared_embedding_name is explictly specified.
    d = tf.contrib.layers.shared_embedding_columns(
        [a1, a2], dimension=4, combiner="mean",
        shared_embedding_name="my_shared_embedding")
    # a3 is a completely different sparse column with a1 and a2, but since the
    # same shared_embedding_name is passed in, a3 will have the same embedding
    # as a1 and a2
    a3 = tf.contrib.layers.sparse_column_with_keys(
        "a3", ["cathy", "tom", "anderson"])
    e = tf.contrib.layers.shared_embedding_columns(
        [a3], dimension=4, combiner="mean",
        shared_embedding_name="my_shared_embedding")
    with tf.variable_scope("run_2"):
      d1 = d[0].to_dnn_input_layer(input_tensor_c1)
      e1 = e[0].to_dnn_input_layer(input_tensor_c1)
    with self.test_session() as sess:
      sess.run(tf.initialize_all_variables())
      d1_value = d1.eval()
      e1_value = e1.eval()
    for i in range(len(d1_value)):
      self.assertAllClose(d1_value[i], e1_value[i])

  def testOneHotColumn(self):
    a = tf.contrib.layers.sparse_column_with_keys("a", ["a", "b", "c", "d"])
    onehot_a = tf.contrib.layers.one_hot_column(a)
    self.assertEqual(onehot_a.sparse_id_column.name, "a")
    self.assertEqual(onehot_a.length, 4)

    b = tf.contrib.layers.sparse_column_with_hash_bucket(
        "b", hash_bucket_size=100, combiner="sum")
    onehot_b = tf.contrib.layers.one_hot_column(b)
    self.assertEqual(onehot_b.sparse_id_column.name, "b")
    self.assertEqual(onehot_b.length, 100)

  def testRealValuedColumn(self):
    a = tf.contrib.layers.real_valued_column("aaa")
    self.assertEqual(a.name, "aaa")
    self.assertEqual(a.dimension, 1)
    b = tf.contrib.layers.real_valued_column("bbb", 10)
    self.assertEqual(b.dimension, 10)
    self.assertTrue(b.default_value is None)

    with self.assertRaisesRegexp(TypeError, "dimension must be an integer"):
      tf.contrib.layers.real_valued_column("d3", dimension=1.0)

    with self.assertRaisesRegexp(ValueError,
                                 "dimension must be greater than 0"):
      tf.contrib.layers.real_valued_column("d3", dimension=0)

    with self.assertRaisesRegexp(ValueError,
                                 "dtype must be convertible to float"):
      tf.contrib.layers.real_valued_column("d3", dtype=tf.string)

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
    with self.assertRaisesRegexp(TypeError,
                                 "default_value must be compatible with dtype"):
      tf.contrib.layers.real_valued_column("d3",
                                           default_value=2.,
                                           dtype=tf.int32)

    # default_value is neither integer nor float.
    with self.assertRaisesRegexp(
        TypeError, "default_value must be compatible with dtype"):
      tf.contrib.layers.real_valued_column("e1", default_value="string")
    with self.assertRaisesRegexp(
        TypeError, "default_value must be compatible with dtype"):
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
    with self.assertRaisesRegexp(
        TypeError, "default_value must be compatible with dtype"):
      tf.contrib.layers.real_valued_column("g3",
                                           default_value=[2.],
                                           dtype=tf.int32)
    with self.assertRaisesRegexp(
        ValueError, "The length of default_value must be equal to dimension"):
      tf.contrib.layers.real_valued_column("g4",
                                           dimension=3,
                                           default_value=[2.])

    # Test that the normalizer_fn gets stored for a real_valued_column
    normalizer = lambda x: x - 1
    h1 = tf.contrib.layers.real_valued_column("h1", normalizer=normalizer)
    self.assertEqual(normalizer(10), h1.normalizer_fn(10))

  def testBucketizedColumnNameEndsWithUnderscoreBucketized(self):
    a = tf.contrib.layers.bucketized_column(
        tf.contrib.layers.real_valued_column("aaa"), [0, 4])
    self.assertEqual(a.name, "aaa_bucketized")

  def testBucketizedColumnRequiresRealValuedColumn(self):
    with self.assertRaisesRegexp(
        TypeError, "source_column must be an instance of _RealValuedColumn"):
      tf.contrib.layers.bucketized_column("bbb", [0])
    with self.assertRaisesRegexp(
        TypeError, "source_column must be an instance of _RealValuedColumn"):
      tf.contrib.layers.bucketized_column(
          tf.contrib.layers.sparse_column_with_integerized_feature(
              column_name="bbb", bucket_size=10),
          [0])

  def testBucketizedColumnRequiresSortedBuckets(self):
    with self.assertRaisesRegexp(
        ValueError, "boundaries must be a sorted list"):
      tf.contrib.layers.bucketized_column(
          tf.contrib.layers.real_valued_column("ccc"), [5, 0, 4])

  def testBucketizedColumnWithSameBucketBoundaries(self):
    a_bucketized = tf.contrib.layers.bucketized_column(
        tf.contrib.layers.real_valued_column("a"), [1., 2., 2., 3., 3.])
    self.assertEqual(a_bucketized.name, "a_bucketized")
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

    self.assertEqual("aaa_X_bbb_X_cost_bucketized", crossed.name,
                     "name should be generated by sorted column names")
    self.assertEqual("aaa", crossed.columns[0].name)
    self.assertEqual("bbb", crossed.columns[1].name)
    self.assertEqual("cost_bucketized", crossed.columns[2].name)

  def testCrossedColumnNotSupportRealValuedColumn(self):
    b = tf.contrib.layers.sparse_column_with_hash_bucket("bbb",
                                                         hash_bucket_size=100)
    with self.assertRaisesRegexp(
        TypeError,
        "columns must be a set of _SparseColumn, _CrossedColumn, "
        "or _BucketizedColumn instances"):
      tf.contrib.layers.crossed_column(
          set([b, tf.contrib.layers.real_valued_column("real")]),
          hash_bucket_size=10000)

  def testWeightedSparseColumnDtypes(self):
    ids = tf.contrib.layers.sparse_column_with_keys(
        "ids", ["marlo", "omar", "stringer"])
    weighted_ids = tf.contrib.layers.weighted_sparse_column(ids, "weights")
    self.assertDictEqual(
        {"ids": tf.VarLenFeature(tf.string),
         "weights": tf.VarLenFeature(tf.float32)},
        weighted_ids.config)

    weighted_ids = tf.contrib.layers.weighted_sparse_column(ids, "weights",
                                                            dtype=tf.int32)
    self.assertDictEqual(
        {"ids": tf.VarLenFeature(tf.string),
         "weights": tf.VarLenFeature(tf.int32)},
        weighted_ids.config)

    with self.assertRaisesRegexp(ValueError,
                                 "dtype is not convertible to float"):
      weighted_ids = tf.contrib.layers.weighted_sparse_column(ids, "weights",
                                                              dtype=tf.string)

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

    with self.assertRaisesRegexp(ValueError,
                                 "dtype must be convertible to float"):
      tf.contrib.layers.real_valued_column("rvc", dtype=tf.string)

  def testSparseColumnDtypes(self):
    sc = tf.contrib.layers.sparse_column_with_integerized_feature("sc", 10)
    self.assertDictEqual({"sc": tf.VarLenFeature(dtype=tf.int64)}, sc.config)

    sc = tf.contrib.layers.sparse_column_with_integerized_feature(
        "sc", 10, dtype=tf.int32)
    self.assertDictEqual({"sc": tf.VarLenFeature(dtype=tf.int32)}, sc.config)

    with self.assertRaisesRegexp(ValueError,
                                 "dtype must be an integer"):
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
    sparse_id_col = tf.contrib.layers.sparse_column_with_keys(
        "id_column", ["marlo", "omar", "stringer"])
    weighted_id_col = tf.contrib.layers.weighted_sparse_column(
        sparse_id_col, "id_weights_column")
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
    feature_columns = set([sparse_col, embedding_col, weighted_id_col,
                           real_valued_col1, real_valued_col2,
                           bucketized_col1, bucketized_col2,
                           cross_col])
    config = tf.contrib.layers.create_feature_spec_for_parsing(feature_columns)
    self.assertDictEqual({
        "sparse_column": tf.VarLenFeature(tf.string),
        "sparse_column_for_embedding": tf.VarLenFeature(tf.string),
        "id_column": tf.VarLenFeature(tf.string),
        "id_weights_column": tf.VarLenFeature(tf.float32),
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
    self.assertGreaterEqual(
        placeholder.name.find(u"Placeholder_real_valued_column"), 0)
    self.assertEqual(tf.float32, placeholder.dtype)
    self.assertEqual([None, 5], placeholder.get_shape().as_list())
    placeholder = placeholders["real_valued_column_for_bucketization"]
    self.assertGreaterEqual(
        placeholder.name.find(
            u"Placeholder_real_valued_column_for_bucketization"), 0)
    self.assertEqual(tf.float32, placeholder.dtype)
    self.assertEqual([None, 1], placeholder.get_shape().as_list())

  def testInitEmbeddingColumnWeightsFromCkpt(self):
    sparse_col = tf.contrib.layers.sparse_column_with_hash_bucket(
        column_name="object_in_image",
        hash_bucket_size=4)
    # Create _EmbeddingColumn which randomly initializes embedding of size
    # [4, 16].
    embedding_col = tf.contrib.layers.embedding_column(sparse_col, dimension=16)

    # Creating a SparseTensor which has all the ids possible for the given
    # vocab.
    input_tensor = tf.SparseTensor(indices=[[0, 0], [1, 1], [2, 2], [3, 3]],
                                   values=[0, 1, 2, 3],
                                   shape=[4, 4])

    # Invoking 'embedding_column.to_dnn_input_layer' will create the embedding
    # variable. Creating under scope 'run_1' so as to prevent name conflicts
    # when creating embedding variable for 'embedding_column_pretrained'.
    with tf.variable_scope("run_1"):
      with tf.variable_scope(embedding_col.name):
        # This will return a [4, 16] tensor which is same as embedding variable.
        embeddings = embedding_col.to_dnn_input_layer(input_tensor)

    save = tf.train.Saver()
    checkpoint_path = os.path.join(self.get_temp_dir(), "model.ckpt")

    with self.test_session() as sess:
      sess.run(tf.initialize_all_variables())
      saved_embedding = embeddings.eval()
      save.save(sess, checkpoint_path)

    embedding_col_initialized = tf.contrib.layers.embedding_column(
        sparse_id_column=sparse_col,
        dimension=16,
        ckpt_to_load_from=checkpoint_path,
        tensor_name_in_ckpt="run_1/object_in_image_embedding/weights")

    with tf.variable_scope("run_2"):
      # This will initialize the embedding from provided checkpoint and return a
      # [4, 16] tensor which is same as embedding variable. Since we didn't
      # modify embeddings, this should be same as 'saved_embedding'.
      pretrained_embeddings = embedding_col_initialized.to_dnn_input_layer(
          input_tensor)

    with self.test_session() as sess:
      sess.run(tf.initialize_all_variables())
      loaded_embedding = pretrained_embeddings.eval()

    self.assertAllClose(saved_embedding, loaded_embedding)

  def testInitCrossedColumnWeightsFromCkpt(self):
    sparse_col_1 = tf.contrib.layers.sparse_column_with_hash_bucket(
        column_name="col_1", hash_bucket_size=4)
    sparse_col_2 = tf.contrib.layers.sparse_column_with_hash_bucket(
        column_name="col_2", hash_bucket_size=4)

    crossed_col = tf.contrib.layers.crossed_column(
        columns=[sparse_col_1, sparse_col_2],
        hash_bucket_size=4)

    input_tensor = tf.SparseTensor(indices=[[0, 0], [1, 1], [2, 2], [3, 3]],
                                   values=[0, 1, 2, 3],
                                   shape=[4, 4])

    # Invoking 'weighted_sum_from_feature_columns' will create the crossed
    # column weights variable.
    with tf.variable_scope("run_1"):
      with tf.variable_scope(crossed_col.name):
        # Returns looked up column weights which is same as crossed column
        # weights as well as actual references to weights variables.
        _, col_weights, _ = (
            tf.contrib.layers.weighted_sum_from_feature_columns(
                {sparse_col_1.name: input_tensor,
                 sparse_col_2.name: input_tensor},
                [crossed_col],
                1))
        # Update the weights since default initializer initializes all weights
        # to 0.0.
        for weight in col_weights.values():
          assign_op = tf.assign(weight[0], weight[0] + 0.5)

    save = tf.train.Saver()
    checkpoint_path = os.path.join(self.get_temp_dir(), "model.ckpt")

    with self.test_session() as sess:
      sess.run(tf.initialize_all_variables())
      sess.run(assign_op)
      saved_col_weights = col_weights[crossed_col][0].eval()
      save.save(sess, checkpoint_path)

    crossed_col_initialized = tf.contrib.layers.crossed_column(
        columns=[sparse_col_1, sparse_col_2],
        hash_bucket_size=4,
        ckpt_to_load_from=checkpoint_path,
        tensor_name_in_ckpt=("run_1/col_1_X_col_2/"
                             "weighted_sum_from_feature_columns/"
                             "col_1_X_col_2/weights"))

    with tf.variable_scope("run_2"):
      # This will initialize the crossed column weights from provided checkpoint
      # and return a [4, 1] tensor which is same as weights variable. Since we
      # won't modify weights, this should be same as 'saved_col_weights'.
      _, col_weights, _ = (
          tf.contrib.layers.weighted_sum_from_feature_columns(
              {sparse_col_1.name: input_tensor,
               sparse_col_2.name: input_tensor},
              [crossed_col_initialized],
              1))
      col_weights_from_ckpt = col_weights[crossed_col_initialized][0]

    with self.test_session() as sess:
      sess.run(tf.initialize_all_variables())
      loaded_col_weights = col_weights_from_ckpt.eval()

    self.assertAllClose(saved_col_weights, loaded_col_weights)


if __name__ == "__main__":
  tf.test.main()

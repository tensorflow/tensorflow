# Copyright 2015 Google Inc. All Rights Reserved.
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

"""Functional tests for ops used with embeddings."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf


def _AsLong(array):
  """Casts arrays elements to long type. Used to convert from numpy tf."""
  return [int(x) for x in array]


class ScatterAddSubTest(tf.test.TestCase):

  def _TestCase(self, shape, indices, scatter_op=tf.scatter_add):
    """Run a random test case with the given shape and indices.

    Args:
      shape: Shape of the parameters array.
      indices: One-dimensional array of ints, the indices of the last dimension
               of the parameters to update.
      scatter_op: ScatterAdd or ScatterSub.
    """
    super(ScatterAddSubTest, self).setUp()
    with self.test_session(use_gpu=False):
      # Create a random parameter array of given shape
      p_init = np.random.rand(*shape).astype("f")
      # Create the shape of the update array. All dimensions except the last
      # match the parameter array, the last dimension equals the # of indices.
      vals_shape = [len(indices)] + shape[1:]
      vals_init = np.random.rand(*vals_shape).astype("f")
      v_i = [float(x) for x in vals_init.ravel()]
      p = tf.Variable(p_init)
      vals = tf.constant(v_i, shape=vals_shape, name="vals")
      ind = tf.constant(indices, dtype=tf.int32)
      p2 = scatter_op(p, ind, vals, name="updated_p")
      # p = init
      tf.initialize_all_variables().run()
      # p += vals
      result = p2.eval()
    # Compute the expected 'p' using numpy operations.
    for i, ind in enumerate(indices):
      if scatter_op == tf.scatter_add:
        p_init.reshape(shape[0], -1)[ind, :] += (
            vals_init.reshape(vals_shape[0], -1)[i, :])
      else:
        p_init.reshape(shape[0], -1)[ind, :] -= (
            vals_init.reshape(vals_shape[0], -1)[i, :])
    self.assertTrue(all((p_init == result).ravel()))

  def testNoRepetitions(self):
    self._TestCase([2, 2], [1])
    self._TestCase([4, 4, 4], [2, 0])
    self._TestCase([43, 20, 10, 10], [42, 5, 6, 1, 3, 5, 7, 9])

  def testWithRepetitions(self):
    self._TestCase([2, 2], [1, 1])
    self._TestCase([5, 3, 9, 5], [2, 0, 4, 1, 3, 1, 4, 0, 4, 3])
    self._TestCase([32, 4, 4], [31] * 8)

  def testRandom(self):
    # Random shapes of rank 4, random indices
    for _ in range(5):
      shape = np.random.randint(1, 20, size=4)
      indices = np.random.randint(shape[0], size=2 * shape[0])
      self._TestCase(_AsLong(list(shape)), list(indices))

  def testSubRandom(self):
    # Random shapes of rank 4, random indices
    for _ in range(5):
      shape = np.random.randint(1, 20, size=4)
      indices = np.random.randint(shape[0], size=2 * shape[0])
      self._TestCase(_AsLong(list(shape)), list(indices),
                     tf.scatter_sub)

  def testWrongShape(self):
    # Indices and values mismatch.
    var = tf.Variable(tf.zeros(shape=[1024, 64, 64], dtype=tf.float32))
    indices = tf.placeholder(tf.int32, shape=[32])
    values = tf.placeholder(tf.float32, shape=[33, 64, 64])
    with self.assertRaises(ValueError):
      tf.scatter_add(var, indices, values)

    # Var and values mismatch.
    values = tf.placeholder(tf.float32, shape=[32, 64, 63])
    with self.assertRaises(ValueError):
      tf.scatter_add(var, indices, values)


def _PName(param_id):
  return "p" + str(param_id)


def _EmbeddingParams(num_shards, vocab_size,
                     dtype=tf.float32,
                     shape=None,
                     use_shapeless_placeholder=False):
  p = []
  params = {}
  feed_dict = {}
  if not shape: shape = [10]
  for i in range(num_shards):
    shard_shape = [vocab_size // num_shards] + shape
    if i < vocab_size % num_shards:  # Excess goes evenly on the first shards
      shard_shape[0] += 1

    param_name = _PName(i)

    if use_shapeless_placeholder:
      param = tf.placeholder(dtype, shape=None, name=param_name)
    else:
      param = tf.constant(1.0, shape=shard_shape, dtype=dtype, name=param_name)
    p.append(param)
    np_type = "f" if dtype == tf.float32 else "d"
    val = (np.random.rand(*shard_shape).astype(np_type)) + 1
    params[param_name + ":0"] = val
    feed_dict[param.name] = val
  return p, params, feed_dict


def _EmbeddingResult(params, id_vals, num_shards, vocab_size,
                     partition_strategy="mod",
                     weight_vals=None):
  if weight_vals is None:
    weight_vals = np.copy(id_vals)
    weight_vals.fill(1)
  values = []
  weights = []
  weights_squared = []
  for ids, wts in zip(id_vals, weight_vals):
    value_aggregation = None
    weight_aggregation = None
    squared_weight_aggregation = None
    if isinstance(ids, tf.compat.integral_types):
      ids = [ids]
      wts = [wts]
    for i, weight_value in zip(ids, wts):
      if partition_strategy == "mod":
        val = np.copy(params[_PName(i % num_shards) + ":0"][
            i // num_shards, :]) * weight_value
      elif partition_strategy == "div":
        ids_per_partition, extras = divmod(vocab_size, num_shards)
        threshold = extras * (ids_per_partition + 1)
        if i < threshold:
          partition = i // (ids_per_partition + 1)
          offset = i % (ids_per_partition + 1)
        else:
          partition = extras + (i - threshold) // ids_per_partition
          offset = (i - threshold) % ids_per_partition
        val = np.copy(
            params[_PName(partition) + ":0"][offset, :]) * weight_value
      else:
        assert False
      if value_aggregation is None:
        assert weight_aggregation is None
        assert squared_weight_aggregation is None
        value_aggregation = val
        weight_aggregation = weight_value
        squared_weight_aggregation = weight_value * weight_value
      else:
        assert weight_aggregation is not None
        assert squared_weight_aggregation is not None
        value_aggregation += val
        weight_aggregation += weight_value
        squared_weight_aggregation += weight_value * weight_value
    values.append(value_aggregation)
    weights.append(weight_aggregation)
    weights_squared.append(squared_weight_aggregation)
  values = np.array(values).astype(np.float32)
  weights = np.array(weights).astype(np.float32)
  weights_squared = np.array(weights_squared).astype(np.float32)
  return values, weights, weights_squared


class EmbeddingLookupTest(tf.test.TestCase):

  # This test looks up [0, 0] in a parameter matrix sharded 2 ways. Since
  # both the ids are in the first shard, one of the resulting lookup
  # vector is going to be empty. The subsequent DivOp fails because of that.
  # TODO(keveman): Disabling the test until the underlying problem is fixed.
  def testSimpleSharded(self):
    with self.test_session():
      num_shards = 2
      vocab_size = 4
      p, params, feed_dict = _EmbeddingParams(num_shards, vocab_size)

      id_vals = np.array([0, 0])
      ids = tf.constant(list(id_vals), dtype=tf.int32)
      print("Construct ids", ids.get_shape())
      embedding = tf.nn.embedding_lookup(p, ids)

      tf_result = embedding.eval(feed_dict=feed_dict)
    np_result, _, _ = _EmbeddingResult(params, id_vals, num_shards, vocab_size)
    self.assertAllEqual(np_result, tf_result)
    self.assertShapeEqual(np_result, embedding)

  def testShardedModPartitioningInt32Ids(self):
    with self.test_session():
      num_shards = 5
      vocab_size = 13
      # Embedding dimensions is 10. The vocab_size x 10 embedding
      # parameters are spread in num_shards matrices, so the first
      # 3 shards are 3 x 10 and the last 2 shards are 2 x 10.
      p, params, feed_dict = _EmbeddingParams(num_shards, vocab_size)

      num_vals = 30
      # Fetch num_vals embeddings for random word ids. Since
      # num_vals > vocab_size, this ought to have repetitions, so
      # will test that aspect.
      id_vals = np.random.randint(vocab_size, size=num_vals)
      ids = tf.constant(list(id_vals), dtype=tf.int32)

      embedding = tf.nn.embedding_lookup(p, ids)
      tf_result = embedding.eval(feed_dict=feed_dict)
    np_result, _, _ = _EmbeddingResult(params, id_vals, num_shards, vocab_size)
    self.assertAllEqual(np_result, tf_result)
    self.assertShapeEqual(np_result, embedding)

  def testShardedModPartitioningInt64Ids(self):
    with self.test_session():
      num_shards = 5
      vocab_size = 13
      # Embedding dimensions is 10. The vocab_size x 10 embedding
      # parameters are spread in num_shards matrices, so the first
      # 3 shards are 3 x 10 and the last 2 shards are 2 x 10.
      p, params, feed_dict = _EmbeddingParams(num_shards, vocab_size)

      num_vals = 30
      # Fetch num_vals embeddings for random word ids. Since
      # num_vals > vocab_size, this ought to have repetitions, so
      # will test that aspect.
      id_vals = np.random.randint(vocab_size, size=num_vals)
      ids = tf.constant(list(id_vals), dtype=tf.int64)

      embedding = tf.nn.embedding_lookup(p, ids)
      tf_result = embedding.eval(feed_dict=feed_dict)
    np_result, _, _ = _EmbeddingResult(params, id_vals, num_shards, vocab_size)
    self.assertAllEqual(np_result, tf_result)
    self.assertShapeEqual(np_result, embedding)

  def testShardedDivPartitioningInt32Ids(self):
    with self.test_session():
      num_shards = 5
      vocab_size = 13
      # Embedding dimensions is 10. The vocab_size x 10 embedding
      # parameters are spread in num_shards matrices, so the first
      # 3 shards are 3 x 10 and the last 2 shards are 2 x 10.
      p, params, feed_dict = _EmbeddingParams(num_shards, vocab_size)

      num_vals = 30
      # Fetch num_vals embeddings for random word ids. Since
      # num_vals > vocab_size, this ought to have repetitions, so
      # will test that aspect.
      id_vals = np.random.randint(vocab_size, size=num_vals)
      ids = tf.constant(list(id_vals), dtype=tf.int32)

      embedding = tf.nn.embedding_lookup(p, ids, partition_strategy="div")
      tf_result = embedding.eval(feed_dict=feed_dict)
    np_result, _, _ = _EmbeddingResult(
        params, id_vals, num_shards, vocab_size, partition_strategy="div")
    self.assertAllEqual(np_result, tf_result)
    self.assertShapeEqual(np_result, embedding)

  def testShardedDivPartitioningInt64Ids(self):
    with self.test_session():
      num_shards = 5
      vocab_size = 13
      # Embedding dimensions is 10. The vocab_size x 10 embedding
      # parameters are spread in num_shards matrices, so the first
      # 3 shards are 3 x 10 and the last 2 shards are 2 x 10.
      p, params, feed_dict = _EmbeddingParams(num_shards, vocab_size)

      num_vals = 30
      # Fetch num_vals embeddings for random word ids. Since
      # num_vals > vocab_size, this ought to have repetitions, so
      # will test that aspect.
      id_vals = np.random.randint(vocab_size, size=num_vals)
      ids = tf.constant(list(id_vals), dtype=tf.int64)

      embedding = tf.nn.embedding_lookup(p, ids, partition_strategy="div")
      tf_result = embedding.eval(feed_dict=feed_dict)
    np_result, _, _ = _EmbeddingResult(
        params, id_vals, num_shards, vocab_size, partition_strategy="div")
    self.assertAllEqual(np_result, tf_result)
    self.assertShapeEqual(np_result, embedding)

  def testShardedDivPartitioningUnknownParamShape(self):
    with self.test_session():
      num_shards = 5
      vocab_size = 13
      # Embedding dimensions is 10. The vocab_size x 10 embedding
      # parameters are spread in num_shards matrices, so the first
      # 3 shards are 3 x 10 and the last 2 shards are 2 x 10.

      # We clear parameter shapes, to test when shape is not statically known.
      p, params, feed_dict = _EmbeddingParams(
          num_shards, vocab_size, use_shapeless_placeholder=True)

      num_vals = 30
      # Fetch num_vals embeddings for random word ids. Since
      # num_vals > vocab_size, this ought to have repetitions, so
      # will test that aspect.
      id_vals = np.random.randint(vocab_size, size=num_vals)
      ids = tf.constant(list(id_vals), dtype=tf.int64)

      embedding = tf.nn.embedding_lookup(p, ids, partition_strategy="div")
      tf_result = embedding.eval(feed_dict=feed_dict)
    np_result, _, _ = _EmbeddingResult(
        params, id_vals, num_shards, vocab_size, partition_strategy="div")
    self.assertAllEqual(np_result, tf_result)

  def testGradientsEmbeddingLookup(self):
    vocab_size = 9
    num_ids = 10
    id_vals = list(np.random.randint(vocab_size, size=num_ids))
    tf.logging.vlog(1, id_vals)
    for ids_shape in [(10,), (2, 5)]:
      for num_shards in [1, 3]:
        with self.test_session():
          ids = tf.constant(id_vals, shape=ids_shape, dtype=tf.int32)
          x, params, _ = _EmbeddingParams(
              num_shards, vocab_size, shape=[2])
          y = tf.nn.embedding_lookup(x, ids)
          y_shape = [num_ids] + list(params[_PName(0) + ":0"].shape[1:])
          x_name = [_PName(i) for i in range(num_shards)]
          x_init_value = [params[x_n + ":0"] for x_n in x_name]
          x_shape = [i.shape for i in x_init_value]
          err = tf.test.compute_gradient_error(x,
                                               x_shape,
                                               y,
                                               y_shape,
                                               x_init_value=x_init_value)
        self.assertLess(err, 1e-4)

  def testGradientsEmbeddingLookupWithComputedParams(self):
    vocab_size = 9
    num_ids = 5
    id_vals = list(np.random.randint(vocab_size, size=num_ids))
    tf.logging.vlog(1, id_vals)
    for num_shards in [1, 3]:
      with self.test_session():
        ids = tf.constant(id_vals, dtype=tf.int32)
        x, params, _ = _EmbeddingParams(
            num_shards, vocab_size, shape=[2])
        # This will force a conversion from IndexedSlices to Tensor.
        x_squared = [tf.square(elem) for elem in x]
        y = tf.nn.embedding_lookup(x_squared, ids)
        y_shape = [num_ids] + list(params[_PName(0) + ":0"].shape[1:])
        x_name = [_PName(i) for i in range(num_shards)]
        x_init_value = [params[x_n + ":0"] for x_n in x_name]
        x_shape = [i.shape for i in x_init_value]
        err = tf.test.compute_gradient_error(x,
                                             x_shape,
                                             y,
                                             y_shape,
                                             x_init_value=x_init_value)
      self.assertLess(err, 1e-3)

  def testConstructionNonSharded(self):
    with tf.Graph().as_default():
      p = tf.Variable(tf.zeros(shape=[100, 100], dtype=tf.float32))
      ids = tf.constant([0, 1, 1, 7], dtype=tf.int32)
      tf.nn.embedding_lookup([p], ids)

  def testConstructionSharded(self):
    with tf.Graph().as_default():
      p = []
      for _ in range(2):
        p += [tf.Variable(tf.zeros(shape=[100, 100], dtype=tf.float32))]
        ids = tf.constant([0, 1, 1, 17], dtype=tf.int32)
      tf.nn.embedding_lookup(p, ids)

  def testHigherRank(self):
    np.random.seed(8)
    with self.test_session():
      for params_shape in (12,), (6, 3):
        params = np.random.randn(*params_shape)
        for ids_shape in (3, 2), (4, 3):
          ids = np.random.randint(params.shape[0],
                                  size=np.prod(ids_shape)).reshape(ids_shape)
          # Compare nonsharded to gather
          simple = tf.nn.embedding_lookup(params, ids).eval()
          self.assertAllEqual(simple, tf.gather(params, ids).eval())
          # Run a few random sharded versions
          for procs in 1, 2, 3:
            stride = procs * tf.range(params.shape[0] // procs)
            split_params = [tf.gather(params, stride + p)
                            for p in xrange(procs)]
            sharded = tf.nn.embedding_lookup(split_params, ids).eval()
            self.assertAllEqual(simple, sharded)


class EmbeddingLookupSparseTest(tf.test.TestCase):

  def _RandomIdsAndWeights(self, batch_size, vocab_size):
    max_val_per_entry = 6
    vals_per_batch_entry = np.random.randint(
        1, max_val_per_entry, size=batch_size)
    num_vals = np.sum(vals_per_batch_entry)

    ids = np.random.randint(vocab_size, size=num_vals)
    weights = 1 + np.random.rand(num_vals)

    indices = []
    for batch_entry, num_val in enumerate(vals_per_batch_entry):
      for val_index in range(num_val):
        indices.append([batch_entry, val_index])

    shape = [batch_size, max_val_per_entry]

    sp_ids = tf.SparseTensor(
        tf.constant(indices, tf.int64),
        tf.constant(ids, tf.int32),
        tf.constant(shape, tf.int64))
    sp_weights = tf.SparseTensor(
        tf.constant(indices, tf.int64),
        tf.constant(weights, tf.float32),
        tf.constant(shape, tf.int64))

    return sp_ids, sp_weights, ids, weights, vals_per_batch_entry

  def _GroupByBatchEntry(self, vals, vals_per_batch_entry):
    grouped_vals = []
    index = 0
    for num_val in vals_per_batch_entry:
      grouped_vals.append(list(vals[index: (index + num_val)]))
      index += num_val
    return grouped_vals

  def testEmbeddingLookupSparse(self):
    vocab_size = 13
    batch_size = 10
    param_shape = [2, 5]
    expected_lookup_result_shape = [None] + param_shape

    sp_ids, sp_weights, ids, weights, vals_per_batch_entry = (
        self._RandomIdsAndWeights(batch_size, vocab_size))

    grouped_ids = self._GroupByBatchEntry(ids, vals_per_batch_entry)
    grouped_weights = self._GroupByBatchEntry(weights, vals_per_batch_entry)
    grouped_ignored_weights = self._GroupByBatchEntry(
        np.ones(np.sum(vals_per_batch_entry)), vals_per_batch_entry)

    for num_shards, combiner, dtype, ignore_weights in itertools.product(
        [1, 5],
        ["sum", "mean", "sqrtn"],
        [tf.float32, tf.float64],
        [True, False]):

      with self.test_session():
        p, params, feed_dict = _EmbeddingParams(num_shards, vocab_size,
                                                shape=param_shape,
                                                dtype=dtype)
        embedding_sum = tf.nn.embedding_lookup_sparse(
            p, sp_ids, None if ignore_weights else sp_weights,
            combiner=combiner)

        self.assertEqual(embedding_sum.get_shape().as_list(),
                         expected_lookup_result_shape)

        tf_embedding_sum = embedding_sum.eval(feed_dict=feed_dict)

        np_embedding_sum, np_weight_sum, np_weight_sq_sum = _EmbeddingResult(
            params, grouped_ids, num_shards, vocab_size,
            weight_vals=grouped_ignored_weights
            if ignore_weights else grouped_weights)
        if combiner == "mean":
          np_embedding_sum /= np.reshape(np_weight_sum, (batch_size, 1, 1))
        if combiner == "sqrtn":
          np_embedding_sum /= np.reshape(
              np.sqrt(np_weight_sq_sum), (batch_size, 1, 1))
        self.assertAllClose(np_embedding_sum, tf_embedding_sum)

  def testGradientsEmbeddingLookupSparse(self):
    vocab_size = 12
    batch_size = 4
    param_shape = [2, 3]
    sp_ids, sp_weights, _, _, _ = (
        self._RandomIdsAndWeights(batch_size, vocab_size))

    for num_shards, combiner, dtype, ignore_weights in itertools.product(
        [1, 3],
        ["sum", "mean", "sqrtn"],
        [tf.float32, tf.float64],
        [True, False]):
      with self.test_session():
        x, params, _ = _EmbeddingParams(num_shards, vocab_size,
                                        shape=param_shape,
                                        dtype=dtype)

        y = tf.nn.embedding_lookup_sparse(
            x, sp_ids, None if ignore_weights else sp_weights,
            combiner=combiner)
        x_name = [_PName(i) for i in range(num_shards)]
        x_init_value = [params[x_n + ":0"] for x_n in x_name]
        x_shape = [i.shape for i in x_init_value]
        y_shape = [batch_size] + list(params[_PName(0) + ":0"].shape[1:])
        err = tf.test.compute_gradient_error(x,
                                             x_shape,
                                             y,
                                             y_shape,
                                             x_init_value=x_init_value)
      self.assertLess(err, 1e-5 if dtype == tf.float64 else 2e-3)

  def testIncompatibleShapes(self):
    with self.test_session():
      x, _, _ = _EmbeddingParams(1, 10, dtype=tf.float32)
      sp_ids = tf.SparseTensor(
          tf.constant([[0, 0], [0, 1], [1, 0]], tf.int64),
          tf.constant([0, 1, 2], tf.int32),
          tf.constant([2, 2], tf.int64))
      sp_weights = tf.SparseTensor(
          tf.constant([[0, 0], [0, 1]], tf.int64),
          tf.constant([12.0, 5.0], tf.float32),
          tf.constant([1, 2], tf.int64))

      with self.assertRaises(ValueError):
        tf.nn.embedding_lookup_sparse(x, sp_ids, sp_weights, combiner="mean")


class DynamicStitchOpTest(tf.test.TestCase):

  def testCint32Cpu(self):
    with self.test_session(use_gpu=False):
      indices = [tf.convert_to_tensor([0, 1, 2]), tf.convert_to_tensor([2, 3])]
      values = [tf.convert_to_tensor([12, 23, 34]), tf.convert_to_tensor([1, 2])]
      self.assertAllEqual(
          tf.dynamic_stitch(indices, values).eval(), [12, 23, 1, 2])

  def testCint32Gpu(self):
    with self.test_session(use_gpu=True):
      indices = [tf.convert_to_tensor([0, 1, 2]), tf.convert_to_tensor([2, 3])]
      values = [tf.convert_to_tensor([12, 23, 34]), tf.convert_to_tensor([1, 2])]
      self.assertAllEqual(
          tf.dynamic_stitch(indices, values).eval(), [12, 23, 1, 2])

  def testInt32Cpu(self):
    with self.test_session(use_gpu=False):
      indices = [tf.convert_to_tensor([0, 1, 2]), tf.convert_to_tensor([2, 3])]
      values = [tf.convert_to_tensor([12, 23, 34]), tf.convert_to_tensor([1, 2])]
      self.assertAllEqual(
          tf.dynamic_stitch(indices, values).eval(), [12, 23, 1, 2])

  def testInt32Gpu(self):
    with self.test_session(use_gpu=True):
      indices = [tf.convert_to_tensor([0, 1, 2]), tf.convert_to_tensor([2, 3])]
      values = [tf.convert_to_tensor([12, 23, 34]), tf.convert_to_tensor([1, 2])]
      self.assertAllEqual(
          tf.dynamic_stitch(indices, values).eval(), [12, 23, 1, 2])

  def testSumGradArgs(self):
    with self.test_session(use_gpu=False):
      indices = [tf.convert_to_tensor([0, 1, 2, 3]),
                 tf.convert_to_tensor([2, 3])]
      values = [tf.convert_to_tensor([2, 3, 5, 7]), tf.convert_to_tensor([1, 1])]
      self.assertAllEqual(
          tf.dynamic_stitch(indices, values).eval(), [2, 3, 1, 1])

  # We expect that the values are merged in order.
  def testStitchOrder(self):
    with self.test_session():
      indices = []
      np_values = []
      values = []
      for _ in range(10):
        indices.extend([tf.convert_to_tensor(np.arange(100).astype(np.int32))])
        np_values.extend([np.random.uniform(size=100)])
        values.extend([tf.convert_to_tensor(np_values[-1])])
      stitched = tf.dynamic_stitch(indices, values).eval()
    self.assertAllEqual(np_values[-1], stitched)


if __name__ == "__main__":
  tf.test.main()

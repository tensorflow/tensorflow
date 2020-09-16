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
"""
Dynamic Embedding is designed for Large-scale Sparse Weights Training.
See [Sparse Domain Isolation](https://github.com/tensorflow/community/pull/237)
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import init_ops_v2
from tensorflow.python.ops import lookup_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.util.tf_export import tf_export
from tensorflow.python.training.tracking import tracking as trackable


def _partition(data, partition_index, shard_num):
  """
  Shard keys to shard_num partitions

  Args:
    data: keys or values, usually the IDs of dynamic features.
    partition_index: partitions index.
    shard_num: partition number
  Returns:
    a pair of tensor: (partition result, partition indices)
  """
  if shard_num <= 1:
    return [
        data,
    ], None
  with ops.colocate_with(data, ignore_existing=True):
    partitions = data_flow_ops.dynamic_partition(data, partition_index,
                                                 shard_num)
    indices = data_flow_ops.dynamic_partition(
        math_ops.range(array_ops.shape(data)[0]),
        math_ops.cast(partition_index, dtypes.int32), shard_num)
  return partitions, indices


def _stitch(values, indices):
  if len(values) == 1:
    return values[0]
  with ops.colocate_with(indices[0]):
    all_values = data_flow_ops.dynamic_stitch(indices, values)
  return all_values


def default_partition_fn(keys, shard_num):
  """The default partition function.
    partition keys by "mod" strategy.

    keys: a tensor presents the keys to be partitioned.
    shard_num: the num of partitions
  Returns:
    a tensor with same shape as keys with type of `tf.int32`,
      represents the corresponding partition-ids of keys.
  """
  with ops.colocate_with(keys):
    ids = math_ops.cast(math_ops.mod(keys, shard_num), dtype=dtypes.int32)
  return ids


@tf_export("dynamic_embedding.Variable")
class Variable(trackable.TrackableResource):
  """
  A Distributed version of HashTable(reference from lookup_ops.MutableHashTable)
  It is designed to dynamically store the Sparse Weights(Parameters) of DLRMs.
  """

  def __init__(self,
               key_dtype=dtypes.int64,
               value_dtype=dtypes.float32,
               dim=1,
               devices=None,
               partitioner=default_partition_fn,
               shared_name=None,
               name="DynamicEmbedding_Variable",
               initializer=None,
               trainable=True,
               checkpoint=True):
    """Creates an empty `Variable` object.

    Creates a group of tables placed on devices,
    the type of its keys and values are specified by key_dtype
    and value_dtype, respectively.

    Args:
      key_dtype: the type of the key tensors.
      value_dtype: the type of the value tensors.
      dim: the length of the value array for each key.
      devices: the list of devices holding the tables.
        One table will be created on each device.
      partitioner: partition function of keys,
        return the partition index for each key.

      Example partition func:
      ```python
      def default_partition_fn(keys, shard_num):
        return tf.cast(keys % shard_num, dtype=tf.int32)
      ```
      shared_name: No used.
      name: A name for the operation (optional).
      initializer: The value to use if a key is missing in the hash table.
        which can be a python number, numpy array or `tf.initializer` instances.
        If initializer is `None` (the default), `0` will be taken.
      trainable: True, will be treated as a trainable Variable, and add to
        to the list of variables collected in the graph under the key
        `GraphKeys.TRAINABLE_VARIABLES`.
      checkpoint: if True, the contents of the SparseVariable are
        saved to and restored from checkpoints.
        If `shared_name` is empty for a checkpointed table,
        it is shared using the table node name.

    Returns:
      A `Variable` object.
    """
    self.key_dtype = key_dtype
    self.value_dtype = value_dtype
    self.dim = dim
    devices_ = devices or [
        "/CPU:0",
    ]
    self.devices = devices_ if isinstance(devices_, list) else [
        devices,
    ]
    self.partition_fn = partitioner
    self.name = name
    self.shared_name = shared_name or "shared_name.{}".format(name)

    self.initializer = None

    self.trainable = trainable
    self.checkpoint = checkpoint

    self._tables = []
    self.size_ops = []
    self.shard_num = len(self.devices)

    if key_dtype not in [dtypes.int32, dtypes.int64]:
      raise TypeError("key_dtype should be int32, int64.")
    _initializer = initializer
    if _initializer is None:
      _initializer = init_ops.zeros_initializer(dtype=self.value_dtype)
    static_default_value = self._convert_anything_to_init(_initializer, dim)

    value_dtype_list = [
        dtypes.int32, dtypes.int64, dtypes.bool, dtypes.float32, dtypes.float64
    ]
    if value_dtype not in value_dtype_list:
      raise TypeError("value_dtype should be ", value_dtype_list)
    scope_name = self.name.split("/")[-1]
    with ops.name_scope(scope_name, "DynamicEmbedding_Variable"):
      with ops.colocate_with(None, ignore_existing=True):
        for idx in range(len(self.devices)):
          with ops.device(self.devices[idx]):
            mht = None
            mht = lookup_ops.MutableHashTable(
                key_dtype=self.key_dtype,
                value_dtype=self.value_dtype,
                default_value=static_default_value,
                name=self._make_name(idx),
                checkpoint=self.checkpoint)

            self._tables.append(mht)
    super(Variable, self).__init__()

  @property
  def tables(self):
    return self._tables

  def _convert_anything_to_init(self, raw_init, dim):
    init = raw_init
    while callable(init):
      if isinstance(init, (init_ops.Initializer, init_ops_v2.Initializer)):
        self.initializer = init
        init = init(shape=[1])
      else:
        init = init()
    init = math_ops.cast(array_ops.fill([dim],
                                        array_ops.reshape(init, [-1])[0]),
                         dtype=self.value_dtype)
    return init

  def _create_resource(self):
    raise NotImplementedError

  def _make_name(self, table_idx):
    return "{}_mht_{}of{}".format(self.name.replace("/", "_"), table_idx + 1,
                                  self.shard_num)

  def upsert(self, keys, values, name=None):
    """Insert or Update `keys` with `values`.

    If key exists already, value will be updated.

    Args:
      keys: Keys to insert. Can be a tensor of any shape. Must match the table's
        key type.
      values: Values to be associated with keys. Must be a tensor of the same
        shape as `keys` and match the table's value type.
      name: A name for the operation (optional).

    Returns:
      The created Operation.

    Raises:
      TypeError: when `keys` or `values` doesn't match the table data
        types.
    """

    partition_index = self.partition_fn(keys, self.shard_num)
    keys_partitions, _ = _partition(keys, partition_index, self.shard_num)
    values_partitions, _ = _partition(values, partition_index, self.shard_num)

    ops_ = []
    for idx in range(len(self.devices)):
      with ops.device(self.devices[idx]):
        ops_.append(self._tables[idx].insert(keys_partitions[idx],
                                             values_partitions[idx],
                                             name=name))

    return control_flow_ops.group(ops_)

  def remove(self, keys, name=None):
    """Removes `keys` and its associated values from the variable.

    If a key is not present in the table, it is silently ignored.

    Args:
      keys: Keys to remove. Can be a tensor of any shape. Must match the table's
        key type.
      name: A name for the operation (optional).

    Returns:
      The created Operation.

    Raises:
      TypeError: when `keys` do not match the table data types.
    """
    partition_index = self.partition_fn(keys, self.shard_num)
    keys_partitions, _ = _partition(keys, partition_index, self.shard_num)

    ops_ = []
    for idx in range(len(self.devices)):
      with ops.device(self.devices[idx]):
        ops_.append(self._tables[idx].remove(keys_partitions[idx], name=name))

    return control_flow_ops.group(ops_)

  def _create_default_values_by_initializer(self, keys):
    if self.initializer is None:
      return None
    try:
      vals_shape = keys.get_shape().concatenate(self.dim)
      init_op = self.initializer(vals_shape)
    except Exception:  # constant.initializer
      init_op = self.initializer([self.dim])
    return init_op

  def lookup(self, keys, name=None):
    """Looks up `keys` in a Variable, outputs the corresponding values.

    The `default_value` is used for keys not present in the table.

    Args:
      keys: Keys to look up. Can be a tensor of any shape. Must match the
        table's key_dtype.
      name: A name for the operation (optional).

    Returns:
      A tensor containing the values in the same shape as `keys` using the
        table's value type.
    """
    partition_index = self.partition_fn(keys, self.shard_num)
    keys_partitions, keys_indices = _partition(keys, partition_index,
                                               self.shard_num)

    ops_ = []
    for idx in range(len(self.devices)):
      with ops.device(self.devices[idx]):
        dynamic_default_values = self._create_default_values_by_initializer(
            keys_partitions[idx])
        ops_.append(self._tables[idx].lookup(
            keys_partitions[idx],
            dynamic_default_values=dynamic_default_values,
            name=name))
    result = _stitch(ops_, keys_indices)

    return result

  def export(self, name=None):
    """Returns tensors of all keys and values in the table.

    Args:
      name: A name for the operation (optional).

    Returns:
      A pair of tensors with the first tensor containing all keys and the
        second tensors containing all values in the table.
    """
    full_keys = []
    full_values = []
    for idx in range(len(self.devices)):
      keys_ = None
      vals_ = None
      with ops.device(self.devices[idx]):
        keys_, vals_ = self._tables[idx].export(name=name)
        full_keys.append(keys_)
        full_values.append(vals_)
    return array_ops.concat(full_keys, 0), array_ops.concat(full_values, 0)

  def size(self, index=None, name=None):
    """Compute the number of elements in the index-th table of this Variable.

    If index is none, the total size of the Variable wil be return.

    Args:
      index: The index of table (optional)
      name: A name for the operation (optional).

    Returns:
      A scalar tensor containing the number of elements in this Variable.
    """
    if context.executing_eagerly():
      self.size_ops = []
    if not self.size_ops:
      for idx in range(len(self.devices)):
        with ops.device(self.devices[idx]):
          self.size_ops.append(self._tables[idx].size(name=name))

    return self.size_ops[index] if index is not None else math_ops.add_n(
        self.size_ops)

  def _gather_saveables_for_checkpoint(self):
    """For object-based checkpointing."""
    saveables = dict()
    for table in self._tables:
      # pylint: disable=protected-access
      saveable_dict = table._gather_saveables_for_checkpoint()
      for (_, saveable) in saveable_dict.items():
        # merge all tables saveable to one dict with their own name.
        saveables[saveable.keywords["name"]] = saveable
    return saveables


@tf_export("dynamic_embedding.get_variable")
def get_variable(
    name,  # unique
    key_dtype=dtypes.int64,
    value_dtype=dtypes.float32,
    dim=1,
    devices=None,
    partitioner=default_partition_fn,
    shared_name="get_variable",
    initializer=None,
    trainable=True,
    checkpoint=True):
  """Gets an `Variable` object with this name if it exists,
       or create a new one.

  Args:
    name: A unique name for the `Variable`.
    key_dtype: the type of the key tensors.
    value_dtype: the type of the value tensors.
    dim: the length of the value array for each key.
    devices: the list of devices holding the tables.
      One table will be created on each device.
    partitioner: partition function of keys,
      return the partition index for each key.

    Example partition func:
    ```python
    def default_partition_fn(keys, shard_num):
      return tf.cast(keys % shard_num, dtype=tf.int32)
    ```
    shared_name: No used.
    initializer: The value to use if a key is missing in the hash table.
      which can a python number, numpy array or `tf.initializer` instances.
      If initializer is `None` (the default), `0` will be used.
    trainable: True, will be treated as a trainable Variable, and add to
      to the list of variables collected in the graph under the key
      `GraphKeys.TRAINABLE_VARIABLES`.
    checkpoint: if True, the contents of the SparseVariable are
      saved to and restored from checkpoints.
      If `shared_name` is empty for a checkpointed table,
      it is shared using the table node name.

  Returns:
    A `Variable` object.
  """
  var_ = None
  scope = variable_scope.get_variable_scope()
  scope_store = variable_scope._get_default_variable_store()
  full_name = scope.name + "/" + name if scope.name else name
  if full_name in scope_store._vars:
    if scope.reuse is False:
      err_msg = ("Variable %s already exists, disallowed."
                 " Did you mean to set reuse=True or "
                 "reuse=tf.AUTO_REUSE in VarScope?" % full_name)

      raise ValueError(err_msg)
  else:
    var_ = Variable(key_dtype=key_dtype,
                    value_dtype=value_dtype,
                    dim=dim,
                    devices=devices,
                    partitioner=partitioner,
                    shared_name=shared_name,
                    name=full_name,
                    initializer=initializer,
                    trainable=trainable,
                    checkpoint=checkpoint)
    scope_store._vars[full_name] = var_
  return scope_store._vars[full_name]


@tf_export("dynamic_embedding.embedding_lookup")
def embedding_lookup(
    params,
    ids,
    partition_strategy=None,  # pylint: disable=unused-argument
    name=None,
    validate_indices=None,  # pylint: disable=unused-argument
    max_norm=None,
    return_trainable=False):
  """Provides a dynamic version of embedding_lookup
    similar with tf.nn.embedding_lookup.

  Ids are flattened to a 1d tensor before being passed to embedding_lookup
  then, they are unflattend to match the original ids shape plus an extra
  leading dimension of the size of the embeddings.

  Args:
    params: A dynamic_embedding.Variable instance.
    ids: a tensor with any shape as same dtype of params.key_dtype.
    partition_strategy: No used, for API compatiblity with `nn.emedding_lookup`.
    name: A name for the operation (optional).
    validate_indices: No used, just for compatible with nn.embedding_lookup .
    max_norm: If not `None`, each embedding is clipped if its l2-norm is larger
      than this value.
    return_trainable: optional, If True, also return TrainableWrapper
  Returns:
    A tensor with shape [shape of ids] + [dim],
      dim is equal to the value dim of params.
      containing the values from the params tensor(s) for keys in ids.
    trainable_wrap:
      A TrainableWrapper object used to fill the Optimizers `var_list`
        Only provided if `return_trainable` is True.
  """
  if isinstance(params, (list, tuple)) and len(params) > 1:
    raise ValueError("Only one params is allowed.")
  if isinstance(params, (list, tuple)):
    params = params[0]
  if not isinstance(params, Variable):
    raise TypeError("params should be a Variable instance.")
  if params.key_dtype != ids.dtype:
    raise TypeError(
        "params.key_dtype should be same with ids.dtype: {} vs. {}".format(
            params.key_dtype, ids.dtype))

  scope = variable_scope.get_variable_scope()
  full_name = scope.name + "/" if scope.name else ""
  full_name += (name + "/") if name else "embedding_lookup/"
  with ops.name_scope(full_name):
    initial_value = None
    trainable_wrap = None
    ids = ops.convert_to_tensor(ids, name="ids")
    if ids.get_shape() == tensor_shape.unknown_shape():
      ids = array_ops.reshape(ids, shape=[-1])
      initial_shape = (1, params.dim)
      trainable_shape = tensor_shape.unknown_shape()
    else:
      initial_shape = [ d if d else 1 for d in ids.get_shape().as_list()] \
                      + [params.dim]
      trainable_shape = ids.get_shape().concatenate([params.dim])
    initial_value = constant_op.constant(0.0,
                                         shape=initial_shape,
                                         dtype=params.value_dtype)

    with ops.colocate_with(None, ignore_existing=True):
      collections = [ops.GraphKeys.LOCAL_VARIABLES]
      if params.trainable:
        collections += [ops.GraphKeys.TRAINABLE_VARIABLES]
      trainable_ = resource_variable_ops.TrainableWrapper(
          params,
          ids,
          max_norm=max_norm,
          initial_value=initial_value,
          dtype=params.value_dtype,
          trainable=params.trainable,
          collections=collections)
      embeddings = array_ops.identity(trainable_)
      embeddings.set_shape(trainable_shape)

  return (embeddings, trainable_) if return_trainable else embeddings


@tf_export("dynamic_embedding.embedding_lookup_sparse")
def embedding_lookup_sparse(
    params,
    sp_ids,
    sp_weights,
    partition_strategy=None,  # no used
    name="embedding_lookup_sparse",
    combiner="mean",
    max_norm=None,
    return_trainable=False):
  """Provides a dynamic version of embedding_lookup_sparse
    similar with tf.nn.embedding_lookup_sparse.

  This op assumes that there is at least one id for each row in the dense tensor
  represented by sp_ids (i.e. there are no rows with empty features), and that
  all the indices of sp_ids are in canonical row-major order.

  It also assumes that all id values lie in the range [0, p0), where p0
  is the sum of the size of params along dimension 0.

  Args:
    params: A single `dynamic_embedding.Variable` instance representing
      the complete embedding tensor.
    sp_ids: N x M `SparseTensor` of int64 ids where N is typically batch size
      and M is arbitrary.
    sp_weights: either a `SparseTensor` of float / double weights, or `None` to
      indicate all weights should be taken to be 1. If specified, `sp_weights`
      must have exactly the same shape and indices as `sp_ids`.
    partition_strategy: No used.
    name: Optional name for the op.
    combiner: A string specifying the reduction op. Currently "mean", "sqrtn"
      and "sum" are supported. "sum" computes the weighted sum of the embedding
      results for each row. "mean" is the weighted sum divided by the total
      weight. "sqrtn" is the weighted sum divided by the square root of the sum
      of the squares of the weights.
    max_norm: If not `None`, each embedding is clipped if its l2-norm is larger
      than this value, before combining.
    return_trainable: optional, If True, also return TrainableWrapper create by
      `dynamic_embedding.embedding_lookup`

  Returns:
    combined_embeddings: A dense tensor representing the combined embeddings
      for the sparse ids. For each row in the dense tensor represented by
      `sp_ids`, the op looks up the embeddings for all ids in that row,
      multiplies them by the corresponding weight, and combines these embeddings
      as specified.

      In other words, if

        `shape(combined params) = [+infinity, dim]`

      and

        `shape(sp_ids) = shape(sp_weights) = [d0, d1, ..., dn]`

      then

        `shape(output) = [d0, dim]`.

      For instance, if params dim=20, and sp_ids / sp_weights are

        ```python
        [0, 0]: id 1, weight 2.0
        [0, 1]: id 3, weight 0.5
        [1, 0]: id 0, weight 1.0
        [2, 3]: id 1, weight 3.0
        ```

      with `combiner`="mean", then the output will be a 3x20 matrix where

        ```python
        output[0, :] = (params[1, :] * 2.0 + params[3, :] * 0.5) / (2.0 + 0.5)
        output[1, :] = (params[0, :] * 1.0) / 1.0
        output[2, :] = (params[1, :] * 3.0) / 3.0
        ```
    trainable_wrap:
      A TrainableWrapper object used to fill the Optimizers `var_list`
        Only provided if `return_trainable` is True.
  Raises:
    TypeError: If `sp_ids` is not a `SparseTensor`, or if `sp_weights` is
      neither `None` nor `SparseTensor`.
    ValueError: If `combiner` is not one of {"mean", "sqrtn", "sum"}.
  """
  if combiner not in ("mean", "sqrtn", "sum"):
    raise ValueError("combiner must be one of 'mean', 'sqrtn' or 'sum'")

  if not isinstance(sp_ids, sparse_tensor.SparseTensor):
    raise TypeError("sp_ids must be SparseTensor")

  ignore_weights = sp_weights is None
  if not ignore_weights:
    if not isinstance(sp_weights, sparse_tensor.SparseTensor):
      raise TypeError("sp_weights must be either None or SparseTensor")

  scope = variable_scope.get_variable_scope()
  full_name = scope.name + "/" + name if scope.name else name
  with ops.name_scope(full_name + "/"):
    segment_ids = sp_ids.indices[:, 0]
    if segment_ids.dtype != dtypes.int32:
      segment_ids = math_ops.cast(segment_ids, dtypes.int32)

    ids = sp_ids.values
    ids, idx = array_ops.unique(ids)

    embeddings, trainable_ = embedding_lookup(
        params,
        ids,
        name=name + '/embedding_lookup',
        partition_strategy=partition_strategy,
        max_norm=max_norm,
        return_trainable=True)
    if embeddings.dtype in (dtypes.float16, dtypes.bfloat16):
      embeddings = math_ops.cast(embeddings, dtypes.float32)
    if not ignore_weights:
      weights = sp_weights.values
      if weights.dtype != embeddings.dtype:
        weights = math_ops.cast(weights, embeddings.dtype)

      embeddings = array_ops.gather(embeddings, idx)

      # Reshape weights to allow broadcast
      ones = array_ops.fill(
          array_ops.expand_dims(array_ops.rank(embeddings) - 1, 0), 1)
      bcast_weights_shape = array_ops.concat([array_ops.shape(weights), ones],
                                             0)

      orig_weights_shape = weights.get_shape()
      weights = array_ops.reshape(weights, bcast_weights_shape)

      # Set the weight shape, since after reshaping to bcast_weights_shape,
      # the shape becomes None.
      if embeddings.get_shape().ndims is not None:
        weights.set_shape(
            orig_weights_shape.concatenate(
                [1 for _ in range(embeddings.get_shape().ndims - 1)]))

      embeddings *= weights

      if combiner == "sum":
        embeddings = math_ops.segment_sum(embeddings, segment_ids, name=name)
      elif combiner == "mean":
        embeddings = math_ops.segment_sum(embeddings, segment_ids)
        weight_sum = math_ops.segment_sum(weights, segment_ids)
        embeddings = math_ops.div(embeddings, weight_sum, name=name)
      elif combiner == "sqrtn":
        embeddings = math_ops.segment_sum(embeddings, segment_ids)
        weights_squared = math_ops.pow(weights, 2)
        weight_sum = math_ops.segment_sum(weights_squared, segment_ids)
        weight_sum_sqrt = math_ops.sqrt(weight_sum)
        embeddings = math_ops.div(embeddings, weight_sum_sqrt, name=name)
      else:
        assert False, "Unrecognized combiner"
    else:
      assert idx is not None
      if combiner == "sum":
        embeddings = math_ops.sparse_segment_sum(embeddings,
                                                 idx,
                                                 segment_ids,
                                                 name=name)
      elif combiner == "mean":
        embeddings = math_ops.sparse_segment_mean(embeddings,
                                                  idx,
                                                  segment_ids,
                                                  name=name)
      elif combiner == "sqrtn":
        embeddings = math_ops.sparse_segment_sqrt_n(embeddings,
                                                    idx,
                                                    segment_ids,
                                                    name=name)
      else:
        assert False, "Unrecognized combiner"

    return (embeddings, trainable_) if return_trainable else embeddings


@tf_export("dynamic_embedding.safe_embedding_lookup_sparse")
def safe_embedding_lookup_sparse(
    embedding_weights,
    sparse_ids,
    sparse_weights=None,
    combiner="mean",
    default_id=None,
    name="safe_embedding_lookup_sparse",
    partition_strategy=None,  # no used
    max_norm=None,
    return_trainable=False):
  """ Provides a dynamic version of `tf.nn.safe_embedding_lookup_sparse`.

  Lookup embedding results, accounting for empty features and invalid weights.

  Any IDs will be treated as valid include non-positive IDs.
  Invalid weights (<= 0) are pruned from input weights, as well as any IDs
  with non-positive weight. For an entry with no features, the embedding vector
  for `default_id` is returned, or the 0-vector if `default_id` is not supplied.

  The ids and weights may be multi-dimensional. Embeddings are always aggregated
  along the last dimension.

  Args:
    embedding_weights: A single `dynamic_embedding.Variable` instance
      representing the complete embedding tensor.
    sparse_ids: `SparseTensor` of shape `[d_0, d_1, ..., d_n]` containing the
      ids. `d_0` is typically batch size.
    sparse_weights: `SparseTensor` of same shape as `sparse_ids`, containing
      float weights corresponding to `sparse_ids`, or `None` if all weights are
      be assumed to be 1.0.
    combiner: A string specifying how to combine embedding results for each
      entry. Currently "mean", "sqrtn" and "sum" are supported, with "mean" the
      default.
    default_id: The id to use for an entry with no features.
    name: A name for this operation (optional).
    partition_strategy: A string specifying the partitioning strategy. Currently
      `"div"` and `"mod"` are supported. Default is `"div"`.
    max_norm: If not `None`, all embeddings are l2-normalized to max_norm before
      combining.

  Returns:
    combined_embeddings:
      A dense `Tensor` of shape `[d_0, d_1, ..., d_{n-1}, e_1, ..., e_m]`.
    trainable_wrap:
      A TrainableWrapper object used to fill the Optimizers `var_list`
        Only provided if `return_trainable` is True.

  Raises:
    ValueError: if `embedding_weights` is empty.
  """
  if embedding_weights is None:
    raise ValueError("Missing embedding_weights %s." % embedding_weights)

  if embedding_weights.key_dtype != sparse_ids.dtype:
    raise TypeError(
        "embedding_weights.key_dtype should be same with sparse_ids.dtype: "
        "{} vs. {}".format(embedding_weights.value_dtype, sparse_ids.dtype))

  weights_dtype = sparse_weights.dtype if sparse_weights is not None else None
  if weights_dtype and embedding_weights.value_dtype != weights_dtype:
    raise TypeError(
        "embedding_weights.value_dtype should be same with sparse_weights.dtype"
        ": {} vs. {}".format(embedding_weights.value_dtype, weights_dtype))

  scope = variable_scope.get_variable_scope()
  full_name = scope.name + "/" + name if scope.name else name
  with ops.name_scope(full_name + "/"):
    # Reshape higher-rank sparse ids and weights to linear segment ids.
    original_shape = sparse_ids.dense_shape
    original_rank_dim = tensor_shape.dimension_value(
        sparse_ids.dense_shape.get_shape()[0])
    original_rank = (array_ops.size(original_shape)
                     if original_rank_dim is None else original_rank_dim)
    sparse_ids = sparse_ops.sparse_reshape(sparse_ids, [
        math_ops.reduce_prod(
            array_ops.slice(original_shape, [0], [original_rank - 1])),
        array_ops.gather(original_shape, original_rank - 1)
    ])
    if sparse_weights is not None:
      sparse_weights = sparse_tensor.SparseTensor(sparse_ids.indices,
                                                  sparse_weights.values,
                                                  sparse_ids.dense_shape)

    # Prune invalid weights.
    if combiner != "sum":
      sparse_ids, sparse_weights = _prune_invalid_weights(
          sparse_ids, sparse_weights)

    # Fill in dummy values for empty features, if necessary.
    sparse_ids, is_row_empty = sparse_ops.sparse_fill_empty_rows(
        sparse_ids, default_id or 0)
    if sparse_weights is not None:
      sparse_weights, _ = sparse_ops.sparse_fill_empty_rows(sparse_weights, 1.0)

    result, trainable_ = embedding_lookup_sparse(
        embedding_weights,
        sparse_ids,
        sparse_weights,
        combiner=combiner,
        partition_strategy=partition_strategy,
        name=name + "/embedding_lookup_sparse",
        max_norm=max_norm,
        return_trainable=True)

    if default_id is None:
      # Broadcast is_row_empty to the same shape as embedding_lookup_result,
      # for use in Select.
      is_row_empty = array_ops.tile(
          array_ops.reshape(is_row_empty, [-1, 1]),
          array_ops.stack([1, array_ops.shape(result)[1]]))

      result = array_ops.where(is_row_empty,
                               array_ops.zeros_like(result),
                               result,
                               name="where")

    # Reshape back from linear ids back into higher-dimensional dense result.
    final_result = array_ops.reshape(
        result,
        array_ops.concat([
            array_ops.slice(math_ops.cast(original_shape, dtypes.int32), [0],
                            [original_rank - 1]),
            array_ops.slice(array_ops.shape(result), [1], [-1])
        ], 0))
    final_result.set_shape(
        tensor_shape.unknown_shape(
            (tensor_shape.Dimension(original_rank_dim) - 1).value).concatenate(
                result.get_shape()[1:]))
    return (final_result, trainable_) if return_trainable else final_result


def _prune_invalid_weights(sparse_ids, sparse_weights):
  """Prune invalid weights (< 0) from the input ids and weights."""
  if sparse_weights is not None:
    is_weights_valid = math_ops.greater(sparse_weights.values, 0)
    sparse_ids = sparse_ops.sparse_retain(sparse_ids, is_weights_valid)
    sparse_weights = sparse_ops.sparse_retain(sparse_weights, is_weights_valid)
  return sparse_ids, sparse_weights


def create_slots(primary, init, slot_name, op_name):
  """Helper function for creating a slot variable for statefull optimizers."""
  params_var_, params_ids_ = primary.params, primary.ids

  scope = variable_scope.get_variable_scope()
  scope_store = variable_scope._get_default_variable_store()
  full_name = params_var_.name + "/" + op_name + "/" + slot_name
  if full_name not in scope_store._vars:
    slot_variable_ = Variable(name=full_name,
                              key_dtype=params_var_.key_dtype,
                              value_dtype=params_var_.value_dtype,
                              dim=params_var_.dim,
                              devices=params_var_.devices,
                              partitioner=params_var_.partition_fn,
                              initializer=init,
                              trainable=False,
                              checkpoint=params_var_.checkpoint)

    scope_store._vars[full_name] = slot_variable_

  slot_trainable = None
  _, slot_trainable = embedding_lookup(params=scope_store._vars[full_name],
                                       ids=params_ids_,
                                       name=slot_name,
                                       return_trainable=True)

  return slot_trainable

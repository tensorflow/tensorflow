# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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
"""Mid level API for Serving TPU Embeddings."""

from typing import Any, Iterable, Optional, Text, Union, Dict
from absl import logging

from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import tpu_strategy
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import variables as tf_variables
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.tpu import tpu_embedding_base
from tensorflow.python.tpu import tpu_embedding_v2_utils
from tensorflow.python.types import core
from tensorflow.python.util import nest
from tensorflow.python.util.tf_export import tf_export


@tf_export("tpu.experimental.embedding.TPUEmbeddingForServing")
class TPUEmbeddingForServing(tpu_embedding_base.TPUEmbeddingBase):
  """The TPUEmbedding mid level API running on CPU for serving.

  Note: This class is intended to be used for embedding tables that are trained
  on TPU and to be served on CPU. Therefore the class should be only initialized
  under non-TPU strategy. Otherwise an error will be raised.

  You can first train your model using the TPUEmbedding class and save the
  checkpoint. Then use this class to restore the checkpoint to do serving.

  First train a model and save the checkpoint.
  ```python
  model = model_fn(...)
  strategy = tf.distribute.TPUStrategy(...)
  with strategy.scope():
    embedding = tf.tpu.experimental.embedding.TPUEmbedding(
        feature_config=feature_config,
        optimizer=tf.tpu.experimental.embedding.SGD(0.1))

  # Your custom training code.

  checkpoint = tf.train.Checkpoint(model=model, embedding=embedding)
  checkpoint.save(...)

  ```

  Then restore the checkpoint and do serving.
  ```python

  # Restore the model on CPU.
  model = model_fn(...)
  embedding = tf.tpu.experimental.embedding.TPUEmbeddingForServing(
        feature_config=feature_config,
        optimizer=tf.tpu.experimental.embedding.SGD(0.1))

  checkpoint = tf.train.Checkpoint(model=model, embedding=embedding)
  checkpoint.restore(...)

  result = embedding(...)
  table = embedding.embedding_table
  ```

  NOTE: This class can also be used to do embedding training on CPU. But it
  requires the conversion between keras optimizer and embedding optimizers so
  that the slot variables can stay consistent between them.
  """

  def __init__(
      self,
      feature_config: Union[tpu_embedding_v2_utils.FeatureConfig, Iterable],  # pylint:disable=g-bare-generic
      optimizer: Optional[tpu_embedding_v2_utils._Optimizer]):  # pylint:disable=protected-access
    """Creates the TPUEmbeddingForServing mid level API object.

    ```python
    embedding = tf.tpu.experimental.embedding.TPUEmbeddingForServing(
        feature_config=tf.tpu.experimental.embedding.FeatureConfig(
            table=tf.tpu.experimental.embedding.TableConfig(
                dim=...,
                vocabulary_size=...)))
    ```

    Args:
      feature_config: A nested structure of
        `tf.tpu.experimental.embedding.FeatureConfig` configs.
      optimizer: An instance of one of `tf.tpu.experimental.embedding.SGD`,
        `tf.tpu.experimental.embedding.Adagrad` or
        `tf.tpu.experimental.embedding.Adam`. When not created under TPUStrategy
        may be set to None to avoid the creation of the optimizer slot
        variables, useful for optimizing memory consumption when exporting the
        model for serving where slot variables aren't needed.

    Raises:
      RuntimeError: If created under TPUStrategy.
    """
    super(TPUEmbeddingForServing, self).__init__(feature_config, optimizer)
    self._strategy = distribute_lib.get_strategy()
    if isinstance(self._strategy,
                  (tpu_strategy.TPUStrategy, tpu_strategy.TPUStrategyV2)):
      raise RuntimeError("Serving on TPU is not yet supported.")

  @property
  def embedding_tables(
      self) -> Dict[tpu_embedding_v2_utils.TableConfig, tf_variables.Variable]:
    """Returns a dict of embedding tables, keyed by `TableConfig`."""
    self._maybe_build()
    # Only return the tables and not the slot variables.
    return {
        table: self._variables[table.name]["parameters"]
        for table in self._table_config
    }

  def _maybe_build(self):
    if not self._built:
      # This can be called while tracing a function, so we wrap the
      # initialization code with init_scope so it runs eagerly, this means that
      # it will not be included the function graph generated by tracing so that
      # we can be sure that we only initialize the TPU for embeddings exactly
      # once.
      with ops.init_scope():
        self.build()

  def _create_variables_and_slots(
      self) -> Dict[Text, Dict[Text, tf_variables.Variable]]:
    """Create variables for TPU embeddings.

    Returns:
      A dict of dicts. The outer dict is keyed by the table names and the inner
      dicts are keyed by 'parameters' and the slot variable names.
    """
    variables = {}
    for table in self._table_config:
      variables[table.name] = self._create_variables(table, trainable=True)
    return variables

  def embedding_lookup(self,
                       features: Any,
                       weights: Optional[Any] = None) -> Any:
    """Apply standard lookup ops on CPU.

    Args:
      features: A nested structure of `tf.Tensor`s, `tf.SparseTensor`s or
        `tf.RaggedTensor`s, with the same structure as `feature_config`. Inputs
        will be downcast to `tf.int32`. Only one type out of `tf.SparseTensor`
        or `tf.RaggedTensor` is supported per call.
      weights: If not `None`, a nested structure of `tf.Tensor`s,
        `tf.SparseTensor`s or `tf.RaggedTensor`s, matching the above, except
        that the tensors should be of float type (and they will be downcast to
        `tf.float32`). For `tf.SparseTensor`s we assume the `indices` are the
        same for the parallel entries from `features` and similarly for
        `tf.RaggedTensor`s we assume the row_splits are the same.

    Returns:
      A nested structure of Tensors with the same structure as input features.
    """
    return cpu_embedding_lookup(features, weights, self.embedding_tables,
                                self._feature_config)


def _ragged_embedding_lookup_with_reduce(table: tf_variables.Variable,
                                         ragged: ragged_tensor.RaggedTensor,
                                         weights: ragged_tensor.RaggedTensor,
                                         combiner: Text) -> core.Tensor:
  """Compute a ragged lookup followed by a reduce on axis 1.

  Args:
    table: The embedding table.
    ragged: A RaggedTensor of ids to look up.
    weights: A RaggedTensor of weights (or None).
    combiner: One of "mean", "sum", "sqrtn".

  Returns:
    A Tensor.
  """
  if weights is None:
    weights = array_ops.ones_like(ragged, dtype=table.dtype)
  weights = array_ops.expand_dims(weights, axis=2)
  ragged_result = embedding_ops.embedding_lookup(table, ragged)
  ragged_result = math_ops.reduce_sum(ragged_result * weights, axis=1)
  if combiner == "mean":
    ragged_result = math_ops.div_no_nan(ragged_result,
                                        math_ops.reduce_sum(weights, axis=1))
  elif combiner == "sqrtn":
    ragged_result = math_ops.div_no_nan(
        ragged_result,
        math_ops.sqrt(math_ops.reduce_sum(weights * weights, axis=1)))
  return ragged_result


@tf_export("tpu.experimental.embedding.serving_embedding_lookup")
def cpu_embedding_lookup(
    inputs: Any,
    weights: Optional[Any],
    tables: Dict[tpu_embedding_v2_utils.TableConfig, tf_variables.Variable],
    feature_config: Union[tpu_embedding_v2_utils.FeatureConfig, Iterable]  # pylint:disable=g-bare-generic
) -> Any:
  """Apply standard lookup ops with `tf.tpu.experimental.embedding` configs.

  This function is a utility which allows using the
  `tf.tpu.experimental.embedding` config objects with standard lookup functions.
  This can be used when exporting a model which uses
  `tf.tpu.experimental.embedding.TPUEmbedding` for serving on CPU. In particular
  `tf.tpu.experimental.embedding.TPUEmbedding` only supports lookups on TPUs and
  should not be part of your serving graph.

  Note that TPU specific options (such as `max_sequence_length`) in the
  configuration objects will be ignored.

  In the following example we take a trained model (see the documentation for
  `tf.tpu.experimental.embedding.TPUEmbedding` for the context) and create a
  saved model with a serving function that will perform the embedding lookup and
  pass the results to your model:

  ```python
  model = model_fn(...)
  embedding = tf.tpu.experimental.embedding.TPUEmbedding(
      feature_config=feature_config,
      batch_size=1024,
      optimizer=tf.tpu.experimental.embedding.SGD(0.1))
  checkpoint = tf.train.Checkpoint(model=model, embedding=embedding)
  checkpoint.restore(...)

  @tf.function(input_signature=[{'feature_one': tf.TensorSpec(...),
                                 'feature_two': tf.TensorSpec(...),
                                 'feature_three': tf.TensorSpec(...)}])
  def serve_tensors(embedding_features):
    embedded_features = tf.tpu.experimental.embedding.serving_embedding_lookup(
        embedding_features, None, embedding.embedding_tables,
        feature_config)
    return model(embedded_features)

  model.embedding_api = embedding
  tf.saved_model.save(model,
                      export_dir=...,
                      signatures={'serving_default': serve_tensors})

  ```

  NOTE: It's important to assign the embedding API object to a member of your
  model as `tf.saved_model.save` only supports saving variables as one
  `Trackable` object. Since the model's weights are in `model` and the
  embedding table are managed by `embedding`, we assign `embedding` to an
  attribute of `model` so that tf.saved_model.save can find the embedding
  variables.

  NOTE: The same `serve_tensors` function and `tf.saved_model.save` call will
  work directly from training.

  Args:
    inputs: a nested structure of Tensors, SparseTensors or RaggedTensors.
    weights: a nested structure of Tensors, SparseTensors or RaggedTensors or
      None for no weights. If not None, structure must match that of inputs, but
      entries are allowed to be None.
    tables: a dict of mapping TableConfig objects to Variables.
    feature_config: a nested structure of FeatureConfig objects with the same
      structure as inputs.

  Returns:
    A nested structure of Tensors with the same structure as inputs.
  """

  nest.assert_same_structure(inputs, feature_config)

  flat_inputs = nest.flatten(inputs)
  flat_weights = [None] * len(flat_inputs)
  if weights is not None:
    nest.assert_same_structure(inputs, weights)
    flat_weights = nest.flatten(weights)
  flat_features = nest.flatten_with_joined_string_paths(feature_config)

  outputs = []
  for inp, weight, (path, feature) in zip(flat_inputs, flat_weights,
                                          flat_features):
    table = tables[feature.table]

    if weight is not None:
      if isinstance(inp, ops.Tensor):
        raise ValueError(
            "Weight specified for {}, but input is dense.".format(path))
      elif type(weight) is not type(inp):
        raise ValueError(
            "Weight for {} is of type {} but it does not match type of the "
            "input which is {}.".format(path, type(weight), type(inp)))
      elif feature.max_sequence_length > 0:
        raise ValueError("Weight specified for {}, but this is a sequence "
                         "feature.".format(path))

    if isinstance(inp, ops.Tensor):
      if feature.max_sequence_length > 0:
        raise ValueError("Feature {} is a sequence feature but a dense tensor "
                         "was passed.".format(path))
      outputs.append(embedding_ops.embedding_lookup_v2(table, inp))

    elif isinstance(inp, sparse_tensor.SparseTensor):
      outputs.append(
          _embedding_lookup_for_sparse_tensor(inp, weight, table, feature))
    elif isinstance(inp, ragged_tensor.RaggedTensor):
      outputs.append(
          _embedding_lookup_for_ragged_tensor(inp, weight, table, feature))
    else:
      raise ValueError("Input {} is type {}. Tensor, SparseTensor or "
                       "RaggedTensor expected.".format(path, type(inp)))
  return nest.pack_sequence_as(feature_config, outputs)


def _embedding_lookup_for_sparse_tensor(
    inp: sparse_tensor.SparseTensor,
    weight: Optional[sparse_tensor.SparseTensor], table: tf_variables.Variable,
    feature: tpu_embedding_v2_utils.FeatureConfig) -> ops.Tensor:
  """Embedding lookup for sparse tensor based on its feature config.

  Args:
    inp: a single SparseTensor input.
    weight: None or SparseTensor which has the same shape of the input.
    table: a table variable.
    feature: a feature config.

  Returns:
    Embedding lookup result.
  """
  inp_rank = inp.shape.rank
  # The input rank can be None for sequence input tensor.
  if (
      not feature.output_shape
      and feature.max_sequence_length > 0
      and (inp_rank is None or inp_rank == 2)
  ):
    batch_size = math_ops.cast(array_ops.shape(inp)[0], dtype=dtypes.int64)
    sparse_shape = array_ops_stack.stack(
        [batch_size, feature.max_sequence_length], axis=0
    )
    # TPU Embedding truncates sequences to max_sequence_length, and if we
    # don't truncate, scatter_nd will error out if the index was out of
    # bounds.
    truncated_inp = sparse_ops.sparse_slice(
        inp, start=[0, 0], size=sparse_shape)

    dense_output_shape = array_ops_stack.stack(
        [batch_size, feature.max_sequence_length, feature.table.dim], axis=0)
    return array_ops.scatter_nd(
        truncated_inp.indices,
        array_ops.gather(table.read_value(), truncated_inp.values),
        dense_output_shape)
  else:
    if feature.max_sequence_length > 0:
      logging.warning(
          (
              "max_sequence_length setting will be ignored because the rank of"
              " the input tensor is %d which is not 2."
          ),
          inp_rank,
      )
    if (not feature.validate_weights_and_indices and inp_rank is not None and
        inp_rank <= 2):
      return embedding_ops.embedding_lookup_sparse_v2(
          table, inp, sp_weights=weight, combiner=feature.table.combiner)
    else:
      return embedding_ops.safe_embedding_lookup_sparse_v2(
          table, inp, sparse_weights=weight, combiner=feature.table.combiner)


def _embedding_lookup_for_ragged_tensor(
    inp: ragged_tensor.RaggedTensor,
    weight: Optional[ragged_tensor.RaggedTensor], table: tf_variables.Variable,
    feature: tpu_embedding_v2_utils.FeatureConfig) -> ops.Tensor:
  """Embedding lookup for ragged tensor based on its feature config.

  Args:
    inp: a single rank 2 RaggedTensor input.
    weight: None or RaggedTensor which has the same shape of the input.
    table: a table variable.
    feature: a feature config.

  Returns:
    Embedding lookup result.

  Raises:
    ValueError: if input ragged tensor is not rank 2 or output shape set in the
      feature config doesn't match with the first dim size of the input.
  """
  if inp.shape.rank != 2:
    raise ValueError(
        "Only rank 2 ragged tensor is supported, but got rank {}".format(
            inp.shape.rank))
  batch_size = inp.shape[0]
  if feature.output_shape:
    output_batch_size = math_ops.reduce_prod(feature.output_shape)
    # If the output batch size matches the data batch size, treat it as
    # normal ragged input.
    if output_batch_size == batch_size:
      ragged_output = _ragged_embedding_lookup_with_reduce(
          table, inp, weight, feature.table.combiner)
      ragged_output = array_ops.reshape(
          ragged_output, shape=feature.output_shape + [feature.table.dim])
    # If the data batch size is a factor of the output batch size, the
    # divide result will be the sequence length. Ignore the weights and
    # combiner.
    elif output_batch_size > batch_size and output_batch_size % batch_size == 0:
      ragged_output = embedding_ops.embedding_lookup_v2(table, inp)
      # Pad or truncate in the sequence dimension
      ragged_output = ragged_output.to_tensor(shape=[
          batch_size, output_batch_size // batch_size, feature.table.dim
      ])
      # Reshape to desire output shape.
      ragged_output = array_ops.reshape(
          ragged_output, feature.output_shape + [feature.table.dim])
    else:
      raise ValueError(
          "Output shape set in the FeatureConfig should be the factor of "
          "the input data batch size. But instead got output shape {}, "
          "input data batch size {}".format(feature.output_shape, batch_size))
  else:
    if feature.max_sequence_length > 0:
      output_shape = [
          batch_size, feature.max_sequence_length, feature.table.dim
      ]
      ragged_lookup = embedding_ops.embedding_lookup_v2(table, inp)
      # Unlike scatter_nd, RaggedTensor.to_tensor truncates to the given
      # shape.
      ragged_output = ragged_lookup.to_tensor(shape=output_shape)
    else:
      ragged_output = _ragged_embedding_lookup_with_reduce(
          table, inp, weight, feature.table.combiner)
  return ragged_output

# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Operations for generating and loading vocab remappings."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_checkpoint_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops

ops.NotDifferentiable("GenerateVocabRemapping")
ops.NotDifferentiable("LoadAndRemapMatrix")


def _load_and_remap_matrix(ckpt_path,
                           old_tensor_name,
                           new_row_vocab_offset,
                           num_rows_to_load,
                           new_col_vocab_size,
                           initializer,
                           old_row_vocab_file=None,
                           new_row_vocab_file=None,
                           old_col_vocab_file=None,
                           new_col_vocab_file=None,
                           num_row_oov_buckets=0,
                           num_col_oov_buckets=0,
                           max_rows_in_memory=-1):
  """Loads a 2-D (matrix) `Tensor` from checkpoint.

  Generates 1D-remappings for rows and columns using the
  `GenerateVocabRemapping` op, and initializes any anticipated values with the
  provided initializer. Then, uses the `LoadAndRemapMatrix` op to create a
  matrix that loads existing values from the checkpoint, while filling out
  "missing" values with the newly initialized values. See
  contrib/framework/ops/checkpoint_ops.cc for more information on the wrapped
  functionality (LoadAndRemapMatrix). This wrapper can be used to perform only
  row remapping or only col remapping. If only row remapping is desired,
  {new,old}_col_vocab_file should be `None`, and vice versa for column
  remapping.

  NOTE: This only supports div-partitioning the vocabulary on the 1st dimension
  (row axis) via `new_row_vocab_offset`.

  Args:
    ckpt_path: Path to the TensorFlow checkpoint (version 2, `TensorBundle`)
      from which the old matrix `Tensor` will be loaded.
    old_tensor_name: Name of the 2-D `Tensor` to load from checkpoint.
    new_row_vocab_offset: A 0-indexed integer representing what line to
      start reading at in the new row vocabulary. Used for partitioned
      variables.
    num_rows_to_load: Number of rows to load for the new vocabulary (note: to
      support variable partitioning and partial loading, this does not need to
      be the same as the number of entries in `new_row_vocab_file`).
    new_col_vocab_size: Number of columns to load - should be the same as the
      number of entries in `new_col_vocab_file`, since we don't support
      partitioning along the column axis.
    initializer: Callable initializer function that accepts a 1-D tensor as the
      arg to specify the shape of the returned tensor. Used to initialize
      missing values.
    old_row_vocab_file: A scalar `Tensor` of type `string` containing the
      path to the old row vocabulary file. Can be None, which represents no
      remapping on the row axis.
    new_row_vocab_file: A scalar `Tensor` of type `string` containing the path
      to the new row vocabulary file. Can be None, which represents no remapping
      on the row axis - in which case, `new_row_vocab_offset` and
      `num_rows_to_load` work under the assumption that the new row vocab is the
      same as the old row vocab.
    old_col_vocab_file: A scalar `Tensor` of type `string` containing the
      path to the old column vocabulary file. Can be None, which represents no
      remapping on the column axis.
    new_col_vocab_file: A scalar `Tensor` of type `string` containing the path
      to the new column vocabulary file. Can be None, which represents no
      remapping on the column axis - in which case, `new_col_vocab_size` works
      under the assumption that the new col vocab is the same as the old col
      vocab.
    num_row_oov_buckets: `int` specifying the number of out-of-vocabulary rows
      to append. Must be >= 0.
    num_col_oov_buckets: `int` specifying the number of out-of-vocabulary
      columns to append. Must be >= 0.
    max_rows_in_memory: `int` specifying the maximum number of rows to load from
      the checkpoint at once. If less than or equal to 0, the entire matrix will
      be loaded into memory. Setting this arg trades increased disk reads for
      lower memory usage.

  Returns:
    A Tensor of shape `[num_rows_to_load + num_row_oov_buckets,
    new_col_vocab_size + num_col_oov_buckets]`, with values loaded from the
    specified tensor in the checkpoint, and any missing or OOV values
    initialized with the given `initializer`.

  Raises:
    ValueError: If `num_row_oov_buckets` or `num_col_oov_buckets` < 0.
    ValueError: If either `old_row_vocab_file` or `new_row_vocab_file` is
      provided, while the other is not. Same for `old_col_vocab_file` and
      `new_col_vocab_file`.
    ValueError: If neither row vocabs or col vocabs are provided.
  """
  if num_row_oov_buckets < 0:
    raise ValueError("num_row_oov_buckets must be >= 0, but received %d" %
                     num_row_oov_buckets)
  if num_col_oov_buckets < 0:
    raise ValueError("num_col_oov_buckets must be >= 0, but received %d" %
                     num_col_oov_buckets)

  if bool(old_row_vocab_file) != bool(new_row_vocab_file):
    raise ValueError(
        "old_row_vocab_file and new_row_vocab_file must both be specified or "
        "left unspecified. old_row_vocab_file='{}', new_row_vocab_file='{}'".
        format(old_row_vocab_file, new_row_vocab_file))
  if bool(old_col_vocab_file) != bool(new_col_vocab_file):
    raise ValueError(
        "old_col_vocab_file and new_col_vocab_file must both be specified or "
        "left unspecified. old_col_vocab_file='{}', new_col_vocab_file='{}'".
        format(old_col_vocab_file, new_col_vocab_file))

  remap_rows = new_row_vocab_file and old_row_vocab_file
  remap_cols = new_col_vocab_file and old_col_vocab_file
  if not (remap_rows or remap_cols):
    raise ValueError(
        "Must provide either row or column vocab files. If no remapping is "
        "necessary, consider using `tf.contrib.framework.init_from_checkpoint` "
        "instead.")

  num_rows_present = num_rows_to_load
  if remap_rows:
    row_remapping, num_rows_present = (
        gen_checkpoint_ops._generate_vocab_remapping(  # pylint: disable=protected-access
            new_vocab_file=new_row_vocab_file,
            old_vocab_file=old_row_vocab_file,
            new_vocab_offset=new_row_vocab_offset,
            num_new_vocab=num_rows_to_load))
  else:
    # Even when the rows are not being reordered, we still need to generate a
    # remapping to account for initializing partitioned Variables (when
    # new_row_vocab_offset is non-zero).
    row_remapping = math_ops.range(
        new_row_vocab_offset,
        new_row_vocab_offset + num_rows_to_load,
        dtype=dtypes.int64)

  col_remapping = []
  num_cols_present = new_col_vocab_size
  if remap_cols:
    col_remapping, num_cols_present = (
        gen_checkpoint_ops._generate_vocab_remapping(  # pylint: disable=protected-access
            new_vocab_file=new_col_vocab_file,
            old_vocab_file=old_col_vocab_file,
            new_vocab_offset=0,  # Offset is unused for cols (no partitioning).
            num_new_vocab=new_col_vocab_size))

  init_vals = initializer([
      num_rows_to_load * new_col_vocab_size -
      num_rows_present * num_cols_present, 1
  ])
  return_tensor = gen_checkpoint_ops._load_and_remap_matrix(  # pylint: disable=protected-access
      ckpt_path=ckpt_path,
      old_tensor_name=old_tensor_name,
      row_remapping=row_remapping,
      col_remapping=col_remapping,
      initializing_values=init_vals,
      num_rows=num_rows_to_load,
      num_cols=new_col_vocab_size,
      max_rows_in_memory=max_rows_in_memory)

  # Add OOV row(s) and column(s).
  if num_row_oov_buckets > 0:
    init_row_oov_val = initializer([num_row_oov_buckets, new_col_vocab_size])
    init_row_oov_val = ops.convert_to_tensor(init_row_oov_val)
    return_tensor = array_ops.concat([return_tensor, init_row_oov_val], 0)
  if num_col_oov_buckets > 0:
    # We need to add any row OOV to the new column shape.
    init_col_oov_val = initializer(
        [num_rows_to_load + num_row_oov_buckets, num_col_oov_buckets])
    init_col_oov_val = ops.convert_to_tensor(init_col_oov_val)
    return_tensor = array_ops.concat([return_tensor, init_col_oov_val], 1)

  return return_tensor


def load_and_remap_matrix_initializer(ckpt_path,
                                      old_tensor_name,
                                      new_row_vocab_size,
                                      new_col_vocab_size,
                                      old_row_vocab_file=None,
                                      new_row_vocab_file=None,
                                      old_col_vocab_file=None,
                                      new_col_vocab_file=None,
                                      num_row_oov_buckets=0,
                                      num_col_oov_buckets=0,
                                      initializer=None,
                                      max_rows_in_memory=-1):
  r"""Returns a var initializer for loading and remapping a 2-D (matrix) tensor.

  The returned initializer loads a 2-D (matrix) `Tensor` with name
  `old_tensor_name` from the checkpoint at `ckpt_path`. It will reorder the
  rows/columns according to the specified vocab files and append additional
  out-of-vocabulary rows/columns according to the number of OOV buckets.

  The format of the file at the `{old,new}_{row,col}_vocab_file` path should be
  a text file, with each line containing a single entity within the vocabulary.
  Let the function `line_of(f, "x")` return the 0-indexed line number of the
  entity "x" in file f, and the function `entity_at(f, i)` return the entity at
  line i of file f. Then, row i of the new output matrix will be taken from row
  `line_of(old_row_vocab_file, entity_at(new_row_vocab_file, i))` of the old
  matrix. If any entity in `new_row_vocab_file` is not found in
  `old_row_vocab_file`, that row is considered a "missing" row, and its values
  will be initialized using the `initializer` arg. The same logic also applies
  for the columns.

  For example, assuming that:

  * `old_row_vocab_file` contains "mercury\nvenus\nmars"
  * `new_row_vocab_file` contains "venus\njupiter\nmercury"
  * `old_col_vocab_file` contains "good\nbetter\nbest"
  * `new_col_vocab_file` contains "good\nbest\nfantastic"
  * `initializer` returns the natural numbers `[1, 2, 3, 4, ...]`
  * `w(i, j)` represents the value from row i, column j of the old matrix

  Then the new output matrix will look like:

  `[[w(1, 0), w(1, 2), 1],
    [2,       3,       4],
    [w(0, 0), w(0, 2), 5]]`

  If we further specify that:

  * `num_row_oov_buckets` == 2
  * `num_col_oov_buckets` == 1

  Then the new output matrix will look like:

  `[[w(1, 0), w(1, 2), 1,  12],
    [2,       3,       4,  13],
    [w(0, 0), w(0, 2), 5,  14],
    [6,       7,       8,  15],
    [9,       10,      11, 16]]`

  If `{old,new}_row_vocab_file` are None, we assume that the old and new row
  vocab files are the same, and no row remapping is done. If
  `{old,new}_col_vocab_file` are None, we assume that the old and new column
  vocab files are the same, and no column remapping is done.

  The returned initializer only supports div-partitioning along the row axis. It
  does not support partitioning along the column axis or mod-partitioning.

  NOTE: When this is used to warm-start variables, client code should use
  `tf.lookup.index_table_from_tensor()` like
  contrib/layers/python/layers/feature_column.py does, as opposed to
  `tf.feature_to_id()` - in order to ensure the underlying lookup tables are the
  same.

  Args:
    ckpt_path: Path to the TensorFlow checkpoint (version 2, `TensorBundle`)
      from which the old matrix `Tensor` will be loaded.
    old_tensor_name: Name of the 2-D `Tensor` to load from checkpoint.
    new_row_vocab_size: `int` specifying the number of entries in
      `new_row_vocab_file`. If no row remapping is needed (no row vocab
      provided), this should be equal to the number of rows to load from the old
      matrix (which can theoretically be smaller than the number of rows in the
      old matrix).
    new_col_vocab_size: `int` specifying the number of entries in
      `new_col_vocab_file`. If no column remapping is needed (no column vocab
      provided), this should be equal to the number of columns in the old
      matrix.
    old_row_vocab_file: A scalar `Tensor` of type `string` containing the
      path to the old row vocabulary file. Can be None, which represents no
      remapping on the row axis.
    new_row_vocab_file: A scalar `Tensor` of type `string` containing the path
      to the new row vocabulary file. Can be None, which represents no remapping
      on the row axis.
    old_col_vocab_file: A scalar `Tensor` of type `string` containing the
      path to the old column vocabulary file. Can be None, which represents no
      remapping on the column axis.
    new_col_vocab_file: A scalar `Tensor` of type `string` containing the path
      to the new column vocabulary file. Can be None, which represents no
      remapping on the column axis.
    num_row_oov_buckets: `int` specifying the number of out-of-vocabulary rows
      to append. Must be >= 0.
    num_col_oov_buckets: `int` specifying the number of out-of-vocabulary
      columns to append. Must be >= 0.
    initializer: Initializer function to initialize missing values. Accepts a
      1-D tensor as the arg to specify the shape of the returned tensor. If
      `None`, defaults to using `zeros_initializer()`.
    max_rows_in_memory: `int` specifying the maximum number of rows to load from
      the checkpoint at once. If less than or equal to 0, the entire matrix will
      be loaded into memory. Setting this arg trades increased disk reads for
      lower memory usage.

  Returns:
    A variable initializer function that should be used to initialize a
    (potentially partitioned) `Variable` whose complete shape is
    `[new_row_vocab_size + num_row_oov_buckets, new_col_vocab_size +
    num_col_oov_buckets]`.

  Raises:
    TypeError: If `initializer` is specified but not callable.
  """
  if initializer is None:
    # TODO(b/25671353): Consider using sqrt(6/(fan_in + fan_out)) instead, from
    # Glorot and Bengio, 2010.
    initializer = init_ops.zeros_initializer()

  if not callable(initializer):
    raise TypeError(
        "initializer must be callable, instead of being {} of type {}.".format(
            initializer, type(initializer)))

  def _initializer(shape, dtype=dtypes.float32, partition_info=None):
    """Variable initializer.

    Args:
      shape: Shape of `Tensor` to return. Should include OOV on both axes.
      dtype: Must be float32.
      partition_info: variable_scope._PartitionInfo.

    Returns:
      `Tensor` of shape `shape`.

    Raises:
      TypeError: If `dtype` is anything other than float32.
      ValueError: For shape mismatch upon invocation.
    """
    # Sanity checks.
    if dtype != dtypes.float32:
      raise TypeError(
          "Currently, only float32 is supported. Received dtype: {}".format(
              dtype))
    if len(shape) != 2:
      raise ValueError("Expected 2-dim shape, but received: {}".format(shape))
    if shape[0] <= 0:
      raise ValueError(
          "Expected 1st dim of shape to be > 0, but received shape: {}".format(
              shape))
    if shape[1] != (new_col_vocab_size + num_col_oov_buckets):
      raise ValueError(
          "Expected 2nd dim of shape to be new_col_vocab_size ({}) + "
          "num_col_oov_buckets ({}) = {}, but received shape: {}".format(
              new_col_vocab_size, num_col_oov_buckets,
              new_col_vocab_size + num_col_oov_buckets, shape))

    offset = 0
    if partition_info is not None:
      offset = partition_info.single_offset(shape)

    if offset + shape[0] > new_row_vocab_size + num_row_oov_buckets:
      raise ValueError(
          "Trying to initialize {} additional rows after {} rows have already "
          "been initialized, which would exceed expected total row count of "
          "new_row_vocab_size ({}) + num_row_oov_buckets ({}) = {}.".format(
              shape[0], offset, new_row_vocab_size, num_row_oov_buckets,
              new_row_vocab_size + num_row_oov_buckets))

    row_oov_buckets_to_use = min(shape[0],
                                 max(0, offset + shape[0] - new_row_vocab_size))
    num_rows_to_load = shape[0] - row_oov_buckets_to_use

    return _load_and_remap_matrix(
        ckpt_path=ckpt_path,
        old_tensor_name=old_tensor_name,
        new_row_vocab_offset=offset,
        num_rows_to_load=num_rows_to_load,
        new_col_vocab_size=new_col_vocab_size,
        initializer=initializer,
        old_row_vocab_file=old_row_vocab_file,
        new_row_vocab_file=new_row_vocab_file,
        old_col_vocab_file=old_col_vocab_file,
        new_col_vocab_file=new_col_vocab_file,
        num_row_oov_buckets=row_oov_buckets_to_use,
        num_col_oov_buckets=num_col_oov_buckets,
        max_rows_in_memory=max_rows_in_memory)

  return _initializer


def load_embedding_initializer(ckpt_path,
                               embedding_tensor_name,
                               new_vocab_size,
                               embedding_dim,
                               old_vocab_file,
                               new_vocab_file,
                               num_oov_buckets=0,
                               initializer=None,
                               max_rows_in_memory=-1):
  """Returns a variable initializer for loading pre-trained embeddings.

  Wrapper around `load_and_remap_matrix_initializer()` specialized for loading
  embedding weights and remapping according to the provided vocab files. See
  docs for `load_and_remap_matrix_initializer()` for more details.

  NOTE: Only for use with div-partitioned variables / vocabularies.

  Args:
    ckpt_path: Path to the TensorFlow checkpoint (version 2, `TensorBundle`)
      from which the old matrix `Tensor` will be loaded.
    embedding_tensor_name: Name of the 2-D `Tensor` to load from checkpoint.
    new_vocab_size: Number of entries in the new vocab.
    embedding_dim: `int` specifying the dimension of the embedding vectors from
      the checkpoint. Must match the number of columns in the old embedding
      matrix.
    old_vocab_file: A scalar `Tensor` of type `string` containing the
      path to the old vocabulary file.
    new_vocab_file: A scalar `Tensor` of type `string` containing the
      path to the new vocabulary file.
    num_oov_buckets: `int` specifying the number of out-of-vocabulary
      buckets to use. Must be >= 0.
    initializer: Initializer function that accepts a 1-D tensor as the arg to
      specify the shape of the returned tensor. If `None`, defaults to using
      `truncated_normal_initializer()`.
    max_rows_in_memory: `int` specifying the maximum number of rows to load from
      the checkpoint at once. If less than or equal to 0, the entire matrix will
      be loaded into memory. Setting this arg trades increased disk reads for
      lower memory usage.

  Returns:
    A variable initializer function.
  """
  if initializer is None:
    # TODO(b/25671353): This should be kept in sync with the stddev used by
    # feature_column.py's _EmbeddingColumn.
    initializer = init_ops.truncated_normal_initializer(
        stddev=1.0 / math.sqrt(embedding_dim))

  return load_and_remap_matrix_initializer(
      ckpt_path=ckpt_path,
      old_tensor_name=embedding_tensor_name,
      new_row_vocab_size=new_vocab_size,
      new_col_vocab_size=embedding_dim,
      old_row_vocab_file=old_vocab_file,
      new_row_vocab_file=new_vocab_file,
      old_col_vocab_file=None,
      new_col_vocab_file=None,
      num_row_oov_buckets=num_oov_buckets,
      num_col_oov_buckets=0,
      initializer=initializer,
      max_rows_in_memory=max_rows_in_memory)


def load_linear_multiclass_bias_initializer(ckpt_path,
                                            bias_tensor_name,
                                            new_class_vocab_size,
                                            old_class_vocab_file,
                                            new_class_vocab_file,
                                            num_class_oov_buckets=0,
                                            initializer=None,
                                            max_rows_in_memory=-1):
  """Loads pre-trained multi-class biases for linear models from checkpoint.

  Wrapper around `load_and_remap_matrix_initializer()` specialized for loading
  multi-class bias and remapping according to the provided vocab files. See docs
  for `load_and_remap_matrix_initializer()` for more details. In this case, the
  provided row_vocab is the class vocabulary, and the expected shape is
  `[new_class_vocab_size, 1]`.

  Args:
    ckpt_path: Path to the TensorFlow checkpoint (version 2, `TensorBundle`)
      from which the old matrix `Tensor` will be loaded.
    bias_tensor_name: Tensor name to load from in the checkpoints.
    new_class_vocab_size: Number of entries in the new class vocab.
    old_class_vocab_file: A scalar `Tensor` of type `string` containing the
      path to the old class vocabulary file.
    new_class_vocab_file: A scalar `Tensor` of type `string` containing the
      path to the new class vocabulary file.
    num_class_oov_buckets: `int` specifying the number of out-of-vocabulary
      buckets to use for the classes. Must be >= 0.
    initializer: Initializer function that accepts a 1-D tensor as the arg to
      specify the shape of the returned tensor. If `None`, defaults to using
      `zeros_initializer()`.
    max_rows_in_memory: `int` specifying the maximum number of rows to load from
      the checkpoint at once. If less than or equal to 0, the entire matrix will
      be loaded into memory. Setting this arg trades increased disk reads for
      lower memory usage.

  Returns:
    A variable initializer function.
  """
  # Linear multi-class biases should be zero-initialized.
  if initializer is None:
    initializer = init_ops.zeros_initializer()

  return load_and_remap_matrix_initializer(
      ckpt_path=ckpt_path,
      old_tensor_name=bias_tensor_name,
      new_row_vocab_size=new_class_vocab_size,
      new_col_vocab_size=1,
      old_row_vocab_file=old_class_vocab_file,
      new_row_vocab_file=new_class_vocab_file,
      old_col_vocab_file=None,
      new_col_vocab_file=None,
      num_row_oov_buckets=num_class_oov_buckets,
      num_col_oov_buckets=0,
      initializer=initializer,
      max_rows_in_memory=max_rows_in_memory)


def load_variable_slot_initializer(ckpt_path,
                                   old_tensor_name,
                                   primary_partition_info,
                                   new_row_vocab_size,
                                   new_col_vocab_size,
                                   old_row_vocab_file=None,
                                   new_row_vocab_file=None,
                                   old_col_vocab_file=None,
                                   new_col_vocab_file=None,
                                   num_row_oov_buckets=0,
                                   num_col_oov_buckets=0,
                                   initializer=None,
                                   max_rows_in_memory=-1):
  """Loads pre-trained multi-class slots for linear models from checkpoint.

  Wrapper around `load_and_remap_matrix_initializer()` specialized for loading
  multi-class slots (such as optimizer accumulators) and remapping them
  according to the provided vocab files. See docs for
  `load_and_remap_matrix_initializer()` for more details.  Takes in a
  `variable_scope._PartitionInfo` representing the slot's primary `Variable`'s
  partitioning.  This is necessary since accumulator `Variable` creation ignores
  primary scoping and partitioning information.

  Args:
    ckpt_path: Path to the TensorFlow checkpoint (version 2, `TensorBundle`)
      from which the old matrix `Tensor` will be loaded.
    old_tensor_name: Name of the 2-D `Tensor` to load from checkpoint.
    primary_partition_info: A `variable_scope._PartitionInfo` containing this
      slot's primary `Variable`'s partitioning information.  This is used to
      calculate the offset and override the partition_info passed to the call to
      _initialize.
    new_row_vocab_size: `int` specifying the number of entries in
      `new_row_vocab_file`. If no row remapping is needed (no row vocab
      provided), this should be equal to the number of rows to load from the old
      matrix (which can theoretically be smaller than the number of rows in the
      old matrix).
    new_col_vocab_size: `int` specifying the number of entries in
      `new_col_vocab_file`. If no column remapping is needed (no column vocab
      provided), this should be equal to the number of columns in the old
      matrix.
    old_row_vocab_file: A scalar `Tensor` of type `string` containing the
      path to the old row vocabulary file. Can be None, which represents no
      remapping on the row axis.
    new_row_vocab_file: A scalar `Tensor` of type `string` containing the path
      to the new row vocabulary file. Can be None, which represents no remapping
      on the row axis.
    old_col_vocab_file: A scalar `Tensor` of type `string` containing the
      path to the old column vocabulary file. Can be None, which represents no
      remapping on the column axis.
    new_col_vocab_file: A scalar `Tensor` of type `string` containing the path
      to the new column vocabulary file. Can be None, which represents no
      remapping on the column axis.
    num_row_oov_buckets: `int` specifying the number of out-of-vocabulary rows
      to append. Must be >= 0.
    num_col_oov_buckets: `int` specifying the number of out-of-vocabulary
      columns to append. Must be >= 0.
    initializer: Initializer function to initialize missing values. Accepts a
      1-D tensor as the arg to specify the shape of the returned tensor. If
      `None`, defaults to using `zeros_initializer()`.
    max_rows_in_memory: `int` specifying the maximum number of rows to load from
      the checkpoint at once. If less than or equal to 0, the entire matrix will
      be loaded into memory. Setting this arg trades increased disk reads for
      lower memory usage.

  Returns:
    A variable initializer function that should be used to initialize a
    (potentially partitioned) `Variable` whose complete shape is
    `[new_row_vocab_size + num_row_oov_buckets, new_col_vocab_size +
    num_col_oov_buckets]`.

  Raises:
    TypeError: If `initializer` is specified but not callable.
  """
  initializer_fn = load_and_remap_matrix_initializer(
      ckpt_path=ckpt_path,
      old_tensor_name=old_tensor_name,
      new_row_vocab_size=new_row_vocab_size,
      new_col_vocab_size=new_col_vocab_size,
      old_row_vocab_file=old_row_vocab_file,
      new_row_vocab_file=new_row_vocab_file,
      old_col_vocab_file=old_col_vocab_file,
      new_col_vocab_file=new_col_vocab_file,
      num_row_oov_buckets=num_row_oov_buckets,
      num_col_oov_buckets=num_col_oov_buckets,
      initializer=initializer,
      max_rows_in_memory=max_rows_in_memory)

  def _initializer(shape, dtype=dtypes.float32, partition_info=None):
    del partition_info  # Unused by this override.
    return initializer_fn(shape, dtype, partition_info=primary_partition_info)

  return _initializer

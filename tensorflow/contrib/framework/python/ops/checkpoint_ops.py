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

from tensorflow.python.framework import dtypes
from tensorflow.python.ops import init_ops
from tensorflow.python.training import checkpoint_ops


# pylint: disable=protected-access,line-too-long
load_and_remap_matrix_initializer = checkpoint_ops._load_and_remap_matrix_initializer
# pylint: enable=line-too-long
load_embedding_initializer = checkpoint_ops._load_embedding_initializer
# pylint: enable=protected-access


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

# Sparse Tensors

Note: Functions taking `Tensor` arguments can also take anything accepted by
@{tf.convert_to_tensor}.

[TOC]

## Sparse Tensor Representation

TensorFlow supports a `SparseTensor` representation for data that is sparse
in multiple dimensions. Contrast this representation with `IndexedSlices`,
which is efficient for representing tensors that are sparse in their first
dimension, and dense along all other dimensions.

*   @{tf.SparseTensor}
*   @{tf.SparseTensorValue}

## Conversion

*   @{tf.sparse_to_dense}
*   @{tf.sparse_tensor_to_dense}
*   @{tf.sparse_to_indicator}
*   @{tf.sparse_merge}

## Manipulation

*   @{tf.sparse_concat}
*   @{tf.sparse_reorder}
*   @{tf.sparse_reshape}
*   @{tf.sparse_split}
*   @{tf.sparse_retain}
*   @{tf.sparse_reset_shape}
*   @{tf.sparse_fill_empty_rows}
*   @{tf.sparse_transpose}

## Reduction
*   @{tf.sparse_reduce_sum}
*   @{tf.sparse_reduce_sum_sparse}

## Math Operations
*   @{tf.sparse_add}
*   @{tf.sparse_softmax}
*   @{tf.sparse_tensor_dense_matmul}
*   @{tf.sparse_maximum}
*   @{tf.sparse_minimum}

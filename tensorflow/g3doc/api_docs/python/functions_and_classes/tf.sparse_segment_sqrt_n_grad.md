### `tf.sparse_segment_sqrt_n_grad(grad, indices, segment_ids, output_dim0, name=None)` {#sparse_segment_sqrt_n_grad}

Computes gradients for SparseSegmentSqrtN.

Returns tensor "output" with same shape as grad, except for dimension 0 whose
value is output_dim0.

##### Args:


*  <b>`grad`</b>: A `Tensor`. Must be one of the following types: `float32`, `float64`.
    gradient propagated to the SparseSegmentSqrtN op.
*  <b>`indices`</b>: A `Tensor` of type `int32`.
    indices passed to the corresponding SparseSegmentSqrtN op.
*  <b>`segment_ids`</b>: A `Tensor` of type `int32`.
    segment_ids passed to the corresponding SparseSegmentSqrtN op.
*  <b>`output_dim0`</b>: A `Tensor` of type `int32`.
    dimension 0 of "data" passed to SparseSegmentSqrtN op.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A `Tensor`. Has the same type as `grad`.


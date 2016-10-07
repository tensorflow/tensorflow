### `tf.sparse_segment_sqrt_n(data, indices, segment_ids, name=None)` {#sparse_segment_sqrt_n}

Computes the sum along sparse segments of a tensor divided by the sqrt of N.

N is the size of the segment being reduced.

Read [the section on
Segmentation](../../api_docs/python/math_ops.md#segmentation) for an explanation
of segments.

##### Args:


*  <b>`data`</b>: A `Tensor`. Must be one of the following types: `float32`, `float64`.
*  <b>`indices`</b>: A `Tensor`. Must be one of the following types: `int32`, `int64`.
    A 1-D tensor. Has same rank as `segment_ids`.
*  <b>`segment_ids`</b>: A `Tensor` of type `int32`.
    A 1-D tensor. Values should be sorted and can be repeated.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A `Tensor`. Has the same type as `data`.
  Has same shape as data, except for dimension 0 which
  has size `k`, the number of segments.


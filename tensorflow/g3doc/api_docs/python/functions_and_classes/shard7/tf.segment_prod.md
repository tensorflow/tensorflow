### `tf.segment_prod(data, segment_ids, name=None)` {#segment_prod}

Computes the product along segments of a tensor.

Read [the section on
Segmentation](../../api_docs/python/math_ops.md#segmentation) for an explanation
of segments.

Computes a tensor such that
\\(output_i = \prod_j data_j\\) where the product is over `j` such
that `segment_ids[j] == i`.

<div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
<img style="width:100%" src="../../images/SegmentProd.png" alt>
</div>

##### Args:


*  <b>`data`</b>: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int64`, `int32`, `uint8`, `uint16`, `int16`, `int8`, `complex64`, `complex128`, `qint8`, `quint8`, `qint32`, `half`.
*  <b>`segment_ids`</b>: A `Tensor`. Must be one of the following types: `int32`, `int64`.
    A 1-D tensor whose rank is equal to the rank of `data`'s
    first dimension.  Values should be sorted and can be repeated.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A `Tensor`. Has the same type as `data`.
  Has same shape as data, except for dimension 0 which
  has size `k`, the number of segments.


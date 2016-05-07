### `tf.segment_mean(data, segment_ids, name=None)` {#segment_mean}

Computes the mean along segments of a tensor.

Read [the section on
Segmentation](../../api_docs/python/math_ops.md#segmentation) for an explanation
of segments.

Computes a tensor such that
\\(output_i = \frac{\sum_j data_j}{N}\\) where `mean` is
over `j` such that `segment_ids[j] == i` and `N` is the total number of
values summed.

<div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
<img style="width:100%" src="../../images/SegmentMean.png" alt>
</div>

##### Args:


*  <b>`data`</b>: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `int64`, `uint8`, `int16`, `int8`, `uint16`, `half`.
*  <b>`segment_ids`</b>: A `Tensor`. Must be one of the following types: `int32`, `int64`.
    A 1-D tensor whose rank is equal to the rank of `data`'s
    first dimension.  Values should be sorted and can be repeated.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A `Tensor`. Has the same type as `data`.
  Has same shape as data, except for dimension 0 which
  has size `k`, the number of segments.


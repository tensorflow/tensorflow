### `tf.unsorted_segment_sum(data, segment_ids, num_segments, name=None)` {#unsorted_segment_sum}

Computes the sum along segments of a tensor.

Read [the section on
Segmentation](../../api_docs/python/math_ops.md#segmentation) for an explanation
of segments.

Computes a tensor such that
`(output[i] = sum_{j...} data[j...]` where the sum is over tuples `j...` such
that `segment_ids[j...] == i`.  Unlike `SegmentSum`, `segment_ids`
need not be sorted and need not cover all values in the full
range of valid values.

If the sum is empty for a given segment ID `i`, `output[i] = 0`.

`num_segments` should equal the number of distinct segment IDs.

<div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
<img style="width:100%" src="../../images/UnsortedSegmentSum.png" alt>
</div>

##### Args:


*  <b>`data`</b>: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int64`, `int32`, `uint8`, `uint16`, `int16`, `int8`, `complex64`, `complex128`, `qint8`, `quint8`, `qint32`, `half`.
*  <b>`segment_ids`</b>: A `Tensor`. Must be one of the following types: `int32`, `int64`.
    A tensor whose shape is a prefix of `data.shape`.
*  <b>`num_segments`</b>: A `Tensor` of type `int32`.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A `Tensor`. Has the same type as `data`.
  Has same shape as data, except for the first `segment_ids.rank`
  dimensions, which are replaced with a single dimension which has size
  `num_segments`.


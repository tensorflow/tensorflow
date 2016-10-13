### `tf.image.non_max_suppression(boxes, scores, max_output_size, iou_threshold=None, name=None)` {#non_max_suppression}

Greedily selects a subset of bounding boxes in descending order of score,

pruning away boxes that have high intersection-over-union (IOU) overlap
with previously selected boxes.  Bounding boxes are supplied as
[y1, x1, y2, x2], where (y1, x1) and (y2, x2) are the coordinates of any
diagonal pair of box corners and the coordinates can be provided as normalized
(i.e., lying in the interval [0, 1]) or absolute.  Note that this algorithm
is agnostic to where the origin is in the coordinate system.  Note that this
algorithm is invariant to orthogonal transformations and translations
of the coordinate system; thus translating or reflections of the coordinate
system result in the same boxes being selected by the algorithm.

The output of this operation is a set of integers indexing into the input
collection of bounding boxes representing the selected boxes.  The bounding
box coordinates corresponding to the selected indices can then be obtained
using the `tf.gather operation`.  For example:

  selected_indices = tf.image.non_max_suppression(
      boxes, scores, max_output_size, iou_threshold)
  selected_boxes = tf.gather(boxes, selected_indices)

##### Args:


*  <b>`boxes`</b>: A `Tensor` of type `float32`.
    A 2-D float tensor of shape `[num_boxes, 4]`.
*  <b>`scores`</b>: A `Tensor` of type `float32`.
    A 1-D float tensor of shape `[num_boxes]` representing a single
    score corresponding to each box (each row of boxes).
*  <b>`max_output_size`</b>: A `Tensor` of type `int32`.
    A scalar integer tensor representing the maximum number of
    boxes to be selected by non max suppression.
*  <b>`iou_threshold`</b>: An optional `float`. Defaults to `0.5`.
    A float representing the threshold for deciding whether boxes
    overlap too much with respect to IOU.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A `Tensor` of type `int32`.
  A 1-D integer tensor of shape `[M]` representing the selected
  indices from the boxes tensor, where `M <= max_output_size`.


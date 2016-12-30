### `tf.image.draw_bounding_boxes(images, boxes, name=None)` {#draw_bounding_boxes}

Draw bounding boxes on a batch of images.

Outputs a copy of `images` but draws on top of the pixels zero or more bounding
boxes specified by the locations in `boxes`. The coordinates of the each
bounding box in `boxes` are encoded as `[y_min, x_min, y_max, x_max]`. The
bounding box coordinates are floats in `[0.0, 1.0]` relative to the width and
height of the underlying image.

For example, if an image is 100 x 200 pixels and the bounding box is
`[0.1, 0.2, 0.5, 0.9]`, the bottom-left and upper-right coordinates of the
bounding box will be `(10, 40)` to `(50, 180)`.

Parts of the bounding box may fall outside the image.

##### Args:


*  <b>`images`</b>: A `Tensor`. Must be one of the following types: `float32`, `half`.
    4-D with shape `[batch, height, width, depth]`. A batch of images.
*  <b>`boxes`</b>: A `Tensor` of type `float32`.
    3-D with shape `[batch, num_bounding_boxes, 4]` containing bounding
    boxes.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A `Tensor`. Has the same type as `images`.
  4-D with the same shape as `images`. The batch of input images with
  bounding boxes drawn on the images.


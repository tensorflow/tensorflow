#include "tensorflow/core/framework/op.h"

namespace tensorflow {

// Operators that deal with SummaryProtos (encoded as DT_STRING tensors) as
// inputs or outputs in various ways.

REGISTER_OP("ScalarSummary")
    .Input("tags: string")
    .Input("values: T")
    .Output("summary: string")
    .Attr("T: {float, double}")
    .Doc(R"doc(
Outputs a `Summary` protocol buffer with scalar values.

The input `tags` and `values` must have the same shape.  The generated summary
has a summary value for each tag-value pair in `tags` and `values`.

tags: 1-D. Tags for the summary.
values: 1-D, same size as `tags.  Values for the summary.
summary: Scalar.  Serialized `Summary` protocol buffer.
)doc");

REGISTER_OP("HistogramSummary")
    .Input("tag: string")
    .Input("values: float")
    .Output("summary: string")
    .Doc(R"doc(
Outputs a `Summary` protocol buffer with a histogram.

The generated
[`Summary`](https://tensorflow.googlesource.com/tensorflow/+/master/tensorflow/core/framework/summary.proto)
has one summary value containing a histogram for `values`.

This op reports an `OutOfRange` error if any value is not finite.

tag: Scalar.  Tag to use for the `Summary.Value`.
values: Any shape. Values to use to build the histogram.
summary: Scalar. Serialized `Summary` protocol buffer.
)doc");

REGISTER_OP("ImageSummary")
    .Input("tag: string")
    .Input("tensor: float")
    .Output("summary: string")
    .Attr("max_images: int >= 1 = 3")
    .Attr(
        "bad_color: tensor = { dtype: DT_UINT8 "
        "tensor_shape: { dim { size: 4 } } "
        "int_val: 255 int_val: 0 int_val: 0 int_val: 255 }")
    .Doc(R"doc(
Outputs a `Summary` protocol buffer with images.

The summary has up to `max_images` summary values containing images. The
images are built from `tensor` which must be 4-D with shape `[batch_size,
height, width, channels]` and where `channels` can be:

*  1: `tensor` is interpreted as Grayscale.
*  3: `tensor` is interpreted as RGB.
*  4: `tensor` is interpreted as RGBA.

The images have the same number of channels as the input tensor. Their values
are normalized, one image at a time, to fit in the range `[0, 255]`.  The
op uses two different normalization algorithms:

*  If the input values are all positive, they are rescaled so the largest one
   is 255.

*  If any input value is negative, the values are shifted so input value 0.0
   is at 127.  They are then rescaled so that either the smallest value is 0,
   or the largest one is 255.

The `tag` argument is a scalar `Tensor` of type `string`.  It is used to
build the `tag` of the summary values:

*  If `max_images` is 1, the summary value tag is '*tag*/image'.
*  If `max_images` is greater than 1, the summary value tags are
   generated sequentially as '*tag*/image/0', '*tag*/image/1', etc.

The `bad_color` argument is the color to use in the generated images for
non-finite input values.  It is a `unit8` 1-D tensor of length `channels`.
Each element must be in the range `[0, 255]` (It represents the value of a
pixel in the output image).  Non-finite values in the input tensor are
replaced by this tensor in the output image.  The default value is the color
red.

tag: Scalar. Used to build the `tag` attribute of the summary values.
tensor: 4-D of shape `[batch_size, height, width, channels]` where
  `channels` is 1, 3, or 4.
max_images: Max number of batch elements to generate images for.
bad_color: Color to use for pixels with non-finite values.
summary: Scalar. Serialized `Summary` protocol buffer.
)doc");

REGISTER_OP("MergeSummary")
    .Input("inputs: N * string")
    .Output("summary: string")
    .Attr("N : int >= 1")
    .Doc(R"doc(
Merges summaries.

This op creates a
[`Summary`](https://tensorflow.googlesource.com/tensorflow/+/master/tensorflow/core/framework/summary.proto)
protocol buffer that contains the union of all the values in the input
summaries.

When the Op is run, it reports an `InvalidArgument` error if multiple values
in the summaries to merge use the same tag.

inputs: Can be of any shape.  Each must contain serialized `Summary` protocol
  buffers.
summary: Scalar. Serialized `Summary` protocol buffer.
)doc");

}  // namespace tensorflow

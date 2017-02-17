/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

namespace tensorflow {

REGISTER_OP("Assert")
    .Input("condition: bool")
    .Input("data: T")
    .SetIsStateful()
    .Attr("T: list(type)")
    .Attr("summarize: int = 3")
    .SetShapeFn(shape_inference::NoOutputs)
    .Doc(R"doc(
Asserts that the given condition is true.

If `condition` evaluates to false, print the list of tensors in `data`.
`summarize` determines how many entries of the tensors to print.

condition: The condition to evaluate.
data: The tensors to print out when condition is false.
summarize: Print this many entries of each tensor.
)doc");

REGISTER_OP("Print")
    .Input("input: T")
    .Input("data: U")
    .Output("output: T")
    .SetIsStateful()
    .Attr("T: type")
    .Attr("U: list(type)")
    .Attr("message: string = ''")
    .Attr("first_n: int = -1")
    .Attr("summarize: int = 3")
    .SetShapeFn(shape_inference::UnchangedShape)
    .Doc(R"doc(
Prints a list of tensors.

Passes `input` through to `output` and prints `data` when evaluating.

input: The tensor passed to `output`
data: A list of tensors to print out when op is evaluated.
output:= The unmodified `input` tensor
message: A string, prefix of the error message.
first_n: Only log `first_n` number of times. -1 disables logging.
summarize: Only print this many entries of each tensor.
)doc");

// ----------------------------------------------------------------------------
// Operators that deal with SummaryProtos (encoded as DT_STRING tensors) as
// inputs or outputs in various ways.

REGISTER_OP("TensorSummary")
    .Input("tensor: T")
    .Output("summary: string")
    .Attr("T: type")
    .Attr("description: string = ''")
    .Attr("labels: list(string) = []")
    .Attr("display_name: string = ''")
    .SetShapeFn(shape_inference::ScalarShape)
    .Doc(R"doc(
Outputs a `Summary` protocol buffer with a tensor.

tensor: A tensor to serialize.
description: A json-encoded SummaryDescription proto.
labels: An unused list of strings.
display_name: An unused string.
)doc");

REGISTER_OP("ScalarSummary")
    .Input("tags: string")
    .Input("values: T")
    .Output("summary: string")
    .Attr("T: realnumbertype")
    .SetShapeFn(shape_inference::ScalarShape)
    .Doc(R"doc(
Outputs a `Summary` protocol buffer with scalar values.

The input `tags` and `values` must have the same shape.  The generated summary
has a summary value for each tag-value pair in `tags` and `values`.

tags: Tags for the summary.
values: Same shape as `tags.  Values for the summary.
summary: Scalar.  Serialized `Summary` protocol buffer.
)doc");

REGISTER_OP("HistogramSummary")
    .Input("tag: string")
    .Input("values: T")
    .Output("summary: string")
    .Attr("T: realnumbertype = DT_FLOAT")
    .SetShapeFn(shape_inference::ScalarShape)
    .Doc(R"doc(
Outputs a `Summary` protocol buffer with a histogram.

The generated
[`Summary`](https://www.tensorflow.org/code/tensorflow/core/framework/summary.proto)
has one summary value containing a histogram for `values`.

This op reports an `InvalidArgument` error if any value is not finite.

tag: Scalar.  Tag to use for the `Summary.Value`.
values: Any shape. Values to use to build the histogram.
summary: Scalar. Serialized `Summary` protocol buffer.
)doc");

REGISTER_OP("ImageSummary")
    .Input("tag: string")
    .Input("tensor: T")
    .Output("summary: string")
    .Attr("max_images: int >= 1 = 3")
    .Attr("T: {uint8, float, half} = DT_FLOAT")
    .Attr(
        "bad_color: tensor = { dtype: DT_UINT8 "
        "tensor_shape: { dim { size: 4 } } "
        "int_val: 255 int_val: 0 int_val: 0 int_val: 255 }")
    .SetShapeFn(shape_inference::ScalarShape)
    .Doc(R"doc(
Outputs a `Summary` protocol buffer with images.

The summary has up to `max_images` summary values containing images. The
images are built from `tensor` which must be 4-D with shape `[batch_size,
height, width, channels]` and where `channels` can be:

*  1: `tensor` is interpreted as Grayscale.
*  3: `tensor` is interpreted as RGB.
*  4: `tensor` is interpreted as RGBA.

The images have the same number of channels as the input tensor. For float
input, the values are normalized one image at a time to fit in the range
`[0, 255]`.  `uint8` values are unchanged.  The op uses two different
normalization algorithms:

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

REGISTER_OP("AudioSummaryV2")
    .Input("tag: string")
    .Input("tensor: float")
    .Input("sample_rate: float")
    .Output("summary: string")
    .Attr("max_outputs: int >= 1 = 3")
    .SetShapeFn(shape_inference::ScalarShape)
    .Doc(R"doc(
Outputs a `Summary` protocol buffer with audio.

The summary has up to `max_outputs` summary values containing audio. The
audio is built from `tensor` which must be 3-D with shape `[batch_size,
frames, channels]` or 2-D with shape `[batch_size, frames]`. The values are
assumed to be in the range of `[-1.0, 1.0]` with a sample rate of `sample_rate`.

The `tag` argument is a scalar `Tensor` of type `string`.  It is used to
build the `tag` of the summary values:

*  If `max_outputs` is 1, the summary value tag is '*tag*/audio'.
*  If `max_outputs` is greater than 1, the summary value tags are
   generated sequentially as '*tag*/audio/0', '*tag*/audio/1', etc.

tag: Scalar. Used to build the `tag` attribute of the summary values.
tensor: 2-D of shape `[batch_size, frames]`.
sample_rate: The sample rate of the signal in hertz.
max_outputs: Max number of batch elements to generate audio for.
summary: Scalar. Serialized `Summary` protocol buffer.
)doc");

REGISTER_OP("AudioSummary")
    .Input("tag: string")
    .Input("tensor: float")
    .Output("summary: string")
    .Attr("sample_rate: float")
    .Attr("max_outputs: int >= 1 = 3")
    .SetShapeFn(shape_inference::ScalarShape)
    .Deprecated(15, "Use AudioSummaryV2.")
    .Doc(R"doc(
Outputs a `Summary` protocol buffer with audio.

The summary has up to `max_outputs` summary values containing audio. The
audio is built from `tensor` which must be 3-D with shape `[batch_size,
frames, channels]` or 2-D with shape `[batch_size, frames]`. The values are
assumed to be in the range of `[-1.0, 1.0]` with a sample rate of `sample_rate`.

The `tag` argument is a scalar `Tensor` of type `string`.  It is used to
build the `tag` of the summary values:

*  If `max_outputs` is 1, the summary value tag is '*tag*/audio'.
*  If `max_outputs` is greater than 1, the summary value tags are
   generated sequentially as '*tag*/audio/0', '*tag*/audio/1', etc.

tag: Scalar. Used to build the `tag` attribute of the summary values.
tensor: 2-D of shape `[batch_size, frames]`.
sample_rate: The sample rate of the signal in hertz.
max_outputs: Max number of batch elements to generate audio for.
summary: Scalar. Serialized `Summary` protocol buffer.
)doc");

REGISTER_OP("MergeSummary")
    .Input("inputs: N * string")
    .Output("summary: string")
    .Attr("N : int >= 1")
    .SetShapeFn(shape_inference::ScalarShape)
    .Doc(R"doc(
Merges summaries.

This op creates a
[`Summary`](https://www.tensorflow.org/code/tensorflow/core/framework/summary.proto)
protocol buffer that contains the union of all the values in the input
summaries.

When the Op is run, it reports an `InvalidArgument` error if multiple values
in the summaries to merge use the same tag.

inputs: Can be of any shape.  Each must contain serialized `Summary` protocol
  buffers.
summary: Scalar. Serialized `Summary` protocol buffer.
)doc");

}  // end namespace tensorflow

/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");

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

REGISTER_OP("SummaryWriter")
    .Output("writer: resource")
    .Attr("shared_name: string = ''")
    .Attr("container: string = ''")
    .SetShapeFn(shape_inference::ScalarShape)
    .Doc(R"doc(
Returns a handle to be used to access a summary writer.

The summary writer is an in-graph resource which can be used by ops to write
summaries to event files.

writer: the summary writer resource. Scalar handle.
)doc");

REGISTER_OP("CreateSummaryFileWriter")
    .Input("writer: resource")
    .Input("logdir: string")
    .Input("max_queue: int32")
    .Input("flush_millis: int32")
    .Input("filename_suffix: string")
    .Doc(R"doc(
Creates a summary file writer accessible by the given resource handle.

writer: A handle to the summary writer resource
logdir: Directory where the event file will be written.
max_queue: Size of the queue of pending events and summaries.
flush_millis: How often, in milliseconds, to flush the pending events and
  summaries to disk.
filename_suffix: Every event file's name is suffixed with this suffix.
)doc");

REGISTER_OP("FlushSummaryWriter")
    .Input("writer: resource")
    .SetShapeFn(shape_inference::NoOutputs)
    .Doc(R"(
Flushes the writer's unwritten events.

writer: A handle to the summary writer resource.
)");

REGISTER_OP("CloseSummaryWriter")
    .Input("writer: resource")
    .SetShapeFn(shape_inference::NoOutputs)
    .Doc(R"(
Flushes and closes the summary writer.

Also removes it from the resource manager. To reopen, use another
CreateSummaryFileWriter op.

writer: A handle to the summary writer resource.
)");

REGISTER_OP("WriteSummary")
    .Input("writer: resource")
    .Input("global_step: int64")
    .Input("tensor: T")
    .Input("tag: string")
    .Input("summary_metadata: string")
    .Attr("T: type")
    .SetShapeFn(shape_inference::NoOutputs)
    .Doc(R"doc(
Outputs a `Summary` protocol buffer with a tensor.

writer: A handle to a summary writer.
global_step: The step to write the summary for.
tensor: A tensor to serialize.
tag: The summary's tag.
summary_metadata: Serialized SummaryMetadata protocol buffer containing
 plugin-related metadata for this summary.
)doc");

REGISTER_OP("WriteScalarSummary")
    .Input("writer: resource")
    .Input("global_step: int64")
    .Input("tag: string")
    .Input("value: T")
    .Attr("T: realnumbertype")
    .SetShapeFn(shape_inference::NoOutputs)
    .Doc(R"doc(
Writes a `Summary` protocol buffer with scalar values.

The input `tag` and `value` must have the scalars.

writer: A handle to a summary writer.
global_step: The step to write the summary for.
tag: Tag for the summary.
value: Value for the summary.
)doc");

REGISTER_OP("WriteHistogramSummary")
    .Input("writer: resource")
    .Input("global_step: int64")
    .Input("tag: string")
    .Input("values: T")
    .Attr("T: realnumbertype = DT_FLOAT")
    .SetShapeFn(shape_inference::NoOutputs)
    .Doc(R"doc(
Writes a `Summary` protocol buffer with a histogram.

The generated
[`Summary`](https://www.tensorflow.org/code/tensorflow/core/framework/summary.proto)
has one summary value containing a histogram for `values`.

This op reports an `InvalidArgument` error if any value is not finite.

writer: A handle to a summary writer.
global_step: The step to write the summary for.
tag: Scalar.  Tag to use for the `Summary.Value`.
values: Any shape. Values to use to build the histogram.
)doc");

REGISTER_OP("WriteImageSummary")
    .Input("writer: resource")
    .Input("global_step: int64")
    .Input("tag: string")
    .Input("tensor: T")
    .Input("bad_color: uint8")
    .Attr("max_images: int >= 1 = 3")
    .Attr("T: {uint8, float, half} = DT_FLOAT")
    .SetShapeFn(shape_inference::NoOutputs)
    .Doc(R"doc(
Writes a `Summary` protocol buffer with images.

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

writer: A handle to a summary writer.
global_step: The step to write the summary for.
tag: Scalar. Used to build the `tag` attribute of the summary values.
tensor: 4-D of shape `[batch_size, height, width, channels]` where
  `channels` is 1, 3, or 4.
max_images: Max number of batch elements to generate images for.
bad_color: Color to use for pixels with non-finite values.
)doc");

REGISTER_OP("WriteAudioSummary")
    .Input("writer: resource")
    .Input("global_step: int64")
    .Input("tag: string")
    .Input("tensor: float")
    .Input("sample_rate: float")
    .Attr("max_outputs: int >= 1 = 3")
    .SetShapeFn(shape_inference::NoOutputs)
    .Doc(R"doc(
Writes a `Summary` protocol buffer with audio.

The summary has up to `max_outputs` summary values containing audio. The
audio is built from `tensor` which must be 3-D with shape `[batch_size,
frames, channels]` or 2-D with shape `[batch_size, frames]`. The values are
assumed to be in the range of `[-1.0, 1.0]` with a sample rate of `sample_rate`.

The `tag` argument is a scalar `Tensor` of type `string`.  It is used to
build the `tag` of the summary values:

*  If `max_outputs` is 1, the summary value tag is '*tag*/audio'.
*  If `max_outputs` is greater than 1, the summary value tags are
   generated sequentially as '*tag*/audio/0', '*tag*/audio/1', etc.

writer: A handle to a summary writer.
global_step: The step to write the summary for.
tag: Scalar. Used to build the `tag` attribute of the summary values.
tensor: 2-D of shape `[batch_size, frames]`.
sample_rate: The sample rate of the signal in hertz.
max_outputs: Max number of batch elements to generate audio for.
)doc");

}  // namespace tensorflow

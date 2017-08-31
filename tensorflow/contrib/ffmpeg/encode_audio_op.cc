// Copyright 2016 The TensorFlow Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================

#include <limits>

#include "tensorflow/contrib/ffmpeg/ffmpeg_lib.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"

namespace tensorflow {
namespace ffmpeg {
namespace {

/*
 * Encoding implementation, shared across V1 and V2 ops. Creates a new
 * output in the context.
 */
void Encode(OpKernelContext* context, const Tensor& contents,
            const string& file_format, const int32 bits_per_second,
            const int32 samples_per_second) {
  std::vector<float> samples;
  samples.reserve(contents.NumElements());
  for (int32 i = 0; i < contents.NumElements(); ++i) {
    samples.push_back(contents.flat<float>()(i));
  }
  const int32 channel_count = contents.dim_size(1);
  string encoded_audio;
  OP_REQUIRES_OK(
      context, CreateAudioFile(file_format, bits_per_second, samples_per_second,
                               channel_count, samples, &encoded_audio));

  // Copy the encoded audio file to the output tensor.
  Tensor* output = nullptr;
  OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape(), &output));
  output->scalar<string>()() = encoded_audio;
}

}  // namespace

/*
 * Supersedes `EncodeAudioOp`. Allows all parameters to be inputs
 * instead of attributes, so that the sample rate (and, probably less
 * usefully, the output file format) can be given as tensors rather than
 * constants only.
 */
class EncodeAudioOpV2 : public OpKernel {
 public:
  explicit EncodeAudioOpV2(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    OP_REQUIRES(
        context, context->num_inputs() == 4,
        errors::InvalidArgument("EncodeAudio requires exactly four inputs."));

    const Tensor& contents = context->input(0);
    const Tensor& file_format_tensor = context->input(1);
    const Tensor& samples_per_second_tensor = context->input(2);
    const Tensor& bits_per_second_tensor = context->input(3);

    OP_REQUIRES(context, TensorShapeUtils::IsMatrix(contents.shape()),
                errors::InvalidArgument(
                    "sampled_audio must be a rank-2 tensor but got shape ",
                    contents.shape().DebugString()));
    OP_REQUIRES(
        context, contents.NumElements() <= std::numeric_limits<int32>::max(),
        errors::InvalidArgument(
            "sampled_audio cannot have more than 2^31 entries. Shape = ",
            contents.shape().DebugString()));
    OP_REQUIRES(context, TensorShapeUtils::IsScalar(file_format_tensor.shape()),
                errors::InvalidArgument(
                    "file_format must be a rank-0 tensor but got shape ",
                    file_format_tensor.shape().DebugString()));
    OP_REQUIRES(context,
                TensorShapeUtils::IsScalar(samples_per_second_tensor.shape()),
                errors::InvalidArgument(
                    "samples_per_second must be a rank-0 tensor but got shape ",
                    samples_per_second_tensor.shape().DebugString()));
    OP_REQUIRES(context,
                TensorShapeUtils::IsScalar(bits_per_second_tensor.shape()),
                errors::InvalidArgument(
                    "bits_per_second must be a rank-0 tensor but got shape ",
                    bits_per_second_tensor.shape().DebugString()));

    const string file_format =
        str_util::Lowercase(file_format_tensor.scalar<string>()());
    const int32 samples_per_second =
        samples_per_second_tensor.scalar<int32>()();
    const int32 bits_per_second = bits_per_second_tensor.scalar<int32>()();

    OP_REQUIRES(context, file_format == "wav",
                errors::InvalidArgument(
                    "file_format must be \"wav\", but got: ", file_format));
    OP_REQUIRES(context, samples_per_second > 0,
                errors::InvalidArgument(
                    "samples_per_second must be positive, but got: ",
                    samples_per_second));
    OP_REQUIRES(
        context, bits_per_second > 0,
        errors::InvalidArgument("bits_per_second must be positive, but got: ",
                                bits_per_second));

    Encode(context, contents, file_format, bits_per_second, samples_per_second);
  }
};

REGISTER_KERNEL_BUILDER(Name("EncodeAudioV2").Device(DEVICE_CPU),
                        EncodeAudioOpV2);

REGISTER_OP("EncodeAudioV2")
    .Input("sampled_audio: float")
    .Input("file_format: string")
    .Input("samples_per_second: int32")
    .Input("bits_per_second: int32")
    .Output("contents: string")
    .SetShapeFn(shape_inference::ScalarShape)
    .Doc(R"doc(
Processes a `Tensor` containing sampled audio with the number of channels
and length of the audio specified by the dimensions of the `Tensor`. The
audio is converted into a string that, when saved to disk, will be equivalent
to the audio in the specified audio format.

The input audio has one row of the tensor for each channel in the audio file.
Each channel contains audio samples starting at the beginning of the audio and
having `1/samples_per_second` time between them. The output file will contain
all of the audio channels contained in the tensor.

sampled_audio: A rank-2 float tensor containing all tracks of the audio.
    Dimension 0 is time and dimension 1 is the channel.
file_format: A string or rank-0 string tensor describing the audio file
    format. This value must be `"wav"`.
samples_per_second: The number of samples per second that the audio should
    have, as an int or rank-0 `int32` tensor. This value must be
    positive.
bits_per_second: The approximate bitrate of the encoded audio file, as
    an int or rank-0 `int32` tensor. This is ignored by the "wav" file
    format.
contents: The binary audio file contents, as a rank-0 string tensor.
)doc");

/*
 * Deprecated in favor of EncodeAudioOpV2.
 */
class EncodeAudioOp : public OpKernel {
 public:
  explicit EncodeAudioOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("file_format", &file_format_));
    file_format_ = str_util::Lowercase(file_format_);
    OP_REQUIRES(context, file_format_ == "wav",
                errors::InvalidArgument("file_format arg must be \"wav\"."));

    OP_REQUIRES_OK(
        context, context->GetAttr("samples_per_second", &samples_per_second_));
    OP_REQUIRES(context, samples_per_second_ > 0,
                errors::InvalidArgument("samples_per_second must be > 0."));
    OP_REQUIRES_OK(context,
                   context->GetAttr("bits_per_second", &bits_per_second_));
  }

  void Compute(OpKernelContext* context) override {
    // Get and verify the input data.
    OP_REQUIRES(
        context, context->num_inputs() == 1,
        errors::InvalidArgument("EncodeAudio requires exactly one input."));
    const Tensor& contents = context->input(0);
    OP_REQUIRES(context, TensorShapeUtils::IsMatrix(contents.shape()),
                errors::InvalidArgument(
                    "sampled_audio must be a rank 2 tensor but got shape ",
                    contents.shape().DebugString()));
    OP_REQUIRES(
        context, contents.NumElements() <= std::numeric_limits<int32>::max(),
        errors::InvalidArgument(
            "sampled_audio cannot have more than 2^31 entries. Shape = ",
            contents.shape().DebugString()));

    Encode(context, contents, file_format_, bits_per_second_,
           samples_per_second_);
  }

 private:
  string file_format_;
  int32 samples_per_second_;
  int32 bits_per_second_;
};

REGISTER_KERNEL_BUILDER(Name("EncodeAudio").Device(DEVICE_CPU), EncodeAudioOp);

REGISTER_OP("EncodeAudio")
    .Input("sampled_audio: float")
    .Output("contents: string")
    .Attr("file_format: string")
    .Attr("samples_per_second: int")
    .Attr("bits_per_second: int = 192000")
    .SetShapeFn(shape_inference::ScalarShape)
    .Doc(R"doc(
Processes a `Tensor` containing sampled audio with the number of channels
and length of the audio specified by the dimensions of the `Tensor`. The
audio is converted into a string that, when saved to disk, will be equivalent
to the audio in the specified audio format.

The input audio has one row of the tensor for each channel in the audio file.
Each channel contains audio samples starting at the beginning of the audio and
having `1/samples_per_second` time between them. The output file will contain
all of the audio channels contained in the tensor.

sampled_audio: A rank 2 tensor containing all tracks of the audio. Dimension 0
    is time and dimension 1 is the channel.
contents: The binary audio file contents.
file_format: A string describing the audio file format. This must be "wav".
samples_per_second: The number of samples per second that the audio should have.
bits_per_second: The approximate bitrate of the encoded audio file. This is
    ignored by the "wav" file format.
)doc");

}  // namespace ffmpeg
}  // namespace tensorflow

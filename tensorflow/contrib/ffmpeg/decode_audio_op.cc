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

#include <stdlib.h>

#include <cstdio>
#include <set>

#include "tensorflow/contrib/ffmpeg/ffmpeg_lib.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/logging.h"

namespace tensorflow {
namespace ffmpeg {
namespace {

// The complete set of audio file formats that are supported by the op. These
// strings are defined by FFmpeg and documented here:
// https://www.ffmpeg.org/ffmpeg-formats.html
const char* kValidFileFormats[] = {"mp3", "mp4", "ogg", "wav"};

/*
 * Decoding implementation, shared across V1 and V2 ops. Creates a new
 * output in the context.
 */
void Decode(OpKernelContext* context,
            const tensorflow::StringPiece& file_contents,
            const string& file_format, const int32 samples_per_second,
            const int32 channel_count, const string& stream) {
  // Write the input data to a temp file.
  const string temp_filename = io::GetTempFilename(file_format);
  OP_REQUIRES_OK(context, WriteFile(temp_filename, file_contents));
  FileDeleter deleter(temp_filename);

  // Run FFmpeg on the data and verify results.
  std::vector<float> output_samples;
  Status result =
      ffmpeg::ReadAudioFile(temp_filename, file_format, samples_per_second,
                            channel_count, stream, &output_samples);
  if (result.code() == error::Code::NOT_FOUND) {
    OP_REQUIRES(
        context, result.ok(),
        errors::Unavailable("FFmpeg must be installed to run this op. FFmpeg "
                            "can be found at http://www.ffmpeg.org."));
  } else if (result.code() == error::UNKNOWN) {
    LOG(ERROR) << "Ffmpeg failed with error '" << result.error_message()
               << "'. Returning empty tensor.";
    Tensor* output = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, TensorShape({0, 0}), &output));
    return;
  } else {
    OP_REQUIRES_OK(context, result);
  }
  OP_REQUIRES(context, !output_samples.empty(),
              errors::Unknown("No output created by FFmpeg."));
  OP_REQUIRES(
      context, output_samples.size() % channel_count == 0,
      errors::Unknown("FFmpeg created non-integer number of audio frames."));

  // Copy the output data to the output Tensor.
  Tensor* output = nullptr;
  const int64 frame_count = output_samples.size() / channel_count;
  OP_REQUIRES_OK(context,
                 context->allocate_output(
                     0, TensorShape({frame_count, channel_count}), &output));
  auto matrix = output->tensor<float, 2>();
  for (int32 frame = 0; frame < frame_count; ++frame) {
    for (int32 channel = 0; channel < channel_count; ++channel) {
      matrix(frame, channel) = output_samples[frame * channel_count + channel];
    }
  }
}

}  // namespace

/*
 * Supersedes `DecodeAudioOp`. Allows all parameters to be inputs
 * instead of attributes, so that they can be given as tensors rather
 * than constants only.
 */
class DecodeAudioOpV2 : public OpKernel {
 public:
  explicit DecodeAudioOpV2(OpKernelConstruction* context) : OpKernel(context) {
    string stream;
    if (context->GetAttr("stream", &stream).ok()) {
      stream_ = stream;
    }
  }

  void Compute(OpKernelContext* context) override {
    OP_REQUIRES(
        context, context->num_inputs() == 4,
        errors::InvalidArgument("DecodeAudio requires exactly four inputs."));

    const Tensor& contents_tensor = context->input(0);
    const Tensor& file_format_tensor = context->input(1);
    const Tensor& samples_per_second_tensor = context->input(2);
    const Tensor& channel_count_tensor = context->input(3);

    OP_REQUIRES(context, TensorShapeUtils::IsScalar(contents_tensor.shape()),
                errors::InvalidArgument(
                    "contents must be a rank-0 tensor but got shape ",
                    contents_tensor.shape().DebugString()));
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
                TensorShapeUtils::IsScalar(channel_count_tensor.shape()),
                errors::InvalidArgument(
                    "channel_count must be a rank-0 tensor but got shape ",
                    channel_count_tensor.shape().DebugString()));

    const tensorflow::StringPiece contents = contents_tensor.scalar<string>()();
    const string file_format =
        str_util::Lowercase(file_format_tensor.scalar<string>()());
    const int32 samples_per_second =
        samples_per_second_tensor.scalar<int32>()();
    const int32 channel_count = channel_count_tensor.scalar<int32>()();

    const std::set<string> valid_file_formats(
        kValidFileFormats, kValidFileFormats + TF_ARRAYSIZE(kValidFileFormats));
    OP_REQUIRES(
        context, valid_file_formats.count(file_format) == 1,
        errors::InvalidArgument("file_format must be one of {",
                                str_util::Join(valid_file_formats, ", "),
                                "}, but was: \"", file_format, "\""));
    OP_REQUIRES(context, samples_per_second > 0,
                errors::InvalidArgument(
                    "samples_per_second must be positive, but got: ",
                    samples_per_second));
    OP_REQUIRES(
        context, channel_count > 0,
        errors::InvalidArgument("channel_count must be positive, but got: ",
                                channel_count));

    Decode(context, contents, file_format, samples_per_second, channel_count,
           stream_);
  }

 private:
  string stream_;
};

REGISTER_KERNEL_BUILDER(Name("DecodeAudioV2").Device(DEVICE_CPU),
                        DecodeAudioOpV2);

REGISTER_OP("DecodeAudioV2")
    .Input("contents: string")
    .Input("file_format: string")
    .Input("samples_per_second: int32")
    .Input("channel_count: int32")
    .Output("sampled_audio: float")
    .Attr("stream: string = ''")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      const Tensor* channels_tensor = c->input_tensor(3);
      if (channels_tensor == nullptr) {
        c->set_output(0, c->Matrix(c->UnknownDim(), c->UnknownDim()));
        return Status::OK();
      }
      const int32 channels = channels_tensor->scalar<int32>()();
      if (channels <= 0) {
        return errors::InvalidArgument(
            "channel_count must be positive, but got: ", channels);
      }
      c->set_output(0, c->Matrix(c->UnknownDim(), channels));
      return Status::OK();
    })
    .Doc(R"doc(
Processes the contents of an audio file into a tensor using FFmpeg to decode
the file.

One row of the tensor is created for each channel in the audio file. Each
channel contains audio samples starting at the beginning of the audio and
having `1/samples_per_second` time between them. If the `channel_count` is
different from the contents of the file, channels will be merged or created.

contents: The binary audio file contents, as a string or rank-0 string
    tensor.
file_format: A string or rank-0 string tensor describing the audio file
    format. This must be one of: "mp3", "mp4", "ogg", "wav".
samples_per_second: The number of samples per second that the audio
    should have, as an `int` or rank-0 `int32` tensor. This value must
    be positive.
channel_count: The number of channels of audio to read, as an int rank-0
    int32 tensor. Must be a positive integer.
sampled_audio: A rank-2 tensor containing all tracks of the audio.
    Dimension 0 is time and dimension 1 is the channel. If ffmpeg fails
    to decode the audio then an empty tensor will be returned.
)doc");

/*
 * Deprecated in favor of DecodeAudioOpV2.
 */
class DecodeAudioOp : public OpKernel {
 public:
  explicit DecodeAudioOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("file_format", &file_format_));
    file_format_ = str_util::Lowercase(file_format_);
    const std::set<string> valid_file_formats(
        kValidFileFormats, kValidFileFormats + TF_ARRAYSIZE(kValidFileFormats));
    OP_REQUIRES(
        context, valid_file_formats.count(file_format_) == 1,
        errors::InvalidArgument("file_format must be one of {",
                                str_util::Join(valid_file_formats, ", "),
                                "}, but was: \"", file_format_, "\""));

    OP_REQUIRES_OK(context, context->GetAttr("channel_count", &channel_count_));
    OP_REQUIRES(context, channel_count_ > 0,
                errors::InvalidArgument("channel_count must be > 0."));
  }

  void Compute(OpKernelContext* context) override {
    // Get and verify the input data.
    OP_REQUIRES(
        context, context->num_inputs() == 1,
        errors::InvalidArgument("DecodeAudio requires exactly one input."));
    const Tensor& contents = context->input(0);
    OP_REQUIRES(
        context, TensorShapeUtils::IsScalar(contents.shape()),
        errors::InvalidArgument("contents must be scalar but got shape ",
                                contents.shape().DebugString()));

    const tensorflow::StringPiece file_contents = contents.scalar<string>()();
    Decode(context, file_contents, file_format_, samples_per_second_,
           channel_count_, "");
  }

 private:
  string file_format_;
  int32 samples_per_second_;
  int32 channel_count_;
};

REGISTER_KERNEL_BUILDER(Name("DecodeAudio").Device(DEVICE_CPU), DecodeAudioOp);

REGISTER_OP("DecodeAudio")
    .Input("contents: string")
    .Output("sampled_audio: float")
    .Attr("file_format: string")
    .Attr("samples_per_second: int")
    .Attr("channel_count: int")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      int64 channels;
      if (c->GetAttr("channel_count", &channels).ok()) {
        c->set_output(0, c->Matrix(c->UnknownDim(), channels));
      } else {
        c->set_output(0, c->Matrix(c->UnknownDim(), c->UnknownDim()));
      }
      return Status::OK();
    })
    .Doc(R"doc(
Processes the contents of an audio file into a tensor using FFmpeg to decode
the file.

One row of the tensor is created for each channel in the audio file. Each
channel contains audio samples starting at the beginning of the audio and
having `1/samples_per_second` time between them. If the `channel_count` is
different from the contents of the file, channels will be merged or created.

contents: The binary audio file contents.
sampled_audio: A rank 2 tensor containing all tracks of the audio. Dimension 0
    is time and dimension 1 is the channel. If ffmpeg fails to decode the audio
    then an empty tensor will be returned.
file_format: A string describing the audio file format. This can be "mp3", "mp4", "ogg", or "wav".
samples_per_second: The number of samples per second that the audio should have.
channel_count: The number of channels of audio to read.
)doc");

}  // namespace ffmpeg
}  // namespace tensorflow

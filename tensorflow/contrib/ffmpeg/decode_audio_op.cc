// Copyright 2016 Google Inc. All Rights Reserved.
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

#include "tensorflow/contrib/ffmpeg/kernels/ffmpeg_lib.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/env.h"

namespace tensorflow {
namespace ffmpeg {
namespace {

// The complete set of audio file formats that are supported by the op. These
// strings are defined by FFmpeg and documented here:
// https://www.ffmpeg.org/ffmpeg-formats.html
const char* kValidFileFormats[] = {"mp3", "ogg", "wav"};

// Writes binary data to a file.
Status WriteFile(const string& filename, tensorflow::StringPiece contents) {
  Env& env = *Env::Default();
  WritableFile* file = nullptr;
  TF_RETURN_IF_ERROR(env.NewWritableFile(filename, &file));
  std::unique_ptr<WritableFile> file_deleter(file);
  TF_RETURN_IF_ERROR(file->Append(contents));
  TF_RETURN_IF_ERROR(file->Close());
  return Status::OK();
}

// Cleans up a file on destruction.
class FileDeleter {
 public:
  explicit FileDeleter(const string& filename) : filename_(filename) {}
  ~FileDeleter() {
    Env& env = *Env::Default();
    env.DeleteFile(filename_);
  }

 private:
  const string filename_;
};

}  // namespace

class DecodeAudioOp : public OpKernel {
 public:
  explicit DecodeAudioOp(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("file_format", &file_format_));
    file_format_ = str_util::Lowercase(file_format_);
    const std::set<string> valid_file_formats(
        kValidFileFormats,
        kValidFileFormats + TF_ARRAYSIZE(kValidFileFormats));
    OP_REQUIRES(context, valid_file_formats.count(file_format_) == 1,
                errors::InvalidArgument(
                    "file_format arg must be in {",
                    str_util::Join(valid_file_formats, ", "), "}."));

    OP_REQUIRES_OK(
        context, context->GetAttr("samples_per_second", &samples_per_second_));
    OP_REQUIRES(context, samples_per_second_ > 0,
                errors::InvalidArgument("samples_per_second must be > 0."));

    OP_REQUIRES_OK(
        context, context->GetAttr("channel_count", &channel_count_));
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

    // Write the input data to a temp file.
    const tensorflow::StringPiece file_contents = contents.scalar<string>()();
    const string input_filename = GetTempFilename(file_format_);
    OP_REQUIRES_OK(context, WriteFile(input_filename, file_contents));
    FileDeleter deleter(input_filename);

    // Run FFmpeg on the data and verify results.
    std::vector<float> output_samples;
    Status result =
        ffmpeg::ReadAudioFile(input_filename, file_format_, samples_per_second_,
                              channel_count_, &output_samples);
    if (result.code() == error::Code::NOT_FOUND) {
      OP_REQUIRES(
          context, result.ok(),
          errors::Unavailable("FFmpeg must be installed to run this op. FFmpeg "
                              "can be found at http://www.ffmpeg.org."));
    } else {
      OP_REQUIRES_OK(context, result);
    }
    OP_REQUIRES(
        context, !output_samples.empty(),
        errors::Unknown("No output created by FFmpeg."));
    OP_REQUIRES(
        context, output_samples.size() % channel_count_ == 0,
        errors::Unknown("FFmpeg created non-integer number of audio frames."));

    // Copy the output data to the output Tensor.
    Tensor* output = nullptr;
    const int64 frame_count = output_samples.size() / channel_count_;
    OP_REQUIRES_OK(
        context, context->allocate_output(
            0, TensorShape({frame_count, channel_count_}), &output));
    auto matrix = output->tensor<float, 2>();
    for (int32 frame = 0; frame < frame_count; ++frame) {
      for (int32 channel = 0; channel < channel_count_; ++channel) {
        matrix(frame, channel) =
            output_samples[frame * channel_count_ + channel];
      }
    }
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
    .Doc(R"doc(
Processes the contents of an audio file into a tensor using FFmpeg to decode
the file.

One row of the tensor is created for each channel in the audio file. Each
channel contains audio samples starting at the beginning of the audio and
having `1/samples_per_second` time between them. If the `channel_count` is
different from the contents of the file, channels will be merged or created.

contents: The binary audio file contents.
sampled_audio: A rank 2 tensor containing all tracks of the audio. Dimension 0
    is time and dimension 1 is the channel.
file_format: A string describing the audio file format. This can be "wav" or
    "mp3".
samples_per_second: The number of samples per second that the audio should have.
channel_count: The number of channels of audio to read.
)doc");

}  // namespace ffmpeg
}  // namespace tensorflow

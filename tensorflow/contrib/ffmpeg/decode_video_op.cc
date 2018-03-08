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

class DecodeVideoOp : public OpKernel {
 public:
  explicit DecodeVideoOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    OP_REQUIRES(
        context, context->num_inputs() == 1,
        errors::InvalidArgument("DecodeVideo requires exactly 1 input."));
    const Tensor& contents_tensor = context->input(0);

    OP_REQUIRES(context, TensorShapeUtils::IsScalar(contents_tensor.shape()),
                errors::InvalidArgument(
                    "contents must be a rank-0 tensor but got shape ",
                    contents_tensor.shape().DebugString()));
    const tensorflow::StringPiece contents = contents_tensor.scalar<string>()();

    // Write the input data to a temp file.
    string extension;
    const string temp_filename = io::GetTempFilename(extension);
    OP_REQUIRES_OK(context, WriteFile(temp_filename, contents));
    FileDeleter deleter(temp_filename);

    uint32 width = 0;
    uint32 height = 0;
    uint32 frames = 0;

    // Run FFmpeg on the data and verify results.
    std::vector<uint8> output_data;
    const Status result = ffmpeg::ReadVideoFile(temp_filename, &output_data,
                                                &width, &height, &frames);
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
    OP_REQUIRES(context, !output_data.empty(),
                errors::Unknown("No output created by FFmpeg."));
    OP_REQUIRES(
        context, output_data.size() == (frames * height * width * 3),
        errors::Unknown("Output created by FFmpeg [", output_data.size(),
                        "] does not match description [", frames, ", ", height,
                        ", ", width, ", 3]"));
    Tensor* output = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(
                       0, TensorShape({frames, height, width, 3}), &output));
    auto output_flat = output->flat<uint8>();
    std::copy_n(output_data.begin(), output_data.size(), &output_flat(0));
  }
};

REGISTER_KERNEL_BUILDER(Name("DecodeVideo").Device(DEVICE_CPU), DecodeVideoOp);

REGISTER_OP("DecodeVideo")
    .Input("contents: string")
    .Output("output: uint8")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      c->set_output(0, c->UnknownShapeOfRank(4));
      return Status::OK();
    })
    .Doc(R"doc(
Processes the contents of an video file into a tensor using FFmpeg to decode
the file.

contents: The binary contents of the video file to decode. This is a
    scalar.
output: A rank-4 `Tensor` that has `[frames, height, width, 3]` RGB as output.
)doc");

}  // namespace ffmpeg
}  // namespace tensorflow

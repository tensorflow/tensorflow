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

// See docs in ../ops/image_ops.cc

#include <memory>
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/png/png_io.h"
#include "tensorflow/core/platform/logging.h"

namespace tensorflow {

// Decode the contents of a PNG file
class DecodePngOp : public OpKernel {
 public:
  explicit DecodePngOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("channels", &channels_));
    OP_REQUIRES(context, channels_ == 0 || channels_ == 1 || channels_ == 3 ||
                             channels_ == 4,
                errors::InvalidArgument("channels must be 0, 1, 3, or 4, got ",
                                        channels_));

    DataType dt;
    OP_REQUIRES_OK(context, context->GetAttr("dtype", &dt));
    OP_REQUIRES(
        context, dt == DataType::DT_UINT8 || dt == DataType::DT_UINT16,
        errors::InvalidArgument("Type must be UINT8 or UINT16, got ", dt));
    if (dt == DataType::DT_UINT8) {
      desired_channel_bits_ = 8;
    } else {
      desired_channel_bits_ = 16;
    }
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& contents = context->input(0);
    OP_REQUIRES(context, TensorShapeUtils::IsScalar(contents.shape()),
                errors::InvalidArgument("contents must be scalar, got shape ",
                                        contents.shape().DebugString()));

    // Start decoding image to get shape details
    const StringPiece data = contents.scalar<string>()();
    png::DecodeContext decode;
    OP_REQUIRES(
        context,
        png::CommonInitDecode(data, channels_, desired_channel_bits_, &decode),
        errors::InvalidArgument("Invalid PNG header, data size ", data.size()));

    // Verify that width and height are not too large:
    // - verify width and height don't overflow int.
    // - width can later be multiplied by channels_ and sizeof(uint16), so
    //   verify single dimension is not too large.
    // - verify when width and height are multiplied together, there are a few
    //   bits to spare as well.
    const int width = static_cast<int>(decode.width);
    const int height = static_cast<int>(decode.height);
    const int64 total_size =
        static_cast<int64>(width) * static_cast<int64>(height);
    if (width != static_cast<int64>(decode.width) || width <= 0 ||
        width >= (1LL << 27) || height != static_cast<int64>(decode.height) ||
        height <= 0 || height >= (1LL << 27) || total_size >= (1LL << 29)) {
      png::CommonFreeDecode(&decode);
      OP_REQUIRES(context, false,
                  errors::InvalidArgument("PNG size too large for int: ",
                                          decode.width, " by ", decode.height));
    }

    // Allocate tensor
    Tensor* output = nullptr;
    const auto status = context->allocate_output(
        0, TensorShape({height, width, decode.channels}), &output);
    if (!status.ok()) png::CommonFreeDecode(&decode);
    OP_REQUIRES_OK(context, status);

    if (desired_channel_bits_ == 8) {
      // Finish decoding image
      OP_REQUIRES(
          context,
          png::CommonFinishDecode(
              reinterpret_cast<png_bytep>(output->flat<uint8>().data()),
              decode.channels * width * sizeof(uint8), &decode),
          errors::InvalidArgument("Invalid PNG data, size ", data.size()));
    } else {
      // Finish decoding image
      OP_REQUIRES(
          context,
          png::CommonFinishDecode(
              reinterpret_cast<png_bytep>(output->flat<uint16>().data()),
              decode.channels * width * sizeof(uint16), &decode),
          errors::InvalidArgument("Invalid PNG data, size ", data.size()));
    }
  }

 private:
  int channels_;
  int desired_channel_bits_;
};
REGISTER_KERNEL_BUILDER(Name("DecodePng").Device(DEVICE_CPU), DecodePngOp);

}  // namespace tensorflow

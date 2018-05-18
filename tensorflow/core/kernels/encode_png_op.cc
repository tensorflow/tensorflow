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
#include "tensorflow/core/kernels/bounds_check.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/png/png_io.h"
#include "tensorflow/core/platform/logging.h"

namespace tensorflow {

// Encode an image to a PNG stream
class EncodePngOp : public OpKernel {
 public:
  explicit EncodePngOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("compression", &compression_));
    OP_REQUIRES(context, -1 <= compression_ && compression_ <= 9,
                errors::InvalidArgument("compression should be in [-1,9], got ",
                                        compression_));

    DataType dt = context->input_type(0);
    OP_REQUIRES(context, dt == DataType::DT_UINT8 || dt == DataType::DT_UINT16,
                errors::InvalidArgument(
                    "image must have type uint8 or uint16, got ", dt));

    if (dt == DataType::DT_UINT8) {
      desired_channel_bits_ = 8;
    } else {
      desired_channel_bits_ = 16;
    }
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& image = context->input(0);
    OP_REQUIRES(context, image.dims() == 3,
                errors::InvalidArgument("image must be 3-dimensional",
                                        image.shape().DebugString()));
    OP_REQUIRES(
        context,
        FastBoundsCheck(image.NumElements(), std::numeric_limits<int32>::max()),
        errors::InvalidArgument("image cannot have >= int32 max elements"));
    const int32 height = static_cast<int32>(image.dim_size(0));
    const int32 width = static_cast<int32>(image.dim_size(1));
    const int32 channels = static_cast<int32>(image.dim_size(2));

    // In some cases, we pass width*channels*2 to png.
    const int32 max_row_width = std::numeric_limits<int32>::max() / 2;

    OP_REQUIRES(context, FastBoundsCheck(width * channels, max_row_width),
                errors::InvalidArgument("image too wide to encode"));

    OP_REQUIRES(context, channels >= 1 && channels <= 4,
                errors::InvalidArgument(
                    "image must have 1, 2, 3, or 4 channels, got ", channels));

    // Encode image to png string
    Tensor* output = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, TensorShape({}), &output));
    if (desired_channel_bits_ == 8) {
      OP_REQUIRES(context,
                  png::WriteImageToBuffer(image.flat<uint8>().data(), width,
                                          height, width * channels, channels,
                                          desired_channel_bits_, compression_,
                                          &output->scalar<string>()(), nullptr),
                  errors::Internal("PNG encoding failed"));
    } else {
      OP_REQUIRES(context,
                  png::WriteImageToBuffer(
                      image.flat<uint16>().data(), width, height,
                      width * channels * 2, channels, desired_channel_bits_,
                      compression_, &output->scalar<string>()(), nullptr),
                  errors::Internal("PNG encoding failed"));
    }
  }

 private:
  int compression_;
  int desired_channel_bits_;
};
REGISTER_KERNEL_BUILDER(Name("EncodePng").Device(DEVICE_CPU), EncodePngOp);

}  // namespace tensorflow

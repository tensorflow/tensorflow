/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/core/framework/bounds_check.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/jpeg/jpeg_mem.h"
#include "tensorflow/core/platform/logging.h"

namespace tensorflow {

// Extract the shape of a JPEG image.
template <typename T>
class ExtractJpegShapeOp : public OpKernel {
 public:
  explicit ExtractJpegShapeOp(OpKernelConstruction* context)
      : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Get input content.
    const Tensor& contents = context->input(0);
    OP_REQUIRES(context, TensorShapeUtils::IsScalar(contents.shape()),
                errors::InvalidArgument("contents must be scalar, got shape ",
                                        contents.shape().DebugString()));
    const StringPiece input = contents.scalar<tstring>()();
    OP_REQUIRES(context, input.size() <= std::numeric_limits<int>::max(),
                errors::InvalidArgument("JPEG contents are too large for int: ",
                                        input.size()));

    // Call GetImageInfo to get image shape.
    int width, height, components;
    OP_REQUIRES(
        context,
        jpeg::GetImageInfo(input.data(), input.size(), &width, &height,
                           &components),
        errors::InvalidArgument("Invalid JPEG data, size ", input.size()));
    // Allocate tensor and set shape size.
    Tensor* image_shape = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, TensorShape({3}), &image_shape));
    auto image_shape_data = image_shape->tensor<T, 1>();
    image_shape_data(0) = height;
    image_shape_data(1) = width;
    image_shape_data(2) = components;
  }
};

#define REGISTER_KERNELS(type)                                      \
  REGISTER_KERNEL_BUILDER(Name("ExtractJpegShape")                  \
                              .Device(DEVICE_CPU)                   \
                              .TypeConstraint<type>("output_type"), \
                          ExtractJpegShapeOp<type>)

TF_CALL_int32(REGISTER_KERNELS);
TF_CALL_int64(REGISTER_KERNELS);
#undef REGISTER_KERNELS

}  // namespace tensorflow

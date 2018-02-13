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

// See docs in ../ops/image_ops.cc.

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/random/simple_philox.h"
#include "tensorflow/core/util/guarded_philox_random.h"

namespace tensorflow {

template <typename T>
class RandomCropOp : public OpKernel {
 public:
  explicit RandomCropOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, generator_.Init(context));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& input = context->input(0);
    OP_REQUIRES(context, input.dims() == 3,
                errors::InvalidArgument("input must be 3-dimensional",
                                        input.shape().DebugString()));
    const Tensor& shape_t = context->input(1);
    OP_REQUIRES(context, shape_t.dims() == 1,
                errors::InvalidArgument("shape_t must be 1-dimensional",
                                        shape_t.shape().DebugString()));
    OP_REQUIRES(context, shape_t.NumElements() == 2,
                errors::InvalidArgument("shape_t must have two elements",
                                        shape_t.shape().DebugString()));

    auto shape_vec = shape_t.vec<int64>();
    const int32 target_height = shape_vec(0);
    const int32 target_width = shape_vec(1);

    const int32 height = input.dim_size(0);
    const int32 width = input.dim_size(1);
    const int32 channels = input.dim_size(2);

    // Initialize shape to the batch size of the input, then add
    // the rest of the dimensions
    Tensor* output = nullptr;
    const auto output_shape =
        TensorShape({target_height, target_width, channels});
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output));

    // If the target size matches the actual size, then do nothing.
    if ((target_height == height) && (target_width == width)) {
      *output = context->input(0);
    }

    // TODO(shlens): Implement edge case to guarantee output size dimensions.
    // Edge case. The target dimensions are larger then the image, so
    // zero-pad the image. This guarantees that the image will *always*
    // be [target_height, target_width] in size.
    OP_REQUIRES(context, width >= target_width,
                errors::FailedPrecondition(
                    "width must be >= target_width: width = ", width,
                    ", target_width = ", target_width));
    OP_REQUIRES(context, height >= target_height,
                errors::FailedPrecondition(
                    "height must be >= target_height: height = ", height,
                    ", target_height = ", target_height));

    int32 offset_height = 0;
    int32 offset_width = 0;

    auto local_gen = generator_.ReserveSamples32(2);
    random::SimplePhilox random(&local_gen);

    if (width > target_width) {
      offset_width = random.Rand32() % (width - target_width + 1);
    }
    if (height > target_height) {
      offset_height = random.Rand32() % (height - target_height + 1);
    }

    // TODO(shlens): Do this more efficiently with memcpy once padding is
    // available for smaller images.
    typename TTypes<T, 3>::ConstTensor input_data = input.tensor<T, 3>();
    typename TTypes<T, 3>::Tensor output_data = output->tensor<T, 3>();

    for (int y = 0; y < target_height; ++y) {
      for (int x = 0; x < target_width; ++x) {
        for (int c = 0; c < channels; ++c) {
          output_data(y, x, c) =
              input_data(y + offset_height, x + offset_width, c);
        }
      }
    }
  }

 private:
  GuardedPhiloxRandom generator_;
};

#define REGISTER_KERNELS(type)                                         \
  REGISTER_KERNEL_BUILDER(                                             \
      Name("RandomCrop").Device(DEVICE_CPU).TypeConstraint<type>("T"), \
      RandomCropOp<type>)

TF_CALL_REAL_NUMBER_TYPES(REGISTER_KERNELS);
#undef REGISTER_KERNELS

}  // namespace tensorflow

/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

// See docs in ../ops/array_ops.cc.

#define EIGEN_USE_THREADS

#include <memory>
#include <string>
#include <utility>

#include "tensorflow/core/kernels/spacetodepth_op.h"

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

template <typename Device, typename T>
class SpaceToDepthOp : public OpKernel {
 public:
  explicit SpaceToDepthOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("block_size", &block_size_));

    OP_REQUIRES(
        context, block_size_ > 1,
        errors::InvalidArgument("Block size should be > 1: ", block_size_));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& input = context->input(0);
    const int dims = input.dims();

    // Check on the input dimensions first.
    // The input is presumed to be [batch, height, width, depth]
    static const int kRequiredDims = 4;
    OP_REQUIRES(context, kRequiredDims == dims,
                errors::InvalidArgument("Input rank should be: ", kRequiredDims,
                                        "instead of: ", dims));

    const int batch_size = input.dim_size(0);
    const int height = input.dim_size(1);
    const int width = input.dim_size(2);
    const int input_depth = input.dim_size(3);

    // Both width and height must be divisible by block_size.
    OP_REQUIRES(
        context, (width % block_size_) == 0 && (height % block_size_) == 0,
        errors::InvalidArgument("Image width ", width, " and height ", height,
                                "should be divisible by block_size: ",
                                block_size_));

    const int block_size_sq = block_size_ * block_size_;

    // The 'spatial' block of size block_size_ X block_size_ will be moved
    // to depth.
    const int output_depth = input_depth * block_size_sq;
    const int output_width = width / block_size_;
    const int output_height = height / block_size_;

    // Allocate output tensor.
    Tensor* outputs_tensor = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(
                                0, TensorShape({batch_size, output_height,
                                                output_width, output_depth}),
                                &outputs_tensor));

    auto Toutput = outputs_tensor->tensor<T, 4>();
    auto Tinput = input.tensor<T, 4>();

    functor::SpaceToDepthOpFunctor<Device, T> functor;
    functor(context->eigen_device<Device>(), Tinput, block_size_, Toutput);
  };

 private:
  int block_size_;
};

// Partial specialization of SpaceToDepthOpFunctor for a CPUDevice.
namespace functor {
template <typename T>
struct SpaceToDepthOpFunctor<CPUDevice, T> {
  void operator()(const CPUDevice& d, typename TTypes<T, 4>::ConstTensor input,
                  int block_size, typename TTypes<T, 4>::Tensor output) {
    const int batch_size = output.dimension(0);
    const int input_height = input.dimension(1);
    const int input_width = input.dimension(2);
    const int input_depth = input.dimension(3);

    for (int b = 0; b < batch_size; ++b) {
      for (int h = 0; h < input_height; ++h) {
        const int out_h = h / block_size;
        const int offset_h = (h % block_size);
        for (int w = 0; w < input_width; ++w) {
          const int out_w = w / block_size;
          const int offset_w = (w % block_size);
          const int offset_d = (offset_h * block_size + offset_w) * input_depth;
          for (int d = 0; d < input_depth; ++d) {
            const int out_d = d + offset_d;
            output(b, out_h, out_w, out_d) = input(b, h, w, d);
          }
        }
      }
    }
  }
};
}  // namespace functor

#define REGISTER(type)                                                   \
  REGISTER_KERNEL_BUILDER(                                               \
      Name("SpaceToDepth").Device(DEVICE_CPU).TypeConstraint<type>("T"), \
      SpaceToDepthOp<CPUDevice, type>);

TF_CALL_ALL_TYPES(REGISTER);
#undef REGISTER

#if GOOGLE_CUDA
REGISTER_KERNEL_BUILDER(
    Name("SpaceToDepth").Device(DEVICE_GPU).TypeConstraint<float>("T"),
    SpaceToDepthOp<GPUDevice, float>);
#endif  // GOOGLE_CUDA

}  // end namespace tensorflow

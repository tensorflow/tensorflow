/* Copyright 2016 Google Inc. All Rights Reserved.

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

#include "tensorflow/core/kernels/depthtospace_op.h"

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
class DepthToSpaceOp : public OpKernel {
 public:
  explicit DepthToSpaceOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("block_size", &block_size_));

    OP_REQUIRES(
        context, block_size_ > 1,
        errors::InvalidArgument("Block size should be > 1: ", block_size_));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& input = context->input(0);

    // Check on the input dimensions first.
    // The input is presumed to be [batch, height, width, depth]
    const int dims = input.dims();
    static const int kRequiredDims = 4;
    OP_REQUIRES(context, kRequiredDims == dims,
                errors::InvalidArgument("Input rank should be: ", kRequiredDims,
                                        "instead of: ", dims));

    const int batch_size = input.dim_size(0);
    const int input_height = input.dim_size(1);
    const int input_width = input.dim_size(2);
    const int input_depth = input.dim_size(3);

    const int block_size_sq = block_size_ * block_size_;

    // The depth must be divisible by block_size_ * block_size_
    OP_REQUIRES(
        context, input_depth % block_size_sq == 0,
        errors::InvalidArgument("Input depth dimension ", input_depth,
                                "should be divisible by: ", block_size_sq));

    const int output_depth = input_depth / block_size_sq;
    const int output_width = input_width * block_size_;
    const int output_height = input_height * block_size_;

    // Allocate output tensor.
    Tensor* output = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(
                                0, TensorShape({batch_size, output_height,
                                                output_width, output_depth}),
                                &output));

    typename TTypes<T, 4>::ConstTensor Tinput = input.tensor<T, 4>();
    typename TTypes<T, 4>::Tensor Toutput = output->tensor<T, 4>();

    functor::DepthToSpaceOpFunctor<Device, T> functor;
    functor(context->eigen_device<Device>(), Tinput, block_size_, Toutput);
  };

 private:
  int block_size_;
};

// Partial specialization of DepthToSpaceOpFunctor for a CPUDevice.
namespace functor {
template <typename T>
struct DepthToSpaceOpFunctor<CPUDevice, T> {
  void operator()(const CPUDevice& d, typename TTypes<T, 4>::ConstTensor input,
                  int block_size, typename TTypes<T, 4>::Tensor output) {
    const int batch_size = output.dimension(0);
    const int output_height = output.dimension(1);
    const int output_width = output.dimension(2);
    const int output_depth = output.dimension(3);

    for (int b = 0; b < batch_size; ++b) {
      for (int h = 0; h < output_height; ++h) {
        const int in_h = h / block_size;
        const int offset_h = (h % block_size);
        for (int w = 0; w < output_width; ++w) {
          const int in_w = w / block_size;
          const int offset_w = (w % block_size);
          const int offset_d =
              (offset_h * block_size + offset_w) * output_depth;
          for (int d = 0; d < output_depth; ++d) {
            const int in_d = d + offset_d;
            output(b, h, w, d) = input(b, in_h, in_w, in_d);
          }
        }
      }
    }
  }
};
}  // namespace functor

#define REGISTER(type)                                                   \
  REGISTER_KERNEL_BUILDER(                                               \
      Name("DepthToSpace").Device(DEVICE_CPU).TypeConstraint<type>("T"), \
      DepthToSpaceOp<CPUDevice, type>);

TF_CALL_ALL_TYPES(REGISTER);
#undef REGISTER

#if GOOGLE_CUDA
REGISTER_KERNEL_BUILDER(
    Name("DepthToSpace").Device(DEVICE_GPU).TypeConstraint<float>("T"),
    DepthToSpaceOp<GPUDevice, float>);
#endif  // GOOGLE_CUDA

}  // end namespace tensorflow

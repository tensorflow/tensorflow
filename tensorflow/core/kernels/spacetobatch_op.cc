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

#include "tensorflow/core/kernels/spacetobatch_op.h"

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
class SpaceToBatchOp : public OpKernel {
 public:
  explicit SpaceToBatchOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("block_size", &block_size_));
    OP_REQUIRES(
        context, block_size_ > 1,
        errors::InvalidArgument("Block size should be > 1: ", block_size_));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& in0 = context->input(0);
    const Tensor& in1 = context->input(1);
    const int dims = in0.dims();

    // Check on the input dimensions first.
    // The input is presumed to be [batch, height, width, depth]
    static const int kRequiredDims = 4;
    OP_REQUIRES(context, kRequiredDims == dims,
                errors::InvalidArgument("Input rank should be: ", kRequiredDims,
                                        "instead of: ", dims));

    // The paddings is presumed to be [2, 2].
    OP_REQUIRES(
        context,
        TensorShapeUtils::IsMatrix(in1.shape()) &&
        in1.dim_size(0) == 2 && in1.dim_size(1) == 2,
        errors::InvalidArgument("paddings must be a 2 x 2 matrix: ",
                                in1.shape().DebugString()));
    TTypes<int32>::ConstMatrix paddings = in1.matrix<int32>();
    OP_REQUIRES(context,
                paddings(0, 0) >= 0 && paddings(0, 1) >= 0 &&
                paddings(1, 0) >= 0 && paddings(1, 1) >= 0,
                errors::InvalidArgument("Paddings must be non-negative"));

    // Compute the shape of the zero-padded input tensor.
    TensorShape padded_shape;
    padded_shape.AddDim(in0.dim_size(0));
    padded_shape.AddDim(paddings(0, 0) + in0.dim_size(1) + paddings(0, 1));
    padded_shape.AddDim(paddings(1, 0) + in0.dim_size(2) + paddings(1, 1));
    padded_shape.AddDim(in0.dim_size(3));

    const int batch = padded_shape.dim_size(0);
    const int height = padded_shape.dim_size(1);
    const int width = padded_shape.dim_size(2);
    const int depth = padded_shape.dim_size(3);

    // Both height and width must be divisible by block_size.
    OP_REQUIRES(
        context, height % block_size_ == 0 && width % block_size_ == 0,
        errors::InvalidArgument("Image height ", height, " and width ", width,
                                "should be divisible by block_size: ",
                                block_size_));

    const int block_size_sq = block_size_ * block_size_;

    // The 'spatial' block of size block_size_ X block_size_ will be moved
    // to batch.
    const int output_batch = batch * block_size_sq;
    const int output_height = height / block_size_;
    const int output_width = width / block_size_;

    // Allocate output tensor.
    Tensor* outputs_tensor = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(
                                0, TensorShape({output_batch, output_height,
                                                output_width, depth}),
                                &outputs_tensor));

    typename TTypes<T, 4>::ConstTensor Tinput = in0.tensor<T, 4>();
    typename TTypes<T, 4>::Tensor Toutput = outputs_tensor->tensor<T, 4>();

    functor::SpaceToBatchOpFunctor<Device, T> functor;
    functor(context->eigen_device<Device>(),
            Tinput, paddings, block_size_, Toutput);
  };

 private:
  int block_size_;
};

// Partial specialization of SpaceToBatchOpFunctor for a CPUDevice.
namespace functor {
template <typename T>
struct SpaceToBatchOpFunctor<CPUDevice, T> {
  void operator()(const CPUDevice& d, typename TTypes<T, 4>::ConstTensor input,
                  typename TTypes<int32>::ConstMatrix paddings,
                  int block_size, typename TTypes<T, 4>::Tensor output) {
    const int output_batch = output.dimension(0);
    const int output_height = output.dimension(1);
    const int output_width = output.dimension(2);
    const int depth = output.dimension(3);

    const int input_batch = input.dimension(0);
    const int input_height = input.dimension(1);
    const int input_width = input.dimension(2);

    const int pad_top = paddings(0, 0);
    const int pad_left = paddings(1, 0);

    for (int out_b = 0; out_b < output_batch; ++out_b) {
      // out_b = (offset_h * block_size + offset_w) * input_batch + in_b
      const int in_b = out_b % input_batch;
      const int offset_w = (out_b / input_batch) % block_size;
      const int offset_h = (out_b / input_batch) / block_size;
      for (int out_h = 0; out_h < output_height; ++out_h) {
        const int in_h = out_h * block_size + offset_h - pad_top;
        for (int out_w = 0; out_w < output_width; ++out_w) {
          const int in_w = out_w * block_size + offset_w - pad_left;
          if (in_h >= 0 && in_w >= 0 &&
              in_h < input_height && in_w < input_width) {
            for (int d = 0; d < depth; ++d) {
              output(out_b, out_h, out_w, d) = input(in_b, in_h, in_w, d);
            }
          } else {
            for (int d = 0; d < depth; ++d) {
              output(out_b, out_h, out_w, d) = static_cast<T>(0);
            }
          }
        }
      }
    }
  }
};
}  // namespace functor

#define REGISTER(T)                                                     \
  REGISTER_KERNEL_BUILDER(Name("SpaceToBatch")                          \
                              .Device(DEVICE_CPU)                       \
                              .TypeConstraint<T>("T")                   \
                              .HostMemory("paddings"),                  \
                          SpaceToBatchOp<CPUDevice, T>);

TF_CALL_REAL_NUMBER_TYPES(REGISTER);
#undef REGISTER

#if GOOGLE_CUDA
#define REGISTER(T)                                                     \
  REGISTER_KERNEL_BUILDER(Name("SpaceToBatch")                          \
                              .Device(DEVICE_GPU)                       \
                              .TypeConstraint<T>("T")                   \
                              .HostMemory("paddings"),                  \
                          SpaceToBatchOp<GPUDevice, T>);

TF_CALL_GPU_NUMBER_TYPES_NO_HALF(REGISTER);
#undef REGISTER
#endif  // GOOGLE_CUDA

}  // end namespace tensorflow

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

#include "tensorflow/core/kernels/batchtospace_op.h"

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
class BatchToSpaceOp : public OpKernel {
 public:
  explicit BatchToSpaceOp(OpKernelConstruction* context) : OpKernel(context) {
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

    // The crops is presumed to be [2, 2] and contain non-negative values.
    OP_REQUIRES(
        context,
        TensorShapeUtils::IsMatrix(in1.shape()) &&
        in1.dim_size(0) == 2 && in1.dim_size(1) == 2,
        errors::InvalidArgument("crops must be a 2 x 2 matrix: ",
                                in1.shape().DebugString()));
    TTypes<int32>::ConstMatrix crops = in1.matrix<int32>();
    OP_REQUIRES(context,
                crops(0, 0) >= 0 && crops(0, 1) >= 0 &&
                crops(1, 0) >= 0 && crops(1, 1) >= 0,
                errors::InvalidArgument("Crops must be non-negative"));

    const int input_batch = in0.dim_size(0);
    const int input_height = in0.dim_size(1);
    const int input_width = in0.dim_size(2);
    const int depth = in0.dim_size(3);

    const int block_size_sq = block_size_ * block_size_;

    // The batch must be divisible by block_size_ * block_size_
    OP_REQUIRES(
        context, input_batch % block_size_sq == 0,
        errors::InvalidArgument("Input batch dimension ", input_batch,
                                "should be divisible by: ", block_size_sq));


    const int output_batch = input_batch / block_size_sq;
    const int output_height =
        input_height * block_size_ - crops(0, 0) - crops(0, 1);
    const int output_width =
        input_width * block_size_ - crops(1, 0) - crops(1, 1);
    OP_REQUIRES(context, output_height > 0 && output_width > 0,
                errors::InvalidArgument("Output dimensions must be positive"));

    // Allocate output tensor.
    Tensor* output = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(
                                0, TensorShape({output_batch, output_height,
                                                output_width, depth}),
                                &output));

    typename TTypes<T, 4>::ConstTensor Tinput = in0.tensor<T, 4>();
    typename TTypes<T, 4>::Tensor Toutput = output->tensor<T, 4>();

    functor::BatchToSpaceOpFunctor<Device, T> functor;
    functor(context->eigen_device<Device>(),
            Tinput, crops, block_size_, Toutput);
  };

 private:
  int block_size_;
};

// Partial specialization of BatchToSpaceOpFunctor for a CPUDevice.
namespace functor {
template <typename T>
struct BatchToSpaceOpFunctor<CPUDevice, T> {
  void operator()(const CPUDevice& d, typename TTypes<T, 4>::ConstTensor input,
                  typename TTypes<int32>::ConstMatrix crops,
                  int block_size, typename TTypes<T, 4>::Tensor output) {
    const int input_batch = input.dimension(0);
    const int input_height = input.dimension(1);
    const int input_width = input.dimension(2);
    const int depth = input.dimension(3);

    const int output_batch = output.dimension(0);
    const int output_height = output.dimension(1);
    const int output_width = output.dimension(2);

    const int crop_top = crops(0, 0);
    const int crop_left = crops(1, 0);

    for (int in_b = 0; in_b < input_batch; ++in_b) {
      // in_b = (offset_h * block_size + offset_w) * output_batch + out_b
      const int out_b = in_b % output_batch;
      const int offset_w = (in_b / output_batch) % block_size;
      const int offset_h = (in_b / output_batch) / block_size;
      for (int in_h = 0; in_h < input_height; ++in_h) {
        const int out_h = in_h * block_size + offset_h - crop_top;
        for (int in_w = 0; in_w < input_width; ++in_w) {
          const int out_w = in_w * block_size + offset_w - crop_left;
          if (out_h >= 0 && out_w >= 0 &&
              out_h < output_height && out_w < output_width) {
            for (int d = 0; d < depth; ++d) {
              output(out_b, out_h, out_w, d) = input(in_b, in_h, in_w, d);
            }
          }
        }
      }
    }
  }
};
}  // namespace functor

#define REGISTER(T)                                                     \
  REGISTER_KERNEL_BUILDER(Name("BatchToSpace")                          \
                              .Device(DEVICE_CPU)                       \
                              .TypeConstraint<T>("T")                   \
                              .HostMemory("crops"),                     \
                          BatchToSpaceOp<CPUDevice, T>);

TF_CALL_REAL_NUMBER_TYPES(REGISTER);
#undef REGISTER

#if GOOGLE_CUDA
#define REGISTER(T)                                                     \
  REGISTER_KERNEL_BUILDER(Name("BatchToSpace")                          \
                              .Device(DEVICE_GPU)                       \
                              .TypeConstraint<T>("T")                   \
                              .HostMemory("crops"),                     \
                          BatchToSpaceOp<GPUDevice, T>);

TF_CALL_GPU_NUMBER_TYPES_NO_HALF(REGISTER);
#undef REGISTER
#endif  // GOOGLE_CUDA

}  // end namespace tensorflow

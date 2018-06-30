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

#if GOOGLE_CUDA

#define EIGEN_USE_GPU

#include "tensorflow/core/kernels/l2loss_op.h"

#include "tensorflow/core/framework/register_types.h"

#include "tensorflow/core/kernels/reduction_gpu_kernels.cu.h"
#include "tensorflow/core/kernels/reduction_ops_common.h"

namespace tensorflow {

typedef Eigen::GpuDevice GPUDevice;

// TODO(eriche): can add specialization for half2
template <typename T>
struct squareHalf {
  __host__ __device__ T operator()(const T& x) const {
    return static_cast<T>(0.5) * x * x;
  }
};

template <typename T>
class L2LossOp<GPUDevice, T> : public OpKernel {
 public:
  explicit L2LossOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // The input tensor can be of any number of dimensions, even though it's
    // 2D in most typical applications.
    const Tensor& input = context->input(0);
    // The output is a single number.
    Tensor* output = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, TensorShape({}), &output));
    typedef cub::TransformInputIterator<T, squareHalf<T>, T*> inputIterType;
    inputIterType input_itr((T*)input.flat<T>().data(), squareHalf<T>());
    typedef const Eigen::array<TTypes<float>::Tensor::Index, 1>& ReductionAxes;

    Constants<GPUDevice> constants;
    functor::ReduceImpl<T, cub::Sum, T*, inputIterType, ReductionAxes>(
        context, (T*)output->flat<T>().data(), input_itr, 1,
        input.flat<T>().size(), 1, 1, 0, constants.kZero, cub::Sum());
  }
};

// Registration of the GPU implementations.
#define REGISTER_GPU_KERNEL(T)                                  \
  REGISTER_KERNEL_BUILDER(                                      \
      Name("L2Loss").Device(DEVICE_GPU).TypeConstraint<T>("T"), \
      L2LossOp<GPUDevice, T>);

REGISTER_GPU_KERNEL(float);
REGISTER_GPU_KERNEL(double);
REGISTER_GPU_KERNEL(Eigen::half);
#undef REGISTER_GPU_KERNEL

}  // namespace tensorflow

#endif  // GOOGLE_CUDA

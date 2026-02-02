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

#define EIGEN_USE_THREADS
#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#endif  // GOOGLE_CUDA

#include "tensorflow/core/framework/numeric_op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/kernels/bounds_check.h"

#include "third_party/eigen3/Eigen/Core"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

#include "tensorflow/core/kernels/scan_ops.h"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

template <typename Device, class T, typename Reducer>
class ScanOp : public OpKernel {
public:
  explicit ScanOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("reverse", &reverse_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("exclusive", &exclusive_));
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor& input = ctx->input(0);
    const Tensor& tensor_axis = ctx->input(1);

    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(tensor_axis.shape()),
                errors::InvalidArgument("ScanOp: axis must be a scalar, not ",
                                        tensor_axis.shape().DebugString()));

    const int axis = internal::SubtleMustCopy(tensor_axis.scalar<int>()());

    OP_REQUIRES(
        ctx, FastBoundsCheck(axis, input.dims()),
        errors::InvalidArgument("ScanOp: Expected scan axis in the range [", 0,
                                ", ", input.dims(), "), but got ", axis));

    TensorShape output_shape = input.shape();
    Tensor* output = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, output_shape, &output));

    const Device& d = ctx->eigen_device<Device>();
    Reducer reducer;

#define HANDLE_SCAN(NDIMS)                                                \
  case NDIMS:                                                             \
    functor::Scan<Device, Reducer, T, NDIMS>()(                           \
        d, input.tensor<T, NDIMS>(), output->tensor<T, NDIMS>(), reducer, \
        axis, reverse_, exclusive_);                                      \
    return;

    switch (input.dims()) {
      // input.dims() == 0 can't occur as there
      // is no valid axis parameter in this case
      HANDLE_SCAN(1);
      HANDLE_SCAN(2);
      HANDLE_SCAN(3);
      HANDLE_SCAN(4);
      HANDLE_SCAN(5);
      HANDLE_SCAN(6);
      HANDLE_SCAN(7);
      HANDLE_SCAN(8);
      default:
        OP_REQUIRES(ctx, false, errors::InvalidArgument(
                                    "Scan does not support tensors with "
                                    "more than 8 dimensions",
                                    input.dims()));
    }
#undef HANDLE_SCAN
  }

private:
  bool reverse_;
  bool exclusive_;
};

#ifdef GOOGLE_CUDA
namespace functor {

// Forward declarations of GPU functors
#define DECLARE(REDUCER, T, D)                                             \
  template <>                                                              \
  void Scan<GPUDevice, REDUCER, T, D>::operator()(                         \
      const GPUDevice& d, TTypes<T, D>::ConstTensor in,                    \
      TTypes<T, D>::Tensor out, const REDUCER& reducer,                    \
      const Eigen::Index& axis, const bool reverse, const bool exclusive); \
  extern template struct Scan<GPUDevice, REDUCER, T, D>;

#define DECLARE_FOR_ALL_DIMS(REDUCER, T) \
  DECLARE(REDUCER, T, 1);                \
  DECLARE(REDUCER, T, 2);                \
  DECLARE(REDUCER, T, 3);                \
  DECLARE(REDUCER, T, 4);                \
  DECLARE(REDUCER, T, 5);                \
  DECLARE(REDUCER, T, 6);                \
  DECLARE(REDUCER, T, 7);                \
  DECLARE(REDUCER, T, 8);

#define DECLARE_FOR_ALL_REDUCERS(T)                        \
  DECLARE_FOR_ALL_DIMS(Eigen::internal::SumReducer<T>, T); \
  DECLARE_FOR_ALL_DIMS(Eigen::internal::ProdReducer<T>, T);

TF_CALL_GPU_NUMBER_TYPES(DECLARE_FOR_ALL_REDUCERS);

#undef DECLARE_FOR_ALL_REDUCERS
#undef DECLARE_FOR_ALL_DIMS
#undef DECLARE

}  // namespace functor
#endif  // GOOGLE_CUDA


// Register Cumsum kernels
#define REGISTER_CPU_KERNELS(type)                                 \
  REGISTER_KERNEL_BUILDER(                                         \
      Name("Cumsum").Device(DEVICE_CPU).TypeConstraint<type>("T"), \
      ScanOp<CPUDevice, type, Eigen::internal::SumReducer<type>>)
TF_CALL_NUMBER_TYPES(REGISTER_CPU_KERNELS);
#undef REGISTER_CPU_KERNELS

#if GOOGLE_CUDA
#define REGISTER_GPU_KERNELS(type)   \
  REGISTER_KERNEL_BUILDER(           \
      Name("Cumsum")                 \
          .Device(DEVICE_GPU)        \
          .TypeConstraint<type>("T") \
          .HostMemory("axis"),       \
      ScanOp<GPUDevice, type, Eigen::internal::SumReducer<type>>)
TF_CALL_GPU_NUMBER_TYPES(REGISTER_GPU_KERNELS)
#undef REGISTER_GPU_KERNELS
#endif // GOOGLE_CUDA


// Register Cumprod kernels
#define REGISTER_CPU_KERNELS(type)                                  \
  REGISTER_KERNEL_BUILDER(                                          \
      Name("Cumprod").Device(DEVICE_CPU).TypeConstraint<type>("T"), \
      ScanOp<CPUDevice, type, Eigen::internal::ProdReducer<type>>)
TF_CALL_NUMBER_TYPES(REGISTER_CPU_KERNELS);
#undef REGISTER_CPU_KERNELS

#if GOOGLE_CUDA
#define REGISTER_GPU_KERNELS(type)   \
  REGISTER_KERNEL_BUILDER(           \
      Name("Cumprod")                \
          .Device(DEVICE_GPU)        \
          .TypeConstraint<type>("T") \
          .HostMemory("axis"),       \
      ScanOp<GPUDevice, type, Eigen::internal::ProdReducer<type>>)
TF_CALL_GPU_NUMBER_TYPES(REGISTER_GPU_KERNELS)
#undef REGISTER_GPU_KERNELS
#endif // GOOGLE_CUDA

}  // namespace tensorflow

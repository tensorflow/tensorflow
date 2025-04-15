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
#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#define EIGEN_USE_GPU
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#include "tensorflow/core/kernels/scan_ops.h"

#include "unsupported/Eigen/CXX11/Tensor"  // from @eigen_archive
#include "tensorflow/core/framework/bounds_check.h"
#include "tensorflow/core/framework/numeric_op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

template <typename Device, class T, typename Reducer, typename Tidx>
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

    const Tidx axis_arg =
        internal::SubtleMustCopy(tensor_axis.scalar<Tidx>()());
    const Tidx axis = (axis_arg < 0) ? input.dims() + axis_arg : axis_arg;
    OP_REQUIRES(ctx, FastBoundsCheck(axis, input.dims()),
                errors::InvalidArgument(
                    "ScanOp: Expected scan axis in the range [", -input.dims(),
                    ", ", input.dims(), "), but got ", axis));

    const TensorShape& output_shape = input.shape();
    Tensor* output = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, output_shape, &output));

    // Exit early if there's nothing to compute
    if (output_shape.num_elements() == 0) return;

    const Device& d = ctx->eigen_device<Device>();
    Reducer reducer;

    // Dim reduction.
    int64_t reduced_shape[3] = {1, 1, 1};
    for (Tidx i = 0; i < axis; ++i) {
      reduced_shape[0] *= input.dim_size(i);
    }
    reduced_shape[1] = input.dim_size(axis);
    for (Tidx i = axis + 1; i < input.dims(); ++i) {
      reduced_shape[2] *= input.dim_size(i);
    }

    functor::Scan<Device, Reducer, T>()(d, input.shaped<T, 3>(reduced_shape),
                                        output->shaped<T, 3>(reduced_shape),
                                        reducer, reverse_, exclusive_);
  }

 private:
  bool reverse_;
  bool exclusive_;
};

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
namespace functor {

// Forward declarations of GPU functors
#define DECLARE(REDUCER, T)                                                 \
  template <>                                                               \
  void Scan<GPUDevice, REDUCER, T>::operator()(                             \
      const GPUDevice& d, TTypes<T, 3>::ConstTensor in,                     \
      TTypes<T, 3>::Tensor out, const REDUCER& reducer, const bool reverse, \
      const bool exclusive);                                                \
  extern template struct Scan<GPUDevice, REDUCER, T>;

#define DECLARE_FOR_ALL_REDUCERS(T)           \
  DECLARE(Eigen::internal::SumReducer<T>, T); \
  DECLARE(Eigen::internal::ProdReducer<T>, T);

TF_CALL_GPU_NUMBER_TYPES(DECLARE_FOR_ALL_REDUCERS);
DECLARE_FOR_ALL_REDUCERS(int32);
DECLARE_FOR_ALL_REDUCERS(int64_t);
#undef DECLARE_FOR_ALL_REDUCERS

#define DECLARE_FOR_LOGSUMEXP_REDUCER(T) DECLARE(LogSumExpReducer<T>, T);
TF_CALL_GPU_NUMBER_TYPES(DECLARE_FOR_LOGSUMEXP_REDUCER);
#undef DECLARE_FOR_LOGSUMEXP_REDUCER

#undef DECLARE

}  // namespace functor
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

// Register Cumsum kernels
#define REGISTER_CPU_KERNELS(type)                                       \
  REGISTER_KERNEL_BUILDER(                                               \
      Name("Cumsum")                                                     \
          .Device(DEVICE_CPU)                                            \
          .TypeConstraint<type>("T")                                     \
          .TypeConstraint<int32>("Tidx"),                                \
      ScanOp<CPUDevice, type, Eigen::internal::SumReducer<type>, int32>) \
  REGISTER_KERNEL_BUILDER(                                               \
      Name("Cumsum")                                                     \
          .Device(DEVICE_CPU)                                            \
          .TypeConstraint<type>("T")                                     \
          .TypeConstraint<int64_t>("Tidx"),                              \
      ScanOp<CPUDevice, type, Eigen::internal::SumReducer<type>, int64>)
TF_CALL_NUMBER_TYPES(REGISTER_CPU_KERNELS);
#undef REGISTER_CPU_KERNELS

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#define REGISTER_GPU_KERNELS(type)                                       \
  REGISTER_KERNEL_BUILDER(                                               \
      Name("Cumsum")                                                     \
          .Device(DEVICE_GPU)                                            \
          .TypeConstraint<type>("T")                                     \
          .TypeConstraint<int32>("Tidx")                                 \
          .HostMemory("axis"),                                           \
      ScanOp<GPUDevice, type, Eigen::internal::SumReducer<type>, int32>) \
  REGISTER_KERNEL_BUILDER(                                               \
      Name("Cumsum")                                                     \
          .Device(DEVICE_GPU)                                            \
          .TypeConstraint<type>("T")                                     \
          .TypeConstraint<int64_t>("Tidx")                               \
          .HostMemory("axis"),                                           \
      ScanOp<GPUDevice, type, Eigen::internal::SumReducer<type>, int64>)
TF_CALL_GPU_NUMBER_TYPES(REGISTER_GPU_KERNELS);
REGISTER_GPU_KERNELS(int32);
REGISTER_GPU_KERNELS(int64_t);
#undef REGISTER_GPU_KERNELS
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

// Register Cumprod kernels
#define REGISTER_CPU_KERNELS(type)                                        \
  REGISTER_KERNEL_BUILDER(                                                \
      Name("Cumprod")                                                     \
          .Device(DEVICE_CPU)                                             \
          .TypeConstraint<type>("T")                                      \
          .TypeConstraint<int32>("Tidx"),                                 \
      ScanOp<CPUDevice, type, Eigen::internal::ProdReducer<type>, int32>) \
  REGISTER_KERNEL_BUILDER(                                                \
      Name("Cumprod")                                                     \
          .Device(DEVICE_CPU)                                             \
          .TypeConstraint<type>("T")                                      \
          .TypeConstraint<int64_t>("Tidx"),                               \
      ScanOp<CPUDevice, type, Eigen::internal::ProdReducer<type>, int64>)
TF_CALL_NUMBER_TYPES(REGISTER_CPU_KERNELS);
#undef REGISTER_CPU_KERNELS

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#define REGISTER_GPU_KERNELS(type)                                        \
  REGISTER_KERNEL_BUILDER(                                                \
      Name("Cumprod")                                                     \
          .Device(DEVICE_GPU)                                             \
          .TypeConstraint<type>("T")                                      \
          .TypeConstraint<int32>("Tidx")                                  \
          .HostMemory("axis"),                                            \
      ScanOp<GPUDevice, type, Eigen::internal::ProdReducer<type>, int32>) \
  REGISTER_KERNEL_BUILDER(                                                \
      Name("Cumprod")                                                     \
          .Device(DEVICE_GPU)                                             \
          .TypeConstraint<type>("T")                                      \
          .TypeConstraint<int64_t>("Tidx")                                \
          .HostMemory("axis"),                                            \
      ScanOp<GPUDevice, type, Eigen::internal::ProdReducer<type>, int64>)
TF_CALL_GPU_NUMBER_TYPES(REGISTER_GPU_KERNELS);
REGISTER_GPU_KERNELS(int32);
REGISTER_GPU_KERNELS(int64_t);
#undef REGISTER_GPU_KERNELS
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#define REGISTER_CUMLOGSUMEXP_KERNEL(device, device_type, type, type_idx) \
  REGISTER_KERNEL_BUILDER(                                                \
      Name("CumulativeLogsumexp")                                         \
          .Device(device)                                                 \
          .TypeConstraint<type>("T")                                      \
          .TypeConstraint<type_idx>("Tidx")                               \
          .HostMemory("axis"),                                            \
      ScanOp<device_type, type, functor::LogSumExpReducer<type>, type_idx>)

#define REGISTER_CPU_KERNELS(type)                                 \
  REGISTER_CUMLOGSUMEXP_KERNEL(DEVICE_CPU, CPUDevice, type, int32) \
  REGISTER_CUMLOGSUMEXP_KERNEL(DEVICE_CPU, CPUDevice, type, int64_t)

TF_CALL_FLOAT_TYPES(REGISTER_CPU_KERNELS);
#undef REGISTER_CPU_KERNELS

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#define REGISTER_GPU_KERNELS(type)                                 \
  REGISTER_CUMLOGSUMEXP_KERNEL(DEVICE_GPU, GPUDevice, type, int32) \
  REGISTER_CUMLOGSUMEXP_KERNEL(DEVICE_GPU, GPUDevice, type, int64_t)

TF_CALL_GPU_NUMBER_TYPES(REGISTER_GPU_KERNELS);
#undef REGISTER_GPU_KERNELS
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#undef REGISTER_CUMLOGSUMEXP_KERNEL

}  // namespace tensorflow

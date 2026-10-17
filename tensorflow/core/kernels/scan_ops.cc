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

#include <type_traits>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"

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
                absl::InvalidArgumentError(
                    absl::StrCat("ScanOp: axis must be a scalar, not ",
                                 tensor_axis.shape().DebugString())));

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

    // Dim reduction.
    int64_t reduced_shape[3] = {1, 1, 1};
    for (Tidx i = 0; i < axis; ++i) {
      reduced_shape[0] *= input.dim_size(i);
    }
    reduced_shape[1] = input.dim_size(axis);
    for (Tidx i = axis + 1; i < input.dims(); ++i) {
      reduced_shape[2] *= input.dim_size(i);
    }

    // bfloat16 and float16 keep only ~8 / ~10 mantissa bits, so accumulating a
    // long sequence directly in the 16-bit type rounds the running total at
    // every step and drifts far from the exact result. Upcast to float32 for
    // the accumulation and cast the result back; this achieves numerical
    // accuracy comparable to the GPU CUB block scan, preserves the documented
    // output dtype, and avoids committing the 16-bit rounding error on the CPU
    // path.
    //
    // The upcast is fused into the reduction: `.cast<float>()` is chained into
    // the scan expression, so no full-size float32 temporaries are allocated.
    // The narrowing `.cast<T>()` must come *before* the trailing
    // `.reverse(dims)`. Eigen's block-based evaluation hands the same
    // destination block down the whole expression chain, so every node that
    // receives it needs the destination's scalar type; with the narrow-cast
    // last, a float32 `reverse` node sits directly over the 16-bit destination
    // and writes 4-byte elements into a 2-byte-per-element buffer.
    if constexpr (std::is_same_v<Device, CPUDevice> &&
                  (std::is_same_v<Reducer,
                                   Eigen::internal::SumReducer<T>> ||
                   std::is_same_v<Reducer,
                                   Eigen::internal::ProdReducer<T>> ||
                   std::is_same_v<Reducer,
                                   functor::LogSumExpReducer<T>>) &&
                  (std::is_same_v<T, ::tensorflow::bfloat16> ||
                   std::is_same_v<T, Eigen::half>)) {
      Eigen::array<bool, 3> dims;
      dims[0] = false;
      dims[1] = reverse_;
      dims[2] = false;

      // Map the 16-bit reducer onto its float32 counterpart, so inputs,
      // outputs and the reducer are all float32 inside the scan.
      auto float_reducer = []() {
        if constexpr (std::is_same_v<Reducer,
                                     Eigen::internal::SumReducer<T>>) {
          return Eigen::internal::SumReducer<float>();
        } else if constexpr (std::is_same_v<Reducer,
                                            Eigen::internal::ProdReducer<T>>) {
          return Eigen::internal::ProdReducer<float>();
        } else {
          return functor::LogSumExpReducer<float>();
        }
      };

      MaybeWith32BitIndexing<Device>(
          [&](auto in32, auto out32) {
            out32.device(d) =
                in32.template cast<float>()
                    .reverse(dims)
                    .scan(1, float_reducer(), exclusive_)
                    .template cast<T>()
                    .reverse(dims);
          },
          input.shaped<T, 3>(reduced_shape),
          output->shaped<T, 3>(reduced_shape));
      return;
    }

    Reducer reducer;
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
DECLARE_FOR_ALL_REDUCERS(int32_t);
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
REGISTER_GPU_KERNELS(int32_t);
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
REGISTER_GPU_KERNELS(int32_t);
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

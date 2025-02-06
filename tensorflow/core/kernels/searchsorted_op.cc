/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/kernels/searchsorted_op.h"

#include "tensorflow/core/framework/bounds_check.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/kernels/fill_functor.h"
#include "tensorflow/core/lib/core/bits.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/threadpool.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

namespace functor {
template <typename T, typename OutType>
struct UpperBoundFunctor<CPUDevice, T, OutType> {
  static absl::Status Compute(
      OpKernelContext* context,
      const typename TTypes<T, 1>::ConstTensor& sorted_inputs,
      const typename TTypes<T, 1>::ConstTensor& values, int batch_size,
      int num_inputs, int num_values,
      typename TTypes<OutType, 1>::Tensor* output) {
    auto work_fn = [&](int64_t first, int64_t last) {
      for (int b = 0; b < batch_size; ++b) {
        const T* sorted_inputs_ptr = sorted_inputs.data() + b * num_inputs;
        OutType* output_ptr = output->data() + b * num_values;
        for (int i = first; i < last; ++i) {
          output_ptr[i] = std::upper_bound(sorted_inputs_ptr,
                                           sorted_inputs_ptr + num_inputs,
                                           values(i + b * num_values)) -
                          sorted_inputs_ptr;
        }
      }
    };
    auto worker_threads = *(context->device()->tensorflow_cpu_worker_threads());
    thread::ThreadPool* thread_pool = worker_threads.workers;
    const float kCostMultiplier = 1.f;  // Can be tuned to minimize overhead
    int64_t cost_per_unit =
        kCostMultiplier * batch_size * Log2Ceiling(num_inputs);
    thread_pool->ParallelFor(num_values, cost_per_unit, work_fn);
    return absl::OkStatus();
  }
};

template <typename T, typename OutType>
struct LowerBoundFunctor<CPUDevice, T, OutType> {
  static absl::Status Compute(
      OpKernelContext* context,
      const typename TTypes<T, 1>::ConstTensor& sorted_inputs,
      const typename TTypes<T, 1>::ConstTensor& values, int batch_size,
      int num_inputs, int num_values,
      typename TTypes<OutType, 1>::Tensor* output) {
    auto work_fn = [&](int64_t first, int64_t last) {
      for (int b = 0; b < batch_size; ++b) {
        const T* sorted_inputs_ptr = sorted_inputs.data() + b * num_inputs;
        OutType* output_ptr = output->data() + b * num_values;
        for (int i = first; i < last; ++i) {
          output_ptr[i] = std::lower_bound(sorted_inputs_ptr,
                                           sorted_inputs_ptr + num_inputs,
                                           values(i + b * num_values)) -
                          sorted_inputs_ptr;
        }
      }
    };
    auto worker_threads = *(context->device()->tensorflow_cpu_worker_threads());
    thread::ThreadPool* thread_pool = worker_threads.workers;
    const float kCostMultiplier = 1.f;  // Can be tuned to minimize overhead
    int64_t cost_per_unit =
        kCostMultiplier * batch_size * Log2Ceiling(num_inputs);
    thread_pool->ParallelFor(num_values, cost_per_unit, work_fn);
    return absl::OkStatus();
  }
};
}  // namespace functor

template <typename Device, typename T, typename OutType>
class UpperBoundOp : public OpKernel {
 public:
  explicit UpperBoundOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    const Tensor& sorted_inputs_t = ctx->input(0);
    const Tensor& values_t = ctx->input(1);

    // Inputs must be a matrix
    // This replicates the shape requirements for the op in array_ops.cc
    OP_REQUIRES(
        ctx, sorted_inputs_t.shape().dims() == 2,
        errors::InvalidArgument(absl::StrCat(
            "Shape must be rank 2 but is rank ", sorted_inputs_t.shape().dims(),
            " for "
            "`sorted_inputs` argument")));
    // Values must be a matrix
    // This replicates the shape requirements for the op in array_ops.cc
    OP_REQUIRES(ctx, values_t.shape().dims() == 2,
                errors::InvalidArgument(absl::StrCat(
                    "Shape must be rank 2 but is rank ",
                    values_t.shape().dims(), " for `values` argument")));
    // must have same batch dim_size for both
    OP_REQUIRES(ctx, sorted_inputs_t.dim_size(0) == values_t.dim_size(0),
                absl::Status(absl::StatusCode::kInvalidArgument,
                             "Leading dim_size of both tensors must match."));

    // this is required because we do indexing in int32 on the GPU
    OP_REQUIRES(ctx, values_t.NumElements() < std::numeric_limits<int>::max(),
                absl::Status(absl::StatusCode::kInvalidArgument,
                             "values tensor size must less than INT_MAX"));

    Tensor* output_t;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, values_t.shape(), &output_t));

    if (output_t->dtype() == DT_INT32) {
      OP_REQUIRES(ctx,
                  FastBoundsCheck(sorted_inputs_t.dim_size(1),
                                  std::numeric_limits<int>::max()),
                  errors::InvalidArgument("trailing dim_size must less than "
                                          "INT_MAX for int32 output type, was ",
                                          sorted_inputs_t.dim_size(1)));
    }

    auto output = output_t->template flat<OutType>();
    const auto sorted_inputs = sorted_inputs_t.template flat<T>();
    const auto values = values_t.template flat<T>();

    // For empty inputs, all values will be placed at the zeroth position.
    if (sorted_inputs.size() == 0) {
      functor::SetZeroFunctor<Device, OutType> set_zero;
      set_zero(ctx->eigen_device<Device>(), output);
      return;
    }

    OP_REQUIRES_OK(
        ctx, functor::UpperBoundFunctor<Device, T, OutType>::Compute(
                 ctx, sorted_inputs, values, sorted_inputs_t.dim_size(0),
                 sorted_inputs_t.dim_size(1), values_t.dim_size(1), &output));
  }
};

template <typename Device, typename T, typename OutType>
class LowerBoundOp : public OpKernel {
 public:
  explicit LowerBoundOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    const Tensor& sorted_inputs_t = ctx->input(0);
    const Tensor& values_t = ctx->input(1);

    // Inputs must be a matrix
    // This replicates the shape requirements for the op in array_ops.cc
    OP_REQUIRES(
        ctx, sorted_inputs_t.shape().dims() == 2,
        errors::InvalidArgument(absl::StrCat(
            "Shape must be rank 2 but is rank ", sorted_inputs_t.shape().dims(),
            " for "
            "`sorted_inputs` argument")));
    // Values must be a matrix
    // This replicates the shape requirements for the op in array_ops.cc
    OP_REQUIRES(ctx, values_t.shape().dims() == 2,
                errors::InvalidArgument(absl::StrCat(
                    "Shape must be rank 2 but is rank ",
                    values_t.shape().dims(), " for `values` argument")));
    // must have same batch dim_size for both
    OP_REQUIRES(ctx, sorted_inputs_t.dim_size(0) == values_t.dim_size(0),
                absl::Status(absl::StatusCode::kInvalidArgument,
                             "Leading dim_size of both tensors must match."));

    // this is required because we do indexing in int32 on the GPU
    OP_REQUIRES(ctx, values_t.NumElements() < std::numeric_limits<int>::max(),
                absl::Status(absl::StatusCode::kInvalidArgument,
                             "values tensor size must less than INT_MAX"));

    Tensor* output_t;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, values_t.shape(), &output_t));

    if (output_t->dtype() == DT_INT32) {
      OP_REQUIRES(ctx,
                  FastBoundsCheck(sorted_inputs_t.dim_size(1),
                                  std::numeric_limits<int>::max()),
                  errors::InvalidArgument("trailing dim_size must less than "
                                          "INT_MAX for int32 output type, was ",
                                          sorted_inputs_t.dim_size(1)));
    }

    auto output = output_t->template flat<OutType>();
    const auto sorted_inputs = sorted_inputs_t.template flat<T>();
    const auto values = values_t.template flat<T>();

    // For empty inputs, all values will be placed at the zeroth position.
    if (sorted_inputs.size() == 0) {
      functor::SetZeroFunctor<Device, OutType> set_zero;
      set_zero(ctx->eigen_device<Device>(), output);
      return;
    }

    OP_REQUIRES_OK(
        ctx, functor::LowerBoundFunctor<Device, T, OutType>::Compute(
                 ctx, sorted_inputs, values, sorted_inputs_t.dim_size(0),
                 sorted_inputs_t.dim_size(1), values_t.dim_size(1), &output));
  }
};

#define REGISTER_KERNELS(type)                                    \
  REGISTER_KERNEL_BUILDER(Name("UpperBound")                      \
                              .Device(DEVICE_CPU)                 \
                              .TypeConstraint<type>("T")          \
                              .TypeConstraint<int32>("out_type"), \
                          UpperBoundOp<CPUDevice, type, int32>);

TF_CALL_REAL_NUMBER_TYPES(REGISTER_KERNELS);
#undef REGISTER_KERNELS

#define REGISTER_KERNELS(type)                                      \
  REGISTER_KERNEL_BUILDER(Name("UpperBound")                        \
                              .Device(DEVICE_CPU)                   \
                              .TypeConstraint<type>("T")            \
                              .TypeConstraint<int64_t>("out_type"), \
                          UpperBoundOp<CPUDevice, type, int64>);

TF_CALL_REAL_NUMBER_TYPES(REGISTER_KERNELS);
#undef REGISTER_KERNELS

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#define REGISTER_KERNELS(type)                                    \
  REGISTER_KERNEL_BUILDER(Name("UpperBound")                      \
                              .Device(DEVICE_GPU)                 \
                              .TypeConstraint<type>("T")          \
                              .TypeConstraint<int32>("out_type"), \
                          UpperBoundOp<GPUDevice, type, int32>);

TF_CALL_REAL_NUMBER_TYPES(REGISTER_KERNELS);
#undef REGISTER_KERNELS

#define REGISTER_KERNELS(type)                                      \
  REGISTER_KERNEL_BUILDER(Name("UpperBound")                        \
                              .Device(DEVICE_GPU)                   \
                              .TypeConstraint<type>("T")            \
                              .TypeConstraint<int64_t>("out_type"), \
                          UpperBoundOp<GPUDevice, type, int64>);

TF_CALL_REAL_NUMBER_TYPES(REGISTER_KERNELS);
#undef REGISTER_KERNELS

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#define REGISTER_KERNELS(type)                                    \
  REGISTER_KERNEL_BUILDER(Name("LowerBound")                      \
                              .Device(DEVICE_CPU)                 \
                              .TypeConstraint<type>("T")          \
                              .TypeConstraint<int32>("out_type"), \
                          LowerBoundOp<CPUDevice, type, int32>);

TF_CALL_REAL_NUMBER_TYPES(REGISTER_KERNELS);
#undef REGISTER_KERNELS

#define REGISTER_KERNELS(type)                                      \
  REGISTER_KERNEL_BUILDER(Name("LowerBound")                        \
                              .Device(DEVICE_CPU)                   \
                              .TypeConstraint<type>("T")            \
                              .TypeConstraint<int64_t>("out_type"), \
                          LowerBoundOp<CPUDevice, type, int64>);

TF_CALL_REAL_NUMBER_TYPES(REGISTER_KERNELS);
#undef REGISTER_KERNELS

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#define REGISTER_KERNELS(type)                                    \
  REGISTER_KERNEL_BUILDER(Name("LowerBound")                      \
                              .Device(DEVICE_GPU)                 \
                              .TypeConstraint<type>("T")          \
                              .TypeConstraint<int32>("out_type"), \
                          LowerBoundOp<GPUDevice, type, int32>);

TF_CALL_REAL_NUMBER_TYPES(REGISTER_KERNELS);
#undef REGISTER_KERNELS

#define REGISTER_KERNELS(type)                                      \
  REGISTER_KERNEL_BUILDER(Name("LowerBound")                        \
                              .Device(DEVICE_GPU)                   \
                              .TypeConstraint<type>("T")            \
                              .TypeConstraint<int64_t>("out_type"), \
                          LowerBoundOp<GPUDevice, type, int64>);

TF_CALL_REAL_NUMBER_TYPES(REGISTER_KERNELS);
#undef REGISTER_KERNELS

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
}  // namespace tensorflow

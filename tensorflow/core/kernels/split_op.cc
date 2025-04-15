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

// See docs in ../ops/array_ops.cc.

#define EIGEN_USE_THREADS

#include "unsupported/Eigen/CXX11/Tensor"  // from @eigen_archive
#include "tensorflow/core/framework/bounds_check.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/kernels/split_lib.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/util/work_sharder.h"
#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#include "tensorflow/core/common_runtime/gpu/gpu_event_mgr.h"
#include "tensorflow/core/kernels/gpu_device_array.h"
#include "tensorflow/core/kernels/split_lib_gpu.h"
#include "tensorflow/core/platform/stream_executor.h"
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

template <typename Device, typename T>
class SplitOpBase : public OpKernel {
 public:
  explicit SplitOpBase(OpKernelConstruction* c) : OpKernel(c) {}

  void ComputeEasyCases(OpKernelContext* context, bool* done) {
    const Tensor& input = context->input(1);
    const TensorShape& input_shape = input.shape();
    const Tensor& split_dim_tensor = context->input(0);
    OP_REQUIRES(
        context, split_dim_tensor.shape().dims() == 0,
        errors::InvalidArgument("split_dim must be a scalar but has rank ",
                                split_dim_tensor.shape().dims()));
    const int32_t split_dim_orig = split_dim_tensor.flat<int32>()(0);
    const int32_t split_dim =
        split_dim_orig < 0 ? split_dim_orig + input.dims() : split_dim_orig;
    const int32_t num_split = num_outputs();

    OP_REQUIRES(
        context, 0 <= split_dim && split_dim < input_shape.dims(),
        errors::InvalidArgument("-input rank(-", input.dims(),
                                ") <= split_dim < input rank (", input.dims(),
                                "), but got ", split_dim_orig));

    OP_REQUIRES(
        context, num_split > 0,
        errors::InvalidArgument(
            "Number of ways to split should be > 0, but got ", num_split));

    OP_REQUIRES(context, input_shape.dim_size(split_dim) % num_split == 0,
                errors::InvalidArgument(
                    "Number of ways to split should evenly divide the split "
                    "dimension, but got split_dim ",
                    split_dim, " (size = ", input_shape.dim_size(split_dim),
                    ") ", "and num_split ", num_split));
    // Special case 1: num_split == 1. Nothing to do.
    if (num_split == 1) {
      VLOG(1) << "Split identity";
      context->set_output(0, context->input(1));
      *done = true;
      return;
    }

    // Special case 2: split along the 1st dimension. We can share the
    // underlying buffer.
    //
    // Apply this optimization conservatively: if input is aligned,
    // the resulting tensors must be aligned. It's conservative
    // because if the immediate consumer of the resulting tensors are
    // not using eigen for computation, its perfectly fine to avoid
    // the copying.
    if ((split_dim == 0) && IsInnerDimsSizeAligned<T>(input_shape)) {
      VLOG(1) << "Slice dim 0: " << input_shape.DebugString();
      const int64_t delta = input_shape.dim_size(0) / num_split;
      for (int i = 0; i < num_split; ++i) {
        context->set_output(i, input.Slice(i * delta, (i + 1) * delta));
      }
      *done = true;
      return;
    }
  }

  template <typename IndexType>
  std::tuple<IndexType, IndexType, IndexType> SetDims(
      const TensorShape& input_shape, int32_t split_dim) const {
    static_assert(std::is_integral<IndexType>::value,
                  "IndexType must be an integer type");
    int32_t prefix_dim_size = 1;
    for (int i = 0; i < split_dim; ++i) {
      prefix_dim_size *= input_shape.dim_size(i);
    }

    // Caller must ensure that dim_size and suffix_dim_size are <
    // std::numeric_limits<IndexType>::max()
    IndexType split_dim_size =
        static_cast<IndexType>(input_shape.dim_size(split_dim));

    IndexType suffix_dim_size = 1;
    for (int i = split_dim + 1; i < input_shape.dims(); ++i) {
      suffix_dim_size *= static_cast<IndexType>(input_shape.dim_size(i));
    }
    return std::make_tuple(prefix_dim_size, split_dim_size, suffix_dim_size);
  }
};

template <typename T, typename InputReshapedType, int NDims>
class SplitOpCPUImpl {
 public:
  template <typename MakeSizesType, typename ReshapeResultType>
  void operator()(OpKernelContext* context,
                  const InputReshapedType& input_reshaped,
                  const TensorShape& input_shape, int32_t split_dim,
                  Eigen::DenseIndex prefix_dim_size,
                  Eigen::DenseIndex split_dim_size,
                  Eigen::DenseIndex suffix_dim_size,
                  const MakeSizesType& make_sizes,
                  const ReshapeResultType& reshape_result, int32_t num_split,
                  int64_t split_dim_output_size) const {
    const auto num_threads =
        context->device()->tensorflow_cpu_worker_threads()->num_threads;
    // TODO(jewillco): Tune heuristic further.
    const auto input_element_count = input_shape.num_elements();
    const bool use_parallelism_between_outputs =
        (num_split >= 4 &&
         input_element_count >= std::max(num_threads, num_split) * 4096 &&
         input_element_count < num_split * 180 * 1024);
    Eigen::DSizes<Eigen::DenseIndex, NDims> indices;
    for (int i = 0; i < NDims; ++i) {
      indices[i] = 0;
    }
    auto sizes = make_sizes(split_dim_output_size);
    TensorShape output_shape(input_shape);
    output_shape.set_dim(split_dim, split_dim_output_size);

    auto range_output_func = [&indices, context, &output_shape, prefix_dim_size,
                              split_dim_output_size, suffix_dim_size, &sizes,
                              use_parallelism_between_outputs, &input_reshaped,
                              &reshape_result](int64_t start, int64_t limit) {
      for (int64_t i = start; i < limit; ++i) {
        Tensor* result = nullptr;
        OP_REQUIRES_OK(context,
                       context->allocate_output(i, output_shape, &result));
        if (prefix_dim_size * split_dim_output_size * suffix_dim_size > 0) {
          Eigen::DSizes<Eigen::DenseIndex, NDims> slice_indices;
          Eigen::DSizes<Eigen::DenseIndex, NDims> slice_sizes;
          for (int j = 0; j < NDims; ++j) {
            slice_indices[j] =
                (j == NDims - 2 ? i * split_dim_output_size : indices[j]);
            slice_sizes[j] = sizes[j];
          }

          auto result_shaped = reshape_result(result, split_dim_output_size);

          if (use_parallelism_between_outputs) {
            // Use sequential implementation for single output.
            result_shaped = input_reshaped.slice(slice_indices, slice_sizes);
          } else {
            // This implementation may be parallel internally.
            functor::Split<CPUDevice, T, NDims>()(
                context->eigen_device<CPUDevice>(), result_shaped,
                input_reshaped, slice_indices, slice_sizes);
          }
        }
      }
    };
    if (use_parallelism_between_outputs) {
      // Run in parallel, disabling parallelism in functor.
      context->device()->tensorflow_cpu_worker_threads()->workers->ParallelFor(
          num_split, input_element_count / num_split, range_output_func);
    } else {
      // Run sequentially, but allow internal parallelism in functor.
      range_output_func(0, num_split);
    }
  }
};

template <typename T>
class SplitOpCPU : public SplitOpBase<CPUDevice, T> {
 public:
  typedef SplitOpBase<CPUDevice, T> Base;
  explicit SplitOpCPU(OpKernelConstruction* c) : Base(c) {}

  void Compute(OpKernelContext* context) override {
    bool done = false;
    Base::ComputeEasyCases(context, &done);
    if (!context->status().ok() || done) {
      return;
    }
    const int32_t num_split = Base::num_outputs();
    const Tensor& input = context->input(1);
    const TensorShape& input_shape = input.shape();
    const int32_t split_dim_orig = context->input(0).flat<int32>()(0);
    const int32_t split_dim =
        split_dim_orig < 0 ? split_dim_orig + input.dims() : split_dim_orig;

    // Android also uses int32 indexing, so check here also.
    OP_REQUIRES(
        context,
        FastBoundsCheck(input.NumElements(),
                        std::numeric_limits<Eigen::DenseIndex>::max()),
        errors::InvalidArgument("Split requires input size < ",
                                std::numeric_limits<Eigen::DenseIndex>::max()));

    Eigen::DenseIndex prefix_dim_size;
    Eigen::DenseIndex split_dim_size;
    Eigen::DenseIndex suffix_dim_size;

    std::tie(prefix_dim_size, split_dim_size, suffix_dim_size) =
        Base::template SetDims<Eigen::DenseIndex>(input_shape, split_dim);

    const int64_t split_dim_output_size = split_dim_size / num_split;

    if (prefix_dim_size == 1) {
      auto input_reshaped =
          input.shaped<T, 2>({split_dim_size, suffix_dim_size});
      auto make_sizes = [&](Eigen::DenseIndex split_size) {
        return Eigen::DSizes<Eigen::DenseIndex, 2>{split_size, suffix_dim_size};
      };
      auto reshape_result = [&](Tensor* result, Eigen::DenseIndex split_size) {
        return result->shaped<T, 2>({split_size, suffix_dim_size});
      };
      SplitOpCPUImpl<T, decltype(input_reshaped), 2>{}(
          context, input_reshaped, input_shape, split_dim, prefix_dim_size,
          split_dim_size, suffix_dim_size, make_sizes, reshape_result,
          num_split, split_dim_output_size);
    } else {
      auto input_reshaped = input.shaped<T, 3>(
          {prefix_dim_size, split_dim_size, suffix_dim_size});
      auto make_sizes = [&](Eigen::DenseIndex split_size) {
        return Eigen::DSizes<Eigen::DenseIndex, 3>{prefix_dim_size, split_size,
                                                   suffix_dim_size};
      };
      auto reshape_result = [&](Tensor* result, Eigen::DenseIndex split_size) {
        return result->shaped<T, 3>(
            {prefix_dim_size, split_size, suffix_dim_size});
      };
      SplitOpCPUImpl<T, decltype(input_reshaped), 3>{}(
          context, input_reshaped, input_shape, split_dim, prefix_dim_size,
          split_dim_size, suffix_dim_size, make_sizes, reshape_result,
          num_split, split_dim_output_size);
    }
  }
};

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM

// Partial specialization for GPU
template <typename T>
class SplitOpGPU : public SplitOpBase<GPUDevice, T> {
 public:
  typedef SplitOpBase<GPUDevice, T> Base;
  explicit SplitOpGPU(OpKernelConstruction* c) : Base(c) {}

  void Compute(OpKernelContext* context) override {
    bool done = false;
    Base::ComputeEasyCases(context, &done);
    if (!context->status().ok() || done) {
      return;
    }
    const Tensor& input = context->input(1);
    const TensorShape& input_shape = input.shape();
    const int32_t split_dim_orig = context->input(0).flat<int32>()(0);
    const int32_t split_dim =
        split_dim_orig < 0 ? split_dim_orig + input.dims() : split_dim_orig;
    const int32_t num_split = Base::num_outputs();
    OP_REQUIRES(
        context,
        FastBoundsCheck(input.NumElements(), std::numeric_limits<int32>::max()),
        errors::InvalidArgument("Split on GPU requires input size "
                                "< max int32"));
    int32_t prefix_dim_size;
    int32_t split_dim_size;
    int32_t suffix_dim_size;
    std::tie(prefix_dim_size, split_dim_size, suffix_dim_size) =
        Base::template SetDims<int32>(input_shape, split_dim);

    const int32_t split_dim_output_size = split_dim_size / num_split;
    TensorShape output_shape(input_shape);
    output_shape.set_dim(split_dim, split_dim_output_size);

    GpuDeviceArrayOnHost<T*> ptrs(context, num_split);
    OP_REQUIRES_OK(context, ptrs.Init());

    for (int i = 0; i < num_split; ++i) {
      Tensor* result = nullptr;
      OP_REQUIRES_OK(context,
                     context->allocate_output(i, output_shape, &result));
      ptrs.Set(i, result->flat<T>().data());
    }
    if (prefix_dim_size * split_dim_output_size * suffix_dim_size == 0) {
      return;
    }
    OP_REQUIRES_OK(context, ptrs.Finalize());

    SplitOpGPULaunch<T>().Run(context->eigen_device<GPUDevice>(),
                              input.flat<T>().data(), prefix_dim_size,
                              split_dim_size, suffix_dim_size, ptrs.data());
    OP_REQUIRES(context, context->op_device_context()->stream()->ok(),
                errors::Internal("Launch of gpu kernel for SplitOp failed"));
  }
};
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#define REGISTER_SPLIT(type)                             \
  REGISTER_KERNEL_BUILDER(Name("Split")                  \
                              .Device(DEVICE_CPU)        \
                              .TypeConstraint<type>("T") \
                              .HostMemory("split_dim"),  \
                          SplitOpCPU<type>)

TF_CALL_ALL_TYPES(REGISTER_SPLIT);
TF_CALL_float8_e5m2(REGISTER_SPLIT);
TF_CALL_float8_e4m3fn(REGISTER_SPLIT);
TF_CALL_int4(REGISTER_SPLIT);
TF_CALL_uint4(REGISTER_SPLIT);
REGISTER_SPLIT(quint8);

#undef REGISTER_SPLIT

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#define REGISTER_GPU(type)                               \
  REGISTER_KERNEL_BUILDER(Name("Split")                  \
                              .Device(DEVICE_GPU)        \
                              .TypeConstraint<type>("T") \
                              .HostMemory("split_dim"),  \
                          SplitOpGPU<type>)

TF_CALL_GPU_NUMBER_TYPES(REGISTER_GPU);
TF_CALL_COMPLEX_TYPES(REGISTER_GPU);
#undef REGISTER_GPU

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

}  // end namespace tensorflow

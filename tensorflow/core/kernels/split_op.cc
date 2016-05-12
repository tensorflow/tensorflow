/* Copyright 2015 Google Inc. All Rights Reserved.

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

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/kernels/bounds_check.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/kernels/split_lib.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#if GOOGLE_CUDA
#include "tensorflow/core/common_runtime/gpu/gpu_event_mgr.h"
#include "tensorflow/core/kernels/cuda_device_array.h"
#include "tensorflow/core/platform/stream_executor.h"
#endif  // GOOGLE_CUDA

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

template <typename Device, typename T>
class SplitOpBase : public OpKernel {
 public:
  explicit SplitOpBase(OpKernelConstruction* c) : OpKernel(c) {}

  void ComputeEasyCases(OpKernelContext* context, bool* done) {
    const int32 split_dim = context->input(0).flat<int32>()(0);
    const int32 num_split = num_outputs();
    const Tensor& input = context->input(1);
    const TensorShape& input_shape = input.shape();

    OP_REQUIRES(
        context, 0 <= split_dim && split_dim < input_shape.dims(),
        errors::InvalidArgument("0 <= split_dim < number of input dimensions (",
                                input_shape.dims(), "), but got ", split_dim));

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
      const int64 delta = input_shape.dim_size(0) / num_split;
      for (int i = 0; i < num_split; ++i) {
        context->set_output(i, input.Slice(i * delta, (i + 1) * delta));
      }
      *done = true;
      return;
    }
  }

  template <typename IndexType>
  std::tuple<IndexType, IndexType, IndexType> SetDims(
      const TensorShape& input_shape, int32 split_dim) const {
    static_assert(std::is_integral<IndexType>::value,
                  "IndexType must be an integer type");
    int32 prefix_dim_size = 1;
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
    const int32 split_dim = context->input(0).flat<int32>()(0);
    const int32 num_split = Base::num_outputs();
    const Tensor& input = context->input(1);
    const TensorShape& input_shape = input.shape();

    // Android also uses int32 indexing, so check here also.
    OP_REQUIRES(
        context, FastBoundsCheck(input.NumElements(),
                                 std::numeric_limits<Eigen::DenseIndex>::max()),
        errors::InvalidArgument("Split requires input size < ",
                                std::numeric_limits<Eigen::DenseIndex>::max()));

    Eigen::DenseIndex prefix_dim_size;
    Eigen::DenseIndex split_dim_size;
    Eigen::DenseIndex suffix_dim_size;

    std::tie(prefix_dim_size, split_dim_size, suffix_dim_size) =
        Base::template SetDims<Eigen::DenseIndex>(input_shape, split_dim);
    auto input_reshaped =
        input.shaped<T, 3>({prefix_dim_size, split_dim_size, suffix_dim_size});

    const int64 split_dim_output_size = split_dim_size / num_split;
    TensorShape output_shape(input_shape);
    output_shape.set_dim(split_dim, split_dim_output_size);

    Eigen::DSizes<Eigen::DenseIndex, 3> indices{0, 0, 0};
    Eigen::DSizes<Eigen::DenseIndex, 3> sizes{
        prefix_dim_size, split_dim_output_size, suffix_dim_size};

    for (int i = 0; i < num_split; ++i) {
      Tensor* result = nullptr;
      OP_REQUIRES_OK(context,
                     context->allocate_output(i, output_shape, &result));
      if (prefix_dim_size * split_dim_output_size * suffix_dim_size > 0) {
        Eigen::DSizes<Eigen::DenseIndex, 3> slice_indices;
        Eigen::DSizes<Eigen::DenseIndex, 3> slice_sizes;
        for (int j = 0; j < 3; ++j) {
          slice_indices[j] = indices[j];
          slice_sizes[j] = sizes[j];
        }

        auto result_shaped = result->shaped<T, 3>(
            {prefix_dim_size, split_dim_output_size, suffix_dim_size});

        functor::Split<CPUDevice, T>()(context->eigen_device<CPUDevice>(),
                                       result_shaped, input_reshaped,
                                       slice_indices, slice_sizes);
      }
      indices[1] += split_dim_output_size;
    }
  }
};

#if GOOGLE_CUDA

template <typename T>
struct SplitOpGPULaunch {
  void Run(const Eigen::GpuDevice& d, const T* input, int32 prefix_dim_size,
           int32 split_dim_size, int32 suffix_dim_size,
           const CudaDeviceArrayStruct<T*>& output_ptr_data);
};

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
    const int32 split_dim = context->input(0).flat<int32>()(0);
    const int32 num_split = Base::num_outputs();
    const Tensor& input = context->input(1);
    const TensorShape& input_shape = input.shape();
    OP_REQUIRES(context, FastBoundsCheck(input.NumElements(),
                                         std::numeric_limits<int32>::max()),
                errors::InvalidArgument("Split on GPU requires input size "
                                        "< max int32"));

    int32 prefix_dim_size;
    int32 split_dim_size;
    int32 suffix_dim_size;
    std::tie(prefix_dim_size, split_dim_size, suffix_dim_size) =
        Base::template SetDims<int32>(input_shape, split_dim);

    const int32 split_dim_output_size = split_dim_size / num_split;
    TensorShape output_shape(input_shape);
    output_shape.set_dim(split_dim, split_dim_output_size);

    CudaDeviceArrayOnHost<T*> ptrs(context, num_split);
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
#endif  // GOOGLE_CUDA

#define REGISTER_SPLIT(type)                             \
  REGISTER_KERNEL_BUILDER(Name("Split")                  \
                              .Device(DEVICE_CPU)        \
                              .TypeConstraint<type>("T") \
                              .HostMemory("split_dim"),  \
                          SplitOpCPU<type>)

TF_CALL_ALL_TYPES(REGISTER_SPLIT);

#undef REGISTER_SPLIT

#if GOOGLE_CUDA

#define REGISTER_GPU(type)                               \
  REGISTER_KERNEL_BUILDER(Name("Split")                  \
                              .Device(DEVICE_GPU)        \
                              .TypeConstraint<type>("T") \
                              .HostMemory("split_dim"),  \
                          SplitOpGPU<type>)

TF_CALL_GPU_NUMBER_TYPES(REGISTER_GPU);
#undef REGISTER_GPU

#endif  // GOOGLE_CUDA

}  // end namespace tensorflow

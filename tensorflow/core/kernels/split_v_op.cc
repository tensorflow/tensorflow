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

#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#endif  // GOOGLE_CUDA

#include <numeric>

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/kernels/bounds_check.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/kernels/split_lib.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/util/work_sharder.h"
#if GOOGLE_CUDA
#include "tensorflow/core/common_runtime/gpu/gpu_event_mgr.h"
#include "tensorflow/core/kernels/cuda_device_array.h"
#include "tensorflow/core/platform/stream_executor.h"
#endif  // GOOGLE_CUDA

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

template <typename Device, typename T, typename Tlen>
class SplitVOpBase : public OpKernel {
 public:
  explicit SplitVOpBase(OpKernelConstruction* c) : OpKernel(c) {}

  void ComputeEasyCases(OpKernelContext* context, bool* done,
                        std::vector<Tlen>* split_sizes_vec) {
    const int32 num_split = context->num_outputs();
    const Tensor& input = context->input(0);
    const TensorShape& input_shape = input.shape();
    const Tensor& split_tensor = context->input(1);
    const Tensor& split_dim_tensor = context->input(2);

    OP_REQUIRES(context, split_dim_tensor.NumElements() == 1,
                errors::InvalidArgument("split_dim_tensor must have "
                                        "exactly one element."));

    const int32 split_dim_orig = split_dim_tensor.flat<int32>()(0);
    const int32 split_dim =
        split_dim_orig < 0 ? split_dim_orig + input.dims() : split_dim_orig;

    OP_REQUIRES(
        context,
        split_tensor.dims() == 1 && split_tensor.NumElements() == num_split,
        errors::InvalidArgument("size of the split_tensor must be 1-D and have "
                                "the same elements as outputs got ",
                                split_tensor.dims(), " -D and ",
                                split_tensor.NumElements(), " elements"));

    auto split_sizes_d = split_tensor.vec<Tlen>();

    split_sizes_vec->resize(split_sizes_d.size());

    std::copy(split_sizes_d.data(), split_sizes_d.data() + split_sizes_d.size(),
              split_sizes_vec->begin());

    OP_REQUIRES(
        context, num_split > 0,
        errors::InvalidArgument(
            "Number of ways to split should be > 0, but got ", num_split));

    OP_REQUIRES(
        context, 0 <= split_dim && split_dim < input.dims(),
        errors::InvalidArgument("-input rank(-", input.dims(),
                                ") <= split_dim < input rank (", input.dims(),
                                "), but got ", split_dim_orig));

    Tlen input_size_split_dim = input_shape.dim_size(split_dim);

    // Special case 1: num_split == 1. Nothing to do.
    if (num_split == 1) {
      context->set_output(0, context->input(0));
      OP_REQUIRES(
          context, (*split_sizes_vec)[0] == input_size_split_dim,
          errors::InvalidArgument("If there is only one output, it must have "
                                  "the same size as the input. Input size: ",
                                  input_size_split_dim,
                                  " output size: ", (*split_sizes_vec)[0]));
      *done = true;
      return;
    }

    // Determine sizes of output, in case of a -1 input value
    int neg_one_dim = -1;
    Tlen determined_size = 0;
    for (int d = 0; d < split_sizes_vec->size(); ++d) {
      Tlen size = (*split_sizes_vec)[d];

      if (size == -1) {
        OP_REQUIRES(context, neg_one_dim == -1,
                    errors::InvalidArgument("There can only be one -1 in the "
                                            "input."));
        neg_one_dim = d;
      } else {
        determined_size += size;
      }
    }

    OP_REQUIRES(
        context,
        (neg_one_dim == -1 && determined_size == input_size_split_dim) ||
            (neg_one_dim >= 0 && determined_size <= input_size_split_dim),
        errors::InvalidArgument("Determined shape must either match "
                                "input shape along split_dim exactly if "
                                "fully specified, or be less than the size of "
                                "the input along split_dim if not fully "
                                "specified.  Got: ",
                                determined_size));

    if (neg_one_dim >= 0) {
      (*split_sizes_vec)[neg_one_dim] = input_size_split_dim - determined_size;
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
      Tlen start = 0;
      for (int i = 0; i < num_split; ++i) {
        context->set_output(i,
                            input.Slice(start, start + (*split_sizes_vec)[i]));
        start += (*split_sizes_vec)[i];
      }
      *done = true;
      return;
    }
  }

  template <typename IndexType>
  std::tuple<IndexType, IndexType, IndexType> SetDims(
      const TensorShape& input_shape, const int32 split_dim) const {
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

template <typename T, typename Tlen, typename InputReshapedType, int NDims>
class SplitVOpCPUImpl {
 public:
  template <typename MakeSizesType, typename ReshapeResultType>
  void operator()(OpKernelContext* context,
                  const InputReshapedType& input_reshaped,
                  const std::vector<int64>& split_start_points,
                  const TensorShape& input_shape, int32 split_dim,
                  Eigen::DenseIndex prefix_dim_size,
                  Eigen::DenseIndex split_dim_size,
                  Eigen::DenseIndex suffix_dim_size,
                  std::vector<Tlen>& split_sizes_vec,
                  const MakeSizesType& make_sizes,
                  const ReshapeResultType& reshape_result) const {
    Eigen::DSizes<Eigen::DenseIndex, NDims> indices;
    for (int i = 0; i < NDims; ++i) {
      indices[i] = 0;
    }
    const auto num_threads =
        context->device()->tensorflow_cpu_worker_threads()->num_threads;
    // TODO(jewillco): Tune heuristic further.
    const auto input_element_count = input_shape.num_elements();
    const int num_split = split_start_points.size();
    const bool use_parallelism_between_outputs =
        (num_split >= 4 &&
         input_element_count >= std::max(num_threads, num_split) * 4096 &&
         input_element_count < num_split * 180 * 1024);

    auto range_output_func = [&indices, context, &input_shape, split_dim,
                              &split_sizes_vec, &split_start_points,
                              use_parallelism_between_outputs, &input_reshaped,
                              &make_sizes,
                              &reshape_result](int64 start, int64 limit) {
      for (int64 i = start; i < limit; ++i) {
        TensorShape output_shape(input_shape);
        output_shape.set_dim(split_dim, split_sizes_vec[i]);
        Tensor* result = nullptr;
        OP_REQUIRES_OK(context,
                       context->allocate_output(i, output_shape, &result));

        const auto sizes = make_sizes(split_sizes_vec[i]);

        if (sizes.TotalSize() > 0) {
          auto result_shaped = reshape_result(result, split_sizes_vec[i]);

          auto current_indices = indices;
          current_indices[NDims - 2] = split_start_points[i];
          if (use_parallelism_between_outputs) {
            // Use sequential implementation for single output.
            result_shaped = input_reshaped.slice(current_indices, sizes);
          } else {
            // This implementation may be parallel internally.
            functor::Split<CPUDevice, T, NDims>()(
                context->eigen_device<CPUDevice>(), result_shaped,
                input_reshaped, current_indices, sizes);
          }
        }
      }
    };
    if (use_parallelism_between_outputs) {
      // Run in parallel, disabling parallelism in functor.
      Shard(num_split,
            context->device()->tensorflow_cpu_worker_threads()->workers,
            num_split, input_element_count / num_split, range_output_func);
    } else {
      // Run sequentially, but allow internal parallelism in functor.
      range_output_func(0, num_split);
    }
  }
};

template <typename T, typename Tlen>
class SplitVOpCPU : public SplitVOpBase<CPUDevice, T, Tlen> {
 public:
  typedef SplitVOpBase<CPUDevice, T, Tlen> Base;
  explicit SplitVOpCPU(OpKernelConstruction* c) : Base(c) {}

  void Compute(OpKernelContext* context) override {
    bool done = false;
    std::vector<Tlen> split_sizes_vec;
    Base::ComputeEasyCases(context, &done, &split_sizes_vec);
    if (!context->status().ok() || done) {
      return;
    }
    const int32 num_split = Base::num_outputs();
    const Tensor& input = context->input(0);
    const TensorShape& input_shape = input.shape();
    const int32 split_dim_orig = context->input(2).flat<int32>()(0);
    const int32 split_dim =
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
    std::vector<int64> split_start_points(num_split);
    for (int i = 0; i < num_split; ++i) {
      if (i == 0) {
        split_start_points[i] = 0;
      } else {
        split_start_points[i] =
            split_start_points[i - 1] + split_sizes_vec[i - 1];
      }
    }

    if (prefix_dim_size == 1) {
      auto input_reshaped =
          input.shaped<T, 2>({split_dim_size, suffix_dim_size});
      auto make_sizes = [&](Eigen::DenseIndex split_size) {
        return Eigen::DSizes<Eigen::DenseIndex, 2>{split_size, suffix_dim_size};
      };
      auto reshape_result = [&](Tensor* result, Tlen split_size) {
        return result->shaped<T, 2>({split_size, suffix_dim_size});
      };
      SplitVOpCPUImpl<T, Tlen, decltype(input_reshaped), 2>{}(
          context, input_reshaped, split_start_points, input_shape, split_dim,
          prefix_dim_size, split_dim_size, suffix_dim_size, split_sizes_vec,
          make_sizes, reshape_result);
    } else {
      auto input_reshaped = input.shaped<T, 3>(
          {prefix_dim_size, split_dim_size, suffix_dim_size});
      auto make_sizes = [&](Eigen::DenseIndex split_size) {
        return Eigen::DSizes<Eigen::DenseIndex, 3>{prefix_dim_size, split_size,
                                                   suffix_dim_size};
      };
      auto reshape_result = [&](Tensor* result, Tlen split_size) {
        return result->shaped<T, 3>(
            {prefix_dim_size, split_size, suffix_dim_size});
      };
      SplitVOpCPUImpl<T, Tlen, decltype(input_reshaped), 3>{}(
          context, input_reshaped, split_start_points, input_shape, split_dim,
          prefix_dim_size, split_dim_size, suffix_dim_size, split_sizes_vec,
          make_sizes, reshape_result);
    }
  }
};

#if GOOGLE_CUDA

template <typename T, typename IntType>
struct SplitVOpGPULaunch {
  void Run(const Eigen::GpuDevice& d, bool fixed, const T* input,
           int total_cols, int total_rows,
           const CudaDeviceArrayStruct<IntType>& output_scan,
           const CudaDeviceArrayStruct<T*>& output_ptr_data);
};

// Partial specialization for GPU
template <typename T, typename Tlen>
class SplitVOpGPU : public SplitVOpBase<GPUDevice, T, Tlen> {
 public:
  typedef SplitVOpBase<GPUDevice, T, Tlen> Base;
  explicit SplitVOpGPU(OpKernelConstruction* c) : Base(c) {}

  void Compute(OpKernelContext* context) override {
    bool done = false;
    std::vector<Tlen> split_sizes_vec;
    Base::ComputeEasyCases(context, &done, &split_sizes_vec);
    if (!context->status().ok() || done) {
      return;
    }
    const int32 num_split = Base::num_outputs();
    const Tensor& input = context->input(0);
    const TensorShape& input_shape = input.shape();
    const int32 split_dim_orig = context->input(2).flat<int32>()(0);
    const int32 split_dim =
        split_dim_orig < 0 ? split_dim_orig + input.dims() : split_dim_orig;
    OP_REQUIRES(
        context,
        FastBoundsCheck(input.NumElements(), std::numeric_limits<int32>::max()),
        errors::InvalidArgument("Split on GPU requires input size "
                                "< max int32"));

    int32 prefix_dim_size;
    int32 split_dim_size;
    int32 suffix_dim_size;
    std::tie(prefix_dim_size, split_dim_size, suffix_dim_size) =
        Base::template SetDims<int32>(input_shape, split_dim);

    // use the same approach as concat (see documentation there)
    // reshape to 2D

    if (num_split > 16) {
      CudaDeviceArrayOnHost<T*> ptrs(context, num_split);
      OP_REQUIRES_OK(context, ptrs.Init());

      CudaDeviceArrayOnHost<Tlen> offsets(context, num_split + 1);
      OP_REQUIRES_OK(context, offsets.Init());

      Tlen offset = 0;
      int entry = split_sizes_vec[0];
      bool fixed_size =
          std::all_of(split_sizes_vec.begin(), split_sizes_vec.end(),
                      [&entry](int n) { return n == entry; });

      for (int i = 0; i < num_split; ++i) {
        TensorShape output_shape(input_shape);
        output_shape.set_dim(split_dim, split_sizes_vec[i]);
        Tensor* result = nullptr;
        OP_REQUIRES_OK(context,
                       context->allocate_output(i, output_shape, &result));
        ptrs.Set(i, result->flat<T>().data());
        offsets.Set(i, offset);
        offset += split_sizes_vec[i] * suffix_dim_size;
      }
      offsets.Set(num_split, offset);
      OP_REQUIRES_OK(context, ptrs.Finalize());
      OP_REQUIRES_OK(context, offsets.Finalize());

      if (input.NumElements() > 0) {
        SplitVOpGPULaunch<T, Tlen>().Run(
            context->eigen_device<GPUDevice>(), fixed_size,
            input.flat<T>().data(), prefix_dim_size,
            input.NumElements() / prefix_dim_size, offsets.data(), ptrs.data());
        OP_REQUIRES(
            context, context->op_device_context()->stream()->ok(),
            errors::Internal("Launch of gpu kernel for SplitVOp failed"));
      }
    } else {
      Eigen::DenseIndex prefix_dim_size;
      Eigen::DenseIndex split_dim_size;
      Eigen::DenseIndex suffix_dim_size;

      std::tie(prefix_dim_size, split_dim_size, suffix_dim_size) =
          Base::template SetDims<Eigen::DenseIndex>(input_shape, split_dim);
      auto input_reshaped = input.shaped<T, 2>(
          {prefix_dim_size, split_dim_size * suffix_dim_size});

      Eigen::DSizes<Eigen::DenseIndex, 2> indices{0, 0};

      for (int i = 0; i < num_split; ++i) {
        TensorShape output_shape(input_shape);
        output_shape.set_dim(split_dim, split_sizes_vec[i]);
        Tensor* result = nullptr;
        OP_REQUIRES_OK(context,
                       context->allocate_output(i, output_shape, &result));

        Eigen::DSizes<Eigen::DenseIndex, 2> sizes{
            prefix_dim_size, split_sizes_vec[i] * suffix_dim_size};

        if (sizes.TotalSize() > 0) {
          auto result_shaped = result->shaped<T, 2>(
              {prefix_dim_size, split_sizes_vec[i] * suffix_dim_size});

          functor::SplitCustom<GPUDevice, T>()(
              context->eigen_device<GPUDevice>(), result_shaped, input_reshaped,
              indices, sizes);
        }
        indices[1] += split_sizes_vec[i] * suffix_dim_size;
      }
    }
  }
};
#endif  // GOOGLE_CUDA

#define REGISTER_SPLIT(type, len_type)                          \
  REGISTER_KERNEL_BUILDER(Name("SplitV")                        \
                              .Device(DEVICE_CPU)               \
                              .TypeConstraint<len_type>("Tlen") \
                              .TypeConstraint<type>("T")        \
                              .HostMemory("size_splits")        \
                              .HostMemory("split_dim"),         \
                          SplitVOpCPU<type, len_type>);

#define REGISTER_SPLIT_LEN(type) \
  REGISTER_SPLIT(type, int32);   \
  REGISTER_SPLIT(type, int64);

TF_CALL_ALL_TYPES(REGISTER_SPLIT_LEN);

#undef REGISTER_SPLIT_LEN
#undef REGISTER_SPLIT

#if GOOGLE_CUDA

#define REGISTER_GPU(type, len_type)                            \
  REGISTER_KERNEL_BUILDER(Name("SplitV")                        \
                              .Device(DEVICE_GPU)               \
                              .TypeConstraint<len_type>("Tlen") \
                              .TypeConstraint<type>("T")        \
                              .HostMemory("size_splits")        \
                              .HostMemory("split_dim"),         \
                          SplitVOpGPU<type, len_type>);

#define REGISTER_GPU_LEN(type) \
  REGISTER_GPU(type, int32);   \
  REGISTER_GPU(type, int64);

TF_CALL_GPU_NUMBER_TYPES(REGISTER_GPU_LEN);
TF_CALL_complex64(REGISTER_GPU_LEN);
TF_CALL_complex128(REGISTER_GPU_LEN);
REGISTER_GPU_LEN(bfloat16);
#undef REGISTER_GPU_LEN
#undef REGISTER_GPU

// special GPU kernel for int32

#define REGISTER_GPU_int32(len_type)                            \
  REGISTER_KERNEL_BUILDER(Name("SplitV")                        \
                              .Device(DEVICE_GPU)               \
                              .TypeConstraint<int32>("T")       \
                              .TypeConstraint<len_type>("Tlen") \
                              .HostMemory("size_splits")        \
                              .HostMemory("split_dim")          \
                              .HostMemory("value")              \
                              .HostMemory("output"),            \
                          SplitVOpCPU<int32, len_type>);

REGISTER_GPU_int32(int32);
REGISTER_GPU_int32(int64);

#undef REGISTER_GPU_int32

#endif  // GOOGLE_CUDA

}  // end namespace tensorflow

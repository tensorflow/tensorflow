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

// See docs in ../ops/array_ops.cc
#define EIGEN_USE_THREADS

#if !defined(PLUGGABLE_DEVICE_SUPPORTED_MACOS) && defined(__APPLE__) && \
    !defined(ANDROID) && !defined(__ANDROID__) && !TARGET_OS_IOS
#define PLUGGABLE_DEVICE_SUPPORTED_MACOS 1
#endif

#include "tensorflow/core/kernels/reverse_op.h"

#include <memory>

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/bounds_check.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/type_traits.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/util/work_sharder.h"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

namespace {

// Reverse rows (middle dimension) of a three dimensional tensor.
// NUM_CHANNELS can be <= 0 to compute it dynamically from <input>
// Otherwise, it must equal input.dim_size(2) and is used as a compile-time
// constant.
template <typename T, int NUM_CHANNELS>
void ReverseRows(OpKernelContext* context, const Tensor& input,
                 Tensor* result) {
  auto work = [&input, result](int64_t start, int64_t end) {
    const int64_t inner_size =
        NUM_CHANNELS > 0 ? NUM_CHANNELS : input.dim_size(2);
    const int64_t middle_size = input.dim_size(1);
    const int64_t row_size = inner_size * middle_size;
    DCHECK_EQ(input.dim_size(2), inner_size);

    const T* in_ptr = input.bit_casted_tensor<T, 3>().data();
    T* out_ptr = result->bit_casted_tensor<T, 3>().data();

    in_ptr += start * row_size;
    out_ptr += start * row_size;

    for (int outer_dim = start; outer_dim < end; ++outer_dim) {
      out_ptr += row_size;
      int remaining = middle_size;
      while (remaining > 0) {
        out_ptr -= inner_size;
        memcpy(out_ptr, in_ptr, inner_size * sizeof(T));
        in_ptr += inner_size;
        --remaining;
      }

      out_ptr += row_size;
    }
  };

  // Shard across outer dimension.
  const int64_t N = input.dim_size(0);
  const int64_t cost_per_unit = input.NumElements() / N;
  auto worker_threads = context->device()->tensorflow_cpu_worker_threads();
  Shard(worker_threads->num_threads, worker_threads->workers, N, cost_per_unit,
        std::move(work));
}

template <typename T>
struct data_type_can_memcpy {
  static constexpr bool value =
      std::is_same<T, uint8>::value || std::is_same<T, int8>::value ||
      std::is_same<T, bool>::value || std::is_same<T, uint16>::value ||
      std::is_same<T, int16>::value || std::is_same<T, Eigen::half>::value ||
      std::is_same<T, Eigen::bfloat16>::value ||
      std::is_same<T, int32>::value || std::is_same<T, float>::value ||
      std::is_same<T, int64_t>::value || std::is_same<T, double>::value ||
      std::is_same<T, complex64>::value || std::is_same<T, complex128>::value;
};

template <typename T, int NUM_CHANNELS>
typename std::enable_if<data_type_can_memcpy<T>::value>::type
DoHandleReverseCase(OpKernelContext* context, const Tensor& input,
                    Tensor* result) {
  if (sizeof(T) == 1) {
    static_assert(sizeof(uint8) == 1, "uint8 must be 1 byte.");
    ReverseRows<uint8, NUM_CHANNELS>(context, input, result);
  } else if (sizeof(T) == 2) {
    static_assert(sizeof(uint16) == 2, "uint16 must be 2 bytes");
    ReverseRows<uint16, NUM_CHANNELS>(context, input, result);
  } else if (sizeof(T) == 4) {
    static_assert(sizeof(uint32) == 4, "uint32 must be 4 bytes");
    ReverseRows<uint32, NUM_CHANNELS>(context, input, result);
  } else if (sizeof(T) == 8) {
    static_assert(sizeof(uint64) == 8, "uint64 must be 8 bytes");
    ReverseRows<uint64, NUM_CHANNELS>(context, input, result);
  } else if (sizeof(T) == 16) {
    static_assert(sizeof(complex128) == 16, "complex128 must be 16 bytes");
    ReverseRows<complex128, NUM_CHANNELS>(context, input, result);
  } else {
    context->CtxFailure(errors::InvalidArgument(DataTypeString(input.dtype()),
                                                " has unexpected size of ",
                                                sizeof(T), " bytes"));
  }
}

template <typename T, int NUM_CHANNELS>
typename std::enable_if<!data_type_can_memcpy<T>::value>::type
DoHandleReverseCase(OpKernelContext* context, const Tensor& input,
                    Tensor* result) {}

}  // namespace

template <typename Device, typename T, int NDIMS>
void HandleReverseCase(OpKernelContext* context,
                       typename TTypes<bool, 1>::ConstTensor dims,
                       Tensor* result) {
  const Tensor& input = context->input(0);

  // Use optimized reverse if possible.
  if (NDIMS == 3 && std::is_same<Device, CPUDevice>::value &&
      data_type_can_memcpy<T>::value && (!dims(0) && dims(1) && !dims(2))) {
    if (input.dim_size(2) == 3) {
      DoHandleReverseCase<T, 3>(context, input, result);
    } else {
      DoHandleReverseCase<T, -1>(context, input, result);
    }
    return;
  }
  typename Eigen::array<bool, NDIMS> axes_di;
  for (int i = 0; i < NDIMS; i++) {
    axes_di[i] = dims(i);
  }
  functor::Reverse<Device, T, NDIMS>()(context->eigen_device<Device>(),
                                       input.tensor<T, NDIMS>(), axes_di,
                                       result->tensor<T, NDIMS>());
}

template <typename Device, typename T>
class ReverseOp : public OpKernel {
 public:
  explicit ReverseOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    const Tensor& input = context->input(0);
    // If input is provided, check to make sure the first dimension is valid.
    if (input.dims() > 0) {
      OP_REQUIRES(
          context, input.dim_size(0) != 0,
          errors::InvalidArgument("Invalid input first dimension. Found 0."));
    }
    const Tensor& dims = context->input(1);

    if (TensorShapeUtils::IsScalar(input.shape())) {
      context->set_output(0, input);
    } else {
      const int input_dims = input.dims();
      OP_REQUIRES(context, TensorShapeUtils::IsVector(dims.shape()),
                  errors::InvalidArgument("'dims' must be 1-dimension, not ",
                                          dims.dims()));

      OP_REQUIRES(
          context, input_dims == dims.dim_size(0),
          errors::InvalidArgument(
              "'dims' must have the same number of values as 'input' has "
              "dimensions. 'input' has ",
              input_dims, "'dims' has ", dims.dim_size(0), " values"));
      OP_REQUIRES(context, input_dims <= 8,
                  errors::Unimplemented(
                      "reverse is not implemented for tensors of rank > 8."));

      Tensor* output = nullptr;
      OP_REQUIRES_OK(context,
                     context->allocate_output(0, input.shape(), &output));

#define HANDLE_REVERSE(NDIMS)                                               \
  case NDIMS:                                                               \
    HandleReverseCase<Device, T, NDIMS>(context, dims.vec<bool>(), output); \
    return;

      switch (input_dims) {
        HANDLE_REVERSE(0);
        HANDLE_REVERSE(1);
        HANDLE_REVERSE(2);
        HANDLE_REVERSE(3);
        HANDLE_REVERSE(4);
        HANDLE_REVERSE(5);
        HANDLE_REVERSE(6);
        HANDLE_REVERSE(7);
        HANDLE_REVERSE(8);
      }
#undef HANDLE_REVERSE
    }
  }
};

template <typename Device, typename T, int NDIMS>
void HandleReverseV2Case(OpKernelContext* context,
                         const gtl::ArraySlice<bool> axes, Tensor* result) {
  const Tensor& input = context->input(0);

  // Use optimized reverse if possible.
  if (NDIMS == 3 && std::is_same<Device, CPUDevice>::value &&
      data_type_can_memcpy<T>::value && (!axes[0] && axes[1] && !axes[2])) {
    if (input.dim_size(2) == 3) {
      DoHandleReverseCase<T, 3>(context, input, result);
    } else {
      DoHandleReverseCase<T, -1>(context, input, result);
    }
    return;
  }

  typename Eigen::array<bool, NDIMS> axes_di;
  for (int i = 0; i < NDIMS; i++) {
    axes_di[i] = axes[i];
  }
  functor::Reverse<Device, T, NDIMS>()(context->eigen_device<Device>(),
                                       input.tensor<T, NDIMS>(), axes_di,
                                       result->tensor<T, NDIMS>());
}

template <typename Device, typename T, typename Tidx>
class ReverseV2Op : public OpKernel {
 public:
  explicit ReverseV2Op(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    const Tensor& input = context->input(0);
    const Tensor& sparse_dims = context->input(1);

    if (TensorShapeUtils::IsScalar(input.shape()) || input.NumElements() == 0) {
      context->set_output(0, input);
    } else {
      const int input_dims = input.dims();
      const TensorShape& sparse_dims_shape = sparse_dims.shape();
      const auto& axes_sparse_flat = sparse_dims.flat<Tidx>();

      OP_REQUIRES(context, TensorShapeUtils::IsVector(sparse_dims_shape),
                  errors::InvalidArgument("'dims' must be 1-dimension, not ",
                                          sparse_dims.dims()));
      gtl::InlinedVector<bool, 8> axes_dense(input_dims, false);
      for (int dummy = 0; dummy < axes_sparse_flat.size(); dummy++) {
        Tidx axis = internal::SubtleMustCopy<Tidx>(axes_sparse_flat(dummy));
        Tidx canonical_axis = axis < 0 ? input_dims + axis : axis;
        OP_REQUIRES(context, canonical_axis >= 0 && canonical_axis < input_dims,
                    errors::InvalidArgument("'axis'[", dummy, "] = ", axis,
                                            " is out of valid range [", 0, ", ",
                                            input_dims - 1));
        OP_REQUIRES(context, !axes_dense[canonical_axis],
                    errors::InvalidArgument("axis ", canonical_axis,
                                            " specified more than once."));
        axes_dense[canonical_axis] = true;
      }

      OP_REQUIRES(context, input_dims <= 8,
                  errors::Unimplemented(
                      "reverse is not implemented for tensors of rank > 8."));

      Tensor* output = nullptr;
      OP_REQUIRES_OK(context,
                     context->allocate_output(0, input.shape(), &output));

      // TODO(cwhipkey): we can do dimension folding to reduce, e.g., a reverse
      // of a single dimension to the dims=3 or dims=2 case, regardless of the
      // number of dimensions in the tensor. This would let some ops use faster
      // lower-dimension code (and use optimized versions).

#define HANDLE_REVERSE(NDIMS)                                           \
  case NDIMS:                                                           \
    HandleReverseV2Case<Device, T, NDIMS>(context, axes_dense, output); \
    return;

      switch (input_dims) {
        HANDLE_REVERSE(0);
        HANDLE_REVERSE(1);
        HANDLE_REVERSE(2);
        HANDLE_REVERSE(3);
        HANDLE_REVERSE(4);
        HANDLE_REVERSE(5);
        HANDLE_REVERSE(6);
        HANDLE_REVERSE(7);
        HANDLE_REVERSE(8);
      }
#undef HANDLE_REVERSE
    }
  }
};

#define REGISTER_KERNELS(T)                                    \
  REGISTER_KERNEL_BUILDER(Name("Reverse")                      \
                              .Device(DEVICE_CPU)              \
                              .TypeConstraint<T>("T")          \
                              .HostMemory("dims"),             \
                          ReverseOp<CPUDevice, T>)             \
  REGISTER_KERNEL_BUILDER(Name("ReverseV2")                    \
                              .Device(DEVICE_CPU)              \
                              .TypeConstraint<T>("T")          \
                              .TypeConstraint<int32>("Tidx")   \
                              .HostMemory("axis"),             \
                          ReverseV2Op<CPUDevice, T, int32>)    \
  REGISTER_KERNEL_BUILDER(Name("ReverseV2")                    \
                              .Device(DEVICE_CPU)              \
                              .TypeConstraint<T>("T")          \
                              .TypeConstraint<int64_t>("Tidx") \
                              .HostMemory("axis"),             \
                          ReverseV2Op<CPUDevice, T, int64>)
TF_CALL_POD_TYPES(REGISTER_KERNELS);
TF_CALL_tstring(REGISTER_KERNELS);
#undef REGISTER_KERNELS

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM

// Forward declarations of the function specializations for GPU (to prevent
// building the GPU versions here, they will be built compiling _gpu.cu.cc).
namespace functor {
#define DECLARE_GPU_SPEC_DIM(T, DIM)                                  \
  template <>                                                         \
  void Reverse<GPUDevice, T, DIM>::operator()(                        \
      const GPUDevice& d, typename TTypes<T, DIM>::ConstTensor input, \
      const Eigen::array<bool, DIM>& reverse_dims,                    \
      typename TTypes<T, DIM>::Tensor output);                        \
  extern template struct Reverse<GPUDevice, T, DIM>;
#define DECLARE_GPU_SPEC(T)  \
  DECLARE_GPU_SPEC_DIM(T, 0) \
  DECLARE_GPU_SPEC_DIM(T, 1) \
  DECLARE_GPU_SPEC_DIM(T, 2) \
  DECLARE_GPU_SPEC_DIM(T, 3) \
  DECLARE_GPU_SPEC_DIM(T, 4) \
  DECLARE_GPU_SPEC_DIM(T, 5) \
  DECLARE_GPU_SPEC_DIM(T, 6) \
  DECLARE_GPU_SPEC_DIM(T, 7) \
  DECLARE_GPU_SPEC_DIM(T, 8)

TF_CALL_uint8(DECLARE_GPU_SPEC);
TF_CALL_int8(DECLARE_GPU_SPEC);
TF_CALL_GPU_ALL_TYPES(DECLARE_GPU_SPEC);
#undef DECLARE_GPU_SPEC
#undef DECLARE_GPU_SPEC_DIM
}  // namespace functor

// Registration of the GPU implementations.
#define REGISTER_GPU_KERNELS(T)                                \
  REGISTER_KERNEL_BUILDER(Name("Reverse")                      \
                              .Device(DEVICE_GPU)              \
                              .TypeConstraint<T>("T")          \
                              .HostMemory("dims"),             \
                          ReverseOp<GPUDevice, T>)             \
  REGISTER_KERNEL_BUILDER(Name("ReverseV2")                    \
                              .Device(DEVICE_GPU)              \
                              .TypeConstraint<T>("T")          \
                              .TypeConstraint<int32>("Tidx")   \
                              .HostMemory("axis"),             \
                          ReverseV2Op<GPUDevice, T, int32>)    \
  REGISTER_KERNEL_BUILDER(Name("ReverseV2")                    \
                              .Device(DEVICE_GPU)              \
                              .TypeConstraint<T>("T")          \
                              .TypeConstraint<int64_t>("Tidx") \
                              .HostMemory("axis"),             \
                          ReverseV2Op<GPUDevice, T, int64>)
TF_CALL_uint8(REGISTER_GPU_KERNELS);
TF_CALL_int8(REGISTER_GPU_KERNELS);
TF_CALL_GPU_ALL_TYPES(REGISTER_GPU_KERNELS);
#undef REGISTER_GPU_KERNEL

// A special GPU kernel for int32.
// TODO(b/25387198): Also enable int32 in device memory. This kernel
// registration requires all int32 inputs and outputs to be in host memory.
REGISTER_KERNEL_BUILDER(Name("Reverse")
                            .Device(DEVICE_GPU)
                            .TypeConstraint<int32>("T")
                            .HostMemory("tensor")
                            .HostMemory("dims")
                            .HostMemory("output"),
                        ReverseOp<CPUDevice, int32>);
REGISTER_KERNEL_BUILDER(Name("ReverseV2")
                            .Device(DEVICE_GPU)
                            .TypeConstraint<int32>("T")
                            .TypeConstraint<int32>("Tidx")
                            .HostMemory("tensor")
                            .HostMemory("axis")
                            .HostMemory("output"),
                        ReverseV2Op<CPUDevice, int32, int32>);
REGISTER_KERNEL_BUILDER(Name("ReverseV2")
                            .Device(DEVICE_GPU)
                            .TypeConstraint<int32>("T")
                            .TypeConstraint<int64_t>("Tidx")
                            .HostMemory("tensor")
                            .HostMemory("axis")
                            .HostMemory("output"),
                        ReverseV2Op<CPUDevice, int32, int64>);
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#if defined(PLUGGABLE_DEVICE_SUPPORTED_MACOS)
#define REGISTER_DEFAULT_KERNELS(T)                         \
REGISTER_KERNEL_BUILDER(Name("Reverse")                     \
                            .Device(DEVICE_DEFAULT)         \
                            .TypeConstraint<T>("T")         \
                            .HostMemory("tensor")           \
                            .HostMemory("dims")             \
                            .HostMemory("output"),          \
                        ReverseOp<CPUDevice, T>)            \
REGISTER_KERNEL_BUILDER(Name("ReverseV2")                   \
                            .Device(DEVICE_DEFAULT)         \
                            .TypeConstraint<T>("T")         \
                            .TypeConstraint<int32>("Tidx")  \
                            .HostMemory("tensor")           \
                            .HostMemory("axis")             \
                            .HostMemory("output"),          \
                        ReverseV2Op<CPUDevice, T, int32>)   \
REGISTER_KERNEL_BUILDER(Name("ReverseV2")                   \
                            .Device(DEVICE_DEFAULT)         \
                            .TypeConstraint<T>("T")         \
                            .TypeConstraint<int64>("Tidx")  \
                            .HostMemory("tensor")           \
                            .HostMemory("axis")             \
                            .HostMemory("output"),          \
                        ReverseV2Op<CPUDevice, T, int64>)
TF_CALL_uint8(REGISTER_DEFAULT_KERNELS);
TF_CALL_int8(REGISTER_DEFAULT_KERNELS);
TF_CALL_int16(REGISTER_DEFAULT_KERNELS);
TF_CALL_uint32(REGISTER_DEFAULT_KERNELS);
TF_CALL_int32(REGISTER_DEFAULT_KERNELS);
TF_CALL_GPU_ALL_TYPES(REGISTER_DEFAULT_KERNELS);
#undef REGISTER_DEFAULT_KERNELS
#endif

}  // namespace tensorflow

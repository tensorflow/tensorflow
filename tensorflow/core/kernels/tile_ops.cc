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

#ifdef GOOGLE_CUDA
#define EIGEN_USE_GPU
#endif  // GOOGLE_CUDA

#include <vector>

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

#include "tensorflow/core/framework/numeric_op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/type_index.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;
#ifdef TENSORFLOW_USE_SYCL
typedef Eigen::SyclDevice SYCLDevice;
#endif  // TENSORFLOW_USE_SYCL

// Forward declarations of functors that will be defined in tile_ops_impl.h
namespace functor {
template <typename Device, typename T, typename Tmultiple>
struct Tile {
  void operator()(const Device& d, Tensor* out, const Tensor& in,
                  const gtl::ArraySlice<Tmultiple> broadcast_array) const;
};

template <typename Device, typename T, int NDIM>
struct TileGrad {
  void operator()(const Device& d, typename TTypes<T, NDIM>::Tensor out,
                  typename TTypes<T, NDIM>::ConstTensor in,
                  const Eigen::DSizes<Eigen::DenseIndex, NDIM>& indices,
                  const Eigen::DSizes<Eigen::DenseIndex, NDIM>& sizes,
                  bool first) const;
};

template <typename Device, typename T>
struct TileGrad<Device, T, 0> {
  void operator()(const Device& d, typename TTypes<T, 0>::Tensor out,
                  typename TTypes<T, 0>::ConstTensor in,
                  const Eigen::DSizes<Eigen::DenseIndex, 0>&,
                  const Eigen::DSizes<Eigen::DenseIndex, 0>&, bool first) const;
};

template <typename Device, typename T, int NDIM, int REDUCEDNDIM>
struct ReduceAndReshape {
  void operator()(
      const Device& d, typename TTypes<T, NDIM>::Tensor out,
      typename TTypes<T, NDIM>::ConstTensor in,
      const Eigen::DSizes<Eigen::DenseIndex, REDUCEDNDIM>& reduce_dim,
      const Eigen::DSizes<Eigen::DenseIndex, NDIM>& reshape_dim) const;
};
}  // namespace functor

// --------------------------------------------------------------------------
template <typename Device, typename Tmultiples>
class TileOp : public OpKernel {
 public:
  explicit TileOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    const Tensor& input = context->input(0);
    const Tensor& multiples = context->input(1);

    OP_REQUIRES(
        context, IsLegacyVector(multiples.shape()),
        errors::InvalidArgument("Expected multiples to be 1-D, but got shape ",
                                multiples.shape().DebugString()));
    OP_REQUIRES(context, input.dims() == multiples.NumElements(),
                errors::InvalidArgument(
                    "Expected multiples argument to be a vector of length ",
                    input.dims(), " but got length ", multiples.dim_size(0)));
    const int input_dims = input.dims();

    // Eigen doesn't support scalars on the GPU, so handle 0-D specially
    if (input_dims == 0) {
      context->set_output(0, input);
      return;
    }

    const gtl::ArraySlice<Tmultiples> multiples_array(
        multiples.flat<Tmultiples>().data(), input_dims);
    TensorShape output_shape;
    for (int i = 0; i < input_dims; ++i) {
      OP_REQUIRES(
          context, multiples_array[i] >= 0,
          errors::InvalidArgument("Expected multiples[", i, "] >= 0, but got ",
                                  multiples_array[i]));
      output_shape.AddDim(input.dim_size(i) * multiples_array[i]);
    }
    if (output_shape == input.shape()) {
      context->set_output(0, input);
      return;
    }
    Tensor* result = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &result));

    // If there's no output, there's nothing to do.
    if (output_shape.num_elements() == 0) return;

#define HANDLE_TYPE(DT)                               \
  if (context->input(0).dtype() == DT) {              \
    HandleCase<DT>(context, multiples_array, result); \
    return;                                           \
  }

#define HANDLE_TYPE_NAME(T) HANDLE_TYPE(DataTypeToEnum<T>::value)

    // Invoke macro using TF_CALL_* so type-filtering for platform applies.
    TF_CALL_bool(HANDLE_TYPE_NAME);
    TF_CALL_float(HANDLE_TYPE_NAME);
    TF_CALL_double(HANDLE_TYPE_NAME);
    TF_CALL_uint8(HANDLE_TYPE_NAME);
    TF_CALL_int32(HANDLE_TYPE_NAME);
    TF_CALL_int16(HANDLE_TYPE_NAME);
    TF_CALL_int64(HANDLE_TYPE_NAME);
    TF_CALL_half(HANDLE_TYPE_NAME);
    TF_CALL_string(HANDLE_TYPE_NAME);  // when DEVICE=CPUDevice.
    TF_CALL_complex64(HANDLE_TYPE_NAME);
    TF_CALL_complex128(HANDLE_TYPE_NAME);

#undef HANDLE_TYPE_NAME
#undef HANDLE_TYPE

    OP_REQUIRES(context, false,
                errors::Unimplemented(
                    "TileOp : Unhandled input dimensions, DT : ",
                    context->input(0).dtype(), ", dims : ", input_dims));
  }

 private:
  template <DataType DT>
  void HandleCaseImpl(OpKernelContext* context,
                      const gtl::ArraySlice<Tmultiples>& multiples_array,
                      Tensor* result) {
    typedef typename EnumToDataType<DT>::Type T;
    functor::Tile<Device, T, Tmultiples>()(context->eigen_device<Device>(),
                                           result, context->input(0),
                                           multiples_array);
  }

  template <DataType DT>
  void HandleCase(OpKernelContext* context,
                  const gtl::ArraySlice<Tmultiples>& multiples_array,
                  Tensor* result);

  TF_DISALLOW_COPY_AND_ASSIGN(TileOp);
};

template <typename Device, typename Tmultiples>
template <DataType DT>
inline void TileOp<Device, Tmultiples>::HandleCase(
    OpKernelContext* context,
    const gtl::ArraySlice<Tmultiples>& multiples_array, Tensor* result) {
  // TODO(vrv): print out the device name if useful. Currently disabled to avoid
  // having to use RTTI.
  LOG(FATAL) << "TileOp: Invalid combination of Device, DT: "
             // << typeid(Device).name() << ", "
             << DataTypeString(DT);
}

#define HANDLE_CASE(device, dtype, Tmultiples)                              \
  template <>                                                               \
  template <>                                                               \
  void TileOp<device, Tmultiples>::HandleCase<dtype>(                       \
      OpKernelContext * context,                                            \
      const gtl::ArraySlice<Tmultiples>& multiples_array, Tensor* result) { \
    HandleCaseImpl<dtype>(context, multiples_array, result);                \
  }

#define HANDLE_TYPE_NAME_CPU(T)                            \
  HANDLE_CASE(CPUDevice, DataTypeToEnum<T>::value, int32); \
  HANDLE_CASE(CPUDevice, DataTypeToEnum<T>::value, int64);

#define HANDLE_TYPE_NAME_GPU(T)                            \
  HANDLE_CASE(GPUDevice, DataTypeToEnum<T>::value, int32); \
  HANDLE_CASE(GPUDevice, DataTypeToEnum<T>::value, int64);

#ifdef TENSORFLOW_USE_SYCL
#define HANDLE_TYPE_NAME_SYCL(T)                            \
  HANDLE_CASE(SYCLDevice, DataTypeToEnum<T>::value, int32); \
  HANDLE_CASE(SYCLDevice, DataTypeToEnum<T>::value, int64);
#endif  // TENSORFLOW_USE_SYCL

TF_CALL_bool(HANDLE_TYPE_NAME_CPU);
TF_CALL_float(HANDLE_TYPE_NAME_CPU);
TF_CALL_double(HANDLE_TYPE_NAME_CPU);
TF_CALL_uint8(HANDLE_TYPE_NAME_CPU);
TF_CALL_int32(HANDLE_TYPE_NAME_CPU);
TF_CALL_int16(HANDLE_TYPE_NAME_CPU);
TF_CALL_int64(HANDLE_TYPE_NAME_CPU);
TF_CALL_half(HANDLE_TYPE_NAME_CPU);
TF_CALL_complex64(HANDLE_TYPE_NAME_CPU);
TF_CALL_complex128(HANDLE_TYPE_NAME_CPU);
TF_CALL_string(HANDLE_TYPE_NAME_CPU);

#if GOOGLE_CUDA
TF_CALL_float(HANDLE_TYPE_NAME_GPU);
TF_CALL_double(HANDLE_TYPE_NAME_GPU);
TF_CALL_int16(HANDLE_TYPE_NAME_GPU);
TF_CALL_int32(HANDLE_TYPE_NAME_GPU);
TF_CALL_int64(HANDLE_TYPE_NAME_GPU);
TF_CALL_half(HANDLE_TYPE_NAME_GPU);
TF_CALL_complex64(HANDLE_TYPE_NAME_GPU);
TF_CALL_complex128(HANDLE_TYPE_NAME_GPU);
#endif  // GOOGLE_CUDA

#ifdef TENSORFLOW_USE_SYCL
TF_CALL_float(HANDLE_TYPE_NAME_SYCL);
TF_CALL_double(HANDLE_TYPE_NAME_SYCL);
TF_CALL_int16(HANDLE_TYPE_NAME_SYCL);
TF_CALL_int32(HANDLE_TYPE_NAME_SYCL);
TF_CALL_int64(HANDLE_TYPE_NAME_SYCL);
#endif  // TENSORFLOW_USE_SYCL

#undef HANDLE_TYPE_NAME_CPU
#undef HANDLE_TYPE_NAME_GPU
#ifdef TENSORFLOW_USE_SYCL
#undef HANDLE_TYPE_NAME_SYCL
#endif  // TENSORFLOW_USE_SYCL
#undef HANDLE_CASE

// --------------------------------------------------------------------------
template <typename Device, typename Tmultiples>
class TileGradientOp : public OpKernel {
 public:
  explicit TileGradientOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    const Tensor& input = context->input(0);
    const Tensor& multiples = context->input(1);
    OP_REQUIRES(
        context, IsLegacyVector(multiples.shape()),
        errors::InvalidArgument("Expected multiples to be 1-D, but got shape ",
                                multiples.shape().DebugString()));
    OP_REQUIRES(context, input.dims() == multiples.NumElements(),
                errors::InvalidArgument(
                    "Expected multiples argument to be a vector of length ",
                    input.dims(), " but got length ", multiples.dim_size(0)));

    const int input_dims = input.dims();

    // Eigen doesn't support scalars on the GPU, so handle 0-D specially
    if (input_dims == 0) {
      context->set_output(0, input);
      return;
    }

    const gtl::ArraySlice<Tmultiples> multiples_array(
        multiples.flat<Tmultiples>().data(), input_dims);
    TensorShape output_shape;
    std::vector<Tmultiples> input_dim_size_vec;
    for (int i = 0; i < input_dims; ++i) {
      OP_REQUIRES(
          context, multiples_array[i] > 0,
          errors::InvalidArgument("Expected multiples[", i, "] > 0, but got ",
                                  multiples_array[i]));
      OP_REQUIRES(context, input.dim_size(i) % multiples_array[i] == 0,
                  errors::InvalidArgument("Expected input_dim[", i,
                                          "] to be divisible by multiples[", i,
                                          "], but ", input.dim_size(i), " % ",
                                          multiples_array[i], " != 0"));
      output_shape.AddDim(input.dim_size(i) / multiples_array[i]);
      input_dim_size_vec.push_back(input.dim_size(i));
    }
    if (output_shape == input.shape()) {
      context->set_output(0, input);
      return;
    }
    Tensor* result = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &result));

#define HANDLE_DIM(DT, NDIM)                                           \
  if (context->input(0).dtype() == DT && input_dims == NDIM) {         \
    HandleCase<DT, NDIM>(context, input_dim_size_vec, multiples_array, \
                         result);                                      \
    return;                                                            \
  }

#define HANDLE_TYPE(T) \
  HANDLE_DIM(T, 1)     \
  HANDLE_DIM(T, 2)     \
  HANDLE_DIM(T, 3)     \
  HANDLE_DIM(T, 4)     \
  HANDLE_DIM(T, 5)     \
  HANDLE_DIM(T, 6)     \
  HANDLE_DIM(T, 7)

#define HANDLE_TYPE_NAME(T) HANDLE_TYPE(DataTypeToEnum<T>::value)

    TF_CALL_float(HANDLE_TYPE_NAME);
    TF_CALL_double(HANDLE_TYPE_NAME);
    TF_CALL_int32(HANDLE_TYPE_NAME);
    TF_CALL_int16(HANDLE_TYPE_NAME);
    TF_CALL_int64(HANDLE_TYPE_NAME);
    TF_CALL_half(HANDLE_TYPE_NAME);
    TF_CALL_complex64(HANDLE_TYPE_NAME);
    TF_CALL_complex128(HANDLE_TYPE_NAME);

#undef HANDLE_TYPE_NAME
#undef HANDLE_TYPE
#undef HANDLE_DIM

    OP_REQUIRES(context, false,
                errors::Unimplemented(
                    "TileGradientOp : Unhandled input dimensions, DT : ",
                    context->input(0).dtype(), ", dims : ", input_dims));
  }

 private:
  template <DataType DT, int NDIM>
  void HandleCase(OpKernelContext* context,
                  const std::vector<Tmultiples>& input_dims,
                  const gtl::ArraySlice<Tmultiples>& multiples_array,
                  Tensor* result);

  template <DataType DT, int NDIM>
  void HandleCaseImpl(OpKernelContext* context,
                      const std::vector<Tmultiples>& input_dims,
                      const gtl::ArraySlice<Tmultiples>& multiples_array,
                      Tensor* result) {
    typedef typename EnumToDataType<DT>::Type T;

    bool reduction_only = true;
    std::vector<Tmultiples> reduction_dims;

    for (int i = 0; i < NDIM; ++i) {
      if (input_dims[i] > multiples_array[i] && multiples_array[i] > 1) {
        reduction_only = false;
        break;
      } else {
        if (multiples_array[i] == input_dims[i]) {
          reduction_dims.push_back(i);
        }
      }
    }

    if (reduction_only) {
#define HANDLE_DIM(D)                                            \
  if (reduction_dims.size() == (D)) {                            \
    HandleReduce<T, NDIM, (D)>(context, reduction_dims, result); \
    return;                                                      \
  }
      // NOTE(keveman): Handling the most common case here.
      // Adding more cases here would require more templating and code
      // explosion. For instance, HANDLE_DIM(2) wouldn't make sense for NDIM=1.
      HANDLE_DIM(1);

// Fall through to the unoptimized version.
#undef HANDLE_DIM
    }

    Eigen::DSizes<Eigen::DenseIndex, NDIM> indices;
    Eigen::DSizes<Eigen::DenseIndex, NDIM> sizes;

    // Accumulate slices along the dimensions into the output. The number of
    // slices along dimension 'i' is simply the multiple along dimension 'i'
    // passed to the original Tile op.
    for (int i = 0; i < NDIM; ++i) {
      sizes[i] = input_dims[i] / multiples_array[i];
      indices[i] = 0;
    }

    bool first = true;
    while (true) {
      functor::TileGrad<Device, T, NDIM>()(
          context->eigen_device<Device>(), result->tensor<T, NDIM>(),
          context->input(0).tensor<T, NDIM>(), indices, sizes, first);
      first = false;
      // Increment the begin indices.
      int i = 0;
      while (i < NDIM && indices[i] / sizes[i] == multiples_array[i] - 1) {
        indices[i] = 0;
        ++i;
      }
      // We are finished if we have iterated to the maximum along all
      // dimensions.
      if (i == NDIM) {
        break;
      }
      indices[i] += sizes[i];
    }
  }

  template <typename T, int NDIM, int REDUCENDIM>
  void HandleReduce(OpKernelContext* context,
                    const std::vector<Tmultiples>& reduce_dim_in,
                    Tensor* result) {
    static_assert(NDIM >= REDUCENDIM, "Too many reduced dimensions");
    Eigen::DSizes<Eigen::DenseIndex, REDUCENDIM> reduce_dim;
    Eigen::DSizes<Eigen::DenseIndex, NDIM> reshape_dim;

    for (int i = 0; i < REDUCENDIM; ++i) {
      reduce_dim[i] = reduce_dim_in[i];
    }

    for (int i = 0; i < NDIM; ++i) {
      reshape_dim[i] = result->dim_size(i);
    }

    functor::ReduceAndReshape<Device, T, NDIM, REDUCENDIM>()(
        context->eigen_device<Device>(), result->tensor<T, NDIM>(),
        context->input(0).tensor<T, NDIM>(), reduce_dim, reshape_dim);
  }

  TF_DISALLOW_COPY_AND_ASSIGN(TileGradientOp);
};

template <typename Device, typename Tmultiples>
template <DataType DT, int NDIM>
inline void TileGradientOp<Device, Tmultiples>::HandleCase(
    OpKernelContext* context, const std::vector<Tmultiples>& input_dims,
    const gtl::ArraySlice<Tmultiples>& multiples_array, Tensor* result) {
  LOG(FATAL) << "TileGradientOp: Invalid combination of Device, DT and NDIM: "
             << MakeTypeIndex<Device>().name() << ", " << DataTypeString(DT)
             << ", " << NDIM;
}

#define HANDLE_CASE(device, T, dtype, Tmultiples, ndim)                        \
  template <>                                                                  \
  template <>                                                                  \
  void TileGradientOp<device, Tmultiples>::HandleCase<dtype, ndim>(            \
      OpKernelContext * context, const std::vector<Tmultiples>& input_dims,    \
      const gtl::ArraySlice<Tmultiples>& multiples_array, Tensor* result) {    \
    HandleCaseImpl<dtype, ndim>(context, input_dims, multiples_array, result); \
  }

// 0-D handled specially above
#define HANDLE_CASE_DIM(device, T, dtype)  \
  HANDLE_CASE(device, T, dtype, int32, 1); \
  HANDLE_CASE(device, T, dtype, int32, 2); \
  HANDLE_CASE(device, T, dtype, int32, 3); \
  HANDLE_CASE(device, T, dtype, int32, 4); \
  HANDLE_CASE(device, T, dtype, int32, 5); \
  HANDLE_CASE(device, T, dtype, int32, 6); \
  HANDLE_CASE(device, T, dtype, int32, 7); \
  HANDLE_CASE(device, T, dtype, int64, 1); \
  HANDLE_CASE(device, T, dtype, int64, 2); \
  HANDLE_CASE(device, T, dtype, int64, 3); \
  HANDLE_CASE(device, T, dtype, int64, 4); \
  HANDLE_CASE(device, T, dtype, int64, 5); \
  HANDLE_CASE(device, T, dtype, int64, 6); \
  HANDLE_CASE(device, T, dtype, int64, 7);

#define HANDLE_TYPE_NAME_CPU(T) \
  HANDLE_CASE_DIM(CPUDevice, T, DataTypeToEnum<T>::value);

#define HANDLE_TYPE_NAME_GPU(T) \
  HANDLE_CASE_DIM(GPUDevice, T, DataTypeToEnum<T>::value);

TF_CALL_float(HANDLE_TYPE_NAME_CPU);
TF_CALL_double(HANDLE_TYPE_NAME_CPU);
TF_CALL_int16(HANDLE_TYPE_NAME_CPU);
TF_CALL_int32(HANDLE_TYPE_NAME_CPU);
TF_CALL_int64(HANDLE_TYPE_NAME_CPU);
TF_CALL_half(HANDLE_TYPE_NAME_CPU);
TF_CALL_complex64(HANDLE_TYPE_NAME_CPU);
TF_CALL_complex128(HANDLE_TYPE_NAME_CPU);

#if GOOGLE_CUDA
TF_CALL_float(HANDLE_TYPE_NAME_GPU);
TF_CALL_double(HANDLE_TYPE_NAME_GPU);
TF_CALL_int16(HANDLE_TYPE_NAME_GPU);
TF_CALL_int32(HANDLE_TYPE_NAME_GPU);
TF_CALL_int64(HANDLE_TYPE_NAME_GPU);
TF_CALL_half(HANDLE_TYPE_NAME_GPU);
TF_CALL_complex64(HANDLE_TYPE_NAME_GPU);
TF_CALL_complex128(HANDLE_TYPE_NAME_GPU);
#endif  // GOOGLE_CUDA

#if TENSORFLOW_USE_SYCL
#define HANDLE_TYPE_NAME_SYCL(T) \
  HANDLE_CASE_DIM(SYCLDevice, T, DataTypeToEnum<T>::value);

TF_CALL_float(HANDLE_TYPE_NAME_SYCL);
TF_CALL_double(HANDLE_TYPE_NAME_SYCL);
TF_CALL_int16(HANDLE_TYPE_NAME_SYCL);
TF_CALL_int32(HANDLE_TYPE_NAME_SYCL);
TF_CALL_int64(HANDLE_TYPE_NAME_SYCL);
#undef HANDLE_TYPE_NAME_SYCL
#endif  // TENSORFLOW_USE_SYCL

#undef HANDLE_TYPE_NAME_CPU
#undef HANDLE_TYPE_NAME_GPU
#undef HANDLE_CASE_DIM
#undef HANDLE_CASE

REGISTER_KERNEL_BUILDER(Name("Tile")
                            .Device(DEVICE_CPU)
                            .HostMemory("multiples")
                            .TypeConstraint<int32>("Tmultiples"),
                        TileOp<CPUDevice, int32>);
REGISTER_KERNEL_BUILDER(Name("Tile")
                            .Device(DEVICE_CPU)
                            .HostMemory("multiples")
                            .TypeConstraint<int64>("Tmultiples"),
                        TileOp<CPUDevice, int64>);
REGISTER_KERNEL_BUILDER(Name("TileGrad")
                            .Device(DEVICE_CPU)
                            .HostMemory("multiples")
                            .TypeConstraint<int32>("Tmultiples"),
                        TileGradientOp<CPUDevice, int32>);
REGISTER_KERNEL_BUILDER(Name("TileGrad")
                            .Device(DEVICE_CPU)
                            .HostMemory("multiples")
                            .TypeConstraint<int64>("Tmultiples"),
                        TileGradientOp<CPUDevice, int64>);

#if GOOGLE_CUDA
#define REGISTER_GPU(type)                                         \
  REGISTER_KERNEL_BUILDER(Name("Tile")                             \
                              .Device(DEVICE_GPU)                  \
                              .TypeConstraint<type>("T")           \
                              .TypeConstraint<int32>("Tmultiples") \
                              .HostMemory("multiples"),            \
                          TileOp<GPUDevice, int32>);               \
  REGISTER_KERNEL_BUILDER(Name("Tile")                             \
                              .Device(DEVICE_GPU)                  \
                              .TypeConstraint<type>("T")           \
                              .TypeConstraint<int64>("Tmultiples") \
                              .HostMemory("multiples"),            \
                          TileOp<GPUDevice, int64>);               \
  REGISTER_KERNEL_BUILDER(Name("TileGrad")                         \
                              .Device(DEVICE_GPU)                  \
                              .TypeConstraint<type>("T")           \
                              .TypeConstraint<int32>("Tmultiples") \
                              .HostMemory("multiples"),            \
                          TileGradientOp<GPUDevice, int32>);       \
  REGISTER_KERNEL_BUILDER(Name("TileGrad")                         \
                              .Device(DEVICE_GPU)                  \
                              .TypeConstraint<type>("T")           \
                              .TypeConstraint<int64>("Tmultiples") \
                              .HostMemory("multiples"),            \
                          TileGradientOp<GPUDevice, int64>);

TF_CALL_float(REGISTER_GPU);
TF_CALL_double(REGISTER_GPU);
TF_CALL_half(REGISTER_GPU);
TF_CALL_int16(REGISTER_GPU);
TF_CALL_int32(REGISTER_GPU);
TF_CALL_complex64(REGISTER_GPU);
TF_CALL_complex128(REGISTER_GPU)

#undef REGISTER_GPU
#endif  // GOOGLE_CUDA

#ifdef TENSORFLOW_USE_SYCL
#define REGISTER_SYCL(type)                                        \
  REGISTER_KERNEL_BUILDER(Name("Tile")                             \
                              .Device(DEVICE_SYCL)                 \
                              .TypeConstraint<type>("T")           \
                              .TypeConstraint<int32>("Tmultiples") \
                              .HostMemory("multiples"),            \
                          TileOp<SYCLDevice, int32>);              \
  REGISTER_KERNEL_BUILDER(Name("Tile")                             \
                              .Device(DEVICE_SYCL)                 \
                              .TypeConstraint<type>("T")           \
                              .TypeConstraint<int64>("Tmultiples") \
                              .HostMemory("multiples"),            \
                          TileOp<SYCLDevice, int64>);              \
  REGISTER_KERNEL_BUILDER(Name("TileGrad")                         \
                              .Device(DEVICE_SYCL)                 \
                              .TypeConstraint<type>("T")           \
                              .TypeConstraint<int32>("Tmultiples") \
                              .HostMemory("multiples"),            \
                          TileGradientOp<SYCLDevice, int32>);      \
  REGISTER_KERNEL_BUILDER(Name("TileGrad")                         \
                              .Device(DEVICE_SYCL)                 \
                              .TypeConstraint<type>("T")           \
                              .TypeConstraint<int64>("Tmultiples") \
                              .HostMemory("multiples"),            \
                          TileGradientOp<SYCLDevice, int64>);

    TF_CALL_float(REGISTER_SYCL);
TF_CALL_double(REGISTER_SYCL);

#undef REGISTER_SYCL
#endif  // TENSORFLOW_USE_SYCL

}  // namespace tensorflow

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

#ifdef GOOGLE_CUDA
#define EIGEN_USE_GPU
#endif  // GOOGLE_CUDA

#include "tensorflow/core/kernels/tile_ops.h"

#include <vector>
#include "tensorflow/core/framework/numeric_op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/type_index.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/platform/macros.h"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

// --------------------------------------------------------------------------
template <typename Device>
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

    const gtl::ArraySlice<int32> multiples_array(multiples.flat<int32>().data(),
                                                 input_dims);
    TensorShape output_shape;
    for (int i = 0; i < input_dims; ++i) {
      OP_REQUIRES(
          context, multiples_array[i] >= 0,
          errors::InvalidArgument("Expected multiples[", i, "] >= 0, but got ",
                                  multiples_array[i]));
      output_shape.AddDim(input.dim_size(i) * multiples_array[i]);
    }
    Tensor* result = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &result));

    // If there's no output, there's nothing to do.
    if (output_shape.num_elements() == 0) return;

#define HANDLE_DIM(DT, NDIM)                                   \
  if (context->input(0).dtype() == DT && input_dims == NDIM) { \
    HandleCase<DT, NDIM>(context, multiples_array, result);    \
    return;                                                    \
  }

#define HANDLE_TYPE(T) \
  HANDLE_DIM(T, 1)     \
  HANDLE_DIM(T, 2)     \
  HANDLE_DIM(T, 3)     \
  HANDLE_DIM(T, 4)     \
  HANDLE_DIM(T, 5)

    HANDLE_TYPE(DT_BOOL);
    HANDLE_TYPE(DT_FLOAT);
    HANDLE_TYPE(DT_DOUBLE);
    HANDLE_TYPE(DT_UINT8);
    HANDLE_TYPE(DT_INT32);
    HANDLE_TYPE(DT_INT16);
    HANDLE_TYPE(DT_INT64);
    HANDLE_TYPE(DT_STRING);  // when DEVICE=CPUDevice.

#undef HANDLE_TYPE
#undef HANDLE_DIM

    OP_REQUIRES(context, false,
                errors::Unimplemented(
                    "TileOp : Unhandled input dimensions, DT : ",
                    context->input(0).dtype(), ", dims : ", input_dims));
  }

 private:
  template <DataType DT, int NDIM>
  void HandleCaseImpl(OpKernelContext* context,
                      const gtl::ArraySlice<int32>& multiples_array,
                      Tensor* result) {
    typedef typename EnumToDataType<DT>::Type T;
    Eigen::array<int32, NDIM> broadcast_array;
    for (int i = 0; i < NDIM; ++i) {
      broadcast_array[i] = multiples_array[i];
    }
    functor::Tile<Device, T, NDIM>()(
        context->eigen_device<Device>(), result->tensor<T, NDIM>(),
        context->input(0).tensor<T, NDIM>(), broadcast_array);
  }

  template <DataType DT, int NDIM>
  void HandleCase(OpKernelContext* context,
                  const gtl::ArraySlice<int32>& multiples_array,
                  Tensor* result);

  TF_DISALLOW_COPY_AND_ASSIGN(TileOp);
};

template <typename Device>
template <DataType DT, int NDIM>
inline void TileOp<Device>::HandleCase(
    OpKernelContext* context, const gtl::ArraySlice<int32>& multiples_array,
    Tensor* result) {
  // TODO(vrv): print out the device name if useful. Currently disabled to avoid
  // having to use RTTI.
  LOG(FATAL) << "TileOp: Invalid combination of Device, DT and NDIM: "
             // << typeid(Device).name() << ", "
             << DataTypeString(DT) << ", " << NDIM;
}

#define HANDLE_CASE(device, dtype, ndim)                               \
  template <>                                                          \
  template <>                                                          \
  void TileOp<device>::HandleCase<dtype, ndim>(                        \
      OpKernelContext * context,                                       \
      const gtl::ArraySlice<int32>& multiples_array, Tensor* result) { \
    HandleCaseImpl<dtype, ndim>(context, multiples_array, result);     \
  }

// 0-D handled above
#define HANDLE_CASE_DIM(device, dtype) \
  HANDLE_CASE(device, dtype, 1);       \
  HANDLE_CASE(device, dtype, 2);       \
  HANDLE_CASE(device, dtype, 3);       \
  HANDLE_CASE(device, dtype, 4);       \
  HANDLE_CASE(device, dtype, 5);

HANDLE_CASE_DIM(CPUDevice, DT_BOOL);
HANDLE_CASE_DIM(CPUDevice, DT_FLOAT);
HANDLE_CASE_DIM(CPUDevice, DT_DOUBLE);
HANDLE_CASE_DIM(CPUDevice, DT_UINT8);
HANDLE_CASE_DIM(CPUDevice, DT_INT32);
HANDLE_CASE_DIM(CPUDevice, DT_INT16);
HANDLE_CASE_DIM(CPUDevice, DT_INT64);
HANDLE_CASE_DIM(CPUDevice, DT_STRING);

#if GOOGLE_CUDA
HANDLE_CASE_DIM(GPUDevice, DT_FLOAT);
HANDLE_CASE_DIM(GPUDevice, DT_DOUBLE);
HANDLE_CASE_DIM(GPUDevice, DT_INT16);
HANDLE_CASE_DIM(GPUDevice, DT_INT32);
HANDLE_CASE_DIM(GPUDevice, DT_INT64);
#endif  // GOOGLE_CUDA

#undef HANDLE_CASE_DIM
#undef HANDLE_CASE

// --------------------------------------------------------------------------
template <typename Device>
class TileGradientOp : public OpKernel {
 public:
  explicit TileGradientOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_DEPRECATED(context, 3, "TileGrad has been replaced with reduce_sum");
  }

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

    const gtl::ArraySlice<int32> multiples_array(multiples.flat<int32>().data(),
                                                 input_dims);
    TensorShape output_shape;
    std::vector<int32> input_dim_size_vec;
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
  HANDLE_DIM(T, 5)

    HANDLE_TYPE(DT_FLOAT);
    HANDLE_TYPE(DT_DOUBLE);
    HANDLE_TYPE(DT_INT32);
    HANDLE_TYPE(DT_INT16);
    HANDLE_TYPE(DT_INT64);

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
                  const std::vector<int32>& input_dims,
                  const gtl::ArraySlice<int32>& multiples_array,
                  Tensor* result);

  template <DataType DT, int NDIM>
  void HandleCaseImpl(OpKernelContext* context,
                      const std::vector<int32>& input_dims,
                      const gtl::ArraySlice<int32>& multiples_array,
                      Tensor* result) {
    typedef typename EnumToDataType<DT>::Type T;

    bool reduction_only = true;
    std::vector<int> reduction_dims;

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
                    const std::vector<int32>& reduce_dim_in, Tensor* result) {
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

template <typename Device>
template <DataType DT, int NDIM>
inline void TileGradientOp<Device>::HandleCase(
    OpKernelContext* context, const std::vector<int32>& input_dims,
    const gtl::ArraySlice<int32>& multiples_array, Tensor* result) {
  LOG(FATAL) << "TileGradientOp: Invalid combination of Device, DT and NDIM: "
             << MakeTypeIndex<Device>().name() << ", " << DataTypeString(DT)
             << ", " << NDIM;
}

#define HANDLE_CASE(device, dtype, ndim)                                       \
  template <>                                                                  \
  template <>                                                                  \
  void TileGradientOp<device>::HandleCase<dtype, ndim>(                        \
      OpKernelContext * context, const std::vector<int32>& input_dims,         \
      const gtl::ArraySlice<int32>& multiples_array, Tensor* result) {         \
    HandleCaseImpl<dtype, ndim>(context, input_dims, multiples_array, result); \
  }

// 0-D handled specially above
#define HANDLE_CASE_DIM(device, dtype) \
  HANDLE_CASE(device, dtype, 1);       \
  HANDLE_CASE(device, dtype, 2);       \
  HANDLE_CASE(device, dtype, 3);       \
  HANDLE_CASE(device, dtype, 4);       \
  HANDLE_CASE(device, dtype, 5);

HANDLE_CASE_DIM(CPUDevice, DT_FLOAT);
HANDLE_CASE_DIM(CPUDevice, DT_DOUBLE);
HANDLE_CASE_DIM(CPUDevice, DT_INT16);
HANDLE_CASE_DIM(CPUDevice, DT_INT32);
HANDLE_CASE_DIM(CPUDevice, DT_INT64);

#if GOOGLE_CUDA
HANDLE_CASE_DIM(GPUDevice, DT_FLOAT);
HANDLE_CASE_DIM(GPUDevice, DT_DOUBLE);
HANDLE_CASE_DIM(GPUDevice, DT_INT16);
HANDLE_CASE_DIM(GPUDevice, DT_INT32);
HANDLE_CASE_DIM(GPUDevice, DT_INT64);
#endif  // GOOGLE_CUDA

#undef HANDLE_CASE_DIM
#undef HANDLE_CASE

REGISTER_KERNEL_BUILDER(Name("Tile").Device(DEVICE_CPU).HostMemory("multiples"),
                        TileOp<CPUDevice>);
REGISTER_KERNEL_BUILDER(Name("TileGrad")
                            .Device(DEVICE_CPU)
                            .HostMemory("multiples"),
                        TileGradientOp<CPUDevice>);

#if GOOGLE_CUDA
#define DEFINE_GPU_TYPE(T) \
  DEFINE_GPU_DIM(T, 1)     \
  DEFINE_GPU_DIM(T, 2)     \
  DEFINE_GPU_DIM(T, 3)     \
  DEFINE_GPU_DIM(T, 4)     \
  DEFINE_GPU_DIM(T, 5)

#define DEFINE_GPU_DIM(T, NDIM)                                               \
  template <>                                                                 \
  void Tile<GPUDevice, T, NDIM>::operator()(                                  \
      const GPUDevice& d, typename TTypes<T, NDIM>::Tensor out,               \
      typename TTypes<T, NDIM>::ConstTensor in,                               \
      const Eigen::array<int32, NDIM>& broadcast_array) const;                \
  extern template struct Tile<GPUDevice, T, NDIM>;                            \
  template <>                                                                 \
  void TileGrad<GPUDevice, T, NDIM>::operator()(                              \
      const GPUDevice& d, typename TTypes<T, NDIM>::Tensor out,               \
      typename TTypes<T, NDIM>::ConstTensor in,                               \
      const Eigen::DSizes<Eigen::DenseIndex, NDIM>& indices,                  \
      const Eigen::DSizes<Eigen::DenseIndex, NDIM>& sizes, bool first) const; \
  extern template struct TileGrad<GPUDevice, T, NDIM>;                        \
  template <>                                                                 \
  void ReduceAndReshape<GPUDevice, T, NDIM, 1>::operator()(                   \
      const GPUDevice& d, typename TTypes<T, NDIM>::Tensor out,               \
      typename TTypes<T, NDIM>::ConstTensor in,                               \
      const Eigen::DSizes<Eigen::DenseIndex, 1>& reduce_dim,                  \
      const Eigen::DSizes<Eigen::DenseIndex, NDIM>& reshape_dim) const;       \
  extern template struct ReduceAndReshape<GPUDevice, T, NDIM, 1>;

namespace functor {
DEFINE_GPU_TYPE(float);
DEFINE_GPU_TYPE(double);
DEFINE_GPU_TYPE(int64);
DEFINE_GPU_TYPE(int32);
DEFINE_GPU_TYPE(int16);
}  // end namespace functor

#undef DEFINE_GPU_DIM
#undef DEFINE_GPU_TYPE

REGISTER_KERNEL_BUILDER(Name("Tile")
                            .Device(DEVICE_GPU)
                            .TypeConstraint<float>("T")
                            .HostMemory("multiples"),
                        TileOp<GPUDevice>);
REGISTER_KERNEL_BUILDER(Name("Tile")
                            .Device(DEVICE_GPU)
                            .TypeConstraint<double>("T")
                            .HostMemory("multiples"),
                        TileOp<GPUDevice>);
REGISTER_KERNEL_BUILDER(Name("Tile")
                            .Device(DEVICE_GPU)
                            .TypeConstraint<int16>("T")
                            .HostMemory("multiples"),
                        TileOp<GPUDevice>);

REGISTER_KERNEL_BUILDER(Name("TileGrad")
                            .Device(DEVICE_GPU)
                            .TypeConstraint<float>("T")
                            .HostMemory("multiples"),
                        TileGradientOp<GPUDevice>);
REGISTER_KERNEL_BUILDER(Name("TileGrad")
                            .Device(DEVICE_GPU)
                            .TypeConstraint<double>("T")
                            .HostMemory("multiples"),
                        TileGradientOp<GPUDevice>);
REGISTER_KERNEL_BUILDER(Name("TileGrad")
                            .Device(DEVICE_GPU)
                            .TypeConstraint<int16>("T")
                            .HostMemory("multiples"),
                        TileGradientOp<GPUDevice>);
#endif  // GOOGLE_CUDA
}  // namespace tensorflow

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

#ifndef TENSORFLOW_CORE_KERNELS_STRIDED_SLICE_OP_IMPL_H_
#define TENSORFLOW_CORE_KERNELS_STRIDED_SLICE_OP_IMPL_H_

// Functor definition for StridedSliceOp, must be compilable by nvcc.

#include "unsupported/Eigen/CXX11/Tensor"  // from @eigen_archive
#include "tensorflow/core/framework/bounds_check.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/register_types_traits.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/variant.h"
#include "tensorflow/core/framework/variant_encode_decode.h"
#include "tensorflow/core/kernels/dense_update_functor.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/kernels/slice_op.h"
#include "tensorflow/core/kernels/strided_slice_op.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/platform/mem.h"

namespace tensorflow {

template <typename Device, typename T, int NDIM>
void HandleStridedSliceCase(OpKernelContext* context,
                            const absl::Span<const int64_t>& begin,
                            const absl::Span<const int64_t>& end,
                            const absl::Span<const int64_t>& strides,
                            const TensorShape& processing_shape,
                            bool is_simple_slice, Tensor* result);

template <typename Device, typename T, int NDIM>
void HandleStridedSliceGradCase(OpKernelContext* context,
                                const absl::Span<const int64_t>& begin,
                                const absl::Span<const int64_t>& end,
                                const absl::Span<const int64_t>& strides,
                                const TensorShape& processing_shape,
                                bool is_simple_slice, Tensor* result);

template <typename Device, typename T, int NDIM>
class HandleStridedSliceAssignCase {
 public:
  void operator()(OpKernelContext* context,
                  const absl::Span<const int64_t>& begin,
                  const absl::Span<const int64_t>& end,
                  const absl::Span<const int64_t>& strides,
                  const StridedSliceAssignBCast& bcast, Tensor* result);
};
}  // namespace tensorflow

// The actual implementation. This is designed so multiple
// translation units can include this file in the form
//
// #define STRIDED_SLICE_INSTANTIATE_DIM 1
// #include <thisfile>
// #undef STRIDED_SLICE_INSTANTIATE_DIM
//
#ifdef STRIDED_SLICE_INSTANTIATE_DIM

namespace tensorflow {

template <typename Device, typename T, int NDIM>
void HandleStridedSliceCase(OpKernelContext* context,
                            const absl::Span<const int64_t>& begin,
                            const absl::Span<const int64_t>& end,
                            const absl::Span<const int64_t>& strides,
                            const TensorShape& processing_shape,
                            bool is_simple_slice, Tensor* result) {
  typedef typename proxy_type<Device, T>::type Proxy;

  absl::InlinedVector<int64_t, 4UL> processing_dims =
      processing_shape.dim_sizes();
  if (is_simple_slice) {
    Eigen::DSizes<Eigen::DenseIndex, NDIM> begin_di;
    Eigen::DSizes<Eigen::DenseIndex, NDIM> sizes_di;
    for (int i = 0; i < NDIM; ++i) {
      begin_di[i] = begin[i];
      sizes_di[i] = end[i] - begin[i];
    }
    functor::Slice<Device, Proxy, NDIM>()(
        context->eigen_device<Device>(),
        result->bit_casted_shaped<Proxy, NDIM>(processing_dims),
        context->input(0).bit_casted_tensor<Proxy, NDIM>(), begin_di, sizes_di);
  } else {
    Eigen::DSizes<Eigen::DenseIndex, NDIM> begin_di;
    Eigen::DSizes<Eigen::DenseIndex, NDIM> end_di;
    Eigen::DSizes<Eigen::DenseIndex, NDIM> strides_di;
    for (int i = 0; i < NDIM; ++i) {
      begin_di[i] = begin[i];
      end_di[i] = end[i];
      strides_di[i] = strides[i];
    }
    functor::StridedSlice<Device, Proxy, NDIM>()(
        context->eigen_device<Device>(),
        result->bit_casted_shaped<Proxy, NDIM>(processing_dims),
        context->input(0).bit_casted_tensor<Proxy, NDIM>(), begin_di, end_di,
        strides_di);
  }
}

template <typename Device, typename T, int NDIM>
void HandleStridedSliceGradCase(OpKernelContext* context,
                                const absl::Span<const int64_t>& begin,
                                const absl::Span<const int64_t>& end,
                                const absl::Span<const int64_t>& strides,
                                const TensorShape& processing_shape,
                                bool is_simple_slice, Tensor* result) {
  absl::InlinedVector<int64_t, 4UL> processing_dims =
      processing_shape.dim_sizes();

  Eigen::DSizes<Eigen::DenseIndex, NDIM> begin_di;
  Eigen::DSizes<Eigen::DenseIndex, NDIM> end_di;
  Eigen::DSizes<Eigen::DenseIndex, NDIM> strides_di;
  for (int i = 0; i < NDIM; ++i) {
    begin_di[i] = begin[i];
    end_di[i] = end[i];
    strides_di[i] = strides[i];
  }

  typedef typename proxy_type<Device, T>::type Proxy;
  functor::StridedSliceGrad<Device, Proxy, NDIM>()(
      context->eigen_device<Device>(), result->bit_casted_tensor<Proxy, NDIM>(),
      context->input(4).bit_casted_shaped<Proxy, NDIM>(processing_dims),
      begin_di, end_di, strides_di);
}

template <typename Device, typename T, int NDIM>
void HandleStridedSliceAssignCase<Device, T, NDIM>::operator()(
    OpKernelContext* context, const absl::Span<const int64_t>& begin,
    const absl::Span<const int64_t>& end,
    const absl::Span<const int64_t>& strides,
    const StridedSliceAssignBCast& bcast, Tensor* result) {
  typedef typename proxy_type<Device, T>::type Proxy;
  Eigen::DSizes<Eigen::DenseIndex, NDIM> begin_di;
  Eigen::DSizes<Eigen::DenseIndex, NDIM> end_di;
  Eigen::DSizes<Eigen::DenseIndex, NDIM> strides_di;
  for (int i = 0; i < NDIM; ++i) {
    begin_di[i] = begin[i];
    end_di[i] = end[i];
    strides_di[i] = strides[i];
  }

  constexpr int kRhsInput = 4;
  const Tensor& input = context->input(kRhsInput);
  functor::StridedSliceAssign<Device, Proxy, NDIM>()(
      context->eigen_device<Device>(), result->bit_casted_tensor<Proxy, NDIM>(),
      input.bit_casted_shaped<Proxy, NDIM>(bcast.reshape()), begin_di, end_di,
      strides_di, bcast);
}

template <typename Device, typename T>
class HandleStridedSliceAssignCase<Device, T, 0> {
 public:
  enum { NDIM_PROXY = 1 };
  void operator()(OpKernelContext* context,
                  const absl::Span<const int64_t>& begin,
                  const absl::Span<const int64_t>& end,
                  const absl::Span<const int64_t>& strides,
                  const StridedSliceAssignBCast& bcast, Tensor* result) {
    absl::InlinedVector<int64_t, 1UL> processing_dims(1);
    processing_dims[0] = 1;

    typedef typename proxy_type<Device, T>::type Proxy;
    functor::StridedSliceAssignScalar<Device, Proxy>()(
        context->eigen_device<Device>(),
        result->bit_casted_shaped<Proxy, 1>(processing_dims),
        context->input(4).bit_casted_shaped<Proxy, 1>(processing_dims));
  }
};

// NOTE(aselle): according to bsteiner, we need this because otherwise
// nvcc instantiates templates that are invalid. strided_slice_op_gpu.cu
// handles instantiates externally. It is important that this is done
// before the HandleXXCase's are instantiated to avoid duplicate
// specialization errors.

#define PREVENT_INSTANTIATE_DIM1_AND_UP(T, NDIM)                   \
  namespace functor {                                              \
  template <>                                                      \
  void StridedSlice<GPUDevice, T, NDIM>::operator()(               \
      const GPUDevice& d, typename TTypes<T, NDIM>::Tensor output, \
      typename TTypes<T, NDIM>::ConstTensor input,                 \
      const Eigen::DSizes<Eigen::DenseIndex, NDIM>& start,         \
      const Eigen::DSizes<Eigen::DenseIndex, NDIM>& stop,          \
      const Eigen::DSizes<Eigen::DenseIndex, NDIM>& strides);      \
  extern template struct StridedSlice<GPUDevice, T, NDIM>;         \
  template <>                                                      \
  void Slice<GPUDevice, T, NDIM>::operator()(                      \
      const GPUDevice& d, typename TTypes<T, NDIM>::Tensor output, \
      typename TTypes<T, NDIM>::ConstTensor input,                 \
      const Eigen::DSizes<Eigen::DenseIndex, NDIM>& indices,       \
      const Eigen::DSizes<Eigen::DenseIndex, NDIM>& sizes);        \
  extern template struct Slice<GPUDevice, T, NDIM>;                \
  template <>                                                      \
  void StridedSliceGrad<GPUDevice, T, NDIM>::operator()(           \
      const GPUDevice& d, typename TTypes<T, NDIM>::Tensor output, \
      typename TTypes<T, NDIM>::ConstTensor input,                 \
      const Eigen::DSizes<Eigen::DenseIndex, NDIM>& start,         \
      const Eigen::DSizes<Eigen::DenseIndex, NDIM>& stop,          \
      const Eigen::DSizes<Eigen::DenseIndex, NDIM>& strides);      \
  extern template struct StridedSliceGrad<GPUDevice, T, NDIM>;     \
  template <>                                                      \
  void StridedSliceAssign<GPUDevice, T, NDIM>::operator()(         \
      const GPUDevice& d, typename TTypes<T, NDIM>::Tensor output, \
      typename TTypes<T, NDIM>::ConstTensor input,                 \
      const Eigen::DSizes<Eigen::DenseIndex, NDIM>& start,         \
      const Eigen::DSizes<Eigen::DenseIndex, NDIM>& stop,          \
      const Eigen::DSizes<Eigen::DenseIndex, NDIM>& strides,       \
      const StridedSliceAssignBCast& bcast);                       \
  extern template struct StridedSliceAssign<GPUDevice, T, NDIM>;   \
  }  // namespace functor
#define PREVENT_INSTANTIATE_DIM0_ONLY(T, NDIM)                   \
  namespace functor {                                            \
  template <>                                                    \
  void StridedSliceAssignScalar<GPUDevice, T>::operator()(       \
      const GPUDevice& d, typename TTypes<T, 1>::Tensor output,  \
      typename TTypes<T, 1>::ConstTensor input);                 \
  extern template struct StridedSliceAssignScalar<GPUDevice, T>; \
  }  // namespace functor

// Dimension 0 only instantiates some functors. So we only need
// to prevent ones defined by PREVENT_INSTANTIATE_DIM0_ONLY
#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#if STRIDED_SLICE_INSTANTIATE_DIM == 0
#define PREVENT_INSTANTIATE(T, NDIM) PREVENT_INSTANTIATE_DIM0_ONLY(T, NDIM)
#else
#define PREVENT_INSTANTIATE(T, NDIM) PREVENT_INSTANTIATE_DIM1_AND_UP(T, NDIM)
#endif
#else
#define PREVENT_INSTANTIATE(T, NDIM)
#endif

#define INSTANTIATE_DIM1_AND_UP_HANDLERS(DEVICE, T, DIM)                \
  template void HandleStridedSliceCase<DEVICE, T, DIM>(                 \
      OpKernelContext * context, const gtl::ArraySlice<int64_t>& begin, \
      const gtl::ArraySlice<int64_t>& end,                              \
      const gtl::ArraySlice<int64_t>& strides,                          \
      const TensorShape& processing_shape, bool is_simple_slice,        \
      Tensor* result);                                                  \
  template void HandleStridedSliceGradCase<DEVICE, T, DIM>(             \
      OpKernelContext * context, const gtl::ArraySlice<int64_t>& begin, \
      const gtl::ArraySlice<int64_t>& end,                              \
      const gtl::ArraySlice<int64_t>& strides,                          \
      const TensorShape& processing_shape, bool is_simple_slice,        \
      Tensor* result);

#define INSTANTIATE_DIM0_AND_UP_HANDLERS(DEVICE, T, DIM) \
  template class HandleStridedSliceAssignCase<DEVICE, T, DIM>;

// Only some kernels need to be instantiated on dim 0.
#if STRIDED_SLICE_INSTANTIATE_DIM == 0
#define INSTANTIATE(DEVICE, T, DIM) \
  INSTANTIATE_DIM0_AND_UP_HANDLERS(DEVICE, T, DIM)
#else
#define INSTANTIATE(DEVICE, T, DIM)                \
  INSTANTIATE_DIM0_AND_UP_HANDLERS(DEVICE, T, DIM) \
  INSTANTIATE_DIM1_AND_UP_HANDLERS(DEVICE, T, DIM)
#endif

#define DECLARE_FOR_N_CPU(T) \
  INSTANTIATE(CPUDevice, T, STRIDED_SLICE_INSTANTIATE_DIM)

#define PREVENT_FOR_N_GPU(T) \
  PREVENT_INSTANTIATE(T, STRIDED_SLICE_INSTANTIATE_DIM)

#define DECLARE_FOR_N_GPU(T) \
  INSTANTIATE(GPUDevice, T, STRIDED_SLICE_INSTANTIATE_DIM)

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
TF_CALL_GPU_PROXY_TYPES(PREVENT_FOR_N_GPU);
TF_CALL_COMPLEX_TYPES(PREVENT_FOR_N_GPU);

TF_CALL_INTEGRAL_TYPES(DECLARE_FOR_N_GPU);
TF_CALL_GPU_ALL_TYPES(DECLARE_FOR_N_GPU);
#endif  // END GOOGLE_CUDA || TENSORFLOW_USE_ROCM

TF_CALL_ALL_TYPES(DECLARE_FOR_N_CPU);
TF_CALL_QUANTIZED_TYPES(DECLARE_FOR_N_CPU);
TF_CALL_float8_e5m2(DECLARE_FOR_N_CPU);
TF_CALL_float8_e4m3fn(DECLARE_FOR_N_CPU);

#undef INSTANTIATE
#undef DECLARE_FOR_N_CPU
#undef DECLARE_FOR_N_GPU

}  // end namespace tensorflow

#endif  // END STRIDED_SLICE_INSTANTIATE_DIM
#endif  // TENSORFLOW_CORE_KERNELS_STRIDED_SLICE_OP_IMPL_H_

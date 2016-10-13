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

#ifndef TENSORFLOW_KERNELS_STRIDED_SLICE_OP_IMPL_H_
#define TENSORFLOW_KERNELS_STRIDED_SLICE_OP_IMPL_H_

// Functor definition for StridedSliceOp, must be compilable by nvcc.

#include "tensorflow/core/kernels/slice_op.h"
#include "tensorflow/core/kernels/strided_slice_op.h"

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/register_types_traits.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/kernels/bounds_check.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/platform/mem.h"

namespace tensorflow {

template <typename Device, typename T, int NDIM>
void HandleStridedSliceCase(OpKernelContext* context,
                            const gtl::ArraySlice<int64>& begin,
                            const gtl::ArraySlice<int64>& end,
                            const gtl::ArraySlice<int64>& strides,
                            const TensorShape& processing_shape,
                            bool is_simple_slice, Tensor* result);

template <typename Device, typename T, int NDIM>
void HandleStridedSliceGradCase(OpKernelContext* context,
                                const gtl::ArraySlice<int64>& begin,
                                const gtl::ArraySlice<int64>& end,
                                const gtl::ArraySlice<int64>& strides,
                                const TensorShape& processing_shape,
                                bool is_simple_slice, Tensor* result);

template <typename Device, typename T, int NDIM>
void HandleStridedSliceAssignCase(OpKernelContext* context,
                                  const gtl::ArraySlice<int64>& begin,
                                  const gtl::ArraySlice<int64>& end,
                                  const gtl::ArraySlice<int64>& strides,
                                  const TensorShape& processing_shape,
                                  bool is_simple_slice, Tensor* result);
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
                            const gtl::ArraySlice<int64>& begin,
                            const gtl::ArraySlice<int64>& end,
                            const gtl::ArraySlice<int64>& strides,
                            const TensorShape& processing_shape,
                            bool is_simple_slice, Tensor* result) {
  typedef typename proxy_type<Device, T>::type Proxy;

  gtl::InlinedVector<int64, 4> processing_dims = processing_shape.dim_sizes();
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
                                const gtl::ArraySlice<int64>& begin,
                                const gtl::ArraySlice<int64>& end,
                                const gtl::ArraySlice<int64>& strides,
                                const TensorShape& processing_shape,
                                bool is_simple_slice, Tensor* result) {
  gtl::InlinedVector<int64, 4> processing_dims = processing_shape.dim_sizes();

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
void HandleStridedSliceAssignCase(OpKernelContext* context,
                                  const gtl::ArraySlice<int64>& begin,
                                  const gtl::ArraySlice<int64>& end,
                                  const gtl::ArraySlice<int64>& strides,
                                  const TensorShape& processing_shape,
                                  bool is_simple_slice, Tensor* result) {
  gtl::InlinedVector<int64, 4> processing_dims = processing_shape.dim_sizes();
  typedef typename proxy_type<Device, T>::type Proxy;
  Eigen::DSizes<Eigen::DenseIndex, NDIM> begin_di;
  Eigen::DSizes<Eigen::DenseIndex, NDIM> end_di;
  Eigen::DSizes<Eigen::DenseIndex, NDIM> strides_di;
  for (int i = 0; i < NDIM; ++i) {
    begin_di[i] = begin[i];
    end_di[i] = end[i];
    strides_di[i] = strides[i];
  }
  functor::StridedSliceAssign<Device, Proxy, NDIM>()(
      context->eigen_device<Device>(), result->bit_casted_tensor<Proxy, NDIM>(),
      context->input(4).bit_casted_shaped<Proxy, NDIM>(processing_dims),
      begin_di, end_di, strides_di);
}

// NODE(aselle): according to bsteiner, we need this because otherwise
// nvcc instantiates templates that are invalid. strided_slice_op_gpu.cu
// handles instantiates externally. It is important that this is done#

// before the HandleXXCase's are instantiated to avoid duplicate
// specialization errors.
#if GOOGLE_CUDA
#define PREVENT_INSTANTIATE(T, NDIM)                               \
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
      const Eigen::DSizes<Eigen::DenseIndex, NDIM>& strides);      \
  extern template struct StridedSliceAssign<GPUDevice, T, NDIM>;   \
  }  // namespace functor

#else
#define PREVENT_INSTANTIATE(T, NDIM)
#endif

#define INSTANTIATE(DEVICE, T, DIM)                                   \
  template void HandleStridedSliceCase<DEVICE, T, DIM>(               \
      OpKernelContext * context, const gtl::ArraySlice<int64>& begin, \
      const gtl::ArraySlice<int64>& end,                              \
      const gtl::ArraySlice<int64>& strides,                          \
      const TensorShape& processing_shape, bool is_simple_slice,      \
      Tensor* result);                                                \
  template void HandleStridedSliceGradCase<DEVICE, T, DIM>(           \
      OpKernelContext * context, const gtl::ArraySlice<int64>& begin, \
      const gtl::ArraySlice<int64>& end,                              \
      const gtl::ArraySlice<int64>& strides,                          \
      const TensorShape& processing_shape, bool is_simple_slice,      \
      Tensor* result);                                                \
  template void HandleStridedSliceAssignCase<DEVICE, T, DIM>(         \
      OpKernelContext * context, const gtl::ArraySlice<int64>& begin, \
      const gtl::ArraySlice<int64>& end,                              \
      const gtl::ArraySlice<int64>& strides,                          \
      const TensorShape& processing_shape, bool is_simple_slice,      \
      Tensor* result);

#define DECLARE_FOR_N_CPU(T) \
  INSTANTIATE(CPUDevice, T, STRIDED_SLICE_INSTANTIATE_DIM)

#define PREVENT_FOR_N_GPU(T) \
  PREVENT_INSTANTIATE(T, STRIDED_SLICE_INSTANTIATE_DIM)

#define DECLARE_FOR_N_GPU(T) \
  INSTANTIATE(GPUDevice, T, STRIDED_SLICE_INSTANTIATE_DIM)

#if GOOGLE_CUDA
TF_CALL_GPU_PROXY_TYPES(PREVENT_FOR_N_GPU);

TF_CALL_GPU_NUMBER_TYPES(DECLARE_FOR_N_GPU);
DECLARE_FOR_N_GPU(int32);
#endif  // END GOOGLE_CUDA

TF_CALL_ALL_TYPES(DECLARE_FOR_N_CPU);
DECLARE_FOR_N_CPU(bfloat16);

#undef INSTANTIATE
#undef DECLARE_FOR_N_CPU
#undef DECLARE_FOR_N_GPU

}  // end namespace tensorflow

#endif  // END STRIDED_SLICE_INSTANTIATE_DIM
#endif  // TENSORFLOW_KERNELS_SLICE_OP_H_

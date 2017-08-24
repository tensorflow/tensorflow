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

#if GOOGLE_CUDA

#define EIGEN_USE_GPU

#include "tensorflow/core/framework/numeric_types.h"
#include "tensorflow/core/kernels/reduction_ops.h"

namespace tensorflow {
namespace functor {

typedef Eigen::GpuDevice GPUDevice;

// Derive Index type. int (32-bit) or long (64-bit) depending on the
// compile-time configuration. "float" here is not relevant.
// TODO(zhifengc): Moves the definition to TTypes.
typedef TTypes<float>::Tensor::Index Index;

template <typename Reducer>
struct ReduceFunctor<GPUDevice, Reducer> {
  template <typename OUT_T, typename IN_T, typename ReductionAxes>
  static void Reduce(const GPUDevice& d, OUT_T out, IN_T in,
                     const ReductionAxes& reduction_axes,
                     const Reducer& reducer) {
    ReduceEigenImpl(d, To32Bit(out), To32Bit(in), reduction_axes, reducer);
  }

  template <typename OUT_T>
  static void FillIdentity(const GPUDevice& d, OUT_T out,
                           const Reducer& reducer) {
    FillIdentityEigenImpl(d, To32Bit(out), reducer);
  }
};

template <typename T>
struct ReduceFunctor<GPUDevice, Eigen::internal::MeanReducer<T>> {
  template <typename OUT_T, typename IN_T, typename ReductionAxes>
  static void Reduce(const GPUDevice& d, OUT_T out, IN_T in,
                     const ReductionAxes& reduction_axes,
                     const Eigen::internal::MeanReducer<T>& reducer) {
    typedef typename IN_T::Index Index;
    // Eigen sum reductions are much faster on GPU than mean reductions:
    // Simply trigger them by computing the sum of the weighted inputs.
    Index num_coeffs_to_reduce = 1;
    for (int i = 0; i < Eigen::internal::array_size<ReductionAxes>::value;
         ++i) {
      num_coeffs_to_reduce *= in.dimension(reduction_axes[i]);
    }
    T scale = T(1.0 / num_coeffs_to_reduce);
    out.device(d) = (in * scale).sum(reduction_axes);
  }

  template <typename OUT_T>
  static void FillIdentity(const GPUDevice& d, OUT_T out,
                           const Eigen::internal::MeanReducer<T>& reducer) {
    FillIdentityEigenImpl(d, To32Bit(out), reducer);
  }
};

// T: the data type
// REDUCER: the reducer functor
// NUM_AXES: the number of axes to reduce
// IN_DIMS: the number of dimensions of the input tensor
#define DEFINE(T, REDUCER, IN_DIMS, NUM_AXES)                        \
  template void ReduceFunctor<GPUDevice, REDUCER>::Reduce(           \
      const GPUDevice& d, TTypes<T, IN_DIMS - NUM_AXES>::Tensor out, \
      TTypes<T, IN_DIMS>::ConstTensor in,                            \
      const Eigen::array<Index, NUM_AXES>& reduction_axes,           \
      const REDUCER& reducer);

#define DEFINE_IDENTITY(T, REDUCER)                              \
  template void ReduceFunctor<GPUDevice, REDUCER>::FillIdentity( \
      const GPUDevice& d, TTypes<T>::Vec out, const REDUCER& reducer);

#define DEFINE_FOR_TYPE_AND_R(T, R) \
  DEFINE(T, R, 1, 1);               \
  DEFINE(T, R, 2, 1);               \
  DEFINE(T, R, 3, 1);               \
  DEFINE(T, R, 3, 2);               \
  DEFINE_IDENTITY(T, R)

#define DEFINE_FOR_ALL_REDUCERS(T)                           \
  DEFINE_FOR_TYPE_AND_R(T, Eigen::internal::SumReducer<T>);  \
  DEFINE_FOR_TYPE_AND_R(T, Eigen::internal::MeanReducer<T>); \
  DEFINE_FOR_TYPE_AND_R(T, Eigen::internal::MinReducer<T>);  \
  DEFINE_FOR_TYPE_AND_R(T, Eigen::internal::MaxReducer<T>);  \
  DEFINE_FOR_TYPE_AND_R(T, Eigen::internal::ProdReducer<T>)

DEFINE_FOR_ALL_REDUCERS(Eigen::half);
DEFINE_FOR_ALL_REDUCERS(int32);
DEFINE_FOR_ALL_REDUCERS(float);
DEFINE_FOR_ALL_REDUCERS(double);
#undef DEFINE_FOR_ALL_REDUCERS

DEFINE_FOR_TYPE_AND_R(complex64, Eigen::internal::SumReducer<complex64>);
DEFINE_FOR_TYPE_AND_R(complex128, Eigen::internal::SumReducer<complex128>);
DEFINE_FOR_TYPE_AND_R(complex64, Eigen::internal::MeanReducer<complex64>);
DEFINE_FOR_TYPE_AND_R(complex128, Eigen::internal::MeanReducer<complex128>);
DEFINE_FOR_TYPE_AND_R(complex64, Eigen::internal::ProdReducer<complex64>);
DEFINE_FOR_TYPE_AND_R(complex128, Eigen::internal::ProdReducer<complex128>);
DEFINE_FOR_TYPE_AND_R(bool, Eigen::internal::AndReducer);
DEFINE_FOR_TYPE_AND_R(bool, Eigen::internal::OrReducer);
#undef DEFINE_FOR_TYPE_AND_R

#undef DEFINE

}  // end namespace functor
}  // end namespace tensorflow

#endif  // GOOGLE_CUDA

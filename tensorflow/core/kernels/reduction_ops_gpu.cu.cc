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

#include "tensorflow/core/kernels/reduction_ops_gpu_kernels.h"

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
  static void Reduce(OpKernelContext* ctx, OUT_T out, IN_T in,
                     const ReductionAxes& reduction_axes,
                     const Reducer& reducer);
};

template <typename T>
struct ReduceFunctor<GPUDevice, Eigen::internal::SumReducer<T>> {
  template <typename OUT_T, typename IN_T, typename ReductionAxes>
  static void Reduce(OpKernelContext* ctx, OUT_T out, IN_T in,
                     const ReductionAxes& reduction_axes,
                     const Eigen::internal::SumReducer<T>& reducer) {
    ReduceImpl<T, cub::Sum, T*, T*, ReductionAxes>(
        ctx, (T*)out.data(), (T*)in.data(), in.rank(), in.dimension(0),
        in.rank() >= 2 ? in.dimension(1) : 1,
        in.rank() >= 3 ? in.dimension(2) : 1, out.rank(), reduction_axes,
        cub::Sum(), T(0));
  }

  template <typename OUT_T>
  static void FillIdentity(const GPUDevice& d, OUT_T out,
                           const Eigen::internal::SumReducer<T>& reducer) {
    FillIdentityEigenImpl(d, To32Bit(out), reducer);
  }
};

template <typename T>
struct ReduceFunctor<GPUDevice, Eigen::internal::MeanReducer<T>> {
  template <typename OUT_T, typename IN_T, typename ReductionAxes>
  static void Reduce(OpKernelContext* ctx, OUT_T out, IN_T in,
                     const ReductionAxes& reduction_axes,
                     const Eigen::internal::MeanReducer<T>& reducer) {
    int divisor = 1;
    if (out.rank() == 0)
      divisor = in.size();
    else if (out.rank() == 1 && in.rank() == 2 && reduction_axes[0] == 0)
      divisor = in.dimension(0);
    else if (out.rank() == 1 && in.rank() == 2 && reduction_axes[0] == 1)
      divisor = in.dimension(1);
    else if (out.rank() == 1 && in.rank() == 3 && reduction_axes[0] == 0 &&
             reduction_axes[1] == 2)
      divisor = in.dimension(0) * in.dimension(2);
    else if (out.rank() == 2 && in.rank() == 3 && reduction_axes[0] == 1)
      divisor = in.dimension(1);

    DividesBy<T> div_op(static_cast<T>(divisor));
    TransformOutputIterator<T, T, DividesBy<T>> itr((T*)out.data(), div_op);
    ReduceImpl<T, cub::Sum, TransformOutputIterator<T, T, DividesBy<T>>, T*,
               ReductionAxes>(ctx, itr, (T*)in.data(), in.rank(),
                              in.dimension(0),
                              in.rank() >= 2 ? in.dimension(1) : 1,
                              in.rank() >= 3 ? in.dimension(2) : 1, out.rank(),
                              reduction_axes, cub::Sum(), T(0));
  }

  template <typename OUT_T>
  static void FillIdentity(const GPUDevice& d, OUT_T out,
                           const Eigen::internal::MeanReducer<T>& reducer) {
    FillIdentityEigenImpl(d, To32Bit(out), reducer);
  }
};

template <>
struct ReduceFunctor<GPUDevice, Eigen::internal::MeanReducer<Eigen::half>> {
  template <typename OUT_T, typename IN_T, typename ReductionAxes>
  static void Reduce(OpKernelContext* ctx, OUT_T out, IN_T in,
                     const ReductionAxes& reduction_axes,
                     const Eigen::internal::MeanReducer<Eigen::half>& reducer) {
    float divisor = 1.f;
    if (out.rank() == 0)
      divisor = in.size();
    else if (out.rank() == 1 && in.rank() == 2 && reduction_axes[0] == 0)
      divisor = in.dimension(0);
    else if (out.rank() == 1 && in.rank() == 2 && reduction_axes[0] == 1)
      divisor = in.dimension(1);
    else if (out.rank() == 1 && in.rank() == 3 && reduction_axes[0] == 0 &&
             reduction_axes[1] == 2)
      divisor = in.dimension(0) * in.dimension(2);
    else if (out.rank() == 2 && in.rank() == 3 && reduction_axes[0] == 1)
      divisor = in.dimension(1);
    DividesBy<float, Eigen::half> div_op(divisor);

    typedef cub::TransformInputIterator<float, HalfToFloat, Eigen::half*>
        inputIterType;
    inputIterType input_itr((Eigen::half*)in.data(), HalfToFloat());

    typedef TransformOutputIterator<Eigen::half, float,
                                    DividesBy<float, Eigen::half>>
        outputIterType;
    outputIterType itr((Eigen::half*)out.data(), div_op);

    ReduceImpl<float, cub::Sum, outputIterType, inputIterType, ReductionAxes>(
        ctx, itr, input_itr, in.rank(), in.dimension(0),
        in.rank() >= 2 ? in.dimension(1) : 1,
        in.rank() >= 3 ? in.dimension(2) : 1, out.rank(), reduction_axes,
        cub::Sum(), 0.f);
  }

  template <typename OUT_T>
  static void FillIdentity(
      const GPUDevice& d, OUT_T out,
      const Eigen::internal::MeanReducer<Eigen::half>& reducer) {
    FillIdentityEigenImpl(d, To32Bit(out), reducer);
  }
};

template <typename T>
struct ReduceFunctor<GPUDevice, Eigen::internal::MaxReducer<T>> {
  template <typename OUT_T, typename IN_T, typename ReductionAxes>
  static void Reduce(OpKernelContext* ctx, OUT_T out, IN_T in,
                     const ReductionAxes& reduction_axes,
                     const Eigen::internal::MaxReducer<T>& reducer) {
    ReduceImpl<T, cub::Max, T*, T*, ReductionAxes>(
        ctx, (T*)out.data(), (T*)in.data(), in.rank(), in.dimension(0),
        in.rank() >= 2 ? in.dimension(1) : 1,
        in.rank() >= 3 ? in.dimension(2) : 1, out.rank(), reduction_axes,
        cub::Max(), std::numeric_limits<T>::min());
  }

  template <typename OUT_T>
  static void FillIdentity(const GPUDevice& d, OUT_T out,
                           const Eigen::internal::MaxReducer<T>& reducer) {
    FillIdentityEigenImpl(d, To32Bit(out), reducer);
  }
};

template <typename T>
struct ReduceFunctor<GPUDevice, Eigen::internal::MinReducer<T>> {
  template <typename OUT_T, typename IN_T, typename ReductionAxes>
  static void Reduce(OpKernelContext* ctx, OUT_T out, IN_T in,
                     const ReductionAxes& reduction_axes,
                     const Eigen::internal::MinReducer<T>& reducer) {
    ReduceImpl<T, cub::Min, T*, T*, ReductionAxes>(
        ctx, (T*)out.data(), (T*)in.data(), in.rank(), in.dimension(0),
        in.rank() >= 2 ? in.dimension(1) : 1,
        in.rank() >= 3 ? in.dimension(2) : 1, out.rank(), reduction_axes,
        cub::Min(), std::numeric_limits<T>::max());
  }

  template <typename OUT_T>
  static void FillIdentity(const GPUDevice& d, OUT_T out,
                           const Eigen::internal::MinReducer<T>& reducer) {
    FillIdentityEigenImpl(d, To32Bit(out), reducer);
  }
};

template <typename T>
struct ReduceFunctor<GPUDevice, Eigen::internal::ProdReducer<T>> {
  template <typename OUT_T, typename IN_T, typename ReductionAxes>
  static void Reduce(OpKernelContext* ctx, OUT_T out, IN_T in,
                     const ReductionAxes& reduction_axes,
                     const Eigen::internal::ProdReducer<T>& reducer) {
    ReduceImpl<T, Prod<T>, T*, T*, ReductionAxes>(
        ctx, (T*)out.data(), (T*)in.data(), in.rank(), in.dimension(0),
        in.rank() >= 2 ? in.dimension(1) : 1,
        in.rank() >= 3 ? in.dimension(2) : 1, out.rank(), reduction_axes,
        Prod<T>(), T(1));
  }

  template <typename OUT_T>
  static void FillIdentity(const GPUDevice& d, OUT_T out,
                           const Eigen::internal::ProdReducer<T>& reducer) {
    FillIdentityEigenImpl(d, To32Bit(out), reducer);
  }
};

template <>
struct ReduceFunctor<GPUDevice, Eigen::internal::AndReducer> {
  template <typename OUT_T, typename IN_T, typename ReductionAxes>
  static void Reduce(OpKernelContext* ctx, OUT_T out, IN_T in,
                     const ReductionAxes& reduction_axes,
                     const Eigen::internal::AndReducer& reducer) {
    ReduceImpl<bool, And, bool*, bool*, ReductionAxes>(
        ctx, (bool*)out.data(), (bool*)in.data(), in.rank(), in.dimension(0),
        in.rank() >= 2 ? in.dimension(1) : 1,
        in.rank() >= 3 ? in.dimension(2) : 1, out.rank(), reduction_axes, And(),
        true);
  }

  template <typename OUT_T>
  static void FillIdentity(const GPUDevice& d, OUT_T out,
                           const Eigen::internal::AndReducer& reducer) {
    FillIdentityEigenImpl(d, To32Bit(out), reducer);
  }
};

template <>
struct ReduceFunctor<GPUDevice, Eigen::internal::OrReducer> {
  template <typename OUT_T, typename IN_T, typename ReductionAxes>
  static void Reduce(OpKernelContext* ctx, OUT_T out, IN_T in,
                     const ReductionAxes& reduction_axes,
                     const Eigen::internal::OrReducer& reducer) {
    ReduceImpl<bool, Or, bool*, bool*, ReductionAxes>(
        ctx, (bool*)out.data(), (bool*)in.data(), in.rank(), in.dimension(0),
        in.rank() >= 2 ? in.dimension(1) : 1,
        in.rank() >= 3 ? in.dimension(2) : 1, out.rank(), reduction_axes, Or(),
        false);
  }

  template <typename OUT_T>
  static void FillIdentity(const GPUDevice& d, OUT_T out,
                           const Eigen::internal::OrReducer& reducer) {
    FillIdentityEigenImpl(d, To32Bit(out), reducer);
  }
};

// T: the data type
// REDUCER: the reducer functor
// NUM_AXES: the number of axes to reduce
// IN_DIMS: the number of dimensions of the input tensor
#define DEFINE(T, REDUCER, IN_DIMS, NUM_AXES)                          \
  template void ReduceFunctor<GPUDevice, REDUCER>::Reduce(             \
      OpKernelContext* ctx, TTypes<T, IN_DIMS - NUM_AXES>::Tensor out, \
      TTypes<T, IN_DIMS>::ConstTensor in,                              \
      const Eigen::array<Index, NUM_AXES>& reduction_axes,             \
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

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

#ifndef TENSORFLOW_CORE_KERNELS_REDUCTION_OPS_H_
#define TENSORFLOW_CORE_KERNELS_REDUCTION_OPS_H_

// Functor definitions for Reduction ops, must be compilable by nvcc.

#include <iostream>
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_types.h"

namespace tensorflow {
namespace functor {

template <typename Reducer>
struct ReducerTraits {
  enum { IsScalarIdentity = true };
};

// Dummy class used for template specialization for mean reduction, which is
// accomplished by SumReducer and on-the-fly division by the reduction factor.
template <typename Scalar>
struct MeanReducer {
  Scalar initialize() const { return Scalar(0); }
};

// Dummy class used for template specialization for l2-norm reduction.
template <typename Scalar>
struct EuclideanNormReducer {
  Scalar initialize() const { return Scalar(0); }
};

template <typename Scalar>
struct ReducerTraits<EuclideanNormReducer<Scalar>> {
  enum { IsScalarIdentity = false };
};

template <typename Device, typename OUT_T, typename IN_T,
          typename ReductionAxes, typename Reducer>
struct ReduceEigenImpl {
  void operator()(const Device& d, OUT_T out, IN_T in,
                  const ReductionAxes& reduction_axes, const Reducer& reducer) {
    out.device(d) = in.reduce(reduction_axes, reducer);
  }
};

template <typename Device, typename OUT_T, typename IN_T,
          typename ReductionAxes, typename Scalar>
struct ReduceEigenImpl<Device, OUT_T, IN_T, ReductionAxes,
                       functor::MeanReducer<Scalar>> {
  void operator()(const Device& d, OUT_T out, IN_T in,
                  const ReductionAxes& reduction_axes,
                  const functor::MeanReducer<Scalar>& reducer) {
    static_assert(std::is_same<Scalar, typename OUT_T::Scalar>::value, "");
    Eigen::internal::SumReducer<Scalar> sum_reducer;
    out.device(d) = in.reduce(reduction_axes, sum_reducer) /
                    static_cast<Scalar>(in.size() / out.size());
  }
};

// Specialization for which we do the reduction in IntermediateType to
// avoid integer overflow.
#define CASTING_SPECIALIZATION(ScalarType, IntermediateType)                  \
  template <typename Device, typename OUT_T, typename IN_T,                   \
            typename ReductionAxes>                                           \
  struct ReduceEigenImpl<Device, OUT_T, IN_T, ReductionAxes,                  \
                         functor::MeanReducer<ScalarType>> {                  \
    void operator()(const Device& d, OUT_T out, IN_T in,                      \
                    const ReductionAxes& reduction_axes,                      \
                    const functor::MeanReducer<ScalarType>& reducer) {        \
      static_assert(std::is_same<ScalarType, typename OUT_T::Scalar>::value,  \
                    "");                                                      \
      Eigen::internal::SumReducer<IntermediateType> sum_reducer;              \
      out.device(d) = (in.template cast<IntermediateType>().reduce(           \
                           reduction_axes, sum_reducer) /                     \
                       static_cast<IntermediateType>(in.size() / out.size())) \
                          .template cast<ScalarType>();                       \
    }                                                                         \
  }

CASTING_SPECIALIZATION(uint8, uint64);
CASTING_SPECIALIZATION(uint16, uint64);
CASTING_SPECIALIZATION(uint32, uint64);
CASTING_SPECIALIZATION(int8, int64);
CASTING_SPECIALIZATION(int16, int64);
CASTING_SPECIALIZATION(int32, int64);
#undef CASTING_SPECIALIZATION

// TODO(rmlarsen): Refactor this such that taking the sqrt can be optional
// controlled by an attribute.
template <typename Device, typename OUT_T, typename IN_T,
          typename ReductionAxes, typename Scalar>
struct ReduceEigenImpl<Device, OUT_T, IN_T, ReductionAxes,
                       functor::EuclideanNormReducer<Scalar>> {
  void operator()(const Device& d, OUT_T out, IN_T in,
                  const ReductionAxes& reduction_axes,
                  const functor::EuclideanNormReducer<Scalar>& reducer) {
    static_assert(std::is_same<Scalar, typename OUT_T::Scalar>::value, "");
    Eigen::internal::SumReducer<Scalar> sum_reducer;
    out.device(d) =
        (in * in.conjugate()).reduce(reduction_axes, sum_reducer).sqrt();
  }
};

template <typename Device, typename OUT_T, typename IN_T,
          typename ReductionAxes>
struct ReduceEigenImpl<Device, OUT_T, IN_T, ReductionAxes,
                       functor::EuclideanNormReducer<bfloat16>> {
  void operator()(const Device& d, OUT_T out, IN_T in,
                  const ReductionAxes& reduction_axes,
                  const functor::EuclideanNormReducer<bfloat16>& reducer) {
    static_assert(std::is_same<bfloat16, typename OUT_T::Scalar>::value, "");
    Eigen::internal::SumReducer<float> sum_reducer;
    auto in_as_float = in.template cast<float>();
    out.device(d) = (in_as_float * in_as_float.conjugate())
                        .reduce(reduction_axes, sum_reducer)
                        .sqrt()
                        .template cast<bfloat16>();
  }
};

// For most reducers, the identity is Reducer::initialize()
template <typename Reducer>
struct Identity {
  static auto identity(const Reducer& reducer)
      -> decltype(reducer.initialize()) {
    return reducer.initialize();
  }
};

// MeanReducer is a special case, since it doesn't technically have an identity.
// Thus, ideally we'd return nan.  However, mean is instantiated for integer
// types as well, so we do the nan override only for floating point types.
#define FIX_MEAN_IDENTITY(T)                            \
  template <>                                           \
  struct Identity<functor::MeanReducer<T>> {            \
    static T identity(const functor::MeanReducer<T>&) { \
      return Eigen::NumTraits<T>::quiet_NaN();          \
    }                                                   \
  };
FIX_MEAN_IDENTITY(Eigen::half)
FIX_MEAN_IDENTITY(float)
FIX_MEAN_IDENTITY(double)
#undef FIX_MEAN_IDENTITY

template <typename Device, typename OUT_T, typename Reducer>
void FillIdentityEigenImpl(const Device& d, OUT_T out, const Reducer& reducer) {
  out.device(d) = out.constant(Identity<Reducer>::identity(reducer));
}

template <typename Device, typename Reducer>
struct ReduceFunctor {
  template <typename OUT_T, typename IN_T, typename ReductionAxes>
  static void Reduce(OpKernelContext* ctx, OUT_T out, IN_T in,
                     const ReductionAxes& reduction_axes,
                     const Reducer& reducer);

  template <typename OUT_T>
  static void FillIdentity(const Device& d, OUT_T out, const Reducer& reducer);
};

}  // namespace functor
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_REDUCTION_OPS_H_

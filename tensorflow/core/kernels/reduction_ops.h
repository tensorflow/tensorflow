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

template <typename Device, typename OUT_T, typename IN_T,
          typename ReductionAxes, typename Reducer>
void ReduceEigenImpl(const Device& d, OUT_T out, IN_T in,
                     const ReductionAxes& reduction_axes,
                     const Reducer& reducer) {
  out.device(d) = in.reduce(reduction_axes, reducer);
}

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
#define FIX_MEAN_IDENTITY(T)                                    \
  template <>                                                   \
  struct Identity<Eigen::internal::MeanReducer<T>> {            \
    static T identity(const Eigen::internal::MeanReducer<T>&) { \
      return Eigen::NumTraits<T>::quiet_NaN();                  \
    }                                                           \
  };
FIX_MEAN_IDENTITY(Eigen::half)
FIX_MEAN_IDENTITY(float)
FIX_MEAN_IDENTITY(double)
FIX_MEAN_IDENTITY(complex64)
FIX_MEAN_IDENTITY(complex128)
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

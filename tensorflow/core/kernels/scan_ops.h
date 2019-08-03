/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_KERNELS_SCAN_OPS_H_
#define TENSORFLOW_CORE_KERNELS_SCAN_OPS_H_

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/tensor_types.h"

namespace tensorflow {
namespace functor {

typedef Eigen::Index Index;

template <typename Device, typename Reducer, typename T>
struct Scan {
  void operator()(const Device& d, typename TTypes<T, 3>::ConstTensor in,
                  typename TTypes<T, 3>::Tensor out, const Reducer& reducer,
                  const bool reverse, const bool exclusive) {
    // Perform the reverse ops directly with Eigen, which avoids copying the
    // tensor twice compared to using individual ops.
    Eigen::array<bool, 3> dims;
    dims[0] = false;
    dims[1] = reverse;
    dims[2] = false;
    To32Bit(out).device(d) =
        To32Bit(in).reverse(dims).scan(1, reducer, exclusive).reverse(dims);
  }
};

template <typename T>
struct LogSumExp {
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE T operator()(const T& a,
                                                     const T& b) const {
    Eigen::internal::scalar_sum_op<T> sum_op;
    Eigen::internal::scalar_exp_op<T> exp_op;
    Eigen::internal::scalar_log_op<T> log_op;
    Eigen::internal::scalar_max_op<T> max_op;
    Eigen::internal::scalar_min_op<T> min_op;
    Eigen::internal::scalar_log1p_op<T> log1p_op;
    Eigen::internal::scalar_difference_op<T> diff_op;

    auto mi = min_op(a, b);
    auto ma = max_op(a, b);

    return sum_op(log1p_op(exp_op(diff_op(mi, ma))), ma);
  }
};

template <typename T>
struct LogSumExpReducer {
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void reduce(const T t, T* accum) const {
    LogSumExp<T> logsumexp;
    *accum = logsumexp(*accum, t);
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE T initialize() const {
    return Eigen::NumTraits<T>::lowest();
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE T finalize(const T accum) const {
    return accum;
  }
};

}  // namespace functor
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_SCAN_OPS_H_

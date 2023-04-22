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

// TODO(b/154339590): Needs to be vectorized.
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
    auto mi = Eigen::internal::scalar_min_op<T>()(a, b);
    auto ma = Eigen::internal::scalar_max_op<T>()(a, b);

    auto sub = Eigen::internal::scalar_difference_op<T>();
    auto add = Eigen::internal::scalar_sum_op<T>();
    auto exp = Eigen::internal::scalar_exp_op<T>();
    auto log1p = Eigen::internal::scalar_log1p_op<T>();
    auto cmp_lt =
        Eigen::internal::scalar_cmp_op<T, T, Eigen::internal::cmp_LT>();

    auto logsumexp = add(log1p(exp(sub(mi, ma))), ma);
    return cmp_lt(ma, Eigen::NumTraits<T>::lowest()) ? ma : logsumexp;
  }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE T packetOp(const T& a,
                                                   const T& b) const {
    auto mi = Eigen::internal::pmin(a, b);
    auto ma = Eigen::internal::pmax(a, b);
    using Eigen::internal::padd;
    using Eigen::internal::pcmp_lt;
    using Eigen::internal::pexp;
    using Eigen::internal::plog1p;
    using Eigen::internal::pset1;
    using Eigen::internal::psub;

    auto logsumexp = padd(plog1p(pexp(psub(mi, ma))), ma);
    return pselect(pcmp_lt(ma, pset1(Eigen::NumTraits<T>::lowest())), ma,
                   logsumexp);
  }
};

template <typename T>
struct LogSumExpReducer {
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void reduce(const T t, T* accum) const {
    LogSumExp<T> logsumexp;
    *accum = logsumexp(*accum, t);
  }

  template <typename Packet>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void reducePacket(const Packet& p,
                                                          Packet* accum) const {
    LogSumExp<T> logsumexp;
    *accum = logsumexp.packetOp(*accum, p);
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE T initialize() const {
    return -Eigen::NumTraits<T>::infinity();
  }

  template <typename Packet>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Packet initializePacket() const {
    return Eigen::internal::pset1(initialize());
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE T finalize(const T accum) const {
    return accum;
  }

  template <typename Packet>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Packet
  finalizePacket(const Packet& vaccum) const {
    return vaccum;
  }

  template <typename Packet>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE T
  finalizeBoth(const T saccum, const Packet& vaccum) const {
    auto max_reducer = Eigen::internal::MaxReducer<T, Eigen::PropagateNaN>();
    auto sum_reducer = Eigen::internal::SumReducer<T>();
    auto exp = Eigen::internal::scalar_exp_op<T>();
    auto cmp_lt =
        Eigen::internal::scalar_cmp_op<T, T, Eigen::internal::cmp_LT>();
    auto log = Eigen::internal::scalar_log_op<T>();
    auto add = Eigen::internal::scalar_sum_op<T>();

    using Eigen::internal::pexp;
    using Eigen::internal::psub;

    // `ma = max(x1, ..., xn)`
    // If the max of all of the `xi` is `-infinity` then the result is
    // -infinity. If the max is larger than `-infinity` then it's safe to use
    // for normalization even if the other elements are `-infinity`.
    //
    // `logsumexp(x1, ..., xn) = ma + log (exp(x1 - ma) + ... + exp(xn - ma))`
    auto ma = max_reducer.finalizeBoth(saccum, vaccum);
    auto logsumexp = add(log(sum_reducer.finalizeBoth(
                             exp(saccum - ma), pexp(psub(vaccum, pset1(ma))))),
                         ma);
    return cmp_lt(ma, Eigen::NumTraits<T>::lowest()) ? initialize() : logsumexp;
  }
};

}  // namespace functor
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_SCAN_OPS_H_

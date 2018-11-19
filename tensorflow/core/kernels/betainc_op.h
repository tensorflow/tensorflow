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

#ifndef TENSORFLOW_CORE_KERNELS_BETAINC_OP_H_
#define TENSORFLOW_CORE_KERNELS_BETAINC_OP_H_
// Functor definition for BetaincOp, must be compilable by nvcc.

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/tensor_types.h"

namespace tensorflow {
namespace functor {

// Functor used by BetaincOp to do the computations.
template <typename Device, typename T, int NDIM>
struct Betainc {
  void operator()(const Device& d, typename TTypes<T, NDIM>::ConstTensor a,
                  typename TTypes<T, NDIM>::ConstTensor b,
                  typename TTypes<T, NDIM>::ConstTensor x,
                  typename TTypes<T, NDIM>::Tensor output) {
    output.device(d) = Eigen::betainc(a, b, x);
  }

  void BCast(const Device& d, typename TTypes<T, NDIM>::ConstTensor a,
             const typename Eigen::array<Eigen::DenseIndex, NDIM>& bcast_a,
             typename TTypes<T, NDIM>::ConstTensor b,
             const typename Eigen::array<Eigen::DenseIndex, NDIM>& bcast_b,
             typename TTypes<T, NDIM>::ConstTensor x,
             const typename Eigen::array<Eigen::DenseIndex, NDIM>& bcast_x,
             typename TTypes<T, NDIM>::Tensor output) {
    output.device(d) = Eigen::betainc(
        a.broadcast(bcast_a), b.broadcast(bcast_b), x.broadcast(bcast_x));
  }
};

}  // namespace functor
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_BETAINC_OP_H_

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

#ifndef THIRD_PARTY_TENSORFLOW_CORE_KERNELS_EIGEN_SOFTMAX_H_
#define THIRD_PARTY_TENSORFLOW_CORE_KERNELS_EIGEN_SOFTMAX_H_

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace Eigen {

/** SoftMax
  * \ingroup CXX11_NeuralNetworks_Module
  *
  * \brief Applies a softmax
  *
  * The input parameter is expected to be a col-major tensor with a rank of 2 (depth and other).
  *
  * The result can be assigned to a tensor of rank and dimensions equal to that of the input. The result will be laid out in col-major order.
  *
*/

namespace {
struct SoftmaxOp {
  SoftmaxOp(const float beta) : beta_(beta) { }

  template <typename Input>
  typename Input::Dimensions dimensions(const Input& input) const {
    return input.dimensions();
  }

  template <typename Input, typename Output, typename Device>
  void eval(const Input& input, Output& output, const Device& device) const
  {
#if !defined(EIGEN_HAS_INDEX_LIST)
    // nvcc doesn't support cxx11
    Eigen::array<typename internal::traits<Input>::Index, 1> depth_dim;
    depth_dim[0] = 0;
    Eigen::array<typename internal::traits<Input>::Index, 2> bcast;
    bcast[0] = dimensions(input)[0];
    bcast[1] = 1;
    DSizes<typename internal::traits<Input>::Index, 2> dims2d;
    dims2d[0] = 1;
    dims2d[1] = dimensions(input)[1];
#else
    // Take advantage of cxx11 to give the compiler information it can use to
    // optimize the code.
    Eigen::IndexList<Eigen::type2index<0>> depth_dim;
    Eigen::IndexList<int, Eigen::type2index<1>> bcast;
    bcast.set(0, dimensions(input)[0]);
    Eigen::IndexList<Eigen::type2index<1>, typename internal::traits<Input>::Index> dims2d;
    dims2d.set(1, dimensions(input)[1]);
#endif

    output.device(device) = ((input - input.maximum(depth_dim).eval().reshape(dims2d).broadcast(bcast)) * beta_).exp();
    output.device(device) = output / (output.sum(depth_dim).eval().reshape(dims2d).broadcast(bcast));
  }

 private:
  const float beta_;
};
}


template <typename Input>
EIGEN_ALWAYS_INLINE
static const TensorCustomUnaryOp<const SoftmaxOp, const Input>
SoftMax(const Input& input, const float beta)
{
  EIGEN_STATIC_ASSERT(internal::traits<Input>::Layout == ColMajor, YOU_MADE_A_PROGRAMMING_MISTAKE);
  EIGEN_STATIC_ASSERT(internal::traits<Input>::NumDimensions == 2, YOU_MADE_A_PROGRAMMING_MISTAKE);

  const SoftmaxOp op(beta);
  return input.customOp(op);
}

} // end namespace Eigen

#endif  // THIRD_PARTY_TENSORFLOW_CORE_KERNELS_EIGEN_SOFTMAX_H_

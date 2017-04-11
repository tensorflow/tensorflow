// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2014 Benoit Steiner <benoit.steiner.goog@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
#ifndef EIGEN_CXX11_NEURAL_NETWORKS_SOFTMAX_H
#define EIGEN_CXX11_NEURAL_NETWORKS_SOFTMAX_H

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
class SoftmaxOp {
 public:
  EIGEN_ALWAYS_INLINE SoftmaxOp(const float beta) : beta_(beta) { }

  template <typename Input> EIGEN_ALWAYS_INLINE
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

#endif // EIGEN_CXX11_NEURAL_NETWORKS_SOFTMAX_H

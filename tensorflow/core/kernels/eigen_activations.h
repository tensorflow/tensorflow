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

#ifndef TENSORFLOW_CORE_KERNELS_EIGEN_ACTIVATIONS_H_
#define TENSORFLOW_CORE_KERNELS_EIGEN_ACTIVATIONS_H_

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace Eigen {

/** scalar_sigmoid_fast_derivative_op
 * \ingroup CXX11_NeuralNetworks_Module
 * \brief Template functor to compute the fast derivative of a sigmoid
 *
 * Input should be the backpropagated gradient.
 *
 * \sa class CwiseUnaryOp, Cwise::sigmoid_fast_derivative()
 */
template <typename T>
struct scalar_sigmoid_fast_derivative_op {
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE T operator()(const T& y) const {
    const T one = T(1);
    return (one - y) * y;
  }

  template <typename Packet>
  inline Packet packetOp(const Packet& y) const {
    const Packet one = internal::pset1<Packet>(1);
    return internal::pmul(internal::psub(one, y), y);
  }
};

namespace internal {
template <typename T>
struct functor_traits<scalar_sigmoid_fast_derivative_op<T> > {
  enum {
    Cost = NumTraits<T>::AddCost * 2 + NumTraits<T>::MulCost,
    PacketAccess = packet_traits<T>::HasAdd && packet_traits<T>::HasMul &&
                   packet_traits<T>::HasNegate
  };
};
}  // namespace internal

/** scalar_tanh_fast_derivative_op
 * \ingroup CXX11_NeuralNetworks_Module
 * \brief Template functor to compute the fast derivative of a tanh
 *
 * Input should be the backpropagated gradient.
 *
 * \sa class CwiseUnaryOp, Cwise::tanh_fast_derivative()
 */
template <typename T>
struct scalar_tanh_fast_derivative_op {
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE T operator()(const T& y) const {
    const T one = T(1);
    return one - (y * y);
  }

  template <typename Packet>
  inline Packet packetOp(const Packet& y) const {
    const Packet one = internal::pset1<Packet>(1);
    return internal::psub(one, internal::pmul(y, y));
  }
};

namespace internal {
template <typename T>
struct functor_traits<scalar_tanh_fast_derivative_op<T> > {
  enum {
    Cost = NumTraits<T>::AddCost * 2 + NumTraits<T>::MulCost * 1,
    PacketAccess = packet_traits<T>::HasAdd && packet_traits<T>::HasMul &&
                   packet_traits<T>::HasNegate
  };
};
}  // namespace internal

/**
 * \ingroup CXX11_NeuralNetworks_Module
 * \brief Template functor to clip the magnitude of the first scalar.
 *
 * \sa class CwiseBinaryOp, MatrixBase::Clip
 */
template <typename Scalar>
struct scalar_clip_op {
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const Scalar
  operator()(const Scalar& a, const Scalar& b) const {
    return numext::mini(numext::maxi(a, -b), b);
  }
  template <typename Packet>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const Packet
  packetOp(const Packet& a, const Packet& b) const {
    return internal::pmin(internal::pmax(a, internal::pnegate(b)), b);
  }
};

namespace internal {
template <typename Scalar>
struct functor_traits<scalar_clip_op<Scalar> > {
  enum {
    Cost = NumTraits<Scalar>::AddCost * 3,
    PacketAccess = packet_traits<Scalar>::HasMax &&
                   packet_traits<Scalar>::HasMin &&
                   packet_traits<Scalar>::HasNegate
  };
};
}  // namespace internal

}  // end namespace Eigen

#endif  // TENSORFLOW_CORE_KERNELS_EIGEN_ACTIVATIONS_H_

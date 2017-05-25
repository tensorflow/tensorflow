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

#ifndef TENSORFLOW_KERNELS_CWISE_OPS_GRADIENTS_H_
#define TENSORFLOW_KERNELS_CWISE_OPS_GRADIENTS_H_

#define EIGEN_USE_THREADS
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/kernels/cwise_ops.h"

namespace Eigen {
namespace internal {

// Gradient for the tanh function
template <typename T>
struct scalar_tanh_gradient_op {
  EIGEN_EMPTY_STRUCT_CTOR(scalar_tanh_gradient_op)
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const T
  operator()(const T& output, const T& output_gradient) const {
    return output_gradient * (T(1) - output * output);
  }
  template <typename Packet>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const Packet
  packetOp(const Packet& output, const Packet& output_gradient) const {
    return pmul(output_gradient,
                psub(pset1<Packet>(T(1)), pmul(output, output)));
  }
};
template <typename T>
struct functor_traits<scalar_tanh_gradient_op<T>> {
  enum {
    Cost = NumTraits<T>::AddCost + 2 * NumTraits<T>::MulCost,
    PacketAccess = packet_traits<T>::HasSub && packet_traits<T>::HasMul,
  };
};

// Gradient for the sigmoid function
template <typename T>
struct scalar_sigmoid_gradient_op {
  EIGEN_EMPTY_STRUCT_CTOR(scalar_sigmoid_gradient_op)
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const T
  operator()(const T& output, const T& output_gradient) const {
    return output_gradient * output * (T(1) - output);
  }
  template <typename Packet>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const Packet
  packetOp(const Packet& output, const Packet& output_gradient) const {
    return pmul(output_gradient,
                pmul(output, psub(pset1<Packet>(T(1)), output)));
  }
};
template <typename T>
struct functor_traits<scalar_sigmoid_gradient_op<T>> {
  enum {
    Cost = NumTraits<T>::AddCost + 2 * NumTraits<T>::MulCost,
    PacketAccess = packet_traits<T>::HasSub && packet_traits<T>::HasMul,
  };
};

// Gradient for the inverse function
template <typename T>
struct scalar_inverse_gradient_op {
  EIGEN_EMPTY_STRUCT_CTOR(scalar_inverse_gradient_op)
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const T
  operator()(const T& output, const T& output_gradient) const {
    const T out_conj = numext::conj(output);
    return -output_gradient * out_conj * out_conj;
  }
  template <typename Packet>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const Packet
  packetOp(const Packet& output, const Packet& output_gradient) const {
    const Packet out_conj = pconj(output);
    return pnegate(pmul(output_gradient, pmul(out_conj, out_conj)));
  }
};
template <typename T>
struct functor_traits<scalar_inverse_gradient_op<T>> {
  enum {
    Cost = NumTraits<T>::AddCost + 2 * NumTraits<T>::MulCost,
    PacketAccess = packet_traits<T>::HasMul,
  };
};

// Gradient for the sqrt function
template <typename T>
struct scalar_sqrt_gradient_op {
  EIGEN_EMPTY_STRUCT_CTOR(scalar_sqrt_gradient_op)
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const T
  operator()(const T& output, const T& output_gradient) const {
    const T out_conj = numext::conj(output);
    return static_cast<T>(0.5) * output_gradient / out_conj;
  }
  template <typename Packet>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const Packet
  packetOp(const Packet& output, const Packet& output_gradient) const {
    const Packet const_half = pset1<Packet>(static_cast<T>(0.5));
    const Packet out_conj = pconj(output);
    return pdiv(pmul(const_half, output_gradient), out_conj);
  }
};
template <typename T>
struct functor_traits<scalar_sqrt_gradient_op<T>> {
  enum {
    PacketAccess = packet_traits<T>::HasMul & packet_traits<T>::HasDiv,
    Cost = NumTraits<T>::MulCost + scalar_div_cost<T, PacketAccess>::value,
  };
};

// Gradient for the rsqrt function
template <typename T>
struct scalar_rsqrt_gradient_op {
  EIGEN_EMPTY_STRUCT_CTOR(scalar_rsqrt_gradient_op)
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const T
  operator()(const T& output, const T& output_gradient) const {
    const T out_conj = numext::conj(output);
    return static_cast<T>(-0.5) * (output_gradient * out_conj) *
           (out_conj * out_conj);
  }
  template <typename Packet>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const Packet
  packetOp(const Packet& output, const Packet& output_gradient) const {
    const Packet const_half = pset1<Packet>(static_cast<T>(-0.5));
    const Packet out_conj = pconj(output);
    return pmul(const_half, pmul(pmul(output_gradient, out_conj),
                                 pmul(out_conj, out_conj)));
  }
};
template <typename T>
struct functor_traits<scalar_rsqrt_gradient_op<T>> {
  enum {
    Cost = 4 * NumTraits<T>::MulCost,
    PacketAccess = packet_traits<T>::HasMul,
  };
};

}  // end namespace internal
}  // end namespace Eigen

namespace tensorflow {

namespace functor {

template <typename Device, typename Functor>
struct SimpleBinaryFunctor {
  void operator()(const Device& d, typename Functor::tout_type out,
                  typename Functor::tin_type in0,
                  typename Functor::tin_type in1);
};

// Partial specialization of BinaryFunctor for CPU devices
typedef Eigen::ThreadPoolDevice CPUDevice;

template <typename Functor>
struct SimpleBinaryFunctor<CPUDevice, Functor> {
  void operator()(const CPUDevice& d, typename Functor::tout_type out,
                  typename Functor::tin_type in0,
                  typename Functor::tin_type in1) {
    out.device(d) = in0.binaryExpr(in1, typename Functor::func());
  }
};


#ifdef TENSORFLOW_USE_SYCL
// Partial specialization of BinaryFunctor for SYCL devices
typedef Eigen::SyclDevice SYCLDevice;
template <typename Functor>
struct SimpleBinaryFunctor<SYCLDevice, Functor> {
  void operator()(const SYCLDevice& d, typename Functor::tout_type out,
                  typename Functor::tin_type in0,
                  typename Functor::tin_type in1) {
    out.device(d) = in0.binaryExpr(in1, typename Functor::func());
  }
};

#endif // TENSORFLOW_USE_SYCL

template <typename T>
struct tanh_grad : base<T, Eigen::internal::scalar_tanh_gradient_op<T>> {};

template <typename T>
struct sigmoid_grad : base<T, Eigen::internal::scalar_sigmoid_gradient_op<T>> {
};

template <typename T>
struct inverse_grad : base<T, Eigen::internal::scalar_inverse_gradient_op<T>> {
};

template <typename T>
struct sqrt_grad : base<T, Eigen::internal::scalar_sqrt_gradient_op<T>> {};

template <typename T>
struct rsqrt_grad : base<T, Eigen::internal::scalar_rsqrt_gradient_op<T>> {};

}  // end namespace functor

}  // end namespace tensorflow
#endif  // TENSORFLOW_KERNELS_CWISE_OPS_GRADIENTS_H_

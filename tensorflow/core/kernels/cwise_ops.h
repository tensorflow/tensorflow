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

#ifndef TENSORFLOW_CORE_KERNELS_CWISE_OPS_H_
#define TENSORFLOW_CORE_KERNELS_CWISE_OPS_H_

#include <cmath>
#include <functional>
#include <type_traits>

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

#include "tensorflow/core/framework/bounds_check.h"
#include "tensorflow/core/framework/numeric_types.h"
#include "tensorflow/core/framework/tensor_types.h"

namespace Eigen {
namespace internal {

#if GOOGLE_CUDA
template <>
struct scalar_arg_op<std::complex<float>> {
  EIGEN_EMPTY_STRUCT_CTOR(scalar_arg_op)
  typedef typename Eigen::NumTraits<std::complex<float>>::Real result_type;
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE float operator()(
      const std::complex<float>& a) const {
    return ::atan2f(a.imag(), a.real());
  }
};

template <>
struct scalar_arg_op<std::complex<double>> {
  EIGEN_EMPTY_STRUCT_CTOR(scalar_arg_op)
  typedef typename Eigen::NumTraits<std::complex<double>>::Real result_type;
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE double operator()(
      const std::complex<double>& a) const {
    return ::atan2(a.imag(), a.real());
  }
};
#endif

#if EIGEN_HAS_CXX11_MATH == 0
template <typename T>
struct scalar_asinh_op {
  EIGEN_EMPTY_STRUCT_CTOR(scalar_asinh_op)
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE T operator()(const T& a) const {
    return std::asinh(a);
  }
};
template <typename T>
struct functor_traits<scalar_asinh_op<T>> {
  enum { Cost = 5 * NumTraits<T>::MulCost, PacketAccess = false };
};

template <typename T>
struct scalar_acosh_op {
  EIGEN_EMPTY_STRUCT_CTOR(scalar_acosh_op)
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE T operator()(const T& a) const {
    return std::acosh(a);
  }
};
template <typename T>
struct functor_traits<scalar_acosh_op<T>> {
  enum { Cost = 5 * NumTraits<T>::MulCost, PacketAccess = false };
};

template <typename T>
struct scalar_atanh_op {
  EIGEN_EMPTY_STRUCT_CTOR(scalar_atanh_op)
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE T operator()(const T& a) const {
    return std::atanh(a);
  }
};
template <typename T>
struct functor_traits<scalar_atanh_op<T>> {
  enum { Cost = 5 * NumTraits<T>::MulCost, PacketAccess = false };
};
#endif

template <typename Scalar, typename Exponent>
struct safe_scalar_binary_pow_op {
  static_assert(std::is_integral<Scalar>::value, "Integer type expected");
  static_assert(std::is_integral<Exponent>::value &&
                    std::is_signed<Exponent>::value,
                "Signed integer type expected");

  bool* const error;

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE safe_scalar_binary_pow_op(bool* error)
      : error(error) {}

  EIGEN_DEVICE_FUNC inline Scalar operator()(const Scalar& a,
                                             const Exponent& b) const {
    const Exponent safe_b = tensorflow::internal::SubtleMustCopy(b);
    if (TF_PREDICT_TRUE(safe_b >= 0)) {
      return numext::pow(a, safe_b);
    } else {
      *error = true;
      return 0;
    }
  }
};

template <typename Scalar, typename Exponent>
struct functor_traits<safe_scalar_binary_pow_op<Scalar, Exponent>> {
  enum { Cost = 5 * NumTraits<Scalar>::MulCost, PacketAccess = false };
};

template <typename T, typename DivOrMod>
struct safe_div_or_mod_op {
  static_assert(std::is_integral<T>::value, "Integer type expected");

  bool* const error;

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE safe_div_or_mod_op(bool* error)
      : error(error) {}

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE T operator()(const T& a,
                                                     const T& b) const {
    const T safe_b = tensorflow::internal::SubtleMustCopy(b);
    if (TF_PREDICT_TRUE(safe_b != 0)) {
      return DivOrMod()(a, safe_b);
    } else {
      *error = true;
      return 0;
    }
  }
};

template <typename T, typename DivOrMod>
struct functor_traits<safe_div_or_mod_op<T, DivOrMod>> {
  enum {
    Cost = functor_traits<DivOrMod>::Cost + NumTraits<T>::AddCost,
    PacketAccess = false,
  };
};

template <typename T, typename Binary>
struct no_nan_op {
  EIGEN_EMPTY_STRUCT_CTOR(no_nan_op)
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE T operator()(const T& a,
                                                     const T& b) const {
    if (b != T(0)) {
      return Binary()(a, b);
    } else {
      return T(0);
    }
  }
  template <typename Packet>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Packet packetOp(const Packet& a,
                                                        const Packet& b) const {
    const Packet mask = pcmp_eq(b, pzero(b));
    const Packet quotient = Binary().packetOp(a, b);
    return pandnot(quotient, mask);
  }
};

template <typename T>
struct div_no_nan_op : public no_nan_op<T, scalar_quotient_op<T>> {
  EIGEN_EMPTY_STRUCT_CTOR(div_no_nan_op)
};

template <typename T>
struct functor_traits<div_no_nan_op<T>> {
  enum {
    Cost = functor_traits<scalar_quotient_op<T>>::Cost + NumTraits<T>::AddCost,
#if TENSORFLOW_USE_ROCM
    PacketAccess = false,
#else
    PacketAccess = true,
#endif  // TENSORFLOW_USE_ROCM
  };
};

template <typename T>
struct mul_no_nan_op : public no_nan_op<T, scalar_product_op<T>> {
  EIGEN_EMPTY_STRUCT_CTOR(mul_no_nan_op)
};

template <typename T>
struct functor_traits<mul_no_nan_op<T>> {
  enum {
    Cost = functor_traits<scalar_product_op<T>>::Cost + NumTraits<T>::AddCost,
#if TENSORFLOW_USE_ROCM
    PacketAccess = false,
#else
    PacketAccess = true,
#endif  // TENSORFLOW_USE_ROCM
  };
};

// scalar_left and scalar_right are template helpers to partially
// apply a binary function.
//
// Suppose Binary is a binary functor f(x, y), scalar_left<> is a
// unary functor g_x(y) = f(x, y), where x is provided via the
// constructor. Similarly, scalar_right<> is a unary functor g_y(x) =
// f(x, y).

template <typename Tout, typename Tin, typename Binary,
          bool is_scalar_in_host_memory = false>
struct scalar_left : private Binary {
  using result_type = Tout;
  using TinPacket = typename Eigen::internal::packet_traits<Tin>::type;

  const Tin* left;
  TinPacket left_packet;  // initialized iff is_scalar_in_host_memory == true

  EIGEN_DEVICE_FUNC inline scalar_left(const scalar_left& other) = default;

  template <typename... Args>
  EIGEN_DEVICE_FUNC inline explicit scalar_left(const Tin* c, Args... args)
      : Binary(args...), left(c) {
    if (is_scalar_in_host_memory) {
      left_packet = Eigen::internal::pset1<TinPacket>(*left);
    }
  }

  EIGEN_DEVICE_FUNC inline Tout operator()(const Tin& right) const {
    return Binary::operator()(*left, right);
  }

  template <typename Packet,
            typename std::enable_if<!is_scalar_in_host_memory ||
                                        !std::is_same<TinPacket, Packet>::value,
                                    int>::type = 0>
  EIGEN_DEVICE_FUNC inline Packet packetOp(const Packet& right_packet) const {
    const Packet left_packet = Eigen::internal::pset1<Packet>(*left);
    return Binary::packetOp(left_packet, right_packet);
  }

  template <typename Packet,
            typename std::enable_if<is_scalar_in_host_memory &&
                                        std::is_same<TinPacket, Packet>::value,
                                    int>::type = 0>
  EIGEN_DEVICE_FUNC inline Packet packetOp(const Packet& right_packet) const {
    return Binary::packetOp(left_packet, right_packet);
  }
};

template <typename Tout, typename Tin, typename Binary,
          bool is_scalar_in_host_memory>
struct functor_traits<
    scalar_left<Tout, Tin, Binary, is_scalar_in_host_memory>> {
  enum {
    Cost = functor_traits<Binary>::Cost,
#if TENSORFLOW_USE_ROCM
    PacketAccess = false,
#else
    PacketAccess = functor_traits<Binary>::PacketAccess,
#endif  // TENSORFLOW_USE_ROCM
  };
};

template <typename Tout, typename Tin, typename Binary,
          bool is_scalar_in_host_memory = false>
struct scalar_right : private Binary {
  using result_type = Tout;
  using TinPacket = typename Eigen::internal::packet_traits<Tin>::type;

  const Tin* right;
  TinPacket right_packet;  // initialized iff is_scalar_in_host_memory == true

  EIGEN_DEVICE_FUNC inline scalar_right(const scalar_right& other) = default;

  template <typename... Args>
  EIGEN_DEVICE_FUNC inline explicit scalar_right(const Tin* c, Args... args)
      : Binary(args...), right(c) {
    if (is_scalar_in_host_memory) {
      right_packet = Eigen::internal::pset1<TinPacket>(*right);
    }
  }

  EIGEN_DEVICE_FUNC inline Tout operator()(const Tin& left) const {
    return Binary::operator()(left, *right);
  }

  template <typename Packet,
            typename std::enable_if<!is_scalar_in_host_memory ||
                                        !std::is_same<TinPacket, Packet>::value,
                                    int>::type = 0>
  EIGEN_DEVICE_FUNC inline Packet packetOp(const Packet& left_packet) const {
    const Packet right_packet = Eigen::internal::pset1<Packet>(*right);
    return Binary::packetOp(left_packet, right_packet);
  }

  template <typename Packet,
            typename std::enable_if<is_scalar_in_host_memory &&
                                        std::is_same<TinPacket, Packet>::value,
                                    int>::type = 0>
  EIGEN_DEVICE_FUNC inline Packet packetOp(const Packet& left_packet) const {
    return Binary::packetOp(left_packet, right_packet);
  }
};

template <typename Tout, typename Tin, typename Binary,
          bool is_scalar_in_host_memory>
struct functor_traits<
    scalar_right<Tout, Tin, Binary, is_scalar_in_host_memory>> {
  enum {
    Cost = functor_traits<Binary>::Cost,
#if TENSORFLOW_USE_ROCM
    PacketAccess = false,
#else
    PacketAccess = functor_traits<Binary>::PacketAccess,
#endif  // TENSORFLOW_USE_ROCM
  };
};

// similar to std::equal_to, but with the DEVICE_FUNC qualifier
template <class T>
struct equal_to : std::binary_function<T, T, bool> {
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE bool operator()(const T& x,
                                                        const T& y) const {
    return x == y;
  }
};

// similar to std::not_equal_to, but with the DEVICE_FUNC qualifier
template <class T>
struct not_equal_to : std::binary_function<T, T, bool> {
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE bool operator()(const T& x,
                                                        const T& y) const {
    return x != y;
  }
};

// similar to std::greater, but with the DEVICE_FUNC qualifier
template <class T>
struct greater : std::binary_function<T, T, bool> {
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE bool operator()(const T& x,
                                                        const T& y) const {
    return x > y;
  }
};

// similar to std::less, but with the DEVICE_FUNC qualifier
template <class T>
struct less : std::binary_function<T, T, bool> {
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE bool operator()(const T& x,
                                                        const T& y) const {
    return x < y;
  }
};

// similar to std::greater_equal, but with the DEVICE_FUNC qualifier
template <class T>
struct greater_equal : std::binary_function<T, T, bool> {
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE bool operator()(const T& x,
                                                        const T& y) const {
    return x >= y;
  }
};

// similar to std::less_equal, but with the DEVICE_FUNC qualifier
template <class T>
struct less_equal : std::binary_function<T, T, bool> {
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE bool operator()(const T& x,
                                                        const T& y) const {
    return x <= y;
  }
};

// Functor that enables squared difference functor.
template <typename Scalar>
struct scalar_squared_difference_op {
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Scalar
  operator()(const Scalar& a, const Scalar& b) const {
    const Scalar v = scalar_difference_op<Scalar>()(a, b);
    return scalar_product_op<Scalar>()(v, scalar_conjugate_op<Scalar>()(v));
  }
  template <typename Packet>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Packet packetOp(const Packet& a,
                                                        const Packet& b) const {
    const Packet v = scalar_difference_op<Scalar>().packetOp(a, b);
    return scalar_product_op<Scalar>().packetOp(
        v, scalar_conjugate_op<Scalar>().packetOp(v));
  }
};

template <typename Scalar>
struct functor_traits<scalar_squared_difference_op<Scalar>> {
  enum {
    Cost = functor_traits<scalar_difference_op<Scalar>>::Cost +
           functor_traits<scalar_conjugate_op<Scalar>>::Cost +
           functor_traits<scalar_product_op<Scalar>>::Cost,
    PacketAccess = functor_traits<scalar_difference_op<Scalar>>::PacketAccess &&
                   functor_traits<scalar_conjugate_op<Scalar>>::PacketAccess &&
                   functor_traits<scalar_product_op<Scalar>>::PacketAccess
  };
};

// TODO(b/32239616): This kernel should be moved into Eigen and vectorized.
template <typename T, typename Enable = void>
struct google_floor_div {
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE T operator()(const T& x,
                                                     const T& y) const {
    if ((x < T(0)) != (y < T(0))) {
// HIP does not have the device version of the abs routine defined
// for all datatypes that T can resolve to
#if defined(__HIP_DEVICE_COMPILE__)
      T abs_x = (x < T(0)) ? -x : x;
      T abs_y = (y < T(0)) ? -y : y;
#else
      T abs_x = std::abs(x);
      T abs_y = std::abs(y);
#endif
      return -(abs_x + abs_y - 1) / abs_y;
    } else {
      return x / y;
    }
  }
  template <typename Packet>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Packet packetOp(const Packet& x,
                                                        const Packet& y) const {
    Packet zeros = pzero(x);
    Packet x_mask = pcmp_lt(x, zeros);
    Packet y_mask = pcmp_lt(y, zeros);
    Packet x_div_y = pdiv(x, y);
    Packet abs_x = pabs(x);
    Packet abs_y = pabs(y);
    Packet ones = pones(x);
    Packet ratio_rounded = pdiv(pnegate(psub(padd(abs_x, abs_y), ones)), abs_y);
    return pselect(pxor(x_mask, y_mask), ratio_rounded, x_div_y);
  }
};

template <typename T>
struct google_floor_div<
    T, typename std::enable_if<std::is_unsigned<T>::value>::type> {
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE T operator()(const T& x,
                                                     const T& y) const {
    return x / y;
  }
  template <typename Packet>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Packet packetOp(const Packet& x,
                                                        const Packet& y) const {
    return pdiv(x, y);
  }
};

template <typename Scalar>
struct functor_traits<google_floor_div<Scalar>> {
  enum {
    Cost = 2 * Eigen::internal::scalar_div_cost<
                   Scalar, packet_traits<Scalar>::HasDiv>::value +
           NumTraits<Scalar>::AddCost,
#if TENSORFLOW_USE_ROCM
    PacketAccess = false,
#else
    PacketAccess = packet_traits<Scalar>::HasDiv
#endif
  };
};

template <typename T, typename Enable = void>
struct google_floor_div_real {
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE T operator()(const T& x,
                                                     const T& y) const {
    return Eigen::numext::floor(x / y);
  }
  template <typename Packet>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Packet packetOp(const Packet& x,
                                                        const Packet& y) const {
    return pfloor(pdiv(x, y));
  }
};

template <typename Scalar>
struct functor_traits<google_floor_div_real<Scalar>> {
  enum {
    Cost = 2 * Eigen::internal::scalar_div_cost<
                   Scalar, packet_traits<Scalar>::HasDiv>::value +
           2 * NumTraits<Scalar>::AddCost,
#if TENSORFLOW_USE_ROCM
    PacketAccess = false,
#else
    PacketAccess =
        packet_traits<Scalar>::HasDiv && packet_traits<Scalar>::HasFloor
#endif
  };
};

// TODO(rmlarsen): Add vectorized mod & fmod in Eigen and use it here.
template <typename T>
struct google_floor_fmod {
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE T operator()(const T& x,
                                                     const T& y) const {
    // EIGEN_STATIC_ASSERT(NUMERIC_TYPE_MUST_BE_REAL);
    T trunc_mod = scalar_fmod_op<T>()(x, y);
    return trunc_mod != T(0) && (y < T(0) != trunc_mod < T(0)) ? trunc_mod + y
                                                               : trunc_mod;
  }
};

template <typename Scalar>
struct functor_traits<google_floor_fmod<Scalar>> {
  enum {
    Cost = functor_traits<Eigen::internal::scalar_fmod_op<Scalar>>::Cost +
           NumTraits<Scalar>::AddCost,
    PacketAccess = false
  };
};

// TODO(rmlarsen): Add vectorized mod & fmod in Eigen and use it here.
template <typename T>
struct google_floor_mod {
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE T operator()(const T& x,
                                                     const T& y) const {
    // EIGEN_STATIC_ASSERT(!NUMERIC_TYPE_MUST_BE_REAL);
    T trunc_mod = Eigen::internal::scalar_mod2_op<T>()(x, y);
    return trunc_mod != T(0) && (y < T(0) != trunc_mod < T(0)) ? trunc_mod + y
                                                               : trunc_mod;
  }
};

template <typename Scalar>
struct functor_traits<google_floor_mod<Scalar>> {
  enum {
    Cost = functor_traits<Eigen::internal::scalar_mod2_op<Scalar>>::Cost +
           NumTraits<Scalar>::AddCost,
    PacketAccess = false
  };
};

#if EIGEN_COMP_GNUC && __cplusplus > 199711L
#define DISABLE_FLOAT_EQUALITY_WARNING \
  _Pragma("GCC diagnostic push")       \
      _Pragma("GCC diagnostic ignored \"-Wfloat-equal\"")
#define ENABLE_FLOAT_EQUALITY_WARNING _Pragma("GCC diagnostic pop")
#else
#define DISABLE_FLOAT_EQUALITY_WARNING
#define ENABLE_FLOAT_EQUALITY_WARNING
#endif

template <typename Scalar, bool IsInteger = Eigen::NumTraits<Scalar>::IsInteger>
struct scalar_round_half_to_even_op {
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Scalar
  operator()(const Scalar& x) const {
    EIGEN_STATIC_ASSERT((!NumTraits<Scalar>::IsComplex),
                        NUMERIC_TYPE_MUST_BE_REAL)

    const Scalar round_val = Eigen::numext::floor(x + Scalar(0.5));
    const Scalar fraction = round_val - x;
    if (TF_PREDICT_FALSE(fraction == Scalar(.5))) {
      return Scalar(2) * Eigen::numext::floor(Scalar(.5) * x + Scalar(0.5));
    } else {
      return round_val;
    }
  }

  template <typename Packet>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Packet packetOp(const Packet& x) const {
    Packet half = pset1<Packet>(Scalar(0.5));
    Packet round_val = pfloor(padd(x, half));
    Packet fraction = psub(round_val, x);
    Packet half_mask = pcmp_eq(fraction, half);
    bool any_halves = predux_any(half_mask);
    if (TF_PREDICT_FALSE(any_halves)) {
      Packet two = pset1<Packet>(Scalar(2));
      Packet nearest_even = pmul(two, pfloor(pmadd(half, x, half)));
      return pselect(half_mask, nearest_even, round_val);
    } else {
      return round_val;
    }
  }
};

template <typename Scalar>
struct scalar_round_half_to_even_op<Scalar, true> {
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Scalar
  operator()(const Scalar& x) const {
    return x;
  }
  template <typename Packet>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Packet packetOp(const Packet& x) const {
    return x;
  }
};

template <typename Scalar>
struct functor_traits<scalar_round_half_to_even_op<Scalar>> {
  enum {
    Cost = Eigen::NumTraits<Scalar>::IsInteger ? 0
                                               : 4 * NumTraits<Scalar>::AddCost,
#if TENSORFLOW_USE_ROCM
    PacketAccess = false,
#else
    PacketAccess = packet_traits<Scalar>::HasFloor &&
                   packet_traits<Scalar>::HasAdd &&
                   packet_traits<Scalar>::HasMul,
#endif
  };
};

template <typename Scalar, bool IsInteger = Eigen::NumTraits<Scalar>::IsInteger>
struct scalar_round_up_op {
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Scalar
  operator()(const Scalar& x) const {
    EIGEN_STATIC_ASSERT((!NumTraits<Scalar>::IsComplex),
                        NUMERIC_TYPE_MUST_BE_REAL)
    return Eigen::numext::floor(x + Scalar(0.5));
  }

  template <typename Packet>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Packet packetOp(const Packet& x) const {
    return pfloor(padd(x, pset1<Packet>(0.5)));
  }
};

template <typename Scalar>
struct scalar_round_up_op<Scalar, true> {
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Scalar
  operator()(const Scalar& x) const {
    return x;
  }

  template <typename Packet>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Packet packetOp(const Packet& x) const {
    return x;
  }
};

template <typename Scalar, bool IsInteger>
struct functor_traits<scalar_round_up_op<Scalar, IsInteger>> {
  enum {
    Cost = IsInteger ? 0 : 4 * NumTraits<Scalar>::AddCost,
#if TENSORFLOW_USE_ROCM
    PacketAccess = false,
#else
    PacketAccess = IsInteger || packet_traits<Scalar>::HasFloor
#endif
  };
};

#undef ENABLE_FLOAT_EQUALITY_WARNING
#undef DISABLE_FLOAT_EQUALITY_WARNING

template <typename Scalar>
struct bitwise_xor_op {
  EIGEN_EMPTY_STRUCT_CTOR(bitwise_xor_op)
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Scalar
  operator()(const Scalar& x, const Scalar& y) const {
    return x ^ y;
  }
  template <typename Packet>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Packet packetOp(const Packet& a,
                                                        const Packet& b) const {
    return Eigen::internal::pxor(a, b);
  }
};

template <typename Scalar>
struct functor_traits<bitwise_xor_op<Scalar>> {
  enum { Cost = Eigen::NumTraits<Scalar>::AddCost, PacketAccess = true };
};

template <typename Scalar>
struct xlogy_op {
  EIGEN_EMPTY_STRUCT_CTOR(xlogy_op)
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Scalar
  operator()(const Scalar& x, const Scalar& y) const {
    if (x == Scalar(0.)) {
      return Scalar(0.);
    }
    return x * numext::log(y);
  }
  template <typename Packet>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Packet packetOp(const Packet& x,
                                                        const Packet& y) const {
    Packet zeros = pzero(x);
    Packet mask = pcmp_eq(x, zeros);
    scalar_log_op<Scalar> log_op;
    Packet log_y = log_op.packetOp(y);
    Packet x_log_y = pmul(x, log_y);
    return pselect(mask, x, x_log_y);
  }
};

template <typename Scalar>
struct functor_traits<xlogy_op<Scalar>> {
  enum {
    Cost = functor_traits<scalar_log_op<Scalar>>::Cost +
           Eigen::NumTraits<Scalar>::MulCost,
#if TENSORFLOW_USE_ROCM
    PacketAccess = false,
#else
    PacketAccess = functor_traits<scalar_log_op<Scalar>>::PacketAccess
#endif
  };
};

template <typename Scalar>
struct xdivy_op {
  EIGEN_EMPTY_STRUCT_CTOR(xdivy_op)
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Scalar
  operator()(const Scalar& x, const Scalar& y) const {
    if (x == Scalar(0.)) {
      return Scalar(0.);
    }
    return x / y;
  }
  template <typename Packet>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Packet packetOp(const Packet& x,
                                                        const Packet& y) const {
    Packet zeros = pzero(x);
    Packet mask = pcmp_eq(x, zeros);
    Packet x_div_y = pdiv(x, y);
    return pselect(mask, x, x_div_y);
  }
};

template <typename Scalar>
struct functor_traits<xdivy_op<Scalar>> {
  enum {
    Cost =
        Eigen::NumTraits<Scalar>::AddCost +
        Eigen::internal::scalar_div_cost<Scalar,
                                         packet_traits<Scalar>::HasDiv>::value,
#if TENSORFLOW_USE_ROCM
    PacketAccess = false,
#else
    PacketAccess = packet_traits<Scalar>::HasDiv
#endif
  };
};

template <typename T>
struct scalar_erfinv_op {
  EIGEN_EMPTY_STRUCT_CTOR(scalar_erfinv_op)
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE T operator()(const T& x) const {
    T y = numext::ndtri((x + static_cast<T>(1.)) / static_cast<T>(2.));
    return y / static_cast<T>(numext::sqrt(2.));
  }
  template <typename Packet>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Packet packetOp(const Packet& x) const {
    Packet y = pndtri<Packet>(pmadd(pset1<Packet>(0.5), x, pset1<Packet>(0.5)));
    return pdiv(y, psqrt(pset1<Packet>(2.)));
  }
};

template <typename T>
struct functor_traits<scalar_erfinv_op<T>> {
  enum {
    Cost = functor_traits<scalar_ndtri_op<T>>::Cost + NumTraits<T>::AddCost,
#if TENSORFLOW_USE_ROCM
    PacketAccess = false,
#else
    PacketAccess = packet_traits<T>::HasNdtri,
#endif
  };
};

}  // end namespace internal
}  // end namespace Eigen

namespace tensorflow {
namespace functor {

////////////////////////////////////////////////////////////////////////////////
// Helpers
////////////////////////////////////////////////////////////////////////////////

// Base template for functors whose input scalar type is T and
// output scalar type is R.
template <typename T, typename F, typename R = T>
struct base {
  // func defines operator() and its vectorized version packetOp().
  typedef F func;

  // If true, the functor's corresponding binary op will instantiate
  // specialized kernels to perform an optimized broadcast
  // operation. Each functor for which this is enabled increases the
  // code size, so by default this is disabled for binary functors and
  // is enabled on a per-op basis as needed.
  static const bool use_bcast_optimization = false;

  // operator() has the signature:
  //  out_type operator()(in_type in0, in_type in1 ...)
  typedef R out_type;
  typedef T in_type;

  // TensorFlow provides tensor-ized version of "func". Roughly
  // speaking, the tensorflow operation has the signature:
  //   tout_type op(tin_type in0)
  //   tout_type op(tin_type in0, tin_type in1)
  //   tout_type op(tin_type in0, in_type scalar)
  typedef typename TTypes<out_type>::Flat tout_type;
  typedef typename TTypes<in_type>::ConstFlat tin_type;
  typedef typename TTypes<in_type>::ConstScalar tscalar_type;

  // Whether the functor can error out.  Currently applies only to integer
  // div and mod.
  static const bool has_errors = false;
};

// For now, we only apply certain speed optimization for
// float/double's broadcast binary op.
template <typename T>
struct use_bcast_optimization {
  static const bool value = false;
};

template <>
struct use_bcast_optimization<float> {
  static const bool value = true;
};

template <>
struct use_bcast_optimization<double> {
  static const bool value = true;
};

////////////////////////////////////////////////////////////////////////////////
// Unary functors
////////////////////////////////////////////////////////////////////////////////

// abs(x) = |x|
// neg(x) = - x
// inverse(x) = 1 / x
// square(x) = x^2
// sqrt(x) = x^(1/2)
// rsqrt(x) = x^(-1/2)
// exp(x) = e^x
// expm1(x) = e^x - 1
// log(x) = natural logarithm of x
// log1p(x) = natural logarithm of 1 + x
// tanh = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
// sigmoid = 1 / (1 + exp(-x))  // a.k.a, logistic
//
// NOTE: We may eventually implement common functions used in NN
// here. E.g., rectifier, softplus, derivatives of tanh, sigmod, etc.
// For reference, see speech/lstm/eigen_functors.h.

template <typename T>
struct abs : base<T, Eigen::internal::scalar_abs_op<T>,
                  typename Eigen::internal::scalar_abs_op<T>::result_type> {};

template <typename T>
struct neg : base<T, Eigen::internal::scalar_opposite_op<T>> {};

template <typename T>
struct inverse : base<T, Eigen::internal::scalar_inverse_op<T>> {};

template <typename T>
struct square : base<T, Eigen::internal::scalar_square_op<T>> {};

template <typename T>
struct sqrt : base<T, Eigen::internal::scalar_sqrt_op<T>> {};

template <typename T>
struct rsqrt : base<T, Eigen::internal::scalar_rsqrt_op<T>> {};

template <typename T>
struct exp : base<T, Eigen::internal::scalar_exp_op<T>> {};

template <typename T>
struct expm1 : base<T, Eigen::internal::scalar_expm1_op<T>> {};

template <typename T>
struct log : base<T, Eigen::internal::scalar_log_op<T>> {};

template <typename T>
struct log1p : base<T, Eigen::internal::scalar_log1p_op<T>> {};

template <typename T>
struct sign : base<T, Eigen::internal::scalar_sign_op<T>> {};

template <typename T>
struct sinh : base<T, Eigen::internal::scalar_sinh_op<T>> {};

template <typename T>
struct cosh : base<T, Eigen::internal::scalar_cosh_op<T>> {};

template <typename T>
struct tanh : base<T, Eigen::internal::scalar_tanh_op<T>> {};

template <typename T>
struct asinh : base<T, Eigen::internal::scalar_asinh_op<T>> {};

template <typename T>
struct acosh : base<T, Eigen::internal::scalar_acosh_op<T>> {};

template <typename T>
struct atanh : base<T, Eigen::internal::scalar_atanh_op<T>> {};

template <typename T>
struct lgamma : base<T, Eigen::internal::scalar_lgamma_op<T>> {};

template <typename T>
struct digamma : base<T, Eigen::internal::scalar_digamma_op<T>> {};

template <typename T>
struct erf : base<T, Eigen::internal::scalar_erf_op<T>> {};

template <typename T>
struct erfc : base<T, Eigen::internal::scalar_erfc_op<T>> {};

template <typename T>
struct ndtri : base<T, Eigen::internal::scalar_ndtri_op<T>> {};

template <typename T>
struct erfinv : base<T, Eigen::internal::scalar_erfinv_op<T>> {};

template <typename T>
struct sigmoid : base<T, Eigen::internal::scalar_logistic_op<T>> {};

template <typename T>
struct sin : base<T, Eigen::internal::scalar_sin_op<T>> {};

template <typename T>
struct cos : base<T, Eigen::internal::scalar_cos_op<T>> {};

template <typename T>
struct tan : base<T, Eigen::internal::scalar_tan_op<T>> {};

template <typename T>
struct asin : base<T, Eigen::internal::scalar_asin_op<T>> {};

template <typename T>
struct acos : base<T, Eigen::internal::scalar_acos_op<T>> {};

template <typename T>
struct atan : base<T, Eigen::internal::scalar_atan_op<T>> {};

template <typename T>
struct bessel_i0e : base<T, Eigen::internal::scalar_bessel_i0e_op<T>> {};

template <typename T>
struct bessel_i1e : base<T, Eigen::internal::scalar_bessel_i1e_op<T>> {};

struct logical_not : base<bool, Eigen::internal::scalar_boolean_not_op<bool>> {
};

// Flip all bits. Named invert to be consistent with numpy.
template <typename T>
struct invert_op {
  EIGEN_EMPTY_STRUCT_CTOR(invert_op)
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE T operator()(const T& a) const {
    return ~a;
  }
};

template <typename T>
struct invert : base<T, invert_op<T>> {};

// NOTE: std::isinf, std::isnan, std::isfinite are plain function.
// Therefore we need to wrap them in functors to be used with Eigen's
// type system.
template <typename T>
struct isinf : base<T, Eigen::internal::scalar_isinf_op<T>, bool> {};

template <typename T>
struct isnan : base<T, Eigen::internal::scalar_isnan_op<T>, bool> {};

template <typename T>
struct isfinite : base<T, Eigen::internal::scalar_isfinite_op<T>, bool> {};

template <typename T>
struct floor : base<T, Eigen::internal::scalar_floor_op<T>> {};

template <typename T>
struct round : base<T, Eigen::internal::scalar_round_half_to_even_op<T>> {};

template <typename T>
struct ceil : base<T, Eigen::internal::scalar_ceil_op<T>> {};

/** TODO(tokarip): This should go in Eigen
 * \brief Template functor to compute the round to int value of a scalar
 */
template <typename Scalar>
struct scalar_rint_op {
  EIGEN_EMPTY_STRUCT_CTOR(scalar_rint_op)
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Scalar
  operator()(const Scalar& a) const {
#if defined(__CUDACC__) || defined(__HIPCC__)
    return ::rint(a);
#elif defined(__ANDROID__)
    return rint(a);
#else
    return std::rint(a);
#endif
  }
};

template <typename T>
struct rint : base<T, scalar_rint_op<T>> {};

////////////////////////////////////////////////////////////////////////////////
// Binary functors
////////////////////////////////////////////////////////////////////////////////

// Binary functors:
//
// add(x, y) = x + y
// sub(x, y) = x - y
// mul(x, y) = x * y
// div(x, y) = x / y
// mod(x, y) = x % y         (int32 and int64 only)
// fmod(x, y) = fmod(x, y)   (float and double only)
// pow(x, y) = x ^ y
// maximum(x, y) = x > y ? x : y
// minimum(x, y) = x < y ? x : y
// squared_difference(x, y) = conj(x - y) * (x - y)

template <typename T>
struct add : base<T, Eigen::internal::scalar_sum_op<T>> {
  static const bool use_bcast_optimization = true;
};

template <typename T>
struct sub : base<T, Eigen::internal::scalar_difference_op<T>> {
  static const bool use_bcast_optimization = true;
};

template <typename T>
struct mul : base<T, Eigen::internal::scalar_product_op<T>> {
  static const bool use_bcast_optimization = true;
};

template <typename T>
struct mul_no_nan : base<T, Eigen::internal::mul_no_nan_op<T>> {};

template <typename T>
struct div : base<T, Eigen::internal::scalar_quotient_op<T>> {};

template <typename T>
struct safe_div : base<T, Eigen::internal::safe_div_or_mod_op<
                              T, Eigen::internal::scalar_quotient_op<T>>> {
  static const bool has_errors = true;
};

template <typename T>
struct div_no_nan : base<T, Eigen::internal::div_no_nan_op<T>> {};

template <typename T>
struct fmod : base<T, Eigen::internal::scalar_fmod_op<T>> {};

template <typename T>
struct mod : base<T, Eigen::internal::scalar_mod2_op<T>> {};

template <typename T>
struct safe_mod : base<T, Eigen::internal::safe_div_or_mod_op<
                              T, Eigen::internal::scalar_mod2_op<T>>> {
  static const bool has_errors = true;
};

template <typename T>
struct floor_fmod : base<T, Eigen::internal::google_floor_fmod<T>> {};

template <typename T>
struct safe_floor_mod : base<T, Eigen::internal::safe_div_or_mod_op<
                                    T, Eigen::internal::google_floor_mod<T>>> {
  static const bool has_errors = true;
};

template <typename T>
struct floor_div : base<T, Eigen::internal::google_floor_div<T>> {};

template <typename T>
struct safe_floor_div : base<T, Eigen::internal::safe_div_or_mod_op<
                                    T, Eigen::internal::google_floor_div<T>>> {
  static const bool has_errors = true;
};

template <typename T>
struct floor_div_real : base<T, Eigen::internal::google_floor_div_real<T>> {};

template <typename T>
struct pow : base<T, Eigen::internal::scalar_pow_op<T, T>> {};

template <typename T>
struct safe_pow : base<T, Eigen::internal::safe_scalar_binary_pow_op<T, T>> {
  static const bool has_errors = true;
};

template <typename T>
struct maximum : base<T, Eigen::internal::scalar_max_op<T>> {};

template <typename T>
struct minimum : base<T, Eigen::internal::scalar_min_op<T>> {};

template <typename T>
struct igamma : base<T, Eigen::internal::scalar_igamma_op<T>> {};

template <typename T>
struct random_gamma_grad
    : base<T, Eigen::internal::scalar_gamma_sample_der_alpha_op<T>> {};

template <typename T>
struct igammac : base<T, Eigen::internal::scalar_igammac_op<T>> {};

template <typename T>
struct zeta : base<T, Eigen::internal::scalar_zeta_op<T>> {};

template <typename T>
struct polygamma : base<T, Eigen::internal::scalar_polygamma_op<T>> {};

template <typename Scalar>
struct scalar_atan2_op {
  EIGEN_EMPTY_STRUCT_CTOR(scalar_atan2_op)
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Scalar
  operator()(const Scalar& y, const Scalar& x) const {
#if GOOGLE_CUDA
    return std::atan2(y, x);
#elif TENSORFLOW_USE_ROCM
    return ::atan2(y, x);
#else
    return std::atan2(y, x);
#endif
  }
};

template <typename T>
struct atan2 : base<T, scalar_atan2_op<T>> {};

template <typename T>
struct squared_difference
    : base<T, Eigen::internal::scalar_squared_difference_op<T>> {};

template <typename T>
struct xdivy : base<T, Eigen::internal::xdivy_op<T>> {};

template <typename T>
struct xlogy : base<T, Eigen::internal::xlogy_op<T>> {};

template <typename T>
struct less : base<T, Eigen::internal::less<T>, bool> {};

template <typename T>
struct less_equal : base<T, Eigen::internal::less_equal<T>, bool> {};

template <typename T>
struct greater : base<T, Eigen::internal::greater<T>, bool> {};

template <typename T>
struct greater_equal : base<T, Eigen::internal::greater_equal<T>, bool> {};

template <typename T>
struct equal_to : base<T, Eigen::internal::equal_to<T>, bool> {};

template <typename T>
struct not_equal_to : base<T, Eigen::internal::not_equal_to<T>, bool> {};

struct logical_and : base<bool, Eigen::internal::scalar_boolean_and_op> {};

struct logical_or : base<bool, Eigen::internal::scalar_boolean_or_op> {};

template <typename T>
struct bitwise_and_op {
  EIGEN_EMPTY_STRUCT_CTOR(bitwise_and_op)
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE T operator()(const T& x,
                                                     const T& y) const {
    return x & y;
  }
};

template <typename T>
struct bitwise_or_op {
  EIGEN_EMPTY_STRUCT_CTOR(bitwise_or_op)
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE T operator()(const T& x,
                                                     const T& y) const {
    return x | y;
  }
};

template <typename T>
struct bitwise_and : base<T, bitwise_and_op<T>> {};

template <typename T>
struct bitwise_or : base<T, bitwise_or_op<T>> {};

template <typename T>
struct bitwise_xor : base<T, Eigen::internal::bitwise_xor_op<T>> {};

template <typename T>
struct left_shift_op {
  EIGEN_EMPTY_STRUCT_CTOR(left_shift_op)
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE T operator()(const T& x,
                                                     const T& y) const {
    // Avoids UB: don't shift by larger than the bitwidth of T, and
    // performs left shifts as unsigned shifts.
    T y_clamped = y;
    if (y_clamped < 0) {
      y_clamped = 0;
    } else if (y_clamped > sizeof(T) * CHAR_BIT - 1) {
      y_clamped = sizeof(T) * CHAR_BIT - 1;
    }
    using U = typename std::make_unsigned<T>::type;
    return static_cast<T>(static_cast<U>(x) << static_cast<U>(y_clamped));
  }
};

template <typename T>
struct right_shift_op {
  EIGEN_EMPTY_STRUCT_CTOR(right_shift_op)
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE T operator()(const T& x,
                                                     const T& y) const {
    // Avoids UB: don't shift by larger than the bitwidth of T.
    T y_clamped = y;
    if (y_clamped < 0) {
      y_clamped = 0;
    } else if (y_clamped > sizeof(T) * CHAR_BIT - 1) {
      y_clamped = sizeof(T) * CHAR_BIT - 1;
    }
    // Technically right shifts of signed integers are not necessarily
    // arithmetic shifts according to the C++ standard. However in practice most
    // implementations are arithmetic shifts. If this proves to be a problem in
    // practice, we may need to use an alternative implementation.
    return x >> y_clamped;
  }
};

template <typename T>
struct left_shift : base<T, left_shift_op<T>> {};

template <typename T>
struct right_shift : base<T, right_shift_op<T>> {};

template <typename T>
struct make_complex_func {
  typedef std::complex<T> result_type;
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE result_type operator()(T real,
                                                               T imag) const {
    return std::complex<T>(real, imag);
  }
};

template <typename T>
struct make_complex : base<T, make_complex_func<T>, std::complex<T>> {};

template <typename T>
struct get_real
    : base<T, Eigen::internal::scalar_real_op<T>, typename T::value_type> {};

template <typename T>
struct get_imag
    : base<T, Eigen::internal::scalar_imag_op<T>, typename T::value_type> {};

template <typename T>
struct get_angle
    : base<T, Eigen::internal::scalar_arg_op<T>, typename T::value_type> {};

template <typename T>
struct conj : base<T, Eigen::internal::scalar_conjugate_op<T>> {};

////////////////////////////////////////////////////////////////////////////////
// Functors takes 1 or 2 tensors, computes the base functor on
// coefficient of the input tensors and puts the results in the output
// tensor.
////////////////////////////////////////////////////////////////////////////////
template <typename Device, typename Functor>
struct UnaryFunctor {
  // Computes on device "d": out[i] = Functor(in[i])
  void operator()(const Device& d, typename Functor::tout_type out,
                  typename Functor::tin_type in);
};

template <typename Device, typename Functor, int NDIMS,
          bool has_errors = Functor::has_errors>
struct BinaryFunctor {
  // Computes on device "d": out[i] = Functor(in0[i], in1[i])
  void operator()(const Device& d, typename Functor::tout_type out,
                  typename Functor::tin_type in0,
                  typename Functor::tin_type in1, bool* error);

  // Computes on device "d": out[i] = Functor(scalar[0], in[i])
  void Left(const Device& d, typename Functor::tout_type out,
            typename Functor::tscalar_type scalar,
            typename Functor::tin_type in, bool* error);

  // Computes on device "d": out[i] = Functor(in[i], scalar[0])
  void Right(const Device& d, typename Functor::tout_type out,
             typename Functor::tin_type in,
             typename Functor::tscalar_type scalar, bool* error);

  // Computes on device "d":
  //   out = Functor(in0.broadcast(bcast0), in1.broadcast(bcast1))
  //
  // TODO(zhifengc): makes BCast a template member function on NDIMS
  // instead making BinaryFunctor templates on NDIMS.
  void BCast(const Device& d,
             typename TTypes<typename Functor::out_type, NDIMS>::Tensor out,
             typename TTypes<typename Functor::in_type, NDIMS>::ConstTensor in0,
             typename Eigen::array<Eigen::DenseIndex, NDIMS> bcast0,
             typename TTypes<typename Functor::in_type, NDIMS>::ConstTensor in1,
             typename Eigen::array<Eigen::DenseIndex, NDIMS> bcast1,
             bool* error);
};

template <typename Device, typename T>
struct ApproximateEqual {
  void operator()(const Device& d, typename TTypes<T>::ConstFlat x,
                  typename TTypes<T>::ConstFlat y, T tolerance,
                  typename TTypes<bool>::Flat z);
};

template <int NDIMS>
bool AllOne(const typename Eigen::array<Eigen::DenseIndex, NDIMS>& a) {
  for (size_t i = 0; i < a.size(); ++i) {
    if (a[i] != 1) return false;
  }
  return true;
}

template <typename Device, typename T>
struct SelectFunctor {
  void operator()(const Device& d, typename TTypes<T>::Flat out,
                  typename TTypes<bool>::ConstFlat cond_flat,
                  typename TTypes<T>::ConstFlat then_flat,
                  typename TTypes<T>::ConstFlat else_flat);
};

template <typename Device, typename T>
struct SelectScalarFunctor {
  void operator()(const Device& d, typename TTypes<T>::Flat out,
                  typename TTypes<bool>::ConstScalar cond,
                  typename TTypes<T>::ConstFlat then_flat,
                  typename TTypes<T>::ConstFlat else_flat);
};

template <typename Device, typename T>
struct BatchSelectFunctor {
  void operator()(const Device& d,
                  typename TTypes<T>::Matrix output_flat_outer_dims,
                  TTypes<bool>::ConstVec cond_vec,
                  typename TTypes<T>::ConstMatrix then_flat_outer_dims,
                  typename TTypes<T>::ConstMatrix else_flat_outer_dims);
};

template <typename Device, typename T, int NDIMS>
struct BCastSelectFunctor {
  void operator()(const Device& d,
                  typename TTypes<T, NDIMS>::Tensor output_tensor,
                  typename TTypes<bool, NDIMS>::ConstTensor cond_tensor,
                  typename TTypes<T, NDIMS>::ConstTensor then_tensor,
                  typename TTypes<T, NDIMS>::ConstTensor else_tensor,
                  typename Eigen::array<Eigen::DenseIndex, NDIMS> cond_bcast,
                  typename Eigen::array<Eigen::DenseIndex, NDIMS> then_bcast,
                  typename Eigen::array<Eigen::DenseIndex, NDIMS> else_bcast);
};

}  // end namespace functor
}  // end namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_CWISE_OPS_H_

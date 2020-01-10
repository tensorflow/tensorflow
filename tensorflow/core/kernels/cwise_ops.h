#ifndef TENSORFLOW_KERNELS_CWISE_OPS_H_
#define TENSORFLOW_KERNELS_CWISE_OPS_H_

#include <cmath>
#include <functional>
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/numeric_types.h"
#include "tensorflow/core/framework/tensor_types.h"

// The following functors (sign, tanh, sigmoid, etc.) are not defined
// by Eigen.  When their equivalent are added into the Eigen, we can
// replace them using type aliases.

namespace Eigen {
namespace internal {

template <typename T>
struct scalar_sign_op {
  // TODO(zhifengc): this only works for real types. In theory,
  // sign(x) = x / |x| works for both real and complex values.
  EIGEN_EMPTY_STRUCT_CTOR(scalar_sign_op);
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE T operator()(const T& x) const {
    return T(x > T(0)) - T(x < T(0));
  }
};

// TODO(zhifengc): Eigen::internal::pow_impl does not have proper
// EIGEN host/device decoration. We duplicate code here for now.
template <typename T, bool IsInteger>
struct pow {
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE T operator()(const T& x,
                                                     const T& y) const {
    return std::pow(x, y);
  }
};

template <typename T>
struct pow<T, true> {
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE T operator()(T x, T y) const {
    T res(1);
    if (y & 1) res *= x;
    y >>= 1;
    while (y) {
      x *= x;
      if (y & 1) res *= x;
      y >>= 1;
    }
    return res;
  }
};

template <typename T>
struct scalar_pow2_op : pow<T, NumTraits<T>::IsInteger> {};

template <typename T>
struct functor_traits<scalar_pow2_op<T> > {
  enum {
    Cost = 5 * NumTraits<T>::MulCost,
    PacketAccess = false,
  };
};

template <typename T>
struct scalar_fmod2_op {
  EIGEN_EMPTY_STRUCT_CTOR(scalar_fmod2_op)
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const T operator()(const T& a,
                                                           const T& b) const {
    return fmod(a, b);
  }
};

template <typename T>
struct scalar_mod2_op {
  EIGEN_EMPTY_STRUCT_CTOR(scalar_mod2_op)
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const T operator()(const T& a,
                                                           const T& b) const {
    return a % b;
  }
};

template <typename T>
struct functor_traits<scalar_mod2_op<T> > {
  enum {
    Cost = 5,  // Roughly the cost of a div
    PacketAccess = false,
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
          bool PacketAccess = functor_traits<Binary>::PacketAccess>
struct scalar_left {
  typedef Tout result_type;
  const Tin* left;
  EIGEN_DEVICE_FUNC inline scalar_left(
      const scalar_left& other)  // NOLINT(runtime/explicit)
      : left(other.left) {}
  EIGEN_DEVICE_FUNC inline explicit scalar_left(const Tin* c) : left(c) {}
  EIGEN_DEVICE_FUNC inline Tout operator()(const Tin& right) const {
    return Binary()(*left, right);
  }
};

template <typename Tout, typename Tin, typename Binary>
struct scalar_left<Tout, Tin, Binary, true> {
  typedef Tout result_type;
  const Tin* left;
  EIGEN_DEVICE_FUNC inline scalar_left(
      const scalar_left& other)  // NOLINT(runtime/explicit)
      : left(other.left) {}
  EIGEN_DEVICE_FUNC inline explicit scalar_left(const Tin* c) : left(c) {}
  EIGEN_DEVICE_FUNC inline Tout operator()(const Tin& right) const {
    return Binary()(*left, right);
  }

  template <typename Packet>
  EIGEN_DEVICE_FUNC inline Packet packetOp(const Packet& right_packet) const {
    const Packet left_packet = Eigen::internal::pset1<Packet>(*left);
    return Binary().packetOp(left_packet, right_packet);
  }
};

template <typename Tout, typename Tin, typename Binary>
struct functor_traits<scalar_left<Tout, Tin, Binary> > {
  enum {
    Cost = functor_traits<Binary>::Cost,
    PacketAccess = functor_traits<Binary>::PacketAccess,
  };
};

template <typename Tout, typename Tin, typename Binary,
          bool PacketAccess = functor_traits<Binary>::PacketAccess>
struct scalar_right {
  typedef Tout result_type;
  const Tin* right;
  EIGEN_DEVICE_FUNC inline scalar_right(
      const scalar_right& other)  // NOLINT(runtime/explicit)
      : right(other.right) {}
  EIGEN_DEVICE_FUNC inline explicit scalar_right(const Tin* c) : right(c) {}
  EIGEN_DEVICE_FUNC inline Tout operator()(const Tin& left) const {
    return Binary()(left, *right);
  }
};

template <typename Tout, typename Tin, typename Binary>
struct scalar_right<Tout, Tin, Binary, true> {
  typedef Tout result_type;
  const Tin* right;
  EIGEN_DEVICE_FUNC inline scalar_right(
      const scalar_right& other)  // NOLINT(runtime/explicit)
      : right(other.right) {}
  EIGEN_DEVICE_FUNC inline explicit scalar_right(const Tin* c) : right(c) {}
  EIGEN_DEVICE_FUNC inline Tout operator()(const Tin& left) const {
    return Binary()(left, *right);
  }

  template <typename Packet>
  EIGEN_DEVICE_FUNC inline Packet packetOp(const Packet& left_packet) const {
    const Packet right_packet = Eigen::internal::pset1<Packet>(*right);
    return Binary().packetOp(left_packet, right_packet);
  }
};

template <typename Tout, typename Tin, typename Binary>
struct functor_traits<scalar_right<Tout, Tin, Binary> > {
  enum {
    Cost = functor_traits<Binary>::Cost,
    PacketAccess = functor_traits<Binary>::PacketAccess,
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
// log(x) = natural logrithm of x
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
struct neg : base<T, Eigen::internal::scalar_opposite_op<T> > {};

template <typename T>
struct inverse : base<T, Eigen::internal::scalar_inverse_op<T> > {};

template <typename T>
struct square : base<T, Eigen::internal::scalar_square_op<T> > {};

template <typename T>
struct sqrt : base<T, Eigen::internal::scalar_sqrt_op<T> > {};

template <typename T>
struct rsqrt : base<T, Eigen::internal::scalar_rsqrt_op<T> > {};

template <typename T>
struct exp : base<T, Eigen::internal::scalar_exp_op<T> > {};

template <typename T>
struct log : base<T, Eigen::internal::scalar_log_op<T> > {};

template <typename T>
struct sign : base<T, Eigen::internal::scalar_sign_op<T> > {};

template <typename T>
struct tanh : base<T, Eigen::internal::scalar_tanh_op<T> > {};

template <typename T>
struct sigmoid : base<T, Eigen::internal::scalar_sigmoid_op<T> > {};

template <typename T>
struct sin : base<T, Eigen::internal::scalar_sin_op<T> > {};

template <typename T>
struct cos : base<T, Eigen::internal::scalar_cos_op<T> > {};

struct logical_not : base<bool, std::logical_not<bool> > {};

namespace impl {

#ifndef __CUDACC__
// Uses STL std cmath functions.
template <typename T>
bool isinf(T v) {
  return std::isinf(v);
}

template <typename T>
bool isnan(T v) {
  return std::isnan(v);
}

template <typename T>
bool isfinite(T v) {
  return std::isfinite(v);
}

template <typename T>
T floor(T v) {
  return std::floor(v);
}

template <typename T>
T ceil(T v) {
  return std::ceil(v);
}
#else
// Uses CUDA's functions for float and double.
template <typename T>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE bool isinf(T v) {
  return ::isinf(v);
}

template <typename T>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE bool isnan(T v) {
  return ::isnan(v);
}

template <typename T>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE bool isfinite(T v) {
  return ::isfinite(v);
}

template <typename T>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE T floor(T v) {
  return ::floor(v);
}

template <typename T>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE T ceil(T v) {
  return ::ceil(v);
}
#endif
}  // end namespace impl

// NOTE: std::isinf, std::isnan, std::isfinite are plain function.
// Therefore we need to wrap them in functors to be used with Eigen's
// type system.

template <typename T>
struct isinf_func {
  typedef bool result_type;
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE bool operator()(T x) const {
    return impl::isinf(x);
  }
};

template <typename T>
struct isinf : base<T, isinf_func<T>, bool> {};

template <typename T>
struct isnan_func {
  typedef bool result_type;
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE bool operator()(T x) const {
    return impl::isnan(x);
  }
};

template <typename T>
struct isnan : base<T, isnan_func<T>, bool> {};

template <typename T>
struct isfinite_func {
  typedef bool result_type;
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE bool operator()(T x) const {
    return impl::isfinite(x);
  }
};

template <typename T>
struct isfinite : base<T, isfinite_func<T>, bool> {};

template <typename T>
struct floor_func {
  typedef T result_type;
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE T operator()(T x) const {
    return impl::floor(x);
  }
};

template <typename T>
struct floor : base<T, floor_func<T> > {};

template <typename T>
struct ceil_func {
  typedef T result_type;
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE T operator()(T x) const {
    return impl::ceil(x);
  }
};

template <typename T>
struct ceil : base<T, ceil_func<T> > {};

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

template <typename T>
struct add : base<T, Eigen::internal::scalar_sum_op<T> > {
  static const bool use_bcast_optimization = true;
};

template <typename T>
struct sub : base<T, Eigen::internal::scalar_difference_op<T> > {
  static const bool use_bcast_optimization = true;
};

template <typename T>
struct mul : base<T, Eigen::internal::scalar_product_op<T> > {};

template <typename T>
struct div : base<T, Eigen::internal::scalar_quotient_op<T> > {};

template <typename T>
struct fmod : base<T, Eigen::internal::scalar_fmod2_op<T> > {};

template <typename T>
struct mod : base<T, Eigen::internal::scalar_mod2_op<T> > {};

template <typename T>
struct pow : base<T, Eigen::internal::scalar_pow2_op<T> > {};

template <typename T>
struct maximum : base<T, Eigen::internal::scalar_max_op<T> > {};

template <typename T>
struct minimum : base<T, Eigen::internal::scalar_min_op<T> > {};

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
struct make_complex_func {
  typedef std::complex<T> result_type;
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE result_type operator()(T real,
                                                               T imag) const {
    return std::complex<T>(real, imag);
  }
};

template <typename T>
struct make_complex : base<T, make_complex_func<T>, std::complex<T> > {};

template <typename T>
struct get_real
    : base<T, Eigen::internal::scalar_real_op<T>, typename T::value_type> {};

template <typename T>
struct get_imag
    : base<T, Eigen::internal::scalar_imag_op<T>, typename T::value_type> {};

template <typename T>
struct conj : base<T, Eigen::internal::scalar_conjugate_op<T> > {};

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

template <typename Device, typename Functor, int NDIMS>
struct BinaryFunctor {
  // Computes on device "d": out[i] = Functor(in0[i], in1[i])
  void operator()(const Device& d, typename Functor::tout_type out,
                  typename Functor::tin_type in0,
                  typename Functor::tin_type in1);

  // Computes on device "d": out[i] = Functor(scalar[0], in[i])
  void Left(const Device& d, typename Functor::tout_type out,
            typename Functor::tscalar_type scalar,
            typename Functor::tin_type in);

  // Computes on device "d": out[i] = Functor(in[i], scalar[0])
  void Right(const Device& d, typename Functor::tout_type out,
             typename Functor::tin_type in,
             typename Functor::tscalar_type scalar);

  // Computes on device "d":
  //   out = Functor(in0.broadcast(bcast0), in1.broadcast(bcast01))
  //
  // TODO(zhifengc): makes BCast a template member function on NDIMS
  // instead making BinaryFunctor templates on NDIMS.
  void BCast(const Device& d,
             typename TTypes<typename Functor::out_type, NDIMS>::Tensor out,
             typename TTypes<typename Functor::in_type, NDIMS>::ConstTensor in0,
             typename Eigen::array<Eigen::DenseIndex, NDIMS> bcast0,
             typename TTypes<typename Functor::in_type, NDIMS>::ConstTensor in1,
             typename Eigen::array<Eigen::DenseIndex, NDIMS> bcast1);
};

template <int NDIMS>
bool AllOne(const typename Eigen::array<Eigen::DenseIndex, NDIMS>& a) {
  for (int i = 0; i < a.size(); ++i) {
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

}  // end namespace functor
}  // end namespace tensorflow

#endif  // TENSORFLOW_KERNELS_CWISE_OPS_H_

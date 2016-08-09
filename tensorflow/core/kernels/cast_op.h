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

#ifndef TENSORFLOW_KERNELS_CAST_OP_H_
#define TENSORFLOW_KERNELS_CAST_OP_H_

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/bfloat16.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/platform/host_info.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace functor {

template <typename Device, typename Tout, typename Tin>
void Cast(const Device& d, typename TTypes<Tout>::Flat o,
          typename TTypes<Tin>::ConstFlat i) {
  o.device(d) = i.template cast<Tout>();
}

template <typename Device, typename Tout, typename Tin>
struct CastFunctor {
  void operator()(const Device& d, typename TTypes<Tout>::Flat o,
                  typename TTypes<Tin>::ConstFlat i);
};

}  // end namespace functor
}  // end namespace tensorflow

namespace Eigen {
namespace internal {

// Eigen can't convert to/from complex numbers, because it is limited to cases
// that can be static_casted. But numpy is able to cast to/from complex, which
// we want to emulate. So we add specializations for complex here.
typedef std::complex<float> complex64;
typedef std::complex<double> complex128;
using tensorflow::uint8;
using tensorflow::int8;
using tensorflow::uint16;
using tensorflow::int16;
using tensorflow::int32;
using tensorflow::int64;

// These are seperate definitions of what should happen when we cast from/to
// complex numbers. The actual template specializations are instantiated below.
template<typename From, typename To>
struct cast_from_complex_impl {
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE To operator()(const From& a) const {
    return static_cast<To>(a.real());
  }
};

template<typename From, typename To>
struct cast_to_complex_impl {
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE To operator()(const From& a) const {
    return To(static_cast<typename To::value_type>(a), typename To::value_type(0));
  }
};

template<typename From, typename To>
struct cast_from_to_complex_impl {
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE To operator()(
    const From& a) const {
    return To(static_cast<typename To::value_type>(a.real()),
              static_cast<typename To::value_type>(a.imag()));
  }
};

template<typename From, typename To>
struct functor_traits_complex_impl {
  enum { Cost = NumTraits<To>::AddCost, PacketAccess = false };
};

#define FROM_COMPLEX(C, T)                    \
  template<>                                  \
  struct scalar_cast_op<C, T>                 \
    : cast_from_complex_impl<C, T> {};        \
  template<>                                  \
  struct functor_traits<scalar_cast_op<C, T>> \
    : functor_traits_complex_impl<C, T> {}

#define TO_COMPLEX(T, C)                      \
  template <>                                 \
  struct scalar_cast_op<T, C>                 \
    : cast_to_complex_impl<T, C> {};          \
  template<>                                  \
  struct functor_traits<scalar_cast_op<T, C>> \
    : functor_traits_complex_impl<T, C> {}

#define FROM_TO_COMPLEX(C1, C2)                 \
  template <>                                   \
  struct scalar_cast_op<C1, C2>                 \
    : cast_from_to_complex_impl<C1, C2> {};     \
  template<>                                    \
  struct functor_traits<scalar_cast_op<C1, C2>> \
    : functor_traits_complex_impl<C1, C2> {}

#define CURRY_TYPES_FROM(FN, arg0) \
  FN(arg0, bool);                  \
  FN(arg0, uint8);                 \
  FN(arg0, int8);                  \
  FN(arg0, uint16);                \
  FN(arg0, int16);                 \
  FN(arg0, int32);                 \
  FN(arg0, int64);                 \
  FN(arg0, Eigen::half);           \
  FN(arg0, float);                 \
  FN(arg0, double)

#define CURRY_TYPES_TO(FN, arg0) \
  FN(bool,        arg0);         \
  FN(uint8,       arg0);         \
  FN(int8,        arg0);         \
  FN(uint16,      arg0);         \
  FN(int16,       arg0);         \
  FN(int32,       arg0);         \
  FN(int64,       arg0);         \
  FN(Eigen::half, arg0);         \
  FN(float,       arg0);         \
  FN(double,      arg0)

CURRY_TYPES_FROM(FROM_COMPLEX, complex64);
CURRY_TYPES_FROM(FROM_COMPLEX, complex128);
CURRY_TYPES_TO(TO_COMPLEX, complex64);
CURRY_TYPES_TO(TO_COMPLEX, complex128);
FROM_TO_COMPLEX(complex64, complex64);
FROM_TO_COMPLEX(complex64, complex128);
FROM_TO_COMPLEX(complex128, complex64);
FROM_TO_COMPLEX(complex128, complex128);

#undef FROM_COMPLEX
#undef TO_COMPLEX
#undef FROM_TO_COMPLEX
#undef CURRY_TYPES_FROM
#undef CURRY_TYPES_TO

// Specialized cast op impls for bfloat16.
template <>
struct scalar_cast_op< ::tensorflow::bfloat16, float> {
  EIGEN_EMPTY_STRUCT_CTOR(scalar_cast_op)
  typedef float result_type;
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE float operator()(
      const ::tensorflow::bfloat16& a) const {
    static_assert(::tensorflow::port::kLittleEndian, "");
    float ret;
    uint16_t* p = reinterpret_cast<uint16_t*>(&ret);
    p[0] = 0;
    p[1] = a.value;
    return ret;
  }
};

template <>
struct functor_traits<scalar_cast_op< ::tensorflow::bfloat16, float> > {
  enum { Cost = NumTraits<float>::AddCost, PacketAccess = false };
};

template <>
struct scalar_cast_op<float, ::tensorflow::bfloat16> {
  EIGEN_EMPTY_STRUCT_CTOR(scalar_cast_op)
  typedef ::tensorflow::bfloat16 result_type;
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const ::tensorflow::bfloat16 operator()(
      const float a) const {
    static_assert(::tensorflow::port::kLittleEndian, "");
    const uint16_t* p = reinterpret_cast<const uint16_t*>(&a);
    return ::tensorflow::bfloat16(p[1]);
  }
};

template <>
struct functor_traits<scalar_cast_op<float, ::tensorflow::bfloat16> > {
  enum { Cost = NumTraits<float>::AddCost, PacketAccess = false };
};

}  // namespace internal
}  // namespace Eigen

#endif  // TENSORFLOW_KERNELS_CAST_OP_H_

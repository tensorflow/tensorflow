/* Copyright 2015 Google Inc. All Rights Reserved.

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

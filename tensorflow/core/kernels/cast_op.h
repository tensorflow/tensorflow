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
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/platform/cpu_info.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

// Common base class of Cast kernels
class CastOpBase : public OpKernel {
 public:
  explicit CastOpBase(OpKernelConstruction* ctx);

  void Compute(OpKernelContext* ctx) override;

 protected:
  DataType src_dtype_;
  DataType dst_dtype_;
  std::function<void(OpKernelContext*, const Tensor&, Tensor*)> work_ = nullptr;

  Status Unimplemented();

  TF_DISALLOW_COPY_AND_ASSIGN(CastOpBase);
};

// CPU implementation of Cast
class CpuCastOp : public CastOpBase {
 public:
  explicit CpuCastOp(OpKernelConstruction* ctx);

 private:
  Status Prepare();
};

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
// we want to replicate. So we add specializations for complex here.
template <typename From, typename To>
struct scalar_cast_op<std::complex<From>, To> {
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE To
  operator()(const std::complex<From>& a) const {
    // Replicate numpy behavior of returning just the real part
    return static_cast<To>(a.real());
  }
};

template <typename From, typename To>
struct scalar_cast_op<From, std::complex<To>> {
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE std::complex<To> operator()(
      const From& a) const {
    // Replicate numpy behavior of setting the imaginary part to 0
    return std::complex<To>(static_cast<To>(a), To(0));
  }
};

template <typename From, typename To>
struct scalar_cast_op<std::complex<From>, std::complex<To>> {
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE std::complex<To> operator()(
      const std::complex<From>& a) const {
    return std::complex<To>(static_cast<To>(a.real()),
                            static_cast<To>(a.imag()));
  }
};

template <typename From, typename To>
struct functor_traits_complex_impl {
  enum { Cost = NumTraits<To>::AddCost, PacketAccess = false };
};

template <typename From, typename To>
struct functor_traits<scalar_cast_op<std::complex<From>, To>>
    : functor_traits_complex_impl<std::complex<From>, To> {};
template <typename From, typename To>
struct functor_traits<scalar_cast_op<From, std::complex<To>>>
    : functor_traits_complex_impl<From, std::complex<To>> {};
// Needed to avoid ambiguous partial specialization
template <typename From, typename To>
struct functor_traits<scalar_cast_op<std::complex<From>, std::complex<To>>>
    : functor_traits_complex_impl<std::complex<From>, std::complex<To>> {};

// Specialized cast op impls for bfloat16.
template <>
struct scalar_cast_op<::tensorflow::bfloat16, float> {
  EIGEN_EMPTY_STRUCT_CTOR(scalar_cast_op)
  typedef float result_type;
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE float operator()(
      const ::tensorflow::bfloat16& a) const {
    float ret;
    uint16_t* p = reinterpret_cast<uint16_t*>(&ret);
#if __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
    p[0] = a.value;  
    p[1] = 0;  
#else  
    static_assert(::tensorflow::port::kLittleEndian, "Not a little endian system!");  
    p[0] = 0;
    p[1] = a.value;
#endif
    return ret;
  }
};

template <>
struct functor_traits<scalar_cast_op<::tensorflow::bfloat16, float>> {
  enum { Cost = NumTraits<float>::AddCost, PacketAccess = false };
};

template <>
struct scalar_cast_op<float, ::tensorflow::bfloat16> {
  EIGEN_EMPTY_STRUCT_CTOR(scalar_cast_op)
  typedef ::tensorflow::bfloat16 result_type;
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const ::tensorflow::bfloat16 operator()(
      const float a) const {
    return ::tensorflow::bfloat16(a);
  }
};

template <>
struct functor_traits<scalar_cast_op<float, ::tensorflow::bfloat16>> {
  enum { Cost = NumTraits<float>::AddCost, PacketAccess = false };
};

}  // namespace internal
}  // namespace Eigen

#endif  // TENSORFLOW_KERNELS_CAST_OP_H_

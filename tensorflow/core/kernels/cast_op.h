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

#ifndef TENSORFLOW_CORE_KERNELS_CAST_OP_H_
#define TENSORFLOW_CORE_KERNELS_CAST_OP_H_

#include "unsupported/Eigen/CXX11/Tensor"  // from @eigen_archive
#include "tensorflow/core/framework/bfloat16.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/platform/byte_order.h"
#include "tensorflow/core/platform/types.h"

// Note that the GPU cast functor templates need to be instantiated unlike the
// CPU ones, and hence their specializations are different than that for CPUs.
#ifdef SPECIALIZE_FOR_GPUS
#define SPECIALIZE_CAST(DEVICE, OUT_TYPE, IN_TYPE)                  \
  template <typename Device>                                        \
  struct CastFunctor<Device, OUT_TYPE, IN_TYPE> {                   \
    void operator()(const Device& d,                                \
                    typename TTypes<OUT_TYPE>::Flat out_tensor,     \
                    typename TTypes<IN_TYPE>::ConstFlat in_tensor,  \
                    bool truncate = false) {                        \
      if (truncate) {                                               \
        out_tensor.device(d) =                                      \
            in_tensor.unaryExpr(LSBZeroSetter<IN_TYPE, OUT_TYPE>()) \
                .template cast<OUT_TYPE>();                         \
      } else {                                                      \
        out_tensor.device(d) = in_tensor.template cast<OUT_TYPE>(); \
      }                                                             \
    }                                                               \
  };                                                                \
  template struct CastFunctor<DEVICE, OUT_TYPE, IN_TYPE>;
#else
#define SPECIALIZE_CAST(DEVICE, OUT_TYPE, IN_TYPE)                  \
  template <>                                                       \
  struct CastFunctor<DEVICE, OUT_TYPE, IN_TYPE> {                   \
    void operator()(const DEVICE& d,                                \
                    typename TTypes<OUT_TYPE>::Flat out_tensor,     \
                    typename TTypes<IN_TYPE>::ConstFlat in_tensor,  \
                    bool truncate = false) {                        \
      if (truncate) {                                               \
        out_tensor.device(d) =                                      \
            in_tensor.unaryExpr(LSBZeroSetter<IN_TYPE, OUT_TYPE>()) \
                .template cast<OUT_TYPE>();                         \
      } else {                                                      \
        out_tensor.device(d) = in_tensor.template cast<OUT_TYPE>(); \
      }                                                             \
    }                                                               \
  };
#endif

#define CAST_FUNCTORS(devname)                                        \
  SPECIALIZE_CAST(devname, float, double)                             \
  SPECIALIZE_CAST(devname, float, std::complex<double>)               \
  SPECIALIZE_CAST(devname, std::complex<float>, std::complex<double>) \
  SPECIALIZE_CAST(devname, std::complex<float>, double)               \
  SPECIALIZE_CAST(devname, Eigen::half, double)                       \
  SPECIALIZE_CAST(devname, Eigen::half, float)                        \
  SPECIALIZE_CAST(devname, Eigen::half, std::complex<double>)         \
  SPECIALIZE_CAST(devname, Eigen::half, std::complex<float>)          \
  SPECIALIZE_CAST(devname, bfloat16, float)                           \
  SPECIALIZE_CAST(devname, float8_e5m2, double)                       \
  SPECIALIZE_CAST(devname, float8_e5m2, float)                        \
  SPECIALIZE_CAST(devname, float8_e5m2, bfloat16)                     \
  SPECIALIZE_CAST(devname, float8_e5m2, Eigen::half)                  \
  SPECIALIZE_CAST(devname, float8_e5m2, float8_e4m3fn)                \
  SPECIALIZE_CAST(devname, float8_e4m3fn, double)                     \
  SPECIALIZE_CAST(devname, float8_e4m3fn, float)                      \
  SPECIALIZE_CAST(devname, float8_e4m3fn, bfloat16)                   \
  SPECIALIZE_CAST(devname, float8_e4m3fn, Eigen::half)                \
  template <typename OUT_TYPE, typename IN_TYPE>                      \
  struct CastFunctor<devname, OUT_TYPE, IN_TYPE> {                    \
    void operator()(const devname& d,                                 \
                    typename TTypes<OUT_TYPE>::Flat out_tensor,       \
                    typename TTypes<IN_TYPE>::ConstFlat in_tensor,    \
                    bool truncate = false) {                          \
      out_tensor.device(d) = in_tensor.template cast<OUT_TYPE>();     \
    }                                                                 \
  };

#if defined(MLIR_GENERATED_GPU_KERNELS_ENABLED)
// If MLIR kernels are enabled, we don't need the specialized cast from float to
// double or from Eigen::half to double. We still need the specialized cast from
// Eigen::half to float, because it is used in depthwise_conv_grad_op.cc. We
// still need the specialized cast from float to double because it is used in
// resize_bilinear_op.cc.
#define CAST_FUNCTORS_SUBSET(devname)                              \
  SPECIALIZE_CAST(devname, float, double)                          \
  SPECIALIZE_CAST(devname, Eigen::half, float)                     \
  SPECIALIZE_CAST(devname, bfloat16, float)                        \
  SPECIALIZE_CAST(devname, float8_e5m2, double)                    \
  SPECIALIZE_CAST(devname, float8_e5m2, float)                     \
  SPECIALIZE_CAST(devname, float8_e5m2, bfloat16)                  \
  SPECIALIZE_CAST(devname, float8_e5m2, Eigen::half)               \
  SPECIALIZE_CAST(devname, float8_e5m2, float8_e4m3fn)             \
  SPECIALIZE_CAST(devname, float8_e4m3fn, double)                  \
  SPECIALIZE_CAST(devname, float8_e4m3fn, float)                   \
  SPECIALIZE_CAST(devname, float8_e4m3fn, bfloat16)                \
  SPECIALIZE_CAST(devname, float8_e4m3fn, Eigen::half)             \
  template <typename OUT_TYPE, typename IN_TYPE>                   \
  struct CastFunctor<devname, OUT_TYPE, IN_TYPE> {                 \
    void operator()(const devname& d,                              \
                    typename TTypes<OUT_TYPE>::Flat out_tensor,    \
                    typename TTypes<IN_TYPE>::ConstFlat in_tensor, \
                    bool truncate = false) {                       \
      out_tensor.device(d) = in_tensor.template cast<OUT_TYPE>();  \
    }                                                              \
  };
#endif

namespace tensorflow {

typedef std::function<void(OpKernelContext*, const Tensor&, Tensor*,
                           bool trunc)>
    CastFunctorType;

// Common base class of Cast kernels
class CastOpBase : public OpKernel {
 public:
  explicit CastOpBase(OpKernelConstruction* ctx);

  void Compute(OpKernelContext* ctx) override;

 protected:
  DataType src_dtype_;
  DataType dst_dtype_;
  DataType external_src_dtype_;
  DataType external_dst_dtype_;
  bool use_truncation_;
  CastFunctorType work_ = nullptr;
  absl::Status Unimplemented();

  CastOpBase(const CastOpBase&) = delete;
  void operator=(const CastOpBase&) = delete;
};

// CPU implementation of Cast
class CpuCastOp : public CastOpBase {
 public:
  explicit CpuCastOp(OpKernelConstruction* ctx);

 private:
  absl::Status Prepare();
};

namespace functor {

template <typename I>
constexpr int MantissaWidth() {
  return std::numeric_limits<I>::digits;
}

template <>
constexpr int MantissaWidth<Eigen::half>() {
  // Remember, there's 1 hidden bit
  return 10 + 1;
}

template <>
constexpr int MantissaWidth<bfloat16>() {
  // Remember, there's 1 hidden bit
  return 7 + 1;
}

template <typename Device, typename Tout, typename Tin>
void Cast(const Device& d, typename TTypes<Tout>::Flat o,
          typename TTypes<Tin>::ConstFlat i) {
  o.device(d) = i.template cast<Tout>();
}

template <typename Device, typename Tout, typename Tin>
struct CastFunctor {
  void operator()(const Device& d, typename TTypes<Tout>::Flat o,
                  typename TTypes<Tin>::ConstFlat i, bool truncate = false);
};

template <typename I>
typename std::enable_if<sizeof(I) == 8, void>::type EIGEN_DEVICE_FUNC
    EIGEN_STRONG_INLINE static LSBZeroSetterHelper(I& t, int n) {
  // Only zero the bits for non-NaNs.
  // For NaNs, let the non-truncation version handle it.
  if (!Eigen::numext::isnan(t)) {
    uint64_t* p = reinterpret_cast<uint64_t*>(&t);
    *p &= (0xFFFFFFFFFFFFFFFF << n);
  }
}

template <typename I>
typename std::enable_if<sizeof(I) == 4, void>::type EIGEN_DEVICE_FUNC
    EIGEN_STRONG_INLINE static LSBZeroSetterHelper(I& t, int n) {
  // Only zero the bits for non-NaNs.
  // For NaNs, let the non-truncation version handle it.
  if (!Eigen::numext::isnan(t)) {
    uint32_t* p = reinterpret_cast<uint32_t*>(&t);
    *p &= (0xFFFFFFFF << n);
  }
}

template <typename I>
typename std::enable_if<sizeof(I) == 2, void>::type EIGEN_DEVICE_FUNC
    EIGEN_STRONG_INLINE static LSBZeroSetterHelper(I& t, int n) {
  // Only zero the bits for non-NaNs.
  // For NaNs, let the non-truncation version handle it.
  if (!Eigen::numext::isnan(t)) {
    uint16_t* p = reinterpret_cast<uint16_t*>(&t);
    *p &= (0xFFFF << n);
  }
}

template <typename I>
typename std::enable_if<sizeof(I) == 1, void>::type EIGEN_DEVICE_FUNC
    EIGEN_STRONG_INLINE static LSBZeroSetterHelper(I& t, int n) {
  // Only zero the bits for non-NaNs.
  // For NaNs, let the non-truncation version handle it.
  if (!Eigen::numext::isnan(t)) {
    uint8_t* p = reinterpret_cast<uint8_t*>(&t);
    *p &= (0xFF << n);
  }
}

// Set n least significant bits to 0
template <typename I, typename O>
struct LSBZeroSetter {
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE I operator()(const I& a) const {
    constexpr int bits = MantissaWidth<I>() - MantissaWidth<O>();
    static_assert(
        bits > 0,
        "The output type must have fewer mantissa bits than the input type\n");
    I t = a;
    LSBZeroSetterHelper(t, bits);
    return t;
  }
};

template <typename I, typename O>
struct LSBZeroSetter<std::complex<I>, std::complex<O>> {
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE std::complex<I> operator()(
      const std::complex<I>& a) const {
    constexpr int bits = MantissaWidth<I>() - MantissaWidth<O>();
    static_assert(
        bits > 0,
        "The output type must have fewer mantissa bits than the input type\n");
    I re = Eigen::numext::real(a);
    I img = Eigen::numext::imag(a);
    LSBZeroSetterHelper(re, bits);
    LSBZeroSetterHelper(img, bits);
    std::complex<I> toReturn(re, img);
    return toReturn;
  }
};

template <typename I, typename O>
struct LSBZeroSetter<std::complex<I>, O> {
  // Sets the 16 LSBits of the float to 0
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE std::complex<I> operator()(
      const std::complex<I>& a) const {
    constexpr int bits = MantissaWidth<I>() - MantissaWidth<O>();
    static_assert(
        bits > 0,
        "The output type must have fewer mantissa bits than the input type\n");
    I re = Eigen::numext::real(a);
    I img = Eigen::numext::imag(a);
    LSBZeroSetterHelper(re, bits);
    LSBZeroSetterHelper(img, bits);
    std::complex<I> toReturn(re, img);
    return toReturn;
  }
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

template <typename From>
struct scalar_cast_op<std::complex<From>, bool> {
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE bool operator()(
      const std::complex<From>& a) const {
    return static_cast<bool>(a.real());
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

template <typename From>
struct functor_traits<scalar_cast_op<std::complex<From>, bool>>
    : functor_traits_complex_impl<std::complex<From>, bool> {};

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

}  // namespace internal
}  // namespace Eigen

#endif  // TENSORFLOW_CORE_KERNELS_CAST_OP_H_

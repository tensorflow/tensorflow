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

#if !GOOGLE_CUDA && !TENSORFLOW_USE_ROCM
#error This file must only be included when building with Cuda or ROCm support
#endif

#ifndef TENSORFLOW_CORE_KERNELS_CWISE_OPS_GPU_COMMON_CU_H_
#define TENSORFLOW_CORE_KERNELS_CWISE_OPS_GPU_COMMON_CU_H_

#define _USE_MATH_DEFINES
#include <cmath>
#include <complex>

#define EIGEN_USE_GPU
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/kernels/cwise_ops.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
namespace tensorflow {
namespace functor {

typedef Eigen::GpuDevice GPUDevice;
typedef std::complex<float> complex64;
typedef std::complex<double> complex128;

// Partial specialization of UnaryFunctor<Device=GPUDevice, Functor>.
template <typename Functor>
struct UnaryFunctor<GPUDevice, Functor> {
  void operator()(const GPUDevice& d, typename Functor::tout_type out,
                  typename Functor::tin_type in) {
    MaybeWith32BitIndexing<GPUDevice>(
        [&](auto out32, auto in32) {
          out32.device(d) = in32.unaryExpr(typename Functor::func());
        },
        out, in);
  }
};

// Partial specialization of BinaryFunctor<Device=GPUDevice, Functor>.
template <typename Functor, int NDIMS, bool has_errors>
struct BinaryFunctor<GPUDevice, Functor, NDIMS, has_errors> {
  void operator()(const GPUDevice& d, typename Functor::tout_type out,
                  typename Functor::tin_type in0,
                  typename Functor::tin_type in1, bool* error) {
    MaybeWith32BitIndexing<GPUDevice>(
        [&](auto out32, auto in0_32, auto in1_32) {
          out32.device(d) = in0_32.binaryExpr(in1_32, typename Functor::func());
        },
        out, in0, in1);
  }

  void Left(const GPUDevice& d, typename Functor::tout_type out,
            typename Functor::tscalar_type scalar,
            typename Functor::tin_type in, bool* error) {
    typedef typename Functor::out_type Tout;
    typedef typename Functor::in_type Tin;
    typedef typename Functor::func Binary;
    typedef typename Eigen::internal::scalar_left<Tout, Tin, Binary> Unary;
    MaybeWith32BitIndexing<GPUDevice>(
        [&](auto out32, auto in32) {
          out32.device(d) = in32.unaryExpr(Unary(scalar.data()));
        },
        out, in);
  }

  void Right(const GPUDevice& d, typename Functor::tout_type out,
             typename Functor::tin_type in,
             typename Functor::tscalar_type scalar, bool* error) {
    typedef typename Functor::out_type Tout;
    typedef typename Functor::in_type Tin;
    typedef typename Functor::func Binary;
    typedef typename Eigen::internal::scalar_right<Tout, Tin, Binary> Unary;
    MaybeWith32BitIndexing<GPUDevice>(
        [&](auto out32, auto in32) {
          out32.device(d) = in32.unaryExpr(Unary(scalar.data()));
        },
        out, in);
  }

  void BCast(const GPUDevice& d,
             typename TTypes<typename Functor::out_type, NDIMS>::Tensor out,
             typename TTypes<typename Functor::in_type, NDIMS>::ConstTensor in0,
             typename Eigen::array<Eigen::DenseIndex, NDIMS> bcast0,
             typename TTypes<typename Functor::in_type, NDIMS>::ConstTensor in1,
             typename Eigen::array<Eigen::DenseIndex, NDIMS> bcast1,
             bool* error) {
    typedef typename Functor::in_type T;
    typename Functor::func func;
    if ((NDIMS == 2) && Functor::use_bcast_optimization &&
        use_bcast_optimization<T>::value) {
      const bool bcast0_all_one = AllOne<NDIMS>(bcast0);
      const bool bcast1_all_one = AllOne<NDIMS>(bcast1);
      if (bcast0_all_one && !bcast1_all_one) {
        MaybeWith32BitIndexing<GPUDevice>(
            [&](auto out32, auto in0_32, auto in1_32) {
              out32.device(d) =
                  in0_32.binaryExpr(in1_32.broadcast(bcast1), func);
            },
            out, in0, in1);
        return;
      }
      if (!bcast0_all_one && bcast1_all_one) {
        MaybeWith32BitIndexing<GPUDevice>(
            [&](auto out32, auto in0_32, auto in1_32) {
              out32.device(d) =
                  in0_32.broadcast(bcast0).binaryExpr(in1_32, func);
            },
            out, in0, in1);
        return;
      }
    }
    MaybeWith32BitIndexing<GPUDevice>(
        [&](auto out32, auto in0_32, auto in1_32) {
          out32.device(d) = in0_32.broadcast(bcast0).binaryExpr(
              in1_32.broadcast(bcast1), func);
        },
        out, in0, in1);
  }
};

// Partial specialization of ApproximateEqual<Device=GPUDevice, T>.
template <typename T>
struct ApproximateEqual<GPUDevice, T> {
  void operator()(const GPUDevice& d, typename TTypes<T>::ConstFlat x,
                  typename TTypes<T>::ConstFlat y, T tolerance,
                  typename TTypes<bool>::Flat z) {
    auto diff = x - y;
    z.device(d) = diff.abs() <= tolerance;
  }
};

// IEEE 754 compliant complex abs specializations for GPU
// Fix for TensorFlow issue #98410: inconsistent inf/nan handling
template <>
struct UnaryFunctor<GPUDevice, functor::abs<complex64>> {
  void operator()(const GPUDevice& d, typename TTypes<complex64>::ConstFlat& in,
                  typename TTypes<float>::Flat& out) {
    auto ieee754_abs = [] __device__ (const complex64& x) -> float {
      float real_part = x.real();
      float imag_part = x.imag();
      
      // IEEE 754: if either component is inf, result is inf
      if (isinf(real_part) || isinf(imag_part)) {
        return INFINITY;
      }
      
      return hypot(real_part, imag_part);
    };
    
    MaybeWith32BitIndexing<GPUDevice>(
        [&](auto out32, auto in32) {
          out32.device(d) = in32.unaryExpr(ieee754_abs);
        },
        out, in);
  }
};

template <>
struct UnaryFunctor<GPUDevice, functor::abs<complex128>> {
  void operator()(const GPUDevice& d, typename TTypes<complex128>::ConstFlat& in,
                  typename TTypes<double>::Flat& out) {
    auto ieee754_abs = [] __device__ (const complex128& x) -> double {
      double real_part = x.real();
      double imag_part = x.imag();
      
      // IEEE 754: if either component is inf, result is inf
      if (isinf(real_part) || isinf(imag_part)) {
        return INFINITY;
      }
      
      return hypot(real_part, imag_part);
    };
    
    MaybeWith32BitIndexing<GPUDevice>(
        [&](auto out32, auto in32) {
          out32.device(d) = in32.unaryExpr(ieee754_abs);
        },
        out, in);
  }
};

// Macros to explicitly instantiate kernels on GPU for multiple types
// (T0, T1, etc.) for UnaryFunctor (e.g., functor::sqrt).
#define DEFINE_UNARY1(F, T) template struct UnaryFunctor<GPUDevice, F<T> >
#define DEFINE_UNARY2(F, T0, T1) \
  DEFINE_UNARY1(F, T0);          \
  DEFINE_UNARY1(F, T1)
#define DEFINE_UNARY3(F, T0, T1, T2) \
  DEFINE_UNARY2(F, T0, T1);          \
  DEFINE_UNARY1(F, T2)
#define DEFINE_UNARY4(F, T0, T1, T2, T3) \
  DEFINE_UNARY2(F, T0, T1);              \
  DEFINE_UNARY2(F, T2, T3)
#define DEFINE_UNARY5(F, T0, T1, T2, T3, T4) \
  DEFINE_UNARY2(F, T0, T1);                  \
  DEFINE_UNARY3(F, T2, T3, T4)
#define DEFINE_UNARY6(F, T0, T1, T2, T3, T4, T5) \
  DEFINE_UNARY2(F, T0, T1);                      \
  DEFINE_UNARY4(F, T2, T3, T4, T5)
#define DEFINE_UNARY7(F, T0, T1, T2, T3, T4, T5, T6) \
  DEFINE_UNARY2(F, T0, T1);                          \
  DEFINE_UNARY5(F, T2, T3, T4, T5, T6)
#define DEFINE_UNARY8(F, T0, T1, T2, T3, T4, T5, T6, T7) \
  DEFINE_UNARY4(F, T0, T1, T2, T3);                      \
  DEFINE_UNARY4(F, T4, T5, T6, T7)

// Macros to explicitly instantiate kernels on GPU for multiple types
// (T0, T1, etc.) for BinaryFunctor.
#define DEFINE_BINARY1(F, T)                         \
  template struct BinaryFunctor<GPUDevice, F<T>, 1>; \
  template struct BinaryFunctor<GPUDevice, F<T>, 2>; \
  template struct BinaryFunctor<GPUDevice, F<T>, 3>; \
  template struct BinaryFunctor<GPUDevice, F<T>, 4>; \
  template struct BinaryFunctor<GPUDevice, F<T>, 5>
#define DEFINE_BINARY2(F, T0, T1) \
  DEFINE_BINARY1(F, T0);          \
  DEFINE_BINARY1(F, T1)
#define DEFINE_BINARY3(F, T0, T1, T2) \
  DEFINE_BINARY2(F, T0, T1);          \
  DEFINE_BINARY1(F, T2)
#define DEFINE_BINARY4(F, T0, T1, T2, T3) \
  DEFINE_BINARY2(F, T0, T1);              \
  DEFINE_BINARY2(F, T2, T3)
#define DEFINE_BINARY5(F, T0, T1, T2, T3, T4) \
  DEFINE_BINARY2(F, T0, T1);                  \
  DEFINE_BINARY3(F, T2, T3, T4)
#define DEFINE_BINARY6(F, T0, T1, T2, T3, T4, T5) \
  DEFINE_BINARY3(F, T0, T1, T2);                  \
  DEFINE_BINARY3(F, T3, T4, T5)
#define DEFINE_BINARY7(F, T0, T1, T2, T3, T4, T5, T6) \
  DEFINE_BINARY3(F, T0, T1, T2);                      \
  DEFINE_BINARY4(F, T3, T4, T5, T6)
#define DEFINE_BINARY8(F, T0, T1, T2, T3, T4, T5, T6, T7) \
  DEFINE_BINARY4(F, T0, T1, T2, T3);                      \
  DEFINE_BINARY4(F, T4, T5, T6, T7)
#define DEFINE_BINARY9(F, T0, T1, T2, T3, T4, T5, T6, T7, T8) \
  DEFINE_BINARY4(F, T0, T1, T2, T3);                          \
  DEFINE_BINARY5(F, T4, T5, T6, T7, T8)
#define DEFINE_BINARY10(F, T0, T1, T2, T3, T4, T5, T6, T7, T8, T9) \
  DEFINE_BINARY5(F, T0, T1, T2, T3, T4);                           \
  DEFINE_BINARY5(F, T5, T6, T7, T8, T9)
#define DEFINE_BINARY11(F, T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10) \
  DEFINE_BINARY5(F, T0, T1, T2, T3, T4);                                \
  DEFINE_BINARY6(F, T5, T6, T7, T8, T9, T10)

#define DEFINE_APPROXIMATE_EQUAL1(T) \
  template struct ApproximateEqual<GPUDevice, T>;
#define DEFINE_APPROXIMATE_EQUAL2(T0, T1) \
  DEFINE_APPROXIMATE_EQUAL1(T0);          \
  DEFINE_APPROXIMATE_EQUAL1(T1);

}  // end namespace functor
}  // end namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_CWISE_OPS_GPU_COMMON_CU_H_
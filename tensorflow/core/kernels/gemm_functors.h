/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

// This is a set of different implementations for the basic matrix by matrix
// multiply function, commonly known as GEMM after the BLAS library's naming.
// Having a standard interface enables us to swap out implementations on
// different platforms, to make sure we're using the optimal version. They are
// implemented as C++ template functors, so they're easy to swap into all of the
// different kernels that use them.

#if !defined(EIGEN_USE_THREADS)
#error "EIGEN_USE_THREADS must be enabled by all .cc files including this."
#endif  // EIGEN_USE_THREADS

#ifndef TENSORFLOW_CORE_KERNELS_GEMM_FUNCTORS_H_
#define TENSORFLOW_CORE_KERNELS_GEMM_FUNCTORS_H_

#include <string.h>
#include <map>
#include <vector>

#include "tensorflow/core/common_runtime/threadpool_device.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_types.h"

#if defined(TENSORFLOW_USE_CUSTOM_CONTRACTION_KERNEL)
#include "tensorflow/core/kernels/eigen_contraction_kernel.h"
#endif

// Apple provides an optimized BLAS library that is better than Eigen for their
// devices, so use that if possible.
#if defined(__APPLE__) && defined(USE_GEMM_FOR_CONV)
#include <Accelerate/Accelerate.h>
#define USE_CBLAS_GEMM
#endif  // __APPLE__

// Older Raspberry Pi systems don't have NEON SIMD acceleration, so Eigen falls
// back to scalar code, but OpenBLAS has much faster support so prefer that.
#if defined(RASPBERRY_PI) && defined(USE_GEMM_FOR_CONV) && defined(USE_OPENBLAS)
#include <cblas.h>
#define USE_CBLAS_GEMM
#endif

// A readable but slow implementation of matrix multiplication, useful for
// debugging and understanding the algorithm. Use instead of FastGemmFunctor in
// the Im2ColConvFunctor template definition inside the op registration to
// enable. Assumes row-major ordering of the values in memory.
template <class T1, class T2, class T3>
class ReferenceGemmFunctor {
 public:
  void operator()(tensorflow::OpKernelContext* ctx, size_t m, size_t n,
                  size_t k, const T1* a, size_t lda, const T2* b, size_t ldb,
                  T3* c, size_t ldc) {
    const size_t a_i_stride = lda;
    const size_t a_l_stride = 1;
    const size_t b_j_stride = 1;
    const size_t b_l_stride = ldb;
    const size_t c_i_stride = ldc;
    const size_t c_j_stride = 1;
    size_t i, j, l;
    for (j = 0; j < n; j++) {
      for (i = 0; i < m; i++) {
        T3 total(0);
        for (l = 0; l < k; l++) {
          const size_t a_index = ((i * a_i_stride) + (l * a_l_stride));
          const T1 a_value = a[a_index];
          const size_t b_index = ((j * b_j_stride) + (l * b_l_stride));
          const T2 b_value = b[b_index];
          total += (a_value * b_value);
        }
        const size_t c_index = ((i * c_i_stride) + (j * c_j_stride));
        c[c_index] = total;
      }
    }
  }
};

// Uses the optimized EigenTensor library to implement the matrix multiplication
// required by the Im2ColConvFunctor class. We supply the two input and one
// output types so that the accumulator can potentially be higher-precision than
// the inputs, even though we don't currently take advantage of this.
template <class T1, class T2, class T3>
class FastGemmFunctor {
 public:
  void operator()(tensorflow::OpKernelContext* ctx, size_t m, size_t n,
                  size_t k, const T1* a, size_t lda, const T2* b, size_t ldb,
                  T3* c, size_t ldc) {
    typename tensorflow::TTypes<const T1>::Matrix a_matrix(a, m, k);
    typename tensorflow::TTypes<const T2>::Matrix b_matrix(b, k, n);
    typename tensorflow::TTypes<T3>::Matrix c_matrix(c, m, n);

    Eigen::array<Eigen::IndexPair<Eigen::DenseIndex>, 1> dim_pair;
    dim_pair[0].first = 1;
    dim_pair[0].second = 0;
    c_matrix.device(ctx->eigen_device<Eigen::ThreadPoolDevice>()) =
        a_matrix.contract(b_matrix, dim_pair);
  }
};

// If we have a fast CBLAS library, use its implementation through a wrapper.
#if defined(USE_CBLAS_GEMM)
template <>
class FastGemmFunctor<float, float, float> {
 public:
  void operator()(tensorflow::OpKernelContext* ctx, size_t m, size_t n,
                  size_t k, const float* a, size_t lda, const float* b,
                  size_t ldb, float* c, size_t ldc) {
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1.0f, a,
                lda, b, ldb, 0.0f, c, ldc);
  }
};
#endif  // USE_CBLAS_GEMM

#endif  // TENSORFLOW_CORE_KERNELS_GEMM_FUNCTORS_H_

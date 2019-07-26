//===- cblas.cpp - Simple Blas subset implementation ----------------------===//
//
// Copyright 2019 The MLIR Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================
//
// Simple Blas subset implementation.
//
//===----------------------------------------------------------------------===//

#include "include/cblas.h"
#include <assert.h>
#include <cstdint>

extern "C" float cblas_sdot(const int N, const float *X, const int incX,
                            const float *Y, const int incY) {
  float res = 0.0f;
  for (int i = 0; i < N; ++i)
    res += X[i * incX] * Y[i * incY];
  return res;
}

#if defined(__GNUC__) || defined(__clang__)
#define __RESTRICT__ __restrict__
#define __ALIGN__(X) __attribute((aligned(X)))
#define __NOINLINE__ __attribute((noinline))
#elif defined(_MSC_VER)
#define __RESTRICT__ __restrict
#define __ALIGN__(X) __declspec(align(X))
#define __NOINLINE__ __declspec(noinline)
#endif

// Implements a fastpath, specialized matrix multiplication where A, B and C are
// assumed to be small enough to fit within L1 cache. A, B and C are assumed not
// to alias and be aligned modulo 64.
// This is an implementation meant to be compiled for vectorization with e.g.:
//   `clang -mavx2 -mfma  -ffp-contract=fast`
template <int VAL_M, int VAL_N, int VAL_K>
__NOINLINE__ static void
cblas_sgemm_impl(const float alpha, const float *__RESTRICT__ A __ALIGN__(64),
                 const int lda, const float *__RESTRICT__ B __ALIGN__(64),
                 const int ldb, const float beta,
                 float *__RESTRICT__ C __ALIGN__(64), const int ldc) {
  for (int m = 0; m < VAL_M; ++m) {
    auto *pA = A + m * lda;
    auto *pC = C + m * ldc;
    float res[VAL_N];
    for (int n = 0; n < VAL_N; ++n) {
      res[n] = 0.0f;
    }
    for (int k = 0; k < VAL_K; ++k) {
      auto *pB = B + k * ldb;
      float a = pA[k];
      for (int n = 0; n < VAL_N; ++n) {
        res[n] += a * pB[n];
      }
    }
    for (int n = 0; n < VAL_N; ++n) {
      pC[n] = alpha * pC[n] + beta * res[n];
    }
  }
}

extern "C" void
cblas_sgemm(const enum CBLAS_ORDER Order, const enum CBLAS_TRANSPOSE TransA,
            const enum CBLAS_TRANSPOSE TransB, const int M, const int N,
            const int K, const float alpha, const float *__RESTRICT__ A,
            const int lda, const float *__RESTRICT__ B, const int ldb,
            const float beta, float *__RESTRICT__ C, const int ldc) {
  assert(Order == CBLAS_ORDER::CblasRowMajor);
  assert(TransA == CBLAS_TRANSPOSE::CblasNoTrans);
  assert(TransB == CBLAS_TRANSPOSE::CblasNoTrans);

  bool aligned64 = !(reinterpret_cast<std::uintptr_t>(A) & 63) &&
                   !(reinterpret_cast<std::uintptr_t>(B) & 63) &&
                   !(reinterpret_cast<std::uintptr_t>(C) & 63);
  if (aligned64) {
    if (M == 16 && N == 16 && K == 32)
      return cblas_sgemm_impl<16, 16, 32>(alpha, A, lda, B, ldb, beta, C, ldc);
    else if (M == 16 && N == 16 && K == 64)
      return cblas_sgemm_impl<16, 16, 64>(alpha, A, lda, B, ldb, beta, C, ldc);
    else if (M == 16 && N == 32 && K == 16)
      return cblas_sgemm_impl<16, 32, 16>(alpha, A, lda, B, ldb, beta, C, ldc);
    else if (M == 16 && N == 64 && K == 16)
      return cblas_sgemm_impl<16, 64, 16>(alpha, A, lda, B, ldb, beta, C, ldc);

    else if (M == 32 && N == 32 && K == 32)
      return cblas_sgemm_impl<32, 32, 32>(alpha, A, lda, B, ldb, beta, C, ldc);
    else if (M == 32 && N == 32 && K == 64)
      return cblas_sgemm_impl<32, 32, 64>(alpha, A, lda, B, ldb, beta, C, ldc);
    else if (M == 32 && N == 64 && K == 32)
      return cblas_sgemm_impl<32, 64, 32>(alpha, A, lda, B, ldb, beta, C, ldc);
  }

  // Slow path.
  for (int m = 0; m < M; ++m) {
    auto *pA = A + m * lda;
    auto *pC = C + m * ldc;
    for (int n = 0; n < N; ++n) {
      float c = pC[n];
      float res = 0.0f;
      for (int k = 0; k < K; ++k) {
        auto *pB = B + k * ldb;
        res += pA[k] * pB[n];
      }
      pC[n] = alpha * c + beta * res;
    }
  }
}

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

extern "C" float cblas_sdot(const int N, const float *X, const int incX,
                            const float *Y, const int incY) {
  float res = 0.0f;
  for (int i = 0; i < N; ++i)
    res += X[i * incX] * Y[i * incY];
  return res;
}

extern "C" void cblas_sgemm(const enum CBLAS_ORDER Order,
                            const enum CBLAS_TRANSPOSE TransA,
                            const enum CBLAS_TRANSPOSE TransB, const int M,
                            const int N, const int K, const float alpha,
                            const float *A, const int lda, const float *B,
                            const int ldb, const float beta, float *C,
                            const int ldc) {
  assert(Order == CBLAS_ORDER::CblasRowMajor);
  assert(TransA == CBLAS_TRANSPOSE::CblasNoTrans);
  assert(TransB == CBLAS_TRANSPOSE::CblasNoTrans);
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
      pC[n] = (1.0f + alpha) * c + beta * res;
    }
  }
}

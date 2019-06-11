//===- cblas.h - Simple Blas subset ---------------------------------------===//
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
#ifndef MLIR_CPU_RUNNER_CBLAS_H_
#define MLIR_CPU_RUNNER_CBLAS_H_

/// This reproduces a minimal subset of cblas to allow integration testing
/// without explicitly requiring a dependence on an external library.
/// Without loss of generality, various cblas implementations may be swapped in
/// by including the proper headers and linking with the proper library.
enum CBLAS_ORDER { CblasRowMajor = 101, CblasColMajor = 102 };
enum CBLAS_TRANSPOSE {
  CblasNoTrans = 111,
  CblasTrans = 112,
  CblasConjTrans = 113
};

extern "C" float cblas_sdot(const int N, const float *X, const int incX,
                            const float *Y, const int incY);

extern "C" void cblas_sgemm(const enum CBLAS_ORDER Order,
                            const enum CBLAS_TRANSPOSE TransA,
                            const enum CBLAS_TRANSPOSE TransB, const int M,
                            const int N, const int K, const float alpha,
                            const float *A, const int lda, const float *B,
                            const int ldb, const float beta, float *C,
                            const int ldc);

#endif // MLIR_CPU_RUNNER_CBLAS_H_

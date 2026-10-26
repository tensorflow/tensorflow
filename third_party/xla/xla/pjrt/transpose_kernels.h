/* Copyright 2021 The OpenXLA Authors.

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

#ifndef XLA_PJRT_TRANSPOSE_KERNELS_H_
#define XLA_PJRT_TRANSPOSE_KERNELS_H_

#include <cstdint>
#include <cstring>

#include "absl/base/optimization.h"

namespace xla {

// Maximum byte size along each dimension of a macroblock.
inline constexpr int kMaxOuterBlockSizeBytes = ABSL_CACHELINE_SIZE;

// Maximum input byte stride (lda) for which the 128-bit square microkernel
// is preferred over the 256-bit rectangular microkernel when lda >= ldb.
inline constexpr int kMaxSquare128StrideBytes = ABSL_CACHELINE_SIZE;

using TransposeMacroKernelFn = void (*)(const char* a, int64_t lda,
                                        int outer_bs_a, char* b, int64_t ldb,
                                        int outer_bs_b);

// Defined in transpose_kernels.cc (where foreach_target.h generates per-target
// kernels); 0-arg getter avoids spilling argument registers in GetChosenTarget.
template <int packed>
TransposeMacroKernelFn GetTransposeMacroKernelPacked();

template <typename T, int bs>
void TransposeMacroKernelDispatch(const char* a, int64_t lda, int outer_bs_a,
                                  char* b, int64_t ldb, int outer_bs_b) {
  constexpr int packed = sizeof(T) | (bs << 16);
  GetTransposeMacroKernelPacked<packed>()(a, lda, outer_bs_a, b, ldb,
                                          outer_bs_b);
}

template <typename T, int bs>
void TransposeMicroKernelDispatch(const char* a, int64_t lda, char* b,
                                  int64_t ldb) {
  TransposeMacroKernelDispatch<T, bs>(a, lda, /*outer_bs_a=*/1, b, ldb,
                                      /*outer_bs_b=*/1);
}

// Kernels for InnerKernelKind::kInterleave and kDeinterleave over `n` elements
// of each of the K rows. They are written as scalar loops with a compile-time
// K for clang to vectorize, e.g. with NEON structured loads/stores (ld2/ld3,
// st2/st3) for K = 2, 3 and shuffles on x86.
template <typename T, int K>
void InterleaveKernel(const char* __restrict a, int64_t lda, char* __restrict b,
                      int64_t n) {
  for (int64_t i = 0; i < n; ++i) {
    for (int k = 0; k < K; ++k) {
      std::memcpy(b + k * sizeof(T), a + k * lda + i * sizeof(T), sizeof(T));
    }
    b += K * sizeof(T);
  }
}

template <typename T, int K>
void DeinterleaveKernel(const char* __restrict a, char* __restrict b,
                        int64_t ldb, int64_t n) {
  for (int64_t i = 0; i < n; ++i) {
    for (int k = 0; k < K; ++k) {
      std::memcpy(b + k * ldb + i * sizeof(T), a + k * sizeof(T), sizeof(T));
    }
    a += K * sizeof(T);
  }
}

}  // namespace xla

#endif  // XLA_PJRT_TRANSPOSE_KERNELS_H_

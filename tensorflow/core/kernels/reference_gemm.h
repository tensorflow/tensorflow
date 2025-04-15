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

#ifndef TENSORFLOW_CORE_KERNELS_REFERENCE_GEMM_H_
#define TENSORFLOW_CORE_KERNELS_REFERENCE_GEMM_H_

#include <stdlib.h>

#include "Eigen/Core"  // from @eigen_archive
#include "tensorflow/core/platform/types.h"

// This is an unoptimized but debuggable implementation of the GEMM matrix
// multiply function, used to compare to faster but more opaque versions, or
// for bit depths or argument combinations that aren't supported by optimized
// code.
// It assumes the row-major convention used by TensorFlow, and implements
// C = A * B, like the standard BLAS GEMM interface. If the transpose flags are
// true, then the relevant matrix is treated as stored in column-major order.

namespace tensorflow {
template <class T1, class T2, class T3>
void ReferenceGemm(bool transpose_a, bool transpose_b, bool transpose_c,
                   size_t m, size_t n, size_t k, const T1* a, int32_t offset_a,
                   size_t lda, const T2* b, int32_t offset_b, size_t ldb, T3* c,
                   int32_t shift_c, int32_t offset_c, int32_t mult_c,
                   size_t ldc) {
  int a_i_stride;
  int a_l_stride;
  if (transpose_a) {
    a_i_stride = 1;
    a_l_stride = lda;
  } else {
    a_i_stride = lda;
    a_l_stride = 1;
  }
  int b_j_stride;
  int b_l_stride;
  if (transpose_b) {
    b_j_stride = ldb;
    b_l_stride = 1;
  } else {
    b_j_stride = 1;
    b_l_stride = ldb;
  }
  int c_i_stride;
  int c_j_stride;
  if (transpose_c) {
    c_i_stride = 1;
    c_j_stride = ldc;
  } else {
    c_i_stride = ldc;
    c_j_stride = 1;
  }

  const int32_t highest = static_cast<int32>(Eigen::NumTraits<T3>::highest());
  const int32_t lowest = static_cast<int32>(Eigen::NumTraits<T3>::lowest());
  const int32_t rounding = (shift_c < 1) ? 0 : (1 << (shift_c - 1));

  int i, j, l;
  for (j = 0; j < n; j++) {
    for (i = 0; i < m; i++) {
      int32_t total = 0;
      for (l = 0; l < k; l++) {
        const size_t a_index = ((i * a_i_stride) + (l * a_l_stride));
        const int32_t a_value = static_cast<int32>(a[a_index]) - offset_a;
        const size_t b_index = ((j * b_j_stride) + (l * b_l_stride));
        const int32_t b_value = static_cast<int32>(b[b_index]) - offset_b;
        total += (a_value * b_value);
      }
      const size_t c_index = ((i * c_i_stride) + (j * c_j_stride));
      int32_t output = ((((total + offset_c) * mult_c) + rounding) >> shift_c);
      if (output > highest) {
        output = highest;
      }
      if (output < lowest) {
        output = lowest;
      }
      c[c_index] = static_cast<T3>(output);
    }
  }
}
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_REFERENCE_GEMM_H_

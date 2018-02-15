/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CONTRIB_LITE_KERNELS_INTERNAL_OPTIMIZED_CBLAS_REFERENCE_H_
#define TENSORFLOW_CONTRIB_LITE_KERNELS_INTERNAL_OPTIMIZED_CBLAS_REFERENCE_H_

#include "tensorflow/contrib/lite/kernels/internal/compatibility.h"

// The reference implementation for a small subset of CBLAS interface.
// This is only used for testing CBLAS implementation, and should never be used
// in production code.

namespace tflite {
namespace cblas_ops {

// The following code follows the original CBLAS specification, and it might
// conflict with the TensorFlow naming convention.
// TODO(ycling): Find another way to test CBLAS with bazel, without writing
// a reference implementation by ourselves.
enum CBLAS_ORDER { CblasRowMajor = 0, CblasColMajor = 1 };

enum CBLAS_TRANSPOSE { CblasNoTrans = 0, CblasTrans = 1, CblasConjTrans = 2 };

// A reference implementation for matrix multiplication.
// The following code computes, c = a * transponse(b) matrix multiplication
// with CBLAS, where:
// * `a` is a matrix with dimensions (m, k).
// * `b` is a matrix with dimensions (n, k), so transpose(b) is (k, n).
// * `c` is a matrix with dimensions (m, n).
// The naming of variables is aligned with CBLAS specification here.
void cblas_sgemm(const enum CBLAS_ORDER order,
                 const enum CBLAS_TRANSPOSE trans_a,
                 const enum CBLAS_TRANSPOSE trans_b, const int m, const int n,
                 const int k, const float alpha, const float *a,
                 const int stride_a, const float *b, const int stride_b,
                 const float beta, float *c, const int stride_c) {
  TFLITE_DCHECK(order == CblasRowMajor);
  TFLITE_DCHECK(trans_a == CblasNoTrans);
  TFLITE_DCHECK(trans_b == CblasTrans);
  TFLITE_DCHECK(beta == 0.0f);
  for (int row = 0; row < m; ++row) {
    for (int col = 0; col < n; ++col) {
      // If `beta` non-zero, multiple it with the original values in output.
      // Otherwise, ignore the original value in output completely.
      float value = 0.0f;
      for (int idx = 0; idx < k; ++idx) {
        value += alpha * a[stride_a * row + idx] * b[stride_b * col + idx];
      }
      c[stride_c * row + col] = value;
    }
  }
}

}  // namespace cblas_ops
}  // namespace tflite

#endif  // TENSORFLOW_CONTRIB_LITE_KERNELS_INTERNAL_OPTIMIZED_CBLAS_REFERENCE_H_

/* Copyright 2019 Google LLC. All Rights Reserved.

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

// This is the only Ruy header that users should #include.

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_RUY_RUY_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_RUY_RUY_H_

#include "tensorflow/lite/experimental/ruy/context.h"
#include "tensorflow/lite/experimental/ruy/dispatch.h"
#include "tensorflow/lite/experimental/ruy/matrix.h"
#include "tensorflow/lite/experimental/ruy/spec.h"

namespace ruy {

// Performs a multiplication of matrices.  This is Ruy's only API entry point.
// Should be self-explanatory given the above documentation for each of Matrix,
// Spec and Context. See reference code in reference.h, with the caveat that
// that is reference code for transpose-multiply (TrMul) not just multiply;
// see the translation between the two in transpose_dispatch.h.
template <Path CompiledPaths, typename LhsScalar, typename RhsScalar,
          typename DstScalar, typename Spec>
void Mul(const Matrix<LhsScalar>& lhs, const Matrix<RhsScalar>& rhs,
         const Spec& spec, Context* context, Matrix<DstScalar>* dst) {
  DispatchMul<CompiledPaths, LhsScalar, RhsScalar, DstScalar, Spec>(
      lhs, rhs, spec, context, dst);
}

}  // namespace ruy

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_RUY_RUY_H_

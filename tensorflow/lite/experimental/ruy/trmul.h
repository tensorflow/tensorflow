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

// As a matrix multiplication library, Ruy offers a Mul entry point, performing
// matrix multiplication. For implementation purposes, it is much nicer to
// be dealing with the transpose-and-multiply operation, doing
//   Destination = Transpose(LHS) * RHS
// Indeed, the latter is performing dot-products between the *columns* of LHS
// and the columns of RHS, whereas a plain matrix multiplication is performing
// dot-products between the *rows* of LHS and the columns of RHS.
// That is why TrMul is nicer to implement, allowing for a more symmetric
// treatment of LHS and RHS.

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_RUY_TRMUL_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_RUY_TRMUL_H_

#include "tensorflow/lite/experimental/ruy/context.h"
#include "tensorflow/lite/experimental/ruy/internal_matrix.h"
#include "tensorflow/lite/experimental/ruy/kernel.h"
#include "tensorflow/lite/experimental/ruy/pack.h"
#include "tensorflow/lite/experimental/ruy/tune.h"

namespace ruy {

// Type-erased data needed for implementing TrMul.
struct TrMulParams {
  // Helper functions for invoking the function pointers.
  void LhsRunPack(Tuning tuning, int start_c, int end_c) {
    lhs_run_pack(tuning, lhs, &packed_lhs, start_c, end_c);
  }
  void RhsRunPack(Tuning tuning, int start_c, int end_c) {
    rhs_run_pack(tuning, rhs, &packed_rhs, start_c, end_c);
  }
  void RunKernel(Tuning tuning, int start_r, int start_c, int end_r,
                 int end_c) {
    run_kernel(tuning, packed_lhs, packed_rhs, spec, start_r, start_c, end_r,
               end_c, &dst);
  }

  // Function pointers to type-erased entry points for kernels and packers.
  RunPackFn* lhs_run_pack = nullptr;
  RunPackFn* rhs_run_pack = nullptr;
  RunKernelFn* run_kernel = nullptr;

  // Matrices and packed matrices.
  DMatrix lhs;
  DMatrix rhs;
  DMatrix dst;
  PMatrix packed_lhs;
  PMatrix packed_rhs;
  bool lhs_is_prepacked = false;
  bool rhs_is_prepacked = false;

  // Type-erased Spec.
  void* spec = nullptr;
};

void TrMul(TrMulParams* params, Context* context);

}  // namespace ruy

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_RUY_TRMUL_H_

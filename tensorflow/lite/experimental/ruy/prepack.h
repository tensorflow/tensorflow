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

// Implementation of low-level pre-packing API.

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_RUY_PREPACK_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_RUY_PREPACK_H_

#include <functional>

#include "tensorflow/lite/experimental/ruy/context.h"
#include "tensorflow/lite/experimental/ruy/dispatch.h"
#include "tensorflow/lite/experimental/ruy/matrix.h"
#include "tensorflow/lite/experimental/ruy/path.h"
#include "tensorflow/lite/experimental/ruy/spec.h"
#include "tensorflow/lite/experimental/ruy/tune.h"

namespace ruy {

template <Path CompiledPaths, typename LhsScalar, typename RhsScalar,
          typename DstScalar, typename Spec>
void PrePackForMulInternal(const Matrix<LhsScalar>& lhs,
                           const Matrix<RhsScalar>& rhs, const Spec& spec,
                           Context* context, Matrix<DstScalar>* dst,
                           PrepackedMatrix* prepacked_lhs,
                           PrepackedMatrix* prepacked_rhs,
                           std::function<void*(std::size_t)> alloc_fn) {
  gemmlowp::ScopedProfilingLabel label("PrePackForMul");
  Path the_path = context->GetPathToTake<CompiledPaths>();
  RUY_CHECK(the_path != Path::kReference);
  constexpr Path TrMulCompiledPaths = CompiledPaths & ~Path::kReference;
  Matrix<LhsScalar> transposed_lhs(lhs);
  Transpose(&transposed_lhs);
  TrMulParams params;
  CreateTrMulParams<TrMulCompiledPaths>(transposed_lhs, rhs, spec, context, dst,
                                        the_path, &params);

  Tuning tuning = context->GetMainThreadTuning();
  if (prepacked_lhs) {
    prepacked_lhs->data_size = DataSize(params.packed_lhs);
    prepacked_lhs->sums_size = SumsSize(params.packed_lhs);
    prepacked_lhs->data = alloc_fn(prepacked_lhs->data_size);
    prepacked_lhs->sums = alloc_fn(prepacked_lhs->sums_size);
    params.packed_lhs.data = prepacked_lhs->data;
    params.packed_lhs.sums = prepacked_lhs->sums;
    params.LhsRunPack(tuning, 0, params.packed_lhs.layout.cols);
  }
  if (prepacked_rhs) {
    prepacked_rhs->data_size = DataSize(params.packed_rhs);
    prepacked_rhs->sums_size = SumsSize(params.packed_rhs);
    prepacked_rhs->data = alloc_fn(prepacked_rhs->data_size);
    prepacked_rhs->sums = alloc_fn(prepacked_rhs->sums_size);
    params.packed_rhs.data = prepacked_rhs->data;
    params.packed_rhs.sums = prepacked_rhs->sums;
    params.RhsRunPack(tuning, 0, params.packed_rhs.layout.cols);
  }
}

template <Path CompiledPaths, typename LhsScalar, typename RhsScalar,
          typename DstScalar, typename Spec>
void MulWithPrepackedInternal(const Matrix<LhsScalar>& lhs,
                              const Matrix<RhsScalar>& rhs, const Spec& spec,
                              Context* context, Matrix<DstScalar>* dst,
                              PrepackedMatrix* prepacked_lhs,
                              PrepackedMatrix* prepacked_rhs) {
  gemmlowp::ScopedProfilingLabel label("MulWithPrepacked");

  EnforceLayoutSupport<Spec>(lhs.layout, rhs.layout, dst->layout);
  EnforceZeroPointSupport<Spec>(lhs.zero_point, rhs.zero_point,
                                dst->zero_point);

  Path the_path = context->GetPathToTake<CompiledPaths>();
  RUY_CHECK(the_path != Path::kReference);
  constexpr Path TrMulCompiledPaths = CompiledPaths & ~Path::kReference;
  Matrix<LhsScalar> transposed_lhs(lhs);
  Transpose(&transposed_lhs);
  TrMulParams params;
  CreateTrMulParams<TrMulCompiledPaths>(transposed_lhs, rhs, spec, context, dst,
                                        the_path, &params);

  if (prepacked_lhs) {
    params.packed_lhs.data = prepacked_lhs->data;
    params.packed_lhs.sums = prepacked_lhs->sums;
    params.lhs_is_prepacked = true;
  }
  if (prepacked_rhs) {
    params.packed_rhs.data = prepacked_rhs->data;
    params.packed_rhs.sums = prepacked_rhs->sums;
    params.rhs_is_prepacked = true;
  }
  TrMul(&params, context);
}

}  // namespace ruy

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_RUY_PREPACK_H_

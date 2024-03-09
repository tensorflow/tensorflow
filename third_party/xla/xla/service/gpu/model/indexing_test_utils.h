/* Copyright 2024 The OpenXLA Authors.

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

#ifndef XLA_SERVICE_GPU_MODEL_INDEXING_TEST_UTILS_H_
#define XLA_SERVICE_GPU_MODEL_INDEXING_TEST_UTILS_H_

#include <string_view>

#include <gmock/gmock.h>
#include "absl/strings/string_view.h"
#include "mlir/IR/AffineExpr.h"  // from @llvm-project
#include "mlir/IR/AffineMap.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "xla/service/gpu/model/indexing_analysis.h"
#include "xla/tests/hlo_test_base.h"

namespace xla {
namespace gpu {

// Matches two strings ignoring whitespaces.
bool ApproximateMatch(std::string_view lhs, std::string_view rhs);

MATCHER(UndefinedMap, "") { return arg.IsUndefined(); }

MATCHER_P(MatchIndexingMap, indexing_string, "") {
  if (arg.IsUndefined()) {
    return false;
  }
  return ExplainMatchResult(
      true, ApproximateMatch(indexing_string, arg.ToString()), result_listener);
}

MATCHER_P(MatchIndexingString, indexing_string, "") {
  return ExplainMatchResult(true, ApproximateMatch(indexing_string, arg),
                            result_listener);
}

HloInstructionIndexing ComputeOutputToInputIndexingForEntryComputation(
    HloTestBase* test_base, mlir::MLIRContext* mlir_context,
    absl::string_view hlo_string, int output_id = 0,
    bool use_physical_layout = false);

HloInstructionIndexing ComputeInputToOutputIndexingForEntryComputation(
    HloTestBase* test_base, mlir::MLIRContext* mlir_context,
    absl::string_view hlo_string, int input_id = 0,
    bool use_physical_layout = false);

mlir::AffineMap ParseAffineMap(absl::string_view serialized_affine_map,
                               mlir::MLIRContext* context);

mlir::AffineExpr ParseAffineExpr(absl::string_view serialized_affine_expr,
                                 mlir::MLIRContext* context);

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_MODEL_INDEXING_TEST_UTILS_H_

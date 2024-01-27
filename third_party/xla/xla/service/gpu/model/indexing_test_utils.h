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
#include "mlir/IR/AffineMap.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "xla/service/gpu/model/affine_map_printer.h"
#include "xla/service/gpu/model/indexing_analysis.h"
#include "xla/tests/hlo_test_base.h"

namespace xla {
namespace gpu {

MATCHER_P2(MatchRange, lower_bound, upper_bound,
           absl::StrCat(negation ? "equals " : "doesn't equal ", "range [",
                        lower_bound, ", ", upper_bound, "]")) {
  return ExplainMatchResult(::testing::FieldsAre(lower_bound, upper_bound), arg,
                            result_listener);
}

MATCHER_P2(MatchDomain, dim_ranges, symbol_ranges, "") {
  return ExplainMatchResult(0, arg.GetExprCount(), result_listener) &&
         ExplainMatchResult(dim_ranges, arg.GetDimensionRanges(),
                            result_listener) &&
         ExplainMatchResult(symbol_ranges, arg.GetSymbolRanges(),
                            result_listener);
}

MATCHER(IsEmptyDomain, "") {
  return ExplainMatchResult(true, arg.IsKnownEmpty(), result_listener);
}

MATCHER_P3(MatchDomainWithGenericConstraints, dim_ranges, symbol_ranges,
           expr_ranges, "") {
  return ExplainMatchResult(dim_ranges, arg.GetDimensionRanges(),
                            result_listener) &&
         ExplainMatchResult(symbol_ranges, arg.GetSymbolRanges(),
                            result_listener) &&
         ExplainMatchResult(expr_ranges, arg.GetExprRanges(), result_listener);
}

MATCHER_P2(MatchExprRange, affine_map_string, range, "") {
  return ExplainMatchResult(::testing::HasSubstr(affine_map_string),
                            AffineMapPrinter().ToString(arg.first),
                            result_listener) &&
         ExplainMatchResult(range, arg.second, result_listener);
}

MATCHER_P2(MatchIndexingMap, affine_map_string, domain, "") {
  if (!arg.has_value()) {
    return false;
  }
  return ExplainMatchResult(::testing::HasSubstr(affine_map_string),
                            AffineMapPrinter().ToString(arg->affine_map),
                            result_listener) &&
         ExplainMatchResult(domain, arg->domain, result_listener);
}

// Matches two strings ignoring whitespaces.
bool ApproximateMatch(std::string_view lhs, std::string_view rhs);

MATCHER_P(MatchIndexingString, indexing_string, "") {
  if (!arg.has_value()) {
    return false;
  }
  return ExplainMatchResult(true,
                            ApproximateMatch(indexing_string, arg->ToString()),
                            result_listener);
}

HloInstructionIndexing ComputeOutputToInputIndexingForEntryComputation(
    HloTestBase* test_base, mlir::MLIRContext* mlir_context,
    absl::string_view hlo_string, int output_id = 0);

HloInstructionIndexing ComputeInputToOutputIndexingForEntryComputation(
    HloTestBase* test_base, mlir::MLIRContext* mlir_context,
    absl::string_view hlo_string, int input_id = 0);

mlir::AffineMap ParseAffineMap(absl::string_view serialized_affine_map,
                               mlir::MLIRContext* context);

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_MODEL_INDEXING_TEST_UTILS_H_

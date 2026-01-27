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

#ifndef XLA_HLO_ANALYSIS_INDEXING_TEST_UTILS_H_
#define XLA_HLO_ANALYSIS_INDEXING_TEST_UTILS_H_

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <vector>

#include <gmock/gmock.h>
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/MLIRContext.h"
#include "xla/hlo/analysis/indexing_analysis.h"
#include "xla/hlo/analysis/indexing_map.h"
#include "xla/hlo/analysis/indexing_map_serialization.h"
#include "xla/hlo/analysis/symbolic_expr.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/testlib/verified_hlo_module.h"

namespace xla {

// Matches two strings ignoring whitespaces.
bool ApproximateMatch(absl::string_view lhs, absl::string_view rhs);

MATCHER(UndefinedMap, "") { return arg.IsUndefined(); }

MATCHER(UndefinedOperandIndexing, "") { return arg.IsUndefined(); }

MATCHER_P(MatchIndexingMap, indexing_string, "") {
  if (arg.IsUndefined()) {
    return false;
  }
  return ExplainMatchResult(
      true, ApproximateMatch(indexing_string, ToString(arg)), result_listener);
}

MATCHER_P(MatchOperandIndexing, indexing_string, "") {
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

class IndexingTestBase : public HloHardwareIndependentTestBase {
 public:
  IndexingTestBase() { RegisterSymbolicExprStorage(&mlir_context_); }
  HloInstruction* ParseAndGetRoot(absl::string_view hlo_string);

  virtual HloInstructionIndexing GetOutputToInputIndexing(
      const HloInstruction* instr, int output_id, bool use_physical_layout);
  HloInstructionIndexing GetOutputToInputIndexing(const HloInstruction* instr,
                                                  int output_id = 0) {
    return GetOutputToInputIndexing(instr, output_id, false);
  }

  virtual HloInstructionIndexing GetInputToOutputIndexing(
      const HloInstruction* instr, int input_id, bool use_physical_layout);
  HloInstructionIndexing GetInputToOutputIndexing(const HloInstruction* instr,
                                                  int input_id = 0) {
    return GetInputToOutputIndexing(instr, input_id, false);
  }

  mlir::MLIRContext mlir_context_;
  std::unique_ptr<VerifiedHloModule> module_;
};

mlir::AffineMap ParseAffineMap(absl::string_view serialized_affine_map,
                               mlir::MLIRContext* mlir_context);

mlir::AffineExpr ParseAffineExpr(absl::string_view serialized_affine_expr,
                                 mlir::MLIRContext* mlir_context);

// Safely evaluates the given expression, returning nullopt if the result is
// undefined (due to undefined behavior, e.g. division by zero or overflow).
std::optional<int64_t> SafeEvaluateAffineExpr(mlir::AffineExpr expr,
                                              absl::Span<int64_t const> dims,
                                              absl::Span<int64_t const> syms);

// Enumerates all the points in the domain of the given indexing map: points
// within the bounds of the dimensions and symbols that do not violate any of
// the constraints.
absl::Status EnumerateDomain(
    const IndexingMap& indexing_map,
    const std::function<absl::Status(absl::Span<int64_t const> dims,
                                     absl::Span<int64_t const> syms)>&
        callback);

// Checks if the indexing map is a bijection: verifies that each point in the
// expected codomain is mapped to a unique point in the domain.
// The codomain is the output of the indexing map. For example, for an
// input->output map for an instruction, it would be the instruction's output
// shape.
absl::Status VerifyBijection(const IndexingMap& indexing_map,
                             absl::Span<Interval const> expected_codomain);

// Checks that two affine expressions map to the same values for all points in
// their domain. If `reference` is undefined at a point, the value of `other` is
// ignored. If `other` is undefined at a point, but `reference` is not, this is
// a failure.
absl::Status VerifyExprsAreIdentical(
    mlir::AffineExpr reference, mlir::AffineExpr other,
    absl::Span<Interval const> dimension_ranges,
    absl::Span<Interval const> symbol_ranges);

// Returns the trip counts for each symbol in the indexing map.
std::vector<int64_t> GetLoopTripCounts(const IndexingMap& indexing_map);

}  // namespace xla

#endif  // XLA_HLO_ANALYSIS_INDEXING_TEST_UTILS_H_

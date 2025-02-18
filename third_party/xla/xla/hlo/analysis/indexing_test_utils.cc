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

#include "xla/hlo/analysis/indexing_test_utils.h"

#include <cctype>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/MathExtras.h"
#include "mlir/AsmParser/AsmParser.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Support/LLVM.h"
#include "xla/hlo/analysis/indexing_analysis.h"
#include "xla/hlo/analysis/indexing_map.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/status_macros.h"
#include "tsl/platform/errors.h"

namespace xla {
namespace {

using ::mlir::AffineExpr;
using ::mlir::AffineMap;
using ::mlir::MLIRContext;

std::string FormatDimsAndSyms(absl::Span<int64_t const> dims,
                              absl::Span<int64_t const> syms) {
  return absl::StrCat("(", absl::StrJoin(dims, ", "), ")[",
                      absl::StrJoin(syms, ", "), "]");
}

}  // namespace

HloInstruction* IndexingTestBase::ParseAndGetRoot(
    absl::string_view hlo_string) {
  auto module_or = ParseAndReturnVerifiedModule(hlo_string);
  CHECK_OK(module_or);
  module_ = std::move(module_or.value());
  return module_->entry_computation()->root_instruction();
}

HloInstructionIndexing IndexingTestBase::GetOutputToInputIndexing(
    const HloInstruction* instr, int output_id, bool use_physical_layout) {
  HloInstructionIndexing indexing =
      ComputeOutputToInputIndexing(instr, output_id, &mlir_context_);

  if (!use_physical_layout) return indexing;

  IndexingMap output_permutation = GetIndexingMapFromPhysicalLayoutToLogical(
      GetOutputShape(instr, output_id), &mlir_context_);

  for (const auto& [operand_id, indexing_maps] :
       llvm::enumerate(indexing.indexing_maps)) {
    IndexingMap operand_permutation = GetIndexingMapFromLogicalToPhysicalLayout(
        instr->operand(operand_id)->shape(), &mlir_context_);

    absl::flat_hash_set<IndexingMap> operand_indexing_maps;
    for (const IndexingMap& indexing_map : indexing_maps) {
      auto normalized_indexing_map = indexing_map;
      if (!output_permutation.GetAffineMap().isIdentity()) {
        normalized_indexing_map =
            ComposeIndexingMaps(output_permutation, normalized_indexing_map);
      }
      if (!operand_permutation.GetAffineMap().isIdentity()) {
        normalized_indexing_map =
            ComposeIndexingMaps(normalized_indexing_map, operand_permutation);
      }
      operand_indexing_maps.insert(normalized_indexing_map);
    }
    indexing.indexing_maps[operand_id] = operand_indexing_maps;
  }
  return indexing;
}

HloInstructionIndexing IndexingTestBase::GetInputToOutputIndexing(
    const HloInstruction* instr, int input_id, bool use_physical_layout) {
  HloInstructionIndexing indexing =
      ComputeInputToOutputIndexing(instr, input_id, &mlir_context_);

  if (!use_physical_layout) return indexing;

  IndexingMap input_permutation = GetIndexingMapFromPhysicalLayoutToLogical(
      instr->operand(input_id)->shape(), &mlir_context_);

  for (const auto& [output_id, indexing_maps] :
       llvm::enumerate(indexing.indexing_maps)) {
    IndexingMap operand_permutation = GetIndexingMapFromLogicalToPhysicalLayout(
        GetOutputShape(instr, output_id), &mlir_context_);

    absl::flat_hash_set<IndexingMap> operand_indexing_maps;
    for (const IndexingMap& indexing_map : indexing_maps) {
      auto normalized_indexing_map = indexing_map;
      if (!input_permutation.GetAffineMap().isIdentity()) {
        normalized_indexing_map =
            ComposeIndexingMaps(input_permutation, normalized_indexing_map);
      }
      if (!operand_permutation.GetAffineMap().isIdentity()) {
        normalized_indexing_map =
            ComposeIndexingMaps(normalized_indexing_map, operand_permutation);
      }
      operand_indexing_maps.insert(normalized_indexing_map);
    }
    indexing.indexing_maps[output_id] = operand_indexing_maps;
  }
  return indexing;
}

AffineMap ParseAffineMap(absl::string_view serialized_affine_map,
                         MLIRContext* context) {
  std::string full_affine_map_string =
      absl::StrCat("affine_map<", serialized_affine_map, ">");
  return mlir::cast<mlir::AffineMapAttr>(
             mlir::parseAttribute(full_affine_map_string, context))
      .getValue();
}

// Since MLIR does not have AffineExprAttr, we construct an AffineMap and then
// retrieve its first result.
AffineExpr ParseAffineExpr(absl::string_view serialized_affine_expr,
                           MLIRContext* context) {
  std::string full_affine_map_string = absl::StrCat(
      "affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8, d9)"
      "[s0, s1, s2, s3, s4, s5, s6, s7, s8, s9] -> (",
      serialized_affine_expr, ")>");
  return mlir::cast<mlir::AffineMapAttr>(
             mlir::parseAttribute(full_affine_map_string, context))
      .getValue()
      .getResult(0);
}

bool ApproximateMatch(absl::string_view lhs, absl::string_view rhs) {
  size_t lhs_length = lhs.size();
  size_t rhs_length = rhs.size();
  size_t l = 0, r = 0;
  while (l < lhs_length || r < rhs_length) {
    while (l < lhs_length && std::isspace(lhs[l])) {
      ++l;
    }
    while (r < rhs_length && std::isspace(rhs[r])) {
      ++r;
    }
    if (l == lhs_length || r == rhs_length) {
      break;
    }
    if (lhs[l++] != rhs[r++]) {
      return false;
    }
  }
  return l == lhs_length && r == rhs_length;
}

std::optional<int64_t> SafeEvaluateAffineExpr(mlir::AffineExpr expr,
                                              absl::Span<int64_t const> dims,
                                              absl::Span<int64_t const> syms) {
  if (auto sym = mlir::dyn_cast<mlir::AffineSymbolExpr>(expr)) {
    if (sym.getPosition() < 0 || sym.getPosition() >= syms.size()) {
      return std::nullopt;
    }
    return syms[sym.getPosition()];
  }
  if (auto dim = mlir::dyn_cast<mlir::AffineDimExpr>(expr)) {
    if (dim.getPosition() < 0 || dim.getPosition() >= dims.size()) {
      return std::nullopt;
    }
    return dims[dim.getPosition()];
  }
  if (auto cst = mlir::dyn_cast<mlir::AffineConstantExpr>(expr)) {
    return cst.getValue();
  }
  auto binary = mlir::cast<mlir::AffineBinaryOpExpr>(expr);
  auto lhs = SafeEvaluateAffineExpr(binary.getLHS(), dims, syms);
  auto rhs = SafeEvaluateAffineExpr(binary.getRHS(), dims, syms);
  if (!lhs || !rhs) return std::nullopt;

  int64_t result;
  bool result_division_is_undefined =
      rhs == 0 || (lhs == std::numeric_limits<int64_t>::min() && rhs == -1);
  switch (binary.getKind()) {
    case mlir::AffineExprKind::Add:
      if (llvm::AddOverflow(*lhs, *rhs, result)) {
        return std::nullopt;
      }
      return result;
    case mlir::AffineExprKind::Mul:
      if (llvm::MulOverflow(*lhs, *rhs, result)) {
        return std::nullopt;
      }
      return result;
    case mlir::AffineExprKind::FloorDiv:
      return result_division_is_undefined
                 ? std::nullopt
                 : std::make_optional(llvm::divideFloorSigned(*lhs, *rhs));
    case mlir::AffineExprKind::CeilDiv:
      return result_division_is_undefined
                 ? std::nullopt
                 : std::make_optional(llvm::divideCeilSigned(*lhs, *rhs));
    case mlir::AffineExprKind::Mod:
      return rhs <= 0 ? std::nullopt
                      : std::make_optional(llvm::mod(*lhs, *rhs));
    default:
      LOG(FATAL) << "Unknown binary op: " << static_cast<int>(binary.getKind());
  }
}

absl::Status EnumerateDomain(
    const IndexingMap& indexing_map,
    const std::function<absl::Status(absl::Span<int64_t const> dims,
                                     absl::Span<int64_t const> syms)>&
        callback) {
  std::vector<int64_t> dims(indexing_map.GetDimensionCount());
  std::vector<int64_t> syms(indexing_map.GetSymbolCount());
  std::function<absl::Status(int64_t dim, int64_t sym)> enumerate;

  absl::Status status = absl::OkStatus();
  auto enumerate_range = [&](int64_t next_dim, int64_t next_sym, Interval range,
                             int64_t& induction_var) -> absl::Status {
    for (int64_t i = range.lower; i <= range.upper; ++i) {
      induction_var = i;
      TF_RETURN_IF_ERROR(enumerate(next_dim, next_sym));
    }
    return absl::OkStatus();
  };

  enumerate = [&](int64_t dim_id, int64_t sym_id) -> absl::Status {
    if (dim_id < dims.size()) {
      return enumerate_range(dim_id + 1, sym_id,
                             indexing_map.GetDimensionBound(dim_id),
                             dims[dim_id]);
    }

    if (sym_id < syms.size()) {
      return enumerate_range(dim_id, sym_id + 1,
                             indexing_map.GetSymbolBound(sym_id), syms[sym_id]);
    }

    for (auto [expr, interval] : indexing_map.GetConstraints()) {
      auto constraint_value = SafeEvaluateAffineExpr(expr, dims, syms);
      TF_RET_CHECK(constraint_value.has_value())
          << "Constraint evaluation triggered undefined behavior at "
          << FormatDimsAndSyms(dims, syms);
      if (!interval.Contains(*constraint_value)) return absl::OkStatus();
    }

    return callback(dims, syms);
  };

  return enumerate(0, 0);
}

absl::Status VerifyBijection(const IndexingMap& indexing_map,
                             absl::Span<Interval const> expected_codomain) {
  mlir::AffineMap affine_map = indexing_map.GetAffineMap();
  absl::flat_hash_map<absl::InlinedVector<int64_t, 4>,
                      std::pair<absl::InlinedVector<int64_t, 6>,
                                absl::InlinedVector<int64_t, 3>>>
      codomain_to_domain;
  TF_RETURN_IF_ERROR(EnumerateDomain(
      indexing_map,
      [&](absl::Span<int64_t const> dims,
          absl::Span<int64_t const> syms) -> absl::Status {
        absl::InlinedVector<int64_t, 4> codomain_point;
        for (auto result : affine_map.getResults()) {
          auto value = SafeEvaluateAffineExpr(result, dims, syms);
          TF_RET_CHECK(value.has_value())
              << "Indexing map evaluation triggered undefined behavior at "
              << FormatDimsAndSyms(dims, syms);
          codomain_point.push_back(*value);
        }

        for (auto [coordinate, interval] :
             llvm::zip(codomain_point, expected_codomain)) {
          TF_RET_CHECK(interval.Contains(coordinate))
              << "Indexing map maps " << FormatDimsAndSyms(dims, syms)
              << " to [" << absl::StrJoin(codomain_point, ", ")
              << "], which lies outside the expected codomain.";
        }

        auto& entry = codomain_to_domain[codomain_point];
        TF_RET_CHECK(entry.first.empty() && entry.second.empty())
            << "Indexing map is not a bijection. Domain points "
            << FormatDimsAndSyms(entry.first, entry.second) << " and "
            << FormatDimsAndSyms(dims, syms) << " map to the same point ["
            << absl::StrJoin(codomain_point, ", ") << "].";

        entry = {{dims.begin(), dims.end()}, {syms.begin(), syms.end()}};
        return absl::OkStatus();
      }));

  int64_t num_expected_points = 1;
  for (auto interval : expected_codomain) {
    num_expected_points *= interval.GetLoopTripCount();
  }

  TF_RET_CHECK(codomain_to_domain.size() == num_expected_points)
      << "Indexing map codomain has " << codomain_to_domain.size()
      << " points, expected " << num_expected_points;

  return absl::OkStatus();
}

std::vector<int64_t> GetLoopTripCounts(const IndexingMap& indexing_map) {
  std::vector<int64_t> trip_counts;
  trip_counts.reserve(indexing_map.GetSymbolCount());
  for (int i = 0; i < indexing_map.GetSymbolCount(); ++i) {
    trip_counts.push_back(indexing_map.GetSymbolBound(i).GetLoopTripCount());
  }
  return trip_counts;
}

absl::Status VerifyExprsAreIdentical(
    mlir::AffineExpr reference, mlir::AffineExpr other,
    absl::Span<Interval const> dimension_ranges,
    absl::Span<Interval const> symbol_ranges) {
  std::vector<IndexingMap::Variable> dims;
  dims.reserve(dimension_ranges.size());
  for (const auto& interval : dimension_ranges) {
    dims.push_back(IndexingMap::Variable{interval});
  }

  std::vector<IndexingMap::Variable> symbols;
  symbols.reserve(symbol_ranges.size());
  for (const auto& interval : symbol_ranges) {
    symbols.push_back(IndexingMap::Variable{interval});
  }

  IndexingMap map(mlir::AffineMap::get(dimension_ranges.size(),
                                       symbol_ranges.size(), reference),
                  dims, symbols, {});
  return EnumerateDomain(
      map,
      [&](absl::Span<int64_t const> dims,
          absl::Span<int64_t const> syms) -> absl::Status {
        auto reference_value = SafeEvaluateAffineExpr(reference, dims, syms);
        // If the reference value is undefined, there is no meaningful way to
        // compare it to the other value.
        if (!reference_value.has_value()) {
          return absl::OkStatus();
        }
        auto other_value = SafeEvaluateAffineExpr(other, dims, syms);
        TF_RET_CHECK(other_value.has_value())
            << "Domain point " << FormatDimsAndSyms(dims, syms)
            << " triggers undefined behavior in `other`.";

        TF_RET_CHECK(reference_value == other_value)
            << "Domain point " << FormatDimsAndSyms(dims, syms)
            << " maps to different values: " << *reference_value << " vs. "
            << *other_value << ".";
        return absl::OkStatus();
      });
}

}  // namespace xla

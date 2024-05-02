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

#include "xla/service/gpu/model/indexing_test_utils.h"

#include <cctype>
#include <cstddef>
#include <string>

#include <gtest/gtest.h>
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/AsmParser/AsmParser.h"  // from @llvm-project
#include "mlir/IR/AffineExpr.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/gpu/model/indexing_analysis.h"
#include "xla/service/gpu/model/indexing_map.h"
#include "xla/tests/hlo_test_base.h"

namespace xla {
namespace gpu {

using ::mlir::AffineExpr;
using ::mlir::AffineMap;
using ::mlir::MLIRContext;

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

bool ApproximateMatch(std::string_view lhs, std::string_view rhs) {
  size_t lhs_length = lhs.size();
  size_t rhs_length = rhs.size();
  size_t l = 0, r = 0;
  while (l < lhs_length && r < rhs_length) {
    while (l < lhs_length && std::isspace(lhs[l])) {
      ++l;
    }
    while (r < rhs_length && std::isspace(rhs[r])) {
      ++r;
    }
    if (l == lhs_length || r == rhs_length) {
      continue;
    }
    if (lhs[l++] != rhs[r++]) {
      return false;
    }
  }
  return l == lhs_length && r == rhs_length;
}

}  // namespace gpu
}  // namespace xla

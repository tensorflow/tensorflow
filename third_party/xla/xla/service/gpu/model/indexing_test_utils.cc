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

inline std::vector<std::string> split_string(std::string s,
                                             std::string pattern) {
  std::vector<std::string> result;
  size_t pos = 0;
  while ((pos = s.find(pattern)) != std::string::npos) {
    result.push_back(s.substr(0, pos));
    s.erase(0, pos + pattern.length());
  }
  if (!s.empty()) result.push_back(s);
  return result;
}

inline bool startswith(const std::string& s, const std::string& pattern) {
  return s.substr(0, pattern.size()) == pattern;
}

bool ApproximateMatch(std::string_view lhs, std::string_view rhs) {
  std::string lhs_unspaced, rhs_unspaced;
  for (auto c : lhs) {
    if (!std::isspace(c)) {
      lhs_unspaced += c;
    }
  }
  for (auto c : rhs) {
    if (!std::isspace(c)) {
      rhs_unspaced += c;
    }
  }

  if (lhs_unspaced.find("###") == std::string::npos)
    return lhs_unspaced == rhs_unspaced;

  std::vector<std::string> frags = split_string(lhs_unspaced, "###");

  while (frags.size() >= 2) {
    if (!startswith(rhs_unspaced, frags[0])) {
      return false;
    }

    rhs_unspaced = rhs_unspaced.substr(frags[0].size());

    auto terms = split_string(frags[1], "+");
    // iterate through permutations of terms
    std::vector<int> indexes(terms.size());
    for (auto i = 0; i < terms.size(); i++) {
      indexes[i] = i;
    }
    bool match = false;
    do {
      std::string permuted = "";
      for (auto i : indexes) {
        permuted += terms[i] + "+";
      }
      permuted.pop_back();
      if (startswith(rhs_unspaced, permuted)) {
        match = true;
        break;
      }
    } while (std::next_permutation(indexes.begin(), indexes.end()));

    if (!match) {
      return false;
    }

    rhs_unspaced = rhs_unspaced.substr(frags[1].size());
    frags.erase(frags.begin());
    frags.erase(frags.begin());
  }
  if (frags.empty())
    return rhs_unspaced.empty();
  else
    return rhs_unspaced == frags[0];
}

}  // namespace gpu
}  // namespace xla

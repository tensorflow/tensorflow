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
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "mlir/AsmParser/AsmParser.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/gpu/model/indexing_analysis.h"
#include "xla/tests/hlo_test_base.h"

namespace xla {
namespace gpu {

using ::mlir::AffineMap;
using ::mlir::AffineMapAttr;

HloInstructionIndexing ComputeOutputToInputIndexingForEntryComputation(
    HloTestBase* test_base, mlir::MLIRContext* mlir_context,
    absl::string_view hlo_string, int output_id) {
  auto module = test_base->ParseAndReturnVerifiedModule(hlo_string);
  EXPECT_TRUE(module.ok());

  HloInstruction* root =
      module.value()->entry_computation()->root_instruction();

  // If there are multiple instructions, they need to be wrapped in a fusion.
  for (auto* operand : root->operands()) {
    if (operand->opcode() != HloOpcode::kParameter &&
        operand->opcode() != HloOpcode::kConstant) {
      return {};
    }
  }
  return ComputeOutputToInputIndexing(root, output_id, mlir_context);
}

HloInstructionIndexing ComputeInputToOutputIndexingForEntryComputation(
    HloTestBase* test_base, mlir::MLIRContext* mlir_context,
    absl::string_view hlo_string, int input_id) {
  auto module = test_base->ParseAndReturnVerifiedModule(hlo_string);
  EXPECT_TRUE(module.ok());

  HloInstruction* root =
      module.value()->entry_computation()->root_instruction();

  // If there are multiple instructions, they need to be wrapped in a fusion.
  for (auto* operand : root->operands()) {
    if (operand->opcode() != HloOpcode::kParameter &&
        operand->opcode() != HloOpcode::kConstant) {
      return {};
    }
  }
  return ComputeInputToOutputIndexing(root, input_id, mlir_context);
}

AffineMap ParseAffineMap(absl::string_view serialized_affine_map,
                         mlir::MLIRContext* context) {
  std::string full_affine_map_string =
      absl::StrCat("affine_map<", serialized_affine_map, ">");
  return mlir::parseAttribute(full_affine_map_string, context)
      .cast<AffineMapAttr>()
      .getValue();
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

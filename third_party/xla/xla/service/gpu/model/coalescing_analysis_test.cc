/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include "xla/service/gpu/model/coalescing_analysis.h"

#include <vector>

#include <gtest/gtest.h>
#include "absl/log/check.h"
#include "absl/strings/string_view.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/gpu/hlo_traversal.h"
#include "xla/service/gpu/model/indexing_analysis.h"
#include "xla/tests/hlo_test_base.h"
#include "tsl/platform/statusor.h"
#include "tsl/platform/test.h"

namespace xla {
namespace gpu {
namespace {

using ::testing::ElementsAreArray;

class CoalescingTest : public HloTestBase {
 public:
  void GetRoot(absl::string_view hlo_string,
               absl::Span<const bool> expected_results) {
    TF_ASSERT_OK_AND_ASSIGN(auto module,
                            ParseAndReturnVerifiedModule(hlo_string));
    HloInstruction* root = module->entry_computation()->root_instruction();

    for (auto* operand : root->operands()) {
      CHECK(operand->opcode() == HloOpcode::kParameter ||
            operand->opcode() == HloOpcode::kConstant)
          << "If there are multiple instructions, they need to be wrapped in a "
             "fusion.";
    }
    auto fusion_adaptor = HloFusionAdaptor::ForInstruction(root);
    TF_ASSERT_OK_AND_ASSIGN(
        auto grouped_indexing_maps,
        ComputeGroupedOutputToInputIndexing(*fusion_adaptor, /*output_id=*/0,
                                            &mlir_context_));

    std::vector<bool> actual_results;
    actual_results.reserve(expected_results.size());
    for (auto [operand_id, is_coalesced] : llvm::enumerate(expected_results)) {
      auto* operand = root->operand(operand_id);
      actual_results.push_back(IsReadCoalesced(
          operand, root, grouped_indexing_maps, &mlir_context_));
    }
    EXPECT_THAT(actual_results, ElementsAreArray(expected_results));
  }

  mlir::MLIRContext mlir_context_;
};

TEST_F(CoalescingTest, IdentityLayout) {
  GetRoot(R"(
    HloModule m
    ENTRY e {
      p0 = f32[10, 20] parameter(0)
      p1 = f32[10, 20] parameter(1)
      ROOT add0 = f32[10, 20] add(p0, p1)
    }
  )",
          {true, true});
}

TEST_F(CoalescingTest, RhsTransposedLayout) {
  GetRoot(R"(
    HloModule m
    ENTRY e {
      p0 = f32[10, 20]{1, 0} parameter(0)
      p1 = f32[10, 20]{0, 1} parameter(1)
      ROOT exp = f32[10, 20]{1, 0} add(p0, p1)
    }
  )",
          {true, false});
}

TEST_F(CoalescingTest, OutputTransposedLayout) {
  GetRoot(R"(
    HloModule m
    ENTRY e {
      p0 = f32[10, 20]{1, 0} parameter(0)
      p1 = f32[10, 20]{1, 0} parameter(1)
      ROOT exp = f32[10, 20]{0, 1} add(p0, p1)
    }
  )",
          {false, false});
}

TEST_F(CoalescingTest, OutputAndLhsTransposedLayout) {
  GetRoot(R"(
    HloModule m
    ENTRY e {
      p0 = f32[10, 20]{1, 0} parameter(0)
      p1 = f32[10, 20]{0, 1} parameter(1)
      ROOT exp = f32[10, 20]{1, 0} add(p0, p1)
    }
  )",
          {true, false});
}

}  // namespace
}  // namespace gpu
}  // namespace xla

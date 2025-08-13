/* Copyright 2025 The OpenXLA Authors.

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

#include "xla/hlo/transforms/simplifiers/slice_mover.h"

#include <memory>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/log/log.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/testlib/pattern_matcher_gmock.h"
#include "xla/service/pattern_matcher.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {
namespace {

namespace m = xla::match;

class SliceMoverTest : public HloHardwareIndependentTestBase {
 public:
  SliceMoverTest()
      : HloHardwareIndependentTestBase(
            /*verifier_layout_sensitive=*/false,
            /*allow_mixed_precision_in_hlo_verifier=*/false) {};
};
// TODO(ramzym): Assert that the slice attribute is as expected.

TEST_F(SliceMoverTest, MoveSliceThroughAdd) {
  absl::string_view module_str = R"(
    HloModule module
    ENTRY main {
      add_op = f32[8,9] add(f32[8,9] parameter(0), f32[8,9] parameter(1))
      ROOT slice_op = f32[2,9] slice(f32[8,9] add_op), slice={[0:2], [0:9]}
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(module_str));

  SliceMover slice_mover;
  TF_ASSERT_OK_AND_ASSIGN(bool changed, RunHloPass(&slice_mover, module.get()));

  SCOPED_TRACE(module->ToString());
  EXPECT_TRUE(changed);
  HloInstruction* root_instruction =
      module->entry_computation()->root_instruction();
  LOG(INFO) << "Module after SliceMover:\n" << module->ToString();
  EXPECT_THAT(root_instruction, GmockMatch(m::Add(m::Slice(m::Parameter(0)),
                                                  m::Slice(m::Parameter(1)))));
}

TEST_F(SliceMoverTest, MoveSliceThroughMultipleAdds) {
  absl::string_view module_str = R"(
    HloModule module
    ENTRY main {
      param_0 = f32[8,9] parameter(0)
      param_1 = f32[8,9] parameter(1)
      add_op_1 = f32[8,9] add(f32[8,9] param_0, f32[8,9] param_1)
      add_op_2 = f32[8,9] add(f32[8,9] add_op_1, f32[8,9] param_1)
      ROOT slice_op = f32[2,9] slice(f32[8,9] add_op_2), slice={[0:2], [0:9]}
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(module_str));

  SliceMover slice_mover;
  TF_ASSERT_OK_AND_ASSIGN(bool changed, RunHloPass(&slice_mover, module.get()));

  SCOPED_TRACE(module->ToString());
  EXPECT_TRUE(changed);
  HloInstruction* root_instruction =
      module->entry_computation()->root_instruction();
  LOG(INFO) << "Module after SliceMover:\n" << module->ToString();
  // Param 1 is only sliced once.
  EXPECT_THAT(root_instruction,
              GmockMatch(m::Add(
                  m::Add(m::Slice(m::Parameter(0)), m::Slice(m::Parameter(1))),
                  m::Slice(m::Parameter(1)))));
}

TEST_F(SliceMoverTest, RemoveDuplicateSlices) {
  absl::string_view module_str = R"(
    HloModule module
    ENTRY main {
      param_0 = f32[8,9] parameter(0)
      slice_1 = f32[2,9] slice(f32[8,9] param_0), slice={[0:2], [0:9]}
      slice_2 = f32[2,9] slice(f32[8,9] param_0), slice={[0:2], [0:9]}
      ROOT add = f32[2,9] add(slice_1, slice_2)
    }
  )";
  // Re-use the first slice in place of the second slice as they are identical.
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(module_str));

  SliceMover slice_mover;
  TF_ASSERT_OK_AND_ASSIGN(bool changed, RunHloPass(&slice_mover, module.get()));

  SCOPED_TRACE(module->ToString());
  EXPECT_TRUE(changed);
  HloInstruction* root_instruction =
      module->entry_computation()->root_instruction();
  LOG(INFO) << "Module after SliceMover:\n" << module->ToString();
  EXPECT_EQ(root_instruction->operand(0), root_instruction->operand(1));
  EXPECT_THAT(root_instruction->operand(0),
              GmockMatch(m::Slice(m::Parameter(0))));
}

}  // anonymous namespace
}  // namespace xla

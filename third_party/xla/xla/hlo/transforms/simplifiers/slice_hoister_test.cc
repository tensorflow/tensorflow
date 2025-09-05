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

#include "xla/hlo/transforms/simplifiers/slice_hoister.h"

#include <memory>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/testlib/pattern_matcher_gmock.h"
#include "xla/service/hlo_cse.h"
#include "xla/service/pattern_matcher.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {
namespace {

namespace m = xla::match;
using ::testing::ElementsAre;

class SliceHoisterTest : public HloHardwareIndependentTestBase {
 public:
  SliceHoisterTest()
      : HloHardwareIndependentTestBase(
            /*verifier_layout_sensitive=*/false,
            /*allow_mixed_precision_in_hlo_verifier=*/true) {};
};

TEST_F(SliceHoisterTest, HoistSliceThroughAdd) {
  absl::string_view module_str = R"(
    HloModule module
    ENTRY main {
      add_op = f32[8,9] add(f32[8,9] parameter(0), f32[8,9] parameter(1))
      ROOT slice_op = f32[2,9] slice(f32[8,9] add_op), slice={[0:2], [0:9]}
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(module_str));

  SliceHoister slice_hoister;
  TF_ASSERT_OK_AND_ASSIGN(bool changed,
                          RunHloPass(&slice_hoister, module.get()));

  SCOPED_TRACE(module->ToString());
  EXPECT_TRUE(changed);
  HloInstruction* root_instruction =
      module->entry_computation()->root_instruction();
  const HloInstruction* param_0_slice = nullptr;
  const HloInstruction* param_1_slice = nullptr;
  EXPECT_THAT(root_instruction,
              GmockMatch(m::Add(m::Slice(&param_0_slice, m::Parameter(0)),
                                m::Slice(&param_1_slice, m::Parameter(1)))));
  EXPECT_THAT(param_0_slice->slice_starts(), ElementsAre(0, 0));
  EXPECT_THAT(param_0_slice->slice_limits(), ElementsAre(2, 9));
  EXPECT_THAT(param_0_slice->slice_strides(), ElementsAre(1, 1));
  EXPECT_THAT(param_1_slice->slice_starts(), ElementsAre(0, 0));
  EXPECT_THAT(param_1_slice->slice_limits(), ElementsAre(2, 9));
  EXPECT_THAT(param_1_slice->slice_strides(), ElementsAre(1, 1));
}

TEST_F(SliceHoisterTest, HoistSliceThroughMultipleAdds) {
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

  SliceHoister slice_hoister;
  TF_ASSERT_OK_AND_ASSIGN(bool changed,
                          RunHloPass(&slice_hoister, module.get()));

  SCOPED_TRACE(module->ToString());
  EXPECT_TRUE(changed);

  HloCSE cse = HloCSE(false);
  TF_ASSERT_OK_AND_ASSIGN(changed, RunHloPass(&cse, module.get()));

  SCOPED_TRACE(module->ToString());
  EXPECT_TRUE(changed);

  HloInstruction* root_instruction =
      module->entry_computation()->root_instruction();
  const HloInstruction* param_0_slice = nullptr;
  const HloInstruction* param_1_first_slice = nullptr;
  const HloInstruction* param_1_second_slice = nullptr;
  EXPECT_THAT(
      root_instruction,
      GmockMatch(m::Add(m::Add(m::Slice(&param_0_slice, m::Parameter(0)),
                               m::Op(&param_1_first_slice)),
                        m::Op(&param_1_second_slice))));
  // The slice of param_1 should be evaluated only once and reused.
  EXPECT_EQ(param_1_first_slice, param_1_second_slice);
  EXPECT_THAT(param_1_first_slice, GmockMatch(m::Slice(m::Parameter(1))));
  EXPECT_THAT(param_0_slice->slice_starts(), ElementsAre(0, 0));
  EXPECT_THAT(param_0_slice->slice_limits(), ElementsAre(2, 9));
  EXPECT_THAT(param_0_slice->slice_strides(), ElementsAre(1, 1));
  EXPECT_THAT(param_1_first_slice->slice_starts(), ElementsAre(0, 0));
  EXPECT_THAT(param_1_first_slice->slice_limits(), ElementsAre(2, 9));
  EXPECT_THAT(param_1_first_slice->slice_strides(), ElementsAre(1, 1));
}

TEST_F(SliceHoisterTest, DoesNotHoistSliceThroughAddIfElementTypesDoNotMatch) {
  absl::string_view module_str = R"(
    HloModule module
    ENTRY main {
      add_op = f32[8,9] add(f16[8,9] parameter(0), f32[8,9] parameter(1))
      ROOT slice_op = f32[2,9] slice(f32[8,9] add_op), slice={[0:2], [0:9]}
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(module_str));

  SliceHoister slice_hoister;
  TF_ASSERT_OK_AND_ASSIGN(bool changed,
                          RunHloPass(&slice_hoister, module.get()));

  SCOPED_TRACE(module->ToString());
  EXPECT_FALSE(changed);
}

TEST_F(SliceHoisterTest,
       DoesNotHoistSliceThroughAddIfAddTypeDoesNotMatchSliceType) {
  absl::string_view module_str = R"(
    HloModule module
    ENTRY main {
      add_op = f32[8,9] add(f32[8,9] parameter(0), f32[8,9] parameter(1))
      ROOT slice_op = f16[2,9] slice(f32[8,9] add_op), slice={[0:2], [0:9]}
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(module_str));

  SliceHoister slice_hoister;
  TF_ASSERT_OK_AND_ASSIGN(bool changed,
                          RunHloPass(&slice_hoister, module.get()));

  SCOPED_TRACE(module->ToString());
  EXPECT_FALSE(changed);
}

TEST_F(SliceHoisterTest,
       DoesNotHoistSliceThroughAddIfAddTypeDoesNotMatchOperandsType) {
  absl::string_view module_str = R"(
    HloModule module
    ENTRY main {
      add_op = f32[8,9] add(f16[8,9] parameter(0), f16[8,9] parameter(1))
      ROOT slice_op = f32[2,9] slice(f32[8,9] add_op), slice={[0:2], [0:9]}
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(module_str));

  SliceHoister slice_hoister;
  TF_ASSERT_OK_AND_ASSIGN(bool changed,
                          RunHloPass(&slice_hoister, module.get()));

  SCOPED_TRACE(module->ToString());
  EXPECT_FALSE(changed);
}
}  // anonymous namespace
}  // namespace xla

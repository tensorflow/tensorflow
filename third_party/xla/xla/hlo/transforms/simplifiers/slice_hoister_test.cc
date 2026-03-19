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
            /*allow_mixed_precision_in_hlo_verifier=*/false) {};
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

TEST_F(SliceHoisterTest, HoistSliceThroughMultipleElementwiseBinaryOperations) {
  absl::string_view module_str = R"(
    HloModule module
    ENTRY main {
      param_0 = f32[8,9] parameter(0)
      param_1 = f32[8,9] parameter(1)
      param_2 = f32[8,9] parameter(2)
      multiply_op = f32[8,9] multiply(f32[8,9] param_0, f32[8,9] param_1)
      add_op = f32[8,9] add(f32[8,9] multiply_op, f32[8,9] param_1)
      divide_op = f32[8,9] divide(f32[8,9] add_op, f32[8,9] param_2)
      power_op = f32[8,9] power(f32[8,9] param_2, f32[8,9] divide_op)
      ROOT slice_op = f32[2,9] slice(f32[8,9] power_op), slice={[0:2], [0:9]}
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(module_str));

  SliceHoister slice_hoister;
  TF_ASSERT_OK_AND_ASSIGN(bool changed,
                          RunHloPass(&slice_hoister, module.get()));

  SCOPED_TRACE(module->ToString());
  EXPECT_TRUE(changed);

  HloCSE cse = HloCSE(
      false);  // CSE to remove the redundant slices of param_1 and param_2.
  TF_ASSERT_OK_AND_ASSIGN(changed, RunHloPass(&cse, module.get()));

  SCOPED_TRACE(module->ToString());
  EXPECT_TRUE(changed);

  HloInstruction* root_instruction =
      module->entry_computation()->root_instruction();
  const HloInstruction* param_0_slice = nullptr;
  const HloInstruction* param_1_slice_1 = nullptr;
  const HloInstruction* param_1_slice_2 = nullptr;
  const HloInstruction* param_2_slice_1 = nullptr;
  const HloInstruction* param_2_slice_2 = nullptr;
  EXPECT_THAT(
      root_instruction,
      GmockMatch(m::Power(
          m::Slice(&param_2_slice_1, m::Parameter(2)),
          m::Divide(
              m::Add(m::Multiply(m::Slice(&param_0_slice, m::Parameter(0)),
                                 m::Slice(&param_1_slice_1, m::Parameter(1))),
                     m::Slice(&param_1_slice_2, m::Parameter(1))),
              m::Slice(&param_2_slice_2, m::Parameter(2))))));
  // The slices of param_1 and param_2 should be evaluated only once and
  // reused.
  EXPECT_EQ(param_1_slice_1, param_1_slice_2);
  EXPECT_EQ(param_2_slice_1, param_2_slice_2);
  auto check_slice_attributes = [](const HloInstruction* slice) {
    EXPECT_THAT(slice->slice_starts(), ElementsAre(0, 0));
    EXPECT_THAT(slice->slice_limits(), ElementsAre(2, 9));
    EXPECT_THAT(slice->slice_strides(), ElementsAre(1, 1));
  };
  check_slice_attributes(param_0_slice);
  check_slice_attributes(param_1_slice_1);
  check_slice_attributes(param_2_slice_1);
}

// Dot is not an element-wise operation.
TEST_F(SliceHoisterTest, DoesNotHoistSliceThroughDot) {
  absl::string_view module_str = R"(
    HloModule module
    ENTRY main {
      p0 = f32[8,10] parameter(0)
      p1 = f32[10,9] parameter(1)
      dot_op = f32[8,9] dot(p0, p1), lhs_contracting_dims={1}, rhs_contracting_dims={0}
      ROOT slice_op = f32[2,9] slice(f32[8,9] dot_op), slice={[0:2], [0:9]}
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

// Negate is not a binary operation.
// TODO(b/434724820): Hoist slices through element-wise unary operations.
TEST_F(SliceHoisterTest, DoesNotHoistSliceThroughNegate) {
  absl::string_view module_str = R"(
    HloModule module
    ENTRY main {
      p0 = f32[8,9] parameter(0)
      neg_op = f32[8,9] negate(p0)
      ROOT slice_op = f32[2,9] slice(f32[8,9] neg_op), slice={[0:2], [0:9]}
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

TEST_F(SliceHoisterTest, HoistSliceThroughTranspose) {
  const absl::string_view hlo_string = R"(
    HloModule module
    ENTRY main {
      param_0 = f32[2,3,5] parameter(0)
      transpose_op = f32[3,2,5] transpose(param_0), dimensions={1,0,2}
      ROOT slice = f32[2,1,2] slice(transpose_op), slice={[1:3], [0:1:2], [0:4:3]}
    }
    )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));

  SliceHoister slice_hoister;
  TF_ASSERT_OK_AND_ASSIGN(bool changed,
                          RunHloPass(&slice_hoister, module.get()));

  SCOPED_TRACE(module->ToString());
  EXPECT_TRUE(changed);
  const HloInstruction* transpose_op = nullptr;
  const HloInstruction* slice_op = nullptr;
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::Transpose(&transpose_op,
                                      m::Slice(&slice_op, m::Parameter(0)))));
  EXPECT_THAT(transpose_op->dimensions(), ElementsAre(1, 0, 2));
  EXPECT_THAT(slice_op->slice_starts(), ElementsAre(0, 1, 0));
  EXPECT_THAT(slice_op->slice_limits(), ElementsAre(1, 3, 4));
  EXPECT_THAT(slice_op->slice_strides(), ElementsAre(2, 1, 3));
}

TEST_F(SliceHoisterTest, HoistSliceThroughTransposeAndAdd) {
  const absl::string_view hlo_string = R"(
    HloModule module
    ENTRY main {
      p0 = f32[2,3,5] parameter(0)
      p1 = f32[2,3,5] parameter(1)
      add_op = f32[2,3,5] add(p0, p1)
      transpose_op = f32[3,2,5] transpose(add_op), dimensions={1,0,2}
      ROOT slice_op = f32[2,1,2] slice(transpose_op), slice={[1:3], [0:1:2], [0:4:3]}
    }
    )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));

  SliceHoister slice_hoister;
  TF_ASSERT_OK_AND_ASSIGN(bool changed,
                          RunHloPass(&slice_hoister, module.get()));

  SCOPED_TRACE(module->ToString());
  EXPECT_TRUE(changed);
  const HloInstruction* transpose_op = nullptr;
  const HloInstruction* p0_slice = nullptr;
  const HloInstruction* p1_slice = nullptr;
  EXPECT_THAT(
      module->entry_computation()->root_instruction(),
      GmockMatch(m::Transpose(&transpose_op,
                              m::Add(m::Slice(&p0_slice, m::Parameter(0)),
                                     m::Slice(&p1_slice, m::Parameter(1))))));
  EXPECT_THAT(transpose_op->dimensions(), ElementsAre(1, 0, 2));
  auto check_slice_attributes = [](const HloInstruction* slice) {
    EXPECT_THAT(slice->slice_starts(), ElementsAre(0, 1, 0));
    EXPECT_THAT(slice->slice_limits(), ElementsAre(1, 3, 4));
    EXPECT_THAT(slice->slice_strides(), ElementsAre(2, 1, 3));
  };
  check_slice_attributes(p0_slice);
  check_slice_attributes(p1_slice);
}

}  // anonymous namespace
}  // namespace xla

/* Copyright 2022 The OpenXLA Authors.

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

#include "xla/hlo/transforms/simplifiers/convert_mover.h"

#include <memory>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/service/pattern_matcher.h"
#include "xla/service/pattern_matcher_gmock.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace {

namespace m = ::xla::match;

class ConvertMoverTest : public HloHardwareIndependentTestBase {
 public:
  ConvertMoverTest()
      : HloHardwareIndependentTestBase(
            /*verifier_layout_sensitive=*/false,
            /*allow_mixed_precision_in_hlo_verifier=*/false) {}
};

template <typename T>
auto MatchConvertToS8(T&& operand) {
  return m::Convert(operand).WithShape(m::Shape().WithElementType(S8));
}
template <typename T>
auto MatchConvertToF16(T&& operand) {
  return m::Convert(operand).WithShape(m::Shape().WithElementType(F16));
}
template <typename T>
auto MatchConvertToF32(T&& operand) {
  return m::Convert(operand).WithShape(m::Shape().WithElementType(F32));
}
template <typename T>
auto MatchConvertToC64(T&& operand) {
  return m::Convert(operand).WithShape(m::Shape().WithElementType(C64));
}

TEST_F(ConvertMoverTest, MoveDownThroughConcat) {
  absl::string_view module_string = R"(
  HloModule module

  ENTRY main {
    ROOT root = concatenate(f32[10] convert(f16[10] parameter(0)),
                            f32[10] convert(f16[10] parameter(1))),
                dimensions={0}
  })";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(module_string));
  ConvertMover pass;
  TF_ASSERT_OK_AND_ASSIGN(bool changed, RunHloPass(&pass, module.get()));
  SCOPED_TRACE(module->ToString());
  EXPECT_TRUE(changed);
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(MatchConvertToF32(
                  m::Concatenate(m::Parameter(0), m::Parameter(1)))));
}

TEST_F(ConvertMoverTest, NoMoveDownThroughConcatWithDifferentSrcTypes) {
  absl::string_view module_string = R"(
  HloModule module

  ENTRY main {
    ROOT root = concatenate(f32[10] convert(bf16[10] parameter(0)),
                            f32[10] convert(f16[10] parameter(1))),
                dimensions={0}
  })";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(module_string));
  ConvertMover pass;
  TF_ASSERT_OK_AND_ASSIGN(bool changed, RunHloPass(&pass, module.get()));
  SCOPED_TRACE(module->ToString());
  EXPECT_FALSE(changed);
}

TEST_F(ConvertMoverTest, MoveUpReshape) {
  absl::string_view module_string = R"(
  HloModule module

  ENTRY main {
    ROOT root = f16[10,10] convert(f32[10,10] reshape(f32[100] parameter(0)))
  })";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(module_string));
  ConvertMover pass;
  TF_ASSERT_OK_AND_ASSIGN(bool changed, RunHloPass(&pass, module.get()));
  SCOPED_TRACE(module->ToString());
  EXPECT_TRUE(changed);
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::Reshape(MatchConvertToF16(m::Parameter(0)))));
}

TEST_F(ConvertMoverTest, MoveUpTwoTransposes) {
  absl::string_view module_string = R"(
  HloModule module

  ENTRY main {
    t1 = transpose(f32[3,4] parameter(0)), dimensions={1,0}
    t2 = transpose(t1), dimensions={1,0}
    ROOT root = f16[3,4] convert(t2)
  })";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(module_string));
  ConvertMover pass;
  TF_ASSERT_OK_AND_ASSIGN(bool changed, RunHloPass(&pass, module.get()));
  SCOPED_TRACE(module->ToString());
  EXPECT_TRUE(changed);
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::Transpose(
                  m::Transpose(MatchConvertToF16(m::Parameter(0))))));
}

TEST_F(ConvertMoverTest, MoveDownTwoSlices) {
  absl::string_view module_string = R"(
  HloModule module

  ENTRY main {
    slice1 = f32[9] slice(f32[10] convert(f16[10] parameter(0))), slice={[0:9]}
    ROOT slice2 = f32[8] slice(slice1), slice={[0:8]}
  })";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(module_string));
  ConvertMover pass;
  TF_ASSERT_OK_AND_ASSIGN(bool changed, RunHloPass(&pass, module.get()));
  SCOPED_TRACE(module->ToString());
  EXPECT_TRUE(changed);
  EXPECT_THAT(
      module->entry_computation()->root_instruction(),
      GmockMatch(MatchConvertToF32(m::Slice(m::Slice(m::Parameter(0))))));
}

TEST_F(ConvertMoverTest, MoveDownC64) {
  absl::string_view module_string = R"(
  HloModule module

  ENTRY main {
    ROOT root = concatenate(c64[10] convert(f32[10] parameter(0)),
                            c64[10] convert(f32[10] parameter(1))),
                dimensions={0}
  })";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(module_string));
  ConvertMover pass;
  TF_ASSERT_OK_AND_ASSIGN(bool changed, RunHloPass(&pass, module.get()));
  SCOPED_TRACE(module->ToString());
  EXPECT_TRUE(changed);
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(MatchConvertToC64(m::Concatenate(  //
                  m::Parameter(0),                          //
                  m::Parameter(1)                           //
                  ))));
}

TEST_F(ConvertMoverTest, MoveDownC64Constant) {
  absl::string_view module_string = R"(
  HloModule module

  ENTRY main {
    ROOT root = concatenate(c64[2] convert(f32[2] parameter(0)),
                            c64[2] convert(f32[2] parameter(1)),
                            c64[2] constant({(1,1), (-1,-1)})),
                dimensions={0}
  })";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(module_string));
  ConvertMover pass;
  TF_ASSERT_OK_AND_ASSIGN(bool changed, RunHloPass(&pass, module.get()));
  SCOPED_TRACE(module->ToString());
  EXPECT_FALSE(changed);
}

TEST_F(ConvertMoverTest, MoveUpPad) {
  absl::string_view module_string = R"(
  HloModule module

  ENTRY main {
    pad = f32[10] pad(f32[8] parameter(0), f32[] constant(0)), padding=1_1
    ROOT root = f16[10] convert(pad)
  })";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(module_string));
  ConvertMover pass;
  TF_ASSERT_OK_AND_ASSIGN(bool changed, RunHloPass(&pass, module.get()));
  SCOPED_TRACE(module->ToString());
  EXPECT_TRUE(changed);
  EXPECT_THAT(
      module->entry_computation()->root_instruction(),
      GmockMatch(m::Pad(MatchConvertToF16(m::Parameter(0)),
                        MatchConvertToF16(m::ConstantEffectiveScalar(0)))));
}

// The out-of-range constant shouldn't prevent us from moving the narrowing
// conversion above the pad op.
TEST_F(ConvertMoverTest, MoveUpPadWithOutOfRangeConstant) {
  absl::string_view module_string = R"(
  HloModule module

  ENTRY main {
    pad = s32[10] pad(s32[8] parameter(0), s32[] constant(1000)), padding=1_1
    ROOT root = s8[10] convert(pad)
  })";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(module_string));
  ConvertMover pass;
  TF_ASSERT_OK_AND_ASSIGN(bool changed, RunHloPass(&pass, module.get()));
  SCOPED_TRACE(module->ToString());
  EXPECT_TRUE(changed);
  EXPECT_THAT(
      module->entry_computation()->root_instruction(),
      GmockMatch(m::Pad(MatchConvertToS8(m::Parameter(0)),
                        MatchConvertToS8(m::ConstantEffectiveScalar(1000)))));
}

TEST_F(ConvertMoverTest, MoveDownPad) {
  absl::string_view module_string = R"(
  HloModule module

  ENTRY main {
    ROOT pad = f32[10] pad(f32[8] convert(f16[8] parameter(0)), f32[] constant(0)),
               padding=1_1
  })";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(module_string));
  ConvertMover pass;
  TF_ASSERT_OK_AND_ASSIGN(bool changed, RunHloPass(&pass, module.get()));
  SCOPED_TRACE(module->ToString());
  EXPECT_TRUE(changed);
  EXPECT_THAT(
      module->entry_computation()->root_instruction(),
      GmockMatch(MatchConvertToF32(m::Pad(
          m::Parameter(0), MatchConvertToF16(m::ConstantEffectiveScalar(0))))));
}

TEST_F(ConvertMoverTest, NoMoveDownPadBecauseConstantIsOutOfRange) {
  absl::string_view module_string = R"(
  HloModule module

  ENTRY main {
    ROOT pad = f32[10] pad(f32[8] convert(f16[8] parameter(0)), f32[] constant(1e9)),
               padding=1_1
  })";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(module_string));
  ConvertMover pass;
  TF_ASSERT_OK_AND_ASSIGN(bool changed, RunHloPass(&pass, module.get()));
  SCOPED_TRACE(module->ToString());
  EXPECT_FALSE(changed);
}

}  // anonymous namespace
}  // namespace xla

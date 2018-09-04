/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/hlo_verifier.h"

#include <memory>
#include <utility>

#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/service/hlo_parser.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/core/status_test_util.h"

namespace xla {
namespace {

using ::testing::HasSubstr;

// This class cannot be converted to use HloVerifiedTestBase. It explicitly
// uses HloTestBase to create and test malformed HLOs.
class HloVerifierTest : public HloTestBase {
 public:
  HloVerifierTest()
      : HloTestBase(/*verifier_layout_sensitive=*/false,
                    /*allow_mixed_precision_in_hlo_verifier=*/false) {}
};

class HloVerifierTestAllowMixedPrecision : public HloTestBase {
 public:
  HloVerifierTestAllowMixedPrecision()
      : HloTestBase(/*verifier_layout_sensitive=*/false,
                    /*allow_mixed_precision_in_hlo_verifier=*/true) {}
};

TEST_F(HloVerifierTest, NullInstructionParent) {
  HloComputation::Builder builder(TestName());
  const Shape scalar_shape = ShapeUtil::MakeShape(F32, {});
  HloInstruction* param = builder.AddInstruction(
      HloInstruction::CreateParameter(0, scalar_shape, "param"));
  HloInstruction* negate = builder.AddInstruction(
      HloInstruction::CreateUnary(scalar_shape, HloOpcode::kNegate, param));
  auto module = CreateNewModule();
  module->AddEntryComputation(builder.Build());

  TF_ASSERT_OK(verifier().Run(module.get()).status());

  negate->set_parent(nullptr);

  auto status = verifier().Run(module.get()).status();
  ASSERT_FALSE(status.ok());
  EXPECT_THAT(status.error_message(), HasSubstr("has a null parent pointer"));
}

TEST_F(HloVerifierTest, NullComputationParent) {
  HloComputation::Builder builder(TestName());
  const Shape scalar_shape = ShapeUtil::MakeShape(F32, {});
  HloInstruction* param = builder.AddInstruction(
      HloInstruction::CreateParameter(0, scalar_shape, "param"));
  builder.AddInstruction(
      HloInstruction::CreateUnary(scalar_shape, HloOpcode::kNegate, param));
  auto module = CreateNewModule();
  HloComputation* computation = module->AddEntryComputation(builder.Build());

  TF_ASSERT_OK(verifier().Run(module.get()).status());

  computation->set_parent(nullptr);

  auto status = verifier().Run(module.get()).status();
  ASSERT_FALSE(status.ok());
  EXPECT_THAT(status.error_message(), HasSubstr("has a null parent pointer"));
}

TEST_F(HloVerifierTest, DifferentOperandParents) {
  HloComputation::Builder builder(TestName());
  const Shape scalar_shape = ShapeUtil::MakeShape(F32, {});
  HloInstruction* param = builder.AddInstruction(
      HloInstruction::CreateParameter(0, scalar_shape, "param"));
  HloInstruction* negate = builder.AddInstruction(
      HloInstruction::CreateUnary(scalar_shape, HloOpcode::kNegate, param));
  auto module = CreateNewModule();
  module->AddEntryComputation(builder.Build());

  HloComputation::Builder emb_builder(TestName());
  HloInstruction* emb_param = emb_builder.AddInstruction(
      HloInstruction::CreateParameter(0, scalar_shape, "param"));
  module->AddEmbeddedComputation(emb_builder.Build());

  TF_ASSERT_OK(verifier().Run(module.get()).status());
  TF_ASSERT_OK(negate->ReplaceOperandWith(0, emb_param));

  auto status = verifier().Run(module.get()).status();
  ASSERT_FALSE(status.ok());
  EXPECT_THAT(status.error_message(),
              HasSubstr("is in a different computation"));
}

TEST_F(HloVerifierTest, ResetsShapeVerifierState) {
  HloComputation::Builder builder(TestName());
  Shape s1 = ShapeUtil::MakeShape(F32, {1});
  Shape s2 = ShapeUtil::MakeShape(F32, {2});

  HloInstruction* param =
      builder.AddInstruction(HloInstruction::CreateParameter(0, s1, "param"));

  // Create an add instruction with the incorrect shape.
  HloInstruction* add = builder.AddInstruction(
      HloInstruction::CreateBinary(s2, HloOpcode::kAdd, param, param));

  // In order to trigger the bug we're checking for, the instruction with the
  // bad shape can't be the root of the computation.
  builder.AddInstruction(
      HloInstruction::CreateBinary(s2, HloOpcode::kMultiply, add, add));

  auto module = CreateNewModule();
  module->AddEntryComputation(builder.Build());

  // Run the verifier twice.  It should fail both times, because it shouldn't
  // carry state in its DFS visitor between runs.
  EXPECT_FALSE(verifier().Run(module.get()).status().ok());
  EXPECT_FALSE(verifier().Run(module.get()).status().ok());
}

TEST_F(HloVerifierTest, CheckCallOperandParameterShapesMismatch) {
  const char* const hlo_string = R"(
HloModule Module

callme {
  ROOT param = (s32[], f32[4]) parameter(0)
}

ENTRY entry {
  p0 = (f32[4], s32[]) parameter(0)
  ROOT mycall = (s32[], f32[4]) call(p0), to_apply=callme
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseHloString(hlo_string));

  auto status = verifier().Run(module.get()).status();
  ASSERT_FALSE(status.ok());
  EXPECT_THAT(status.error_message(),
              HasSubstr("shape does not match parameter"));
}

TEST_F(HloVerifierTest, CheckConditionalOperandParameterShapesMismatch) {
  const char* const hlo_string = R"(
HloModule Module

true_branch {
  tparam = (s32[], f32[4]) parameter(0)
  ROOT tgte1 = f32[4] get-tuple-element(tparam), index=1
}

false_branch {
  fparam = (s32[], f32[4]) parameter(0)
  ROOT fgte1 = f32[4] get-tuple-element(fparam), index=1
}

ENTRY entry {
  p0 = (f32[4], s32[]) parameter(0)
  constant = pred[] constant(true)
  ROOT conditional = f32[4] conditional(constant, p0, p0),
    true_computation=true_branch, false_computation=false_branch
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseHloString(hlo_string));

  auto status = verifier().Run(module.get()).status();
  ASSERT_FALSE(status.ok());
  EXPECT_THAT(status.error_message(),
              HasSubstr("shape does not match parameter"));
}

TEST_F(HloVerifierTest, RngOpnd0NotScalar) {
  const char* const hlo_string = R"(
  HloModule Module

  ENTRY RngOpnd0NotScalar {
   constant.0 = f32[] constant(0)
   constant.1 = f16[2] constant({1, 3})
   ROOT rng.0 = f32[10]{0} rng(f32[] constant.0, f16[2] constant.1),
    distribution=rng_uniform
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseHloString(hlo_string));

  auto status = verifier().Run(module.get()).status();
  ASSERT_FALSE(status.ok());
  EXPECT_THAT(status.error_message(), HasSubstr("Expected scalar type"));
}

TEST_F(HloVerifierTest, RngOperandElementTypesDoNotMatch) {
  const char* const hlo_string = R"(
  HloModule Module

  ENTRY RngOperandElementTypesNotMatch {
   constant.0 = f32[] constant(0)
   constant.1 = f16[] constant(1)
   ROOT rng.0 = f32[10]{0} rng(f32[] constant.0, f16[] constant.1),
    distribution=rng_normal
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseHloString(hlo_string));

  auto status = verifier().Run(module.get()).status();
  ASSERT_FALSE(status.ok());
  EXPECT_THAT(status.error_message(),
              HasSubstr("Expected compatible element types"));
}

TEST_F(HloVerifierTest, RngMixedPrecisionNotAllowed) {
  const char* const hlo_string = R"(
  HloModule Module

  ENTRY RngResultElementTypeNotMatch {
   constant.0 = f32[] constant(0)
   constant.1 = f32[] constant(1)
   ROOT rng.0 = f16[10]{0} rng(f32[] constant.0, f32[] constant.1),
    distribution=rng_normal
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseHloString(hlo_string));

  auto status = verifier().Run(module.get()).status();
  ASSERT_FALSE(status.ok());
  EXPECT_THAT(status.error_message(),
              HasSubstr("Expected compatible element types"));
}

TEST_F(HloVerifierTestAllowMixedPrecision, RngMixedPrecisionAllowed) {
  const char* const hlo_string = R"(
  HloModule Module

  ENTRY RngResultElementTypeNotMatch {
   constant.0 = f32[] constant(0)
   constant.1 = f32[] constant(1)
   ROOT rng.0 = f16[10]{0} rng(f32[] constant.0, f32[] constant.1),
    distribution=rng_normal
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseHloString(hlo_string));

  auto status = verifier().Run(module.get()).status();
  ASSERT_TRUE(status.ok());
}

TEST_F(HloVerifierTest, RngElementTypeNotSupported) {
  const char* const hlo_string = R"(
  HloModule Module

  ENTRY RngElementTypeNotSupported {
   constant.0 = s32[] constant(0)
   constant.1 = s32[] constant(1)
   ROOT rng.0 = s32[10]{0} rng(s32[] constant.0, s32[] constant.1),
    distribution=rng_normal
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseHloString(hlo_string));

  auto status = verifier().Run(module.get()).status();
  ASSERT_FALSE(status.ok());
  EXPECT_THAT(status.error_message(), HasSubstr("Element type not supported"));
}

TEST_F(HloVerifierTest, NegativeInteriorPaddingNotAllowed) {
  // This testcase can't be written using textual HLO, because it doesn't parse
  // negative interior padding.  That's probably a feature.  :)
  HloComputation::Builder builder(TestName());
  HloInstruction* param =
      builder.AddInstruction(HloInstruction::CreateParameter(
          0, ShapeUtil::MakeShape(F32, {100}), "param"));
  PaddingConfig padding_config;
  padding_config.add_dimensions()->set_interior_padding(-1);
  builder.AddInstruction(HloInstruction::CreatePad(
      ShapeUtil::MakeShape(F32, {100}), param,
      builder.AddInstruction(HloInstruction::CreateConstant(
          LiteralUtil::Zero(F32).CloneToUnique())),
      padding_config));

  auto module = CreateNewModule();
  module->AddEntryComputation(builder.Build());

  auto status = verifier().Run(module.get()).status();
  ASSERT_FALSE(status.ok());
  EXPECT_THAT(status.error_message(),
              HasSubstr("Interior padding cannot be negative"));
}

TEST_F(HloVerifierTest, PadNegativeInteriorDilationNotAllowed) {
  // This testcase can't be written using textual HLO, because it doesn't parse
  // negative interior padding.  That's probably a feature.  :)
  HloComputation::Builder builder(TestName());
  HloInstruction* param =
      builder.AddInstruction(HloInstruction::CreateParameter(
          0, ShapeUtil::MakeShape(F32, {100}), "param"));
  PaddingConfig padding_config;
  padding_config.add_dimensions()->set_interior_padding(-1);
  builder.AddInstruction(HloInstruction::CreatePad(
      ShapeUtil::MakeShape(F32, {100}), param,
      builder.AddInstruction(HloInstruction::CreateConstant(
          LiteralUtil::Zero(F32).CloneToUnique())),
      padding_config));

  auto module = CreateNewModule();
  module->AddEntryComputation(builder.Build());

  EXPECT_THAT(verifier().Run(module.get()).status().error_message(),
              HasSubstr("Interior padding cannot be negative"));
}

// Simple module containing a convolution as the root.
static const char* const kConvHloString = R"(
HloModule module
ENTRY entry_computation {
  param0 = f16[128,128,56,56] parameter(0)
  param1 = f16[3,3,128,128] parameter(1)
  zero_f16 = f16[] constant(0)
  ROOT conv = f16[128,128,28,28] convolution(param0, param1),
    window={size=3x3 stride=2x2}, dim_labels=bf01_01io->bf01
})";

TEST_F(HloVerifierTest, ConvNegativeWindowDilationNotAllowed) {
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseHloString(kConvHloString));
  auto* conv = module->entry_computation()->root_instruction();
  Window w = conv->window();
  w.mutable_dimensions(0)->set_window_dilation(-1);
  conv->set_window(w);

  EXPECT_THAT(verifier().Run(module.get()).status().error_message(),
              HasSubstr("non-positive window dilation factor"));
}

TEST_F(HloVerifierTest, ConvNegativeBaseDilationNotAllowed) {
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseHloString(kConvHloString));
  auto* conv = module->entry_computation()->root_instruction();
  Window w = conv->window();
  w.mutable_dimensions(0)->set_base_dilation(-1);
  conv->set_window(w);

  EXPECT_THAT(verifier().Run(module.get()).status().error_message(),
              HasSubstr("non-positive base area dilation factor"));
}

}  // namespace
}  // namespace xla

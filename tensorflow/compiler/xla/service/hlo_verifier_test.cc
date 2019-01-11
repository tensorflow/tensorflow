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
#include "tensorflow/compiler/xla/service/hlo_module_config.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/service/hlo_parser.h"
#include "tensorflow/compiler/xla/service/layout_assignment.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/xla.pb.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/core/status_test_util.h"

namespace xla {
namespace {

using ::testing::HasSubstr;

std::unique_ptr<HloModule> CreateUnverifiedModule() {
  return absl::make_unique<HloModule>("module", HloModuleConfig());
}

// This class cannot be converted to use HloTestBase. It explicitly
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

class HloVerifierTestLayoutSensitive : public HloTestBase {
 public:
  HloVerifierTestLayoutSensitive()
      : HloTestBase(/*verifier_layout_sensitive=*/true,
                    /*allow_mixed_precision_in_hlo_verifier=*/false,
                    LayoutAssignment::InstructionCanChangeLayout) {}
};

TEST_F(HloVerifierTest, NullInstructionParent) {
  HloComputation::Builder builder(TestName());
  const Shape scalar_shape = ShapeUtil::MakeShape(F32, {});
  HloInstruction* param = builder.AddInstruction(
      HloInstruction::CreateParameter(0, scalar_shape, "param"));
  HloInstruction* negate = builder.AddInstruction(
      HloInstruction::CreateUnary(scalar_shape, HloOpcode::kNegate, param));
  auto module = CreateUnverifiedModule();
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
  auto module = CreateUnverifiedModule();
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
  auto module = CreateUnverifiedModule();
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

  auto module = CreateUnverifiedModule();
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
      builder.AddInstruction(
          HloInstruction::CreateConstant(LiteralUtil::Zero(F32))),
      padding_config));

  auto module = CreateUnverifiedModule();
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
      builder.AddInstruction(
          HloInstruction::CreateConstant(LiteralUtil::Zero(F32).Clone())),
      padding_config));

  auto module = CreateUnverifiedModule();
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

static const char* const kAddWithLayoutChangeHlo = R"(
   HloModule AddWithLayoutChange
    ENTRY AddWithLayoutChange {
      par0 = f32[3,4]{1,0} parameter(0)
      par1 = f32[3,4]{0,1} parameter(1)
      ROOT add0 = f32[3,4]{1,0} add(par0,par1)
    }
  )";

TEST_F(HloVerifierTest, AddWithLayoutChange) {
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseHloString(kAddWithLayoutChangeHlo));
  auto status = verifier().Run(module.get()).status();
  ASSERT_TRUE(status.ok());
}

TEST_F(HloVerifierTest, ScalarIndexDynamicSlice) {
  const char* const kScalarIndexDynamicSlice = R"(
    HloModule DynamicSlice_module

    ENTRY %DynamicSlice.v5 (original_parameter: s32[2,2,258], start_index: s32[]) -> s32[2,2,258] {
      %original_parameter = s32[2,2,258] parameter(0)
      %constant = s32[] constant(0)
      %start_index = s32[] parameter(1)
      ROOT %dynamic-slice = s32[2,2,258] dynamic-slice(s32[2,2,258] %original_parameter, s32[] %constant, s32[] %constant, s32[] %start_index), dynamic_slice_sizes={2,2,258}
    }
  )";

  HloModuleConfig config;
  DebugOptions debug_options = config.debug_options();
  debug_options.set_xla_allow_scalar_index_dynamic_ops(true);
  config.set_debug_options(debug_options);

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseHloString(kScalarIndexDynamicSlice, config));
  auto status = verifier().Run(module.get()).status();
  ASSERT_TRUE(status.ok());
}

TEST_F(HloVerifierTest, ScalarIndexDynamicUpdateSlice) {
  const char* const kScalarIndexDynamicSlice = R"(
    HloModule DynamicUpdateSlice_module

    ENTRY %DynamicUpdateSlice.v4 (input: s32[1,1,25,1], update: s32[1,1,2,1], start_index.0: s32[], start_index.1: s32[], start_index.2: s32[], start_index.3: s32[]) -> s32[1,1,25,1] {
      %input = s32[1,1,25,1]{3,2,1,0} parameter(0)
      %update = s32[1,1,2,1]{3,2,1,0} parameter(1)
      %start_index.0 = s32[] parameter(2)
      %start_index.1 = s32[] parameter(3)
      %start_index.2 = s32[] parameter(4)
      %start_index.3 = s32[] parameter(5)
      ROOT %dynamic-update-slice = s32[1,1,25,1]{3,2,1,0} dynamic-update-slice(s32[1,1,25,1]{3,2,1,0} %input, s32[1,1,2,1]{3,2,1,0} %update, s32[] %start_index.0, s32[] %start_index.1, s32[] %start_index.2, s32[] %start_index.3)
    }
  )";

  HloModuleConfig config;
  DebugOptions debug_options = config.debug_options();
  debug_options.set_xla_allow_scalar_index_dynamic_ops(true);
  config.set_debug_options(debug_options);

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseHloString(kScalarIndexDynamicSlice, config));
  auto status = verifier().Run(module.get()).status();
  ASSERT_TRUE(status.ok());
}

TEST_F(HloVerifierTestLayoutSensitive, AddWithLayoutChangeNotAllowed) {
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseHloString(kAddWithLayoutChangeHlo));
  auto status = verifier().Run(module.get()).status();
  ASSERT_FALSE(status.ok());
  EXPECT_THAT(status.error_message(),
              HasSubstr("Instruction shouldn't change layouts"));
}

TEST_F(HloVerifierTestLayoutSensitive, SliceWithLayoutChangeNotAllowed) {
  const char* const kSliceWithLayoutChangeHlo = R"(
   HloModule SliceWithLayoutChange
    ENTRY SliceWithLayoutChange {
      par0 = f32[4,5]{0,1} parameter(0)
      par1 = s32[] parameter(1)
      par2 = s32[] parameter(2)
      ROOT dslice0 = f32[3,4]{1,0} dynamic-slice(par0, par1, par2),
        dynamic_slice_sizes={3,4}
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseHloString(kSliceWithLayoutChangeHlo));
  auto status = verifier().Run(module.get()).status();
  ASSERT_FALSE(status.ok());
  EXPECT_THAT(status.error_message(),
              HasSubstr("Instruction shouldn't change layouts"));
}

TEST_F(HloVerifierTestLayoutSensitive, ConcatWithLayoutChangeNotAllowed) {
  const char* const kConcatWithLayoutChangeHlo = R"(
   HloModule ConcatWithLayoutChange
   ENTRY ConcatWithLayoutChange {
      par0 = f32[3,5]{0,1} parameter(0)
      par1 = f32[3,3]{1,0} parameter(1)
      ROOT concat0 = f32[3,8]{1,0} concatenate(f32[3,5] par0, f32[3,3] par1),
        dimensions={1}
   }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseHloString(kConcatWithLayoutChangeHlo));
  auto status = verifier().Run(module.get()).status();
  ASSERT_FALSE(status.ok());
  EXPECT_THAT(status.error_message(),
              HasSubstr("Instruction shouldn't change layouts"));
}

TEST_F(HloVerifierTest, BitcastCanNotChangeElementType) {
  const char* const hlo_string = R"(
  HloModule Module

  ENTRY BitcastCanNotChangeElementType {
   constant.0 = f32[2] constant({0.0, 0.0})
   ROOT bitcast = s32[2] bitcast(constant.0)
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseHloString(hlo_string));

  auto status = verifier().Run(module.get()).status();
  ASSERT_FALSE(status.ok());
  EXPECT_THAT(status.error_message(),
              HasSubstr("Bitcast can not change the element type"));
}

}  // namespace
}  // namespace xla

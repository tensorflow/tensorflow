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
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/str_replace.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_computation.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_instruction.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_opcode.h"
#include "tensorflow/compiler/xla/service/hlo_module_config.h"
#include "tensorflow/compiler/xla/service/hlo_parser.h"
#include "tensorflow/compiler/xla/service/layout_assignment.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/compiler/xla/xla.pb.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/tsl/lib/core/status_test_util.h"

namespace xla {
namespace {

using ::testing::HasSubstr;

std::unique_ptr<HloModule> CreateUnverifiedModule() {
  return std::make_unique<HloModule>("module", HloModuleConfig());
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

class HloVerifierTestLayoutFusion : public HloTestBase {
 public:
  HloVerifierTestLayoutFusion()
      : HloTestBase(/*verifier_layout_sensitive=*/true,
                    /*allow_mixed_precision_in_hlo_verifier=*/false) {}
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
  EXPECT_THAT(status.message(), HasSubstr("has a null parent pointer"));
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
  EXPECT_THAT(status.message(), HasSubstr("has a null parent pointer"));
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
  EXPECT_THAT(status.message(), HasSubstr("is in a different computation"));
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
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnUnverifiedModule(hlo_string));

  auto status = verifier().Run(module.get()).status();
  ASSERT_FALSE(status.ok());
  EXPECT_THAT(status.message(), HasSubstr("shape does not match parameter"));
}

TEST_F(HloVerifierTest, CheckCallThreadMismatch) {
  constexpr absl::string_view hlo = R"(
    HloModule Module

    callme {
      ROOT param = (s32[], f32[4]) parameter(0)
    }, execution_thread="parallel_thread"

    ENTRY entry {
      p0 = (s32[], f32[4]) parameter(0)
      ROOT mycall = (s32[], f32[4]) call(p0), to_apply=callme
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnUnverifiedModule(hlo));

  auto status = verifier().Run(module.get()).status();
  ASSERT_FALSE(status.ok());
  EXPECT_THAT(status.message(),
              HasSubstr("expects parent computation thread name same as called "
                        "computation's thread name"));
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
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnUnverifiedModule(hlo_string));

  auto status = verifier().Run(module.get()).status();
  ASSERT_FALSE(status.ok());
  EXPECT_THAT(status.message(), HasSubstr("shape does not match parameter"));
}

TEST_F(HloVerifierTest, CheckConditionalBranchIndexOperandShape) {
  const char* const hlo_string = R"(
  HloModule Module

  branch0 {
    tparam = f32[4] parameter(0)
    ROOT tgte1 = f32[4] ceil(tparam)
  }

  branch1 {
    fparam = f32[4] parameter(0)
    ROOT fgte1 = f32[4] floor(fparam)
  }

  branch2 {
    sparam = f32[4] parameter(0)
    ROOT sgte1 = f32[4] ceil(sparam)
  }

  ENTRY entry {
    p0 = f32[4] parameter(0)
    b0 = s32[] parameter(1)
    ROOT conditional = f32[4] conditional(b0, p0, p0, p0),
      branch_computations={branch0, branch1, branch2}
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnUnverifiedModule(hlo_string));
  auto status = verifier().Run(module.get()).status();

  HloInstruction* condition = FindInstruction(module.get(), "b0");
  *condition->mutable_shape() = ShapeUtil::MakeShape(F32, {});
  status = verifier().Run(module.get()).status();
  ASSERT_FALSE(status.ok());
  EXPECT_THAT(
      status.message(),
      HasSubstr(
          "first operand of indexed conditional must be a scalar of S32"));

  *condition->mutable_shape() = ShapeUtil::MakeShape(S32, {4});
  status = verifier().Run(module.get()).status();
  ASSERT_FALSE(status.ok());
  EXPECT_THAT(status.message(),
              HasSubstr("first operand of conditional must be a scalar"));
}

TEST_F(HloVerifierTest, CheckConditionalBranchThread) {
  const char* const hlo_string = R"(
    HloModule Module

    branch0 {
      tparam = f32[4] parameter(0)
      ROOT tgte1 = f32[4] ceil(tparam)
    }

    branch1 {
      fparam = f32[4] parameter(0)
      ROOT fgte1 = f32[4] floor(fparam)
    }, execution_thread="parallel_thread"

    branch2 {
      sparam = f32[4] parameter(0)
      ROOT sgte1 = f32[4] ceil(sparam)
    }

    ENTRY entry {
      p0 = f32[4] parameter(0)
      b0 = s32[] parameter(1)
      ROOT conditional = f32[4] conditional(b0, p0, p0, p0),
        branch_computations={branch0, branch1, branch2}
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnUnverifiedModule(hlo_string));
  auto status = verifier().Run(module.get()).status();
  EXPECT_THAT(status.message(),
              HasSubstr("expects parent computation thread name same as called "
                        "computation's thread name"));
}

TEST_F(HloVerifierTest, CheckConditionalBranchContainsAsyncThread) {
  const char* const hlo_string = R"(
    HloModule Module

    branch0 {
      tparam = f32[4] parameter(0)
      ROOT tgte1 = f32[4] ceil(tparam)
    }

    branch1 {
      fparam = f32[4] parameter(0)
      %async-start = ((f32[4]), f32[4], s32[]) custom-call-start(f32[4] fparam), async_execution_thread="parallel_thread", custom_call_target="foo"
      ROOT %async-done = f32[4] custom-call-done(((f32[4]), f32[4], s32[]) %async-start), async_execution_thread="parallel_thread", custom_call_target="foo"
    }

    branch2 {
      sparam = f32[4] parameter(0)
      ROOT sgte1 = f32[4] ceil(sparam)
    }

    ENTRY entry {
      p0 = f32[4] parameter(0)
      b0 = s32[] parameter(1)
      ROOT conditional = f32[4] conditional(b0, p0, p0, p0),
        branch_computations={branch0, branch1, branch2}
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnUnverifiedModule(hlo_string));
  TF_ASSERT_OK(verifier().Run(module.get()).status());
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
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnUnverifiedModule(hlo_string));

  auto status = verifier().Run(module.get()).status();
  ASSERT_FALSE(status.ok());
  EXPECT_THAT(status.message(), HasSubstr("Expected scalar type"));
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
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnUnverifiedModule(hlo_string));

  auto status = verifier().Run(module.get()).status();
  ASSERT_FALSE(status.ok());
  EXPECT_THAT(status.message(), HasSubstr("Expected compatible element types"));
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
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnUnverifiedModule(hlo_string));

  auto status = verifier().Run(module.get()).status();
  ASSERT_FALSE(status.ok());
  EXPECT_THAT(status.message(), HasSubstr("Expected compatible element types"));
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
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

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
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnUnverifiedModule(hlo_string));

  auto status = verifier().Run(module.get()).status();
  ASSERT_FALSE(status.ok());
  EXPECT_THAT(status.message(), HasSubstr("Element type not supported"));
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
  EXPECT_THAT(status.message(),
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

  EXPECT_THAT(verifier().Run(module.get()).status().message(),
              HasSubstr("Interior padding cannot be negative"));
}

TEST_F(HloVerifierTest, DotMixedPrecisionAllowed) {
  static const char* const kDotHloString = R"(
HloModule module
ENTRY entry_computation {
  a = f32[2,10] parameter(0)
  b = bf16[10,2] parameter(1)
  ROOT dot = f32[2,2] dot(a, b), lhs_contracting_dims={1}, rhs_contracting_dims={0}
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(kDotHloString));

  auto status = verifier().Run(module.get()).status();
  EXPECT_TRUE(status.ok()) << status;
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
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnUnverifiedModule(kConvHloString));
  auto* conv = module->entry_computation()->root_instruction();
  Window w = conv->window();
  w.mutable_dimensions(0)->set_window_dilation(-1);
  conv->set_window(w);

  EXPECT_THAT(verifier().Run(module.get()).status().message(),
              HasSubstr("non-positive window dilation factor"));
}

TEST_F(HloVerifierTest, ConvNegativeBaseDilationNotAllowed) {
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnUnverifiedModule(kConvHloString));
  auto* conv = module->entry_computation()->root_instruction();
  Window w = conv->window();
  w.mutable_dimensions(0)->set_base_dilation(-1);
  conv->set_window(w);

  EXPECT_THAT(verifier().Run(module.get()).status().message(),
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
  TF_ASSERT_OK_AND_ASSIGN(
      auto module, ParseAndReturnVerifiedModule(kAddWithLayoutChangeHlo));
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

  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(
                                           kScalarIndexDynamicSlice, config));
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

  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(
                                           kScalarIndexDynamicSlice, config));
  auto status = verifier().Run(module.get()).status();
  ASSERT_TRUE(status.ok());
}

TEST_F(HloVerifierTestAllowMixedPrecision, DynamicUpdateSliceMixedPrecision) {
  const char* const kDynamicUpdateSliceMixedPrecision = R"(
    HloModule kDynamicUpdateSliceMixedPrecision

    ENTRY %entry (parameter.0: f32[32,511,2048], parameter.1: bf16[32,511,512], parameter.2: s32[], parameter.3: s32[], parameter.4: s32[]) -> bf16[32,511,2048] {
      %parameter.0 = f32[32,511,2048] parameter(0)
      %parameter.1 = bf16[32,511,512] parameter(1)
      %parameter.2 = s32[] parameter(2)
      %parameter.3 = s32[] parameter(3)
      %parameter.4 = s32[] parameter(4)
      ROOT %dus = bf16[32,511,2048] dynamic-update-slice(f32[32,511,2048] %parameter.0, bf16[32,511,512] %parameter.1, s32[] %parameter.2, s32[] %parameter.3, s32[] %parameter.4)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnUnverifiedModule(
                                           kDynamicUpdateSliceMixedPrecision));
  auto status = verifier().Run(module.get()).status();
  ASSERT_FALSE(status.ok());
  EXPECT_THAT(status.message(),
              HasSubstr("Expected instruction to have shape equal to "
                        "f32[32,511,2048], actual shape is bf16[32,511,2048]"));
}

TEST_F(HloVerifierTestLayoutSensitive, AddWithLayoutChangeNotAllowed) {
  TF_ASSERT_OK_AND_ASSIGN(
      auto module, ParseAndReturnUnverifiedModule(kAddWithLayoutChangeHlo));
  auto status = verifier().Run(module.get()).status();
  ASSERT_FALSE(status.ok());
  EXPECT_THAT(status.message(),
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
  TF_ASSERT_OK_AND_ASSIGN(
      auto module, ParseAndReturnUnverifiedModule(kSliceWithLayoutChangeHlo));
  auto status = verifier().Run(module.get()).status();
  ASSERT_FALSE(status.ok());
  EXPECT_THAT(status.message(),
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
  TF_ASSERT_OK_AND_ASSIGN(
      auto module, ParseAndReturnUnverifiedModule(kConcatWithLayoutChangeHlo));
  auto status = verifier().Run(module.get()).status();
  ASSERT_FALSE(status.ok());
  EXPECT_THAT(status.message(),
              HasSubstr("Instruction shouldn't change layouts"));
}

TEST_F(HloVerifierTestLayoutSensitive, BitcastNeedsSameNumberOfElements) {
  const char* const hlo_string = R"(
  HloModule Module

  ENTRY BitcastNeedsToBeNoOp {
   constant.0 = f32[2] constant({0.0, 0.0})
   ROOT bitcast = f32[3] bitcast(constant.0)
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnUnverifiedModule(hlo_string));

  auto status = verifier().Run(module.get()).status();
  ASSERT_FALSE(status.ok());
  EXPECT_THAT(status.message(),
              HasSubstr("Bitcast cannot have different shape sizes of output "
                        "(12) and operand (8)"));
}

TEST_F(HloVerifierTest, SelectMixedPrecisionNotAllowed) {
  const char* const hlo_string = R"(
  HloModule Module

  ENTRY SelectMixedPrecisionNotAllowed {
   p0 = pred[32] parameter(0)
   p1 = f32[32] parameter(1)
   p2 = bf16[32] parameter(2)
   ROOT select = f32[32] select(p0, p1, p2)
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnUnverifiedModule(hlo_string));

  auto status = verifier().Run(module.get()).status();
  ASSERT_FALSE(status.ok());
  EXPECT_THAT(status.message(),
              HasSubstr("Seen floating point types of different precisions"));
}

TEST_F(HloVerifierTestAllowMixedPrecision, SelectMixedPrecisionAllowed) {
  const char* const hlo_string = R"(
  HloModule Module

  ENTRY SelectMixedPrecisionAllowed {
   p0 = pred[32] parameter(0)
   p1 = f32[32] parameter(1)
   p2 = bf16[32] parameter(2)
   ROOT select = f32[32] select(p0, p1, p2)
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  auto status = verifier().Run(module.get()).status();
  ASSERT_TRUE(status.ok());
}

TEST_F(HloVerifierTest, SelectTupleNotAllowed) {
  const char* const hlo_string = R"(
  HloModule Module

  ENTRY SelectWithTuple {
    p0 = (f32[], f32[]) parameter(0)
    p1 = (f32[], f32[]) parameter(1)
    p2 = pred[] parameter(2)
    ROOT select = (f32[], f32[]) select(p2, p0, p1)
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnUnverifiedModule(hlo_string));

  auto status = verifier().Run(module.get()).status();
  ASSERT_FALSE(status.ok());
  EXPECT_THAT(status.message(),
              HasSubstr("Expected array argument for select"));
}

TEST_F(HloVerifierTestLayoutSensitive, CopyStartAndCopyDone) {
  const char* const hlo_string = R"(
  HloModule Module

  ENTRY CopyStartAndCopyDone {
    p0 = f32[2,3]{1,0:S(1)} parameter(0)
    copy-start = (f32[2,3]{1,0:S(2)}, f32[2,3]{1,0:S(1)}, u32[]) copy-start(p0)
    ROOT copy-done = f32[2,3]{1,0:S(2)} copy-done(copy-start)
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  auto status = verifier().Run(module.get()).status();
  ASSERT_TRUE(status.ok());
}

TEST_F(HloVerifierTestLayoutSensitive, CopyStartAndCopyDoneWrongLayout) {
  const char* const hlo_string = R"(
  HloModule Module

  ENTRY CopyStartAndCopyDone {
    p0 = f32[2,3]{1,0:S(1)} parameter(0)
    copy-start = (f32[2,3]{0,1:S(2)}, f32[2,3]{1,0:S(1)}, u32[]) copy-start(p0)
    ROOT copy-done = f32[2,3]{1,0:S(2)} copy-done(copy-start)
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnUnverifiedModule(hlo_string));

  auto status = verifier().Run(module.get()).status();
  ASSERT_FALSE(status.ok());
  EXPECT_THAT(status.message(),
              HasSubstr("Expected instruction to have shape equal to"));
}

TEST_F(HloVerifierTest, CopyStartAndCopyDoneWrongType) {
  const char* const hlo_string = R"(
  HloModule Module

  ENTRY CopyStartAndCopyDone {
    p0 = f32[2,3] parameter(0)
    copy-start = f32[2,3] copy-start(p0)
    ROOT copy-done = f32[2,3] copy-done(copy-start)
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnUnverifiedModule(hlo_string));

  auto status = verifier().Run(module.get()).status();
  ASSERT_FALSE(status.ok());
  EXPECT_THAT(status.message(),
              HasSubstr("Expected instruction to have shape equal to "
                        "(f32[2,3], f32[2,3], u32[])"));
}

TEST_F(HloVerifierTest, CopyStartMultipleCopyDone) {
  const char* const hlo_string = R"(
  HloModule Module

  ENTRY CopyStartAndCopyDone {
    p0 = f32[2,3] parameter(0)
    copy-start = (f32[2,3], f32[2,3], u32[]) copy-start(p0)
    copy-done.1 = f32[2,3] copy-done(copy-start)
    copy-done.2 = f32[2,3] copy-done(copy-start)
    ROOT tuple = (f32[2,3], f32[2,3]) tuple(copy-done.1, copy-done.2)
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnUnverifiedModule(hlo_string));

  auto status = verifier().Run(module.get()).status();
  ASSERT_FALSE(status.ok());
  EXPECT_THAT(
      status.message(),
      HasSubstr("copy-start instruction requires one consumer, found 2"));
}

TEST_F(HloVerifierTest, CopyDoneNoCopyStart) {
  const char* const hlo_string = R"(
  HloModule Module

  ENTRY CopyStartAndCopyDone {
    p0 = f32[2,3] parameter(0)
    p1 = u32[] parameter(1)
    tuple = (f32[2,3], f32[2,3], u32[]) tuple(p0, p0, p1)
    ROOT copy-done = f32[2,3] copy-done(tuple)
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnUnverifiedModule(hlo_string));

  auto status = verifier().Run(module.get()).status();
  ASSERT_FALSE(status.ok());
  EXPECT_THAT(status.message(),
              HasSubstr("The operand of a copy-done instruction needs to be "
                        "copy-start, found tuple"));
}

TEST_F(HloVerifierTestLayoutSensitive, AsyncStartAndAsyncDone) {
  const char* const hlo_string = R"(
  HloModule Module

  ENTRY AsyncStartAndAsyncDone {
    p0 = f32[2,3]{1,0:S(1)} parameter(0)
    async-start = ((f32[2,3]{1,0:S(1)}), f32[2,3]{1,0:S(2)}, u32[]) custom-call-start(p0), custom_call_target="foo"
    ROOT async-done = f32[2,3]{1,0:S(2)} custom-call-done(async-start), custom_call_target="foo"
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  auto status = verifier().Run(module.get()).status();
  ASSERT_TRUE(status.ok());
}

TEST_F(HloVerifierTestLayoutSensitive, AsyncStartAndAsyncUpdateAndAsyncDone) {
  const char* const hlo_string = R"(
  HloModule Module

  ENTRY AsyncStartAndAsyncUpdateAndAsyncDone {
    p0 = f32[2,3]{1,0:S(1)} parameter(0)
    async-start = ((f32[2,3]{1,0:S(1)}), f32[2,3]{1,0:S(2)}, u32[]) custom-call-start(p0), custom_call_target="foo"
    async-update.1 = ((f32[2,3]{1,0:S(1)}), f32[2,3]{1,0:S(2)}, u32[]) custom-call-update(async-start), custom_call_target="foo"
    async-update.2 = ((f32[2,3]{1,0:S(1)}), f32[2,3]{1,0:S(2)}, u32[]) custom-call-update(async-update.1), custom_call_target="foo"
    ROOT async-done = f32[2,3]{1,0:S(2)} custom-call-done(async-update.2), custom_call_target="foo"
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  auto status = verifier().Run(module.get()).status();
  ASSERT_TRUE(status.ok());
}

TEST_F(HloVerifierTestLayoutSensitive,
       AsyncStartAndAsyncUpdateAndAsyncDoneWithThreadName) {
  const char* const hlo_string = R"(
  HloModule Module

  ENTRY AsyncStartAndAsyncUpdateAndAsyncDone {
    p0 = f32[2,3]{1,0:S(1)} parameter(0)
    async-start = ((f32[2,3]{1,0:S(1)}), f32[2,3]{1,0:S(2)}, u32[]) custom-call-start(p0), async_execution_thread="parallel_thread", custom_call_target="foo"
    async-update.1 = ((f32[2,3]{1,0:S(1)}), f32[2,3]{1,0:S(2)}, u32[]) custom-call-update(async-start), async_execution_thread="parallel_thread", custom_call_target="foo"
    async-update.2 = ((f32[2,3]{1,0:S(1)}), f32[2,3]{1,0:S(2)}, u32[]) custom-call-update(async-update.1), async_execution_thread="parallel_thread", custom_call_target="foo"
    ROOT async-done = f32[2,3]{1,0:S(2)} custom-call-done(async-update.2), async_execution_thread="parallel_thread", custom_call_target="foo"
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  auto status = verifier().Run(module.get()).status();
  ASSERT_TRUE(status.ok());
}

TEST_F(HloVerifierTest, AsyncStartAndAsyncDoneWrongType) {
  const char* const hlo_string = R"(
  HloModule Module

  ENTRY AsyncStartAndAsyncDone {
    p0 = f32[2,3] parameter(0)
    async-start = ((f32[2,3]), f32[3,2], u32[]) custom-call-start(p0), custom_call_target="foo"
    ROOT async-done = f32[2,3] custom-call-done(async-start), custom_call_target="foo"
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnUnverifiedModule(hlo_string));

  auto status = verifier().Run(module.get()).status();
  ASSERT_FALSE(status.ok());
  EXPECT_THAT(status.message(),
              HasSubstr("async-done expects the async shape at index {1} to "
                        "match the async computation root shape"));
}

TEST_F(HloVerifierTest, AsyncStartAndAsyncDoneWrongThreadName) {
  const char* const hlo_string = R"(
  HloModule Module

  ENTRY AsyncStartAndAsyncDone {
    p0 = f32[2,3] parameter(0)
    async-start = ((f32[2,3]), f32[2,3], u32[]) custom-call-start(p0), async_execution_thread="parallel_thread", custom_call_target="foo"
    ROOT async-done = f32[2,3] custom-call-done(async-start), async_execution_thread="main_thread", custom_call_target="bar"
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnUnverifiedModule(hlo_string));

  auto status = verifier().Run(module.get()).status();
  ASSERT_FALSE(status.ok());
  EXPECT_THAT(status.message(),
              HasSubstr("thread name (main_thread vs parallel_thread)."));
}

TEST_F(HloVerifierTest, AsyncStartAndAsyncDoneWrongAttr) {
  const char* const hlo_string = R"(
  HloModule Module

  ENTRY AsyncStartAndAsyncDone {
    p0 = f32[2,3] parameter(0)
    async-start = ((f32[2,3]), f32[2,3], u32[]) custom-call-start(p0), custom_call_target="foo"
    ROOT async-done = f32[2,3] custom-call-done(async-start), custom_call_target="bar"
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnUnverifiedModule(hlo_string));

  auto status = verifier().Run(module.get()).status();
  ASSERT_FALSE(status.ok());
  EXPECT_THAT(status.message(),
              HasSubstr("async-done expects its wrapped async computation to "
                        "be identical to its operand's"));
}

TEST_F(HloVerifierTest, AsyncStartMultipleAsyncDone) {
  const char* const hlo_string = R"(
  HloModule Module

  ENTRY AsyncStartAndAsyncDone {
    p0 = f32[2,3] parameter(0)
    async-start = ((f32[2,3]), f32[2,3], u32[]) custom-call-start(p0), custom_call_target="foo"
    async-done.1 = f32[2,3] custom-call-done(async-start), custom_call_target="foo"
    async-done.2 = f32[2,3] custom-call-done(async-start), custom_call_target="foo"
    ROOT tuple = (f32[2,3], f32[2,3]) tuple(async-done.1, async-done.2)
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnUnverifiedModule(hlo_string));

  auto status = verifier().Run(module.get()).status();
  ASSERT_FALSE(status.ok());
  EXPECT_THAT(
      status.message(),
      HasSubstr("async-start instruction requires one consumer, found 2"));
}

TEST_F(HloVerifierTest, AsyncStartNoAsyncDone) {
  const char* const hlo_string = R"(
  HloModule Module

  ENTRY AsyncStartAndAsyncDone {
    p0 = f32[2,3] parameter(0)
    ROOT async-start = ((f32[2,3]), f32[2,3], u32[]) custom-call-start(p0), custom_call_target="foo"
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnUnverifiedModule(hlo_string));

  auto status = verifier().Run(module.get()).status();
  ASSERT_FALSE(status.ok());
  EXPECT_THAT(
      status.message(),
      HasSubstr("async-start instruction requires one consumer, found 0"));
}

TEST_F(HloVerifierTest, AsyncStartAndAsyncUpdateNoAsyncDone) {
  const char* const hlo_string = R"(
  HloModule Module

  ENTRY AsyncStartAndAsyncDone {
    p0 = f32[2,3] parameter(0)
    async-start = ((f32[2,3]), f32[2,3], u32[]) custom-call-start(p0), custom_call_target="foo"
    ROOT async-update = ((f32[2,3]), f32[2,3], u32[]) custom-call-update(async-start), custom_call_target="foo"
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnUnverifiedModule(hlo_string));

  auto status = verifier().Run(module.get()).status();
  ASSERT_FALSE(status.ok());
  EXPECT_THAT(
      status.message(),
      HasSubstr("async-update instruction requires one consumer, found 0"));
}

TEST_F(HloVerifierTest, AsyncDoneNoAsyncStart) {
  const char* const hlo_string = R"(
  HloModule Module

  ENTRY AsyncStartAndAsyncDone {
    p0 = f32[2,3] parameter(0)
    p1 = u32[] parameter(1)
    tuple = ((f32[2,3]), f32[2,3], u32[]) tuple(p0, p0, p1)
    ROOT async-done = f32[2,3] custom-call-done(tuple), custom_call_target="foo"
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnUnverifiedModule(hlo_string));

  auto status = verifier().Run(module.get()).status();
  ASSERT_FALSE(status.ok());
  EXPECT_THAT(status.message(),
              HasSubstr("The operand of a async-done instruction needs to be "
                        "async-start or async-update, found tuple"));
}

TEST_F(HloVerifierTest, AsyncUpdateAndAsyncDoneNoAsyncStart) {
  const char* const hlo_string = R"(
  HloModule Module

  ENTRY AsyncStartAndAsyncDone {
    p0 = f32[2,3] parameter(0)
    p1 = u32[] parameter(1)
    tuple = ((f32[2,3]), f32[2,3], u32[]) tuple(p0, p0, p1)
    async-update = ((f32[2,3]), f32[2,3], u32[]) custom-call-update(tuple), custom_call_target="foo"
    ROOT async-done = f32[2,3] custom-call-done(tuple), custom_call_target="foo"
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnUnverifiedModule(hlo_string));

  auto status = verifier().Run(module.get()).status();
  ASSERT_FALSE(status.ok());
  EXPECT_THAT(status.message(),
              HasSubstr("The operand of a async-update instruction needs to be "
                        "async-start or async-update, found tuple"));
}

TEST_F(HloVerifierTest, AsyncOpComputationParamWrongType) {
  const char* const hlo_string = R"(
  HloModule Module

  async_computation {
    p = f32[2,3] parameter(0)
    ROOT custom-call = f32[3,2] custom-call(p), custom_call_target="foo"
  }

  ENTRY AsyncStartAndAsyncDone {
    p0 = f32[2,3] parameter(0)
    async-start = ((f32[3,2]), f32[3,2], u32[]) async-start(p0), calls=async_computation
    ROOT async-done = f32[3,2] async-done(async-start), calls=async_computation
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnUnverifiedModule(hlo_string));

  auto status = verifier().Run(module.get()).status();
  ASSERT_FALSE(status.ok());
  EXPECT_THAT(status.message(),
              HasSubstr("async-start expects the async shape at index {0} to "
                        "match async computation parameter shape"));
}

TEST_F(HloVerifierTest, AsyncOpComputationRootWrongType) {
  const char* const hlo_string = R"(
  HloModule Module

  async_computation {
    p = f32[2,3] parameter(0)
    ROOT custom-call = f32[3,2] custom-call(p), custom_call_target="foo"
  }

  ENTRY AsyncStartAndAsyncDone {
    p0 = f32[2,3] parameter(0)
    async-start = ((f32[2,3]), f32[2,3], u32[]) async-start(p0), calls=async_computation
    ROOT async-done = f32[3,2] async-done(async-start), calls=async_computation
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnUnverifiedModule(hlo_string));

  auto status = verifier().Run(module.get()).status();
  ASSERT_FALSE(status.ok());
  EXPECT_THAT(status.message(),
              HasSubstr("async-start expects the async shape at index {1} to "
                        "match the async computation root shape"));
}

TEST_F(HloVerifierTest, AsyncOpTupleWrongType) {
  const char* const hlo_string = R"(
  HloModule Module

  async_computation {
    p = f32[2,3] parameter(0)
    ROOT custom-call = f32[3,2] custom-call(p), custom_call_target="foo"
  }

  ENTRY AsyncStartAndAsyncDone {
    p0 = f32[2,3] parameter(0)
    async-start = ((f32[2,3])) async-start(p0), calls=async_computation
    ROOT async-done = f32[3,2] async-done(async-start), calls=async_computation
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnUnverifiedModule(hlo_string));

  auto status = verifier().Run(module.get()).status();
  ASSERT_FALSE(status.ok());
  EXPECT_THAT(status.message(),
              HasSubstr("async-start expects the async shape to be a tuple of "
                        "at least two elements"));
}

TEST_F(HloVerifierTest, AsyncStartOperandWrongType) {
  const char* const hlo_string = R"(
  HloModule Module

  async_computation {
    p = f32[2,3] parameter(0)
    ROOT custom-call = f32[3,2] custom-call(p), custom_call_target="foo"
  }

  ENTRY AsyncStartAndAsyncDone {
    p0 = f32[3,2] parameter(0)
    async-start = ((f32[2,3]), f32[3,2], u32[]) async-start(p0), calls=async_computation
    ROOT async-done = f32[3,2] async-done(async-start), calls=async_computation
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnUnverifiedModule(hlo_string));

  auto status = verifier().Run(module.get()).status();
  ASSERT_FALSE(status.ok());
  EXPECT_THAT(status.message(),
              HasSubstr("async-start expects the shape of operand 0 to match "
                        "the async shape at index {0}"));
}

TEST_F(HloVerifierTest, AsyncDoneOutputWrongType) {
  const char* const hlo_string = R"(
  HloModule Module

  async_computation {
    p = f32[2,3] parameter(0)
    ROOT custom-call = f32[3,2] custom-call(p), custom_call_target="foo"
  }

  ENTRY AsyncStartAndAsyncDone {
    p0 = f32[2,3] parameter(0)
    async-start = ((f32[2,3]), f32[3,2], u32[]) async-start(p0), calls=async_computation
    ROOT async-done = f32[2,3] async-done(async-start), calls=async_computation
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnUnverifiedModule(hlo_string));

  auto status = verifier().Run(module.get()).status();
  ASSERT_FALSE(status.ok());
  EXPECT_THAT(status.message(),
              HasSubstr("async-done expects the shape of output to match the "
                        "async shape at index {1}"));
}

TEST_F(HloVerifierTest, AsyncUpdateWrongType) {
  const char* const hlo_string = R"(
  HloModule Module

  async_computation {
    p = f32[2,3] parameter(0)
    ROOT custom-call = f32[3,2] custom-call(p), custom_call_target="foo"
  }

  ENTRY AsyncStartAndAsyncDone {
    p0 = f32[2,3] parameter(0)
    async-start = ((f32[2,3]), f32[3,2], u32[]) async-start(p0), calls=async_computation
    async-update = ((f32[3,2]), f32[3,2], u32[]) async-update(async-start), calls=async_computation
    ROOT async-done = f32[3,2] async-done(async-update), calls=async_computation
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnUnverifiedModule(hlo_string));

  auto status = verifier().Run(module.get()).status();
  ASSERT_FALSE(status.ok());
  EXPECT_THAT(
      status.message(),
      HasSubstr(
          "async-update expects the shape of operand and output to match"));
}

TEST_F(HloVerifierTestLayoutSensitive, AsyncDoneWrongGroupId) {
  const char* const hlo_string = R"(
  HloModule Module

  ENTRY AsyncStartAndAsyncUpdateAndAsyncDone {
    p0 = f32[2,3]{1,0:S(1)} parameter(0)
    async-start = ((f32[2,3]{1,0:S(1)}), f32[2,3]{1,0:S(2)}, u32[]) custom-call-start(p0), async_group_id=0, custom_call_target="foo"
    async-update.1 = ((f32[2,3]{1,0:S(1)}), f32[2,3]{1,0:S(2)}, u32[]) custom-call-update(async-start), async_group_id=0, custom_call_target="foo"
    async-update.2 = ((f32[2,3]{1,0:S(1)}), f32[2,3]{1,0:S(2)}, u32[]) custom-call-update(async-update.1), async_group_id=0, custom_call_target="foo"
    ROOT async-done = f32[2,3]{1,0:S(2)} custom-call-done(async-update.2), async_group_id=1, custom_call_target="foo"
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnUnverifiedModule(hlo_string));

  auto status = verifier().Run(module.get()).status();
  ASSERT_FALSE(status.ok());
  EXPECT_THAT(status.message(),
              HasSubstr("async-done expects its operand to have the same group "
                        "id (1 vs 0)."));
}

TEST_F(HloVerifierTestLayoutSensitive, AsyncUpdateWrongGroupId) {
  const char* const hlo_string = R"(
  HloModule Module

  ENTRY AsyncStartAndAsyncUpdateAndAsyncDone {
    p0 = f32[2,3]{1,0:S(1)} parameter(0)
    async-start = ((f32[2,3]{1,0:S(1)}), f32[2,3]{1,0:S(2)}, u32[]) custom-call-start(p0), async_group_id=0, custom_call_target="foo"
    async-update.1 = ((f32[2,3]{1,0:S(1)}), f32[2,3]{1,0:S(2)}, u32[]) custom-call-update(async-start), custom_call_target="foo"
    async-update.2 = ((f32[2,3]{1,0:S(1)}), f32[2,3]{1,0:S(2)}, u32[]) custom-call-update(async-update.1), async_group_id=0, custom_call_target="foo"
    ROOT async-done = f32[2,3]{1,0:S(2)} custom-call-done(async-update.2), async_group_id=0, custom_call_target="foo"
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnUnverifiedModule(hlo_string));

  auto status = verifier().Run(module.get()).status();
  ASSERT_FALSE(status.ok());
  EXPECT_THAT(status.message(),
              HasSubstr("async-update expects its operand to have the same "
                        "group id (none vs 0)."));
}

TEST_F(HloVerifierTest, IotaNonArrayResult) {
  const char* const hlo_string = R"(
  HloModule IotaTupleResult

  ENTRY  kernelEntry {
    ROOT iota = () iota(), iota_dimension=24
  }
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnUnverifiedModule(hlo_string));

  auto status = verifier().Run(module.get()).status();
  ASSERT_FALSE(status.ok());
  EXPECT_THAT(status.message(), HasSubstr("does not support non-array result"));
}

TEST_F(HloVerifierTest, IotaNegativeDimension) {
  const char* const hlo_string = R"(
  HloModule IotaTupleResult

  ENTRY  kernelEntry {
    ROOT iota = s32[128,1001]{1,0} iota(), iota_dimension=-1
  }
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnUnverifiedModule(hlo_string));

  auto status = verifier().Run(module.get()).status();
  ASSERT_FALSE(status.ok());
  EXPECT_THAT(status.message(), HasSubstr("negative"));
}

TEST_F(HloVerifierTest, IotaPredResultNotAllowed) {
  const char* const hlo_string = R"(
  HloModule IotaPredResult

  ENTRY  kernelEntry {
    ROOT iota = pred[128] iota(), iota_dimension=0
  }
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnUnverifiedModule(hlo_string));

  auto status = verifier().Run(module.get()).status();
  ASSERT_FALSE(status.ok());
  EXPECT_THAT(status.message(), HasSubstr("got PRED"));
}

static const char* const kMapOperandComputationMismatchHlo = R"(
  HloModule MapOperandComputationMismatch

  Computation {
    param0 = f32[] parameter(0)
    constant = f32[] constant(1)
    ROOT add = f32[] add(param0, constant)
  }

  ENTRY kernelEntry {
  param = f64[] parameter(0)
  ROOT map = f32[] map(param), dimensions={}, to_apply=Computation
})";

TEST_F(HloVerifierTest, MapOperandComputationMismatch) {
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnUnverifiedModule(
                                           kMapOperandComputationMismatchHlo));
  auto status = verifier().Run(module.get()).status();
  ASSERT_FALSE(status.ok());
  EXPECT_THAT(
      status.message(),
      HasSubstr(
          "Shape mismatch between to_apply computation parameter and operand"));
}

TEST_F(HloVerifierTestAllowMixedPrecision, MapOperandComputationMismatch) {
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(
                                           kMapOperandComputationMismatchHlo));
  auto status = verifier().Run(module.get()).status();
  ASSERT_TRUE(status.ok());
}

static const char* const kReduceOperandComputationMismatchHlo = R"(
  HloModule ReduceOperandComputationMismatch
  computation {
    x = f32[] parameter(0)
    y = f32[] parameter(1)
    ROOT add = f32[] add(x, y)
  }

  ENTRY kernelEntry {
    arg0 = f16[64,64,224,224]{3,2,1,0} parameter(0)
    constant = f16[] constant(0)
    reduce = f16[64]{0} reduce(arg0, constant), dimensions={0,2,3}, to_apply=computation
  })";

TEST_F(HloVerifierTest, ReduceOperandComputationMismatch) {
  TF_ASSERT_OK_AND_ASSIGN(
      auto module,
      ParseAndReturnUnverifiedModule(kReduceOperandComputationMismatchHlo));
  auto status = verifier().Run(module.get()).status();
  ASSERT_FALSE(status.ok());
  EXPECT_THAT(status.message(),
              HasSubstr("Expected instruction to have shape equal to f32[64]"));
}

TEST_F(HloVerifierTestAllowMixedPrecision, ReduceOperandComputationMismatch) {
  TF_ASSERT_OK_AND_ASSIGN(
      auto module,
      ParseAndReturnVerifiedModule(kReduceOperandComputationMismatchHlo));
  auto status = verifier().Run(module.get()).status();
  ASSERT_TRUE(status.ok());
}

std::string ReplicaGroupsStr(std::vector<std::vector<int64_t>> replica_groups) {
  std::vector<std::string> replica_group_strs;
  replica_group_strs.reserve(replica_groups.size());
  for (const auto& g : replica_groups) {
    replica_group_strs.push_back(
        absl::StrFormat("{%s}", absl::StrJoin(g, ",")));
  }
  return absl::StrFormat("{%s}", absl::StrJoin(replica_group_strs, ", "));
}

int64_t ReplicaCount(const std::vector<std::vector<int64_t>>& replica_groups) {
  int64_t replica_count = 0;
  for (auto group : replica_groups) {
    replica_count += group.size();
  }
  return replica_count;
}

StatusOr<std::unique_ptr<HloModule>> MakeCollectiveCommOpComputation(
    std::vector<std::vector<int64_t>> replica_groups,
    std::optional<int64_t> replica_count, std::optional<int64_t> num_partitions,
    absl::string_view other_attributes, absl::string_view template_str) {
  HloModuleConfig config;
  config.set_replica_count(
      replica_count.value_or(ReplicaCount(replica_groups)));
  config.set_num_partitions(num_partitions.value_or(1));
  return ParseAndReturnUnverifiedModule(
      absl::StrReplaceAll(
          template_str,
          {{"REPLICA_GROUPS", ReplicaGroupsStr(replica_groups)},
           {"OTHER_ATTRIBUTES", other_attributes.empty()
                                    ? ""
                                    : absl::StrCat(",", other_attributes)}}),
      config);
}

StatusOr<std::unique_ptr<HloModule>> MakeAllReduceComputation(
    std::vector<std::vector<int64_t>> replica_groups,
    std::optional<int64_t> replica_count = std::nullopt,
    std::optional<int64_t> num_partitions = std::nullopt,
    absl::string_view other_attributes = "") {
  const char* kTemplate = R"(
  HloModule test
  add {
    x = f32[] parameter(0)
    y = f32[] parameter(1)
    ROOT add = f32[] add(x, y)
  }
  ENTRY entry {
    p = f32[128]{0} parameter(0)
    crs = f32[128]{0} all-reduce(p), to_apply=add, replica_groups=REPLICA_GROUPS
                                     OTHER_ATTRIBUTES
  })";
  return MakeCollectiveCommOpComputation(replica_groups, replica_count,
                                         num_partitions, other_attributes,
                                         kTemplate);
}

TEST_F(HloVerifierTest, AllReduce_NoReplicaGroupsOK) {
  TF_ASSERT_OK_AND_ASSIGN(auto module, MakeAllReduceComputation({}));
  TF_ASSERT_OK(verifier().Run(module.get()).status());
}

TEST_F(HloVerifierTest, AllReduce_DifferentGroupSizesOk) {
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          MakeAllReduceComputation({{0}, {1, 3}, {2}}));
  TF_ASSERT_OK(verifier().Run(module.get()).status());
}

TEST_F(HloVerifierTest, AllReduce_EmptyReplicaGroup) {
  TF_ASSERT_OK_AND_ASSIGN(auto module, MakeAllReduceComputation({{0}, {}}));
  EXPECT_THAT(verifier().Run(module.get()).status().message(),
              HasSubstr("empty replica group"));
}

TEST_F(HloVerifierTest, AllReduce_RepeatedReplicaId) {
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          MakeAllReduceComputation({{0, 1}, {2, 3}, {4, 0}}));
  EXPECT_THAT(verifier().Run(module.get()).status().message(),
              HasSubstr("Replica 0 is repeated"));
}

TEST_F(HloVerifierTest, AllReduce_MissingReplicaId) {
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          MakeAllReduceComputation({{0, 1}, {2, 3}, {5, 6}}));
  EXPECT_THAT(verifier().Run(module.get()).status().message(),
              HasSubstr("Replica 4 is not named"));
}

TEST_F(HloVerifierTest, AllReduce_NotEnougReplicasInGroupConfig) {
  TF_ASSERT_OK_AND_ASSIGN(auto module, MakeAllReduceComputation({{0, 1}}, 8));
  EXPECT_THAT(verifier().Run(module.get()).status().message(),
              HasSubstr("In kCrossReplica mode, replica groups should contain "
                        "8 replicas, but found 2"));
}

TEST_F(HloVerifierTest, AllReduce_TooManyReplicasInGroupConfig) {
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          MakeAllReduceComputation({{0, 1}, {2, 3}}, 2));
  EXPECT_THAT(verifier().Run(module.get()).status().message(),
              HasSubstr("In kCrossReplica mode, replica groups should contain "
                        "2 replicas, but found 4"));
}

TEST_F(HloVerifierTest, AllReduce_CrossReplicaAndPartition_Invalid) {
  TF_ASSERT_OK_AND_ASSIGN(
      auto module,
      MakeAllReduceComputation({{0, 1}, {2, 3}}, 2, 1, "channel_id=1"));
  EXPECT_THAT(
      verifier().Run(module.get()).status().message(),
      HasSubstr(
          "In kCrossReplicaAndPartition mode, replica groups should contain "
          "2 replicas, but found 4"));
}

TEST_F(HloVerifierTest, AllReduce_CrossReplicaAndPartition_Valid) {
  TF_ASSERT_OK_AND_ASSIGN(
      auto module,
      MakeAllReduceComputation({{0, 1}, {2, 3}}, 4, 1, "channel_id=1"));
  TF_ASSERT_OK(verifier().Run(module.get()).status());
}

TEST_F(HloVerifierTest, AllReduce_FlattenedID_Invalid) {
  TF_ASSERT_OK_AND_ASSIGN(
      auto module,
      MakeAllReduceComputation({{0, 1}, {2, 3}}, 1, 2,
                               "channel_id=1, use_global_device_ids=true"));
  EXPECT_THAT(verifier().Run(module.get()).status().message(),
              HasSubstr("In kFlattenedID mode, replica groups should contain "
                        "2 flattened IDs, but found 4"));
}

TEST_F(HloVerifierTest, AllReduce_FlattenedID_Valid) {
  TF_ASSERT_OK_AND_ASSIGN(
      auto module,
      MakeAllReduceComputation({{0, 1}, {2, 3}}, 2, 2,
                               "channel_id=1, use_global_device_ids=true"));
  TF_ASSERT_OK(verifier().Run(module.get()).status());
}

TEST_F(HloVerifierTest, AllReduceStartAndDone) {
  const char* const kModuleStr = R"(
  HloModule test
  add {
    x = f32[] parameter(0)
    y = f32[] parameter(1)
    ROOT add = f32[] add(x, y)
  }
  ENTRY entry {
    p0 = f32[2,3] parameter(0)
    start = f32[2,3] all-reduce-start(p0), to_apply=add
    ROOT done = f32[2,3] all-reduce-done(start)
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnUnverifiedModule(kModuleStr));

  auto status = verifier().Run(module.get()).status();
  ASSERT_TRUE(status.ok());
}

TEST_F(HloVerifierTest, AllReduceStartAndDoneWrongType) {
  const char* const kModuleStr = R"(
  HloModule test
  add {
    x = f32[] parameter(0)
    y = f32[] parameter(1)
    ROOT add = f32[] add(x, y)
  }
  ENTRY entry {
    p0 = f32[2,3] parameter(0)
    start = (f32[2,3], f32[2,3]) all-reduce-start(p0), to_apply=add
    ROOT done = f32[2,3] all-reduce-done(start)
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnUnverifiedModule(kModuleStr));

  auto status = verifier().Run(module.get()).status();
  EXPECT_THAT(status.message(),
              HasSubstr("Expected instruction to have shape equal to "
                        "f32[2,3]"));
}

TEST_F(HloVerifierTest, AllReduceStartAndMultipleDone) {
  const char* const kModuleStr = R"(
  HloModule test
  add {
    x = f32[] parameter(0)
    y = f32[] parameter(1)
    ROOT add = f32[] add(x, y)
  }
  ENTRY entry {
    p0 = f32[2,3] parameter(0)
    start = (f32[2,3], f32[2,3]) all-reduce-start(p0), to_apply=add
    done1 = f32[2,3] all-reduce-done(start)
    ROOT done2 = f32[2,3] all-reduce-done(start)
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnUnverifiedModule(kModuleStr));

  auto status = verifier().Run(module.get()).status();
  ASSERT_FALSE(status.ok());
  EXPECT_THAT(
      status.message(),
      HasSubstr("all-reduce-start instruction requires one consumer, found 2"));
}

TEST_F(HloVerifierTest, AllReduceDoneWithoutStart) {
  const char* const kModuleStr = R"(
  HloModule test
  ENTRY entry {
    p0 = f32[2,3] parameter(0)
    p1 = u32[] parameter(1)
    tuple = (f32[2,3], f32[2,3]) tuple(p0, p0, p1, p1)
    ROOT done = f32[2,3] all-reduce-done(tuple)
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnUnverifiedModule(kModuleStr));

  auto status = verifier().Run(module.get()).status();
  ASSERT_FALSE(status.ok());
  EXPECT_THAT(status.message(),
              HasSubstr("The operand of a all-reduce-done instruction "
                        "needs to be all-reduce-start, found tuple"));
}

StatusOr<std::unique_ptr<HloModule>> MakeAllToAllComputation(
    std::vector<std::vector<int64_t>> replica_groups,
    std::optional<int64_t> replica_count = std::nullopt,
    std::optional<int64_t> num_partitions = std::nullopt,
    absl::string_view other_attributes = "") {
  const char* kTemplate = R"(
  HloModule test
  ENTRY entry {
    p0 = f32[128]{0} parameter(0)
    p1 = f32[128]{0} parameter(1)
    a2a = (f32[128], f32[128]) all-to-all(p0, p1), replica_groups=REPLICA_GROUPS
                                                   OTHER_ATTRIBUTES
  })";
  return MakeCollectiveCommOpComputation(replica_groups, replica_count,
                                         num_partitions, other_attributes,
                                         kTemplate);
}

TEST_F(HloVerifierTest, AllToAll_NoReplicaGroupsOK) {
  TF_ASSERT_OK_AND_ASSIGN(auto module, MakeAllToAllComputation({}, 2));
  TF_ASSERT_OK(verifier().Run(module.get()).status());
}

TEST_F(HloVerifierTest, AllToAll_EmptyReplicaGroup) {
  TF_ASSERT_OK_AND_ASSIGN(auto module, MakeAllToAllComputation({{0, 1}, {}}));
  EXPECT_THAT(verifier().Run(module.get()).status().message(),
              HasSubstr("cannot have an empty replica group"));
}

TEST_F(HloVerifierTest, AllToAll_RepeatedReplicaId) {
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          MakeAllToAllComputation({{0, 1}, {2, 3}, {4, 0}}));
  EXPECT_THAT(verifier().Run(module.get()).status().message(),
              HasSubstr("Replica 0 is repeated"));
}

TEST_F(HloVerifierTest, AllToAll_MissingReplicaId) {
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          MakeAllToAllComputation({{0, 1}, {2, 3}, {5, 6}}));
  EXPECT_THAT(verifier().Run(module.get()).status().message(),
              HasSubstr("Replica 4 is not named"));
}

TEST_F(HloVerifierTest, AllToAll_UniformSizeOfReplicasInGroup) {
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          MakeAllToAllComputation({{0, 1}, {2}, {3, 4}}));
  EXPECT_THAT(verifier().Run(module.get()).status().message(),
              HasSubstr("Replica groups expected to be of uniform size"));
}

TEST_F(HloVerifierTest, AllToAll_CrossPartition_Invalid) {
  TF_ASSERT_OK_AND_ASSIGN(
      auto module,
      MakeAllToAllComputation({{0, 1}, {2, 3}}, 1, 2, "channel_id=1"));
  EXPECT_THAT(verifier().Run(module.get()).status().message(),
              HasSubstr("In kCrossPartition mode, replica groups should "
                        "contain 2 partitions, but found 4"));
}

TEST_F(HloVerifierTest, AllToAll_CrossPartition_Valid) {
  TF_ASSERT_OK_AND_ASSIGN(
      auto module,
      MakeAllToAllComputation({{0, 1}, {2, 3}}, 1, 4, "channel_id=1"));
  TF_ASSERT_OK(verifier().Run(module.get()).status());
}

TEST_F(HloVerifierTest, AllToAll_LayoutConstrained) {
  const char* const kModuleStr = R"(
  HloModule test
  ENTRY entry {
    p0 = f32[128,4]{0,1} parameter(0)
    p1 = f32[128,4]{1,0} parameter(1)
    ROOT a2a = (f32[128,4]{0,1}, f32[128,4]{1,0}) all-to-all(p0, p1),
      replica_groups={{0,1}}
  }
  )";
  HloModuleConfig config;
  config.set_replica_count(2);
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnUnverifiedModule(kModuleStr, config));
  EXPECT_THAT(verifier().Run(module.get()).status().message(),
              HasSubstr("HLO all-to-all has operands with different shapes"));
}

TEST_F(HloVerifierTest, AllToAll_OperandCountMismatchWithReplicaGroupSize) {
  const char* const kModuleStr = R"(
  HloModule test
  ENTRY entry {
    p0 = f32[128,4] parameter(0)
    p1 = f32[128,4] parameter(1)
    ROOT a2a = (f32[128,4], f32[128,4], f32[128,4]) all-to-all(p0, p1, p1),
      replica_groups={{0,1}}
  }
  )";
  HloModuleConfig config;
  config.set_replica_count(2);
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnUnverifiedModule(kModuleStr, config));
  EXPECT_THAT(verifier().Run(module.get()).status().message(),
              HasSubstr("hlo->operand_count() == split_count"));
}

TEST_F(HloVerifierTest, CollectivePermuteSameSourceTwice) {
  const char* const kModuleStr = R"(
  HloModule test
  ENTRY entry {
    p0 = f32[128] parameter(0)
    ROOT permute = f32[128] collective-permute(p0),
      source_target_pairs={{0,1}, {0,2}, {1,0}}
  }
  )";
  HloModuleConfig config;
  config.set_replica_count(3);
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnUnverifiedModule(kModuleStr, config));
  EXPECT_THAT(verifier().Run(module.get()).status().message(),
              HasSubstr("Source 0 appears more than once"));
}

TEST_F(HloVerifierTest, CollectivePermuteSameTargetTwice) {
  const char* const kModuleStr = R"(
  HloModule test
  ENTRY entry {
    p0 = f32[128] parameter(0)
    ROOT permute = f32[128] collective-permute(p0),
      source_target_pairs={{0,2}, {1,2}, {2,0}}
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnUnverifiedModule(kModuleStr));
  EXPECT_THAT(verifier().Run(module.get()).status().message(),
              HasSubstr("Target 2 appears more than once"));
}

TEST_F(HloVerifierTest, CollectivePermuteSameSourceTooManyTimes) {
  const char* const kModuleStr = R"(
  HloModule test
  ENTRY entry {
    replica_id = u32[] replica-id()
    broadcast.0 = u32[2,8,128]{2,1,0:T(2,128)} broadcast(u32[] replica_id), dimensions={}
    constant.1 = u32[] constant(1000)
    broadcast.1 = u32[2,8,128]{2,1,0:T(2,128)} broadcast(u32[] constant.1), dimensions={}
    constant.2 = s32[] constant(0)
    constant.3 = s32[] constant(1)
    tuple.2 = (s32[],s32[],s32[]) tuple(constant.2, constant.2, constant.2)
    tuple.3 = (s32[],s32[],s32[]) tuple(constant.3, constant.2, constant.2)
    tuple.4 = ((s32[],s32[],s32[]), (s32[],s32[],s32[])) tuple((s32[],s32[],s32[]) tuple.2, (s32[],s32[],s32[]) tuple.3)
    ROOT collective-permute = u32[2,8,128]{2,1,0:T(2,128)} collective-permute(u32[2,8,128] broadcast.0, u32[2,8,128] broadcast.1, ((s32[],s32[],s32[]), (s32[],s32[],s32[])) tuple.4, ((s32[],s32[],s32[]), (s32[],s32[],s32[])) tuple.4), source_target_pairs={{0,1},{1,2},{2,3},{3,0},{0,3},{0,2},{2,1},{1,0}}, slice_sizes={{1,8,128},{1,8,128}}
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnUnverifiedModule(kModuleStr));
  EXPECT_THAT(verifier().Run(module.get()).status().message(),
              HasSubstr("Source 0 appears more than 2 times in instruction's "
                        "source-target pairs:"));
}

TEST_F(HloVerifierTest, CollectivePermuteSameTargetTooManyTimes) {
  const char* const kModuleStr = R"(
  HloModule test
  ENTRY entry {
    replica_id = u32[] replica-id()
    broadcast.0 = u32[2,8,128]{2,1,0:T(2,128)} broadcast(u32[] replica_id), dimensions={}
    constant.1 = u32[] constant(1000)
    broadcast.1 = u32[2,8,128]{2,1,0:T(2,128)} broadcast(u32[] constant.1), dimensions={}
    constant.2 = s32[] constant(0)
    constant.3 = s32[] constant(1)
    tuple.2 = (s32[],s32[],s32[]) tuple(constant.2, constant.2, constant.2)
    tuple.3 = (s32[],s32[],s32[]) tuple(constant.3, constant.2, constant.2)
    tuple.4 = ((s32[],s32[],s32[]), (s32[],s32[],s32[])) tuple((s32[],s32[],s32[]) tuple.2, (s32[],s32[],s32[]) tuple.3)
    ROOT collective-permute = u32[2,8,128]{2,1,0:T(2,128)} collective-permute(u32[2,8,128] broadcast.0, u32[2,8,128] broadcast.1, ((s32[],s32[],s32[]), (s32[],s32[],s32[])) tuple.4, ((s32[],s32[],s32[]), (s32[],s32[],s32[])) tuple.4), source_target_pairs={{0,1},{1,2},{2,3},{3,0},{0,3},{3,2},{2,3},{1,0}}, slice_sizes={{1,8,128},{1,8,128}}
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnUnverifiedModule(kModuleStr));
  EXPECT_THAT(verifier().Run(module.get()).status().message(),
              HasSubstr("Target 3 appears more than 2 times in instruction's "
                        "source-target pairs:"));
}

TEST_F(HloVerifierTest, CollectivePermuteUnmatchingSourceTarget) {
  const char* const kModuleStr = R"(
  HloModule test
  ENTRY entry {
    replica_id = u32[] replica-id()
    broadcast.0 = u32[2,8,128]{2,1,0:T(2,128)} broadcast(u32[] replica_id), dimensions={}
    constant.1 = u32[] constant(1000)
    broadcast.1 = u32[2,8,128]{2,1,0:T(2,128)} broadcast(u32[] constant.1), dimensions={}
    broadcast.2 = u32[4,8,128]{2,1,0:T(2,128)} broadcast(u32[] constant.1), dimensions={}
    constant.2 = s32[] constant(0)
    constant.3 = s32[] constant(1)
    tuple.output = (u32[2,8,128]{2,1,0:T(2,128)}, u32[4,8,128]{2,1,0:T(2,128)}) tuple(u32[2,8,128]{2,1,0:T(2,128)} broadcast.1, u32[4,8,128]{2,1,0:T(2,128)} broadcast.2)
    tuple.2 = (s32[],s32[],s32[]) tuple(constant.2, constant.2, constant.2)
    tuple.3 = (s32[],s32[],s32[]) tuple(constant.3, constant.2, constant.2)
    tuple.4 = ((s32[],s32[],s32[]), (s32[],s32[],s32[])) tuple((s32[],s32[],s32[]) tuple.2, (s32[],s32[],s32[]) tuple.3)
    constant.4 = s32[] constant(2)
    tuple.5 = (s32[],s32[],s32[]) tuple(constant.4, constant.2, constant.2)
    tuple.6 = ((s32[],s32[],s32[]), (s32[],s32[],s32[])) tuple((s32[],s32[],s32[]) tuple.2, (s32[],s32[],s32[]) tuple.5)
    tuple.9 = (((s32[],s32[],s32[]), (s32[],s32[],s32[])), ((s32[],s32[],s32[]), (s32[],s32[],s32[]))) tuple(((s32[],s32[],s32[]), (s32[],s32[],s32[])) tuple.4, ((s32[],s32[],s32[]), (s32[],s32[],s32[])) tuple.6)
    ROOT collective-permute.53 = (u32[2,8,128]{2,1,0:T(2,128)}, u32[4,8,128]{2,1,0:T(2,128)}) collective-permute(u32[2,8,128]{2,1,0:T(2,128)} broadcast.0, (u32[2,8,128]{2,1,0:T(2,128)}, u32[4,8,128]{2,1,0:T(2,128)}) tuple.output, ((s32[],s32[],s32[]), (s32[],s32[],s32[])) tuple.4, (((s32[],s32[],s32[]), (s32[],s32[],s32[])), ((s32[],s32[],s32[]), (s32[],s32[],s32[]))) tuple.9), source_target_pairs={{0,1},{1,2},{2,3},{3,0},{0,3},{3,2},{2,1},{1,0}}, slice_sizes={{1,8,128},{1,8,128},{2,8,128},{2,8,128}}
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnUnverifiedModule(kModuleStr));
  EXPECT_THAT(verifier().Run(module.get()).status().message(),
              HasSubstr("Unmatching input buffers and output buffers"));
}

TEST_F(HloVerifierTest, CollectivePermuteUnmatchingInputAndInputOffset) {
  const char* const kModuleStr = R"(
  HloModule test
  ENTRY entry {
    replica_id = u32[] replica-id()
    broadcast.0 = u32[2,8,128]{2,1,0:T(2,128)} broadcast(u32[] replica_id), dimensions={}
    constant.1 = u32[] constant(1000)
    broadcast.1 = u32[2,8,128]{2,1,0:T(2,128)} broadcast(u32[] constant.1), dimensions={}
    broadcast.2 = u32[4,8,128]{2,1,0:T(2,128)} broadcast(u32[] constant.1), dimensions={}
    constant.2 = s32[] constant(0)
    constant.3 = s32[] constant(1)
    tuple.input = (u32[2,8,128]{2,1,0:T(2,128)}, u32[2,8,128]{2,1,0:T(2,128)}) tuple(u32[2,8,128]{2,1,0:T(2,128)} broadcast.0, u32[2,8,128]{2,1,0:T(2,128)} broadcast.0)
    tuple.output = (u32[2,8,128]{2,1,0:T(2,128)}, u32[4,8,128]{2,1,0:T(2,128)}) tuple(u32[2,8,128]{2,1,0:T(2,128)} broadcast.1, u32[4,8,128]{2,1,0:T(2,128)} broadcast.2)
    tuple.2 = (s32[],s32[],s32[]) tuple(constant.2, constant.2, constant.2)
    tuple.3 = (s32[],s32[],s32[]) tuple(constant.3, constant.2, constant.2)
    tuple.4 = ((s32[],s32[],s32[]), (s32[],s32[],s32[])) tuple((s32[],s32[],s32[]) tuple.2, (s32[],s32[],s32[]) tuple.3)
    constant.4 = s32[] constant(2)
    tuple.5 = (s32[],s32[],s32[]) tuple(constant.4, constant.2, constant.2)
    tuple.6 = ((s32[],s32[],s32[]), (s32[],s32[],s32[])) tuple((s32[],s32[],s32[]) tuple.2, (s32[],s32[],s32[]) tuple.5)
    tuple.9 = (((s32[],s32[],s32[]), (s32[],s32[],s32[])), ((s32[],s32[],s32[]), (s32[],s32[],s32[]))) tuple(((s32[],s32[],s32[]), (s32[],s32[],s32[])) tuple.4, ((s32[],s32[],s32[]), (s32[],s32[],s32[])) tuple.6)
    ROOT collective-permute.53 = (u32[2,8,128]{2,1,0:T(2,128)}, u32[4,8,128]{2,1,0:T(2,128)}) collective-permute((u32[2,8,128]{2,1,0:T(2,128)}, u32[2,8,128]{2,1,0:T(2,128)}) tuple.input, (u32[2,8,128]{2,1,0:T(2,128)}, u32[4,8,128]{2,1,0:T(2,128)}) tuple.output, (s32[],s32[],s32[]) tuple.3, (((s32[],s32[],s32[]), (s32[],s32[],s32[])), ((s32[],s32[],s32[]), (s32[],s32[],s32[]))) tuple.9), source_target_pairs={{0,1},{1,2},{2,3},{3,0},{0,3},{3,2},{2,1},{1,0}}, slice_sizes={{1,8,128},{1,8,128},{2,8,128},{2,8,128}}
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnUnverifiedModule(kModuleStr));
  EXPECT_THAT(verifier().Run(module.get()).status().message(),
              HasSubstr("Unmatching input buffers and input offset."));
}

TEST_F(HloVerifierTest, CollectivePermuteUnmatchingOutputAndOutputOffset) {
  const char* const kModuleStr = R"(
  HloModule test
  ENTRY entry {
    replica_id = u32[] replica-id()
      broadcast.0 = u32[2,8,128]{2,1,0:T(2,128)} broadcast(u32[] replica_id), dimensions={}
      constant.1 = u32[] constant(1000)
      broadcast.1 = u32[2,8,128]{2,1,0:T(2,128)} broadcast(u32[] constant.1), dimensions={}
      broadcast.2 = u32[4,8,128]{2,1,0:T(2,128)} broadcast(u32[] constant.1), dimensions={}
      constant.2 = s32[] constant(0)
      constant.3 = s32[] constant(1)
      tuple.input = (u32[2,8,128]{2,1,0:T(2,128)}, u32[2,8,128]{2,1,0:T(2,128)}) tuple(u32[2,8,128]{2,1,0:T(2,128)} broadcast.0, u32[2,8,128]{2,1,0:T(2,128)} broadcast.0)
      tuple.output = (u32[2,8,128]{2,1,0:T(2,128)}, u32[4,8,128]{2,1,0:T(2,128)}) tuple(u32[2,8,128]{2,1,0:T(2,128)} broadcast.1, u32[4,8,128]{2,1,0:T(2,128)} broadcast.2)
      tuple.2 = (s32[],s32[],s32[]) tuple(constant.2, constant.2, constant.2)
      tuple.3 = (s32[],s32[],s32[]) tuple(constant.3, constant.2, constant.2)
      tuple.4 = ((s32[],s32[],s32[]), (s32[],s32[],s32[])) tuple((s32[],s32[],s32[]) tuple.2, (s32[],s32[],s32[]) tuple.3)
      constant.4 = s32[] constant(2)
      tuple.5 = (s32[],s32[],s32[]) tuple(constant.4, constant.2, constant.2)
      tuple.7 = ((s32[],s32[],s32[]), (s32[],s32[],s32[])) tuple((s32[],s32[],s32[]) tuple.2, (s32[],s32[],s32[]) tuple.2)
      tuple.8 = (((s32[],s32[],s32[]), (s32[],s32[],s32[])), ((s32[],s32[],s32[]), (s32[],s32[],s32[]))) tuple(((s32[],s32[],s32[]), (s32[],s32[],s32[])) tuple.4, ((s32[],s32[],s32[]), (s32[],s32[],s32[])) tuple.7)
      ROOT collective-permute.53 = (u32[2,8,128]{2,1,0:T(2,128)}, u32[4,8,128]{2,1,0:T(2,128)}) collective-permute((u32[2,8,128]{2,1,0:T(2,128)}, u32[2,8,128]{2,1,0:T(2,128)}) tuple.input, (u32[2,8,128]{2,1,0:T(2,128)}, u32[4,8,128]{2,1,0:T(2,128)}) tuple.output, (((s32[],s32[],s32[]), (s32[],s32[],s32[])), ((s32[],s32[],s32[]), (s32[],s32[],s32[]))) tuple.8, (s32[],s32[],s32[]) tuple.2), source_target_pairs={{0,1},{1,2},{2,3},{3,0},{0,3},{3,2},{2,1},{1,0}}, slice_sizes={{1,8,128},{1,8,128},{2,8,128},{2,8,128}}
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnUnverifiedModule(kModuleStr));
  EXPECT_THAT(verifier().Run(module.get()).status().message(),
              HasSubstr("Unmatching output buffers and output offset."));
}

TEST_F(HloVerifierTest, CollectivePermuteCrossReplicaSourceOOR) {
  const char* const kModuleStr = R"(
  HloModule test
  ENTRY entry {
    p0 = f32[128] parameter(0)
    ROOT permute = f32[128] collective-permute(p0),
      source_target_pairs={{5,2}, {1,2}, {2,0}}
  }
  )";
  HloModuleConfig config;
  config.set_replica_count(3);
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnUnverifiedModule(kModuleStr, config));
  const std::string error_message(
      verifier().Run(module.get()).status().message());
  EXPECT_THAT(error_message, HasSubstr("Source 5"));
  EXPECT_THAT(error_message, HasSubstr("must be < 3"));
}

TEST_F(HloVerifierTest, CollectivePermuteCrossReplicaTargetOOR) {
  const char* const kModuleStr = R"(
  HloModule test
  ENTRY entry {
    p0 = f32[128] parameter(0)
    ROOT permute = f32[128] collective-permute(p0),
      source_target_pairs={{0,1}, {1,2}, {2,7}}
  }
  )";
  HloModuleConfig config;
  config.set_replica_count(3);
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnUnverifiedModule(kModuleStr, config));
  const std::string error_message(
      verifier().Run(module.get()).status().message());
  EXPECT_THAT(error_message, HasSubstr("Target 7"));
  EXPECT_THAT(error_message, HasSubstr("must be < 3"));
}

TEST_F(HloVerifierTest, CollectivePermuteCrossPartitionSourceOOR) {
  const char* const kModuleStr = R"(
  HloModule test
  ENTRY entry {
    p0 = f32[128] parameter(0)
    ROOT permute = f32[128] collective-permute(p0),
      source_target_pairs={{5,2}, {1,2}, {2,0}}, channel_id=1
  }
  )";
  HloModuleConfig config;
  config.set_num_partitions(3);
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnUnverifiedModule(kModuleStr, config));
  const std::string error_message(
      verifier().Run(module.get()).status().message());
  EXPECT_THAT(error_message, HasSubstr("Source 5"));
  EXPECT_THAT(error_message, HasSubstr("must be < 3"));
}

TEST_F(HloVerifierTest, CollectivePermuteCrossPartitionTargetOOR) {
  const char* const kModuleStr = R"(
  HloModule test
  ENTRY entry {
    p0 = f32[128] parameter(0)
    ROOT permute = f32[128] collective-permute(p0),
      source_target_pairs={{0,2}, {1,7}, {2,0}}, channel_id=1
  }
  )";
  HloModuleConfig config;
  config.set_num_partitions(3);
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnUnverifiedModule(kModuleStr, config));
  const std::string error_message(
      verifier().Run(module.get()).status().message());
  EXPECT_THAT(error_message, HasSubstr("Target 7"));
  EXPECT_THAT(error_message, HasSubstr("must be < 3"));
}

TEST_F(HloVerifierTest, FusionShapeVerifier) {
  const char* const kModuleStr = R"(
  HloModule test

  fused_computation {
    ROOT p0 = f32[10,10] parameter(0)
  }

  ENTRY entry {
    p0 = f32[10,10] parameter(0)
    ROOT out = f32[10] fusion(p0), kind=kInput, calls=fused_computation
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnUnverifiedModule(kModuleStr));
  EXPECT_THAT(verifier().Run(module.get()).status().message(),
              HasSubstr("Fused computation shape"));
}

TEST_F(HloVerifierTest, FusionThreadVerifier) {
  const char* const kModuleStr = R"(
  HloModule test

  fused_computation {
    ROOT p0 = f32[8,12] parameter(0)
  }, execution_thread="parallel_thread"

  ENTRY entry {
    p0 = f32[8,12] parameter(0)
    ROOT out = f32[8,12] fusion(p0), kind=kInput, calls=fused_computation
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnUnverifiedModule(kModuleStr));
  EXPECT_THAT(verifier().Run(module.get()).status().message(),
              HasSubstr("expects parent computation thread name same as called "
                        "computation's thread name"));
}

TEST_F(HloVerifierTest, FusionNestedComputationThreadVerifier) {
  const char* const kModuleStr = R"(
  HloModule test

  add {
    lhs = f32[] parameter(0)
    rhs = f32[] parameter(1)
    ROOT add = f32[] add(lhs, rhs)
  }, execution_thread="parallel_thread"

  fused_computation {
    p0 = f32[8,12] parameter(0)
    p1 = f32[8,12] parameter(1)
    crs0 = f32[8,12] all-reduce(p1), replica_groups={}, to_apply=add
    ROOT result = add(p0, crs0)
  }

  ENTRY entry {
    p0 = f32[8,12] parameter(0)
    p1 = f32[8,12] parameter(1)
    ROOT out = f32[8,12] fusion(p0, p1), kind=kInput, calls=fused_computation
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnUnverifiedModule(kModuleStr));
  EXPECT_THAT(
      verifier().Run(module.get()).status().message(),
      HasSubstr("Nested computations expects same computation's thread name"));
}

TEST_F(HloVerifierTest, AllReduceVerifier) {
  const char* const kModuleStr = R"(
  HloModule test

  add {
    lhs = f32[] parameter(0)
    rhs = f32[] parameter(1)
    ROOT add = f32[] add(lhs, rhs)
  }

  ENTRY entry {
    input = f32[8,12]{0,1} parameter(0)
    crs0 = f32[8,12]{0,1} all-reduce(input), replica_groups={}, to_apply=add
    crs1 = f32[8,12]{0,1} all-reduce(input), replica_groups={}, to_apply=add,
      constrain_layout=true
    ROOT result = (f32[8,12]{0,1}, f32[8,12]{0,1}) tuple(crs0, crs1)
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnUnverifiedModule(kModuleStr));
  EXPECT_THAT(
      verifier().Run(module.get()).status().message(),
      HasSubstr("mix of layout constrained and unconstrained AllReduce"));
}

TEST_F(HloVerifierTest, ChannelVerifier) {
  const char* const kModuleStr = R"(
  HloModule test

  add {
    lhs = f32[] parameter(0)
    rhs = f32[] parameter(1)
    ROOT add = f32[] add(lhs, rhs)
  }

  ENTRY entry {
    %input = f32[8,12] parameter(0)
    %token0 = token[] after-all()
    %send = (f32[8,12], u32[], token[]) send(%input, %token0), channel_id=1
    %send-done = token[] send-done(%send), channel_id=1
    %crs = f32[8,12] all-reduce(%input), replica_groups={}, to_apply=add,
      channel_id=1
    ROOT result = (f32[8,12]{0,1}, f32[8,12]{0,1}) tuple(%input, %crs)
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnUnverifiedModule(kModuleStr));
  EXPECT_THAT(verifier().Run(module.get()).status().message(),
              HasSubstr("used for different types of channel instructions"));
}

TEST_F(HloVerifierTest, CollectiveChannelVerifier) {
  const char* const kModuleStr = R"(
  HloModule test

  add {
    lhs = f32[] parameter(0)
    rhs = f32[] parameter(1)
    ROOT add = f32[] add(lhs, rhs)
  }

  ENTRY entry {
    %input = f32[8,12] parameter(0)
    %permute = f32[8,12] collective-permute(%input),
      source_target_pairs={{0,1},{1,0}}, channel_id=1
    %crs = f32[8,12] all-reduce(%input), replica_groups={}, to_apply=add,
      channel_id=1
    ROOT result = (f32[8,12]{0,1}, f32[8,12]{0,1}) tuple(%permute, %crs)
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnUnverifiedModule(kModuleStr));
  EXPECT_THAT(verifier().Run(module.get()).status().message(),
              HasSubstr("used for different types of channel instructions"));
}

TEST_F(HloVerifierTestLayoutSensitive, CollectivePermuteStartAndDone) {
  const char* const kModuleStr = R"(
  HloModule Module

  ENTRY CollectivePermuteStartAndDone {
    p0 = f32[2,3]{1,0:S(1)} parameter(0)
    collective-permute-start.1 = (f32[2,3]{1,0:S(1)}, f32[2,3]{1,0:S(1)}, u32[], u32[]) collective-permute-start(p0), source_target_pairs={{0,1},{1,0}}, channel_id=1
    ROOT collective-permute-done.1 = f32[2,3]{1,0:S(1)} collective-permute-done(collective-permute-start.1)
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnUnverifiedModule(kModuleStr));

  auto status = verifier().Run(module.get()).status();
  ASSERT_TRUE(status.ok());
}

TEST_F(HloVerifierTest, CollectivePermuteStartAndDoneWrongType) {
  const char* const kModuleStr = R"(
  HloModule Module

  ENTRY CollectivePermuteStartAndDoneWrongType {
    p0 = f32[2,3]{1,0:S(1)} parameter(0)
    collective-permute-start.1 = f32[2,3]{1,0:S(1)} collective-permute-start(p0), source_target_pairs={{0,1},{1,0}}, channel_id=1
    ROOT collective-permute-done.1 = f32[2,3]{1,0:S(1)} collective-permute-done(collective-permute-start.1)
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnUnverifiedModule(kModuleStr));

  auto status = verifier().Run(module.get()).status();
  ASSERT_FALSE(status.ok());
  EXPECT_THAT(status.message(),
              HasSubstr("Expected instruction to have shape equal to "
                        "(f32[2,3], f32[2,3], u32[], u32[])"));
}

TEST_F(HloVerifierTest, CollectivePermuteStartAndMultipleDone) {
  const char* const kModuleStr = R"(
  HloModule Module

  ENTRY CollectivePermuteStartAndMultipleDone {
    p0 = f32[2,3]{1,0:S(1)} parameter(0)
    collective-permute-start.1 = (f32[2,3]{1,0:S(1)}, f32[2,3]{1,0:S(1)}, u32[], u32[]) collective-permute-start(p0), source_target_pairs={{0,1},{1,0}}, channel_id=1
    collective-permute-done.1 = f32[2,3]{1,0:S(1)} collective-permute-done(collective-permute-start.1)
    ROOT collective-permute-done.2 = f32[2,3]{1,0:S(1)} collective-permute-done(collective-permute-start.1)
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnUnverifiedModule(kModuleStr));

  auto status = verifier().Run(module.get()).status();
  ASSERT_FALSE(status.ok());
  EXPECT_THAT(
      status.message(),
      HasSubstr("collective-permute-start instruction requires one consumer, "
                "found 2"));
}

TEST_F(HloVerifierTest, CollectivePermuteDoneNoCollectivePermuteStart) {
  const char* const kModuleStr = R"(
  HloModule Module

  ENTRY CollectivePermuteDoneNoCollectivePermuteStart {
    p0 = f32[2,3]{1,0:S(1)} parameter(0)
    p1 = f32[2,3]{1,0:S(1)} parameter(1)
    p2 = u32[] parameter(2)
    p3 = u32[] parameter(3)
    tuple.1 = (f32[2,3], f32[2,3], u32[], u32[]) tuple(p0, p1, p2, p3)
    ROOT collective-permute-done.1 = f32[2,3]{1,0:S(1)} collective-permute-done(tuple.1)
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnUnverifiedModule(kModuleStr));

  auto status = verifier().Run(module.get()).status();
  ASSERT_FALSE(status.ok());
  EXPECT_THAT(status.message(),
              HasSubstr("The operand of a collective-permute-done instruction "
                        "needs to be collective-permute-start, found tuple"));
}

TEST_F(HloVerifierTest, ComparisonTypeFloat) {
  const char* const hlo_string = R"(
  HloModule Module

  ENTRY RngOperandElementTypesNotMatch {
   p0 = f32[] parameter(0)
   ROOT cmp = pred[] compare(f32[] p0, f32[] p0), direction=LT, type=UNSIGNED
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnUnverifiedModule(hlo_string));

  auto status = verifier().Run(module.get()).status();
  ASSERT_FALSE(status.ok());
  EXPECT_THAT(status.message(),
              HasSubstr("Expected comparison type FLOAT or TOTALORDER"));
}

TEST_F(HloVerifierTest, ComparisonTypeSigned) {
  const char* const hlo_string = R"(
  HloModule Module

  ENTRY RngOperandElementTypesNotMatch {
   p0 = s32[] parameter(0)
   ROOT cmp = pred[] compare(s32[] p0, s32[] p0), direction=LT, type=UNSIGNED
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnUnverifiedModule(hlo_string));

  auto status = verifier().Run(module.get()).status();
  ASSERT_FALSE(status.ok());
  EXPECT_THAT(status.message(), HasSubstr("Expected comparison type SIGNED"));
}

TEST_F(HloVerifierTest, ComparisonTypeUnsigned) {
  const char* const hlo_string = R"(
  HloModule Module

  ENTRY RngOperandElementTypesNotMatch {
   p0 = u32[] parameter(0)
   ROOT cmp = pred[] compare(u32[] p0, u32[] p0), direction=LT, type=SIGNED
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnUnverifiedModule(hlo_string));

  auto status = verifier().Run(module.get()).status();
  ASSERT_FALSE(status.ok());
  EXPECT_THAT(status.message(), HasSubstr("Expected comparison type UNSIGNED"));
}

TEST_F(HloVerifierTest, ComparisonTypePred) {
  const char* const hlo_string = R"(
  HloModule Module

  ENTRY RngOperandElementTypesNotMatch {
   p0 = pred[] parameter(0)
   ROOT cmp = pred[] compare(pred[] p0, pred[] p0), direction=LT, type=SIGNED
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnUnverifiedModule(hlo_string));

  auto status = verifier().Run(module.get()).status();
  ASSERT_FALSE(status.ok());
  EXPECT_THAT(status.message(), HasSubstr("Expected comparison type UNSIGNED"));
}

TEST_F(HloVerifierTest, UseGlobalDeviceIdsEmptyReplicaGroup) {
  const char* const hlo_string = R"(
  HloModule Module
  add {
    lhs = f32[] parameter(0)
    rhs = f32[] parameter(1)
    ROOT add = f32[] add(lhs, rhs)
  }

  ENTRY CRS {
    input = f32[8]{0} parameter(0)
    ROOT crs = f32[8]{0} all-reduce(input), replica_groups={}, channel_id=1,
                         use_global_device_ids=true, to_apply=add
  })";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnUnverifiedModule(hlo_string));

  auto status = verifier().Run(module.get()).status();
  ASSERT_FALSE(status.ok());
  EXPECT_THAT(
      status.message(),
      HasSubstr("Replica groups must be specified in flattened-id mode"));
}

TEST_F(HloVerifierTest, InvalidChannelIDandUseGlobalDeviceIDs) {
  const char* const hlo_string = R"(
  HloModule Module
  add {
    lhs = f32[] parameter(0)
    rhs = f32[] parameter(1)
    ROOT add = f32[] add(lhs, rhs)
  }

  ENTRY CRS {
    input = f32[8]{0} parameter(0)
    ROOT crs = f32[8]{0} all-reduce(input), replica_groups={},
                         use_global_device_ids=true, to_apply=add
  })";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnUnverifiedModule(hlo_string));

  auto status = verifier().Run(module.get()).status();
  ASSERT_FALSE(status.ok());
  EXPECT_THAT(
      status.message(),
      HasSubstr(
          "Invalid combination of has_channel_id and use_global_device_ids"));
}

TEST_F(HloVerifierTest, ReduceScatterInvalidOutputSize0) {
  const char* const hlo_string = R"(
  HloModule Module
  add {
    lhs = f32[] parameter(0)
    rhs = f32[] parameter(1)
    ROOT add = f32[] add(lhs, rhs)
  }

  ENTRY CRS {
    input = f32[8]{0} parameter(0)
    ROOT crs = f32[8]{0} reduce-scatter(input), replica_groups={{0,1}},
                         to_apply=add, dimensions={0}
  })";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnUnverifiedModule(hlo_string));

  auto status = verifier().Run(module.get()).status();
  ASSERT_FALSE(status.ok());
  EXPECT_THAT(status.message(),
              HasSubstr("shard_count = 1, subgroup_size = 2"));
}

TEST_F(HloVerifierTest, ReduceScatterInvalidScatterDim) {
  const char* const hlo_string = R"(
  HloModule Module
  add {
    lhs = f32[] parameter(0)
    rhs = f32[] parameter(1)
    ROOT add = f32[] add(lhs, rhs)
  }

  ENTRY CRS {
    input = f32[8]{0} parameter(0)
    ROOT crs = f32[4]{0} reduce-scatter(input), replica_groups={{0,1}},
                         to_apply=add, dimensions={1}
  })";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnUnverifiedModule(hlo_string));

  auto status = verifier().Run(module.get()).status();
  ASSERT_FALSE(status.ok());
  EXPECT_THAT(
      status.message(),
      HasSubstr("ars->scatter_dimension() < ars->operand(i)->shape().rank()"));
}

TEST_F(HloVerifierTest, ReduceScatterNonUniformGroups) {
  const char* const hlo_string = R"(
  HloModule Module
  add {
    lhs = f32[] parameter(0)
    rhs = f32[] parameter(1)
    ROOT add = f32[] add(lhs, rhs)
  }

  ENTRY CRS {
    input = f32[8]{0} parameter(0)
    ROOT crs = f32[4]{0} reduce-scatter(input), replica_groups={{0,1}, {2,3,4}},
                         to_apply=add, dimensions={0}
  })";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnUnverifiedModule(hlo_string));

  auto status = verifier().Run(module.get()).status();
  ASSERT_FALSE(status.ok());
  EXPECT_THAT(status.message(),
              HasSubstr("Replica groups expected to be of uniform size"));
}

TEST_F(HloVerifierTest, VerifyBroadcastDimensionsOrder) {
  const char* const hlo = R"(
HloModule module

ENTRY computation {
  mul = f32[32,32,32]{2,1,0} parameter(0)
  ROOT broadcast = f32[32,32,32,32]{3,2,1,0} broadcast(mul), dimensions={3,2,1}
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnUnverifiedModule(hlo));
  auto status = HloVerifier{HloVerifierOpts{}.VerifyBroadcastDimensionsOrder()}
                    .Run(module.get())
                    .status();
  ASSERT_FALSE(status.ok());
  EXPECT_THAT(status.message(),
              HasSubstr("Broadcast dimensions should be ordered"));
}

TEST_F(HloVerifierTest, VerifyBroadcastDimensionsOrderOK) {
  const char* const hlo = R"(
HloModule module

ENTRY computation {
  mul = f32[4,5] parameter(0)
  ROOT broadcast = f32[4,3,2,5] broadcast(mul), dimensions={0,3}
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnUnverifiedModule(hlo));
  TF_ASSERT_OK(HloVerifier{HloVerifierOpts{}.VerifyBroadcastDimensionsOrder()}
                   .Run(module.get())
                   .status());
}

TEST_F(HloVerifierTest, ReshapeIsNotBitcast) {
  const char* const hlo = R"(
HloModule Module

ENTRY main {
  p = f32[8,3]{1,0} parameter(0)
  ROOT r = f32[4,2,3]{0,1,2} reshape(p)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnUnverifiedModule(hlo));
  auto status =
      HloVerifier{
          HloVerifierOpts{}.MakeLayoutSensitive().VerifyReshapeIsBitcast()}
          .Run(module.get())
          .status();
  ASSERT_FALSE(status.ok());
  EXPECT_THAT(status.message(),
              HasSubstr("Reshape should be a physical bitcast"));
}

TEST_F(HloVerifierTest, ReshapeIsBitcast) {
  const char* const hlo = R"(
HloModule Module

ENTRY main {
  p = f32[8]{0} parameter(0)
  ROOT r = f32[4,2]{1,0} reshape(p)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnUnverifiedModule(hlo));
  TF_ASSERT_OK(HloVerifier{
      HloVerifierOpts{}.MakeLayoutSensitive().VerifyReshapeIsBitcast()}
                   .Run(module.get())
                   .status());
}

TEST_F(HloVerifierTest, VerifyCustomCallThread) {
  const char* const hlo = R"(
    HloModule module
    %call_body (prev.2: s32[]) -> pred[] {
      %constant.1 = s32[] constant(5)
      %prev.2 = s32[] parameter(0)
      ROOT %greater-than = pred[] compare(s32[] %constant.1, s32[] %prev.2), direction=GT
    }, execution_thread="parallel_thread"

    ENTRY %WhileWithScalarS32Result.v2 () -> s32[] {
      %constant.2 = s32[] constant(0)
      ROOT %custom = s32[] custom-call(s32[] %constant.2), custom_call_target="MyCustomCall", to_apply=%call_body
    }
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnUnverifiedModule(hlo));
  auto status =
      HloVerifier{
          HloVerifierOpts{}.VerifyCustomCallNestedComputationThreadName()}
          .Run(module.get())
          .status();
  ASSERT_FALSE(status.ok());
  EXPECT_THAT(status.message(),
              HasSubstr("expects parent computation thread name same as called "
                        "computation's thread name"));
}

TEST_F(HloVerifierTest, CheckWhileThread) {
  const char* const hlo_string = R"(
    HloModule While, entry_computation_layout={()->s32[]}

    %body.v3 (prev.1: s32[]) -> s32[] {
      %constant = s32[] constant(1)
      %prev.1 = s32[] parameter(0)
      ROOT %add = s32[] add(s32[] %constant, s32[] %prev.1)
    }

    %condition.v3 (prev.2: s32[]) -> pred[] {
      %constant.1 = s32[] constant(5)
      %prev.2 = s32[] parameter(0)
      ROOT %greater-than = pred[] compare(s32[] %constant.1, s32[] %prev.2), direction=GT
    }, execution_thread="parallel_thread"

    ENTRY %WhileWithScalarS32Result.v2 () -> s32[] {
      %constant.2 = s32[] constant(0)
      ROOT %while = s32[] while(s32[] %constant.2), condition=%condition.v3, body=%body.v3
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnUnverifiedModule(hlo_string));
  auto status = verifier().Run(module.get()).status();
  ASSERT_FALSE(status.ok());
  EXPECT_THAT(status.message(),
              HasSubstr("expects parent computation thread name same as called "
                        "computation's thread name"));
}

TEST_F(HloVerifierTest, CheckWhileContainsAsyncThread) {
  const char* const hlo_string = R"(
    HloModule While, entry_computation_layout={()->s32[]}

    %async_add (prev.1: s32[]) -> s32[] {
      %constant = s32[] constant(1)
      %prev.1 = s32[] parameter(0)
      ROOT %add = s32[] add(s32[] %constant, s32[] %prev.1)
    }, execution_thread="parallel_thread"

    %body.v3 (prev.1: s32[]) -> s32[] {
      %constant = s32[] constant(1)
      %prev.1 = s32[] parameter(0)
      ROOT %add = s32[] add(s32[] %constant, s32[] %prev.1)
    }

    %condition.v3 (prev.2: s32[]) -> pred[] {
      %constant.1 = s32[] constant(5)
      %prev.2 = s32[] parameter(0)
      %async-start = ((s32[]), s32[], s32[]) custom-call-start(s32[] %prev.2), async_execution_thread="parallel_thread", custom_call_target="async_add"
      %async-done = s32[] custom-call-done(((s32[]), s32[], s32[]) %async-start), async_execution_thread="parallel_thread", custom_call_target="async_add"
      ROOT %greater-than = pred[] compare(s32[] %constant.1, s32[] %async-done), direction=GT
    }

    ENTRY %WhileWithScalarS32Result.v2 () -> s32[] {
      %constant.2 = s32[] constant(0)
      ROOT %while = s32[] while(s32[] %constant.2), condition=%condition.v3, body=%body.v3
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnUnverifiedModule(hlo_string));
  auto status = verifier().Run(module.get()).status();
  ASSERT_TRUE(status.ok());
}

TEST_F(HloVerifierTestLayoutFusion, DynamicUpdateSliceWithMemorySpace) {
  const char* const hlo_string = R"(
HloModule fusion, is_scheduled=true

fused_computation {
  %parameter.0 = bf16[1,8,1,8,320]{4,0,3,2,1:T(2,128)(2,1)S(3)} parameter(0)
  %parameter.1 = bf16[1,8,6,8,320]{4,0,3,2,1:T(2,128)(2,1)S(3)} parameter(1)
  %c = bf16[1,8,6,8,320]{4,0,3,2,1:T(2,128)(2,1)} copy(parameter.1)
  %constant.1 = s32[] constant(0)
  ROOT %dynamic-update-slice.1 = bf16[1,8,6,8,320]{4,0,3,2,1:T(2,128)(2,1)S(3)}
    dynamic-update-slice(%c, %parameter.0, %constant.1, %constant.1,
    %constant.1, %constant.1, %constant.1)
}

ENTRY entry (parameter.0: bf16[1,8,1,8,320], parameter.1: bf16[1,8,6,8,320]) -> bf16[1,8,6,8,320]{
  %p0 = bf16[1,8,1,8,320]{4,0,3,2,1:T(2,128)(2,1)S(3)} parameter(0)
  %p1 = bf16[1,8,6,8,320]{4,0,3,2,1:T(2,128)(2,1)S(3)} parameter(1)
  ROOT out = bf16[1,8,6,8,320]{4,0,3,2,1:T(2,128)(2,1)S(3)} fusion(p0, p1), kind=kLoop, calls=fused_computation
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnUnverifiedModule(hlo_string));

  auto status = verifier().Run(module.get()).status();
  ASSERT_TRUE(status.ok());
}

TEST_F(HloVerifierTest, InvalidShardingRank) {
  const char* const hlo = R"(
HloModule Module

ENTRY main {
  p = f32[4,2] parameter(0), sharding={devices=[1,2,2,1]0,1,2,3}
  ROOT r = f32[4,2] copy(p)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnUnverifiedModule(hlo));

  auto status = verifier().Run(module.get()).status();
  ASSERT_FALSE(status.ok());
  EXPECT_THAT(status.message(),
              HasSubstr("tile assignment dimensions (excluding subgroups) is "
                        "different than the input rank."));
}

TEST_F(HloVerifierTest, InvalidShardingDevices) {
  const char* const hlo = R"(
HloModule Module

ENTRY main {
  p = f32[4,2] parameter(0), sharding={devices=[2,2]0,1,2,3}
  ROOT r = f32[4,2] copy(p)
}
)";

  HloModuleConfig config;
  config.set_num_partitions(2);
  config.set_use_spmd_partitioning(true);
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnUnverifiedModule(hlo, config));
  ASSERT_TRUE(module->config().use_spmd_partitioning());

  auto status = verifier().Run(module.get()).status();
  ASSERT_FALSE(status.ok());
  EXPECT_THAT(status.message(),
              HasSubstr("device 2 > num_devices (2) in tile assignment"));
}

TEST_F(HloVerifierTest, InconsistentWhileSharding) {
  const char* const hlo = R"(
    HloModule While

    %body.v3 (prev.1: s32[]) -> s32[] {
       %prev.1 = s32[] parameter(0), sharding={replicated}
      %constant = s32[] constant(1)
      ROOT %add = s32[] add(s32[] %constant, s32[] %prev.1)
    }

    %condition.v3 (prev.2: s32[]) -> pred[] {
      %prev.2 = s32[] parameter(0), sharding={maximal device=0}
      %constant.1 = s32[] constant(5)
      ROOT %greater-than = pred[] compare(s32[] %constant.1, s32[] %prev.2), direction=GT
    }

    ENTRY %WhileWithScalarS32Result.v2 () -> s32[] {
      %constant.2 = s32[] constant(0)
      ROOT %while = s32[] while(s32[] %constant.2), condition=%condition.v3, body=%body.v3
    }
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnUnverifiedModule(hlo));
  auto status = verifier().Run(module.get()).status();
  ASSERT_FALSE(status.ok());
  EXPECT_THAT(status.message(),
              HasSubstr("Inconsistent while sharding among instructions"));
}

TEST_F(HloVerifierTest, InconsistentConditionSharding) {
  const char* const hlo = R"(
  HloModule Module

  true_branch {
    tparam = (s32[], f32[4]) parameter(0)
    ROOT tgte1 = f32[4] get-tuple-element(tparam), index=1
  }

  false_branch {
    fparam = (s32[], f32[4]) parameter(0)
    ROOT fgte1 = f32[4] get-tuple-element(fparam), index=1, sharding={replicated}
  }

  ENTRY entry {
    p0 = (s32[], f32[4]) parameter(0)
    constant = pred[] constant(true)
    ROOT conditional = f32[4] conditional(constant, p0, p0),
      true_computation=true_branch, false_computation=false_branch,
      sharding={maximal device=0}
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnUnverifiedModule(hlo));
  auto status = verifier().Run(module.get()).status();
  ASSERT_FALSE(status.ok());
  EXPECT_THAT(
      status.message(),
      HasSubstr("Inconsistent conditional sharding among instructions"));
}

TEST_F(HloVerifierTest, InvalidS4Usage) {
  const char* const hlo = R"(
  HloModule Module

  ENTRY entry {
    param0 = s32[] parameter(0)
    x = s4[] convert(param0)
    ROOT add = s4[] add(x, x)
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnUnverifiedModule(hlo));
  auto status = verifier().Run(module.get()).status();
  ASSERT_FALSE(status.ok());
  EXPECT_THAT(
      status.message(),
      HasSubstr("S4/U4 is currently only supported in matmul and convolution"));
}

TEST_F(HloVerifierTest, InvalidU4Usage) {
  const char* const hlo = R"(
  HloModule Module

  ENTRY entry {
    param0 = u32[] parameter(0)
    x = u4[] convert(param0)
    ROOT add = u4[] add(x, x)
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnUnverifiedModule(hlo));
  auto status = verifier().Run(module.get()).status();
  ASSERT_FALSE(status.ok());
  EXPECT_THAT(
      status.message(),
      HasSubstr("S4/U4 is currently only supported in matmul and convolution"));
}

}  // namespace
}  // namespace xla

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

#include "xla/service/cpu/cpu_float_support.h"

#include <memory>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/str_cat.h"
#include "absl/strings/str_replace.h"
#include "absl/strings/string_view.h"
#include "xla/backends/cpu/codegen/target_machine_test_base.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/testlib/verified_hlo_module.h"
#include "xla/hlo/transforms/simplifiers/float_normalization.h"
#include "xla/service/hlo_module_config.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/xla_data.pb.h"

namespace xla::cpu {
namespace {

struct SkipInstructionTestSpec {
  HloOpcode op;
  bool call_library_for_instruction;
  bool upcast;
};

class SkipInstructionTest
    : public TargetMachineTestBase,
      public ::testing::WithParamInterface<SkipInstructionTestSpec> {
 public:
  static std::string Name(
      const ::testing::TestParamInfo<SkipInstructionTestSpec>& info) {
    std::string op =
        absl::StrReplaceAll(HloOpcodeString(info.param.op), {{"-", "_"}});
    absl::string_view dot_strategy =
        info.param.call_library_for_instruction ? "LibDot" : "NoLibDot";
    return absl::StrCat(op, "_", dot_strategy);
  }

  void SetUp() override { TargetMachineTestBase::SetUp(); }

  void CheckDtype(HloModule* module, PrimitiveType lhs_type,
                  PrimitiveType rhs_type, PrimitiveType out_type) {
    HloInstruction* op = module->entry_computation()->root_instruction();
    EXPECT_EQ(op->operand(0)->shape().element_type(), lhs_type);
    EXPECT_EQ(op->operand(1)->shape().element_type(), rhs_type);
    EXPECT_EQ(op->shape().element_type(), out_type);
  }
};

TEST_P(SkipInstructionTest, Bf16InF32Out) {
  SkipInstructionTestSpec spec = GetParam();

  auto module = std::make_unique<VerifiedHloModule>(
      "test", HloModuleConfig(),
      /*verifier_layout_sensitive=*/false,
      /*allow_mixed_precision_in_hlo_verifier=*/true,
      ShapeUtil::ByteSizeOfElements);

  // Create the HLO module: p0 <op> p1.
  HloComputation::Builder builder("SkipInstructionTest");
  Shape input_shape = ShapeUtil::MakeShape(BF16, {100, 100});
  Shape output_shape = ShapeUtil::MakeShape(F32, {100, 100});
  HloInstruction* p0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, input_shape, "p0"));

  HloInstruction* p1;
  Shape scalar_shape = ShapeUtil::MakeShape(BF16, {});
  if (spec.op == HloOpcode::kReduce || spec.op == HloOpcode::kReduceWindow) {
    p1 = builder.AddInstruction(
        HloInstruction::CreateParameter(1, scalar_shape, "p1"));
    output_shape = (spec.op == HloOpcode::kReduce)
                       ? ShapeUtil::MakeShape(F32, {100})
                       : ShapeUtil::MakeShape(F32, {100, 100});
  } else {
    p1 = builder.AddInstruction(
        HloInstruction::CreateParameter(1, input_shape, "p1"));
  }

  HloInstruction* root_op = nullptr;

  if (spec.op == HloOpcode::kDot) {
    DotDimensionNumbers dot_dimensions;
    dot_dimensions.add_lhs_contracting_dimensions(1);
    dot_dimensions.add_rhs_contracting_dimensions(0);
    root_op = builder.AddInstruction(HloInstruction::CreateDot(
        output_shape, p0, p1, dot_dimensions, PrecisionConfig()));
  } else if (spec.op == HloOpcode::kReduce ||
             spec.op == HloOpcode::kReduceWindow) {
    HloComputation::Builder add_builder("add");
    HloInstruction* a = add_builder.AddInstruction(
        HloInstruction::CreateParameter(0, scalar_shape, "a"));
    HloInstruction* b = add_builder.AddInstruction(
        HloInstruction::CreateParameter(1, scalar_shape, "b"));
    add_builder.AddInstruction(
        HloInstruction::CreateBinary(scalar_shape, HloOpcode::kAdd, a, b));
    HloComputation* add_comp =
        module->AddEmbeddedComputation(add_builder.Build());

    if (spec.op == HloOpcode::kReduce) {
      root_op = builder.AddInstruction(
          HloInstruction::CreateReduce(output_shape, p0, p1, {1}, add_comp));
    } else {
      Window window;
      WindowDimension* dim0 = window.add_dimensions();
      dim0->set_size(1);
      dim0->set_stride(1);
      dim0->set_base_dilation(1);
      dim0->set_window_dilation(1);
      WindowDimension* dim1 = window.add_dimensions();
      dim1->set_size(1);
      dim1->set_stride(1);
      dim1->set_base_dilation(1);
      dim1->set_window_dilation(1);
      root_op = builder.AddInstruction(HloInstruction::CreateReduceWindow(
          output_shape, p0, p1, window, add_comp));
    }
  } else {
    root_op = builder.AddInstruction(
        HloInstruction::CreateBinary(output_shape, spec.op, p0, p1));
  }

  module->AddEntryComputation(builder.Build());

  // Create CpuFloatSupport.
  CpuFloatSupport::CallLibraryChecker call_library_for_instruction =
      [&spec](const HloInstruction& hlo) {
        return spec.call_library_for_instruction;
      };
  CpuFloatSupport cpu_float_support(BF16, call_library_for_instruction);

  // Run FloatNormalization and check the results.
  FloatNormalization float_normalization(&cpu_float_support);
  ASSERT_OK_AND_ASSIGN(bool upcast, float_normalization.Run(module.get()));
  EXPECT_EQ(upcast, spec.upcast);
  PrimitiveType expected_input_dtype = spec.upcast ? F32 : BF16;
  CheckDtype(module.get(), expected_input_dtype, expected_input_dtype, F32);

  if (spec.op == HloOpcode::kReduce || spec.op == HloOpcode::kReduceWindow) {
    HloComputation* add_comp = root_op->to_apply();
    EXPECT_EQ(add_comp->root_instruction()->shape().element_type(),
              expected_input_dtype);
  }
}

std::vector<SkipInstructionTestSpec> GetSkipInstructionTestSpecs() {
  return std::vector<SkipInstructionTestSpec>{
      // Add op, always upcast.
      SkipInstructionTestSpec{HloOpcode::kAdd,
                              /*call_library_for_instruction=*/true,
                              /*upcast=*/true},
      // Library dot is disabled.
      SkipInstructionTestSpec{HloOpcode::kDot,
                              /*call_library_for_instruction=*/false,
                              /*upcast=*/true},
      // Library dot is enabled.
      SkipInstructionTestSpec{HloOpcode::kDot,
                              /*call_library_for_instruction=*/true,
                              /*upcast=*/false},
      // Reduce is disabled.
      SkipInstructionTestSpec{HloOpcode::kReduce,
                              /*call_library_for_instruction=*/false,
                              /*upcast=*/true},
      // Reduce is enabled.
      SkipInstructionTestSpec{HloOpcode::kReduce,
                              /*call_library_for_instruction=*/true,
                              /*upcast=*/false},
      // ReduceWindow is disabled.
      SkipInstructionTestSpec{HloOpcode::kReduceWindow,
                              /*call_library_for_instruction=*/false,
                              /*upcast=*/true},
      // ReduceWindow is enabled.
      SkipInstructionTestSpec{HloOpcode::kReduceWindow,
                              /*call_library_for_instruction=*/true,
                              /*upcast=*/false},
  };
}

INSTANTIATE_TEST_SUITE_P(SkipInstructionTestSuite, SkipInstructionTest,
                         ::testing::ValuesIn(GetSkipInstructionTestSpecs()),
                         SkipInstructionTest::Name);

TEST_F(TargetMachineTestBase, SortNotUpcast) {
  for (absl::string_view type_str : {"bf16", "f16"}) {
    std::string hlo_text = absl::StrReplaceAll(R"(
HloModule test_module

compare {
  p0 = $type$[] parameter(0)
  p1 = $type$[] parameter(1)
  p2 = s32[] parameter(2)
  p3 = s32[] parameter(3)
  ROOT cmp = pred[] compare(p0, p1), direction=LT
}

ENTRY main {
  k = $type$[100] parameter(0)
  v = s32[100] parameter(1)
  ROOT sort = ($type$[100], s32[100]) sort(k, v), dimensions={0}, to_apply=compare, is_stable=true
}
)",
                                               {{"$type$", type_str}});

    ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_text));

    PrimitiveType low_precision_type = (type_str == "bf16") ? BF16 : F16;

    CpuFloatSupport cpu_float_support(
        low_precision_type, [](const HloInstruction&) { return false; });

    FloatNormalization float_normalization(&cpu_float_support);
    ASSERT_OK_AND_ASSIGN(bool upcast, float_normalization.Run(module.get()));
    EXPECT_FALSE(upcast);

    HloInstruction* root = module->entry_computation()->root_instruction();
    EXPECT_EQ(root->opcode(), HloOpcode::kSort);
    EXPECT_EQ(root->operand(0)->shape().element_type(), low_precision_type);
    EXPECT_EQ(root->operand(1)->shape().element_type(), S32);

    HloComputation* compare_comp = root->to_apply();
    EXPECT_EQ(compare_comp->parameter_instruction(0)->shape().element_type(),
              low_precision_type);
  }
}

TEST_F(TargetMachineTestBase, SortUpcastForUnsupportedType) {
  std::string hlo_text = R"(
HloModule test_module

compare {
  p0 = f8e5m2[] parameter(0)
  p1 = f8e5m2[] parameter(1)
  p2 = s32[] parameter(2)
  p3 = s32[] parameter(3)
  ROOT cmp = pred[] compare(p0, p1), direction=LT
}

ENTRY main {
  k = f8e5m2[100] parameter(0)
  v = s32[100] parameter(1)
  ROOT sort = (f8e5m2[100], s32[100]) sort(k, v), dimensions={0}, to_apply=compare, is_stable=true
}
)";

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_text));

  CpuFloatSupport cpu_float_support(
      F8E5M2, [](const HloInstruction&) { return false; });

  FloatNormalization float_normalization(&cpu_float_support);
  ASSERT_OK_AND_ASSIGN(bool upcast, float_normalization.Run(module.get()));
  EXPECT_TRUE(upcast);

  HloInstruction* sort_instr = FindInstruction(module.get(), HloOpcode::kSort);
  ASSERT_NE(sort_instr, nullptr);
  EXPECT_EQ(sort_instr->operand(0)->shape().element_type(), F32);
  EXPECT_EQ(sort_instr->operand(1)->shape().element_type(), S32);
  HloComputation* updated_cmp = sort_instr->to_apply();
  EXPECT_EQ(updated_cmp->parameter_instruction(0)->shape().element_type(), F32);
  EXPECT_EQ(updated_cmp->parameter_instruction(1)->shape().element_type(), F32);
}

TEST_F(TargetMachineTestBase, CompareAndSelectNotUpcast) {
  for (absl::string_view type_str : {"bf16", "f16"}) {
    std::string hlo_text = absl::StrReplaceAll(R"(
HloModule test_module

ENTRY main {
  p0 = $type$[100] parameter(0)
  p1 = $type$[100] parameter(1)
  cmp = pred[100] compare(p0, p1), direction=LT
  ROOT select = $type$[100] select(cmp, p0, p1)
}
)",
                                               {{"$type$", type_str}});

    ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_text));

    PrimitiveType low_precision_type = (type_str == "bf16") ? BF16 : F16;

    CpuFloatSupport cpu_float_support(
        low_precision_type, [](const HloInstruction&) { return false; });

    FloatNormalization float_normalization(&cpu_float_support);
    ASSERT_OK_AND_ASSIGN(bool upcast, float_normalization.Run(module.get()));
    EXPECT_FALSE(upcast);

    HloInstruction* root = module->entry_computation()->root_instruction();
    EXPECT_EQ(root->opcode(), HloOpcode::kSelect);
    EXPECT_EQ(root->shape().element_type(), low_precision_type);
    EXPECT_EQ(root->operand(1)->shape().element_type(), low_precision_type);
    EXPECT_EQ(root->operand(2)->shape().element_type(), low_precision_type);

    const HloInstruction* cmp = root->operand(0);
    EXPECT_EQ(cmp->opcode(), HloOpcode::kCompare);
    EXPECT_EQ(cmp->operand(0)->shape().element_type(), low_precision_type);
    EXPECT_EQ(cmp->operand(1)->shape().element_type(), low_precision_type);
  }
}

}  // namespace
}  // namespace xla::cpu

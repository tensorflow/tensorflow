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
#include <utility>
#include <vector>

#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "xla/backends/cpu/codegen/target_machine_features.h"
#include "xla/backends/cpu/codegen/target_machine_test_base.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/testlib/verified_hlo_module.h"
#include "xla/hlo/transforms/simplifiers/float_normalization.h"
#include "xla/service/hlo_module_config.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/test.h"
#include "xla/xla_data.pb.h"

namespace xla::cpu {
namespace {

struct SkipInstructionTestSpec {
  HloOpcode op;
  bool call_library_for_dot;
  std::string cpu_name;
  std::string features;
  bool upcast;
};

class SkipInstructionTest
    : public TargetMachineTestBase,
      public ::testing::WithParamInterface<SkipInstructionTestSpec> {
 public:
  static std::string Name(
      const ::testing::TestParamInfo<SkipInstructionTestSpec>& info) {
    absl::string_view op = HloOpcodeString(info.param.op);
    absl::string_view dot_strategy =
        info.param.call_library_for_dot ? "LibDot" : "NoLibDot";
    absl::string_view bf16_strategy =
        absl::StrContains(info.param.features, "+avx512bf16") ? "Bf16"
                                                              : "NoBf16";
    return absl::StrCat(op, "_", dot_strategy, "_", bf16_strategy);
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

  // Create the HLO module: p0 <op> p1.
  HloComputation::Builder builder("SkipInstructionTest");
  Shape input_shape = ShapeUtil::MakeShape(BF16, {100, 100});
  Shape output_shape = ShapeUtil::MakeShape(F32, {100, 100});
  HloInstruction* p0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, input_shape, "p0"));
  HloInstruction* p1 = builder.AddInstruction(
      HloInstruction::CreateParameter(1, input_shape, "p1"));
  if (spec.op == HloOpcode::kDot) {
    DotDimensionNumbers dot_dimensions;
    dot_dimensions.add_lhs_contracting_dimensions(1);
    dot_dimensions.add_rhs_contracting_dimensions(0);
    builder.AddInstruction(HloInstruction::CreateDot(
        output_shape, p0, p1, dot_dimensions, PrecisionConfig()));
  } else {
    builder.AddInstruction(
        HloInstruction::CreateBinary(output_shape, spec.op, p0, p1));
  }
  std::unique_ptr<HloComputation> computation = builder.Build();
  std::unique_ptr<HloModule> module = std::make_unique<VerifiedHloModule>(
      "test", HloModuleConfig(),
      /*verifier_layout_sensitive=*/false,
      /*allow_mixed_precision_in_hlo_verifier=*/true,
      ShapeUtil::ByteSizeOfElements);
  module->AddEntryComputation(std::move(computation));

  // Create CpuFloatSupport.
  CpuFloatSupport::DotStrategyChecker call_library_for_dot =
      [&spec](const HloInstruction& hlo) { return spec.call_library_for_dot; };
  std::unique_ptr<TargetMachineFeatures> features = CreateTargetMachineFeatures(
      "x86_64-unknown-linux-gnu", spec.cpu_name, spec.features);
  CpuFloatSupport cpu_float_support(BF16, call_library_for_dot, features.get());

  // Run FloatNormalization and check the results.
  FloatNormalization float_normalization(&cpu_float_support);
  TF_ASSERT_OK_AND_ASSIGN(bool upcast, float_normalization.Run(module.get()));
  EXPECT_EQ(upcast, spec.upcast);
  PrimitiveType expected_input_dtype = spec.upcast ? F32 : BF16;
  CheckDtype(module.get(), expected_input_dtype, expected_input_dtype, F32);
}

std::vector<SkipInstructionTestSpec> GetSkipInstructionTestSpecs() {
  return std::vector<SkipInstructionTestSpec>{
      // Add op, always upcast.
      SkipInstructionTestSpec{HloOpcode::kAdd,
                              /*call_library_for_dot=*/true,
                              /*cpu_name=*/"sapphirerapids",
                              /*features=*/"+avx512bf16",
                              /*upcast=*/true},
      // CPU has BF16, but library dot is disabled.
      SkipInstructionTestSpec{HloOpcode::kDot,
                              /*call_library_for_dot=*/false,
                              /*cpu_name=*/"sapphirerapids",
                              /*features=*/"+avx512bf16",
                              /*upcast=*/true},
      // Library dot is enabled, but CPU does not have BF16.
      SkipInstructionTestSpec{HloOpcode::kDot,
                              /*call_library_for_dot=*/true,
                              /*cpu_name=*/"znver3",
                              /*features=*/"+avx2",
                              /*upcast=*/true},
      // Library dot is enabled and CPU has BF16. Use mixed precision.
      SkipInstructionTestSpec{HloOpcode::kDot,
                              /*call_library_for_dot=*/true,
                              /*cpu_name=*/"sapphirerapids",
                              /*features=*/"+avx512bf16",
                              /*upcast=*/false}};
}

INSTANTIATE_TEST_SUITE_P(SkipInstructionTestSuite, SkipInstructionTest,
                         ::testing::ValuesIn(GetSkipInstructionTestSpecs()),
                         SkipInstructionTest::Name);

}  // namespace
}  // namespace xla::cpu

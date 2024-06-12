/* Copyright 2024 The OpenXLA Authors.

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

// TODO(b/343158720): Simplify the tests in this file after a generic emitter
// has landed.
#include "xla/service/gpu/triton_support.h"

#include <memory>
#include <string>
#include <tuple>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/strings/substitute.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/utils/hlo_query.h"
#include "xla/primitive_util.h"
#include "xla/service/gpu/gpu_device_info_for_tests.h"
#include "xla/service/gpu/ir_emitter_triton.h"
#include "xla/service/gpu/matmul_utils.h"
#include "xla/service/gpu/triton_fusion_analysis.h"
#include "xla/service/gpu/triton_test_utils.h"
#include "xla/stream_executor/device_description.h"
#include "xla/xla.pb.h"
#include "xla/xla_data.pb.h"
#include "tsl/lib/core/status_test_util.h"
#include "tsl/platform/status_matchers.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace gpu {
namespace {

using UnaryElementwiseTest = TritonSupportTestWithParam;

// TODO(b/331636835): updates elementwise op tests to directly emit single op
// instead of relying on triton gemm kernel.
TEST_P(UnaryElementwiseTest, IsTritonSupportedUnaryElementwise) {
  PrimitiveType data_type;
  HloOpcode opcode;
  std::tie(data_type, opcode) = GetParam();
  if (!GetCudaComputeCapability().IsAtLeast(
          se::CudaComputeCapability::AMPERE) &&
      data_type == BF16) {
    GTEST_SKIP() << "No BF16 before Ampere.";
  }

  const std::string kHloTestTemplate = R"(
triton_computation {
  parameter_0 = $0[33,68]{1,0} parameter(0)
  unary = $0[33,68]{1,0} $1(parameter_0)
  ROOT convert = f32[33,68]{1,0} convert(unary)
}

ENTRY e {
  parameter_0 = $0[33,68]{1,0} parameter(0)
  ROOT root_op = f32[33,68]{1,0} fusion(parameter_0),
    kind=kCustom, calls=triton_computation,
    backend_config={"fusion_backend_config":{"kind":"__triton"}}
})";
  const std::string hlo_test = absl::Substitute(
      kHloTestTemplate, primitive_util::LowercasePrimitiveTypeName(data_type),
      HloOpcodeString(opcode));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_test));
  const HloComputation* computation =
      module->GetComputationWithName("triton_computation");
  ASSERT_TRUE(computation != nullptr);
  const HloInstruction* instr =
      hlo_query::GetFirstInstructionWithOpcode(*computation, opcode);
  if (IsTritonSupportedInstruction(*instr, GetCudaComputeCapability())) {
    TF_EXPECT_OK(ApplyFloatNormalization(module.get()));
    TF_EXPECT_OK(CreateTritonIrAndFileCheck(
        *computation, /*config=*/{}, /*output_tile_sizes=*/{1, 32}, EmitGeneric,
        "CHECK: tt.func @triton_fn"));
  } else {
    // TODO(b/331632717): update the check to use SymbolicTileAnalysis to avoid
    // tiling failures and check triton emitter fails gracefully.
    EXPECT_THAT(TritonFusionAnalysis::Execute(*computation),
                tsl::testing::StatusIs(
                    absl::StatusCode::kFailedPrecondition,
                    ::testing::HasSubstr(
                        "Can not propagate dim orders and requirements")));
  }
}

INSTANTIATE_TEST_SUITE_P(
    UnaryElementwiseTestSuite, UnaryElementwiseTest,
    ::testing::Combine(::testing::Values(S8, S16, S32, F16, F32, BF16),
                       ::testing::Values(HloOpcode::kConvert, HloOpcode::kAbs,
                                         HloOpcode::kNegate)),
    TritonSupportTestParamsToString);
INSTANTIATE_TEST_SUITE_P(
    UnaryPREDTestSuite, UnaryElementwiseTest,
    ::testing::Combine(::testing::Values(PRED),
                       ::testing::Values(HloOpcode::kConvert, HloOpcode::kNot)),
    TritonSupportTestParamsToString);
INSTANTIATE_TEST_SUITE_P(
    UnaryMathTestSuite, UnaryElementwiseTest,
    ::testing::Combine(::testing::Values(F16, F32, BF16),
                       ::testing::Values(HloOpcode::kCeil, HloOpcode::kCos,
                                         HloOpcode::kExp, HloOpcode::kExpm1,
                                         HloOpcode::kFloor, HloOpcode::kLog,
                                         HloOpcode::kLog1p, HloOpcode::kRsqrt,
                                         HloOpcode::kSin, HloOpcode::kSqrt,
                                         HloOpcode::kCbrt, HloOpcode::kTan,
                                         HloOpcode::kTanh, HloOpcode::kErf)),
    TritonSupportTestParamsToString);

using BinaryElementwiseTest = TritonSupportTestWithParam;

TEST_P(BinaryElementwiseTest, IsTritonSupportedBinaryElementwise) {
  PrimitiveType data_type;
  HloOpcode opcode;
  std::tie(data_type, opcode) = GetParam();
  if (!GetCudaComputeCapability().IsAtLeast(
          se::CudaComputeCapability::AMPERE) &&
      data_type == BF16) {
    GTEST_SKIP() << "No BF16 before Ampere.";
  }

  const std::string kHloTestTemplate = R"(
triton_computation {
  parameter_0 = $0[11,63]{1,0} parameter(0)
  parameter_1 = $0[11,63]{1,0} parameter(1)
  ROOT binary = $0[11,63]{1,0} $1(parameter_0, parameter_1)
}

ENTRY e {
  parameter_0 = $0[11,63]{1,0} parameter(0)
  parameter_1 = $0[11,63]{1,0} parameter(1)
  ROOT triton_op = $0[11,63]{1,0} fusion(parameter_0, parameter_1),
    kind=kCustom, calls=triton_computation,
    backend_config={"fusion_backend_config":{"kind":"__triton"}}
})";
  const std::string hlo_test = absl::Substitute(
      kHloTestTemplate, primitive_util::LowercasePrimitiveTypeName(data_type),
      HloOpcodeString(opcode));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_test));
  const HloComputation* computation =
      module->GetComputationWithName("triton_computation");
  ASSERT_TRUE(computation != nullptr);
  const HloInstruction* instr =
      hlo_query::GetFirstInstructionWithOpcode(*computation, opcode);
  if (IsTritonSupportedInstruction(*instr, GetCudaComputeCapability())) {
    TF_EXPECT_OK(ApplyFloatNormalization(module.get()));
    TF_EXPECT_OK(CreateTritonIrAndFileCheck(
        *computation, /*config=*/{}, /*output_tile_sizes=*/{1, 32}, EmitGeneric,
        "CHECK: tt.func @triton_fn"));
  } else {
    EXPECT_THAT(TritonFusionAnalysis::Execute(*computation),
                ::testing::AnyOf(
                    tsl::testing::StatusIs(
                        absl::StatusCode::kInternal,
                        ::testing::HasSubstr(
                            "std::holds_alternative<DimOrdersAndReqs>")),
                    tsl::testing::StatusIs(
                        absl::StatusCode::kFailedPrecondition,
                        ::testing::HasSubstr(
                            "Can not propagate dim orders and requirements"))));
  }
}

INSTANTIATE_TEST_SUITE_P(
    BinaryElementwiseTestSuite, BinaryElementwiseTest,
    ::testing::Combine(::testing::Values(S8, S16, S32, F16, F32, BF16),
                       ::testing::Values(HloOpcode::kAdd, HloOpcode::kMultiply,
                                         HloOpcode::kMaximum,
                                         HloOpcode::kMinimum,
                                         HloOpcode::kSubtract)),
    TritonSupportTestParamsToString);

INSTANTIATE_TEST_SUITE_P(BinaryPREDTestSuite, BinaryElementwiseTest,
                         ::testing::Combine(::testing::Values(PRED),
                                            ::testing::Values(HloOpcode::kAnd,
                                                              HloOpcode::kOr,
                                                              HloOpcode::kXor)),
                         TritonSupportTestParamsToString);
INSTANTIATE_TEST_SUITE_P(
    BinaryMathTestSuite, BinaryElementwiseTest,
    ::testing::Combine(::testing::Values(F16, F32, BF16),
                       ::testing::Values(HloOpcode::kAtan2, HloOpcode::kDivide,
                                         HloOpcode::kPower)),
    TritonSupportTestParamsToString);

using CompareTest = TritonSupportTestWithParam;

TEST_P(CompareTest, IsTritonSupportedCompare) {
  PrimitiveType data_type;
  HloOpcode opcode;
  std::tie(data_type, opcode) = GetParam();
  if (!GetCudaComputeCapability().IsAtLeast(
          se::CudaComputeCapability::AMPERE) &&
      data_type == BF16) {
    GTEST_SKIP() << "No BF16 before Ampere.";
  }

  const std::string kHloTestTemplate = R"(
triton_computation {
  parameter_0 = $0[11,63]{1,0} parameter(0)
  parameter_1 = $0[11,63]{1,0} parameter(1)
  compare = pred[11,63]{1,0} $1(parameter_0, parameter_1), direction=GE
  ROOT convert = f32[11,63]{1,0} convert(compare)
}

ENTRY e {
  parameter_0 = $0[11,63]{1,0} parameter(0)
  parameter_1 = $0[11,63]{1,0} parameter(1)
  ROOT triton_op = f32[11,63]{1,0} fusion(parameter_0, parameter_1),
    kind=kCustom, calls=triton_computation,
    backend_config={"fusion_backend_config":{"kind":"__triton"}}
})";
  const std::string hlo_test = absl::Substitute(
      kHloTestTemplate, primitive_util::LowercasePrimitiveTypeName(data_type),
      HloOpcodeString(opcode));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_test));
  const HloComputation* computation =
      module->GetComputationWithName("triton_computation");
  ASSERT_TRUE(computation != nullptr);
  const HloInstruction* instr =
      hlo_query::GetFirstInstructionWithOpcode(*computation, opcode);
  if (IsTritonSupportedInstruction(*instr, GetCudaComputeCapability())) {
    TF_EXPECT_OK(ApplyFloatNormalization(module.get()));
    TF_EXPECT_OK(CreateTritonIrAndFileCheck(
        *computation, /*config=*/{}, /*output_tile_sizes=*/{1, 32}, EmitGeneric,
        "CHECK: tt.func @triton_fn"));
  } else {
    EXPECT_THAT(
        TritonFusionAnalysis::Execute(*computation),
        tsl::testing::StatusIs(
            absl::StatusCode::kInternal,
            ::testing::HasSubstr("std::holds_alternative<DimOrdersAndReqs>")));
  }
}

INSTANTIATE_TEST_SUITE_P(
    CompareTestSuite, CompareTest,
    ::testing::Combine(::testing::Values(PRED, S8, S16, S32, F16, F32, BF16),
                       ::testing::Values(HloOpcode::kCompare)),
    TritonSupportTestParamsToString);

using TernaryElementwiseTest = TritonSupportTestWithParam;

TEST_P(TernaryElementwiseTest, IsTritonSupportedTernaryElementwise) {
  PrimitiveType data_type;
  HloOpcode opcode;
  std::tie(data_type, opcode) = GetParam();
  if (!GetCudaComputeCapability().IsAtLeast(
          se::CudaComputeCapability::AMPERE) &&
      data_type == BF16) {
    GTEST_SKIP() << "No BF16 before Ampere.";
  }

  const std::string kHloTestTemplate = R"(
triton_computation {
  parameter_0 = $0[13,63]{1,0} parameter(0)
  parameter_1 = $0[13,63]{1,0} parameter(1)
  parameter_2 = pred[13,63]{1,0} parameter(2)
  ternary = $0[13,63]{1,0} $1(parameter_2, parameter_0, parameter_1)
  ROOT convert = f32[13,63]{1,0} convert(ternary)
}

ENTRY e {
  parameter_0 = $0[13,63]{1,0} parameter(0)
  parameter_1 = $0[13,63]{1,0} parameter(1)
  parameter_2 = pred[13,63]{1,0} parameter(2)
  ROOT triton_op = f32[13,63]{1,0} fusion(parameter_0, parameter_1, parameter_2),
    kind=kCustom, calls=triton_computation,
    backend_config={"fusion_backend_config":{"kind":"__triton"}}
})";
  const std::string hlo_test = absl::Substitute(
      kHloTestTemplate, primitive_util::LowercasePrimitiveTypeName(data_type),
      HloOpcodeString(opcode));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_test));
  const HloComputation* computation =
      module->GetComputationWithName("triton_computation");
  ASSERT_TRUE(computation != nullptr);
  const HloInstruction* instr =
      hlo_query::GetFirstInstructionWithOpcode(*computation, opcode);
  if (IsTritonSupportedInstruction(*instr, GetCudaComputeCapability())) {
    TF_EXPECT_OK(ApplyFloatNormalization(module.get()));
    TF_EXPECT_OK(CreateTritonIrAndFileCheck(
        *computation, /*config=*/{}, /*output_tile_sizes=*/{1, 32}, EmitGeneric,
        "CHECK: tt.func @triton_fn"));
  } else {
    EXPECT_THAT(
        TritonFusionAnalysis::Execute(*computation),
        tsl::testing::StatusIs(
            absl::StatusCode::kInternal,
            ::testing::HasSubstr("std::holds_alternative<DimOrdersAndReqs>")));
  }
}

INSTANTIATE_TEST_SUITE_P(
    TernaryElementwiseTestSuite, TernaryElementwiseTest,
    ::testing::Combine(::testing::Values(PRED, S8, S16, S32, F16, F32, BF16),
                       ::testing::Values(HloOpcode::kSelect)),
    TritonSupportTestParamsToString);

using ReduceConstTest = TritonSupportTestWithParam;
TEST_P(ReduceConstTest, IsTritonSupportedReduceWithConstInit) {
  PrimitiveType data_type;
  HloOpcode opcode;
  std::tie(data_type, opcode) = GetParam();
  if (!GetCudaComputeCapability().IsAtLeast(
          se::CudaComputeCapability::AMPERE) &&
      data_type == BF16) {
    GTEST_SKIP() << "No BF16 before Ampere.";
  }

  const std::string kHloTestTemplate = R"(
HloModule t
add {
  Arg_0 = $0[] parameter(0)
  Arg_1 = $0[] parameter(1)
  ROOT add = $0[] add(Arg_0, Arg_1)
}

triton_computation {
  parameter_0 = $0[125,127]{1,0} parameter(0)
  constant_0 = $0[] constant(0)
  ROOT reduce = $0[125]{0} $1(parameter_0, constant_0), dimensions={1}, to_apply=add
}

ENTRY main {
  parameter_0 = $0[125,127]{1,0} parameter(0)
  ROOT triton_op = $0[125]{0} fusion(parameter_0),
                          kind=kCustom, calls=triton_computation,
                          backend_config={"fusion_backend_config":
                                           {"kind":"__triton"}}
})";
  const std::string hlo_test = absl::Substitute(
      kHloTestTemplate, primitive_util::LowercasePrimitiveTypeName(data_type),
      HloOpcodeString(opcode));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_test));

  const HloFusionInstruction* fusion = Cast<HloFusionInstruction>(
      module->entry_computation()->root_instruction());
  const HloComputation* computation = fusion->fused_instructions_computation();
  ASSERT_TRUE(computation != nullptr);
  const HloInstruction* instr =
      hlo_query::GetFirstInstructionWithOpcode(*computation, opcode);
  if (IsTritonSupportedInstruction(*instr, GetCudaComputeCapability())) {
    TF_EXPECT_OK(ApplyFloatNormalization(module.get()));
    TF_EXPECT_OK(CreateTritonIrAndFileCheck(
        *computation, /*config=*/{}, /*output_tile_sizes=*/{1}, EmitGeneric,
        "CHECK: tt.func @triton_fn"));
  } else {
    const se::DeviceDescription dev_info =
        TestGpuDeviceInfo::RTXA6000DeviceInfo(GetCudaComputeCapability());
    EXPECT_THAT(
        TritonWrapper(*TritonFusionAnalysis::Execute(*computation), "test_fn",
                      fusion, GetCudaComputeCapability(), dev_info,
                      /*config=*/{}, /*output_tile_sizes=*/{1}, &llvm_module_,
                      &EmitGeneric, mlir_context_),
        tsl::testing::StatusIs(
            absl::StatusCode::kInternal,
            ::testing::HasSubstr("Failed to compile Triton kernel")));
  }
}

INSTANTIATE_TEST_SUITE_P(
    ReduceConstTestSuite, ReduceConstTest,
    ::testing::Combine(::testing::Values(F16, F32, BF16),
                       ::testing::Values(HloOpcode::kReduce)),
    TritonSupportTestParamsToString);

TEST_F(TritonSupportTest,
       SupportedReduceWithConvertConstantIsCodegenedSuccessfullyWithTriton) {
  if (!GetCudaComputeCapability().IsAtLeast(
          se::CudaComputeCapability::AMPERE)) {
    GTEST_SKIP() << "No BF16 before Ampere.";
  }
  const std::string kHloTest = R"(
HloModule t
add {
  Arg_0 = f32[] parameter(0)
  Arg_1 = f32[] parameter(1)
  ROOT add = f32[] add(Arg_0, Arg_1)
}

triton_computation {
  parameter_0 = f32[125,127]{1,0} parameter(0)
  constant_0 = bf16[] constant(0)
  convert_0 = f32[] convert(constant_0)
  ROOT reduce = f32[125]{0} reduce(parameter_0, convert_0), dimensions={1}, to_apply=add
}

ENTRY main {
  parameter_0 = f32[125,127]{1,0} parameter(0)
  ROOT triton_op = f32[125]{0} fusion(parameter_0), kind=kCustom,
  calls=triton_computation,
                        backend_config={"fusion_backend_config":
                        {"kind":"__triton"}}
})";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kHloTest));

  const HloComputation* computation =
      module->GetComputationWithName("triton_computation");
  ASSERT_TRUE(computation != nullptr);
  const HloInstruction* instr = hlo_query::GetFirstInstructionWithOpcode(
      *computation, HloOpcode::kReduce);
  EXPECT_TRUE(IsTritonSupportedInstruction(*instr, GetCudaComputeCapability())
                  .CanFuse());
  TF_EXPECT_OK(ApplyFloatNormalization(module.get()));
  TF_EXPECT_OK(CreateTritonIrAndFileCheck(
      *computation, /*config=*/{}, /*output_tile_sizes=*/{1}, EmitGeneric,
      "CHECK: tt.func @triton_fn"));
}

TEST_F(
    TritonSupportTest,
    UnsupportedReduceWithMoreThanOneReduceDimensionsFailsGracefullyWithTriton) {
  const std::string kHloTest = R"(
HloModule t
add {
  Arg_0 = f32[] parameter(0)
  Arg_1 = f32[] parameter(1)
  ROOT add = f32[] add(Arg_0, Arg_1)
}

triton_computation {
  parameter_0 = f32[2,125,127]{2,1,0} parameter(0)
  constant_0 = f32[] constant(0)
  ROOT reduce = f32[2]{0} reduce(parameter_0, constant_0), dimensions={1,2}, to_apply=add
}

ENTRY main {
  parameter_0 = f32[2,125,127]{2,1,0} parameter(0)
  ROOT triton_op = f32[2]{0} fusion(parameter_0),
                          kind=kCustom, calls=triton_computation,
                          backend_config={"fusion_backend_config":
                                            {"kind":"__triton"}}
})";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> hlo_module,
                          ParseAndReturnVerifiedModule(kHloTest));

  const HloComputation* computation =
      hlo_module->GetComputationWithName("triton_computation");
  ASSERT_TRUE(computation != nullptr);
  const HloInstruction* instr = hlo_query::GetFirstInstructionWithOpcode(
      *computation, HloOpcode::kReduce);
  EXPECT_THAT(IsTritonSupportedInstruction(*instr, GetCudaComputeCapability())
                  .Explain(),
              ::testing::HasSubstr(
                  "Reduction is not a row-reduction of a single operand."));
  EXPECT_THAT(TritonFusionAnalysis::Execute(*computation),
              tsl::testing::StatusIs(
                  absl::StatusCode::kFailedPrecondition,
                  ::testing::HasSubstr(
                      "Can not propagate dim orders and requirements")));
}

TEST_F(TritonSupportTest,
       UnsupportedReduceWithNonLastReduceDimensionFailsGracefullyWithTriton) {
  const std::string kHloTest = R"(
HloModule t
add {
  Arg_0 = f32[] parameter(0)
  Arg_1 = f32[] parameter(1)
  ROOT add = f32[] add(Arg_0, Arg_1)
}

triton_computation {
  parameter_0 = f32[125,127]{1,0} parameter(0)
  constant_0 = f32[] constant(0)
  ROOT reduce = f32[127]{0} reduce(parameter_0, constant_0), dimensions={0}, to_apply=add
}

ENTRY main {
  parameter_0 = f32[125,127]{1,0} parameter(0)
  ROOT triton_op = f32[127]{0} fusion(parameter_0),
                          kind=kCustom, calls=triton_computation,
                          backend_config={"fusion_backend_config":
                                            {"kind":"__triton"}}
})";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> hlo_module,
                          ParseAndReturnVerifiedModule(kHloTest));

  const HloComputation* computation =
      hlo_module->GetComputationWithName("triton_computation");
  ASSERT_TRUE(computation != nullptr);
  const HloInstruction* instr = hlo_query::GetFirstInstructionWithOpcode(
      *computation, HloOpcode::kReduce);
  EXPECT_THAT(IsTritonSupportedInstruction(*instr, GetCudaComputeCapability())
                  .Explain(),
              ::testing::HasSubstr(
                  "Reduction is not a row-reduction of a single operand."));
  EXPECT_THAT(TritonFusionAnalysis::Execute(*computation),
              tsl::testing::StatusIs(
                  absl::StatusCode::kFailedPrecondition,
                  ::testing::HasSubstr(
                      "Can not propagate dim orders and requirements")));
}

TEST_F(TritonSupportTest,
       UnsupportedReduceWithMoreThanOneOperandsFailsGracefullyWithTriton) {
  const std::string kHloTest = R"(
HloModule t
add {
  Arg_0 = f32[] parameter(0)
  Arg_2 = f32[] parameter(1)
  Arg_1 = f32[] parameter(2)
  Arg_3 = f32[] parameter(3)
  add_0 = f32[] add(Arg_0, Arg_2)
  add_1 = f32[] add(Arg_1, Arg_3)
  ROOT pair = (f32[], f32[]) tuple(add_0, add_1)
}

triton_computation {
  parameter_0 = f32[125,127] parameter(0)
  constant_0 = f32[] constant(0)
  tuple_0 = (f32[125]{0}, f32[125]{0}) reduce(parameter_0, parameter_0, constant_0, constant_0), dimensions={1}, to_apply=add
  ROOT reduce = f32[125]{0} get-tuple-element(tuple_0), index=0
}

ENTRY main {
  parameter_0 = f32[125,127]{1,0} parameter(0)
  ROOT triton_op = f32[125]{0} fusion(parameter_0),
                          kind=kCustom, calls=triton_computation,
                          backend_config={"fusion_backend_config":
                                           {"kind":"__triton"}}
})";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> hlo_module,
                          ParseAndReturnVerifiedModule(kHloTest));

  const HloComputation* computation =
      hlo_module->GetComputationWithName("triton_computation");
  ASSERT_TRUE(computation != nullptr);
  const HloInstruction* instr = hlo_query::GetFirstInstructionWithOpcode(
      *computation, HloOpcode::kReduce);
  EXPECT_THAT(
      IsTritonSupportedInstruction(*instr, GetCudaComputeCapability())
          .Explain(),
      ::testing::HasSubstr("Unsupported output data type for Reduce op."));
  EXPECT_THAT(TritonFusionAnalysis::Execute(*computation),
              tsl::testing::StatusIs(
                  absl::StatusCode::kFailedPrecondition,
                  ::testing::HasSubstr(
                      "Can not propagate dim orders and requirements")));
}

TEST_F(TritonSupportTest,
       UnsupportedReduceWithNonConstReduceValueFailsGracefullyWithTriton) {
  const std::string kHloTest = R"(
HloModule t
add {
  Arg_0 = f32[] parameter(0)
  Arg_1 = f32[] parameter(1)
  ROOT add = f32[] add(Arg_0, Arg_1)
}

triton_computation {
  parameter_0 = f32[125,127]{1,0} parameter(0)
  init = f32[] parameter(1)
  ROOT reduce = f32[125]{0} reduce(parameter_0, init), dimensions={1}, to_apply=add
}

ENTRY main {
  parameter_0 = f32[125,127]{1,0} parameter(0)
  parameter_1 = f32[] parameter(1)
  ROOT triton_op = f32[125]{0} fusion(parameter_0, parameter_1),
                          kind=kCustom, calls=triton_computation,
                        backend_config={"fusion_backend_config":
                                         {"kind":"__triton"}}
})";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> hlo_module,
                          ParseAndReturnVerifiedModule(kHloTest));

  const HloFusionInstruction* fusion = Cast<HloFusionInstruction>(
      hlo_module->entry_computation()->root_instruction());
  const HloComputation* computation = fusion->fused_instructions_computation();
  ASSERT_TRUE(computation != nullptr);
  const HloInstruction* instr = hlo_query::GetFirstInstructionWithOpcode(
      *computation, HloOpcode::kReduce);
  const se::DeviceDescription dev_info =
      TestGpuDeviceInfo::RTXA6000DeviceInfo(GetCudaComputeCapability());
  EXPECT_THAT(IsTritonSupportedInstruction(*instr, GetCudaComputeCapability())
                  .Explain(),
              ::testing::HasSubstr("Reduction init value should be a constant "
                                   "or a convert of a constant."));
  EXPECT_THAT(
      TritonWrapper(*TritonFusionAnalysis::Execute(*computation), "test_fn",
                    fusion, GetCudaComputeCapability(), dev_info, /*config=*/{},
                    /*output_tile_sizes=*/{1}, &llvm_module_, &EmitGeneric,
                    mlir_context_),
      tsl::testing::StatusIs(
          absl::StatusCode::kInternal,
          ::testing::HasSubstr("operand->opcode() == HloOpcode::kConstant")));
}

TEST_F(TritonSupportTest,
       UnsupportedReductionComputationFailsGracefullyWithTriton) {
  const std::string kHloTest = R"(
HloModule t
custom_call {
  Arg_0 = f32[] parameter(0)
  Arg_1 = f32[] parameter(1)
  ROOT custom_call = f32[] custom-call(Arg_0, Arg_1), custom_call_target="foo"
}

triton_computation {
  parameter_0 = f32[125,127]{1,0} parameter(0)
  constant_0 = f32[] constant(0)
  ROOT reduce = f32[125]{0} reduce(parameter_0, constant_0), dimensions={1}, to_apply=custom_call
}

ENTRY main {
  parameter_0 = f32[125,127]{1,0} parameter(0)
  ROOT triton_op = f32[125]{0} fusion(parameter_0),
                          kind=kCustom, calls=triton_computation,
                          backend_config={"fusion_backend_config":
                                         {"kind":"__triton"}}
})";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> hlo_module,
                          ParseAndReturnVerifiedModule(kHloTest));

  const HloFusionInstruction* fusion = Cast<HloFusionInstruction>(
      hlo_module->entry_computation()->root_instruction());
  const HloComputation* computation = fusion->fused_instructions_computation();
  ASSERT_TRUE(computation != nullptr);
  const HloInstruction* instr = hlo_query::GetFirstInstructionWithOpcode(
      *computation, HloOpcode::kReduce);
  const se::DeviceDescription dev_info =
      TestGpuDeviceInfo::RTXA6000DeviceInfo(GetCudaComputeCapability());
  EXPECT_THAT(
      IsTritonSupportedInstruction(*instr, GetCudaComputeCapability())
          .Explain(),
      ::testing::HasSubstr("Unsupported reduction computation by Triton."));
  EXPECT_THAT(
      TritonWrapper(*TritonFusionAnalysis::Execute(*computation), "test_fn",
                    fusion, GetCudaComputeCapability(), dev_info, /*config=*/{},
                    /*output_tile_sizes=*/{1}, &llvm_module_, &EmitGeneric,
                    mlir_context_),
      tsl::testing::StatusIs(absl::StatusCode::kInvalidArgument,
                             ::testing::HasSubstr("Unsupported operation")));
}
}  // namespace
}  // namespace gpu
}  // namespace xla

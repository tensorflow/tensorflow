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

#include "xla/service/gpu/triton_support.h"

#include <memory>
#include <string>
#include <tuple>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/base/optimization.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_replace.h"
#include "absl/strings/string_view.h"
#include "absl/strings/substitute.h"
#include "llvm/IR/Module.h"
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "xla/error_spec.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/utils/hlo_query.h"
#include "xla/primitive_util.h"
#include "xla/service/float_normalization.h"
#include "xla/service/gpu/gpu_device_info_for_tests.h"
#include "xla/service/gpu/gpu_float_support.h"
#include "xla/service/gpu/ir_emitter_triton.h"
#include "xla/service/gpu/matmul_utils.h"
#include "xla/service/gpu/tests/gpu_codegen_test.h"
#include "xla/service/gpu/triton_fusion_analysis.h"
#include "xla/service/hlo_pass_pipeline.h"
#include "xla/stream_executor/device_description.h"
#include "xla/xla.pb.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/status_matchers.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace gpu {
namespace {

class TritonSupportTest : public GpuCodegenTest {
 public:
  se::CudaComputeCapability GetCudaComputeCapability() {
    return backend()
        .default_stream_executor()
        ->GetDeviceDescription()
        .cuda_compute_capability();
  }
  absl::StatusOr<bool> ApplyFloatNormalization(HloModule* module) {
    const GpuFloatSupport bf16_support(GetCudaComputeCapability(), BF16);
    HloPassPipeline pipeline("hlo float normalization");
    pipeline.AddPass<FloatNormalization>(&bf16_support);
    return pipeline.Run(module);
  }

  float getTolerance(PrimitiveType data_type) {
    float tolerance;
    switch (data_type) {
      case F64:
      case F32:
        tolerance = 1e-6;
        break;
      case F16:
        tolerance = 2e-4;
        break;
      case BF16:
        tolerance = 2e-2;
        break;
      case PRED:
      case S8:
        tolerance = 3e-2;
        break;
      case S16:
        tolerance = 3e-3;
        break;
      case S32:
        tolerance = 3e-3;
        break;
      default:
        ABSL_UNREACHABLE();
    }
    return tolerance;
  }

 protected:
  llvm::LLVMContext llvm_ctx_;
  llvm::Module llvm_module_{"module", llvm_ctx_};
  mlir::MLIRContext mlir_context_;
  TritonGemmConfig config_{16, 32, 512, 1, 4, 8};
};

class TritonSupportTestWithParam : public TritonSupportTest,
                                   public ::testing::WithParamInterface<
                                       std::tuple<PrimitiveType, HloOpcode>> {};

std::string TestParamsToString(
    const ::testing::TestParamInfo<std::tuple<PrimitiveType, HloOpcode>>&
        data) {
  PrimitiveType data_type;
  HloOpcode opcode;
  std::tie(data_type, opcode) = data.param;
  return absl::StrCat(
      primitive_util::LowercasePrimitiveTypeName(data_type), "_",
      absl::StrReplaceAll(HloOpcodeString(opcode), {{"-", "_"}}));
}

using UnaryElementwiseTest = TritonSupportTestWithParam;

// TODO(b/331636835): updates elementwise op tests to directly emit single op
// instead of relying on triton gemm kernel.
TEST_P(UnaryElementwiseTest, IsTritonSupportedExecutesCorrectlyForUnary) {
  PrimitiveType data_type;
  HloOpcode opcode;
  std::tie(data_type, opcode) = GetParam();
  if (!GetCudaComputeCapability().IsAtLeast(
          se::CudaComputeCapability::AMPERE) &&
      data_type == BF16) {
    GTEST_SKIP() << "No BF16 before Ampere.";
  }

  const std::string kHloTestTemplate = R"(
triton_gemm___computation {
  parameter_0 = f32[15,33]{1,0} parameter(0)
  parameter_1 = $0[33,68]{1,0} parameter(1)
  unary = $0[33,68]{1,0} $1(parameter_1)
  convert = f32[33,68]{1,0} convert(unary)
  ROOT dot = f32[15,68]{1,0} dot(parameter_0, convert),
    lhs_contracting_dims={1}, rhs_contracting_dims={0},
    operand_precision={HIGH, HIGH}
}

ENTRY e {
  parameter_0 = f32[15,33]{1,0} parameter(0)
  parameter_1 = $0[33,68]{1,0} parameter(1)
  ROOT triton_gemm = f32[15,68]{1,0} fusion(parameter_0, parameter_1),
    kind=kCustom, calls=triton_gemm___computation,
    backend_config={"fusion_backend_config":{"kind":"__triton_gemm"}}
})";
  const std::string hlo_test = absl::Substitute(
      kHloTestTemplate, primitive_util::LowercasePrimitiveTypeName(data_type),
      HloOpcodeString(opcode));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_test));
  const HloComputation* computation =
      module->GetComputationWithName("triton_gemm___computation");
  ASSERT_TRUE(computation != nullptr);
  const HloInstruction* instr =
      hlo_query::GetFirstInstructionWithOpcode(*computation, opcode);
  if (IsTritonSupportedInstruction(*instr, GetCudaComputeCapability())) {
    float tolerance = getTolerance(data_type);
    EXPECT_OK(ApplyFloatNormalization(module.get()));
    EXPECT_TRUE(RunAndCompareNoHloPasses(
        std::move(module), ErrorSpec{/*aabs=*/tolerance, /*arel=*/tolerance}));
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
    TestParamsToString);
INSTANTIATE_TEST_SUITE_P(
    UnaryPREDTestSuite, UnaryElementwiseTest,
    ::testing::Combine(::testing::Values(PRED),
                       ::testing::Values(HloOpcode::kConvert, HloOpcode::kNot)),
    TestParamsToString);
INSTANTIATE_TEST_SUITE_P(
    UnaryMathTestSuite, UnaryElementwiseTest,
    ::testing::Combine(::testing::Values(F16, F32, BF16),
                       ::testing::Values(HloOpcode::kCos, HloOpcode::kExp,
                                         HloOpcode::kExpm1, HloOpcode::kLog,
                                         HloOpcode::kLog1p, HloOpcode::kRsqrt,
                                         HloOpcode::kSin, HloOpcode::kSqrt,
                                         HloOpcode::kCbrt, HloOpcode::kTan,
                                         HloOpcode::kTanh, HloOpcode::kErf)),
    TestParamsToString);

using BinaryElementwiseTest = TritonSupportTestWithParam;

TEST_P(BinaryElementwiseTest, IsTritonSupportedExecutesCorrectlyForBinaryE) {
  PrimitiveType data_type;
  HloOpcode opcode;
  std::tie(data_type, opcode) = GetParam();
  if (!GetCudaComputeCapability().IsAtLeast(
          se::CudaComputeCapability::AMPERE) &&
      data_type == BF16) {
    GTEST_SKIP() << "No BF16 before Ampere.";
  }

  const std::string kHloTestTemplate = R"(
triton_gemm___computation {
  parameter_0 = f32[92,11]{1,0} parameter(0)
  parameter_1 = $0[11,63]{1,0} parameter(1)
  parameter_2 = $0[11,63]{1,0} parameter(2)
  binary = $0[11,63]{1,0} $1(parameter_1, parameter_2)
  convert = f32[11,63]{1,0} convert(binary)
  ROOT dot = f32[92,63]{1,0} dot(parameter_0, convert),
    lhs_contracting_dims={1}, rhs_contracting_dims={0},
    operand_precision={HIGH, HIGH}
}

ENTRY e {
  parameter_0 = f32[92,11]{1,0} parameter(0)
  parameter_1 = $0[11,63]{1,0} parameter(1)
  parameter_2 = $0[11,63]{1,0} parameter(2)
  ROOT triton_gemm = f32[92,63]{1,0} fusion(parameter_0, parameter_1, parameter_2),
    kind=kCustom, calls=triton_gemm___computation,
    backend_config={"fusion_backend_config":{"kind":"__triton_gemm"}}
})";
  const std::string hlo_test = absl::Substitute(
      kHloTestTemplate, primitive_util::LowercasePrimitiveTypeName(data_type),
      HloOpcodeString(opcode));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_test));
  const HloComputation* computation =
      module->GetComputationWithName("triton_gemm___computation");
  ASSERT_TRUE(computation != nullptr);
  const HloInstruction* instr =
      hlo_query::GetFirstInstructionWithOpcode(*computation, opcode);
  if (IsTritonSupportedInstruction(*instr, GetCudaComputeCapability())) {
    float tolerance = getTolerance(data_type);
    EXPECT_OK(ApplyFloatNormalization(module.get()));
    EXPECT_TRUE(RunAndCompareNoHloPasses(
        std::move(module), ErrorSpec{/*aabs=*/tolerance, /*arel=*/tolerance}));
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
    TestParamsToString);

INSTANTIATE_TEST_SUITE_P(BinaryPREDTestSuite, BinaryElementwiseTest,
                         ::testing::Combine(::testing::Values(PRED),
                                            ::testing::Values(HloOpcode::kAnd,
                                                              HloOpcode::kOr,
                                                              HloOpcode::kXor)),
                         TestParamsToString);
INSTANTIATE_TEST_SUITE_P(
    BinaryMathTestSuite, BinaryElementwiseTest,
    ::testing::Combine(::testing::Values(F16, F32, BF16),
                       ::testing::Values(HloOpcode::kAtan2, HloOpcode::kDivide,
                                         HloOpcode::kPower)),
    TestParamsToString);

using CompareTest = TritonSupportTestWithParam;

TEST_P(CompareTest, IsTritonSupportedExecutesCorrectlyForCompare) {
  PrimitiveType data_type;
  HloOpcode opcode;
  std::tie(data_type, opcode) = GetParam();
  if (!GetCudaComputeCapability().IsAtLeast(
          se::CudaComputeCapability::AMPERE) &&
      data_type == BF16) {
    GTEST_SKIP() << "No BF16 before Ampere.";
  }

  const std::string kHloTestTemplate = R"(
triton_gemm___computation {
  parameter_0 = f32[92,11]{1,0} parameter(0)
  parameter_1 = $0[11,63]{1,0} parameter(1)
  parameter_2 = $0[11,63]{1,0} parameter(2)
  compare = pred[11,63]{1,0} $1(parameter_1, parameter_2), direction=GE
  convert = f32[11,63]{1,0} convert(compare)
  ROOT dot = f32[92,63]{1,0} dot(parameter_0, convert),
    lhs_contracting_dims={1}, rhs_contracting_dims={0},
    operand_precision={HIGH, HIGH}
}

ENTRY e {
  parameter_0 = f32[92,11]{1,0} parameter(0)
  parameter_1 = $0[11,63]{1,0} parameter(1)
  parameter_2 = $0[11,63]{1,0} parameter(2)
  ROOT triton_gemm = f32[92,63]{1,0} fusion(parameter_0, parameter_1, parameter_2),
    kind=kCustom, calls=triton_gemm___computation,
    backend_config={"fusion_backend_config":{"kind":"__triton_gemm"}}
})";
  const std::string hlo_test = absl::Substitute(
      kHloTestTemplate, primitive_util::LowercasePrimitiveTypeName(data_type),
      HloOpcodeString(opcode));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_test));
  const HloComputation* computation =
      module->GetComputationWithName("triton_gemm___computation");
  ASSERT_TRUE(computation != nullptr);
  const HloInstruction* instr =
      hlo_query::GetFirstInstructionWithOpcode(*computation, opcode);
  if (IsTritonSupportedInstruction(*instr, GetCudaComputeCapability())) {
    float tolerance = getTolerance(data_type);
    EXPECT_OK(ApplyFloatNormalization(module.get()));
    EXPECT_TRUE(RunAndCompareNoHloPasses(
        std::move(module), ErrorSpec{/*aabs=*/tolerance, /*arel=*/tolerance}));
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
    TestParamsToString);

using TernaryElementwiseTest = TritonSupportTestWithParam;

TEST_P(TernaryElementwiseTest, IsTritonSupportedExecutesCorrectlyForTernary) {
  PrimitiveType data_type;
  HloOpcode opcode;
  std::tie(data_type, opcode) = GetParam();
  if (!GetCudaComputeCapability().IsAtLeast(
          se::CudaComputeCapability::AMPERE) &&
      data_type == BF16) {
    GTEST_SKIP() << "No BF16 before Ampere.";
  }

  const std::string kHloTestTemplate = R"(
triton_gemm___computation {
  parameter_0 = f32[92,13]{1,0} parameter(0)
  parameter_1 = $0[13,63]{1,0} parameter(1)
  parameter_2 = $0[13,63]{1,0} parameter(2)
  parameter_3 = pred[13,63]{1,0} parameter(3)
  ternary = $0[13,63]{1,0} $1(parameter_3, parameter_1, parameter_2)
  convert = f32[13,63]{1,0} convert(ternary)
  ROOT dot = f32[92,63]{1,0} dot(parameter_0, convert),
    lhs_contracting_dims={1}, rhs_contracting_dims={0},
    operand_precision={HIGH, HIGH}
}

ENTRY e {
  parameter_0 = f32[92,13]{1,0} parameter(0)
  parameter_1 = $0[13,63]{1,0} parameter(1)
  parameter_2 = $0[13,63]{1,0} parameter(2)
  parameter_3 = pred[13,63]{1,0} parameter(3)
  ROOT triton_gemm = f32[92,63]{1,0} fusion(parameter_0, parameter_1, parameter_2, parameter_3),
    kind=kCustom, calls=triton_gemm___computation,
    backend_config={"fusion_backend_config":{"kind":"__triton_gemm"}}
})";
  const std::string hlo_test = absl::Substitute(
      kHloTestTemplate, primitive_util::LowercasePrimitiveTypeName(data_type),
      HloOpcodeString(opcode));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_test));
  const HloComputation* computation =
      module->GetComputationWithName("triton_gemm___computation");
  ASSERT_TRUE(computation != nullptr);
  const HloInstruction* instr =
      hlo_query::GetFirstInstructionWithOpcode(*computation, opcode);
  if (IsTritonSupportedInstruction(*instr, GetCudaComputeCapability())) {
    float tolerance = getTolerance(data_type);
    EXPECT_OK(ApplyFloatNormalization(module.get()));
    EXPECT_TRUE(RunAndCompareNoHloPasses(
        std::move(module), ErrorSpec{/*aabs=*/tolerance, /*arel=*/tolerance}));
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
    TestParamsToString);

using DotTest = TritonSupportTestWithParam;

TEST_P(DotTest, IsTritonSupportedExecutesCorrectlyForDot) {
  PrimitiveType data_type;
  HloOpcode opcode;
  std::tie(data_type, opcode) = GetParam();
  if (!GetCudaComputeCapability().IsAtLeast(
          se::CudaComputeCapability::AMPERE) &&
      data_type == BF16) {
    GTEST_SKIP() << "No BF16 before Ampere.";
  }

  const std::string kHloTestTemplate = R"(
triton_gemm___computation {
  parameter_0 = $0[92,11]{1,0} parameter(0)
  parameter_1 = $0[11,63]{1,0} parameter(1)
  ROOT dot = $0[92,63]{1,0} $1(parameter_0, parameter_1),
    lhs_contracting_dims={1}, rhs_contracting_dims={0}
}

ENTRY e {
  parameter_0 = $0[92,11]{1,0} parameter(0)
  parameter_1 = $0[11,63]{1,0} parameter(1)
  ROOT triton_gemm = $0[92,63]{1,0} fusion(parameter_0, parameter_1), kind=kCustom,
    calls=triton_gemm___computation,
    backend_config={"fusion_backend_config":{"kind":"__triton_gemm"}}
})";
  const std::string hlo_test = absl::Substitute(
      kHloTestTemplate, primitive_util::LowercasePrimitiveTypeName(data_type),
      HloOpcodeString(opcode));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_test));
  const HloComputation* computation =
      module->GetComputationWithName("triton_gemm___computation");
  ASSERT_TRUE(computation != nullptr);
  const HloInstruction* instr =
      hlo_query::GetFirstInstructionWithOpcode(*computation, opcode);
  if (IsTritonSupportedInstruction(*instr, GetCudaComputeCapability())) {
    EXPECT_OK(ApplyFloatNormalization(module.get()));
    EXPECT_TRUE(RunAndCompareNoHloPasses(
        std::move(module), ErrorSpec{/*aabs=*/2e-4, /*arel=*/2e-4}));
  } else {
    const se::DeviceDescription dev_info =
        TestGpuDeviceInfo::RTXA6000DeviceInfo(GetCudaComputeCapability());
    EXPECT_THAT(
        TritonWrapper(*TritonFusionAnalysis::Execute(*computation), "test_fn",
                      computation, GetCudaComputeCapability(), dev_info,
                      config_, &llvm_module_, &EmitMatMul, mlir_context_),
        tsl::testing::StatusIs(
            absl::StatusCode::kInternal,
            ::testing::HasSubstr("Failed to compile Triton kernel")));
  }
}

INSTANTIATE_TEST_SUITE_P(DotTestTestSuite, DotTest,
                         ::testing::Combine(::testing::Values(F16, F32, BF16),
                                            ::testing::Values(HloOpcode::kDot)),
                         TestParamsToString);

TEST_F(TritonSupportTest, UnsupportedDotOutputTypeFailsGracefullyWithTriton) {
  const std::string kHloTest = R"(
triton_gemm___computation {
  parameter_0 = f32[92,11]{1,0} parameter(0)
  parameter_1 = f32[11,63]{1,0} parameter(1)
  ROOT dot = pred[92,63]{1,0} dot(parameter_0, parameter_1),
    lhs_contracting_dims={1}, rhs_contracting_dims={0}
}

ENTRY e {
  parameter_0 = f32[92,11]{1,0} parameter(0)
  parameter_1 = f32[11,63]{1,0} parameter(1)
  ROOT triton_gemm = pred[92,63]{1,0} fusion(parameter_0, parameter_1), kind=kCustom,
    calls=triton_gemm___computation,
    backend_config={"fusion_backend_config":{"kind":"__triton_gemm"}}
})";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> hlo_module,
                          ParseAndReturnVerifiedModule(kHloTest));

  const HloComputation* computation =
      hlo_module->GetComputationWithName("triton_gemm___computation");
  ASSERT_TRUE(computation != nullptr);
  const HloInstruction* instr =
      hlo_query::GetFirstInstructionWithOpcode(*computation, HloOpcode::kDot);
  const se::DeviceDescription dev_info =
      TestGpuDeviceInfo::RTXA6000DeviceInfo(GetCudaComputeCapability());
  EXPECT_THAT(IsTritonSupportedInstruction(*instr, GetCudaComputeCapability())
                  .Explain(),
              ::testing::HasSubstr("Unsupported output data type for Dot op."));
  EXPECT_THAT(
      TritonWrapper(*TritonFusionAnalysis::Execute(*computation), "test_fn",
                    computation, GetCudaComputeCapability(), dev_info, config_,
                    &llvm_module_, &EmitMatMul, mlir_context_),
      tsl::testing::StatusIs(
          absl::StatusCode::kInternal,
          ::testing::HasSubstr("pm.run(triton_module.get()).succeeded()")));
}

TEST_F(TritonSupportTest,
       UnsupportedDotWithMultipleBatchDimensionsFailsGracefullyWithTriton) {
  const std::string kHloTest = R"(
triton_gemm___computation {
  parameter_0 = f32[2,2,2,2]{3,2,1,0} parameter(0)
  parameter_1 = f32[2,2,2,2]{3,2,1,0} parameter(1)
  ROOT dot = f32[2,2,2,2]{3,2,1,0} dot(parameter_0, parameter_1),
    lhs_contracting_dims={3}, lhs_batch_dims={1,0}, rhs_contracting_dims={2},
    rhs_batch_dims={1,0}
}

ENTRY e {
  parameter_0 = f32[2,2,2,2]{3,2,1,0} parameter(0)
  parameter_1 = f32[2,2,2,2]{3,2,1,0} parameter(1)
  ROOT triton_gemm = f32[2,2,2,2]{3,2,1,0} fusion(parameter_0, parameter_1),
    kind=kCustom, calls=triton_gemm___computation,
    backend_config={"fusion_backend_config":{"kind":"__triton_gemm"}}
})";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> hlo_module,
                          ParseAndReturnVerifiedModule(kHloTest));

  const HloComputation* computation =
      hlo_module->GetComputationWithName("triton_gemm___computation");
  ASSERT_TRUE(computation != nullptr);
  const HloInstruction* instr =
      hlo_query::GetFirstInstructionWithOpcode(*computation, HloOpcode::kDot);
  const se::DeviceDescription dev_info =
      TestGpuDeviceInfo::RTXA6000DeviceInfo(GetCudaComputeCapability());
  EXPECT_THAT(IsTritonSupportedInstruction(*instr, GetCudaComputeCapability())
                  .Explain(),
              ::testing::HasSubstr("Multiple batch dimensions"));
  EXPECT_THAT(
      TritonWrapper(*TritonFusionAnalysis::Execute(*computation), "test_fn",
                    computation, GetCudaComputeCapability(), dev_info, config_,
                    &llvm_module_, &EmitMatMul, mlir_context_),
      tsl::testing::StatusIs(absl::StatusCode::kInternal,
                             ::testing::HasSubstr("num_batch_dims <= 1")));
}

TEST_F(TritonSupportTest,
       UnsupportedDotWithNoNonContractingDimensionsFailsGracefullyWithTriton) {
  const std::string kHloTest = R"(
triton_gemm___computation {
  parameter_0 = f32[2]{0} parameter(0)
  parameter_1 = f32[2]{0} parameter(1)
  ROOT dot = f32[] dot(parameter_0, parameter_1),
    lhs_contracting_dims={0}, rhs_contracting_dims={0}
}

ENTRY e {
  parameter_0 = f32[2]{0} parameter(0)
  parameter_1 = f32[2]{0} parameter(1)
  ROOT triton_gemm = f32[] fusion(parameter_0, parameter_1), kind=kCustom,
    calls=triton_gemm___computation,
    backend_config={"fusion_backend_config":{"kind":"__triton_gemm"}}
})";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> hlo_module,
                          ParseAndReturnVerifiedModule(kHloTest));

  const HloComputation* computation =
      hlo_module->GetComputationWithName("triton_gemm___computation");
  ASSERT_TRUE(computation != nullptr);
  const HloInstruction* instr =
      hlo_query::GetFirstInstructionWithOpcode(*computation, HloOpcode::kDot);
  EXPECT_THAT(IsTritonSupportedInstruction(*instr, GetCudaComputeCapability())
                  .Explain(),
              ::testing::HasSubstr("No non-contracting dimensions."));
  EXPECT_THAT(TritonFusionAnalysis::Execute(*computation),
              tsl::testing::StatusIs(
                  absl::StatusCode::kInternal,
                  ::testing::HasSubstr("non_contracting_dims.size() == 1")));
}

using ReduceConstTest = TritonSupportTestWithParam;
TEST_P(ReduceConstTest,
       IsTritonSupportedExecutesCorrectlyForReduceWithConstInit) {
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

triton_softmax_computation {
  parameter_0 = $0[125,127]{1,0} parameter(0)
  multiply_0 = $0[125,127]{1,0} multiply(parameter_0, parameter_0)
  constant_0 = $0[] constant(0)
  reduce = $0[125]{0} $1(multiply_0, constant_0), dimensions={1}, to_apply=add
  broadcast = $0[125,127]{1,0} broadcast(reduce), dimensions={0}
  ROOT multiply = $0[125,127]{1,0} multiply(multiply_0, broadcast)
}

ENTRY main {
  parameter_0 = $0[125,127]{1,0} parameter(0)
  ROOT triton_softmax = $0[125,127]{1,0} fusion(parameter_0),
                          kind=kCustom, calls=triton_softmax_computation,
                          backend_config={"fusion_backend_config":
                                           {"kind":"__triton_softmax"}}
})";
  const std::string hlo_test = absl::Substitute(
      kHloTestTemplate, primitive_util::LowercasePrimitiveTypeName(data_type),
      HloOpcodeString(opcode));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_test));

  const HloComputation* computation =
      module->GetComputationWithName("triton_softmax_computation");
  ASSERT_TRUE(computation != nullptr);
  const HloInstruction* instr =
      hlo_query::GetFirstInstructionWithOpcode(*computation, opcode);
  if (IsTritonSupportedInstruction(*instr, GetCudaComputeCapability())) {
    float tolerance = getTolerance(data_type);
    EXPECT_OK(ApplyFloatNormalization(module.get()));
    EXPECT_TRUE(RunAndCompareNoHloPasses(
        std::move(module), ErrorSpec{/*aabs=*/tolerance, /*arel=*/tolerance}));
  } else {
    const se::DeviceDescription dev_info =
        TestGpuDeviceInfo::RTXA6000DeviceInfo(GetCudaComputeCapability());
    EXPECT_THAT(
        TritonWrapper(*TritonFusionAnalysis::Execute(*computation), "test_fn",
                      computation, GetCudaComputeCapability(), dev_info,
                      config_, &llvm_module_, &EmitSoftMax, mlir_context_),
        tsl::testing::StatusIs(
            absl::StatusCode::kInternal,
            ::testing::HasSubstr("Failed to compile Triton kernel")));
  }
}

INSTANTIATE_TEST_SUITE_P(
    ReduceConstTestSuite, ReduceConstTest,
    ::testing::Combine(::testing::Values(F16, F32, BF16),
                       ::testing::Values(HloOpcode::kReduce)),
    TestParamsToString);

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

triton_softmax_computation {
  parameter_0 = f32[125,127]{1,0} parameter(0)
  multiply_0 = f32[125,127]{1,0} multiply(parameter_0, parameter_0)
  constant_0 = bf16[] constant(0)
  convert_0 = f32[] convert(constant_0)
  reduce = f32[125]{0} reduce(multiply_0, convert_0), dimensions={1}, to_apply=add
  broadcast = f32[125,127]{1,0} broadcast(reduce), dimensions={0}
  ROOT multiply = f32[125,127]{1,0} multiply(multiply_0, broadcast)
}

ENTRY main {
  parameter_0 = f32[125,127]{1,0} parameter(0)
  ROOT triton_softmax = f32[125,127]{1,0} fusion(parameter_0), kind=kCustom,
  calls=triton_softmax_computation,
                        backend_config={"fusion_backend_config":
                        {"kind":"__triton_softmax"}}
})";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> hlo_module,
                          ParseAndReturnVerifiedModule(kHloTest));

  const HloComputation* computation =
      hlo_module->GetComputationWithName("triton_softmax_computation");
  ASSERT_TRUE(computation != nullptr);
  const HloInstruction* instr = hlo_query::GetFirstInstructionWithOpcode(
      *computation, HloOpcode::kReduce);
  EXPECT_TRUE(IsTritonSupportedInstruction(*instr, GetCudaComputeCapability())
                  .CanFuse());
  EXPECT_OK(ApplyFloatNormalization(hlo_module.get()));
  EXPECT_TRUE(RunAndCompareNoHloPasses(
      std::move(hlo_module), ErrorSpec{/*aabs=*/2e-4, /*arel=*/2e-4}));
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

triton_softmax_computation {
  parameter_0 = f32[2,125,127]{2,1,0} parameter(0)
  multiply_0 = f32[2,125,127]{2,1,0} multiply(parameter_0, parameter_0)
  constant_0 = f32[] constant(0)
  reduce = f32[2]{0} reduce(multiply_0, constant_0), dimensions={1,2}, to_apply=add
  broadcast = f32[2,125,127]{2,1,0} broadcast(reduce), dimensions={0}
  ROOT multiply = f32[2,125,127]{2,1,0} multiply(multiply_0, broadcast)
}

ENTRY main {
  parameter_0 = f32[2,125,127]{2,1,0} parameter(0)
  ROOT triton_softmax = f32[2,125,127]{2,1,0} fusion(parameter_0),
                          kind=kCustom, calls=triton_softmax_computation,
                          backend_config={"fusion_backend_config":
                                            {"kind":"__triton_softmax"}}
})";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> hlo_module,
                          ParseAndReturnVerifiedModule(kHloTest));

  const HloComputation* computation =
      hlo_module->GetComputationWithName("triton_softmax_computation");
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
       UnsupportedReduceWithNoneLastReduceDimensionFailsGracefullyWithTriton) {
  const std::string kHloTest = R"(
HloModule t
add {
  Arg_0 = f32[] parameter(0)
  Arg_1 = f32[] parameter(1)
  ROOT add = f32[] add(Arg_0, Arg_1)
}

triton_softmax_computation {
  parameter_0 = f32[2,125,127]{2,1,0} parameter(0)
  multiply_0 = f32[2,125,127]{2,1,0} multiply(parameter_0, parameter_0)
  constant_0 = f32[] constant(0)
  reduce = f32[2,127]{1,0} reduce(multiply_0, constant_0), dimensions={1}, to_apply=add
  broadcast = f32[2,125,127]{2,1,0} broadcast(reduce), dimensions={0,2}
  ROOT multiply = f32[2,125,127]{2,1,0} multiply(multiply_0, broadcast)
}

ENTRY main {
  parameter_0 = f32[2,125,127]{2,1,0} parameter(0)
  ROOT triton_softmax = f32[2,125,127]{2,1,0} fusion(parameter_0),
                          kind=kCustom, calls=triton_softmax_computation,
                          backend_config={"fusion_backend_config":
                                            {"kind":"__triton_softmax"}}
})";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> hlo_module,
                          ParseAndReturnVerifiedModule(kHloTest));

  const HloComputation* computation =
      hlo_module->GetComputationWithName("triton_softmax_computation");
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

triton_softmax_computation {
  parameter_0 = f32[125,127] parameter(0)
  multiply_0 = f32[125,127]{1,0} multiply(parameter_0, parameter_0)
  constant_0 = f32[] constant(0)
  tuple_0 = (f32[125]{0}, f32[125]{0}) reduce(multiply_0, multiply_0, constant_0, constant_0), dimensions={1}, to_apply=add
  reduce = f32[125]{0} get-tuple-element(tuple_0), index=0
  broadcast = f32[125,127]{1,0} broadcast(reduce), dimensions={0}
  ROOT multiply = f32[125,127]{1,0} multiply(multiply_0, broadcast)
}

ENTRY main {
  parameter_0 = f32[125,127]{1,0} parameter(0)
  ROOT triton_softmax = f32[125,127]{1,0} fusion(parameter_0),
                          kind=kCustom, calls=triton_softmax_computation,
                          backend_config={"fusion_backend_config":
                                           {"kind":"__triton_softmax"}}
})";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> hlo_module,
                          ParseAndReturnVerifiedModule(kHloTest));

  const HloComputation* computation =
      hlo_module->GetComputationWithName("triton_softmax_computation");
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

triton_softmax_computation {
  parameter_0 = f32[125,127]{1,0} parameter(0)
  multiply_0 = f32[125,127]{1,0} multiply(parameter_0, parameter_0)
  init = f32[] parameter(1)
  reduce = f32[125]{0} reduce(multiply_0, init), dimensions={1}, to_apply=add
  broadcast = f32[125,127]{1,0} broadcast(reduce), dimensions={0}
  ROOT multiply = f32[125,127]{1,0} multiply(multiply_0, broadcast)
}

ENTRY main {
  parameter_0 = f32[125,127]{1,0} parameter(0)
  parameter_1 = f32[] parameter(1)
  ROOT triton_softmax = f32[125,127]{1,0} fusion(parameter_0, parameter_1),
                          kind=kCustom, calls=triton_softmax_computation,
                        backend_config={"fusion_backend_config":
                                         {"kind":"__triton_softmax"}}
})";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> hlo_module,
                          ParseAndReturnVerifiedModule(kHloTest));

  const HloComputation* computation =
      hlo_module->GetComputationWithName("triton_softmax_computation");
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
                    computation, GetCudaComputeCapability(), dev_info, config_,
                    &llvm_module_, &EmitSoftMax, mlir_context_),
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

triton_softmax_computation {
  parameter_0 = f32[125,127]{1,0} parameter(0)
  multiply_0 = f32[125,127]{1,0} multiply(parameter_0, parameter_0)
  constant_0 = f32[] constant(0)
  reduce = f32[125]{0} reduce(multiply_0, constant_0), dimensions={1}, to_apply=custom_call
  broadcast = f32[125,127]{1,0} broadcast(reduce), dimensions={0}
  ROOT multiply = f32[125,127]{1,0} multiply(multiply_0, broadcast)
}

ENTRY main {
  parameter_0 = f32[125,127]{1,0} parameter(0)
  ROOT triton_softmax = f32[125,127]{1,0} fusion(parameter_0),
                          kind=kCustom, calls=triton_softmax_computation,
                          backend_config={"fusion_backend_config":
                                         {"kind":"__triton_softmax"}}
})";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> hlo_module,
                          ParseAndReturnVerifiedModule(kHloTest));

  const HloComputation* computation =
      hlo_module->GetComputationWithName("triton_softmax_computation");
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
                    computation, GetCudaComputeCapability(), dev_info, config_,
                    &llvm_module_, &EmitSoftMax, mlir_context_),
      tsl::testing::StatusIs(absl::StatusCode::kInvalidArgument,
                             ::testing::HasSubstr("Unsupported operation")));
}
}  // namespace
}  // namespace gpu
}  // namespace xla

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

#include "xla/service/gpu/gpu_float_support.h"

#include <memory>
#include <string>

#include <gtest/gtest.h>
#include "absl/log/check.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/strings/substitute.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/testlib/test_helpers.h"
#include "xla/hlo/transforms/simplifiers/float_normalization.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/service/hlo_verifier.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/device_description.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/statusor.h"

namespace xla::gpu {
namespace {

class FloatSupportTest : public HloHardwareIndependentTestBase {
 protected:
  FloatSupportTest()
      : HloHardwareIndependentTestBase(
            /*verifier_layout_sensitive=*/false,
            /*allow_mixed_precision_in_hlo_verifier=*/true) {}

  bool Normalize(HloModule* module, se::GpuComputeCapability cc,
                 PrimitiveType low_precision_type,
                 PrimitiveType high_precision_type) {
    GpuFloatSupport float_support(cc, low_precision_type, high_precision_type);
    FloatNormalization normalization(&float_support);
    absl::StatusOr<bool> result = normalization.Run(module);
    EXPECT_IS_OK(result.status());

    HloVerifier verifier(/*layout_sensitive=*/false,
                         /*allow_mixed_precision=*/true);
    EXPECT_IS_OK(verifier.Run(module).status());

    return result.value();
  }

  std::unique_ptr<HloComputation> CreateComputation(PrimitiveType lhs_type,
                                                    PrimitiveType rhs_type,
                                                    PrimitiveType result_type) {
    auto builder = HloComputation::Builder(TestName());
    Shape lhs_shape = ShapeUtil::MakeShape(lhs_type, {3, 3});
    Shape rhs_shape = ShapeUtil::MakeShape(rhs_type, {3, 3});
    Shape result_shape = ShapeUtil::MakeShape(result_type, {3, 3});

    HloInstruction* a = builder.AddInstruction(
        HloInstruction::CreateParameter(0, lhs_shape, "a"));
    HloInstruction* b = builder.AddInstruction(
        HloInstruction::CreateParameter(1, rhs_shape, "b"));
    PrecisionConfig precision_config;
    DotDimensionNumbers dot_dnums;
    dot_dnums.add_lhs_contracting_dimensions(1);
    dot_dnums.add_rhs_contracting_dimensions(1);

    builder.AddInstruction(HloInstruction::CreateDot(
        result_shape, a, b, dot_dnums, precision_config));

    return builder.Build();
  }

  void TestDotConversion(PrimitiveType lhs_type, PrimitiveType rhs_type,
                         PrimitiveType result_type, se::GpuComputeCapability cc,
                         bool should_convert_lhs, bool should_convert_rhs,
                         PrimitiveType low_precision_type,
                         PrimitiveType high_precision_type = F16) {
    auto module = CreateNewVerifiedModule();
    auto computation = module->AddEntryComputation(
        CreateComputation(lhs_type, rhs_type, result_type));

    EXPECT_EQ(
        Normalize(module.get(), cc, low_precision_type, high_precision_type),
        should_convert_lhs || should_convert_rhs);

    EXPECT_EQ(computation->root_instruction()->opcode(), HloOpcode::kDot);
    EXPECT_EQ(computation->root_instruction()->operand(0)->opcode() ==
                  HloOpcode::kConvert,
              should_convert_lhs);
    EXPECT_EQ(computation->root_instruction()->operand(1)->opcode() ==
                  HloOpcode::kConvert,
              should_convert_rhs);
  }

  void TestTritonFusedDot(PrimitiveType lhs_type, PrimitiveType rhs_type,
                          PrimitiveType result_type,
                          se::GpuComputeCapability cc, bool should_convert_lhs,
                          bool should_convert_rhs,
                          PrimitiveType low_precision_type,
                          PrimitiveType high_precision_type = F16) {
    auto module = CreateNewVerifiedModule();

    auto computation = module->AddComputationAndUnifyNamesAndIds(
        CreateComputation(lhs_type, rhs_type, result_type), /*is_entry=*/false);

    Shape lhs_shape = ShapeUtil::MakeShape(lhs_type, {3, 3});
    Shape rhs_shape = ShapeUtil::MakeShape(rhs_type, {3, 3});
    Shape result_shape = ShapeUtil::MakeShape(result_type, {3, 3});

    auto builder = HloComputation::Builder("main");
    HloInstruction* a = builder.AddInstruction(
        HloInstruction::CreateParameter(0, lhs_shape, "a"));
    HloInstruction* b = builder.AddInstruction(
        HloInstruction::CreateParameter(1, rhs_shape, "b"));
    HloInstruction* fusion =
        builder.AddInstruction(HloInstruction::CreateFusion(
            result_shape, HloInstruction::FusionKind::kCustom, {a, b},
            computation));
    GpuBackendConfig config;
    config.mutable_fusion_backend_config()->set_kind(
        std::string(kTritonGemmFusionKind));
    CHECK_OK(fusion->set_backend_config(config));

    module->AddEntryComputation(builder.Build());

    EXPECT_EQ(
        Normalize(module.get(), cc, low_precision_type, high_precision_type),
        should_convert_lhs || should_convert_rhs);
    EXPECT_EQ(computation->root_instruction()->opcode(), HloOpcode::kDot);
    EXPECT_EQ(computation->root_instruction()->operand(0)->opcode() ==
                  HloOpcode::kConvert,
              should_convert_lhs);
    EXPECT_EQ(computation->root_instruction()->operand(1)->opcode() ==
                  HloOpcode::kConvert,
              should_convert_rhs);
  }
};

TEST_F(FloatSupportTest, ShouldAlwaysConvertFp8Dot) {
  TestDotConversion(F8E4M3FN, F8E4M3FN, F16,
                    se::CudaComputeCapability::Hopper(),
                    /*should_convert_lhs=*/true,
                    /*should_convert_rhs=*/true, F8E4M3FN);

  TestDotConversion(F8E4M3FN, F8E4M3FN, F32,
                    se::CudaComputeCapability::Hopper(),
                    /*should_convert_lhs=*/true,
                    /*should_convert_rhs=*/true, F8E4M3FN);

  TestDotConversion(F8E4M3FN, F8E4M3FN, F16,
                    se::CudaComputeCapability::Ampere(),
                    /*should_convert_lhs=*/true,
                    /*should_convert_rhs=*/true, F8E4M3FN);

  TestDotConversion(F8E4M3FN, F8E4M3FN, F32,
                    se::CudaComputeCapability::Hopper(),
                    /*should_convert_lhs=*/true,
                    /*should_convert_rhs=*/true, F8E4M3FN);

  TestDotConversion(F8E5M2, F8E5M2, F16, se::CudaComputeCapability::Ampere(),
                    /*should_convert_lhs=*/true,
                    /*should_convert_rhs=*/true, F8E5M2);

  TestDotConversion(F8E5M2, F8E5M2, F32, se::CudaComputeCapability::Ampere(),
                    /*should_convert_lhs=*/true,
                    /*should_convert_rhs=*/true, F8E5M2);

  TestDotConversion(F8E5M2, F8E4M3FN, F16, se::CudaComputeCapability::Hopper(),
                    /*should_convert_lhs=*/true,
                    /*should_convert_rhs=*/false, F8E5M2);

  TestDotConversion(F8E5M2, F8E4M3FN, F32, se::CudaComputeCapability::Hopper(),
                    /*should_convert_lhs=*/true,
                    /*should_convert_rhs=*/false, F8E5M2);

  TestDotConversion(F8E5M2, F16, F16, se::CudaComputeCapability::Hopper(),
                    /*should_convert_lhs=*/true,
                    /*should_convert_rhs=*/false, F8E5M2);

  TestDotConversion(F8E5M2, F16, F32, se::CudaComputeCapability::Hopper(),
                    /*should_convert_lhs=*/true,
                    /*should_convert_rhs=*/false, F8E5M2);
}

TEST_F(FloatSupportTest, ShouldConvertTritonUnsupportedFp8Dot) {
  TestTritonFusedDot(F8E4M3FN, F8E4M3FN, F16,
                     se::CudaComputeCapability::Hopper(),
                     /*should_convert_lhs=*/true,
                     /*should_convert_rhs=*/true, F8E4M3FN);

  TestTritonFusedDot(F8E4M3FN, F8E4M3FN, F32,
                     se::CudaComputeCapability::Hopper(),
                     /*should_convert_lhs=*/false,
                     /*should_convert_rhs=*/false, F8E4M3FN);

  TestTritonFusedDot(F8E4M3FN, F8E4M3FN, F16,
                     se::CudaComputeCapability::Ampere(),
                     /*should_convert_lhs=*/true,
                     /*should_convert_rhs=*/true, F8E4M3FN);

  TestTritonFusedDot(F8E4M3FN, F8E4M3FN, F32,
                     se::CudaComputeCapability::Hopper(),
                     /*should_convert_lhs=*/false,
                     /*should_convert_rhs=*/false, F8E4M3FN);

  TestTritonFusedDot(F8E5M2, F8E5M2, F16, se::CudaComputeCapability::Ampere(),
                     /*should_convert_lhs=*/true,
                     /*should_convert_rhs=*/true, F8E5M2);

  TestTritonFusedDot(F8E5M2, F8E5M2, F32, se::CudaComputeCapability::Ampere(),
                     /*should_convert_lhs=*/true,
                     /*should_convert_rhs=*/true, F8E5M2);

  TestTritonFusedDot(F8E5M2, F8E4M3FN, F16, se::CudaComputeCapability::Hopper(),
                     /*should_convert_lhs=*/true,
                     /*should_convert_rhs=*/false, F8E5M2);

  TestTritonFusedDot(F8E5M2, F8E4M3FN, F32, se::CudaComputeCapability::Hopper(),
                     /*should_convert_lhs=*/false,
                     /*should_convert_rhs=*/false, F8E5M2);

  TestTritonFusedDot(F8E5M2, F16, F16, se::CudaComputeCapability::Hopper(),
                     /*should_convert_lhs=*/true,
                     /*should_convert_rhs=*/false, F8E5M2);

  TestTritonFusedDot(F8E5M2, F16, F32, se::CudaComputeCapability::Hopper(),
                     /*should_convert_lhs=*/true,
                     /*should_convert_rhs=*/false, F8E5M2);
}

TEST_F(FloatSupportTest, ShouldKeepBf16OnAmpere) {
  TestDotConversion(BF16, BF16, F32, se::CudaComputeCapability::Ampere(),
                    /*should_convert_lhs=*/false,
                    /*should_convert_rhs=*/false, BF16);
}

TEST_F(FloatSupportTest, ShouldKeepBf16OnHopper) {
  TestDotConversion(BF16, BF16, F32, se::CudaComputeCapability::Hopper(),
                    /*should_convert_lhs=*/false,
                    /*should_convert_rhs=*/false, BF16);
}

TEST_F(FloatSupportTest, Bf16ReducePrecisionIsNotNormalized) {
  auto cc = se::CudaComputeCapability::Ampere();
  constexpr absl::string_view kHloModule = R"(
HloModule m

ENTRY main {
  p0 = bf16[] parameter(0)
  ROOT r = bf16[] reduce-precision(p0), exponent_bits=8, mantissa_bits=7
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(kHloModule));
  EXPECT_FALSE(Normalize(module.get(), cc, BF16, F32));
}

TEST_F(FloatSupportTest, Bf16ExpIsNotNormalized) {
  auto cc = se::CudaComputeCapability::Ampere();
  constexpr absl::string_view kHloModule = R"(
HloModule m

ENTRY main {
  p0 = bf16[] parameter(0)
  ROOT r = bf16[] exponential(p0)
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(kHloModule));
  EXPECT_FALSE(Normalize(module.get(), cc, BF16, F32));
}

TEST_F(FloatSupportTest, Bf16LogIsNotNormalized) {
  auto cc = se::CudaComputeCapability::Ampere();
  constexpr absl::string_view kHloModule = R"(
HloModule m

ENTRY main {
  p0 = bf16[] parameter(0)
  ROOT r = bf16[] log(p0)
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(kHloModule));
  EXPECT_FALSE(Normalize(module.get(), cc, BF16, F32));
}

TEST_F(FloatSupportTest,
       BF16ReductionOnHopperIsOnlyNormalizedIfReducerIsUnsupported) {
  auto cc = se::CudaComputeCapability::Hopper();
  constexpr absl::string_view kHloModuleTemplate = R"(
HloModule m

reducer {
  p0 = bf16[] parameter(0)
  p1 = bf16[] parameter(1)
  ROOT reducer = bf16[] $0(p0, p1)
}

ENTRY main {
  p0 = bf16[1024] parameter(0)
  init = bf16[] constant(1337)
  ROOT r = bf16[] reduce(p0, init), dimensions={0}, to_apply=reducer
})";

  // add.bf16 was added in Hopper.
  TF_ASSERT_OK_AND_ASSIGN(auto module_with_supported_reducer,
                          ParseAndReturnVerifiedModule(
                              absl::Substitute(kHloModuleTemplate, "add")));
  EXPECT_FALSE(Normalize(module_with_supported_reducer.get(), cc, BF16, F32));

  // There is no bf16 instruction for divide, however.
  TF_ASSERT_OK_AND_ASSIGN(auto module_with_unsupported_reducer,
                          ParseAndReturnVerifiedModule(
                              absl::Substitute(kHloModuleTemplate, "divide")));
  EXPECT_TRUE(Normalize(module_with_unsupported_reducer.get(), cc, BF16, F32));
}

TEST_F(FloatSupportTest, BF16LogAndExpOnRocmIsNormalized) {
  auto cc = se::RocmComputeCapability();
  constexpr absl::string_view kHloModule = R"(
HloModule module

ENTRY main {
      p0 = bf16[4] parameter(0)
      ROOT r = bf16[4] $0(p0)
})";

  TF_ASSERT_OK_AND_ASSIGN(
      auto module_log,
      ParseAndReturnVerifiedModule(absl::Substitute(kHloModule, "log")));
  EXPECT_TRUE(Normalize(module_log.get(), cc, BF16, F32));

  TF_ASSERT_OK_AND_ASSIGN(auto module_exp,
                          ParseAndReturnVerifiedModule(
                              absl::Substitute(kHloModule, "exponential")));
  EXPECT_TRUE(Normalize(module_exp.get(), cc, BF16, F32));
}

}  // namespace
}  // namespace xla::gpu

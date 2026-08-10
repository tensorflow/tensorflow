/* Copyright 2026 The OpenXLA Authors.

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

#include "xla/backends/gpu/transforms/conv_fp8_fallback.h"

// Tests for the FP8 -> BF16 convolution-fusion fallback pass.
//
// The ConvFp8FallbackRewriteTest cases exercise the pure HLO rewrite helper
// (RewriteFp8FusionToBf16) and probe no cuDNN plans.
//
// The ConvFp8FallbackDevicelessTest cases run the full pass against
// stream_executor::DeviceDescriptions built from checked-in target-config
// specs and open no GPU; they need a loadable host cuDNN >= 9.8 (like the
// SupportsFusionDeviceless probe they drive) and skip otherwise. The sm_120
// cases additionally skip on cuDNN runtimes < 9.19, whose deviceless
// heuristics cannot probe Blackwell-generation targets.
//
// The fusions under test are produced by the real ConvKindAssignment +
// ConvFusionRewriter passes, so they cannot drift from pipeline output.

#include <memory>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/backends/gpu/target_config/target_config.h"
#include "xla/backends/gpu/transforms/conv_fusion_rewriter.h"
#include "xla/backends/gpu/transforms/conv_kind_assignment.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/testlib/verified_hlo_module.h"
#include "xla/primitive_util.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/cuda/cuda_dnn.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/device_description.pb.h"
#include "xla/stream_executor/dnn.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace gpu {
namespace {

namespace se = ::stream_executor;

class ConvFp8FallbackTestBase : public HloHardwareIndependentTestBase {
 protected:
  // Builds a deviceless GpuTargetConfig for the given GPU model from its
  // checked-in target-config spec (no StreamExecutor / GPU required).
  static absl::StatusOr<GpuTargetConfig> DevicelessTargetConfig(
      GpuModel model) {
    ABSL_ASSIGN_OR_RETURN(stream_executor::GpuTargetConfigProto proto,
                     GetGpuTargetConfig(model));
    return GpuTargetConfig::FromProto(proto);
  }

  // Runs the conv-fusion pipeline passes on `hlo_text` and returns the module
  // containing the resulting __cudnn$fusion.
  absl::StatusOr<std::unique_ptr<VerifiedHloModule>> BuildConvFusionModule(
      absl::string_view hlo_text, const GpuTargetConfig& target_config) {
    const se::DeviceDescription& device_info = target_config.device_description;
    ABSL_ASSIGN_OR_RETURN(std::unique_ptr<VerifiedHloModule> module,
                     ParseAndReturnVerifiedModule(hlo_text));
    ConvKindAssignment kind_assignment(device_info.gpu_compute_capability(),
                                       target_config.dnn_version_info);
    ABSL_RETURN_IF_ERROR(RunHloPass(&kind_assignment, module.get()).status());
    ConvFusionRewriter rewriter(device_info);
    ABSL_RETURN_IF_ERROR(RunHloPass(&rewriter, module.get()).status());
    if (FindCudnnFusion(*module) == nullptr) {
      return absl::InternalError("No __cudnn$fusion produced.");
    }
    return module;
  }

  static int CountCudnnFusions(const HloModule& module) {
    int count = 0;
    for (const HloComputation* computation :
         module.MakeNonfusionComputations()) {
      for (const HloInstruction* instruction : computation->instructions()) {
        if (instruction->opcode() != HloOpcode::kFusion) {
          continue;
        }
        auto config = instruction->backend_config<GpuBackendConfig>();
        count += config.ok() &&
                 config->fusion_backend_config().kind() == kCuDnnFusionKind;
      }
    }
    return count;
  }

  static HloFusionInstruction* FindCudnnFusion(const HloModule& module) {
    for (HloInstruction* instruction :
         module.entry_computation()->instructions()) {
      if (instruction->opcode() != HloOpcode::kFusion) {
        continue;
      }
      auto config = instruction->backend_config<GpuBackendConfig>();
      if (config.ok() &&
          config->fusion_backend_config().kind() == kCuDnnFusionKind) {
        return Cast<HloFusionInstruction>(instruction);
      }
    }
    return nullptr;
  }

  static bool FusionContainsF8(const HloFusionInstruction& fusion) {
    for (const HloInstruction* instruction :
         fusion.fused_instructions_computation()->instructions()) {
      bool contains_f8 = false;
      ShapeUtil::ForEachSubshape(
          instruction->shape(), [&](const Shape& subshape, const ShapeIndex&) {
            contains_f8 |= subshape.IsArray() &&
                           primitive_util::IsF8Type(subshape.element_type());
          });
      if (contains_f8) {
        return true;
      }
    }
    return false;
  }
};

// Bare FP8 convolution; the conv result itself is the fusion output.
constexpr absl::string_view kF8BareConvHlo = R"(
  ENTRY e {
    input = f8e4m3fn[1,6,6,128] parameter(0)
    filter = f8e4m3fn[16,3,3,128] parameter(1)
    ROOT conv = f8e4m3fn[1,6,6,16] convolution(input, filter),
      window={size=3x3 pad=1_1x1_1}, dim_labels=b01f_o01i->b01f
  })";

// FP8 convolution accumulating in f32 with a scale / clamp / quantize
// epilogue, in the shape ConvFusionRewriter fuses.
constexpr absl::string_view kF8ScaleConvHlo = R"(
  ENTRY e {
    input = f8e4m3fn[1,6,6,128] parameter(0)
    filter = f8e4m3fn[16,3,3,128] parameter(1)
    input_f32 = f32[1,6,6,128] convert(input)
    filter_f32 = f32[16,3,3,128] convert(filter)
    z_scale = f32[] parameter(2)
    z_scale_bcast = f32[1,6,6,16] broadcast(z_scale), dimensions={}
    conv = f32[1,6,6,16] convolution(input_f32, filter_f32),
      window={size=3x3 pad=1_1x1_1}, dim_labels=b01f_o01i->b01f
    scaled = f32[1,6,6,16] multiply(conv, z_scale_bcast)
    lower = f32[] constant(-448.)
    lower_bcast = f32[1,6,6,16] broadcast(lower), dimensions={}
    upper = f32[] constant(448.)
    upper_bcast = f32[1,6,6,16] broadcast(upper), dimensions={}
    clamped = f32[1,6,6,16] clamp(lower_bcast, scaled, upper_bcast)
    ROOT out = f8e4m3fn[1,6,6,16] convert(clamped)
  })";

// FP8 conv -> scale -> f8 output, plus an amax of the f32 tensor as a second
// (scalar) fusion output.
constexpr absl::string_view kF8ScaleAmaxConvHlo = R"(
  apply {
    a = f32[] parameter(0)
    b = f32[] parameter(1)
    ROOT m = f32[] maximum(a, b)
  }

  ENTRY e {
    input = f8e4m3fn[1,6,6,128] parameter(0)
    filter = f8e4m3fn[16,3,3,128] parameter(1)
    input_f32 = f32[1,6,6,128] convert(input)
    filter_f32 = f32[16,3,3,128] convert(filter)
    z_scale = f32[] parameter(2)
    z_scale_bcast = f32[1,6,6,16] broadcast(z_scale), dimensions={}
    conv = f32[1,6,6,16] convolution(input_f32, filter_f32),
      window={size=3x3 pad=1_1x1_1}, dim_labels=b01f_o01i->b01f
    scaled = f32[1,6,6,16] multiply(conv, z_scale_bcast)
    lower = f32[] constant(-448.)
    lower_bcast = f32[1,6,6,16] broadcast(lower), dimensions={}
    upper = f32[] constant(448.)
    upper_bcast = f32[1,6,6,16] broadcast(upper), dimensions={}
    clamped = f32[1,6,6,16] clamp(lower_bcast, scaled, upper_bcast)
    out = f8e4m3fn[1,6,6,16] convert(clamped)
    abs_conv = f32[1,6,6,16] abs(conv)
    zero = f32[] constant(0)
    amax = f32[] reduce(abs_conv, zero), dimensions={0,1,2,3},
      to_apply=apply
    ROOT result = (f8e4m3fn[1,6,6,16], f32[]) tuple(out, amax)
  })";

// Plain f32 convolution — never a fallback candidate.
constexpr absl::string_view kF32ConvHlo = R"(
  ENTRY e {
    input = f32[8,28,28,64] parameter(0)
    filter = f32[16,3,3,64] parameter(1)
    ROOT conv = f32[8,28,28,16] convolution(input, filter),
      window={size=3x3 pad=1_1x1_1}, dim_labels=b01f_o01i->b01f
  })";

// Grouped FP8 convolutions on sm_120 (RTX 5090 / RTX 6000 Pro), keyed on
// channels per group: 16 channels/group has cuDNN heuristic-mode-A plans,
// fewer channels per group have none the conv-fusion pipeline can execute
// (see openxla/xla#40021).
constexpr absl::string_view kGroupedF8Conv16ChannelsHlo = R"(
  ENTRY e {
    input = f8e4m3fn[1,16,16,32] parameter(0)
    filter = f8e4m3fn[32,3,3,16] parameter(1)
    ROOT conv = f8e4m3fn[1,16,16,32] convolution(input, filter),
      window={size=3x3 pad=1_1x1_1}, dim_labels=b01f_o01i->b01f,
      feature_group_count=2
  })";

constexpr absl::string_view kGroupedF8Conv8ChannelsHlo = R"(
  ENTRY e {
    input = f8e4m3fn[1,16,16,32] parameter(0)
    filter = f8e4m3fn[32,3,3,8] parameter(1)
    ROOT conv = f8e4m3fn[1,16,16,32] convolution(input, filter),
      window={size=3x3 pad=1_1x1_1}, dim_labels=b01f_o01i->b01f,
      feature_group_count=4
  })";

constexpr absl::string_view kGroupedF8Conv2ChannelsHlo = R"(
  ENTRY e {
    input = f8e4m3fn[1,16,16,32] parameter(0)
    filter = f8e4m3fn[32,3,3,2] parameter(1)
    ROOT conv = f8e4m3fn[1,16,16,32] convolution(input, filter),
      window={size=3x3 pad=1_1x1_1}, dim_labels=b01f_o01i->b01f,
      feature_group_count=16
  })";

// ----------------------------------------------------------------------
// Pure rewrite tests: RewriteFp8FusionToBf16, no cuDNN plan probing.
// ----------------------------------------------------------------------

using ConvFp8FallbackRewriteTest = ConvFp8FallbackTestBase;

TEST_F(ConvFp8FallbackRewriteTest, BareF8ConvBecomesBf16WithF8Boundary) {
  ASSERT_OK_AND_ASSIGN(GpuTargetConfig target_config,
                       DevicelessTargetConfig(GpuModel::H100_SXM));
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       BuildConvFusionModule(kF8BareConvHlo, target_config));
  HloFusionInstruction* fusion = FindCudnnFusion(*module);
  ASSERT_OK_AND_ASSIGN(HloFusionInstruction * bf16_fusion,
                       RewriteFp8FusionToBf16(fusion));

  EXPECT_FALSE(FusionContainsF8(*bf16_fusion));
  // Operands are converts of the original f8 parameters to bf16.
  for (const HloInstruction* operand : bf16_fusion->operands()) {
    EXPECT_EQ(operand->opcode(), HloOpcode::kConvert);
    EXPECT_EQ(operand->shape().element_type(), BF16);
    EXPECT_TRUE(
        primitive_util::IsF8Type(operand->operand(0)->shape().element_type()));
  }
  // The fusion output is bf16, converted back to the original f8 type.
  EXPECT_EQ(bf16_fusion->shape().element_type(), BF16);
  const HloInstruction* root = module->entry_computation()->root_instruction();
  EXPECT_EQ(root->opcode(), HloOpcode::kConvert);
  EXPECT_EQ(root->shape().element_type(), F8E4M3FN);
  EXPECT_EQ(root->operand(0), bf16_fusion);
  // The rewritten module still verifies (shapes are consistent).
  EXPECT_OK(verifier().Run(module.get()).status());
}

TEST_F(ConvFp8FallbackRewriteTest, ScaleEpilogueKeepsF32InternalsAndScale) {
  ASSERT_OK_AND_ASSIGN(GpuTargetConfig target_config,
                       DevicelessTargetConfig(GpuModel::H100_SXM));
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       BuildConvFusionModule(kF8ScaleConvHlo, target_config));
  HloFusionInstruction* fusion = FindCudnnFusion(*module);
  const int original_operand_count = fusion->operand_count();
  ASSERT_OK_AND_ASSIGN(HloFusionInstruction * bf16_fusion,
                       RewriteFp8FusionToBf16(fusion));

  EXPECT_FALSE(FusionContainsF8(*bf16_fusion));
  EXPECT_EQ(bf16_fusion->operand_count(), original_operand_count);
  // The f32 scale operand is passed through unconverted.
  bool has_f32_operand = false;
  for (const HloInstruction* operand : bf16_fusion->operands()) {
    has_f32_operand |= operand->shape().element_type() == F32;
  }
  EXPECT_TRUE(has_f32_operand);
  // The convolution inside still accumulates in f32.
  bool found_f32_conv = false;
  for (const HloInstruction* instruction :
       bf16_fusion->fused_instructions_computation()->instructions()) {
    if (instruction->opcode() == HloOpcode::kConvolution) {
      found_f32_conv = instruction->shape().element_type() == F32;
    }
  }
  EXPECT_TRUE(found_f32_conv);
  EXPECT_OK(verifier().Run(module.get()).status());
}

TEST_F(ConvFp8FallbackRewriteTest, TwoOutputAmaxFusionOnlyConvertsF8Output) {
  ASSERT_OK_AND_ASSIGN(GpuTargetConfig target_config,
                       DevicelessTargetConfig(GpuModel::H100_SXM));
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<VerifiedHloModule> module,
      BuildConvFusionModule(kF8ScaleAmaxConvHlo, target_config));
  HloFusionInstruction* fusion = FindCudnnFusion(*module);
  ASSERT_TRUE(fusion->shape().IsTuple());
  ASSERT_OK_AND_ASSIGN(HloFusionInstruction * bf16_fusion,
                       RewriteFp8FusionToBf16(fusion));

  EXPECT_FALSE(FusionContainsF8(*bf16_fusion));
  ASSERT_TRUE(bf16_fusion->shape().IsTuple());
  EXPECT_EQ(bf16_fusion->shape().tuple_shapes(0).element_type(), BF16);
  // The f32 amax output keeps its type.
  EXPECT_EQ(bf16_fusion->shape().tuple_shapes(1).element_type(), F32);
  EXPECT_OK(verifier().Run(module.get()).status());
}

// ----------------------------------------------------------------------
// Full-pass deviceless tests: probe verdicts drive the rewrite decision.
// ----------------------------------------------------------------------

class ConvFp8FallbackDevicelessTest : public ConvFp8FallbackTestBase {
 protected:
  void SetUp() override {
    if (!se::gpu::SupportsDevicelessDeviceProperties()) {
      GTEST_SKIP() << "cuDNN runtime < 9.8 does not support deviceless "
                      "DeviceProperties.";
    }
  }
};

TEST_F(ConvFp8FallbackDevicelessTest, F32ConvIsNotModified) {
  ASSERT_OK_AND_ASSIGN(GpuTargetConfig target_config,
                       DevicelessTargetConfig(GpuModel::H100_SXM));
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       BuildConvFusionModule(kF32ConvHlo, target_config));
  ConvFp8Fallback pass(target_config.device_description);
  ASSERT_OK_AND_ASSIGN(bool changed, RunHloPass(&pass, module.get()));
  EXPECT_FALSE(changed);
}

TEST_F(ConvFp8FallbackDevicelessTest, Fp8ConvIsKeptOnH100) {
  ASSERT_OK_AND_ASSIGN(GpuTargetConfig target_config,
                       DevicelessTargetConfig(GpuModel::H100_SXM));
  for (absl::string_view hlo :
       {kF8BareConvHlo, kF8ScaleConvHlo, kF8ScaleAmaxConvHlo}) {
    ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                         BuildConvFusionModule(hlo, target_config));
    ConvFp8Fallback pass(target_config.device_description);
    ASSERT_OK_AND_ASSIGN(bool changed, RunHloPass(&pass, module.get()));
    EXPECT_FALSE(changed);
    EXPECT_TRUE(FusionContainsF8(*FindCudnnFusion(*module)));
  }
}

// FP8 convolutions require sm_89+; on an A100 (sm_80, BF16-capable) target
// the pass must rewrite the fusion to BF16. The fusion is constructed with
// an FP8-capable config, as models are: the fallback exists precisely for
// fusions whose FP8 form the target GPU cannot run.
TEST_F(ConvFp8FallbackDevicelessTest, Fp8ConvFallsBackToBf16OnA100) {
  ASSERT_OK_AND_ASSIGN(GpuTargetConfig construction_config,
                       DevicelessTargetConfig(GpuModel::H100_SXM));
  ASSERT_OK_AND_ASSIGN(GpuTargetConfig target_config,
                       DevicelessTargetConfig(GpuModel::A100_SXM_80));
  for (absl::string_view hlo :
       {kF8BareConvHlo, kF8ScaleConvHlo, kF8ScaleAmaxConvHlo}) {
    ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                         BuildConvFusionModule(hlo, construction_config));
    ConvFp8Fallback pass(target_config.device_description);
    ASSERT_OK_AND_ASSIGN(bool changed, RunHloPass(&pass, module.get()));
    EXPECT_TRUE(changed);
    HloFusionInstruction* fusion = FindCudnnFusion(*module);
    ASSERT_NE(fusion, nullptr);
    EXPECT_FALSE(FusionContainsF8(*fusion));
    EXPECT_OK(verifier().Run(module.get()).status());
  }
}

// A target with no cuDNN plans for either the FP8 fusion or its BF16
// replacement (V100, sm_70: no FP8 engines, and no BF16 conv engines either)
// must keep the FP8 fusion rather than rewrite to something equally
// unrunnable — and must drop the BF16 clone it built to probe with.
TEST_F(ConvFp8FallbackDevicelessTest, Fp8ConvIsKeptWhenBf16AlsoHasNoPlans) {
  ASSERT_OK_AND_ASSIGN(GpuTargetConfig construction_config,
                       DevicelessTargetConfig(GpuModel::H100_SXM));
  ASSERT_OK_AND_ASSIGN(GpuTargetConfig target_config,
                       DevicelessTargetConfig(GpuModel::V100));
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<VerifiedHloModule> module,
      BuildConvFusionModule(kF8BareConvHlo, construction_config));
  ConvFp8Fallback pass(target_config.device_description);
  ASSERT_OK_AND_ASSIGN(bool changed, RunHloPass(&pass, module.get()));
  EXPECT_FALSE(changed);
  HloFusionInstruction* fusion = FindCudnnFusion(*module);
  ASSERT_NE(fusion, nullptr);
  EXPECT_TRUE(FusionContainsF8(*fusion));
  // The BF16 clone built only to probe is gone again, computation included.
  EXPECT_EQ(CountCudnnFusions(*module), 1);
  EXPECT_OK(verifier().Run(module.get()).status());
}

class ConvFp8FallbackSm120DevicelessTest
    : public ConvFp8FallbackDevicelessTest {
 protected:
  void SetUp() override {
    ConvFp8FallbackDevicelessTest::SetUp();
    if (IsSkipped()) {
      return;
    }
    absl::StatusOr<GpuTargetConfig> target_config =
        DevicelessTargetConfig(GpuModel::RTX6000PRO);
    ASSERT_TRUE(target_config.ok()) << target_config.status();
    if (!se::gpu::SupportsDevicelessConvGraphs(
            target_config->device_description)) {
      GTEST_SKIP() << "cuDNN runtime cannot probe conv graphs devicelessly "
                      "for sm_120 targets (requires cuDNN >= 9.19).";
    }
  }
};

// channels/group == 16 has cuDNN plans on sm_120; the fusion must be kept
// FP8.
TEST_F(ConvFp8FallbackSm120DevicelessTest,
       GroupedConvWith16ChannelsPerGroupKeptFp8) {
  ASSERT_OK_AND_ASSIGN(GpuTargetConfig target_config,
                       DevicelessTargetConfig(GpuModel::RTX6000PRO));
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<VerifiedHloModule> module,
      BuildConvFusionModule(kGroupedF8Conv16ChannelsHlo, target_config));
  ConvFp8Fallback pass(target_config.device_description);
  ASSERT_OK_AND_ASSIGN(bool changed, RunHloPass(&pass, module.get()));
  EXPECT_FALSE(changed);
  EXPECT_TRUE(FusionContainsF8(*FindCudnnFusion(*module)));
}

// Fewer than 16 channels/group has no FP8 plans the conv-fusion pipeline can
// execute on sm_120 (cuDNN heuristic mode A; 8 channels/group appears only
// in the FALLBACK heuristic list that the legacy conv runner path uses, but
// CudnnGraph::Prepare — deviceless probe and live compilation alike — does
// not enumerate). The pass must fall back to BF16; this is the configuration
// that motivated it.
TEST_F(ConvFp8FallbackSm120DevicelessTest,
       GroupedConvWithFewChannelsPerGroupFallsBackToBf16) {
  ASSERT_OK_AND_ASSIGN(GpuTargetConfig target_config,
                       DevicelessTargetConfig(GpuModel::RTX6000PRO));
  for (absl::string_view hlo :
       {kGroupedF8Conv8ChannelsHlo, kGroupedF8Conv2ChannelsHlo}) {
    ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                         BuildConvFusionModule(hlo, target_config));
    ConvFp8Fallback pass(target_config.device_description);
    ASSERT_OK_AND_ASSIGN(bool changed, RunHloPass(&pass, module.get()));
    EXPECT_TRUE(changed);
    HloFusionInstruction* fusion = FindCudnnFusion(*module);
    ASSERT_NE(fusion, nullptr);
    EXPECT_FALSE(FusionContainsF8(*fusion));
    EXPECT_OK(verifier().Run(module.get()).status());
  }
}

}  // namespace
}  // namespace gpu
}  // namespace xla

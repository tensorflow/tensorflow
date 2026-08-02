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

// Tests for CuDnnFusionCompiler::SupportsFusionDeviceless.
//
// The deviceless cases build DeviceDescriptions from checked-in target-config
// specs and open no GPU; they need a loadable host cuDNN >= 9.8 and skip
// otherwise. DevicelessSupportMatchesLivePlanEnumeration needs a real GPU
// whose deviceless conv probing is not gated off and skips otherwise.
//
// The fusions under test are produced by the real ConvKindAssignment +
// ConvFusionRewriter passes, so they cannot drift from pipeline output.

#include <memory>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "xla/tsl/platform/status_macros.h"
#include "xla/backends/gpu/target_config/target_config.h"
#include "xla/backends/gpu/transforms/conv_fusion_rewriter.h"
#include "xla/backends/gpu/transforms/conv_kind_assignment.h"
#include "xla/backends/gpu/transforms/cudnn_fusion_compiler.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/testlib/verified_hlo_module.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/service/hlo.pb.h"
#include "xla/stream_executor/cuda/cuda_dnn.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/device_description.pb.h"
#include "xla/stream_executor/dnn.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/semantic_version.h"
#include "xla/stream_executor/stream_executor.h"

namespace xla {
namespace gpu {
namespace {

namespace se = ::stream_executor;

using DevicelessFusionSupport = CuDnnFusionCompiler::DevicelessFusionSupport;

class CudnnFusionCompilerDevicelessTest
    : public HloHardwareIndependentTestBase {
 protected:
  void SetUp() override {
    if (!se::gpu::SupportsDevicelessDeviceProperties()) {
      GTEST_SKIP() << "cuDNN runtime < 9.8 does not support deviceless "
                      "DeviceProperties.";
    }
  }

  // Builds a deviceless GpuTargetConfig for the given GPU model from its
  // checked-in target-config spec (no StreamExecutor / GPU required).
  static absl::StatusOr<GpuTargetConfig> DevicelessTargetConfig(
      GpuModel model) {
    ASSIGN_OR_RETURN(stream_executor::GpuTargetConfigProto proto,
                     GetGpuTargetConfig(model));
    return GpuTargetConfig::FromProto(proto);
  }

  // Runs the conv-fusion pipeline passes on `hlo_text` and returns the module
  // containing the resulting __cudnn$fusion. Fails if the fused convolution
  // was not assigned `expected_kind`, so the HLO patterns below cannot
  // silently match a different kind.
  absl::StatusOr<std::unique_ptr<VerifiedHloModule>> BuildConvFusionModule(
      absl::string_view hlo_text, const se::DeviceDescription& device_info,
      se::dnn::VersionInfo dnn_version, ConvolutionKind expected_kind) {
    ASSIGN_OR_RETURN(std::unique_ptr<VerifiedHloModule> module,
                     ParseAndReturnVerifiedModule(hlo_text));
    ConvKindAssignment kind_assignment(device_info.gpu_compute_capability(),
                                       dnn_version);
    RETURN_IF_ERROR(RunHloPass(&kind_assignment, module.get()).status());
    ConvFusionRewriter rewriter(device_info);
    RETURN_IF_ERROR(RunHloPass(&rewriter, module.get()).status());

    const HloFusionInstruction* fusion = FindCudnnFusion(*module);
    if (fusion == nullptr) {
      return absl::InternalError(absl::StrCat(
          "No __cudnn$fusion produced for:\n", module->ToString()));
    }
    const HloInstruction* conv = nullptr;
    for (const HloInstruction* instruction :
         fusion->fused_instructions_computation()->instructions()) {
      if (instruction->opcode() == HloOpcode::kConvolution) {
        conv = instruction;
        break;
      }
    }
    if (conv == nullptr ||
        Cast<HloConvolutionInstruction>(conv)->convolution_kind() !=
            expected_kind) {
      return absl::InternalError(absl::StrCat(
          "Fused convolution does not have expected kind ",
          ConvolutionKind_Name(expected_kind), ":\n", module->ToString()));
    }
    return module;
  }

  static const HloFusionInstruction* FindCudnnFusion(const HloModule& module) {
    for (const HloInstruction* instruction :
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

  static se::StreamExecutor* MaybeGetGpuExecutor() {
    absl::StatusOr<se::Platform*> platform =
        se::PlatformManager::PlatformWithName("CUDA");
    if (!platform.ok() || (*platform)->VisibleDeviceCount() == 0) {
      return nullptr;
    }
    absl::StatusOr<se::StreamExecutor*> executor =
        (*platform)->ExecutorForDevice(0);
    return executor.ok() ? *executor : nullptr;
  }
};

constexpr absl::string_view kF32ForwardConvHlo = R"(
  ENTRY e {
    input = f32[8,28,28,64] parameter(0)
    filter = f32[16,3,3,64] parameter(1)
    ROOT conv = f32[8,28,28,16] convolution(input, filter),
      window={size=3x3 pad=1_1x1_1}, dim_labels=b01f_o01i->b01f
  })";

// Backward-filter pattern (all filter dimensions larger than the output
// dimensions), matching ConvKindAssignmentTest.TestBackwardFilterPatternMatch.
constexpr absl::string_view kF32BackwardFilterConvHlo = R"(
  ENTRY e {
    activations = f32[8,120,64,64] parameter(0)
    gradients = f32[8,120,64,64] parameter(1)
    ROOT conv = f32[120,120,3,3] convolution(activations, gradients),
      window={size=64x64 pad=1_1x1_1}, dim_labels=fb01_io01->fb01
  })";

// Backward-input pattern (reversed filter with symmetric padding), matching
// ConvKindAssignmentTest.BackwardInputConvolveEvenPadding.
constexpr absl::string_view kF32BackwardInputConvHlo = R"(
  ENTRY e {
    gradients = f32[4,5,16,16] parameter(0)
    kernel = f32[5,3,7,7] parameter(1)
    reverse = f32[5,3,7,7] reverse(kernel), dimensions={2,3}
    ROOT conv = f32[4,3,16,16] convolution(gradients, reverse),
      window={size=7x7 pad=3_3x3_3}, dim_labels=bf01_io01->bf01
  })";

// Bare FP8 convolution; the conv result itself is the fusion output.
constexpr absl::string_view kF8BareConvHlo = R"(
  ENTRY e {
    input = f8e4m3fn[1,6,6,128] parameter(0)
    filter = f8e4m3fn[16,3,3,128] parameter(1)
    ROOT conv = f8e4m3fn[1,6,6,16] convolution(input, filter),
      window={size=3x3 pad=1_1x1_1}, dim_labels=b01f_o01i->b01f
  })";

// FP8 convolution accumulating in f32 with a scale / clamp / quantize
// epilogue, in the shape ConvFusionRewriter fuses (prologue converts consumed
// by the compiler, pointwise epilogue).
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

// A grouped FP8 convolution with fewer than 16 channels per group — the
// configuration that motivated compile-time conv support probing (such
// convolutions can have no valid cuDNN plans on some GPU generations, see
// openxla/xla#40021).
constexpr absl::string_view kGroupedFp8ConvHlo = R"(
  ENTRY e {
    input = f8e4m3fn[1,16,16,32] parameter(0)
    filter = f8e4m3fn[32,3,3,2] parameter(1)
    ROOT conv = f8e4m3fn[1,16,16,32] convolution(input, filter),
      window={size=3x3 pad=1_1x1_1}, dim_labels=b01f_o01i->b01f,
      feature_group_count=16
  })";

// Plain f32 convolutions of every kind are supported on any GPU generation.
// Also the control for the FP8 tests below: their V100 kUnsupported cannot be
// blamed on an unusable V100 device description.
TEST_F(CudnnFusionCompilerDevicelessTest, PlainF32ConvAllKindsSupported) {
  struct KindCase {
    absl::string_view name;
    absl::string_view hlo;
    ConvolutionKind kind;
  };
  const KindCase kKindCases[] = {
      {"fprop", kF32ForwardConvHlo, CONVOLUTION_KIND_FPROP},
      {"wgrad", kF32BackwardFilterConvHlo, CONVOLUTION_KIND_WGRAD},
      {"dgrad", kF32BackwardInputConvHlo, CONVOLUTION_KIND_DGRAD},
  };
  ASSERT_OK_AND_ASSIGN(GpuTargetConfig construction_config,
                       DevicelessTargetConfig(GpuModel::H100_SXM));
  for (const KindCase& kind_case : kKindCases) {
    SCOPED_TRACE(kind_case.name);
    ASSERT_OK_AND_ASSIGN(
        std::unique_ptr<VerifiedHloModule> module,
        BuildConvFusionModule(
            kind_case.hlo, construction_config.device_description,
            construction_config.dnn_version_info, kind_case.kind));
    const HloFusionInstruction* fusion = FindCudnnFusion(*module);
    ASSERT_NE(fusion, nullptr);
    for (GpuModel model : {GpuModel::H100_SXM, GpuModel::V100}) {
      SCOPED_TRACE(absl::StrCat("model=", static_cast<int>(model)));
      ASSERT_OK_AND_ASSIGN(GpuTargetConfig target_config,
                           DevicelessTargetConfig(model));
      EXPECT_EQ(CuDnnFusionCompiler::SupportsFusionDeviceless(
                    target_config.device_description, *fusion),
                DevicelessFusionSupport::kSupported);
    }
  }
}

// FP8 convolutions require sm_89+, so the same host cuDNN must probe
// kSupported for H100 (sm_90) and kUnsupported for V100 (sm_70). The V100
// case also checks end to end that cuDNN's negative verdicts surface as
// kUnsupported rather than kUnknown.
TEST_F(CudnnFusionCompilerDevicelessTest, Fp8ConvVerdictTracksTargetGpu) {
  struct GraphCase {
    absl::string_view name;
    absl::string_view hlo;
  };
  const GraphCase kGraphCases[] = {
      {"bare_conv", kF8BareConvHlo},
      // conv (f32 accumulate) -> scale by a scalar -> clamp -> f8 output.
      {"scale", kF8ScaleConvHlo},
      // conv -> relu -> quantizing scale.
      {"relu_scale", R"(
        ENTRY e {
          input = f8e4m3fn[1,6,6,128] parameter(0)
          filter = f8e4m3fn[16,3,3,128] parameter(1)
          input_f32 = f32[1,6,6,128] convert(input)
          filter_f32 = f32[16,3,3,128] convert(filter)
          zero = f32[] constant(0)
          zero_bcast = f32[1,6,6,16] broadcast(zero), dimensions={}
          z_scale = f32[] parameter(2)
          z_scale_bcast = f32[1,6,6,16] broadcast(z_scale), dimensions={}
          conv = f32[1,6,6,16] convolution(input_f32, filter_f32),
            window={size=3x3 pad=1_1x1_1}, dim_labels=b01f_o01i->b01f
          relu = f32[1,6,6,16] maximum(conv, zero_bcast)
          scaled = f32[1,6,6,16] multiply(relu, z_scale_bcast)
          lower = f32[] constant(-448.)
          lower_bcast = f32[1,6,6,16] broadcast(lower), dimensions={}
          upper = f32[] constant(448.)
          upper_bcast = f32[1,6,6,16] broadcast(upper), dimensions={}
          clamped = f32[1,6,6,16] clamp(lower_bcast, scaled, upper_bcast)
          ROOT out = f8e4m3fn[1,6,6,16] convert(clamped)
        })"},
      // conv -> scale -> f8 output, plus an amax of the f32 tensor as a
      // second (scalar) fusion output.
      {"scale_amax", R"(
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
        })"},
  };
  struct ModelExpectation {
    GpuModel model;
    DevicelessFusionSupport expected;
  };
  // Fusions are constructed once with an FP8-capable target, then probed
  // per target.
  ASSERT_OK_AND_ASSIGN(GpuTargetConfig construction_config,
                       DevicelessTargetConfig(GpuModel::H100_SXM));
  for (const GraphCase& graph_case : kGraphCases) {
    SCOPED_TRACE(graph_case.name);
    ASSERT_OK_AND_ASSIGN(
        std::unique_ptr<VerifiedHloModule> module,
        BuildConvFusionModule(
            graph_case.hlo, construction_config.device_description,
            construction_config.dnn_version_info, CONVOLUTION_KIND_FPROP));
    const HloFusionInstruction* fusion = FindCudnnFusion(*module);
    ASSERT_NE(fusion, nullptr);
    for (const auto& [model, expected] :
         {ModelExpectation{GpuModel::H100_SXM,
                           DevicelessFusionSupport::kSupported},
          ModelExpectation{GpuModel::V100,
                           DevicelessFusionSupport::kUnsupported}}) {
      SCOPED_TRACE(absl::StrCat("model=", static_cast<int>(model)));
      ASSERT_OK_AND_ASSIGN(GpuTargetConfig target_config,
                           DevicelessTargetConfig(model));
      EXPECT_EQ(CuDnnFusionCompiler::SupportsFusionDeviceless(
                    target_config.device_description, *fusion),
                expected);
    }
  }
}

// The grouped FP8 configuration that motivated compile-time conv support
// probing must get a verdict, not kUnknown. Which verdict is correct depends
// on the cuDNN version; parity with live enumeration is asserted on GPU in
// DevicelessSupportMatchesLivePlanEnumeration.
TEST_F(CudnnFusionCompilerDevicelessTest, GroupedFp8ConvDeliversVerdict) {
  ASSERT_OK_AND_ASSIGN(GpuTargetConfig target_config,
                       DevicelessTargetConfig(GpuModel::H100_SXM));
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       BuildConvFusionModule(kGroupedFp8ConvHlo,
                                             target_config.device_description,
                                             target_config.dnn_version_info,
                                             CONVOLUTION_KIND_FPROP));
  const HloFusionInstruction* fusion = FindCudnnFusion(*module);
  ASSERT_NE(fusion, nullptr);
  EXPECT_NE(CuDnnFusionCompiler::SupportsFusionDeviceless(
                target_config.device_description, *fusion),
            DevicelessFusionSupport::kUnknown);
}

// The deviceless verdict must agree with live plan enumeration on the
// executor's own device: SupportsFusionDeviceless(desc, fusion) is kSupported
// iff GetAvailablePlanCount(executor, desc, fusion) > 0.
TEST_F(CudnnFusionCompilerDevicelessTest,
       DevicelessSupportMatchesLivePlanEnumeration) {
  se::StreamExecutor* executor = MaybeGetGpuExecutor();
  if (executor == nullptr) {
    GTEST_SKIP() << "No CUDA GPU available.";
  }
  const se::DeviceDescription& device_description =
      executor->GetDeviceDescription();
  if (!se::gpu::SupportsDevicelessConvGraphs(device_description)) {
    GTEST_SKIP() << "Deviceless conv graph probing is gated off for this "
                    "GPU / cuDNN runtime combination, so every deviceless "
                    "verdict is kUnknown and there is no parity to check.";
  }
  const se::SemanticVersion dnn_version = device_description.dnn_version();
  struct ParityCase {
    absl::string_view name;
    absl::string_view hlo;
    ConvolutionKind kind;
  };
  const ParityCase kParityCases[] = {
      {"f32_fprop", kF32ForwardConvHlo, CONVOLUTION_KIND_FPROP},
      {"f32_wgrad", kF32BackwardFilterConvHlo, CONVOLUTION_KIND_WGRAD},
      {"f32_dgrad", kF32BackwardInputConvHlo, CONVOLUTION_KIND_DGRAD},
      {"f8_bare_conv", kF8BareConvHlo, CONVOLUTION_KIND_FPROP},
      {"f8_scale", kF8ScaleConvHlo, CONVOLUTION_KIND_FPROP},
      {"f8_grouped", kGroupedFp8ConvHlo, CONVOLUTION_KIND_FPROP},
  };
  for (const ParityCase& parity_case : kParityCases) {
    SCOPED_TRACE(parity_case.name);
    absl::StatusOr<std::unique_ptr<VerifiedHloModule>> module =
        BuildConvFusionModule(parity_case.hlo, device_description,
                              se::dnn::VersionInfo(dnn_version.major_version(),
                                                   dnn_version.minor_version(),
                                                   dnn_version.patch_version()),
                              parity_case.kind);
    if (absl::IsUnimplemented(module.status())) {
      // The conv-fusion pipeline refuses this case on the test GPU (e.g. FP8
      // below sm_89); there is no fusion whose parity could be checked.
      continue;
    }
    ASSERT_TRUE(module.ok()) << module.status();
    const HloFusionInstruction* fusion = FindCudnnFusion(**module);
    ASSERT_NE(fusion, nullptr);

    // Live enumeration reports unsupported fusions as an error from
    // PrepareGraph, not as a zero count.
    const absl::StatusOr<int> live_plan_count =
        CuDnnFusionCompiler::GetAvailablePlanCount(executor, device_description,
                                                   *fusion);
    const bool live_has_plans = live_plan_count.ok() && *live_plan_count > 0;
    EXPECT_EQ(CuDnnFusionCompiler::SupportsFusionDeviceless(device_description,
                                                            *fusion),
              live_has_plans ? DevicelessFusionSupport::kSupported
                             : DevicelessFusionSupport::kUnsupported);
  }
}

}  // namespace
}  // namespace gpu
}  // namespace xla

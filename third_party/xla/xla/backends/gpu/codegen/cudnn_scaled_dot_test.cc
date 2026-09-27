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

#include <cmath>
#include <cstdint>
#include <memory>
#include <ostream>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/algorithm/container.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/random/uniform_int_distribution.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_replace.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/backends/autotuner/backends.pb.h"
#include "xla/backends/gpu/tests/hlo_pjrt_gpu_test_base.h"
#include "xla/backends/gpu/transforms/composite_rewriter.h"
#include "xla/backends/gpu/transforms/cudnn_custom_call_compiler.h"
#include "xla/backends/gpu/transforms/cudnn_fusion_compiler.h"
#include "xla/backends/gpu/transforms/scaled_dot_rewriter.h"
#include "xla/error_spec.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/hlo/testlib/verified_hlo_module.h"
#include "xla/literal.h"
#include "xla/primitive_util.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/stream_executor/cuda/cuda_compute_capability.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/semantic_version.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tests/hlo_interpreter_reference_mixin.h"
#include "xla/tests/test_utils.h"
#include "xla/tsl/platform/test.h"
#include "xla/types.h"
#include "xla/xla.pb.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace gpu {
namespace {

class CudnnScaledDotTestBase
    : public HloInterpreterReferenceMixin<HloPjRtGpuTestBase> {
 public:
  se::StreamExecutor* stream_executor() const {
    auto platform =
        se::PlatformManager::PlatformWithId(stream_executor_platform_id());
    CHECK_OK(platform);
    auto executor = (*platform)->ExecutorForDevice(0);
    CHECK_OK(executor);
    return *executor;
  }

  DebugOptions GetDebugOptionsForTest() const override {
    DebugOptions debug_options = HloPjRtGpuTestBase::GetDebugOptionsForTest();
    debug_options.set_xla_gpu_autotune_level(0);
    debug_options.set_xla_gpu_cudnn_gemm_fusion_level(2);
    debug_options.clear_xla_gpu_experimental_autotune_backends();
    debug_options.add_xla_gpu_experimental_autotune_backends(
        autotuner::Backend::CUDNN);
    debug_options.add_xla_gpu_experimental_autotune_backends(
        autotuner::Backend::CUBLASLT);
    debug_options.set_xla_gpu_experimental_scaled_dot_with_triton(false);
    return debug_options;
  }

  se::CudaComputeCapability get_cuda_cc() const {
    return device_description().cuda_compute_capability();
  }

  bool IsAtLeastBlackwellWithCuDnn9() const {
    return get_cuda_cc().IsAtLeastBlackwell() &&
           gpu_target_config()
                   .device_description.dnn_version()
                   .major_version() >= 9;
  }

  absl::Status PopulateScale(Literal* scale, std::minstd_rand0* engine) {
    switch (scale->shape().element_type()) {
      case F8E8M0FNU: {
        absl::uniform_int_distribution<int> exponent_distribution(-4, 1);
        return scale->Populate<float8_e8m0fnu>(
            [&](absl::Span<const int64_t> /*indices*/) {
              return float8_e8m0fnu(
                  std::ldexp(1.0f, exponent_distribution(*engine)));
            });
      }
      case F8E4M3FN:
        for (float8_e4m3fn& value : scale->data<float8_e4m3fn>()) {
          value = float8_e4m3fn(std::abs(static_cast<float>(value)));
        }
        return absl::OkStatus();
      case F8E5M2:
        for (float8_e5m2& value : scale->data<float8_e5m2>()) {
          value = float8_e5m2(std::abs(static_cast<float>(value)));
        }
        return absl::OkStatus();
      default:
        return absl::InvalidArgumentError(
            absl::StrCat("Unsupported scale type: ",
                         PrimitiveType_Name(scale->shape().element_type())));
    }
  }

  absl::StatusOr<std::vector<Literal>> MakeScaledDotArguments(
      const HloModule* module) {
    std::minstd_rand0 engine;
    FakeArgumentsOptions options;
    options.engine = &engine;
    ABSL_ASSIGN_OR_RETURN(std::vector<Literal> arguments,
                     MakeFakeArguments(module, options));
    if (arguments.size() < 4) {
      return absl::InternalError(absl::StrCat(
          "Expected at least 4 scaled-dot arguments, got ", arguments.size()));
    }
    ABSL_RETURN_IF_ERROR(PopulateScale(&arguments[2], &engine));
    ABSL_RETURN_IF_ERROR(PopulateScale(&arguments[3], &engine));
    return arguments;
  }
};

enum class CuDnnSupportResult {
  kSupported,  // Lowered to native cuDNN block-scaled graph, execution succeeds
  kNotSupported,  // Not supported by cuDNN block scaling, falls back to
                  // decomposed dot
  kFailed,        // Lowered to native cuDNN block-scaled graph, but execution
                  // exhibits discrepancy
};

absl::string_view CuDnnSupportResultToString(CuDnnSupportResult result) {
  switch (result) {
    case CuDnnSupportResult::kSupported:
      return "supported";
    case CuDnnSupportResult::kNotSupported:
      return "not_supported";
    case CuDnnSupportResult::kFailed:
      return "failed";
  }
}

std::ostream& operator<<(std::ostream& os, CuDnnSupportResult result) {
  return os << CuDnnSupportResultToString(result);
}

struct ScaledDotCoverageTestCase {
  PrimitiveType lhs_type;
  PrimitiveType rhs_type;
  PrimitiveType scale_type;
  int block_size;
  bool lhs_k_minor;
  bool rhs_k_minor;
  CuDnnSupportResult expected_result;
};

std::vector<ScaledDotCoverageTestCase> GetCoverageTestCases() {
  return {
      // 1. MXFP8 E4M3FN x E4M3FN (scale E8M0FNU, block 32)
      {F8E4M3FN, F8E4M3FN, F8E8M0FNU, 32, false, false,
       CuDnnSupportResult::kNotSupported},
      {F8E4M3FN, F8E4M3FN, F8E8M0FNU, 32, false, true,
       CuDnnSupportResult::kNotSupported},
      {F8E4M3FN, F8E4M3FN, F8E8M0FNU, 32, true, false,
       CuDnnSupportResult::kNotSupported},
      {F8E4M3FN, F8E4M3FN, F8E8M0FNU, 32, true, true,
       CuDnnSupportResult::kSupported},

      // 2. MXFP8 E4M3FN x E5M2 (scale E8M0FNU, block 32)
      {F8E4M3FN, F8E5M2, F8E8M0FNU, 32, false, false,
       CuDnnSupportResult::kNotSupported},
      {F8E4M3FN, F8E5M2, F8E8M0FNU, 32, false, true,
       CuDnnSupportResult::kNotSupported},
      {F8E4M3FN, F8E5M2, F8E8M0FNU, 32, true, false,
       CuDnnSupportResult::kNotSupported},
      {F8E4M3FN, F8E5M2, F8E8M0FNU, 32, true, true,
       CuDnnSupportResult::kSupported},

      // 3. E4M3FN x F4E2M1FN (scale E8M0FNU, block 32)
      {F8E4M3FN, F4E2M1FN, F8E8M0FNU, 32, false, false,
       CuDnnSupportResult::kNotSupported},
      {F8E4M3FN, F4E2M1FN, F8E8M0FNU, 32, false, true,
       CuDnnSupportResult::kNotSupported},
      {F8E4M3FN, F4E2M1FN, F8E8M0FNU, 32, true, false,
       CuDnnSupportResult::kNotSupported},
      {F8E4M3FN, F4E2M1FN, F8E8M0FNU, 32, true, true,
       CuDnnSupportResult::kNotSupported},

      // 4. MXFP8 E5M2 x E4M3FN (scale E8M0FNU, block 32)
      {F8E5M2, F8E4M3FN, F8E8M0FNU, 32, false, false,
       CuDnnSupportResult::kNotSupported},
      {F8E5M2, F8E4M3FN, F8E8M0FNU, 32, false, true,
       CuDnnSupportResult::kNotSupported},
      {F8E5M2, F8E4M3FN, F8E8M0FNU, 32, true, false,
       CuDnnSupportResult::kNotSupported},
      {F8E5M2, F8E4M3FN, F8E8M0FNU, 32, true, true,
       CuDnnSupportResult::kSupported},

      // 5. MXFP8 E5M2 x E5M2 (scale E8M0FNU, block 32) - unsupported by cuDNN
      {F8E5M2, F8E5M2, F8E8M0FNU, 32, false, false,
       CuDnnSupportResult::kNotSupported},
      {F8E5M2, F8E5M2, F8E8M0FNU, 32, false, true,
       CuDnnSupportResult::kNotSupported},
      {F8E5M2, F8E5M2, F8E8M0FNU, 32, true, false,
       CuDnnSupportResult::kNotSupported},
      {F8E5M2, F8E5M2, F8E8M0FNU, 32, true, true,
       CuDnnSupportResult::kNotSupported},

      // 6. E5M2 x F4E2M1FN (scale E8M0FNU, block 32)
      {F8E5M2, F4E2M1FN, F8E8M0FNU, 32, false, false,
       CuDnnSupportResult::kNotSupported},
      {F8E5M2, F4E2M1FN, F8E8M0FNU, 32, false, true,
       CuDnnSupportResult::kNotSupported},
      {F8E5M2, F4E2M1FN, F8E8M0FNU, 32, true, false,
       CuDnnSupportResult::kNotSupported},
      {F8E5M2, F4E2M1FN, F8E8M0FNU, 32, true, true,
       CuDnnSupportResult::kNotSupported},

      // 7. F4E2M1FN x E4M3FN (scale E8M0FNU, block 32)
      {F4E2M1FN, F8E4M3FN, F8E8M0FNU, 32, false, false,
       CuDnnSupportResult::kNotSupported},
      {F4E2M1FN, F8E4M3FN, F8E8M0FNU, 32, false, true,
       CuDnnSupportResult::kNotSupported},
      {F4E2M1FN, F8E4M3FN, F8E8M0FNU, 32, true, false,
       CuDnnSupportResult::kNotSupported},
      {F4E2M1FN, F8E4M3FN, F8E8M0FNU, 32, true, true,
       CuDnnSupportResult::kNotSupported},

      // 8. F4E2M1FN x E5M2 (scale E8M0FNU, block 32)
      {F4E2M1FN, F8E5M2, F8E8M0FNU, 32, false, false,
       CuDnnSupportResult::kNotSupported},
      {F4E2M1FN, F8E5M2, F8E8M0FNU, 32, false, true,
       CuDnnSupportResult::kNotSupported},
      {F4E2M1FN, F8E5M2, F8E8M0FNU, 32, true, false,
       CuDnnSupportResult::kNotSupported},
      {F4E2M1FN, F8E5M2, F8E8M0FNU, 32, true, true,
       CuDnnSupportResult::kNotSupported},

      // 9. F4E2M1FN x F4E2M1FN (scale E8M0FNU, block 32)
      {F4E2M1FN, F4E2M1FN, F8E8M0FNU, 32, false, false,
       CuDnnSupportResult::kNotSupported},
      {F4E2M1FN, F4E2M1FN, F8E8M0FNU, 32, false, true,
       CuDnnSupportResult::kNotSupported},
      {F4E2M1FN, F4E2M1FN, F8E8M0FNU, 32, true, false,
       CuDnnSupportResult::kNotSupported},
      {F4E2M1FN, F4E2M1FN, F8E8M0FNU, 32, true, true,
       CuDnnSupportResult::kNotSupported},

      // 10. F4E2M1FN x F4E2M1FN (scale E8M0FNU, block 16)
      {F4E2M1FN, F4E2M1FN, F8E8M0FNU, 16, false, false,
       CuDnnSupportResult::kNotSupported},
      {F4E2M1FN, F4E2M1FN, F8E8M0FNU, 16, false, true,
       CuDnnSupportResult::kNotSupported},
      {F4E2M1FN, F4E2M1FN, F8E8M0FNU, 16, true, false,
       CuDnnSupportResult::kNotSupported},
      {F4E2M1FN, F4E2M1FN, F8E8M0FNU, 16, true, true,
       CuDnnSupportResult::kNotSupported},

      // 11. NVFP4 F4E2M1FN x F4E2M1FN (scale E4M3FN, block 16)
      {F4E2M1FN, F4E2M1FN, F8E4M3FN, 16, false, false,
       CuDnnSupportResult::kNotSupported},
      {F4E2M1FN, F4E2M1FN, F8E4M3FN, 16, false, true,
       CuDnnSupportResult::kNotSupported},
      {F4E2M1FN, F4E2M1FN, F8E4M3FN, 16, true, false,
       CuDnnSupportResult::kNotSupported},
      {F4E2M1FN, F4E2M1FN, F8E4M3FN, 16, true, true,
       CuDnnSupportResult::kFailed},
  };
}

std::string ScaledDotCoverageTestParamToString(
    const ::testing::TestParamInfo<ScaledDotCoverageTestCase>& info) {
  const auto& tc = info.param;
  return absl::StrCat(
      PrimitiveType_Name(tc.lhs_type), "_", PrimitiveType_Name(tc.rhs_type),
      "_", PrimitiveType_Name(tc.scale_type), "_Block", tc.block_size, "_Lhs",
      tc.lhs_k_minor, "_Rhs", tc.rhs_k_minor);
}

class CudnnScaledDotCoverageTest
    : public CudnnScaledDotTestBase,
      public ::testing::WithParamInterface<ScaledDotCoverageTestCase> {};

TEST_P(CudnnScaledDotCoverageTest, Executes) {
  const auto& param = GetParam();
  const bool lhs_k_minor = param.lhs_k_minor;
  const bool rhs_k_minor = param.rhs_k_minor;

  auto cc = get_cuda_cc();
  std::string device_name = "Unknown";
  if (cc.IsAtLeastBlackwell()) {
    device_name = "B200";
  } else if (cc.IsAtLeastHopper()) {
    device_name = "H100";
  } else {
    device_name = "Pre-Hopper";
  }

  std::string lhs_name =
      primitive_util::LowercasePrimitiveTypeName(param.lhs_type);
  std::string rhs_name =
      primitive_util::LowercasePrimitiveTypeName(param.rhs_type);
  std::string scale_name =
      primitive_util::LowercasePrimitiveTypeName(param.scale_type);
  std::string lhs_display =
      lhs_k_minor ? lhs_name : absl::StrCat(lhs_name, ".T");
  std::string rhs_display =
      rhs_k_minor ? absl::StrCat(rhs_name, ".T") : rhs_name;

  LOG(ERROR) << "Report Device: " << device_name;
  LOG(ERROR) << "Report LHS: " << lhs_display;
  LOG(ERROR) << "Report RHS: " << rhs_display;
  LOG(ERROR) << "Report Scale: " << scale_name;
  LOG(ERROR) << "Report BlockSize: " << param.block_size;

  constexpr int64_t m = 128;
  constexpr int64_t n = 128;
  constexpr int64_t k = 256;
  const int64_t scale_k = k / param.block_size;

  const std::string lhs_shape =
      lhs_k_minor ? absl::StrCat(m, ",", k) : absl::StrCat(k, ",", m);
  const std::string rhs_shape =
      rhs_k_minor ? absl::StrCat(n, ",", k) : absl::StrCat(k, ",", n);
  const std::string lhs_scale_shape = lhs_k_minor
                                          ? absl::StrCat(m, ",", scale_k)
                                          : absl::StrCat(scale_k, ",", m);
  const std::string rhs_scale_shape = rhs_k_minor
                                          ? absl::StrCat(n, ",", scale_k)
                                          : absl::StrCat(scale_k, ",", n);

  // 1. Stage: After Composite
  std::string after_composite = "N/A";
  constexpr absl::string_view kCompositeHloTemplate = R"hlo(
HloModule test_composite

%xla.scaled_dot.1 {
  %p0 = $lhs_type[$lhs_shape]{1,0} parameter(0)
  %p1 = $rhs_type[$rhs_shape]{1,0} parameter(1)
  %p2 = $scale_type[$lhs_scale_shape]{1,0} parameter(2)
  %p3 = $scale_type[$rhs_scale_shape]{1,0} parameter(3)
  %zero = bf16[] constant(0)
  ROOT %dummy = bf16[$output_shape]{1,0} broadcast(%zero), dimensions={}
}

ENTRY %main {
  %lhs = $lhs_type[$lhs_shape]{1,0} parameter(0)
  %rhs = $rhs_type[$rhs_shape]{1,0} parameter(1)
  %lhs_scale = $scale_type[$lhs_scale_shape]{1,0} parameter(2)
  %rhs_scale = $scale_type[$rhs_scale_shape]{1,0} parameter(3)
  ROOT %call = bf16[$output_shape]{1,0} call(%lhs, %rhs, %lhs_scale, %rhs_scale),
      to_apply=%xla.scaled_dot.1,
      is_composite=true,
      frontend_attributes={
        composite.attributes="{dimension_numbers=[[[$lhs_contracting_dim],[$rhs_contracting_dim]],[[],[]]]}",
        composite.name="xla.scaled_dot",
        composite.version="1"
      }
}
)hlo";

  std::string composite_hlo =
      absl::StrReplaceAll(kCompositeHloTemplate,
                          {{"$lhs_type", lhs_name},
                           {"$rhs_type", rhs_name},
                           {"$scale_type", scale_name},
                           {"$lhs_shape", lhs_shape},
                           {"$rhs_shape", rhs_shape},
                           {"$lhs_scale_shape", lhs_scale_shape},
                           {"$rhs_scale_shape", rhs_scale_shape},
                           {"$output_shape", absl::StrCat(m, ",", n)},
                           {"$lhs_contracting_dim", lhs_k_minor ? "1" : "0"},
                           {"$rhs_contracting_dim", rhs_k_minor ? "1" : "0"}});

  if (auto comp_module = ParseAndReturnVerifiedModule(composite_hlo);
      comp_module.ok()) {
    CompositeRewriter rewriter;
    if (auto rewrite_status = rewriter.Run(comp_module->get());
        rewrite_status.ok()) {
      const auto* root =
          (*comp_module)->entry_computation()->root_instruction();
      after_composite = HloOpcodeString(root->opcode());
    }
  }
  LOG(ERROR) << "Report AfterComposite: " << after_composite;
  EXPECT_EQ(after_composite, "scaled-dot");

  // 2. Stage: Optimized HLO (compiled directly from composite_hlo)
  std::string optimized_hlo = "N/A";
  HloInstruction* custom_call_or_fusion = nullptr;
  auto optimized_module_or = GetOptimizedModule(composite_hlo);
  if (!optimized_module_or.ok()) {
    LOG(ERROR) << "GetOptimizedModule error: " << optimized_module_or.status();
  }
  if (optimized_module_or.ok()) {
    HloModule& optimized_module = **optimized_module_or;
    for (HloComputation* comp : optimized_module.computations()) {
      for (HloInstruction* instr : comp->instructions()) {
        if (instr->opcode() == HloOpcode::kCustomCall &&
            instr->custom_call_target() == "__cudnn$blockScaledDot") {
          optimized_hlo = "__cudnn$blockScaledDot";
          custom_call_or_fusion = instr;
          break;
        }
        if (instr->opcode() == HloOpcode::kFusion) {
          if (auto gpu_cfg = instr->backend_config<GpuBackendConfig>();
              gpu_cfg.ok() &&
              gpu_cfg->fusion_backend_config().kind() == "__cudnn$fusion") {
            const HloComputation* fused_comp =
                instr->fused_instructions_computation();
            bool has_scaled_dot = false;
            for (const HloInstruction* fused_instr :
                 fused_comp->instructions()) {
              if (fused_instr->opcode() == HloOpcode::kScaledDot) {
                has_scaled_dot = true;
                break;
              }
            }
            optimized_hlo = has_scaled_dot ? "__cudnn$fusion" : "dot";
            custom_call_or_fusion = instr;
            break;
          }
        }
        if (instr->opcode() == HloOpcode::kScaledDot) {
          optimized_hlo = "scaled-dot";
        } else if (instr->opcode() == HloOpcode::kDot &&
                   optimized_hlo == "N/A") {
          optimized_hlo = "dot";
        }
      }
      if (custom_call_or_fusion != nullptr) {
        break;
      }
    }
  }
  LOG(ERROR) << "Report OptimizedHLO: " << optimized_hlo;

  // 3. Stage: cuDNN Graph
  std::string cudnn_graph_pattern = "N/A";
  if (custom_call_or_fusion != nullptr && IsAtLeastBlackwellWithCuDnn9() &&
      optimized_hlo != "dot" && optimized_module_or.ok()) {
    HloModule& optimized_module = **optimized_module_or;
    if (custom_call_or_fusion->opcode() == HloOpcode::kFusion) {
      BinaryMap compilation_results;
      CuDnnFusionCompiler cudnn_compiler(stream_executor()->AsDnn(),
                                         device_description(),
                                         compilation_results);
      auto cloned = optimized_module.Clone();
      auto res = cudnn_compiler.Run(cloned.get());
      if (res.ok()) {
        cudnn_graph_pattern = "BLOCK_SCALE_DEQUANTIZE + MATMUL";
      } else {
        cudnn_graph_pattern = absl::StrCat("FAILED: ", res.status().message());
      }
    } else if (custom_call_or_fusion->opcode() == HloOpcode::kCustomCall) {
      BinaryMap compilation_results;
      CuDnnCustomCallCompiler custom_call_compiler(stream_executor()->AsDnn(),
                                                   device_description(),
                                                   compilation_results);
      auto cloned = optimized_module.Clone();
      auto res = custom_call_compiler.Run(cloned.get());
      if (res.ok() && !compilation_results.empty()) {
        cudnn_graph_pattern = "BLOCK_SCALE_DEQUANTIZE + MATMUL";
      } else if (!res.ok()) {
        cudnn_graph_pattern = absl::StrCat("FAILED: ", res.status().message());
      }
    }
  }

  // 4. Stage: Execution Status
  std::string status = "FAILURE";
  if (!IsAtLeastBlackwellWithCuDnn9()) {
    GTEST_SKIP()
        << "Block scaled dot cuDNN execution requires Blackwell and cuDNN 9+.";
  }
  if (optimized_module_or.ok()) {
    auto ref_module_or = ParseAndReturnVerifiedModule(composite_hlo);
    if (ref_module_or.ok()) {
      CompositeRewriter composite_pass;
      (void)composite_pass.Run(ref_module_or->get());
      ScaledDotRewriter decompose_pass;
      if (auto decomposed = decompose_pass.Run(ref_module_or->get());
          decomposed.ok()) {
        auto ref_optimized_or =
            GetOptimizedModule((*ref_module_or)->ToString());
        if (ref_optimized_or.ok()) {
          auto run_result = RunAndCompareTwoModules(
              std::move(*optimized_module_or), std::move(*ref_optimized_or),
              ErrorSpec{/*aabs=*/0.02, /*arel=*/0.05},
              /*run_hlo_passes=*/false);
          status = run_result ? "SUCCESS" : "FAILURE";
          if (!run_result) {
            LOG(ERROR) << "RunAndCompare error: " << run_result.message();
          }
        } else {
          LOG(ERROR) << "ref_optimized error: " << ref_optimized_or.status();
        }
      }
    }
  }

  CuDnnSupportResult actual = CuDnnSupportResult::kNotSupported;
  if (optimized_hlo == "__cudnn$fusion" ||
      optimized_hlo == "__cudnn$blockScaledDot") {
    actual = (cudnn_graph_pattern == "BLOCK_SCALE_DEQUANTIZE + MATMUL" &&
              status == "SUCCESS")
                 ? CuDnnSupportResult::kSupported
                 : CuDnnSupportResult::kFailed;
  }

  LOG(ERROR) << "Report CuDnnGraph: " << CuDnnSupportResultToString(actual);
  LOG(ERROR) << "Report CuDnnGraphPattern: " << cudnn_graph_pattern;
  LOG(ERROR) << "Report Status: " << status;

  CuDnnSupportResult expected = param.expected_result;
  EXPECT_EQ(actual, expected);

  switch (expected) {
    case CuDnnSupportResult::kSupported:
      EXPECT_EQ(optimized_hlo, "__cudnn$fusion");
      EXPECT_EQ(cudnn_graph_pattern, "BLOCK_SCALE_DEQUANTIZE + MATMUL");
      EXPECT_EQ(status, "SUCCESS");
      break;
    case CuDnnSupportResult::kFailed:
      EXPECT_EQ(optimized_hlo, "__cudnn$fusion");
      EXPECT_EQ(cudnn_graph_pattern, "BLOCK_SCALE_DEQUANTIZE + MATMUL");
      EXPECT_EQ(status, "FAILURE");
      break;
    case CuDnnSupportResult::kNotSupported:
      EXPECT_EQ(optimized_hlo, "dot");
      EXPECT_EQ(cudnn_graph_pattern, "N/A");
      EXPECT_EQ(status, "SUCCESS");
      break;
  }
}

INSTANTIATE_TEST_SUITE_P(CudnnScaledDotCoverageTestSuite,
                         CudnnScaledDotCoverageTest,
                         ::testing::ValuesIn(GetCoverageTestCases()),
                         ScaledDotCoverageTestParamToString);

}  // namespace
}  // namespace gpu
}  // namespace xla

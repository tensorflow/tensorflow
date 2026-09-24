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
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <optional>
#include <ostream>
#include <random>
#include <string>
#include <tuple>
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
#include "absl/status/status_matchers.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_replace.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "absl/strings/substitute.h"
#include "absl/types/span.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "xla/backends/gpu/codegen/triton/test_utils.h"
#include "xla/backends/gpu/codegen/triton/xtile_compiler.h"
#include "xla/backends/gpu/codegen/triton/xtile_test_base.h"
#include "xla/backends/gpu/tests/gpu_pjrt_codegen_test.h"
#include "xla/backends/gpu/transforms/composite_rewriter.h"
#include "xla/codegen/xtile/block_level_parameters.h"
#include "xla/error_spec.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/hlo/testlib/filecheck.h"
#include "xla/hlo/testlib/verified_hlo_module.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/primitive_util.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/gpu_compiler.h"
#include "xla/service/gpu/gpu_device_info_for_tests.h"
#include "xla/stream_executor/cuda/cuda_compute_capability.h"
#include "xla/stream_executor/device_description.h"
#include "xla/tests/hlo_interpreter_reference_mixin.h"
#include "xla/tests/test_utils.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/types.h"
#include "xla/xla.pb.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace gpu {
namespace {

using ::absl_testing::IsOk;
using ::absl_testing::IsOkAndHolds;
using ::testing::HasSubstr;
using ::xla::xtile::BlockLevelFusionConfig;
using ::xla::xtile::BlockLevelParameters;

std::string TilingParametersToString(bool tiling_propagation_enabled) {
  return tiling_propagation_enabled ? "SymbolicTiling" : "ExperimentalTiling";
}

class TritonEmitterTest
    : public HloInterpreterReferenceMixin<GpuPjRtCodegenTest>,
      public XTileTestBase {
 public:
  virtual bool EnableTilingPropagation() const = 0;
  DebugOptions GetDebugOptionsForTest() const override {
    DebugOptions debug_options = HloInterpreterReferenceMixin<
        GpuPjRtCodegenTest>::GetDebugOptionsForTest();
    debug_options.set_xla_gpu_unsupported_enable_triton_multi_output_fusion(
        true);
    debug_options.set_xla_gpu_experimental_disable_binary_libraries(true);
    debug_options.set_xla_gpu_experimental_enable_tiling_propagation(
        EnableTilingPropagation());
    return debug_options;
  }

  const stream_executor::GpuComputeCapability& GpuComputeCapability() {
    return device_description().gpu_compute_capability();
  }
  stream_executor::CudaComputeCapability GetCudaComputeCapability() {
    return device_description().cuda_compute_capability();
  }
  absl::StatusOr<
      std::pair<mlir::OwningOpRef<mlir::ModuleOp>, std::unique_ptr<HloModule>>>
  CreateXTileIrAndFileCheck(absl::string_view hlo_text,
                            absl::string_view triton_fusion_name,
                            absl::string_view filecheck_pattern) {
    ABSL_ASSIGN_OR_RETURN(std::unique_ptr<VerifiedHloModule> module,
                     ParseAndReturnVerifiedModule(hlo_text));
    return XTileTestBase::CreateXTileIrAndFileCheck(
        std::move(module), triton_fusion_name, filecheck_pattern);
  }
  absl::Status CreateTritonIrFromHloTextAndFileCheck(
      absl::string_view hlo_text, absl::string_view triton_fusion_name,
      absl::string_view filecheck_pattern) {
    ABSL_ASSIGN_OR_RETURN(std::unique_ptr<VerifiedHloModule> module,
                     ParseAndReturnVerifiedModule(hlo_text));
    return CreateTritonIrAndFileCheck(module.get(), triton_fusion_name,
                                      filecheck_pattern);
  }
  absl::Status CreateTritonIrFromHloTextAndFileCheckForDot(
      absl::string_view hlo_text, absl::string_view triton_fusion_name,
      absl::string_view filecheck_pattern) {
    ABSL_ASSIGN_OR_RETURN(std::unique_ptr<VerifiedHloModule> module,
                     ParseAndReturnVerifiedModule(hlo_text));
    return CreateTritonIrAndFileCheckForDot(module.get(), triton_fusion_name,
                                            filecheck_pattern);
  }
};

class TritonEmitterTestWithTilingParam
    : public TritonEmitterTest,
      public ::testing::WithParamInterface<bool> {
 public:
  bool EnableTilingPropagation() const override { return GetParam(); }
};

TEST_P(TritonEmitterTestWithTilingParam,
       ScaledDotIsSupportedByReferencePlatform) {
  constexpr absl::string_view kHloText = R"(
    HloModule ScaledDotIsSupportedByReferencePlatform

    ENTRY entry {
     lhs = bf16[16,128] parameter(0)
     rhs = bf16[128,16] parameter(1)
     lhs_scale = bf16[1,4] parameter(2)
     rhs_scale = bf16[4,1] parameter(3)
     ROOT dot = bf16[16,16] scaled-dot(lhs, rhs, lhs_scale, rhs_scale),
         lhs_contracting_dims={1},
         rhs_contracting_dims={0}
    }
  )";

  EXPECT_TRUE(RunAndCompare(kHloText, ErrorSpec{/*aabs=*/1e-3, /*arel=*/1e-3}));
}

INSTANTIATE_TEST_SUITE_P(TritonEmitterTestWithTilingParamTestSuite,
                         TritonEmitterTestWithTilingParam, ::testing::Bool(),
                         [](const ::testing::TestParamInfo<bool>& info) {
                           return TilingParametersToString(info.param);
                         });

struct ScaleDotTestParams {
  std::string lhs_type;
  std::string rhs_type;
  std::string lhs_scale_type;
  std::string rhs_scale_type;
  std::string output_type;
  std::string expected_triton_type;

  std::string PrepareHloText(absl::string_view hlo_template) const {
    return absl::StrReplaceAll(hlo_template,
                               {{"$lhs_type", lhs_type},
                                {"$rhs_type", rhs_type},
                                {"$lhs_scale_type", lhs_scale_type},
                                {"$rhs_scale_type", rhs_scale_type},
                                {"$output_type", output_type}});
  }
  static std::string ToString(
      const ::testing::TestParamInfo<ScaleDotTestParams>& info) {
    const ScaleDotTestParams& params = info.param;
    auto name = absl::StrCat(params.lhs_type, "_", params.rhs_type, "_",
                             params.lhs_scale_type, "_", params.rhs_scale_type,
                             "_", params.output_type);
    absl::StrReplaceAll({{"[", "_"}, {"]", "_"}, {",", "x"}}, &name);
    return name;
  }
};

std::ostream& operator<<(std::ostream& stream, const ScaleDotTestParams& tc) {
  return stream << "{\n\tlhs_type:" << tc.lhs_type
                << ",\n\trhs_type:" << tc.rhs_type
                << ",\n\tlhs_scale_type:" << tc.lhs_scale_type
                << ",\n\trhs_scale_type:" << tc.rhs_scale_type
                << ",\n\toutput_type:" << tc.output_type << "\n}";
}

class TritonScaledDotGemmTest : public TritonEmitterTest,
                                public ::testing::WithParamInterface<
                                    std::tuple<ScaleDotTestParams, bool>> {
 public:
  bool EnableTilingPropagation() const override {
    return std::get<1>(GetParam());
  }

 public:
  DebugOptions GetDebugOptionsForTest() const override {
    DebugOptions debug_options = TritonEmitterTest::GetDebugOptionsForTest();
    debug_options.set_xla_gpu_experimental_scaled_dot_with_triton(true);
    debug_options.set_xla_gpu_autotune_level(0);
    debug_options.set_xla_gpu_cublas_fallback(false);
    return debug_options;
  }
};

TEST_P(TritonScaledDotGemmTest,
       FP8ScaledDotCompilesToPtxIntrinsicsWhenAvailable) {
  const ScaleDotTestParams& params = std::get<0>(GetParam());
  constexpr absl::string_view kHloTextTemplate = R"hlo(
HloModule m

triton_dot {
  lhs = $lhs_type parameter(0)
  rhs = $rhs_type parameter(1)
  lhs_scale = $lhs_scale_type parameter(2)
  rhs_scale = $rhs_scale_type parameter(3)
  ROOT _ = $output_type{1,0} scaled-dot(lhs, rhs, lhs_scale, rhs_scale),
    lhs_contracting_dims={1},
    rhs_contracting_dims={0},
    backend_config={sizes:[128]}
}

ENTRY e {
  lhs = $lhs_type{1,0} parameter(0)
  rhs = $rhs_type{1,0} parameter(1)
  lhs_scale = $lhs_scale_type{1,0} parameter(2)
  rhs_scale = $rhs_scale_type{1,0} parameter(3)
  ROOT _ = $output_type{1,0} fusion(lhs, rhs, lhs_scale, rhs_scale),
    kind=kCustom,
    calls=triton_dot,
    backend_config={
      "fusion_backend_config": {
        kind: "__triton_nested_gemm_fusion",
        "block_level_fusion_config":{
          "output_tiles":[{"sizes":["128", "256"]}],
          "num_warps":"4",
          "num_stages":"1",
          "num_ctas":"1"
        }
      }
    }
}
)hlo";

  auto hlo_text = params.PrepareHloText(kHloTextTemplate);

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(hlo_text));

  constexpr absl::string_view kExpectedTritonIrTmpl = R"(
      CHECK: tt.dot_scaled
      CHECK: tensor<128x128x$triton_type>, tensor<128x4xi8>
      CHECK: tensor<128x256x$triton_type>, tensor<256x4xi8>
      CHECK: -> tensor<128x256xf32>
  )";
  auto expected_triton_ir = absl::StrReplaceAll(
      kExpectedTritonIrTmpl, {{"$triton_type", params.expected_triton_type}});
  EXPECT_THAT(
      CreateTritonIrAndFileCheckForDot(
          *module->GetComputationWithName("triton_dot"), expected_triton_ir),
      IsOk());
  if (GetCudaComputeCapability().IsAtLeastBlackwell()) {
    CompileAndOptionallyVerifyPtx(
        std::move(module), R"(CHECK: mxf8f6f4.block_scale.scale_vec::1X)");
  }
}

TEST_P(TritonScaledDotGemmTest, FP8ScaledDotGetsFusedAndExecutesCorrectly) {
  const ScaleDotTestParams& params = std::get<0>(GetParam());
  if (auto cc = GpuComputeCapability().cuda_compute_capability();
      cc && !cc->IsAtLeastBlackwell()) {
    GTEST_SKIP() << "Skipping test for pre-Blackwell GPUs.";
  }
  constexpr absl::string_view kHloTextTemplate = R"hlo(
HloModule FP8ScaledDotGetsFusedAndExecutesCorrectly

ENTRY e {
  lhs = $lhs_type parameter(0)
  rhs = $rhs_type parameter(2)
  lhs_scale = $lhs_scale_type parameter(1)
  rhs_scale = $rhs_scale_type parameter(3)
  ROOT _ = $output_type{1,0} scaled-dot(lhs, rhs, lhs_scale, rhs_scale),
    lhs_contracting_dims={1},
    rhs_contracting_dims={0}
}
)hlo";

  auto hlo_text = params.PrepareHloText(kHloTextTemplate);

  ASSERT_OK_AND_ASSIGN(auto optimized_module, GetOptimizedModule(hlo_text));
  EXPECT_THAT(RunFileCheck(optimized_module->ToString(), R"(
    CHECK: fusion
    CHECK: ROOT {{.*}} scaled-dot
    CHECK: ENTRY
    CHECK: __triton_nested_gemm_fusion
  )"),
              IsOkAndHolds(true));
  EXPECT_TRUE(RunAndCompareNoHloPasses(
      std::move(optimized_module), ErrorSpec{/*aabs=*/1e-3, /*arel=*/1e-3}));
}

INSTANTIATE_TEST_SUITE_P(
    TritonScaledDotGemmTest, TritonScaledDotGemmTest,
    ::testing::Combine(
        ::testing::Values(
            ScaleDotTestParams{"f8e4m3fn[128,128]", "f8e4m3fn[128,256]",
                               "f8e8m0fnu[128,4]", "f8e8m0fnu[4,256]",
                               "bf16[128,256]", "f8E4M3FN"},
            ScaleDotTestParams{"f8e5m2[128,128]", "f8e5m2[128,256]",
                               "f8e8m0fnu[128,4]", "f8e8m0fnu[4,256]",
                               "bf16[128,256]", "f8E5M2"}),
        ::testing::Bool()),
    [](const ::testing::TestParamInfo<std::tuple<ScaleDotTestParams, bool>>&
           info) {
      return absl::StrCat(ScaleDotTestParams::ToString(
                              ::testing::TestParamInfo<ScaleDotTestParams>(
                                  std::get<0>(info.param), info.index)),
                          TilingParametersToString(std::get<1>(info.param)));
    });

class TritonScaledDotTestBase : public TritonEmitterTest {
 public:
  DebugOptions GetDebugOptionsForTest() const override {
    DebugOptions debug_options = TritonEmitterTest::GetDebugOptionsForTest();
    debug_options.set_xla_gpu_experimental_scaled_dot_with_triton(true);
    debug_options.set_xla_gpu_autotune_level(0);
    debug_options.set_xla_gpu_cublas_fallback(false);
    return debug_options;
  }

  HloComputation* GetFirstComputationWithInstruction(const HloModule& module,
                                                     HloOpcode opcode) const {
    for (const auto& computation : module.computations()) {
      for (const auto& instruction : computation->instructions()) {
        if (instruction->opcode() == opcode) {
          return computation;
        }
      }
    }
    return nullptr;
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
    if (arguments.size() != 4) {
      return absl::InternalError(absl::StrCat(
          "Expected 4 scaled-dot arguments, got ", arguments.size()));
    }
    ABSL_RETURN_IF_ERROR(PopulateScale(&arguments[2], &engine));
    ABSL_RETURN_IF_ERROR(PopulateScale(&arguments[3], &engine));
    return arguments;
  }
};

class TritonScaledDotTest : public TritonScaledDotTestBase,
                            public ::testing::WithParamInterface<bool> {
 public:
  bool EnableTilingPropagation() const override { return GetParam(); }
};

struct Fp4ScaledDotTypeCase {
  PrimitiveType lhs_type;
  PrimitiveType rhs_type;
  PrimitiveType scale_type;
  int block_size = 32;
};

using Fp4ScaledDotTestParam = std::tuple<Fp4ScaledDotTypeCase,
                                         /*lhs_k_minor=*/bool,
                                         /*rhs_k_minor=*/bool,
                                         /*tiling_enabled=*/bool>;

std::string Fp4ScaledDotTestParamToString(
    const ::testing::TestParamInfo<Fp4ScaledDotTestParam>& info) {
  const auto& [type_case, lhs_k_minor, rhs_k_minor, tiling_enabled] =
      info.param;
  return absl::StrCat(PrimitiveType_Name(type_case.lhs_type), "_",
                      PrimitiveType_Name(type_case.rhs_type), "_",
                      PrimitiveType_Name(type_case.scale_type), "_Block",
                      type_case.block_size, "_Lhs", lhs_k_minor, "_Rhs",
                      rhs_k_minor, "_",
                      TilingParametersToString(tiling_enabled));
}

class TritonFp4ScaledDotTest
    : public TritonScaledDotTestBase,
      public ::testing::WithParamInterface<Fp4ScaledDotTestParam> {
 public:
  bool EnableTilingPropagation() const override {
    return std::get<3>(GetParam());
  }

  void RunFp4ScaledDotExecutionTest(PrimitiveType lhs_type,
                                    PrimitiveType rhs_type,
                                    PrimitiveType scale_type, int block_size,
                                    bool lhs_k_minor, bool rhs_k_minor) {
    constexpr absl::string_view kHloTemplate = R"hlo(
HloModule m

ENTRY e {
  lhs = $lhs_type[$lhs_shape] parameter(0)
  rhs = $rhs_type[$rhs_shape] parameter(1)
  lhs_scale = $scale_type[$lhs_scale_shape] parameter(2)
  rhs_scale = $scale_type[$rhs_scale_shape] parameter(3)
  ROOT dot = bf16[$output_shape] scaled-dot(lhs, rhs, lhs_scale, rhs_scale),
    lhs_contracting_dims={$lhs_contracting_dim},
    rhs_contracting_dims={$rhs_contracting_dim}
}
)hlo";

    ASSERT_TRUE(scale_type == F8E8M0FNU || scale_type == F8E4M3FN);

    constexpr int64_t m = 128;
    constexpr int64_t n = 128;
    constexpr int64_t k = 256;
    const int64_t scale_k = k / block_size;

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

    LOG(INFO) << "TritonFp4ScaledDotTest Params:"
              << "\n  LHS Type: "
              << primitive_util::LowercasePrimitiveTypeName(lhs_type)
              << "\n  RHS Type: "
              << primitive_util::LowercasePrimitiveTypeName(rhs_type)
              << "\n  Scale Type: "
              << primitive_util::LowercasePrimitiveTypeName(scale_type)
              << "\n  Block Size: " << block_size
              << "\n  LHS Contracting Dim: " << (lhs_k_minor ? "1" : "0")
              << "\n  RHS Contracting Dim: " << (rhs_k_minor ? "1" : "0");

    std::string hlo = absl::StrReplaceAll(
        kHloTemplate,
        {{"$lhs_type", primitive_util::LowercasePrimitiveTypeName(lhs_type)},
         {"$rhs_type", primitive_util::LowercasePrimitiveTypeName(rhs_type)},
         {"$scale_type",
          primitive_util::LowercasePrimitiveTypeName(scale_type)},
         {"$lhs_shape", lhs_shape},
         {"$rhs_shape", rhs_shape},
         {"$lhs_scale_shape", lhs_scale_shape},
         {"$rhs_scale_shape", rhs_scale_shape},
         {"$output_shape", absl::StrCat(m, ",", n)},
         {"$lhs_contracting_dim", lhs_k_minor ? "1" : "0"},
         {"$rhs_contracting_dim", rhs_k_minor ? "1" : "0"}});
    if (scale_type == F8E8M0FNU && block_size == 16 &&
        GetCudaComputeCapability().IsAtLeastBlackwell() && !lhs_k_minor) {
#ifndef NDEBUG
      EXPECT_DEATH(
          { (void)GetOptimizedModule(hlo); },
          "MMAv5 with kind=mxf4nvf4 does not support transpose");
      return;
#endif
    }
    ASSERT_OK_AND_ASSIGN(auto optimized_module, GetOptimizedModule(hlo));
    HloComputation* scaled_dot_computation = GetFirstComputationWithInstruction(
        *optimized_module, HloOpcode::kScaledDot);
    EXPECT_THAT(CreateTritonIrAndFileCheckForDot(*scaled_dot_computation,
                                                 "CHECK: tt.dot_scaled"),
                IsOk());
    ASSERT_OK_AND_ASSIGN(std::vector<Literal> arguments,
                         MakeScaledDotArguments(optimized_module.get()));
    if (block_size == 16 && GetCudaComputeCapability().IsAtLeastBlackwell()) {
      EXPECT_EXIT(
          {
            EXPECT_THAT(RunAndCompareNoHloPasses(
                            std::move(optimized_module),
                            LiteralUtil::MakePointers(arguments),
                            ErrorSpec{/*aabs=*/1e-3, /*arel=*/1e-3})
                            .message(),
                        HasSubstr("CUDA_ERROR_INVALID_INSTRUCTION"));
            std::_Exit(1);
          },
          ::testing::ExitedWithCode(1), "");
      return;
    }
    EXPECT_TRUE(RunAndCompareNoHloPasses(
        std::move(optimized_module), LiteralUtil::MakePointers(arguments),
        ErrorSpec{/*aabs=*/1e-3, /*arel=*/1e-3}));
  }
};

INSTANTIATE_TEST_SUITE_P(TritonScaledDotTestSuite, TritonScaledDotTest,
                         ::testing::Bool(),
                         [](const ::testing::TestParamInfo<bool>& info) {
                           return TilingParametersToString(info.param);
                         });

TEST_P(TritonFp4ScaledDotTest, Executes) {
  auto cc = GpuComputeCapability().cuda_compute_capability();
  if (!cc || !cc->IsAtLeastHopper()) {
    GTEST_SKIP() << "Scaled dot isn't supported by Triton for pre-Hopper GPUs.";
  }
  const Fp4ScaledDotTypeCase& type_case = std::get<0>(GetParam());
  RunFp4ScaledDotExecutionTest(
      type_case.lhs_type, type_case.rhs_type, type_case.scale_type,
      type_case.block_size, std::get<1>(GetParam()), std::get<2>(GetParam()));
}

INSTANTIATE_TEST_SUITE_P(
    TritonFp4ScaledDotTestSuite, TritonFp4ScaledDotTest,
    ::testing::Combine(
        ::testing::Values(
            Fp4ScaledDotTypeCase{F4E2M1FN, F4E2M1FN, F8E8M0FNU,
                                 32},  // MXFP4 x MXFP4 (block 32)
            Fp4ScaledDotTypeCase{F4E2M1FN, F4E2M1FN, F8E8M0FNU,
                                 16},  // MXFP4 x MXFP4 (block 16)
            Fp4ScaledDotTypeCase{F4E2M1FN, F8E4M3FN, F8E8M0FNU,
                                 32},  // MXFP4 x MXFP8 (MMAv5 mxf8f6f4)
            Fp4ScaledDotTypeCase{F8E4M3FN, F4E2M1FN, F8E8M0FNU,
                                 32},  // MXFP8 x MXFP4 (MMAv5 mxf8f6f4)
            Fp4ScaledDotTypeCase{F4E2M1FN, F8E5M2, F8E8M0FNU,
                                 32},  // MXFP4 x MXE5M2
            Fp4ScaledDotTypeCase{F8E5M2, F4E2M1FN, F8E8M0FNU,
                                 32},  // MXE5M2 x MXFP4
            Fp4ScaledDotTypeCase{F4E2M1FN, F4E2M1FN, F8E4M3FN,
                                 16}),  // NVFP4 x NVFP4 (block 16)
        ::testing::Bool(), ::testing::Bool(), ::testing::Bool()),
    Fp4ScaledDotTestParamToString);

TEST_P(TritonScaledDotTest,
       ScaledDotWithOmmittedLhsScaleGetFusedAndExecutedCorrectly) {
  if (auto cc = GpuComputeCapability().cuda_compute_capability();
      cc && !cc->IsAtLeastHopper()) {
    GTEST_SKIP() << "Scaled dot isn't supported by Triton for pre-Hopper GPUs.";
  }
  constexpr absl::string_view kHloTextTemplate = R"hlo(
HloModule ScaledDotWithOmmittedLhsScaleGetFusedAndExecutedCorrectly

ENTRY e {
  lhs = bf16[3,128,128] parameter(0)
  rhs = f8e4m3fn[3,128,128] parameter(1)
  constant = bf16[1,1,1] constant(1.0)
  rhs_scale = f8e8m0fnu[3,128,4] parameter(2)
  ROOT _ = bf16[3,128,128] scaled-dot(lhs, rhs, constant, rhs_scale),
    lhs_batch_dims={0},
    rhs_batch_dims={0},
    lhs_contracting_dims={2},
    rhs_contracting_dims={2}
}
)hlo";

  ASSERT_OK_AND_ASSIGN(auto optimized_module,
                       GetOptimizedModule(kHloTextTemplate));
  constexpr absl::string_view kExpectedOptimizedHLO = R"(
    CHECK: fusion
    CHECK: ROOT {{.*}} scaled-dot
    CHECK: ENTRY
    CHECK: __triton_nested_gemm_fusion
  )";
  EXPECT_THAT(RunFileCheck(optimized_module->ToString(), kExpectedOptimizedHLO),
              true);
  for (const auto& computation : optimized_module->computations()) {
    for (const auto& instruction : computation->instructions()) {
      if (instruction->opcode() == HloOpcode::kScaledDot) {
        LOG(INFO) << "Instruction: " << instruction->name();
      }
    }
  }

  HloComputation* scaled_dot_computation = GetFirstComputationWithInstruction(
      *optimized_module, HloOpcode::kScaledDot);
  constexpr absl::string_view kExpectedTritonIr = R"(
      CHECK: tt.dot_scaled
      CHECK: tensor<128x128xbf16>
      CHECK: tensor<128x16xf8E4M3FN>, tensor<16x4xi8>
      CHECK: -> tensor<128x16xf32>
  )";
  EXPECT_THAT(CreateTritonIrAndFileCheckForDot(*scaled_dot_computation,
                                               kExpectedTritonIr),
              IsOk());

  EXPECT_TRUE(RunAndCompareNoHloPasses(
      std::move(optimized_module), ErrorSpec{/*aabs=*/1e-3, /*arel=*/1e-3}));
}

TEST_P(TritonScaledDotTest, FP8ScaledDotLhsKNotMinorDim) {
  if (!GetCudaComputeCapability().IsAtLeastBlackwell()) {
    GTEST_SKIP() << "FP8 scaled dot requires Blackwell+";
  }
  constexpr absl::string_view kHloTextTemplate = R"hlo(
HloModule FP8ScaledDotLhsKNotMinorDim

ENTRY e {
  lhs = f8e4m3fn[128,64] parameter(0)
  lhs_scale = f8e8m0fnu[4,64] parameter(1)
  rhs = f8e4m3fn[128,256] parameter(2)
  rhs_scale = f8e8m0fnu[4,256] parameter(3)
  ROOT _ = bf16[64,256]{1,0} scaled-dot(lhs, rhs, lhs_scale, rhs_scale),
    lhs_contracting_dims={0},
    rhs_contracting_dims={0}
}
)hlo";

  ASSERT_OK_AND_ASSIGN(auto optimized_module,
                       GetOptimizedModule(kHloTextTemplate));
  constexpr absl::string_view kExpectedOptimizedHLO = R"(
    CHECK: fusion
    CHECK: ROOT {{.*}} scaled-dot
    CHECK: ENTRY
    CHECK: __triton_nested_gemm_fusion
  )";
  EXPECT_THAT(RunFileCheck(optimized_module->ToString(), kExpectedOptimizedHLO),
              true);

  HloComputation* scaled_dot_computation = GetFirstComputationWithInstruction(
      *optimized_module, HloOpcode::kScaledDot);
  constexpr absl::string_view kExpectedTritonIr = R"(
      CHECK: tt.dot_scaled
  )";
  EXPECT_THAT(CreateTritonIrAndFileCheckForDot(*scaled_dot_computation,
                                               kExpectedTritonIr),
              IsOk());

  EXPECT_TRUE(RunAndCompareNoHloPasses(
      std::move(optimized_module), ErrorSpec{/*aabs=*/1e-3, /*arel=*/1e-3}));
}

TEST_P(TritonScaledDotTest, ScaledDotWithBatchGetFusedAndExecutedCorrectly) {
  if (auto cc = GpuComputeCapability().cuda_compute_capability();
      cc && !cc->IsAtLeastHopper()) {
    GTEST_SKIP() << "Scaled dot isn't supported by Triton for pre-Hopper GPUs.";
  }
  constexpr absl::string_view kHloTextTemplate = R"hlo(
HloModule ScaledDotWithBatchGetFusedAndExecutedCorrectly

ENTRY e {
  lhs = f8e4m3fn[3,128,128] parameter(0)
  rhs = f8e4m3fn[3,128,128] parameter(1)
  lhs_scale = f8e8m0fnu[3,128,4] parameter(2)
  rhs_scale = f8e8m0fnu[3,128,4 ] parameter(3)
  ROOT _ = bf16[3,128,128] scaled-dot(lhs, rhs, lhs_scale, rhs_scale),
    lhs_batch_dims={0},
    rhs_batch_dims={0},
    lhs_contracting_dims={2},
    rhs_contracting_dims={2}
}
)hlo";

  ASSERT_OK_AND_ASSIGN(auto optimized_module,
                       GetOptimizedModule(kHloTextTemplate));
  constexpr absl::string_view kExpectedOptimizedHLO = R"(
    CHECK: fusion
    CHECK: ROOT {{.*}} scaled-dot
    CHECK: ENTRY
    CHECK: __triton_nested_gemm_fusion
  )";
  EXPECT_THAT(RunFileCheck(optimized_module->ToString(), kExpectedOptimizedHLO),
              true);

  HloComputation* scaled_dot_computation = GetFirstComputationWithInstruction(
      *optimized_module, HloOpcode::kScaledDot);
  constexpr absl::string_view kExpectedTritonIr = R"(
      CHECK: tt.dot_scaled
      CHECK: tensor<128x128xf8E4M3FN>, tensor<128x4xi8>
      CHECK: tensor<128x16xf8E4M3FN>, tensor<16x4xi8>
      CHECK: -> tensor<128x16xf32>
  )";
  EXPECT_THAT(CreateTritonIrAndFileCheckForDot(*scaled_dot_computation,
                                               kExpectedTritonIr),
              IsOk());

  EXPECT_TRUE(RunAndCompareNoHloPasses(
      std::move(optimized_module), ErrorSpec{/*aabs=*/1e-3, /*arel=*/1e-3}));
}

TEST_P(TritonScaledDotTest, BroadcastAndReshapeGetFused) {
  if (auto cc = GpuComputeCapability().cuda_compute_capability();
      cc && !cc->IsAtLeastHopper()) {
    GTEST_SKIP() << "Scaled dot isn't supported by Triton for pre-Hopper GPUs.";
  }
  constexpr absl::string_view kHloTextTemplate = R"hlo(
HloModule ScaledDotWithBatchGetFusedAndExecutedCorrectly

ENTRY e {
  lhs = f8e4m3fn[3,128,128] parameter(0)
  rhs = f8e4m3fn[3,128,128] parameter(1)
  lhs_scale = f8e8m0fnu[3,128,1] parameter(2)
  lhs_scale_broadcasted = f8e8m0fnu[3,128,1,4] broadcast(lhs_scale),
      dimensions={0,1,2}
  lhs_scale_reshaped = f8e8m0fnu[3,128,4] reshape(lhs_scale_broadcasted)
  rhs_scale = f8e8m0fnu[3,128,1] parameter(3)
  rhs_scale_broadcasted = f8e8m0fnu[3,128,1,4] broadcast(rhs_scale),
      dimensions={0,1,2}
  rhs_scale_reshaped = f8e8m0fnu[3,128,4] reshape(rhs_scale_broadcasted)
  ROOT _ = bf16[3,128,128] scaled-dot(
      lhs,
      rhs,
      lhs_scale_reshaped,
      rhs_scale_reshaped),
    lhs_batch_dims={0},
    rhs_batch_dims={0},
    lhs_contracting_dims={2},
    rhs_contracting_dims={2}
}
  )hlo";

  ASSERT_OK_AND_ASSIGN(auto optimized_module,
                       GetOptimizedModule(kHloTextTemplate));
  constexpr absl::string_view kExpectedOptimizedHLO = R"(
    CHECK: %fusion
    CHECK: %{{.*}} = f8e8m0fnu[3,128,4]{2,1,0} broadcast(%{{.*}}), dimensions={0,1}
    CHECK: %{{.*}} = f8e8m0fnu[3,128,4]{2,1,0} broadcast(%{{.*}}), dimensions={0,1}
    CHECK: ROOT {{.*}} scaled-dot
    CHECK: ENTRY
    CHECK: __triton_nested_gemm_fusion
  )";
  EXPECT_THAT(RunFileCheck(optimized_module->ToString(), kExpectedOptimizedHLO),
              true);

  HloComputation* scaled_dot_computation = GetFirstComputationWithInstruction(
      *optimized_module, HloOpcode::kScaledDot);
  constexpr absl::string_view kExpectedTritonIr = R"(
      CHECK: tt.dot_scaled
      CHECK: tensor<128x128xf8E4M3FN>, tensor<128x4xi8>
      CHECK: tensor<128x16xf8E4M3FN>, tensor<16x4xi8>
      CHECK: -> tensor<128x16xf32>
  )";
  EXPECT_THAT(CreateTritonIrAndFileCheckForDot(*scaled_dot_computation,
                                               kExpectedTritonIr),
              IsOk());

  EXPECT_TRUE(RunAndCompareNoHloPasses(
      std::move(optimized_module), ErrorSpec{/*aabs=*/1e-3, /*arel=*/1e-3}));
}

TEST_P(TritonScaledDotTest, Mxfp8ScaledDotSmallBlockKAndNExecutes) {
  if (!GetCudaComputeCapability().IsAtLeastHopper()) {
    GTEST_SKIP() << "Requires Hopper+.";
  }
  constexpr absl::string_view kHloText = R"hlo(
HloModule m

fusion__ {
  parameter_0 = f8e4m3fn[128,256]{1,0} parameter(0)
  parameter_1 = f8e4m3fn[256,128]{1,0} parameter(1)
  parameter_2 = f8e8m0fnu[128,8]{1,0} parameter(2)
  parameter_3 = f8e8m0fnu[8,128]{1,0} parameter(3)
  ROOT _.1 = bf16[128,128]{1,0} scaled-dot(
      parameter_0, parameter_1, parameter_2, parameter_3),
    lhs_contracting_dims={1}, rhs_contracting_dims={0},
    backend_config={"sizes":["64"]}
}

ENTRY e {
  lhs = f8e4m3fn[128,256]{1,0} parameter(0)
  rhs = f8e4m3fn[256,128]{1,0} parameter(1)
  lhs_scale = f8e8m0fnu[128,8]{1,0} parameter(2)
  rhs_scale = f8e8m0fnu[8,128]{1,0} parameter(3)
  ROOT fusion = bf16[128,128]{1,0} fusion(
      lhs, rhs, lhs_scale, rhs_scale),
    kind=kCustom, calls=fusion__,
    backend_config={"fusion_backend_config":{
      "kind":"__triton_nested_gemm_fusion",
      "block_level_fusion_config":{
        "output_tiles":[{"sizes":["128","16"]}],
        "num_warps":"4","num_ctas":"1","num_stages":"1"}}}
}
)hlo";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHloText));
  EXPECT_TRUE(RunAndCompareNoHloPasses(
      std::move(module), ErrorSpec{/*aabs=*/1e-3, /*arel=*/1e-3}));
}

TEST_P(TritonScaledDotTest, Fp4Succeeds) {
  if (!GetCudaComputeCapability().IsAtLeastBlackwell()) {
    GTEST_SKIP() << "Scaled dot with FP4 isn't supported by Triton for "
                    "pre-Blackwell GPUs.";
  }
  constexpr absl::string_view kHloTextTemplate = R"hlo(
    HloModule jit_scaled_dot_fn

    ENTRY %main.2 {
      %lhs = f4e2m1fn[1,1024,256]{2,1,0} parameter(0)
      %rhs = f4e2m1fn[1,256,256]{2,1,0} parameter(1)
      %lhs_scale = f8e8m0fnu[1,1024,8]{2,1,0} parameter(2)
      %rhs_scale = f8e8m0fnu[1,8,256]{2,1,0} parameter(3)
      ROOT %scaled-dot = bf16[1,1024,256]{2,1,0} scaled-dot(%lhs, %rhs, %lhs_scale, %rhs_scale),
          lhs_batch_dims={0},
          lhs_contracting_dims={2},
          rhs_batch_dims={0},
          rhs_contracting_dims={1}
    }
  )hlo";
  ASSERT_OK_AND_ASSIGN(auto optimized_module,
                       GetOptimizedModule(kHloTextTemplate));
  HloComputation* scaled_dot_computation = GetFirstComputationWithInstruction(
      *optimized_module, HloOpcode::kScaledDot);
  ASSERT_OK_AND_ASSIGN(GpuBackendConfig gpu_config,
                       scaled_dot_computation->FusionInstruction()
                           ->backend_config<GpuBackendConfig>());
  const BlockLevelFusionConfig& block_level_fusion_config =
      gpu_config.fusion_backend_config().block_level_fusion_config();
  ASSERT_EQ(block_level_fusion_config.output_tiles_size(), 1);
  const auto& output_tile_sizes =
      block_level_fusion_config.output_tiles(0).sizes();
  ASSERT_EQ(output_tile_sizes.size(), 3);
  const int64_t output_m = output_tile_sizes.Get(1);
  const int64_t output_n = output_tile_sizes.Get(2);
  ASSERT_EQ(output_n % 2, 0);
  const std::string expected_triton_ir =
      absl::Substitute(R"(
      CHECK: tt.dot_scaled
      CHECK: tensor<$0x64xi8>, tensor<$0x4xi8>
      CHECK: tensor<128x$1xi8>, tensor<$2x4xi8>
      CHECK: -> tensor<$0x$2xf32>
  )",
                       output_m, output_n / 2, output_n);

  EXPECT_THAT(CreateTritonIrAndFileCheckForDot(*scaled_dot_computation,
                                               expected_triton_ir),
              IsOk());

  EXPECT_TRUE(RunAndCompareNoHloPasses(
      std::move(optimized_module), ErrorSpec{/*aabs=*/1e-3, /*arel=*/1e-3}));
}

TEST_P(TritonScaledDotTest, Mxfp4KPackedRhsBlackwellLowersWithTranspose) {
  if (!GetCudaComputeCapability().IsAtLeastBlackwell()) {
    GTEST_SKIP() << "Requires Blackwell+.";
  }
  constexpr absl::string_view kHloText = R"hlo(
HloModule m
fusion__ {
  parameter_0 = f4e2m1fn[128,256]{1,0:E(4)} parameter(0)
  parameter_1 = f4e2m1fn[128,256]{1,0:E(4)} parameter(1)
  parameter_2 = f8e8m0fnu[128,8]{1,0} parameter(2)
  parameter_3 = f8e8m0fnu[128,8]{1,0} parameter(3)
  ROOT _.1 = bf16[128,128] scaled-dot(parameter_0, parameter_1, parameter_2, parameter_3),
    lhs_contracting_dims={1}, rhs_contracting_dims={1},
    backend_config={"sizes":["64"]}
}
ENTRY e {
  lhs = f4e2m1fn[128,256]{1,0:E(4)} parameter(0)
  rhs = f4e2m1fn[128,256]{1,0:E(4)} parameter(1)
  lhs_scale = f8e8m0fnu[128,8]{1,0} parameter(2)
  rhs_scale = f8e8m0fnu[128,8]{1,0} parameter(3)
  ROOT fusion = bf16[128,128] fusion(lhs, rhs, lhs_scale, rhs_scale),
    kind=kCustom, calls=fusion__,
    backend_config={"fusion_backend_config":{"kind":"__triton_nested_gemm_fusion",
      "block_level_fusion_config":{
        "output_tiles":[{"sizes":["128","128"]}],
        "num_warps":"4","num_ctas":"1","num_stages":"1"}}}
}
)hlo";
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHloText));
  HloComputation* scaled_dot_computation =
      GetFirstComputationWithInstruction(*module, HloOpcode::kScaledDot);
  EXPECT_THAT(CreateTritonIrAndFileCheckForDot(*scaled_dot_computation, R"(
      CHECK: tt.trans
      CHECK: tensor<128x32xi8> -> tensor<32x128xi8>
      CHECK-NOT: unrealized_conversion_cast
      CHECK: tt.dot_scaled
      CHECK: tensor<128x32xi8>, tensor<128x2xi8> * tensor<32x128xi8>, tensor<128x2xi8>
  )"),
              IsOk())
      << "We expect to see tt.trans and the follow up dot_scaled";
  EXPECT_TRUE(RunAndCompareNoHloPasses(
      std::move(module), ErrorSpec{/*aabs=*/1e-3, /*arel=*/1e-3}));
}

TEST_P(TritonScaledDotTest, GlobalScalerSucceeds) {
  if (!GetCudaComputeCapability().IsAtLeastHopper()) {
    GTEST_SKIP() << "Scaled dot isn't supported by Triton for pre-Hopper GPUs.";
  }
  constexpr absl::string_view kHloTextTemplate = R"hlo(
HloModule ScaledDotWithGlobalScaler

ENTRY e {
  lhs = f8e4m3fn[3,128,128] parameter(0)
  rhs = f8e4m3fn[3,128,128] parameter(1)
  lhs_scale = f8e8m0fnu[3,128,4] parameter(2)
  rhs_scale = f8e8m0fnu[3,128,4] parameter(3)
  scaled_dot = bf16[3,128,128] scaled-dot(lhs, rhs, lhs_scale, rhs_scale),
    lhs_batch_dims={0},
    rhs_batch_dims={0},
    lhs_contracting_dims={2},
    rhs_contracting_dims={2}
  global_scaler = bf16[] constant(1.42)
  global_scaler_broadcasted = bf16[3,128,128] broadcast(global_scaler),
      dimensions={}
  ROOT _ = bf16[3,128,128] multiply(scaled_dot, global_scaler_broadcasted)
}
  )hlo";

  ASSERT_OK_AND_ASSIGN(auto optimized_module,
                       GetOptimizedModule(kHloTextTemplate));
  constexpr absl::string_view kExpectedOptimizedHLO = R"(
    CHECK: %[[fusion_name:.*]] (parameter
    CHECK: %[[scaled_dot:.*]] = bf16[3,128,128]{2,1,0} scaled-dot
    CHECK: %[[global_scaler:.*]] = bf16[3,128,128]{2,1,0} broadcast
    CHECK: ROOT %{{.*}} = bf16[3,128,128]{2,1,0} multiply(%[[scaled_dot]], %[[global_scaler]])
    CHECK: ENTRY
    CHECK: ROOT {{.*}} fusion({{.*}}), kind=kCustom, calls=%[[fusion_name]]
  )";
  EXPECT_THAT(RunFileCheck(optimized_module->ToString(), kExpectedOptimizedHLO),
              true);

  HloComputation* scaled_dot_computation = GetFirstComputationWithInstruction(
      *optimized_module, HloOpcode::kScaledDot);
  constexpr absl::string_view kExpectedTritonIr = R"(
      CHECK: tt.dot_scaled
      CHECK: tensor<128x128xf8E4M3FN>, tensor<128x4xi8>
      CHECK: tensor<128x16xf8E4M3FN>, tensor<16x4xi8>
      CHECK: -> tensor<128x16xf32>
  )";
  EXPECT_THAT(CreateTritonIrAndFileCheckForDot(*scaled_dot_computation,
                                               kExpectedTritonIr),
              IsOk());

  EXPECT_TRUE(RunAndCompareNoHloPasses(
      std::move(optimized_module), ErrorSpec{/*aabs=*/1e-3, /*arel=*/1e-3}));
}

TEST_P(TritonScaledDotTest, ScaledDotWithE8m0Scale) {
  if (!GetCudaComputeCapability().IsAtLeastHopper()) {
    GTEST_SKIP()
        << "ScaledDot with Triton requires Hopper or newer architecture.";
  }
  constexpr absl::string_view kHloText = R"hlo(
HloModule E8m0ScaledDot

ENTRY e {
  lhs = f8e4m3fn[128,128] parameter(0)
  rhs = f8e4m3fn[128,256] parameter(2)
  lhs_scale = f8e8m0fnu[128,4] parameter(1)
  rhs_scale = f8e8m0fnu[4,256] parameter(3)
  ROOT _ = bf16[128,256]{1,0} scaled-dot(lhs, rhs, lhs_scale, rhs_scale),
    lhs_contracting_dims={1},
    rhs_contracting_dims={0}
}
)hlo";

  ASSERT_OK_AND_ASSIGN(auto optimized_module, GetOptimizedModule(kHloText));

  // Verify HLO fusion
  constexpr absl::string_view kExpectedOptimizedHLO = R"(
    CHECK: fusion
    CHECK: ROOT {{.*}} scaled-dot
    CHECK: ENTRY
    CHECK: __triton_nested_gemm_fusion
  )";
  EXPECT_THAT(RunFileCheck(optimized_module->ToString(), kExpectedOptimizedHLO),
              absl_testing::IsOkAndHolds(true));

  // Verify Triton IR
  HloComputation* scaled_dot_computation = GetFirstComputationWithInstruction(
      *optimized_module, HloOpcode::kScaledDot);
  constexpr absl::string_view kExpectedTritonIr = R"(
      CHECK: tt.dot_scaled
      CHECK: tensor<128x4xi8>
  )";
  EXPECT_THAT(CreateTritonIrAndFileCheckForDot(*scaled_dot_computation,
                                               kExpectedTritonIr),
              IsOk());

  // Execute on GPU hardware and compare with reference
  EXPECT_TRUE(RunAndCompareNoHloPasses(
      std::move(optimized_module), ErrorSpec{/*aabs=*/1e-3, /*arel=*/1e-3}));
}

TEST_P(TritonScaledDotTest, ScaledDotWithE4m3Scale) {
  if (!GetCudaComputeCapability().IsAtLeastHopper()) {
    GTEST_SKIP()
        << "ScaledDot with Triton requires Hopper or newer architecture.";
  }
  constexpr absl::string_view kHloText = R"hlo(
HloModule E4m3ScaledDot

ENTRY e {
  lhs = f8e4m3fn[128,128] parameter(0)
  rhs = f8e4m3fn[128,256] parameter(2)
  lhs_scale = f8e4m3fn[128,4] parameter(1)
  rhs_scale = f8e4m3fn[4,256] parameter(3)
  ROOT _ = bf16[128,256]{1,0} scaled-dot(lhs, rhs, lhs_scale, rhs_scale),
    lhs_contracting_dims={1},
    rhs_contracting_dims={0}
}
)hlo";

  ASSERT_OK_AND_ASSIGN(auto optimized_module, GetOptimizedModule(kHloText));

  // Verify Triton IR contains tt.dot_scaled with f8e4m3fn scale tensors
  HloComputation* scaled_dot_computation = GetFirstComputationWithInstruction(
      *optimized_module, HloOpcode::kScaledDot);
  constexpr absl::string_view kExpectedTritonIr = R"(
      CHECK: tt.dot_scaled
      CHECK: tensor<128x4xf8E4M3FN>
  )";
  EXPECT_THAT(CreateTritonIrAndFileCheckForDot(*scaled_dot_computation,
                                               kExpectedTritonIr),
              IsOk());

  // Execute on GPU hardware and compare with reference (Hopper only; Blackwell
  // TMEM scale expects E8M0 scale).
  if (!GetCudaComputeCapability().IsAtLeastBlackwell()) {
    EXPECT_TRUE(RunAndCompareNoHloPasses(
        std::move(optimized_module), ErrorSpec{/*aabs=*/1e-3, /*arel=*/1e-3}));
  }
}

TEST_P(TritonScaledDotTest, ScaledDotWithE5m2Scale) {
  if (!GetCudaComputeCapability().IsAtLeastHopper()) {
    GTEST_SKIP()
        << "ScaledDot with Triton requires Hopper or newer architecture.";
  }
  constexpr absl::string_view kHloText = R"hlo(
HloModule E5m2ScaledDot

ENTRY e {
  lhs = f8e4m3fn[128,128] parameter(0)
  rhs = f8e4m3fn[128,256] parameter(2)
  lhs_scale = f8e5m2[128,4] parameter(1)
  rhs_scale = f8e5m2[4,256] parameter(3)
  ROOT _ = bf16[128,256]{1,0} scaled-dot(lhs, rhs, lhs_scale, rhs_scale),
    lhs_contracting_dims={1},
    rhs_contracting_dims={0}
}
)hlo";

  ASSERT_OK_AND_ASSIGN(auto optimized_module, GetOptimizedModule(kHloText));

  // Verify Triton IR contains tt.dot_scaled with f8e5m2 scale tensors
  HloComputation* scaled_dot_computation = GetFirstComputationWithInstruction(
      *optimized_module, HloOpcode::kScaledDot);
  constexpr absl::string_view kExpectedTritonIr = R"(
      CHECK: tt.dot_scaled
      CHECK: tensor<128x4xf8E5M2>
  )";
  EXPECT_THAT(CreateTritonIrAndFileCheckForDot(*scaled_dot_computation,
                                               kExpectedTritonIr),
              IsOk());

  // Execute on GPU hardware and compare with reference (Hopper only; Blackwell
  // TMEM scale expects E8M0 scale).
  if (!GetCudaComputeCapability().IsAtLeastBlackwell()) {
    EXPECT_TRUE(RunAndCompareNoHloPasses(
        std::move(optimized_module), ErrorSpec{/*aabs=*/1e-3, /*arel=*/1e-3}));
  }
}

struct ScaledDotCoverageTestCase {
  PrimitiveType lhs_type;
  PrimitiveType rhs_type;
  PrimitiveType scale_type;
  int block_size;
};

class TritonScaledDotCoverageTest : public TritonScaledDotTestBase,
                                    public ::testing::WithParamInterface<
                                        std::tuple<ScaledDotCoverageTestCase,
                                                   /*lhs_k_minor=*/bool,
                                                   /*rhs_k_minor=*/bool,
                                                   /*tiling_enabled=*/bool>> {
 public:
  bool EnableTilingPropagation() const override {
    return std::get<3>(GetParam());
  }
};

std::vector<ScaledDotCoverageTestCase> GetCoverageTestCases() {
  std::vector<PrimitiveType> input_types = {F8E4M3FN, F8E5M2, F4E2M1FN};
  std::vector<ScaledDotCoverageTestCase> cases;

  // 1. All input combinations with E8M0 scale and block size 32 (9 cases)
  for (auto lhs : input_types) {
    for (auto rhs : input_types) {
      cases.push_back({lhs, rhs, F8E8M0FNU, 32});
    }
  }

  // 2. Block size 16 cases
  cases.push_back({F4E2M1FN, F4E2M1FN, F8E8M0FNU, 16});
  cases.push_back({F4E2M1FN, F4E2M1FN, F8E4M3FN, 16});

  return cases;
}

std::string ScaledDotCoverageTestParamToString(
    const ::testing::TestParamInfo<
        std::tuple<ScaledDotCoverageTestCase, bool, bool, bool>>& info) {
  const auto& [type_case, lhs_k_minor, rhs_k_minor, tiling_enabled] =
      info.param;
  return absl::StrCat(PrimitiveType_Name(type_case.lhs_type), "_",
                      PrimitiveType_Name(type_case.rhs_type), "_",
                      PrimitiveType_Name(type_case.scale_type), "_Block",
                      type_case.block_size, "_Lhs", lhs_k_minor, "_Rhs",
                      rhs_k_minor, "_",
                      TilingParametersToString(tiling_enabled));
}

TEST_P(TritonScaledDotCoverageTest, Executes) {
  const auto& [param, lhs_k_minor, rhs_k_minor, tiling_enabled] = GetParam();

  auto cc = GpuComputeCapability().cuda_compute_capability();
  std::string device_name = "Unknown";
  if (cc) {
    device_name = cc->IsAtLeastBlackwell()
                      ? "B200"
                      : (cc->IsAtLeastHopper() ? "H100" : "Pre-Hopper");
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
  std::string tiling_name = tiling_enabled ? "Symbolic" : "Experimental";

  LOG(ERROR) << "Report Device: " << device_name;
  LOG(ERROR) << "Report LHS: " << lhs_display;
  LOG(ERROR) << "Report RHS: " << rhs_display;
  LOG(ERROR) << "Report Scale: " << scale_name;
  LOG(ERROR) << "Report BlockSize: " << param.block_size;
  LOG(ERROR) << "Report Tiling: " << tiling_name;

  if (!cc || !cc->IsAtLeastHopper()) {
    GTEST_SKIP() << "Scaled dot isn't supported by Triton for pre-Hopper GPUs.";
  }

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
  ROOT %dummy = bf16[$output_shape]{1,0} constant({...})
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

  if (auto comp_module = ParseAndReturnUnverifiedModule(composite_hlo);
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

  // 2. Stage: Optimized HLO
  constexpr absl::string_view kHloTemplate = R"hlo(
HloModule m

ENTRY e {
  lhs = $lhs_type[$lhs_shape] parameter(0)
  rhs = $rhs_type[$rhs_shape] parameter(1)
  lhs_scale = $scale_type[$lhs_scale_shape] parameter(2)
  rhs_scale = $scale_type[$rhs_scale_shape] parameter(3)
  ROOT dot = bf16[$output_shape] scaled-dot(lhs, rhs, lhs_scale, rhs_scale),
    lhs_contracting_dims={$lhs_contracting_dim},
    rhs_contracting_dims={$rhs_contracting_dim}
}
)hlo";

  std::string hlo = absl::StrReplaceAll(
      kHloTemplate, {{"$lhs_type", lhs_name},
                     {"$rhs_type", rhs_name},
                     {"$scale_type", scale_name},
                     {"$lhs_shape", lhs_shape},
                     {"$rhs_shape", rhs_shape},
                     {"$lhs_scale_shape", lhs_scale_shape},
                     {"$rhs_scale_shape", rhs_scale_shape},
                     {"$output_shape", absl::StrCat(m, ",", n)},
                     {"$lhs_contracting_dim", lhs_k_minor ? "1" : "0"},
                     {"$rhs_contracting_dim", rhs_k_minor ? "1" : "0"}});

  if (param.scale_type == F8E8M0FNU && param.block_size == 16 &&
      GetCudaComputeCapability().IsAtLeastBlackwell() && !lhs_k_minor) {
#ifndef NDEBUG
    EXPECT_DEATH(
        { (void)GetOptimizedModule(hlo); },
        "MMAv5 with kind=mxf4nvf4 does not support transpose");
    return;
#endif
  }

  std::string optimized_hlo = "N/A";
  auto optimized_module_or = GetOptimizedModule(hlo);
  HloComputation* scaled_dot_computation = nullptr;
  if (optimized_module_or.ok()) {
    scaled_dot_computation = GetFirstComputationWithInstruction(
        **optimized_module_or, HloOpcode::kScaledDot);
    if (scaled_dot_computation) {
      optimized_hlo = "scaled-dot";
    } else if (GetFirstComputationWithInstruction(**optimized_module_or,
                                                  HloOpcode::kDot)) {
      optimized_hlo = "dot";
    }
  }
  LOG(ERROR) << "Report OptimizedHLO: " << optimized_hlo;

  // 3. Stage: Triton Dot Lowering
  std::string triton_dot = "N/A";
  if (scaled_dot_computation && scaled_dot_computation->FusionInstruction()) {
    mlir::MLIRContext mlir_context;
    BlockLevelParameters block_level_parameters;
    if (auto gpu_config = scaled_dot_computation->FusionInstruction()
                              ->backend_config<GpuBackendConfig>();
        gpu_config.ok() && gpu_config->has_fusion_backend_config() &&
        gpu_config->fusion_backend_config().has_block_level_fusion_config()) {
      block_level_parameters = BlockLevelParameters::FromBlockLevelFusionConfig(
          gpu_config->fusion_backend_config().block_level_fusion_config());
    }
    auto triton_source_or =
        CreateTritonModule("triton_fn",
                           *Cast<HloFusionInstruction>(
                               scaled_dot_computation->FusionInstruction()),
                           TestGpuDeviceInfo::RTXA6000DeviceInfo(*cc),
                           block_level_parameters, mlir_context);
    if (triton_source_or.ok()) {
      std::string mlir_str;
      llvm::raw_string_ostream os(mlir_str);
      triton_source_or->module()->print(os);
      if (absl::StrContains(mlir_str, "tt.dot_scaled")) {
        triton_dot = "tt.dot_scaled";
      } else if (absl::StrContains(mlir_str, "tt.dot")) {
        triton_dot = "tt.dot";
      } else if (absl::StrContains(mlir_str, "xtile.dot_scaled")) {
        triton_dot = "xtile.dot_scaled";
      }
    }
  }
  LOG(ERROR) << "Report TritonDot: " << triton_dot;

  // 4. Stage: PTX Generation
  std::vector<std::string> ptx_instructions;
  GpuCompiler* gpu_compiler = dynamic_cast<GpuCompiler*>(compiler());
  bool compilation_succeeded = false;
  if (gpu_compiler && optimized_module_or.ok()) {
    gpu_compiler->SetAsmHook([&](absl::string_view ptx) {
      std::string ptx_str(ptx);
      std::string line;
      size_t pos = 0;
      while ((pos = ptx_str.find('\n')) != std::string::npos) {
        line = ptx_str.substr(0, pos);
        size_t first_non_space = line.find_first_not_of(" \t");
        if (first_non_space != std::string::npos) {
          line = line.substr(first_non_space);
        }
        std::vector<std::string> tokens =
            absl::StrSplit(line, absl::ByAnyChar(" \t;"), absl::SkipEmpty());
        if (!tokens.empty()) {
          std::string inst = tokens[0];
          if (absl::StartsWith(inst, "@") && tokens.size() > 1) {
            inst = tokens[1];
          }
          if (absl::StrContains(inst, "mma") &&
              (absl::StartsWith(inst, "wgmma.mma") ||
               absl::StartsWith(inst, "tcgen05.mma") ||
               absl::StartsWith(inst, "mma.sync"))) {
            ptx_instructions.push_back(inst);
          }
        }
        ptx_str.erase(0, pos + 1);
      }
    });

    auto cloned_module = (*optimized_module_or)->Clone();
    auto executable_or = CompileToExecutable(std::move(cloned_module),
                                             /*run_optimization_passes=*/true);
    compilation_succeeded = executable_or.ok();
    gpu_compiler->RemoveAsmHook();
  }

  std::vector<std::string> unique_ptx;
  for (const auto& inst : ptx_instructions) {
    if (absl::c_find(unique_ptx, inst) == unique_ptx.end()) {
      unique_ptx.push_back(inst);
    }
  }
  std::string ptx_str =
      unique_ptx.empty() ? "N/A" : absl::StrJoin(unique_ptx, ", ");
  LOG(ERROR) << "Report PTX: " << ptx_str;

  // 5. Stage: Execution Status
  std::string status = "FAILURE";
  if (compilation_succeeded) {
    auto arguments_or = MakeScaledDotArguments((*optimized_module_or).get());
    if (arguments_or.ok()) {
      auto run_result =
          RunAndCompareNoHloPasses(std::move(*optimized_module_or),
                                   LiteralUtil::MakePointers(*arguments_or),
                                   ErrorSpec{/*aabs=*/1e-3, /*arel=*/1e-3});
      status = run_result ? "SUCCESS" : "FAILURE";
      if (!run_result) {
        if (absl::StrContains(run_result.message(), "ILLEGAL_INSTRUCTION")) {
          status = "ILLEGAL_INSTRUCTION";
        }
      }
    }
  }
  LOG(ERROR) << "Report Status: " << status;
  if (GetCudaComputeCapability().IsAtLeastBlackwell() &&
      status == "ILLEGAL_INSTRUCTION") {
    EXPECT_DEATH(
        { CHECK_NE(status, "ILLEGAL_INSTRUCTION"); }, "ILLEGAL_INSTRUCTION");
  }
}

INSTANTIATE_TEST_SUITE_P(
    TritonScaledDotCoverageTestSuite, TritonScaledDotCoverageTest,
    ::testing::Combine(::testing::ValuesIn(GetCoverageTestCases()),
                       ::testing::Bool(), ::testing::Bool(), ::testing::Bool()),
    ScaledDotCoverageTestParamToString);

}  // namespace
}  // namespace gpu
}  // namespace xla

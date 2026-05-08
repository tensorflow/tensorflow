/* Copyright 2019 The OpenXLA Authors.

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

#include "xla/backends/gpu/transforms/gemm_rewriter.h"

#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/container/flat_hash_map.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_replace.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/backends/gpu/transforms/gemm_rewriter_test_lib.h"
#include "xla/error_spec.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/testlib/filecheck.h"
#include "xla/hlo/testlib/pattern_matcher_gmock.h"
#include "xla/hlo/testlib/test.h"
#include "xla/service/hlo_module_config.h"
#include "xla/service/hlo_verifier.h"
#include "xla/service/pattern_matcher.h"
#include "xla/stream_executor/cuda/cuda_compute_capability.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/semantic_version.h"
#include "xla/tests/hlo_pjrt_interpreter_reference_mixin.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/xla.pb.h"

namespace xla {
namespace gpu {

namespace {

namespace m = ::xla::match;

using GemmRewriteTest = HloPjRtInterpreterReferenceMixin<GemmRewriteTestBase>;

TEST_F(GemmRewriteTest, CheckCustomCallTarget) {
  if (SkipGpuBlasLtTest()) {
    GTEST_SKIP() << "BlasLt is not supported on this GPU architecture";
  }

  const char* hlo_text = R"(
HloModule SimpleGemm

ENTRY AddDotsFunc {
  x = f32[2,3] parameter(0)
  y = f32[3,4] parameter(1)
  ROOT dot_a = f32[2,4] dot(x, y), lhs_contracting_dims={1}, rhs_contracting_dims={0}
}

)";
  DebugOptions debug_options = GetDebugOptionsForTest();
  if (debug_options.xla_gpu_enable_cublaslt()) {
    MatchOptimizedHlo(hlo_text,
                      R"(; CHECK: custom_call_target="__cublas$lt$matmul")");
  } else {
    MatchOptimizedHlo(hlo_text,
                      R"(; CHECK: custom_call_target="__cublas$gemm")");
  }
}

TEST_F(GemmRewriteTest, NormalizeMultipleBatchDimensions) {
  if (SkipGpuBlasLtTest()) {
    GTEST_SKIP() << "BlasLt is not supported on this GPU architecture";
  }

  const char* hlo_text = R"(
HloModule module

ENTRY test {
  lhs = f32[2,3,16,10240]{3,2,1,0} parameter(0)
  rhs = f32[2,3,10240,128]{3,2,1,0} parameter(1)
  ROOT dot = f32[2,3,16,128]{3,2,1,0} dot(lhs, rhs),
                lhs_batch_dims={0,1}, lhs_contracting_dims={3},
                rhs_batch_dims={0,1}, rhs_contracting_dims={2}
})";

  MatchOptimizedHlo(hlo_text,
                    R"(
CHECK-DAG: %[[LHS_BITCAST:[a-zA-Z0-9_.-]+]] = f32[6,16,10240]{{.*}} {{bitcast}}
CHECK-DAG: %[[RHS_BITCAST:[a-zA-Z0-9_.-]+]] = f32[6,10240,128]{{.*}} {{bitcast}}
CHECK: = (f32[6,16,128]{2,1,0}, s8[{{[0-9]+}}]{0}) custom-call(%[[LHS_BITCAST]], %[[RHS_BITCAST]]), custom_call_target="__cublas{{.*}}matmul"
CHECK: ROOT {{.*}} = f32[2,3,16,128]{3,2,1,0} {{bitcast}}
)");
}

TEST_F(GemmRewriteTest, CheckCustomCallHipblasLtBF16) {
  if (!IsRocm()) {
    GTEST_SKIP() << "Test is ROCm-specific: verifies that MI200 falls back to "
                    "legacy cuBLAS for BF16->F32 GEMMs (hipblasLt limitation), "
                    "while other ROCm architectures use hipblasLt";
  }
  DebugOptions debug_options = GetDebugOptionsForTest();
  HloModuleConfig config;
  config.set_debug_options(debug_options);
  config.mutable_debug_options().set_xla_gpu_enable_cublaslt(true);
  constexpr absl::string_view kHloText = R"(
ENTRY e {
  p0 = bf16[32,32] parameter(0)
  p1 = bf16[32,32] parameter(1)
  ROOT d = f32[32,32] dot(p0, p1),
    lhs_contracting_dims={1}, rhs_contracting_dims={0}
})";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> optimized_module,
                          GetOptimizedModule(kHloText, config));
  absl::StatusOr<bool> filecheck_result;
  if (IsRocm() && Capability().rocm_compute_capability()->gfx9_mi200()) {
    filecheck_result = RunFileCheck(optimized_module->ToString(), R"(
    ; CHECK-NOT: convert
    ; CHECK: __cublas$gemm
    )");
  } else {
    filecheck_result = RunFileCheck(optimized_module->ToString(), R"(
    ; CHECK-NOT: convert
    ; CHECK: __cublas$lt$matmul
    )");
  }
  TF_ASSERT_OK(filecheck_result.status());
  EXPECT_TRUE(filecheck_result.value());
}

TEST_F(GemmRewriteTest, TestBatchedAutotuning) {
  if (HasCudaComputeCapability(se::CudaComputeCapability::Ampere())) {
    GTEST_SKIP()
        << "There is no autotuning starting with the Nvidia Ampere generation";
  }

  const char* hlo_text = R"(
HloModule ComplexDotMultipleNonContracting

ENTRY test {
  %lhs = f32[7,17,10,13]{3,2,1,0} parameter(0)
  %rhs = f32[7,9,10,13,6]{4,3,2,1,0} parameter(1)
  ROOT %dot = f32[10,7,17,9,6]{4,3,2,1,0} dot(%lhs, %rhs), lhs_batch_dims={2,0}, rhs_batch_dims={2,0}, lhs_contracting_dims={3}, rhs_contracting_dims={3}
}

)";

  MatchOptimizedHlo(hlo_text,
                    R"(
; CHECK: selected_algorithm
      )");
}

TEST_F(GemmRewriteTest, SimpleRewriteDeterministic) {
  if (SkipGpuBlasLtTest()) {
    GTEST_SKIP() << "BlasLt is not supported on this GPU architecture";
  }

  const char* hlo_text = R"(
HloModule SimpleGemm

ENTRY AddDotsFunc {
  x = f32[128,128] parameter(0)
  y = f32[128,128] parameter(1)
  ROOT dot_a = f32[128,128] dot(x, y), lhs_contracting_dims={1}, rhs_contracting_dims={0}
}
)";

  auto get_config = [&]() {
    HloModuleConfig config;
    DebugOptions debug_options = GetDebugOptionsForTest();
    debug_options.set_xla_gpu_exclude_nondeterministic_ops(true);
    config.set_debug_options(debug_options);
    return config;
  };

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> optimized_module,
                          GetOptimizedModule(hlo_text, get_config()));

  absl::StatusOr<bool> filecheck_result =
      RunFileCheck(optimized_module->ToString(),
                   R"(
; CHECK:    custom_call_target="__cublas${{(lt\$matmul|gemm)}}"
    )");
  ASSERT_OK(filecheck_result.status());
  EXPECT_TRUE(filecheck_result.value());
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_text, get_config()));
  EXPECT_TRUE(RunAndCompare(std::move(module), ErrorSpec{1e-3, 1e-3}));
}

TEST_F(GemmRewriteTest, BF16GemmCodeGen) {
  const char* hlo_text = R"(
HloModule bf16codegendgemm

ENTRY bf16gemm {
  %parameter.1 = bf16[3]{0} parameter(0)
  %parameter.2 = bf16[3]{0} parameter(1)
  ROOT %dot.3 = bf16[] dot(bf16[3]{0} %parameter.1, bf16[3]{0} %parameter.2), lhs_contracting_dims={0}, rhs_contracting_dims={0}, operand_precision={highest,highest}
}
  )";

  if (HasCudaComputeCapability(se::CudaComputeCapability::Hopper())) {
    // The Hopper optimized HLO has a BF16 multiply instruction since Hopper has
    // native BF16 multiply support.
    MatchOptimizedHlo(hlo_text, R"(
  ; CHECK:  [[P0:%[^ ]+]] = bf16[3]{0} parameter(0)
  ; CHECK:  [[P1:%[^ ]+]] = bf16[3]{0} parameter(1)
  ; CHECK:  [[INSTR_2:%[^ ]+]] = bf16[3]{0} multiply([[P0]], [[P1]])
  ; CHECK:  [[INSTR_3:%[^ ]+]] = f32[3]{0} convert([[INSTR_2]])
  ; CHECK:  [[INSTR_4:%[^ ]+]] = f32[] constant(0)
  ; CHECK:  [[INSTR_5:%[^ ]+]] = f32[] reduce([[INSTR_3]], [[INSTR_4]]), dimensions={0}, to_apply=[[INSTR_6:%[^ ]+]]
  ; CHECK:  ROOT [[INSTR_7:%[^ ]+]] = bf16[] convert([[INSTR_5]])
    )");
  } else {
    MatchOptimizedHlo(hlo_text, R"(
  ; CHECK:  [[P1:%[^ ]+]] = bf16[3]{0} parameter(1)
  ; CHECK:  [[INSTR_1:%[^ ]+]] = f32[3]{0} convert([[P1]])
  ; CHECK:  [[P0:%[^ ]+]] = bf16[3]{0} parameter(0)
  ; CHECK:  [[INSTR_3:%[^ ]+]] = f32[3]{0} convert([[P0]])
  ; CHECK:  [[INSTR_4:%[^ ]+]] = f32[3]{0} multiply([[INSTR_1]], [[INSTR_3]])
  ; CHECK:  [[INSTR_5:%[^ ]+]] = f32[] constant(0)
  ; CHECK:  [[INSTR_6:%[^ ]+]] = f32[] reduce([[INSTR_4]], [[INSTR_5]]), dimensions={0}, to_apply=[[INSTR_7:%[^ ]+]]
  ; CHECK:  ROOT [[INSTR_8:%[^ ]+]] = bf16[] convert([[INSTR_6]])
    )");
  }

  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{1e-4, 1e-4}));
}

TEST_F(GemmRewriteTest, BF16Transpose) {
  const char* hlo_text = R"(
HloModule broadcast

ENTRY broadcast {
  p = bf16[9] parameter(0)
  ROOT out = bf16[1,9] broadcast(p), dimensions={1}
}
)";

  MatchOptimizedHlo(hlo_text, R"(
; CHECK: bf16[1,9]{1,0} bitcast
; CHECK: bf16[1,9]{1,0} copy
)");

  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{1e-5, 1e-5}));
}

TEST_F(GemmRewriteTest, NoFuseBiasBroadcast) {
  const char* hlo = R"(

HloModule module

ENTRY main.10 {
  Arg_0.1 = f16[384,128]{1,0} parameter(0)
  Arg_1.2 = f16[128,256]{1,0} parameter(1)
  dot.4 = f16[384,256]{1,0} dot(Arg_0.1, Arg_1.2), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  Arg_2.3 = f16[256]{0} parameter(2)
  reshape.5 = f16[1,256]{1,0} reshape(Arg_2.3)
  broadcast.6 = f16[1,256]{1,0} broadcast(reshape.5), dimensions={0,1}
  reshape.7 = f16[256]{0} reshape(broadcast.6)
  broadcast.8 = f16[384,256]{1,0} broadcast(reshape.7), dimensions={1}
  ROOT add.9 = f16[384,256]{1,0} add(dot.4, broadcast.8)
})";

  MatchOptimizedHlo(hlo, R"(
// CHECK: "beta":0
  )");
}

TEST_F(GemmRewriteTest, ReduceOfBatchDot) {
  absl::string_view hlo_string =
      R"(
HloModule test

region_5.50 {
  Arg_0.51 = f32[] parameter(0)
  Arg_1.52 = f32[] parameter(1)
  ROOT add.53 = f32[] add(Arg_0.51, Arg_1.52)
}

ENTRY main {
  p0 = bf16[3,32,3,13]{3,2,1,0} parameter(0)
  p1 = bf16[3,32,3,64]{3,2,1,0} parameter(1)
  dot.95 = bf16[3,3,13,64]{3,2,1,0} dot(p0, p1), lhs_batch_dims={0,2}, lhs_contracting_dims={1}, rhs_batch_dims={0,2}, rhs_contracting_dims={1}, operand_precision={highest,highest}
  transpose.96 = bf16[3,64,3,13]{1,3,2,0} transpose(dot.95), dimensions={0,3,1,2}
  convert.101 = f32[3,64,3,13]{1,3,2,0} convert(transpose.96)
  constant.66 = f32[] constant(0.0)
  ROOT reduce.102 = f32[3,64,13]{2,1,0} reduce(convert.101, constant.66), dimensions={2}, to_apply=region_5.50
}
)";
  // Make sure the dot is lowered to a custom call. There is an algebraic
  // simplifier simplification which could turn the dot into a non-canonical dot
  // late in the pipeline, which will make it unsupported by the GemmRewriter.
  MatchOptimizedHlo(hlo_string, R"(
  // CHECK: custom_call_target="__cublas${{gemm|lt\$matmul}}"
  )");
}

TEST_F(GemmRewriteTest, DotWithBias) {
  const char* hlo = R"(
      HloModule m

      ENTRY main {
        p0 = f32[1024,1024] parameter(0)
        p1 = f32[1024,1024] parameter(1)
        p2 = f32[1024,1024] parameter(2)
        p3 = f32[1024,1024] parameter(3)
        dot0 = f32[1024,1024] dot(p0, p1),
            lhs_contracting_dims={1}, rhs_contracting_dims={0}
        dot1 = f32[1024,1024] dot(p2, p3),
            lhs_contracting_dims={1}, rhs_contracting_dims={0}
        ROOT root = f32[1024,1024] add(dot0, dot1)
  })";

  const char* expected = R"()
    // CHECK: %[[P0:.*]] = f32[1024,1024]{1,0} parameter(0)
    // CHECK: %[[P1:.*]] = f32[1024,1024]{1,0} parameter(1)
    // CHECK: %[[P2:.*]] = f32[1024,1024]{1,0} parameter(2)
    // CHECK: %[[P3:.*]] = f32[1024,1024]{1,0} parameter(3)
    // CHECK: %[[TUPLE0:.*]] = (f32[1024,1024]{1,0}, s8[4194304]{0}) custom-call(%[[P2]], %[[P3]])
    // CHECK: %[[S0:.*]] = f32[1024,1024]{1,0} get-tuple-element(%[[TUPLE0]]), index=0
    // CHECK: %[[TUPLE1:.*]] = (f32[1024,1024]{1,0}, s8[4194304]{0}) custom-call(%[[P0]], %[[P1]], %[[S0]])
    // CHECK: ROOT %[[S1:.*]] = f32[1024,1024]{1,0} get-tuple-element(%[[TUPLE1]]), index=0
  })";

  RunAndFilecheckHloRewrite(
      hlo,
      GemmRewriter(
          se::CudaComputeCapability{},
          /*toolkit_version=*/stream_executor::SemanticVersion{0, 0, 0},
          GemmRewriterOptions{GemmRewriterOptions::DType::kNonFp8Only}),
      expected);
}

TEST_F(GemmRewriteTest, DotWithoutBias) {
  const char* hlo = R"(
      HloModule m

      ENTRY main {
        p0 = f32[1024,1024] parameter(0)
        p1 = f32[1024,1024] parameter(1)
        p2 = f32[1024,1024] parameter(2)
        p3 = f32[1024,1024] parameter(3)
        dot0 = f32[1024,1024] dot(p0, p1),
            lhs_contracting_dims={1}, rhs_contracting_dims={0}
        dot1 = f32[1024,1024] dot(p2, p3),
            lhs_contracting_dims={1}, rhs_contracting_dims={0}
        ROOT root = f32[1024,1024] add(dot0, dot1)
  })";

  const char* expected = R"()
    // CHECK: %[[P0:.*]] = f32[1024,1024]{1,0} parameter(0)
    // CHECK: %[[P1:.*]] = f32[1024,1024]{1,0} parameter(1)
    // CHECK: %[[TUPLE0:.*]] = (f32[1024,1024]{1,0}, s8[4194304]{0}) custom-call(%[[P0]], %[[P1]])
    // CHECK: %[[S0:.*]] = f32[1024,1024]{1,0} get-tuple-element(%[[TUPLE0]]), index=0
    // CHECK: %[[P2:.*]] = f32[1024,1024]{1,0} parameter(2)
    // CHECK: %[[P3:.*]] = f32[1024,1024]{1,0} parameter(3)
    // CHECK: %[[TUPLE1:.*]] = (f32[1024,1024]{1,0}, s8[4194304]{0}) custom-call(%[[P2]], %[[P3]])
    // CHECK: %[[S1:.*]] = f32[1024,1024]{1,0} get-tuple-element(%[[TUPLE1]]), index=0
    // CHECK: ROOT %[[S2:.*]] = f32[1024,1024]{1,0} add(%[[S0]], %[[S1]])
  })";

  RunAndFilecheckHloRewrite(
      hlo,
      GemmRewriter(
          se::CudaComputeCapability{},
          /*toolkit_version=*/stream_executor::SemanticVersion{0, 0, 0},
          GemmRewriterOptions{GemmRewriterOptions::DType::kNonFp8Only,
                              GemmRewriterOptions::BiasMode::kNoBias}),
      expected);
}

using ParameterizedGemmRewriteTest =
    HloPjRtInterpreterReferenceMixin<ParameterizedGemmRewriteTestBase>;

// A test fixture class for tests which are specific to cublasLt
class CublasLtGemmRewriteTest
    : public HloPjRtInterpreterReferenceMixin<GemmRewriteTestBase> {
 public:
  DebugOptions GetDebugOptionsForTest() const override {
    DebugOptions debug_options = GemmRewriteTestBase::GetDebugOptionsForTest();
    debug_options.set_xla_gpu_enable_cublaslt(true);
    debug_options.set_xla_gpu_enable_triton_gemm(false);
    return debug_options;
  }

 protected:
  void SetUp() override {
    if (SkipGpuBlasLtTest()) {
      GTEST_SKIP() << "BlasLt is not supported on this GPU architecture";
    }
  }
};

TEST_F(CublasLtGemmRewriteTest, AlphaBetaRewrite) {
  const char* hlo_text = R"(
HloModule NonZeroAlphaBeta

ENTRY AddDotsFunc {
  x = f32[2,2] parameter(0)
  y = f32[2,2] parameter(1)
  bias = f32[2,2] parameter(2)
  k = f32[] constant(3.0)
  k_broadcast = f32[2, 2] broadcast(k), dimensions={}
  dot_a = f32[2,2] dot(x, y), lhs_contracting_dims={1}, rhs_contracting_dims={0}, operand_precision={highest,highest}
  dot_a_multiplied = f32[2, 2] multiply(dot_a, k_broadcast)
  ROOT out = f32[2,2] add(dot_a_multiplied, bias)
}

)";

  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{1e-5, 1e-5}));
  MatchOptimizedHlo(hlo_text,
                    R"(
; CHECK-LABEL: ENTRY %{{.*}} ({{.*}}: f32[2,2], {{.*}}: f32[2,2], {{.*}}: f32[2,2]) -> f32[2,2] {
; CHECK-DAG:     [[X:%[^ ]+]] = f32[2,2]{1,0} parameter(0)
; CHECK-DAG:     [[Y:%[^ ]+]] = f32[2,2]{1,0} parameter(1)
; CHECK-DAG:     [[BIAS:%[^ ]+]] = f32[2,2]{1,0} parameter(2)
; CHECK-NEXT:    [[GEMM:%[^ ]+]] = (f32[2,2]{1,0}, s8[{{[0-9]+}}]{0}) custom-call([[X]], [[Y]], [[BIAS]]),
; CHECK:           custom_call_target="__cublas$lt$matmul",
; CHECK:           backend_config={
; CHECK-DAG:         "alpha_real":3
; CHECK-DAG:         "alpha_imag":0
; CHECK-DAG:         "beta":1
; CHECK-DAG:         "dot_dimension_numbers":{
; CHECK-DAG:           "lhs_contracting_dimensions":["1"]
; CHECK-DAG:           "rhs_contracting_dimensions":["0"]
; CHECK-DAG:           "lhs_batch_dimensions":[]
; CHECK-DAG:           "rhs_batch_dimensions":[]
; CHECK-DAG:         }
; CHECK-DAG:         "precision_config":{
; CHECK-DAG:           "operand_precision":["HIGHEST","HIGHEST"]
; CHECK-DAG:         }
; CHECK-DAG:         "epilogue":"DEFAULT"
; CHECK:           }
; CHECK-NEXT  ROOT [[OUT:%[^ ]+]] = f32[2,2]{1,0} get-tuple-element(%cublas-lt-matmul.2.0), index=0
)");
}

TEST_F(CublasLtGemmRewriteTest, BF16GemmWithBias) {
  const char* hlo_text = R"(
HloModule test

ENTRY BF16GemmWithBias {
  x = bf16[8,8]{1,0} parameter(0)
  y = bf16[8,8]{1,0} parameter(1)
  dot.5 = bf16[8,8]{1,0} dot(x, y), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  bias = bf16[8,8]{1,0} parameter(2)
  ROOT add.6 = bf16[8,8]{1,0} add(dot.5, bias)
}
  )";

  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{1e-3, 1e-3}));

  if (IsCuda() &&
      !HasCudaComputeCapability(se::CudaComputeCapability::Ampere())) {
    GTEST_SKIP() << "Pre-Ampere casts up bf16 to fp32";
  }

  MatchOptimizedHlo(hlo_text,
                    R"(
; CHECK-LABEL: ENTRY %{{.*}} ({{.*}}: bf16[8,8], {{.*}}: bf16[8,8], {{.*}}: bf16[8,8]) -> bf16[8,8] {
; CHECK-DAG:    [[X:%[^ ]+]] = bf16[8,8]{1,0} parameter(0)
; CHECK-DAG:    [[Y:%[^ ]+]] = bf16[8,8]{1,0} parameter(1)
; CHECK-DAG:    [[BIAS:%[^ ]+]] = bf16[8,8]{1,0} parameter(2)
; CHECK-NEXT:   [[GEMM:%[^ ]+]] = (bf16[8,8]{1,0}, s8[{{[0-9]+}}]{0}) custom-call([[X]], [[Y]], [[BIAS]]),
; CHECK:           custom_call_target="__cublas$lt$matmul",
; CHECK:           backend_config={
; CHECK-DAG:         "alpha_real":1
; CHECK-DAG:         "alpha_imag":0
; CHECK-DAG:         "beta":1
; CHECK-DAG:         "dot_dimension_numbers":{
; CHECK-DAG:           "lhs_contracting_dimensions":["1"]
; CHECK-DAG:           "rhs_contracting_dimensions":["0"]
; CHECK-DAG:           "lhs_batch_dimensions":[]
; CHECK-DAG:           "rhs_batch_dimensions":[]
; CHECK-DAG:         }
; CHECK-DAG:         "precision_config":{
; CHECK-DAG:           "operand_precision":["DEFAULT","DEFAULT"]
; CHECK-DAG:         }
; CHECK-DAG:         "epilogue":"DEFAULT"
; CHECK:           }
)");
}

TEST_F(CublasLtGemmRewriteTest, MatrixBias) {
  const char* hlo_text = R"(
HloModule test

ENTRY test {
  x = f32[2,3] parameter(0)
  y = f32[3,4] parameter(1)
  z = f32[2,4] parameter(2)
  dot_a = f32[2,4] dot(x, y), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  ROOT out = f32[2,4] add(dot_a, z)
}

)";

  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{1e-5, 1e-5}));
  MatchOptimizedHlo(hlo_text,
                    R"(
; CHECK-LABEL: ENTRY %{{.*}} ({{.*}}: f32[2,3], {{.*}}: f32[3,4], {{.*}}: f32[2,4]) -> f32[2,4] {
; CHECK-DAG:     [[P0:%[^ ]+]] = f32[2,3]{1,0} parameter(0)
; CHECK-DAG:     [[P1:%[^ ]+]] = f32[3,4]{1,0} parameter(1)
; CHECK-DAG:     [[P2:%[^ ]+]] = f32[2,4]{1,0} parameter(2)
; CHECK:         [[GEMM:%[^ ]+]] = (f32[2,4]{1,0}, s8[{{[0-9]+}}]{0}) custom-call([[P0]], [[P1]], [[P2]]),
; CHECK:           custom_call_target="__cublas$lt$matmul",
; CHECK:           backend_config={
; CHECK-DAG:         "alpha_real":1
; CHECK-DAG:         "alpha_imag":0
; CHECK-DAG:         "beta":1
; CHECK-DAG:         "dot_dimension_numbers":{
; CHECK-DAG:           "lhs_contracting_dimensions":["1"]
; CHECK-DAG:           "rhs_contracting_dimensions":["0"]
; CHECK-DAG:           "lhs_batch_dimensions":[]
; CHECK-DAG:           "rhs_batch_dimensions":[]
; CHECK-DAG:         }
; CHECK-DAG:         "precision_config":{
; CHECK-DAG:           "operand_precision":["DEFAULT","DEFAULT"]
; CHECK-DAG:         }
; CHECK-DAG:         "epilogue":"DEFAULT"
; CHECK:           }
)");
}

TEST_F(CublasLtGemmRewriteTest, MatrixBiasWhereBiasIsNotAParameter) {
  const char* hlo_text = R"(
HloModule test

ENTRY test {
  w = f32[2,3] parameter(0)
  x = f32[3,4] parameter(1)
  first_dot = f32[2,4] dot(w, x), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  y = f32[2,3] parameter(2)
  z = f32[3,4] parameter(3)
  second_dot = f32[2,4] dot(y, z), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  ROOT out = f32[2,4] add(second_dot, first_dot)
}

)";

  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{1e-5, 1e-5}));
  MatchOptimizedHlo(hlo_text,
                    R"(
; CHECK-LABEL: ENTRY %{{.*}} ({{.*}}: f32[2,3], {{.*}}: f32[3,4], {{.*}}: f32[2,3], {{.*}}: f32[3,4]) -> f32[2,4] {
; CHECK-DAG:     [[P0:%[^ ]+]] = f32[2,3]{1,0} parameter(0)
; CHECK-DAG:     [[P1:%[^ ]+]] = f32[3,4]{1,0} parameter(1)
; CHECK-DAG:     [[P2:%[^ ]+]] = f32[2,3]{1,0} parameter(2)
; CHECK-DAG:     [[P3:%[^ ]+]] = f32[3,4]{1,0} parameter(3)
; CHECK-NEXT:    [[FIRST_GEMM_TUPLE:%[^ ]+]] = (f32[2,4]{1,0}, s8[{{[0-9]+}}]{0}) custom-call([[P0]], [[P1]]),
; CHECK:           custom_call_target="__cublas$lt$matmul",
; CHECK:           backend_config={
; CHECK-DAG:         "alpha_real":1
; CHECK-DAG:         "alpha_imag":0
; CHECK-DAG:         "beta":0
; CHECK-DAG:         "dot_dimension_numbers":{
; CHECK-DAG:           "lhs_contracting_dimensions":["1"]
; CHECK-DAG:           "rhs_contracting_dimensions":["0"]
; CHECK-DAG:           "lhs_batch_dimensions":[]
; CHECK-DAG:           "rhs_batch_dimensions":[]
; CHECK-DAG:         }
; CHECK-DAG:         "precision_config":{
; CHECK-DAG:           "operand_precision":["DEFAULT","DEFAULT"]
; CHECK-DAG:         }
; CHECK-DAG:         "epilogue":"DEFAULT"
; CHECK:           }
; CHECK:         [[FIRST_GEMM:%[^ ]+]] = f32[2,4]{1,0} get-tuple-element([[FIRST_GEMM_TUPLE]]), index=0
; CHECK-NEXT:    [[SECOND_GEMM:%[^ ]+]] = (f32[2,4]{1,0}, s8[{{[0-9]+}}]{0}) custom-call([[P2]], [[P3]], [[FIRST_GEMM]]),
; CHECK:           custom_call_target="__cublas$lt$matmul",
; CHECK:           output_to_operand_aliasing={
; CHECK:              {0}: (2, {})
; CHECK:           }
; CHECK:           backend_config={
; CHECK-DAG:         "alpha_real":1
; CHECK-DAG:         "alpha_imag":0
; CHECK-DAG:         "beta":1
; CHECK-DAG:         "dot_dimension_numbers":{
; CHECK-DAG:           "lhs_contracting_dimensions":["1"]
; CHECK-DAG:           "rhs_contracting_dimensions":["0"]
; CHECK-DAG:           "lhs_batch_dimensions":[]
; CHECK-DAG:           "rhs_batch_dimensions":[]
; CHECK-DAG:         }
; CHECK-DAG:         "precision_config":{
; CHECK-DAG:           "operand_precision":["DEFAULT","DEFAULT"]
; CHECK-DAG:         }
; CHECK-DAG:         "epilogue":"DEFAULT"
; CHECK:           }
)");
}

TEST_F(CublasLtGemmRewriteTest, VectorBias) {
  const char* hlo_text = R"(
HloModule test

ENTRY test {
  x = f32[2,3] parameter(0)
  y = f32[3,4] parameter(1)
  z = f32[4] parameter(2)
  dot_a = f32[2,4] dot(x, y), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  z_bcast = f32[2,4] broadcast(z), dimensions={1}
  ROOT out = f32[2,4] add(dot_a, z_bcast)
}

)";

  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{1e-5, 1e-5}));
  MatchOptimizedHlo(hlo_text,
                    R"(
; CHECK-LABEL: ENTRY %{{.*}} ({{.*}}: f32[2,3], {{.*}}: f32[3,4], {{.*}}: f32[4]) -> f32[2,4] {
; CHECK-DAG:     [[P0:%[^ ]+]] = f32[2,3]{1,0} parameter(0)
; CHECK-DAG:     [[P1:%[^ ]+]] = f32[3,4]{1,0} parameter(1)
; CHECK-DAG:     [[P2:%[^ ]+]] = f32[4]{0} parameter(2)
; CHECK:         [[OUT:%[^ ]+]] = (f32[2,4]{1,0}, s8[{{[0-9]+}}]{0}) custom-call([[P0]], [[P1]], [[P2]]),
; CHECK:           custom_call_target="__cublas$lt$matmul",
; CHECK:           backend_config={
; CHECK-DAG:         "alpha_real":1
; CHECK-DAG:         "alpha_imag":0
; CHECK-DAG:         "beta":0
; CHECK-DAG:         "dot_dimension_numbers":{
; CHECK-DAG:           "lhs_contracting_dimensions":["1"]
; CHECK-DAG:           "rhs_contracting_dimensions":["0"]
; CHECK-DAG:           "lhs_batch_dimensions":[]
; CHECK-DAG:           "rhs_batch_dimensions":[]
; CHECK-DAG:         }
; CHECK-DAG:         "precision_config":{
; CHECK-DAG:           "operand_precision":["DEFAULT","DEFAULT"]
; CHECK-DAG:         }
; CHECK-DAG:         "epilogue":"BIAS"
; CHECK:           }
)");
}

TEST_F(CublasLtGemmRewriteTest, BatchedVectorBias) {
  const char* hlo_text = R"(
HloModule test

ENTRY test {
  x = f32[2,3,4] parameter(0)
  y = f32[4,5,6] parameter(1)
  z = f32[3,5,6] parameter(2)
  dot_a = f32[2,3,5,6] dot(x, y), lhs_contracting_dims={2}, rhs_contracting_dims={0}, operand_precision={highest,highest}
  z_bcast = f32[2,3,5,6] broadcast(z), dimensions={1,2,3}
  ROOT out = f32[2,3,5,6] add(dot_a, z_bcast)
}

)";

  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{1e-5, 1e-5}));
  MatchOptimizedHlo(hlo_text,
                    R"(

; CHECK-LABEL: ENTRY %{{.*}} ({{.*}}: f32[2,3,4], {{.*}}: f32[4,5,6], {{.*}}: f32[3,5,6]) -> f32[2,3,5,6] {
; CHECK:         [[MATMUL_TUPLE:%[^ ]+]] = (f32[6,30]{1,0}, s8[{{[0-9]+}}]{0}) custom-call(
; CHECK:           custom_call_target="__cublas$lt$matmul",
; CHECK:           output_to_operand_aliasing={
; CHECK:              {0}: (2, {})
; CHECK:           }
; CHECK:           backend_config={
; CHECK-DAG:         "alpha_real":1
; CHECK-DAG:         "alpha_imag":0
; CHECK-DAG:         "beta":1
; CHECK-DAG:         "dot_dimension_numbers":{
; CHECK-DAG:           "lhs_contracting_dimensions":["1"]
; CHECK-DAG:           "rhs_contracting_dimensions":["0"]
; CHECK-DAG:           "lhs_batch_dimensions":[]
; CHECK-DAG:           "rhs_batch_dimensions":[]
; CHECK-DAG:         }
; CHECK-DAG:         "precision_config":{
; CHECK-DAG:           "operand_precision":["HIGHEST","HIGHEST"]
; CHECK-DAG:         }
; CHECK-DAG:         "epilogue":"DEFAULT"
; CHECK:           }
; CHECK-NEXT:   [[MATMUL:%[^ ]+]] = f32[6,30]{1,0} get-tuple-element([[MATMUL_TUPLE]]), index=0
; CHECK-NEXT:    ROOT [[OUT:%[^ ]+]] = f32[2,3,5,6]{3,2,1,0} bitcast([[MATMUL]])
      )");
}

TEST_F(CublasLtGemmRewriteTest, BatchedSharedVectorBias) {
  const char* hlo_text = R"(
HloModule test

ENTRY test {
  x = f32[2,3,4] parameter(0)
  y = f32[4,5,6] parameter(1)
  z = f32[6] parameter(2)
  dot_a = f32[2,3,5,6] dot(x, y), lhs_contracting_dims={2}, rhs_contracting_dims={0}, operand_precision={highest,highest}
  z_bcast = f32[2,3,5,6] broadcast(z), dimensions={3}
  ROOT out = f32[2,3,5,6] add(dot_a, z_bcast)
}

)";

  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{1e-5, 1e-5}));
  MatchOptimizedHlo(hlo_text,
                    R"(

; CHECK-LABEL: ENTRY %{{.*}} ({{.*}}: f32[2,3,4], {{.*}}: f32[4,5,6], {{.*}}: f32[6]) -> f32[2,3,5,6] {
; CHECK:         [[MATMUL_TUPLE:%[^ ]+]] = (f32[6,30]{1,0}, s8[{{[0-9]+}}]{0}) custom-call(
; CHECK:           custom_call_target="__cublas$lt$matmul",
; CHECK:           output_to_operand_aliasing={
; CHECK:              {0}: (2, {})
; CHECK:           }
; CHECK:           backend_config={
; CHECK-DAG:         "alpha_real":1
; CHECK-DAG:         "alpha_imag":0
; CHECK-DAG:         "beta":1
; CHECK-DAG:         "dot_dimension_numbers":{
; CHECK-DAG:           "lhs_contracting_dimensions":["1"]
; CHECK-DAG:           "rhs_contracting_dimensions":["0"]
; CHECK-DAG:           "lhs_batch_dimensions":[]
; CHECK-DAG:           "rhs_batch_dimensions":[]
; CHECK-DAG:         }
; CHECK-DAG:         "precision_config":{
; CHECK-DAG:           "operand_precision":["HIGHEST","HIGHEST"]
; CHECK-DAG:         }
; CHECK-DAG:         "epilogue":"DEFAULT"
; CHECK:           }
; CHECK:         [[MATMUL:%[^ ]+]] = f32[6,30]{1,0} get-tuple-element([[MATMUL_TUPLE]]), index=0
; CHECK-NEXT:    ROOT [[OUT:%[^ ]+]] = f32[2,3,5,6]{3,2,1,0} bitcast([[MATMUL]])
      )");
}

TEST_F(CublasLtGemmRewriteTest, VectorBiasIncorrectAxisFusedAsMatrix) {
  const char* hlo_text = R"(
HloModule test

ENTRY test {
  x = f32[2,3] parameter(0)
  y = f32[3,4] parameter(1)
  z = f32[2] parameter(2)
  dot_a = f32[2,4] dot(x, y), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  z_bcast = f32[2,4] broadcast(z), dimensions={0}
  add = f32[2,4] add(dot_a, z_bcast)
  ROOT out = f32[4,2] transpose(add), dimensions={1,0}
}

)";

  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{1e-5, 1e-5}));
  MatchOptimizedHlo(hlo_text,
                    R"(
; CHECK-LABEL: ENTRY %{{.*}} ({{.*}}: f32[2,3], {{.*}}: f32[3,4], {{.*}}: f32[2]) -> f32[4,2] {
; CHECK-DAG:     [[P0:%[^ ]+]] = f32[2,3]{1,0} parameter(0)
; CHECK-DAG:     [[P1:%[^ ]+]] = f32[3,4]{1,0} parameter(1)
; CHECK-DAG:     [[P2:%[^ ]+]] = f32[2]{0} parameter(2)
; CHECK:         [[MATMUL_TUPLE:%[^ ]+]] = (f32[2,4]{0,1}, s8[{{[0-9]+}}]{0}) custom-call([[P0]], [[P1]], [[P2]]),
; CHECK:           custom_call_target="__cublas$lt$matmul",
; CHECK:           backend_config={
; CHECK-DAG:         "alpha_real":1
; CHECK-DAG:         "alpha_imag":0
; CHECK-DAG:         "beta":0
; CHECK-DAG:         "dot_dimension_numbers":{
; CHECK-DAG:           "lhs_contracting_dimensions":["1"]
; CHECK-DAG:           "rhs_contracting_dimensions":["0"]
; CHECK-DAG:           "lhs_batch_dimensions":[]
; CHECK-DAG:           "rhs_batch_dimensions":[]
; CHECK-DAG:         }
; CHECK-DAG:         "precision_config":{
; CHECK-DAG:           "operand_precision":["DEFAULT","DEFAULT"]
; CHECK-DAG:         }
; CHECK-DAG:         "epilogue":"BIAS"
; CHECK:           }
; CHECK-NEXT:    [[MATMUL:%[^ ]+]] = f32[2,4]{0,1} get-tuple-element([[MATMUL_TUPLE]]), index=0
; CHECK-NEXT:    ROOT [[OUT:%[^ ]+]] = f32[4,2]{1,0} bitcast([[MATMUL]])
)");
}

TEST_F(CublasLtGemmRewriteTest, VectorBiasSliced) {
  const char* hlo_text = R"(
HloModule test

ENTRY test {
  x = f32[4,3] parameter(0)
  y = f32[3,4] parameter(1)
  z = f32[3] parameter(2)
  dot_a = f32[4,4] dot(x, y), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  slice_a = f32[2,3] slice(dot_a), slice={[0:2], [0:3]}
  z_bcast = f32[2,3] broadcast(z), dimensions={1}
  ROOT out = f32[2,3] add(slice_a, z_bcast)
}

)";

  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{1e-5, 1e-5}));
  MatchOptimizedHlo(hlo_text,
                    R"(

; CHECK-LABEL: ENTRY %{{.*}} ({{.*}}: f32[4,3], {{.*}}: f32[3,4], {{.*}}: f32[3]) -> f32[2,3] {
; CHECK-DAG:     [[P0:%[^ ]+]] = f32[4,3]{1,0} parameter(0)
; CHECK-DAG:     [[P1:%[^ ]+]] = f32[3,4]{1,0} parameter(1)
; CHECK-DAG:     [[P2:%[^ ]+]] = f32[3]{0} parameter(2)
; CHECK:         [[MATMUL:%[^ ]+]] = (f32[4,4]{1,0}, s8[{{[0-9]+}}]{0}) custom-call([[P0]], [[P1]], [[P2]]),
; CHECK:           custom_call_target="__cublas$lt$matmul",
; CHECK:           backend_config={
; CHECK-DAG:         "alpha_real":1
; CHECK-DAG:         "alpha_imag":0
; CHECK-DAG:         "beta":0
; CHECK-DAG:         "dot_dimension_numbers":{
; CHECK-DAG:           "lhs_contracting_dimensions":["1"]
; CHECK-DAG:           "rhs_contracting_dimensions":["0"]
; CHECK-DAG:           "lhs_batch_dimensions":[]
; CHECK-DAG:           "rhs_batch_dimensions":[]
; CHECK-DAG:         }
; CHECK-DAG:         "precision_config":{
; CHECK-DAG:           "operand_precision":["DEFAULT","DEFAULT"]
; CHECK-DAG:         }
; CHECK-DAG:         "epilogue":"BIAS"
; CHECK:           }
; CHECK-NEXT:    [[GETTUPLE:%[^ ]+]] = f32[4,4]{1,0} get-tuple-element([[MATMUL]]), index=0
; CHECK:    ROOT [[OUT:%[^ ]+]] = f32[2,3]{1,0} fusion([[GETTUPLE]]), kind=kLoop
      )");
}

TEST_F(CublasLtGemmRewriteTest, VectorBiasSlicedMultipleUsers) {
  const char* hlo_text = R"(
HloModule test

ENTRY test {
  x = f32[2,3] parameter(0)
  y = f32[3,4] parameter(1)
  z = f32[2] parameter(2)
  c = f32[] constant(5)
  dot_a = f32[2,4] dot(x, y), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  slice_a = f32[2,2] slice(dot_a), slice={[0:2], [0:2]}
  z_bcast = f32[2,2] broadcast(z), dimensions={1}
  add_a = f32[2,2] add(slice_a, z_bcast)
  c_bcast = f32[2,2] broadcast(c), dimensions={}
  dot_b = f32[2,2] dot(slice_a, c_bcast), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  ROOT out = f32[2,2] dot(add_a, dot_b), lhs_contracting_dims={1}, rhs_contracting_dims={0}
}

)";

  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{1e-5, 1e-5}));
  MatchOptimizedHlo(hlo_text,
                    R"(
; CHECK-LABEL: ENTRY %{{.*}} ({{.*}}: f32[2,3], {{.*}}: f32[3,4], {{.*}}: f32[2]) -> f32[2,2] {
; CHECK-DAG:     [[P0:%[^ ]+]] = f32[2,3]{1,0} parameter(0)
; CHECK-DAG:     [[P1:%[^ ]+]] = f32[3,4]{1,0} parameter(1)
; CHECK-DAG:     [[P2:%[^ ]+]] = f32[2]{0} parameter(2)
; CHECK:         [[MATMUL0_TUPLE:%[^ ]+]] = (f32[2,4]{1,0}, s8[{{[0-9]+}}]{0}) custom-call([[P0]], [[P1]]),
; CHECK:           custom_call_target="__cublas$lt$matmul",
; CHECK:           backend_config={
; CHECK-DAG:         "alpha_real":1
; CHECK-DAG:         "alpha_imag":0
; CHECK-DAG:         "beta":0
; CHECK-DAG:         "dot_dimension_numbers":{
; CHECK-DAG:           "lhs_contracting_dimensions":["1"]
; CHECK-DAG:           "rhs_contracting_dimensions":["0"]
; CHECK-DAG:           "lhs_batch_dimensions":[]
; CHECK-DAG:           "rhs_batch_dimensions":[]
; CHECK-DAG:         }
; CHECK-DAG:         "precision_config":{
; CHECK-DAG:           "operand_precision":["DEFAULT","DEFAULT"]
; CHECK-DAG:         }
; CHECK-DAG:         "epilogue":"DEFAULT"
; CHECK:           }
; CHECK:         [[MATMUL1_TUPLE:%[^ ]+]] = (f32[2,2]{1,0}, s8[{{[0-9]+}}]{0}) custom-call(
; CHECK:           custom_call_target="__cublas$lt$matmul",
; CHECK:           backend_config={
; CHECK-DAG:         "alpha_real":1
; CHECK-DAG:         "alpha_imag":0
; CHECK-DAG:         "beta":0
; CHECK-DAG:         "dot_dimension_numbers":{
; CHECK-DAG:           "lhs_contracting_dimensions":["1"]
; CHECK-DAG:           "rhs_contracting_dimensions":["0"]
; CHECK-DAG:           "lhs_batch_dimensions":[]
; CHECK-DAG:           "rhs_batch_dimensions":[]
; CHECK-DAG:         }
; CHECK-DAG:         "precision_config":{
; CHECK-DAG:           "operand_precision":["DEFAULT","DEFAULT"]
; CHECK-DAG:         }
; CHECK-DAG:         "epilogue":"DEFAULT"
; CHECK:           }
; CHECK:         [[MATMUL1:%[^ ]+]] = f32[2,2]{1,0} get-tuple-element([[MATMUL1_TUPLE]]), index=0
; CHECK-NEXT:    [[OUT:%[^ ]+]] = (f32[2,2]{1,0}, s8[{{[0-9]+}}]{0}) custom-call{{.*}}[[MATMUL1]]
; CHECK:           custom_call_target="__cublas$lt$matmul",
; CHECK:           backend_config={
; CHECK-DAG:         "alpha_real":1
; CHECK-DAG:         "alpha_imag":0
; CHECK-DAG:         "beta":0
; CHECK-DAG:         "dot_dimension_numbers":{
; CHECK-DAG:           "lhs_contracting_dimensions":["1"]
; CHECK-DAG:           "rhs_contracting_dimensions":["0"]
; CHECK-DAG:           "lhs_batch_dimensions":[]
; CHECK-DAG:           "rhs_batch_dimensions":[]
; CHECK-DAG:         }
; CHECK-DAG:         "precision_config":{
; CHECK-DAG:           "operand_precision":["DEFAULT","DEFAULT"]
; CHECK-DAG:         }
; CHECK-DAG:         "epilogue":"DEFAULT"
; CHECK:           }
      )");
}

TEST_F(CublasLtGemmRewriteTest, VectorBiasTransposed) {
  const char* hlo_text = R"(
HloModule test

ENTRY test {
  x = f32[2,3] parameter(0)
  y = f32[3,4] parameter(1)
  z = f32[2] parameter(2)
  dot_a = f32[2,4] dot(x, y), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  z_bcast = f32[2,4] parameter(3)
  ROOT out = f32[2,4] add(dot_a, z_bcast)
}

)";

  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{1e-5, 1e-5}));
  MatchOptimizedHlo(hlo_text,
                    R"(
; CHECK-DAG:     [[P0:%[^ ]+]] = f32[2,3]{1,0} parameter(0)
; CHECK-DAG:     [[P1:%[^ ]+]] = f32[3,4]{1,0} parameter(1)
; CHECK-DAG:     [[P2_BCAST:%[^ ]+]] = f32[2,4]{1,0} parameter(3)
; CHECK:         [[OUT:%[^ ]+]] = (f32[2,4]{1,0}, s8[{{[0-9]+}}]{0}) custom-call([[P0]], [[P1]], [[P2_BCAST]]),
; CHECK:           custom_call_target="__cublas$lt$matmul",
; CHECK:           backend_config={
; CHECK-DAG:         "alpha_real":1
; CHECK-DAG:         "alpha_imag":0
; CHECK-DAG:         "beta":1
; CHECK-DAG:         "dot_dimension_numbers":{
; CHECK-DAG:           "lhs_contracting_dimensions":["1"]
; CHECK-DAG:           "rhs_contracting_dimensions":["0"]
; CHECK-DAG:           "lhs_batch_dimensions":[]
; CHECK-DAG:           "rhs_batch_dimensions":[]
; CHECK-DAG:         }
; CHECK-DAG:         "precision_config":{
; CHECK-DAG:           "operand_precision":["DEFAULT","DEFAULT"]
; CHECK-DAG:         }
; CHECK-DAG:         "epilogue":"DEFAULT"
; CHECK:           }
)");
}

TEST_F(CublasLtGemmRewriteTest, VectorBiasThenMatrixBias) {
  const char* hlo_text = R"(
HloModule test

ENTRY test {
  x = f32[2,3] parameter(0)
  y = f32[3,4] parameter(1)
  z = f32[4] parameter(2)
  z2 = f32[2,4] parameter(3)
  dot_a = f32[2,4] dot(x, y), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  z_bcast = f32[2,4] broadcast(z), dimensions={1}
  add0 = f32[2,4] add(dot_a, z_bcast)
  ROOT add1 = f32[2,4] add(add0, z2)
}

)";

  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{1e-5, 1e-5}));
  MatchOptimizedHlo(hlo_text,
                    R"(
; CHECK-LABEL: ENTRY %{{.*}} ({{.*}}: f32[2,3], {{.*}}: f32[3,4], {{.*}}: f32[4], {{.*}}: f32[2,4]) -> f32[2,4] {
; CHECK-DAG:     [[P0:%[^ ]+]] = f32[2,3]{1,0} parameter(0)
; CHECK-DAG:     [[P1:%[^ ]+]] = f32[3,4]{1,0} parameter(1)
; CHECK-DAG:     [[VECTOR_BIAS:%[^ ]+]] = f32[4]{0} parameter(2)
; CHECK-DAG:     [[MATRIX_BIAS:%[^ ]+]] = f32[2,4]{1,0} parameter(3)
; CHECK-NEXT:    [[OUT:%[^ ]+]] = (f32[2,4]{1,0}, s8[{{[0-9]+}}]{0}) custom-call([[P0]], [[P1]], [[MATRIX_BIAS]], [[VECTOR_BIAS]]),
; CHECK:           custom_call_target="__cublas$lt$matmul",
; CHECK:           backend_config={
; CHECK-DAG:         "alpha_real":1
; CHECK-DAG:         "alpha_imag":0
; CHECK-DAG:         "beta":1
; CHECK-DAG:         "dot_dimension_numbers":{
; CHECK-DAG:           "lhs_contracting_dimensions":["1"]
; CHECK-DAG:           "rhs_contracting_dimensions":["0"]
; CHECK-DAG:           "lhs_batch_dimensions":[]
; CHECK-DAG:           "rhs_batch_dimensions":[]
; CHECK-DAG:         }
; CHECK-DAG:         "precision_config":{
; CHECK-DAG:           "operand_precision":["DEFAULT","DEFAULT"]
; CHECK-DAG:         }
; CHECK-DAG:         "epilogue":"BIAS"
; CHECK:           }
)");
}

TEST_F(CublasLtGemmRewriteTest, BF16VectorBias) {
  const char* hlo_text = R"(
HloModule test

ENTRY test {
  x = bf16[16,24] parameter(0)
  y = bf16[24,32] parameter(1)
  z = bf16[32] parameter(2)
  dot_a = bf16[16,32] dot(x, y), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  z_bcast = bf16[16,32] broadcast(z), dimensions={1}
  ROOT out = bf16[16,32] add(dot_a, z_bcast)
}

)";

  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{3e-3, 1e-3}));

  if (IsCuda() &&
      !HasCudaComputeCapability(se::CudaComputeCapability::Ampere())) {
    GTEST_SKIP() << "Pre-Ampere casts up bf16 to fp32";
  }

  MatchOptimizedHlo(hlo_text,
                    R"(

; CHECK-LABEL: ENTRY %{{.*}} ({{.*}}: bf16[16,24], {{.*}}: bf16[24,32], {{.*}}: bf16[32]) -> bf16[16,32] {
; CHECK-DAG:     [[P0:%[^ ]+]] = bf16[16,24]{1,0} parameter(0)
; CHECK-DAG:     [[P1:%[^ ]+]] = bf16[24,32]{1,0} parameter(1)
; CHECK-DAG:     [[P2:%[^ ]+]] = bf16[32]{0} parameter(2)
; CHECK:         [[OUT:%[^ ]+]] = (bf16[16,32]{1,0}, s8[{{[0-9]+}}]{0}) custom-call([[P0]], [[P1]], [[P2]]),
; CHECK:           custom_call_target="__cublas$lt$matmul",
; CHECK:           backend_config={
; CHECK-DAG:         "alpha_real":1
; CHECK-DAG:         "alpha_imag":0
; CHECK-DAG:         "beta":0
; CHECK-DAG:         "dot_dimension_numbers":{
; CHECK-DAG:           "lhs_contracting_dimensions":["1"]
; CHECK-DAG:           "rhs_contracting_dimensions":["0"]
; CHECK-DAG:           "lhs_batch_dimensions":[]
; CHECK-DAG:           "rhs_batch_dimensions":[]
; CHECK-DAG:         }
; CHECK-DAG:         "precision_config":{
; CHECK-DAG:           "operand_precision":["DEFAULT","DEFAULT"]
; CHECK-DAG:         }
; CHECK-DAG:         "epilogue":"BIAS"
      )");
}

TEST_F(CublasLtGemmRewriteTest, BF16VectorBiasPadded) {
  if (IsCuda() &&
      !HasCudaComputeCapability(se::CudaComputeCapability::Ampere())) {
    GTEST_SKIP() << "Padding of GEMM bf16 operands only implemented on "
                    "architectures with bf16 Tensor Cores.";
  }
  const char* hlo_text = R"(
HloModule test

ENTRY test {
  x = bf16[2,3] parameter(0)
  y = bf16[3,4] parameter(1)
  z = bf16[4] parameter(2)
  dot_a = bf16[2,4] dot(x, y), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  z_bcast = bf16[2,4] broadcast(z), dimensions={1}
  ROOT out = bf16[2,4] add(dot_a, z_bcast)
})";

  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{1e-3, 1e-3}));
  if (IsCuda()) {
    MatchOptimizedHlo(hlo_text, R"(
    ; CHECK-DAG: ENTRY %{{.*}} ({{.*}}: bf16[2,3], {{.*}}: bf16[3,4], {{.*}}: bf16[4]) -> bf16[2,4] {
    ; CHECK-DAG:    bf16[8,8]{1,0} pad({{.*}}), padding=0_6x0_5
    ; CHECK-DAG:    bf16[8,8]{1,0} pad({{.*}}), padding=0_5x0_4
      )");
  } else {
    MatchOptimizedHlo(hlo_text, R"(
    ; CHECK-DAG: ENTRY %{{.*}} ({{.*}}: bf16[2,3], {{.*}}: bf16[3,4], {{.*}}: bf16[4]) -> bf16[2,4] {
    ; CHECK-DAG:    bf16[2,3]{1,0}
    ; CHECK-DAG:    bf16[3,4]{1,0}
    )");
  }
}

TEST_F(CublasLtGemmRewriteTest, ReluActivation) {
  const char* hlo_text = R"(
HloModule test

ENTRY test {
  x = f32[2,3] parameter(0)
  y = f32[3,4] parameter(1)
  dot_a = f32[2,4] dot(x, y), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  c = f32[] constant(0)
  c_bcast = f32[2,4] broadcast(c), dimensions={}
  ROOT out = f32[2,4] maximum(dot_a, c_bcast)
}

)";

  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{1e-5, 1e-5}));
  MatchOptimizedHlo(hlo_text,
                    R"(

; CHECK-LABEL: ENTRY %{{.*}} ({{.*}}: f32[2,3], {{.*}}: f32[3,4]) -> f32[2,4] {
; CHECK-DAG:     [[P0:%[^ ]+]] = f32[2,3]{1,0} parameter(0)
; CHECK-DAG:     [[P1:%[^ ]+]] = f32[3,4]{1,0} parameter(1)
; CHECK:         [[OUT:%[^ ]+]] = (f32[2,4]{1,0}, s8[{{[0-9]+}}]{0}) custom-call([[P0]], [[P1]]),
; CHECK:           custom_call_target="__cublas$lt$matmul",
; CHECK:           backend_config={
; CHECK-DAG:         "alpha_real":1
; CHECK-DAG:         "alpha_imag":0
; CHECK-DAG:         "beta":0
; CHECK-DAG:         "dot_dimension_numbers":{
; CHECK-DAG:           "lhs_contracting_dimensions":["1"]
; CHECK-DAG:           "rhs_contracting_dimensions":["0"]
; CHECK-DAG:           "lhs_batch_dimensions":[]
; CHECK-DAG:           "rhs_batch_dimensions":[]
; CHECK-DAG:         }
; CHECK-DAG:         "precision_config":{
; CHECK-DAG:           "operand_precision":["DEFAULT","DEFAULT"]
; CHECK-DAG:         }
; CHECK-DAG:         "epilogue":"RELU"
; CHECK:           }
)");
}

TEST_F(CublasLtGemmRewriteTest, BatchedReluActivation) {
  const char* hlo_text = R"(
HloModule test

ENTRY test {
  x = f32[2,3,4] parameter(0)
  y = f32[4,5,6] parameter(1)
  dot_a = f32[2,3,5,6] dot(x, y), lhs_contracting_dims={2}, rhs_contracting_dims={0}, operand_precision={highest,highest}
  c = f32[] constant(0)
  c_bcast = f32[2,3,5,6] broadcast(c), dimensions={}
  ROOT out = f32[2,3,5,6] maximum(dot_a, c_bcast)
}

)";

  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{1e-5, 1e-5}));
  MatchOptimizedHlo(hlo_text,
                    R"(

; CHECK-LABEL: ENTRY %{{.*}} ({{.*}}: f32[2,3,4], {{.*}}: f32[4,5,6]) -> f32[2,3,5,6] {
; CHECK-DAG:     [[P0:%[^ ]+]] = f32[2,3,4]{2,1,0} parameter(0)
; CHECK-DAG:     [[P0_BITCAST:%[^ ]+]] = f32[6,4]{1,0} bitcast([[P0]])
; CHECK-DAG:     [[P1:%[^ ]+]] = f32[4,5,6]{2,1,0} parameter(1)
; CHECK-DAG:     [[P1_BITCAST:%[^ ]+]] = f32[4,30]{1,0} bitcast([[P1]])
; CHECK:         [[MATMUL_TUPLE:%[^ ]+]] = (f32[6,30]{1,0}, s8[{{[0-9]+}}]{0}) custom-call([[P0_BITCAST]], [[P1_BITCAST]]),
; CHECK:           custom_call_target="__cublas$lt$matmul",
; CHECK:           backend_config={
; CHECK-DAG:         "alpha_real":1
; CHECK-DAG:         "alpha_imag":0
; CHECK-DAG:         "beta":0
; CHECK-DAG:         "dot_dimension_numbers":{
; CHECK-DAG:           "lhs_contracting_dimensions":["1"]
; CHECK-DAG:           "rhs_contracting_dimensions":["0"]
; CHECK-DAG:           "lhs_batch_dimensions":[]
; CHECK-DAG:           "rhs_batch_dimensions":[]
; CHECK-DAG:         }
; CHECK-DAG:         "precision_config":{
; CHECK-DAG:           "operand_precision":["HIGHEST","HIGHEST"]
; CHECK-DAG:         }
; CHECK-DAG:         "epilogue":"RELU"
; CHECK:           }
; CHECK:         [[MATMUL:%[^ ]+]] = f32[6,30]{1,0} get-tuple-element([[MATMUL_TUPLE]]), index=0
; CHECK-NEXT:    ROOT [[OUT:%[^ ]+]] = f32[2,3,5,6]{3,2,1,0} bitcast([[MATMUL]])
      )");
}

TEST_F(CublasLtGemmRewriteTest, ReluActivationSliced) {
  const char* hlo_text = R"(
HloModule test

ENTRY test {
  x = f32[2,3] parameter(0)
  y = f32[3,4] parameter(1)
  dot_a = f32[2,4] dot(x, y), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  c = f32[] constant(0)
  c_bcast = f32[2,2] broadcast(c), dimensions={}
  slice_a = f32[2,2] slice(dot_a), slice={[0:2], [0:2]}
  ROOT out = f32[2,2] maximum(slice_a, c_bcast)
}

)";

  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{1e-5, 1e-5}));
  MatchOptimizedHlo(hlo_text,
                    R"(

; CHECK-LABEL: ENTRY %{{.*}} ({{.*}}: f32[2,3], {{.*}}: f32[3,4]) -> f32[2,2] {
; CHECK-DAG:     [[P0:%[^ ]+]] = f32[2,3]{1,0} parameter(0)
; CHECK-DAG:     [[P1:%[^ ]+]] = f32[3,4]{1,0} parameter(1)
; CHECK:         [[MATMUL_TUPLE:%[^ ]+]] = (f32[2,4]{1,0}, s8[{{[0-9]+}}]{0}) custom-call([[P0]], [[P1]]),
; CHECK:           custom_call_target="__cublas$lt$matmul",
; CHECK:           backend_config={
; CHECK-DAG:         "alpha_real":1
; CHECK-DAG:         "alpha_imag":0
; CHECK-DAG:         "beta":0
; CHECK-DAG:         "dot_dimension_numbers":{
; CHECK-DAG:           "lhs_contracting_dimensions":["1"]
; CHECK-DAG:           "rhs_contracting_dimensions":["0"]
; CHECK-DAG:           "lhs_batch_dimensions":[]
; CHECK-DAG:           "rhs_batch_dimensions":[]
; CHECK-DAG:         }
; CHECK-DAG:         "precision_config":{
; CHECK-DAG:           "operand_precision":["DEFAULT","DEFAULT"]
; CHECK-DAG:         }
; CHECK-DAG:         "epilogue":"RELU"
; CHECK:           }
; CHECK:         [[MATMUL:%[^ ]+]] = f32[2,4]{1,0} get-tuple-element([[MATMUL_TUPLE]]), index=0
; CHECK:    ROOT [[OUT:%[^ ]+]] = f32[2,2]{1,0} fusion([[MATMUL]]), kind=kLoop
      )");
}

TEST_F(CublasLtGemmRewriteTest, MatrixBiasReluActivation) {
  const char* hlo_text = R"(
HloModule test

ENTRY test {
  x = f32[2,3] parameter(0)
  y = f32[3,4] parameter(1)
  z = f32[2,4] parameter(2)
  dot_a = f32[2,4] dot(x, y), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  add = f32[2,4] add(dot_a, z)
  c = f32[] constant(0)
  c_bcast = f32[2,4] broadcast(c), dimensions={}
  ROOT out = f32[2,4] maximum(add, c_bcast)
}

)";

  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{1e-5, 1e-5}));
  MatchOptimizedHlo(hlo_text,
                    R"(

; CHECK-LABEL: ENTRY %{{.*}} ({{.*}}: f32[2,3], {{.*}}: f32[3,4], {{.*}}: f32[2,4]) -> f32[2,4] {
; CHECK-DAG:     [[P0:%[^ ]+]] = f32[2,3]{1,0} parameter(0)
; CHECK-DAG:     [[P1:%[^ ]+]] = f32[3,4]{1,0} parameter(1)
; CHECK-DAG:     [[P2:%[^ ]+]] = f32[2,4]{1,0} parameter(2)
; CHECK:         [[OUT:%[^ ]+]] = (f32[2,4]{1,0}, s8[{{[0-9]+}}]{0}) custom-call([[P0]], [[P1]], [[P2]]),
; CHECK:           custom_call_target="__cublas$lt$matmul",
; CHECK:           backend_config={
; CHECK-DAG:         "alpha_real":1
; CHECK-DAG:         "alpha_imag":0
; CHECK-DAG:         "beta":1
; CHECK-DAG:         "dot_dimension_numbers":{
; CHECK-DAG:           "lhs_contracting_dimensions":["1"]
; CHECK-DAG:           "rhs_contracting_dimensions":["0"]
; CHECK-DAG:           "lhs_batch_dimensions":[]
; CHECK-DAG:           "rhs_batch_dimensions":[]
; CHECK-DAG:         }
; CHECK-DAG:         "precision_config":{
; CHECK-DAG:           "operand_precision":["DEFAULT","DEFAULT"]
; CHECK-DAG:         }
; CHECK-DAG:         "epilogue":"RELU"
; CHECK:           }
      )");
}

TEST_F(CublasLtGemmRewriteTest, SquareMatrixBiasReluActivation) {
  const char* hlo_text = R"(
HloModule test

ENTRY test {
  x = f32[4,4] parameter(0)
  y = f32[4,4] parameter(1)
  z = f32[4,4] parameter(2)
  dot_a = f32[4,4] dot(x, y), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  add = f32[4,4] add(dot_a, z)
  c = f32[] constant(0)
  c_bcast = f32[4,4] broadcast(c), dimensions={}
  ROOT out = f32[4,4] maximum(add, c_bcast)
}

)";

  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{1e-5, 1e-5}));
  MatchOptimizedHlo(hlo_text,
                    R"(

; CHECK-LABEL: ENTRY %{{.*}} ({{.*}}: f32[4,4], {{.*}}: f32[4,4], {{.*}}: f32[4,4]) -> f32[4,4] {
; CHECK-DAG:     [[P0:%[^ ]+]] = f32[4,4]{1,0} parameter(0)
; CHECK-DAG:     [[P1:%[^ ]+]] = f32[4,4]{1,0} parameter(1)
; CHECK-DAG:     [[P2:%[^ ]+]] = f32[4,4]{1,0} parameter(2)
; CHECK:         [[OUT:%[^ ]+]] = (f32[4,4]{1,0}, s8[{{[0-9]+}}]{0}) custom-call([[P0]], [[P1]], [[P2]]),
; CHECK:           custom_call_target="__cublas$lt$matmul",
; CHECK:           backend_config={
; CHECK-DAG:         "alpha_real":1
; CHECK-DAG:         "alpha_imag":0
; CHECK-DAG:         "beta":1
; CHECK-DAG:         "dot_dimension_numbers":{
; CHECK-DAG:           "lhs_contracting_dimensions":["1"]
; CHECK-DAG:           "rhs_contracting_dimensions":["0"]
; CHECK-DAG:           "lhs_batch_dimensions":[]
; CHECK-DAG:           "rhs_batch_dimensions":[]
; CHECK-DAG:         }
; CHECK-DAG:         "precision_config":{
; CHECK-DAG:           "operand_precision":["DEFAULT","DEFAULT"]
; CHECK-DAG:         }
; CHECK-DAG:         "epilogue":"RELU"
; CHECK:           }
      )");
}

TEST_F(CublasLtGemmRewriteTest, VectorBiasReluActivation) {
  const char* hlo_text = R"(
HloModule test

ENTRY test {
  x = f32[2,3] parameter(0)
  y = f32[3,4] parameter(1)
  z = f32[4] parameter(2)
  dot_a = f32[2,4] dot(x, y), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  z_bcast = f32[2,4] broadcast(z), dimensions={1}
  add = f32[2,4] add(dot_a, z_bcast)
  c = f32[] constant(0)
  c_bcast = f32[2,4] broadcast(c), dimensions={}
  ROOT out = f32[2,4] maximum(add, c_bcast)
}

)";

  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{1e-5, 1e-5}));
  MatchOptimizedHlo(hlo_text,
                    R"(

; CHECK-LABEL: ENTRY %{{.*}} ({{.*}}: f32[2,3], {{.*}}: f32[3,4], {{.*}}: f32[4]) -> f32[2,4] {
; CHECK-DAG:     [[P0:%[^ ]+]] = f32[2,3]{1,0} parameter(0)
; CHECK-DAG:     [[P1:%[^ ]+]] = f32[3,4]{1,0} parameter(1)
; CHECK-DAG:     [[P2:%[^ ]+]] = f32[4]{0} parameter(2)
; CHECK:         [[OUT:%[^ ]+]] = (f32[2,4]{1,0}, s8[{{[0-9]+}}]{0}) custom-call([[P0]], [[P1]], [[P2]]),
; CHECK:           custom_call_target="__cublas$lt$matmul",
; CHECK:           backend_config={
; CHECK-DAG:         "alpha_real":1
; CHECK-DAG:         "alpha_imag":0
; CHECK-DAG:         "beta":0
; CHECK-DAG:         "dot_dimension_numbers":{
; CHECK-DAG:           "lhs_contracting_dimensions":["1"]
; CHECK-DAG:           "rhs_contracting_dimensions":["0"]
; CHECK-DAG:           "lhs_batch_dimensions":[]
; CHECK-DAG:           "rhs_batch_dimensions":[]
; CHECK-DAG:         }
; CHECK-DAG:         "precision_config":{
; CHECK-DAG:           "operand_precision":["DEFAULT","DEFAULT"]
; CHECK-DAG:         }
; CHECK-DAG:         "epilogue":"BIAS_RELU"
; CHECK:           }
      )");
}

TEST_F(CublasLtGemmRewriteTest, BatchedVectorBiasReluActivation) {
  const char* hlo_text = R"(
HloModule test

ENTRY test {
  x = f32[2,3,4] parameter(0)
  y = f32[4,5,6] parameter(1)
  z = f32[3,5,6] parameter(2)
  dot_a = f32[2,3,5,6] dot(x, y), lhs_contracting_dims={2}, rhs_contracting_dims={0}, operand_precision={highest,highest}
  z_bcast = f32[2,3,5,6] broadcast(z), dimensions={1,2,3}
  add = f32[2,3,5,6] add(dot_a, z_bcast)
  c = f32[] constant(0)
  c_bcast = f32[2,3,5,6] broadcast(c), dimensions={}
  ROOT out = f32[2,3,5,6] maximum(add, c_bcast)
}

)";

  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{1e-5, 1e-5}));
  MatchOptimizedHlo(hlo_text,
                    R"(

; CHECK-LABEL: ENTRY %{{.*}} ({{.*}}: f32[2,3,4], {{.*}}: f32[4,5,6], {{.*}}: f32[3,5,6]) -> f32[2,3,5,6] {
; CHECK-DAG:     [[P0:%[^ ]+]] = f32[2,3,4]{{{.*}}} parameter(0)
; CHECK-DAG:     [[P0_BITCAST:%[^ ]+]] = f32[6,4]{{{.*}}} bitcast([[P0]])
; CHECK-DAG:     [[P1:%[^ ]+]] = f32[4,5,6]{{{.*}}} parameter(1)
; CHECK-DAG:     [[P1_BITCAST:%[^ ]+]] = f32[4,30]{{{.*}}} bitcast([[P1]])
; CHECK:         [[MATMUL_TUPLE:%[^ ]+]] = (f32[6,30]{1,0}, s8[{{[0-9]+}}]{0}) custom-call([[P0_BITCAST]], [[P1_BITCAST]], {{.*}}),
; CHECK:           custom_call_target="__cublas$lt$matmul",
; CHECK:           backend_config={
; CHECK-DAG:         "alpha_real":1
; CHECK-DAG:         "alpha_imag":0
; CHECK-DAG:         "beta":1
; CHECK-DAG:         "dot_dimension_numbers":{
; CHECK-DAG:           "lhs_contracting_dimensions":["1"]
; CHECK-DAG:           "rhs_contracting_dimensions":["0"]
; CHECK-DAG:           "lhs_batch_dimensions":[]
; CHECK-DAG:           "rhs_batch_dimensions":[]
; CHECK-DAG:         }
; CHECK-DAG:         "precision_config":{
; CHECK-DAG:           "operand_precision":["HIGHEST","HIGHEST"]
; CHECK-DAG:         }
; CHECK-DAG:         "epilogue":"RELU"
; CHECK:           }
; CHECK:         [[MATMUL:%[^ ]+]] = f32[6,30]{1,0} get-tuple-element([[MATMUL_TUPLE]]), index=0
; CHECK-NEXT:    ROOT [[OUT:%[^ ]+]] = f32[2,3,5,6]{3,2,1,0} bitcast([[MATMUL]])
      )");
}

TEST_F(CublasLtGemmRewriteTest, VectorBiasTransposedReluActivation) {
  const char* hlo_text = R"(
HloModule test

ENTRY test {
  x = f32[2,3] parameter(0)
  y = f32[3,4] parameter(1)
  z = f32[2] parameter(2)
  dot_a = f32[2,4] dot(x, y), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  z_bcast = f32[2,4] broadcast(z), dimensions={0}
  add = f32[2,4] add(dot_a, z_bcast)
  c = f32[] constant(0)
  c_bcast = f32[2,4] broadcast(c), dimensions={}
  maximum = f32[2,4] maximum(add, c_bcast)
  ROOT out = f32[4,2] transpose(maximum), dimensions={1,0}
}

)";

  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{1e-5, 1e-5}));
  MatchOptimizedHlo(hlo_text,
                    R"(

; CHECK-LABEL: ENTRY %{{.*}} ({{.*}}: f32[2,3], {{.*}}: f32[3,4], {{.*}}: f32[2]) -> f32[4,2] {
; CHECK-DAG:     [[P0:%[^ ]+]] = f32[2,3]{1,0} parameter(0)
; CHECK-DAG:     [[P1:%[^ ]+]] = f32[3,4]{1,0} parameter(1)
; CHECK-DAG:     [[P2:%[^ ]+]] = f32[2]{0} parameter(2)
; CHECK:         [[MATMUL_TUPLE:%[^ ]+]] = (f32[2,4]{0,1}, s8[{{[0-9]+}}]{0}) custom-call([[P0]], [[P1]], [[P2]]),
; CHECK:           custom_call_target="__cublas$lt$matmul",
; CHECK:           backend_config={
; CHECK-DAG:       "alpha_real":1
; CHECK-DAG:       "alpha_imag":0
; CHECK-DAG:       "beta":0
; CHECK-DAG:         "dot_dimension_numbers":{
; CHECK-DAG:           "lhs_contracting_dimensions":["1"]
; CHECK-DAG:           "rhs_contracting_dimensions":["0"]
; CHECK-DAG:           "lhs_batch_dimensions":[]
; CHECK-DAG:           "rhs_batch_dimensions":[]
; CHECK-DAG:         }
; CHECK-DAG:         "precision_config":{
; CHECK-DAG:           "operand_precision":["DEFAULT","DEFAULT"]
; CHECK-DAG:         }
; CHECK-DAG:         "epilogue":"BIAS_RELU"
; CHECK:           }
; CHECK-NEXT:    [[MATMUL:%[^ ]+]] = f32[2,4]{0,1} get-tuple-element([[MATMUL_TUPLE]]), index=0
; CHECK-NEXT:    ROOT [[OUT:%[^ ]+]] = f32[4,2]{1,0} bitcast([[MATMUL]])
)");
}

TEST_F(CublasLtGemmRewriteTest, VectorBiasThenMatrixBiasReluActivation) {
  const char* hlo_text = R"(
HloModule test

ENTRY test {
  x = f32[2,3] parameter(0)
  y = f32[3,4] parameter(1)
  z_vec = f32[4] parameter(2)
  z_matrix = f32[2,4] parameter(3)
  dot_a = f32[2,4] dot(x, y), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  z_bcast = f32[2,4] broadcast(z_vec), dimensions={1}
  add0 = f32[2,4] add(dot_a, z_bcast)
  add1 = f32[2,4] add(add0, z_matrix)
  c = f32[] constant(0)
  c_bcast = f32[2,4] broadcast(c), dimensions={}
  ROOT out = f32[2,4] maximum(add1, c_bcast)
}

)";

  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{1e-5, 1e-5}));
  MatchOptimizedHlo(hlo_text,
                    R"(

; CHECK-LABEL: ENTRY %{{.*}} ({{.*}}: f32[2,3], {{.*}}: f32[3,4], {{.*}}: f32[4], {{.*}}: f32[2,4]) -> f32[2,4] {
; CHECK-DAG:     [[P0:%[^ ]+]] = f32[2,3]{1,0} parameter(0)
; CHECK-DAG:     [[P1:%[^ ]+]] = f32[3,4]{1,0} parameter(1)
; CHECK-DAG:     [[P2:%[^ ]+]] = f32[4]{0} parameter(2)
; CHECK-DAG:     [[P3:%[^ ]+]] = f32[2,4]{1,0} parameter(3)
; CHECK-NEXT:    [[OUT:%[^ ]+]] = (f32[2,4]{1,0}, s8[{{[0-9]+}}]{0}) custom-call([[P0]], [[P1]], [[P3]], [[P2]]),
; CHECK:           custom_call_target="__cublas$lt$matmul",
; CHECK:           backend_config={
; CHECK-DAG:         "alpha_real":1
; CHECK-DAG:         "alpha_imag":0
; CHECK-DAG:         "beta":1
; CHECK-DAG:         "dot_dimension_numbers":{
; CHECK-DAG:           "lhs_contracting_dimensions":["1"]
; CHECK-DAG:           "rhs_contracting_dimensions":["0"]
; CHECK-DAG:           "lhs_batch_dimensions":[]
; CHECK-DAG:           "rhs_batch_dimensions":[]
; CHECK-DAG:         }
; CHECK-DAG:         "precision_config":{
; CHECK-DAG:           "operand_precision":["DEFAULT","DEFAULT"]
; CHECK-DAG:         }
; CHECK-DAG:         "epilogue":"BIAS_RELU"
; CHECK:           }
      )");
}

TEST_F(CublasLtGemmRewriteTest, ApproxGeluActivation) {
  const char* hlo_text = R"(
HloModule test

ENTRY test {
  x = f32[2,3] parameter(0)
  y = f32[3,4] parameter(1)
  dot = f32[2,4] dot(x, y), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  mul.0 = f32[2,4] multiply(dot, dot)
  mul.1 = f32[2,4] multiply(dot, mul.0)
  const.0 = f32[] constant(0.044715)
  bcast.0 = f32[2,4] broadcast(const.0), dimensions={}
  mul.2 = f32[2,4] multiply(mul.1, bcast.0)
  add.0 = f32[2,4] add(dot, mul.2)
  const.1 = f32[] constant(0.797884583)
  bcast.1 = f32[2,4] broadcast(const.1), dimensions={}
  mul.3 = f32[2,4] multiply(add.0, bcast.1)
  tanh = f32[2,4] tanh(mul.3)
  const.2 = f32[] constant(1)
  bcast.2 = f32[2,4] broadcast(const.2), dimensions={}
  add.2 = f32[2,4] add(tanh, bcast.2)
  const.3 = f32[] constant(0.5)
  bcast.3 = f32[2,4] broadcast(const.3), dimensions={}
  mul.4 = f32[2,4] multiply(add.2, bcast.3)
  ROOT out = f32[2,4] multiply(dot, mul.4)
}

)";

  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{1e-5, 1e-5}));
  MatchOptimizedHlo(hlo_text,
                    R"(

; CHECK-LABEL: ENTRY %{{.*}} ({{.*}}: f32[2,3], {{.*}}: f32[3,4]) -> f32[2,4] {
; CHECK-DAG:     [[P0:%[^ ]+]] = f32[2,3]{1,0} parameter(0)
; CHECK-DAG:     [[P1:%[^ ]+]] = f32[3,4]{1,0} parameter(1)
; CHECK:         [[OUT:%[^ ]+]] = (f32[2,4]{1,0}, s8[{{[0-9]+}}]{0}) custom-call([[P0]], [[P1]]),
; CHECK:           custom_call_target="__cublas$lt$matmul",
; CHECK:           backend_config={
; CHECK-DAG:         "alpha_real":1
; CHECK-DAG:         "alpha_imag":0
; CHECK-DAG:         "beta":0
; CHECK-DAG:         "dot_dimension_numbers":{
; CHECK-DAG:           "lhs_contracting_dimensions":["1"]
; CHECK-DAG:           "rhs_contracting_dimensions":["0"]
; CHECK-DAG:           "lhs_batch_dimensions":[]
; CHECK-DAG:           "rhs_batch_dimensions":[]
; CHECK-DAG:         }
; CHECK-DAG:         "precision_config":{
; CHECK-DAG:           "operand_precision":["DEFAULT","DEFAULT"]
; CHECK-DAG:         }
; CHECK-DAG:         "epilogue":"GELU"
; CHECK:           }
      )");
}

TEST_F(CublasLtGemmRewriteTest, ApproxGeluActivationWrongConstant) {
  // Modify one constant slightly, so it should no longer pattern match.
  const char* hlo_text = R"(
HloModule test

ENTRY test {
  x = f32[2,3] parameter(0)
  y = f32[3,4] parameter(1)
  dot = f32[2,4] dot(x, y), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  mul.0 = f32[2,4] multiply(dot, dot)
  mul.1 = f32[2,4] multiply(dot, mul.0)
  const.0 = f32[] constant(0.05)
  bcast.0 = f32[2,4] broadcast(const.0), dimensions={}
  mul.2 = f32[2,4] multiply(mul.1, bcast.0)
  add.0 = f32[2,4] add(dot, mul.2)
  const.1 = f32[] constant(0.797884583)
  bcast.1 = f32[2,4] broadcast(const.1), dimensions={}
  mul.3 = f32[2,4] multiply(add.0, bcast.1)
  tanh = f32[2,4] tanh(mul.3)
  const.2 = f32[] constant(1)
  bcast.2 = f32[2,4] broadcast(const.2), dimensions={}
  add.2 = f32[2,4] add(tanh, bcast.2)
  const.3 = f32[] constant(0.5)
  bcast.3 = f32[2,4] broadcast(const.3), dimensions={}
  mul.4 = f32[2,4] multiply(add.2, bcast.3)
  ROOT out = f32[2,4] multiply(dot, mul.4)
}

)";

  MatchOptimizedHlo(hlo_text,
                    R"(

; CHECK-NOT: GELU
      )");
}

TEST_F(CublasLtGemmRewriteTest, MatrixBiasSwishActivation) {
  auto runtime_version = GetToolkitVersion();
  bool rocm_gelu_available =
      IsRocm() &&
      (runtime_version >= stream_executor::SemanticVersion(7, 0, 0));
  if (!rocm_gelu_available) {
    GTEST_SKIP() << "TODO: Unsupported blas-lt epilogue on ROCM";
  }
  const char* hlo_text = R"(
HloModule test

ENTRY test {
  x = f32[2,3] parameter(0)
  y = f32[3,4] parameter(1)
  dot = f32[2,4] dot(x, y), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  neg = f32[2,4] negate(dot)
  exp = f32[2,4] exponential(neg)
  one = f32[] constant(1)
  one_bcast = f32[2,4] broadcast(one), dimensions={}
  denom = f32[2,4] add(one_bcast, exp)
  sigmoid = f32[2,4] divide(one_bcast, denom)
  ROOT swish = f32[2,4] multiply(dot, sigmoid)
}

)";

  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{1e-3, 1e-3}));
  MatchOptimizedHlo(hlo_text,
                    R"(

; CHECK-LABEL: ENTRY %{{.*}} ({{.*}}: f32[2,3], {{.*}}: f32[3,4]) -> f32[2,4] {
; CHECK-DAG:    [[P0:%[^ ]+]] = f32[2,3]{1,0} parameter(0)
; CHECK-DAG:    [[P1:%[^ ]+]] = f32[3,4]{1,0} parameter(1)
; CHECK:        [[OUT:%[^ ]+]] = (f32[2,4]{1,0}, s8[{{[0-9]+}}]{0}) custom-call([[P0]], [[P1]]),
; CHECK:           custom_call_target="__cublas$lt$matmul",
; CHECK:           backend_config={
; CHECK-DAG:         "alpha_real":1
; CHECK-DAG:         "alpha_imag":0
; CHECK-DAG:         "beta":0
; CHECK-DAG:         "dot_dimension_numbers":{
; CHECK-DAG:           "lhs_contracting_dimensions":["1"]
; CHECK-DAG:           "rhs_contracting_dimensions":["0"]
; CHECK-DAG:           "lhs_batch_dimensions":[]
; CHECK-DAG:           "rhs_batch_dimensions":[]
; CHECK-DAG:         }
; CHECK-DAG:         "precision_config":{
; CHECK-DAG:           "operand_precision":["DEFAULT","DEFAULT"]
; CHECK-DAG:         }
; CHECK-DAG:         "epilogue":"SILU"
; CHECK:           }
      )");
}

TEST_F(CublasLtGemmRewriteTest, SwishActivationWithBitcastAndAuxiliaryOutput) {
  auto runtime_version = GetToolkitVersion();
  bool rocm_swish_available =
      IsRocm() &&
      (runtime_version >= stream_executor::SemanticVersion(7, 0, 0));
  if (!rocm_swish_available) {
    GTEST_SKIP() << "Swish/SILU activation fusion only available on ROCm 7.0+";
  }

  const char* hlo_text = R"(
HloModule test

ENTRY test (x: bf16[49152,11008], y: bf16[11008,11008]) -> (bf16[12,4096,11008], bf16[49152,11008]) {
  x = bf16[49152,11008]{1,0} parameter(0)
  y = bf16[11008,11008]{1,0} parameter(1)
  dot = bf16[49152,11008]{1,0} dot(x, y), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  bitcast = bf16[12,4096,11008]{2,1,0} bitcast(dot)

  one = bf16[] constant(1)
  one_bcast = bf16[12,4096,11008]{2,1,0} broadcast(one), dimensions={}
  neg = bf16[12,4096,11008]{2,1,0} negate(bitcast)
  exp = bf16[12,4096,11008]{2,1,0} exponential(neg)
  add = bf16[12,4096,11008]{2,1,0} add(exp, one_bcast)
  sigmoid = bf16[12,4096,11008]{2,1,0} divide(one_bcast, add)
  swish = bf16[12,4096,11008]{2,1,0} multiply(bitcast, sigmoid)

  extra_user = bf16[49152,11008]{1,0} negate(dot)

  ROOT out = (bf16[12,4096,11008]{2,1,0}, bf16[49152,11008]{1,0}) tuple(swish, extra_user)
}
)";

  HloModuleConfig config;
  DebugOptions debug_options = GetDebugOptionsForTest();
  config.set_debug_options(debug_options);
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_text, config));

  GemmRewriterOptions options;
  options.enable_cublaslt = true;
  GemmRewriter pass(Capability(), GetToolkitVersion(), options);

  TF_ASSERT_OK_AND_ASSIGN(bool changed, RunHloPass(&pass, module.get()));
  EXPECT_TRUE(changed);

  TF_ASSERT_OK_AND_ASSIGN(bool filecheck_result,
                          RunFileCheck(module->ToString(), R"(
; CHECK:           [[GEMM_TUPLE:%[^ ]+]] = {{.*}} custom-call
; CHECK:           custom_call_target="__cublas$lt$matmul",
; CHECK-DAG:         "epilogue":"DEFAULT"
      )"));
  EXPECT_TRUE(filecheck_result);
}

TEST_F(CublasLtGemmRewriteTest, VectorBiasThenApproxGeluActivation) {
  auto runtime_version = GetToolkitVersion();
  bool rocm_gelu_available =
      IsRocm() &&
      (runtime_version >= stream_executor::SemanticVersion(6, 0, 0));
  if (IsRocm() && !rocm_gelu_available) {
    GTEST_SKIP() << "TODO: Unsupported blas-lt epilogue on ROCM";
  }
  const char* hlo_text = R"(
HloModule test

ENTRY test {
  x = f32[2,3] parameter(0)
  y = f32[3,4] parameter(1)
  z = f32[4] parameter(2)
  dot = f32[2,4] dot(x, y), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  z_bcast = f32[2,4] broadcast(z), dimensions={1}
  add = f32[2,4] add(dot, z_bcast)
  mul.0 = f32[2,4] multiply(add, add)
  mul.1 = f32[2,4] multiply(add, mul.0)
  const.0 = f32[] constant(0.044715)
  bcast.0 = f32[2,4] broadcast(const.0), dimensions={}
  mul.2 = f32[2,4] multiply(mul.1, bcast.0)
  add.0 = f32[2,4] add(add, mul.2)
  const.1 = f32[] constant(0.797884583)
  bcast.1 = f32[2,4] broadcast(const.1), dimensions={}
  mul.3 = f32[2,4] multiply(add.0, bcast.1)
  tanh = f32[2,4] tanh(mul.3)
  const.2 = f32[] constant(1)
  bcast.2 = f32[2,4] broadcast(const.2), dimensions={}
  add.2 = f32[2,4] add(tanh, bcast.2)
  const.3 = f32[] constant(0.5)
  bcast.3 = f32[2,4] broadcast(const.3), dimensions={}
  mul.4 = f32[2,4] multiply(add.2, bcast.3)
  ROOT out = f32[2,4] multiply(add, mul.4)
}

)";

  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{1e-5, 1e-5}));
  MatchOptimizedHlo(hlo_text,
                    R"(

; CHECK-LABEL: ENTRY %{{.*}} ({{.*}}: f32[2,3], {{.*}}: f32[3,4], {{.*}}: f32[4]) -> f32[2,4] {
; CHECK-DAG:     [[P0:%[^ ]+]] = f32[2,3]{1,0} parameter(0)
; CHECK-DAG:     [[P1:%[^ ]+]] = f32[3,4]{1,0} parameter(1)
; CHECK-DAG:     [[P2:%[^ ]+]] = f32[4]{0} parameter(2)
; CHECK:         [[OUT:%[^ ]+]] = (f32[2,4]{1,0}, s8[{{[0-9]+}}]{0}) custom-call([[P0]], [[P1]], [[P2]]),
; CHECK:           custom_call_target="__cublas$lt$matmul",
; CHECK:           backend_config={
; CHECK-DAG:         "alpha_real":1
; CHECK-DAG:         "alpha_imag":0
; CHECK-DAG:         "beta":0
; CHECK-DAG:         "dot_dimension_numbers":{
; CHECK-DAG:           "lhs_contracting_dimensions":["1"]
; CHECK-DAG:           "rhs_contracting_dimensions":["0"]
; CHECK-DAG:           "lhs_batch_dimensions":[]
; CHECK-DAG:           "rhs_batch_dimensions":[]
; CHECK-DAG:         }
; CHECK-DAG:         "precision_config":{
; CHECK-DAG:           "operand_precision":["DEFAULT","DEFAULT"]
; CHECK-DAG:         }
; CHECK-DAG:         "epilogue":"BIAS_GELU"
; CHECK:           }
      )");
}

TEST_F(CublasLtGemmRewriteTest, ApproxGeluActivationWithAux) {
  if (IsRocm()) {
    GTEST_SKIP() << "TODO: Unsupported blas-lt epilogue on ROCM";
  }
  const char* hlo_text = R"(
HloModule test

ENTRY test {
  x = f32[2,3] parameter(0)
  y = f32[3,4] parameter(1)
  dot = f32[2,4] dot(x, y), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  mul.0 = f32[2,4] multiply(dot, dot)
  mul.1 = f32[2,4] multiply(dot, mul.0)
  const.0 = f32[] constant(0.044715)
  bcast.0 = f32[2,4] broadcast(const.0), dimensions={}
  mul.2 = f32[2,4] multiply(mul.1, bcast.0)
  add.0 = f32[2,4] add(dot, mul.2)
  const.1 = f32[] constant(0.797884583)
  bcast.1 = f32[2,4] broadcast(const.1), dimensions={}
  mul.3 = f32[2,4] multiply(add.0, bcast.1)
  tanh = f32[2,4] tanh(mul.3)
  const.2 = f32[] constant(1)
  bcast.2 = f32[2,4] broadcast(const.2), dimensions={}
  add.2 = f32[2,4] add(tanh, bcast.2)
  const.3 = f32[] constant(0.5)
  bcast.3 = f32[2,4] broadcast(const.3), dimensions={}
  mul.4 = f32[2,4] multiply(add.2, bcast.3)
  mul.5 = f32[2,4] multiply(dot, mul.4)
  ROOT out = (f32[2,4], f32[2,4]) tuple(mul.5, dot)
}

)";

  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{1e-5, 1e-5}));
  MatchOptimizedHlo(hlo_text,
                    R"(

; CHECK-LABEL: ENTRY %{{.*}} ({{.*}}: f32[2,3], {{.*}}: f32[3,4]) -> (f32[2,4], f32[2,4]) {
; CHECK-DAG:     [[P0:%[^ ]+]] = f32[2,3]{1,0} parameter(0)
; CHECK-DAG:     [[P1:%[^ ]+]] = f32[3,4]{1,0} parameter(1)
; CHECK:         [[OUT:%[^ ]+]] = (f32[2,4]{1,0}, f32[2,4]{1,0}, s8[{{[0-9]+}}]{0}) custom-call([[P0]], [[P1]]),
; CHECK:           custom_call_target="__cublas$lt$matmul",
; CHECK:           backend_config={
; CHECK-DAG:         "alpha_real":1
; CHECK-DAG:         "alpha_imag":0
; CHECK-DAG:         "beta":0
; CHECK-DAG:         "dot_dimension_numbers":{
; CHECK-DAG:           "lhs_contracting_dimensions":["1"]
; CHECK-DAG:           "rhs_contracting_dimensions":["0"]
; CHECK-DAG:           "lhs_batch_dimensions":[]
; CHECK-DAG:           "rhs_batch_dimensions":[]
; CHECK-DAG:         }
; CHECK-DAG:         "precision_config":{
; CHECK-DAG:           "operand_precision":["DEFAULT","DEFAULT"]
; CHECK-DAG:         }
; CHECK-DAG:         "epilogue":"GELU_AUX"
; CHECK:           }
      )");
}

TEST_F(CublasLtGemmRewriteTest,
       ApproxGeluActivationWithAuxAndBitcastAndExtraBitcastUser) {
  if (IsRocm()) {
    GTEST_SKIP() << "TODO: Unsupported blas-lt epilogue on ROCM";
  }
  const char* hlo_text = R"(
HloModule test

ENTRY test {
  x = f32[2,3] parameter(0)
  y = f32[3,4] parameter(1)
  dot = f32[2,4] dot(x, y), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  bitcasted_dot = f32[8] bitcast(dot)
  mul.0 = f32[8] multiply(bitcasted_dot, bitcasted_dot)
  mul.1 = f32[8] multiply(bitcasted_dot, mul.0)
  const.0 = f32[] constant(0.044715)
  bcast.0 = f32[8] broadcast(const.0), dimensions={}
  mul.2 = f32[8] multiply(mul.1, bcast.0)
  add.0 = f32[8] add(bitcasted_dot, mul.2)
  const.1 = f32[] constant(0.797884583)
  bcast.1 = f32[8] broadcast(const.1), dimensions={}
  mul.3 = f32[8] multiply(add.0, bcast.1)
  tanh = f32[8] tanh(mul.3)
  const.2 = f32[] constant(1)
  bcast.2 = f32[8] broadcast(const.2), dimensions={}
  add.2 = f32[8] add(tanh, bcast.2)
  const.3 = f32[] constant(0.5)
  bcast.3 = f32[8] broadcast(const.3), dimensions={}
  mul.4 = f32[8] multiply(add.2, bcast.3)
  mul.5 = f32[8] multiply(bitcasted_dot, mul.4)
  ROOT out = (f32[8], f32[8]) tuple(mul.5, bitcasted_dot)
}

)";

  GemmRewriterOptions options;
  options.enable_cublaslt = true;
  GemmRewriter pass(Capability(), GetToolkitVersion(), options);
  RunAndFilecheckHloRewrite(hlo_text, std::move(pass),
                            R"(

; CHECK-LABEL: ENTRY %{{.*}} ({{.*}}: f32[2,3], {{.*}}: f32[3,4]) -> (f32[8], f32[8]) {
; CHECK-DAG:     [[P0:%[^ ]+]] = f32[2,3]{1,0} parameter(0)
; CHECK-DAG:     [[P1:%[^ ]+]] = f32[3,4]{1,0} parameter(1)
; CHECK:         [[OUT:%[^ ]+]] = (f32[2,4]{1,0}, f32[2,4]{1,0}, s8[{{[0-9]+}}]{0}) custom-call([[P0]], [[P1]]),
; CHECK:           custom_call_target="__cublas$lt$matmul",
; CHECK:           backend_config={
; CHECK-DAG:         "alpha_real":1
; CHECK-DAG:         "alpha_imag":0
; CHECK-DAG:         "beta":0
; CHECK-DAG:         "dot_dimension_numbers":{
; CHECK-DAG:           "lhs_contracting_dimensions":["1"]
; CHECK-DAG:           "rhs_contracting_dimensions":["0"]
; CHECK-DAG:           "lhs_batch_dimensions":[]
; CHECK-DAG:           "rhs_batch_dimensions":[]
; CHECK-DAG:         }
; CHECK-DAG:         "precision_config":{
; CHECK-DAG:           "operand_precision":["DEFAULT","DEFAULT"]
; CHECK-DAG:         }
; CHECK-DAG:         "epilogue":"GELU_AUX"
; CHECK:           }
; CHECK-DAG:     [[GELU_OUTPUT:%[^ ]+]] = f32[2,4]{1,0} get-tuple-element([[OUT]]), index=0
; CHECK-DAG:     [[DOT_OUTPUT:%[^ ]+]] = f32[2,4]{1,0} get-tuple-element([[OUT]]), index=1
; CHECK-DAG:     [[GELU_BITCAST:%[^ ]+]] = f32[8]{0} bitcast([[GELU_OUTPUT]])
; CHECK-DAG:     [[DOT_BITCAST:%[^ ]+]] = f32[8]{0} bitcast([[DOT_OUTPUT]])
; CHECK-DAG:     ROOT [[RESULT:%[^ ]+]] = (f32[8]{0}, f32[8]{0}) tuple([[GELU_BITCAST]], [[DOT_BITCAST]])
      )");
}

TEST_F(CublasLtGemmRewriteTest,
       ApproxGeluActivationWithAuxAndBitcastAndExtraDotUser) {
  if (IsRocm()) {
    GTEST_SKIP() << "TODO: Unsupported blas-lt epilogue on ROCM";
  }
  const char* hlo_text = R"(
HloModule test

ENTRY test {
  x = f32[2,3] parameter(0)
  y = f32[3,4] parameter(1)
  dot = f32[2,4] dot(x, y), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  bitcasted_dot = f32[8] bitcast(dot)
  mul.0 = f32[8] multiply(bitcasted_dot, bitcasted_dot)
  mul.1 = f32[8] multiply(bitcasted_dot, mul.0)
  const.0 = f32[] constant(0.044715)
  bcast.0 = f32[8] broadcast(const.0), dimensions={}
  mul.2 = f32[8] multiply(mul.1, bcast.0)
  add.0 = f32[8] add(bitcasted_dot, mul.2)
  const.1 = f32[] constant(0.797884583)
  bcast.1 = f32[8] broadcast(const.1), dimensions={}
  mul.3 = f32[8] multiply(add.0, bcast.1)
  tanh = f32[8] tanh(mul.3)
  const.2 = f32[] constant(1)
  bcast.2 = f32[8] broadcast(const.2), dimensions={}
  add.2 = f32[8] add(tanh, bcast.2)
  const.3 = f32[] constant(0.5)
  bcast.3 = f32[8] broadcast(const.3), dimensions={}
  mul.4 = f32[8] multiply(add.2, bcast.3)
  mul.5 = f32[8] multiply(bitcasted_dot, mul.4)
  ROOT out = (f32[8], f32[2,4]) tuple(mul.5, dot)
}

)";

  GemmRewriterOptions options;
  options.enable_cublaslt = true;
  GemmRewriter pass(Capability(), GetToolkitVersion(), options);
  RunAndFilecheckHloRewrite(hlo_text, std::move(pass),
                            R"(

; CHECK-LABEL: ENTRY %{{.*}} ({{.*}}: f32[2,3], {{.*}}: f32[3,4]) -> (f32[8], f32[2,4]) {
; CHECK-DAG:     [[P0:%[^ ]+]] = f32[2,3]{1,0} parameter(0)
; CHECK-DAG:     [[P1:%[^ ]+]] = f32[3,4]{1,0} parameter(1)
; CHECK:         [[OUT:%[^ ]+]] = (f32[2,4]{1,0}, f32[2,4]{1,0}, s8[{{[0-9]+}}]{0}) custom-call([[P0]], [[P1]]),
; CHECK:           custom_call_target="__cublas$lt$matmul",
; CHECK:           backend_config={
; CHECK-DAG:         "alpha_real":1
; CHECK-DAG:         "alpha_imag":0
; CHECK-DAG:         "beta":0
; CHECK-DAG:         "dot_dimension_numbers":{
; CHECK-DAG:           "lhs_contracting_dimensions":["1"]
; CHECK-DAG:           "rhs_contracting_dimensions":["0"]
; CHECK-DAG:           "lhs_batch_dimensions":[]
; CHECK-DAG:           "rhs_batch_dimensions":[]
; CHECK-DAG:         }
; CHECK-DAG:         "precision_config":{
; CHECK-DAG:           "operand_precision":["DEFAULT","DEFAULT"]
; CHECK-DAG:         }
; CHECK-DAG:         "epilogue":"GELU_AUX"
; CHECK:           }
; CHECK-DAG:     [[GELU_OUTPUT:%[^ ]+]] = f32[2,4]{1,0} get-tuple-element([[OUT]]), index=0
; CHECK-DAG:     [[DOT_OUTPUT:%[^ ]+]] = f32[2,4]{1,0} get-tuple-element([[OUT]]), index=1
; CHECK-DAG:     [[GELU_BITCAST:%[^ ]+]] = f32[8]{0} bitcast([[GELU_OUTPUT]])
; CHECK-DAG:     ROOT [[RESULT:%[^ ]+]] = (f32[8]{0}, f32[2,4]{1,0}) tuple([[GELU_BITCAST]], [[DOT_OUTPUT]])
      )");
}

TEST_F(CublasLtGemmRewriteTest, VectorBiasThenApproxGeluActivationWithAux) {
  if (IsRocm()) {
    GTEST_SKIP() << "TODO: Unsupported blas-lt epilogue on ROCM";
  }
  const char* hlo_text = R"(
HloModule test

ENTRY test {
  x = f32[2,3] parameter(0)
  y = f32[3,4] parameter(1)
  z = f32[4] parameter(2)
  dot = f32[2,4] dot(x, y), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  z_bcast = f32[2,4] broadcast(z), dimensions={1}
  add = f32[2,4] add(dot, z_bcast)
  mul.0 = f32[2,4] multiply(add, add)
  mul.1 = f32[2,4] multiply(add, mul.0)
  const.0 = f32[] constant(0.044715)
  bcast.0 = f32[2,4] broadcast(const.0), dimensions={}
  mul.2 = f32[2,4] multiply(mul.1, bcast.0)
  add.0 = f32[2,4] add(add, mul.2)
  const.1 = f32[] constant(0.797884583)
  bcast.1 = f32[2,4] broadcast(const.1), dimensions={}
  mul.3 = f32[2,4] multiply(add.0, bcast.1)
  tanh = f32[2,4] tanh(mul.3)
  const.2 = f32[] constant(1)
  bcast.2 = f32[2,4] broadcast(const.2), dimensions={}
  add.2 = f32[2,4] add(tanh, bcast.2)
  const.3 = f32[] constant(0.5)
  bcast.3 = f32[2,4] broadcast(const.3), dimensions={}
  mul.4 = f32[2,4] multiply(add.2, bcast.3)
  mul.5 = f32[2,4] multiply(add, mul.4)
  ROOT out = (f32[2,4], f32[2,4]) tuple(mul.5, add)
}

)";

  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{1e-5, 1e-5}));
  MatchOptimizedHlo(hlo_text,
                    R"(

; CHECK-LABEL: ENTRY %{{.*}} ({{.*}}: f32[2,3], {{.*}}: f32[3,4], {{.*}}: f32[4]) -> (f32[2,4], f32[2,4]) {
; CHECK-DAG:     [[P0:%[^ ]+]] = f32[2,3]{1,0} parameter(0)
; CHECK-DAG:     [[P1:%[^ ]+]] = f32[3,4]{1,0} parameter(1)
; CHECK-DAG:     [[P2:%[^ ]+]] = f32[4]{0} parameter(2)
; CHECK:         [[OUT:%[^ ]+]] = (f32[2,4]{1,0}, f32[2,4]{1,0}, s8[{{[0-9]+}}]{0}) custom-call([[P0]], [[P1]], [[P2]]),
; CHECK:           custom_call_target="__cublas$lt$matmul",
; CHECK:           backend_config={
; CHECK-DAG:         "alpha_real":1
; CHECK-DAG:         "alpha_imag":0
; CHECK-DAG:         "beta":0
; CHECK-DAG:         "dot_dimension_numbers":{
; CHECK-DAG:           "lhs_contracting_dimensions":["1"]
; CHECK-DAG:           "rhs_contracting_dimensions":["0"]
; CHECK-DAG:           "lhs_batch_dimensions":[]
; CHECK-DAG:           "rhs_batch_dimensions":[]
; CHECK-DAG:         }
; CHECK-DAG:         "precision_config":{
; CHECK-DAG:           "operand_precision":["DEFAULT","DEFAULT"]
; CHECK-DAG:         }
; CHECK-DAG:         "epilogue":"BIAS_GELU_AUX"
; CHECK:           }
      )");
}

TEST_F(CublasLtGemmRewriteTest, ApproxGeluActivationBF16) {
  if (IsCuda() &&
      !HasCudaComputeCapability(se::CudaComputeCapability::Ampere())) {
    GTEST_SKIP() << "Padding of GEMM bf16 operands only implemented on "
                    "architectures with bf16 Tensor Cores.";
  }
  const char* hlo_text = R"(
HloModule test

ENTRY test {
  x = bf16[2,3] parameter(0)
  y = bf16[3,4] parameter(1)
  dot = bf16[2,4] dot(x, y), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  mul.0 = bf16[2,4] multiply(dot, dot)
  mul.1 = bf16[2,4] multiply(dot, mul.0)
  const.0 = bf16[] constant(0.044715)
  bcast.0 = bf16[2,4] broadcast(const.0), dimensions={}
  mul.2 = bf16[2,4] multiply(mul.1, bcast.0)
  add.0 = bf16[2,4] add(dot, mul.2)
  const.1 = bf16[] constant(0.797884583)
  bcast.1 = bf16[2,4] broadcast(const.1), dimensions={}
  mul.3 = bf16[2,4] multiply(add.0, bcast.1)
  tanh = bf16[2,4] tanh(mul.3)
  const.2 = bf16[] constant(1)
  bcast.2 = bf16[2,4] broadcast(const.2), dimensions={}
  add.2 = bf16[2,4] add(tanh, bcast.2)
  const.3 = bf16[] constant(0.5)
  bcast.3 = bf16[2,4] broadcast(const.3), dimensions={}
  mul.4 = bf16[2,4] multiply(add.2, bcast.3)
  ROOT out = bf16[2,4] multiply(dot, mul.4)
})";

  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{5e-5, 1e-5}));
  if (IsCuda()) {
    MatchOptimizedHlo(hlo_text, R"(
    ; CHECK-DAG: ENTRY %{{.*}} ({{.*}}: bf16[2,3], {{.*}}: bf16[3,4]) -> bf16[2,4] {
    ; CHECK-DAG:    bf16[8,8]{1,0} pad({{.*}}), padding=0_6x0_5
    ; CHECK-DAG:    bf16[8,8]{1,0} pad({{.*}}), padding=0_5x0_4
      )");
  } else {
    MatchOptimizedHlo(hlo_text, R"(
    ; CHECK-DAG: ENTRY %{{.*}} ({{.*}}: bf16[2,3], {{.*}}: bf16[3,4]) -> bf16[2,4] {
    ; CHECK-DAG:    bf16[2,3]{1,0}
    ; CHECK-DAG:    bf16[3,4]{1,0}
    )");
  }
}

TEST_F(CublasLtGemmRewriteTest, ApproxGeluActivationBitcast) {
  const char* hlo_text = R"(
HloModule test

ENTRY test {
  x = f32[2,3] parameter(0)
  y = f32[3,4] parameter(1)
  dot = f32[2,4] dot(x, y), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  dot_bitcast = f32[2,2,2] bitcast(dot)
  mul.0 = f32[2,2,2] multiply(dot_bitcast, dot_bitcast)
  mul.1 = f32[2,2,2] multiply(dot_bitcast, mul.0)
  const.0 = f32[] constant(0.044715)
  bcast.0 = f32[2,2,2] broadcast(const.0), dimensions={}
  mul.2 = f32[2,2,2] multiply(mul.1, bcast.0)
  add.0 = f32[2,2,2] add(dot_bitcast, mul.2)
  const.1 = f32[] constant(0.797884583)
  bcast.1 = f32[2,2,2] broadcast(const.1), dimensions={}
  mul.3 = f32[2,2,2] multiply(add.0, bcast.1)
  tanh = f32[2,2,2] tanh(mul.3)
  const.2 = f32[] constant(1)
  bcast.2 = f32[2,2,2] broadcast(const.2), dimensions={}
  add.2 = f32[2,2,2] add(tanh, bcast.2)
  const.3 = f32[] constant(0.5)
  bcast.3 = f32[2,2,2] broadcast(const.3), dimensions={}
  mul.4 = f32[2,2,2] multiply(add.2, bcast.3)
  ROOT out = f32[2,2,2] multiply(dot_bitcast, mul.4)
}

)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_text));
  GemmRewriterOptions options;
  options.enable_cublaslt = true;
  GemmRewriter pass(Capability(), GetToolkitVersion(), options);
  TF_ASSERT_OK_AND_ASSIGN(bool changed, this->RunHloPass(&pass, module.get()));
  EXPECT_TRUE(changed);

  EXPECT_THAT(
      module->entry_computation()->root_instruction(),
      GmockMatch(
          m::Bitcast(m::GetTupleElement(
                         m::CustomCall({"__cublas$lt$matmul"},
                                       m::Parameter(0).WithShape(F32, {2, 3}),
                                       m::Parameter(1).WithShape(F32, {3, 4})),
                         0))
              .WithShape(F32, {2, 2, 2})));
}

TEST_F(CublasLtGemmRewriteTest, MatrixBiasF16) {
  const char* hlo_text = R"(
HloModule test

ENTRY test {
  x = f16[8,16] parameter(0)
  y = f16[16,8] parameter(1)
  z = f16[8,8] parameter(2)
  dot_a = f16[8,8] dot(x, y), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  ROOT out = f16[8,8] add(dot_a, z)
}

)";

  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{1e-3, 1e-3}));
  MatchOptimizedHlo(hlo_text,
                    R"(

; CHECK-LABEL: ENTRY %{{.*}} ({{.*}}: f16[8,16], {{.*}}: f16[16,8], {{.*}}: f16[8,8]) -> f16[8,8] {
; CHECK-DAG:     [[P0:%[^ ]+]] = f16[8,16]{1,0} parameter(0)
; CHECK-DAG:     [[P1:%[^ ]+]] = f16[16,8]{1,0} parameter(1)
; CHECK-DAG:     [[P2:%[^ ]+]] = f16[8,8]{1,0} parameter(2)
; CHECK:         [[OUT:%[^ ]+]] = (f16[8,8]{1,0}, s8[{{[0-9]+}}]{0}) custom-call([[P0]], [[P1]], [[P2]]),
; CHECK:           custom_call_target="__cublas$lt$matmul",
; CHECK:           backend_config={
; CHECK-DAG:         "alpha_real":1
; CHECK-DAG:         "alpha_imag":0
; CHECK-DAG:         "beta":1
; CHECK-DAG:         "dot_dimension_numbers":{
; CHECK-DAG:           "lhs_contracting_dimensions":["1"]
; CHECK-DAG:           "rhs_contracting_dimensions":["0"]
; CHECK-DAG:           "lhs_batch_dimensions":[]
; CHECK-DAG:           "rhs_batch_dimensions":[]
; CHECK-DAG:         }
; CHECK-DAG:         "precision_config":{
; CHECK-DAG:           "operand_precision":["DEFAULT","DEFAULT"]
; CHECK-DAG:         }
; CHECK-DAG:         "epilogue":"DEFAULT"
; CHECK:           }
      )");
}

TEST_F(ParameterizedGemmRewriteTest, Simple) {
  const char* hlo_text = R"(
HloModule test

ENTRY test {
  x = f32[2,3] parameter(0)
  y = f32[3,4] parameter(1)
  ROOT dot_a = f32[2,4] dot(x, y), lhs_contracting_dims={1}, rhs_contracting_dims={0}
}

)";

  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{1e-5, 1e-5}));
  MatchOptimizedHlo(hlo_text,
                    R"(
; CHECK-LABEL: ENTRY %{{.*}} ({{.*}}: f32[2,3], {{.*}}: f32[3,4]) -> f32[2,4] {
; CHECK-DAG:     [[P0:%[^ ]+]] = f32[2,3]{1,0} parameter(0)
; CHECK-DAG:     [[P1:%[^ ]+]] = f32[3,4]{1,0} parameter(1)
; CHECK:         [[GEMM:%[^ ]+]] = {{.*}} custom-call([[P0]], [[P1]]),
; CHECK:           custom_call_target="__cublas$lt$matmul",
; CHECK:           backend_config={
; CHECK-DAG:         "alpha_real":1
; CHECK-DAG:         "alpha_imag":0
; CHECK-DAG:         "beta":0
; CHECK-DAG:         "dot_dimension_numbers":{
; CHECK-DAG:           "lhs_contracting_dimensions":["1"]
; CHECK-DAG:           "rhs_contracting_dimensions":["0"]
; CHECK-DAG:           "lhs_batch_dimensions":[]
; CHECK-DAG:           "rhs_batch_dimensions":[]
; CHECK-DAG:         }
; CHECK-DAG:         "precision_config":{
; CHECK-DAG:           "operand_precision":["DEFAULT","DEFAULT"]
; CHECK-DAG:         }
; CHECK-DAG:         "epilogue":"DEFAULT"
; CHECK:           }
)");
}

TEST_F(ParameterizedGemmRewriteTest, SimpleRewrite) {
  const char* hlo_text = R"(
HloModule SimpleGemm

ENTRY AddDotsFunc {
  x = f32[2,3] parameter(0)
  y = f32[3,4] parameter(1)
  ROOT dot_a = f32[2,4] dot(x, y), lhs_contracting_dims={1}, rhs_contracting_dims={0}
}

)";

  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{1e-5, 1e-5}));
  MatchOptimizedHlo(hlo_text,
                    R"(
; CHECK-LABEL: ENTRY %{{.*}} ({{.*}}: f32[2,3], {{.*}}: f32[3,4]) -> f32[2,4] {
; CHECK-DAG:     [[P0:%[^ ]+]] = f32[2,3]{1,0} parameter(0)
; CHECK-DAG:     [[P1:%[^ ]+]] = f32[3,4]{1,0} parameter(1)
; CHECK:         [[GEMM:%[^ ]+]] = {{.*}} custom-call([[P0]], [[P1]]),
; CHECK:           custom_call_target="__cublas$lt$matmul",
; CHECK:           backend_config={
; CHECK-DAG:         "alpha_real":1
; CHECK-DAG:         "alpha_imag":0
; CHECK-DAG:         "beta":0
; CHECK-DAG:         "dot_dimension_numbers":{
; CHECK-DAG:           "lhs_contracting_dimensions":["1"]
; CHECK-DAG:           "rhs_contracting_dimensions":["0"]
; CHECK-DAG:           "lhs_batch_dimensions":[]
; CHECK-DAG:           "rhs_batch_dimensions":[]
; CHECK-DAG:         }
; CHECK-DAG:         "precision_config":{
; CHECK-DAG:           "operand_precision":["DEFAULT","DEFAULT"]
; CHECK-DAG:         }
; CHECK-DAG:         "epilogue":"DEFAULT"
; CHECK:           }
)");
}

TEST_F(ParameterizedGemmRewriteTest, MultipleContractingDims) {
  const char* hlo_text = R"(
HloModule MultipleContractingCheckGemm

ENTRY AddDotsFunc {
  x = f32[3,4,2] parameter(0)
  y = f32[3,4,5] parameter(1)
  ROOT dot_a = f32[2,5] dot(x, y), lhs_contracting_dims={0,1}, rhs_contracting_dims={0,1}, operand_precision={highest,highest}
}

)";

  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{1e-5, 1e-5}));
  MatchOptimizedHlo(hlo_text,
                    R"(
; CHECK-NOT:     copy
;
; CHECK-LABEL: ENTRY %{{.*}} ({{.*}}: f32[3,4,2], {{.*}}: f32[3,4,5]) -> f32[2,5] {
; CHECK-DAG:     [[P0:%[^ ]+]] = f32[3,4,2]{2,1,0} parameter(0)
; CHECK-DAG:     [[P1:%[^ ]+]] = f32[3,4,5]{2,1,0} parameter(1)
; CHECK-DAG:     [[BITCAST0:%[^ ]+]] = f32[2,12]{0,1} bitcast([[P0]])
; CHECK-DAG:     [[BITCAST1:%[^ ]+]] = f32[12,5]{1,0} bitcast([[P1]])
; CHECK:         [[GEMM:%[^ ]+]] = {{.*}} custom-call([[BITCAST0]], [[BITCAST1]]),
; CHECK:           custom_call_target="__cublas$lt$matmul",
; CHECK:           backend_config={
; CHECK-DAG:         "alpha_real":1
; CHECK-DAG:         "alpha_imag":0
; CHECK-DAG:         "beta":0
; CHECK-DAG:         "dot_dimension_numbers":{
; CHECK-DAG:           "lhs_contracting_dimensions":["1"]
; CHECK-DAG:           "rhs_contracting_dimensions":["0"]
; CHECK-DAG:           "lhs_batch_dimensions":[]
; CHECK-DAG:           "rhs_batch_dimensions":[]
; CHECK-DAG:         }
; CHECK-DAG:         "precision_config":{
; CHECK-DAG:           "operand_precision":["HIGHEST","HIGHEST"]
; CHECK-DAG:         }
; CHECK-DAG:         "epilogue":"DEFAULT"
; CHECK:           }
)");
}

TEST_F(ParameterizedGemmRewriteTest, ArgTransposeFoldCheck) {
  const char* hlo_text = R"(
HloModule ArgTransposeFoldGemm

ENTRY AddDotsFunc {
  x = f32[3,2] parameter(0)
  y = f32[3,4] parameter(1)
  x_transposed = f32[2,3] transpose(x), dimensions={1, 0}
  ROOT dot_a = f32[2,4] dot(x_transposed, y), lhs_contracting_dims={1}, rhs_contracting_dims={0}
}

)";

  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{1e-5, 1e-5}));
  MatchOptimizedHlo(hlo_text,
                    R"(
; CHECK-LABEL: ENTRY %{{.*}} ({{.*}}: f32[3,2], {{.*}}: f32[3,4]) -> f32[2,4] {
; CHECK-DAG:     [[P0:%[^ ]+]] = f32[3,2]{1,0} parameter(0)
; CHECK-DAG:     [[P1:%[^ ]+]] = f32[3,4]{1,0} parameter(1)
; CHECK:         [[GEMM:%[^ ]+]] = {{.*}} custom-call([[P0]], [[P1]]),
; CHECK:           custom_call_target="__cublas$lt$matmul",
; CHECK:           backend_config={
; CHECK-DAG:         "alpha_real":1
; CHECK-DAG:         "alpha_imag":0
; CHECK-DAG:         "beta":0
; CHECK-DAG:         "dot_dimension_numbers":{
; CHECK-DAG:           "lhs_contracting_dimensions":["0"]
; CHECK-DAG:           "rhs_contracting_dimensions":["0"]
; CHECK-DAG:           "lhs_batch_dimensions":[]
; CHECK-DAG:           "rhs_batch_dimensions":[]
; CHECK-DAG:         }
; CHECK-DAG:         "precision_config":{
; CHECK-DAG:           "operand_precision":["DEFAULT","DEFAULT"]
; CHECK-DAG:         }
; CHECK-DAG:         "epilogue":"DEFAULT"
; CHECK:           }
)");
}

TEST_F(ParameterizedGemmRewriteTest, BatchedArgRowColTransposeFoldCheck) {
  const char* hlo_text = R"(
HloModule BatchedArgRowColTransposeFoldGemm

ENTRY AddDotsFunc {
  x = f32[5,3,2] parameter(0)
  y = f32[5,3,4] parameter(1)
  x_transposed = f32[5,2,3] transpose(x), dimensions={0, 2, 1}
  ROOT dot_a = f32[5,2,4] dot(x_transposed, y), lhs_contracting_dims={2}, rhs_contracting_dims={1}, lhs_batch_dims={0}, rhs_batch_dims={0}
}

)";

  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{1e-3, 1e-3}));
  MatchOptimizedHlo(hlo_text,
                    R"(
; CHECK-LABEL: ENTRY %{{.*}} ({{.*}}: f32[5,3,2], {{.*}}: f32[5,3,4]) -> f32[5,2,4] {
; CHECK-DAG:     [[P0:%[^ ]+]] = f32[5,3,2]{2,1,0} parameter(0)
; CHECK-DAG:     [[P1:%[^ ]+]] = f32[5,3,4]{2,1,0} parameter(1)
; CHECK:         [[GEMM:%[^ ]+]] = {{.*}} custom-call([[P0]], [[P1]]),
; CHECK:           custom_call_target="__cublas$lt$matmul",
; CHECK:           backend_config={
; CHECK-DAG:         "alpha_real":1
; CHECK-DAG:         "alpha_imag":0
; CHECK-DAG:         "beta":0
; CHECK-DAG:         "dot_dimension_numbers":{
; CHECK-DAG:           "lhs_contracting_dimensions":["1"]
; CHECK-DAG:           "rhs_contracting_dimensions":["1"]
; CHECK-DAG:           "lhs_batch_dimensions":["0"]
; CHECK-DAG:           "rhs_batch_dimensions":["0"]
; CHECK-DAG:         }
; CHECK-DAG:         "precision_config":{
; CHECK-DAG:           "operand_precision":["DEFAULT","DEFAULT"]
; CHECK-DAG:         }
; CHECK-DAG:         "epilogue":"DEFAULT"
; CHECK:           }
)");
}

TEST_F(ParameterizedGemmRewriteTest, BatchRowTransposeFoldCheck) {
  const char* hlo_text = R"(
HloModule BatchRowTransposeFoldCheck

ENTRY AddDotsFunc {
  x = f32[2,5,3] parameter(0)
  y = f32[5,3,4] parameter(1)
  x_transposed = f32[5,2,3] transpose(x), dimensions={1, 0, 2}
  ROOT dot_a = f32[5,2,4] dot(x_transposed, y), lhs_contracting_dims={2}, rhs_contracting_dims={1}, lhs_batch_dims={0}, rhs_batch_dims={0}
}

)";

  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{2.5e-5, 1e-5}));
  MatchOptimizedHlo(hlo_text,
                    R"(
; CHECK-LABEL: ENTRY %{{.*}} ({{.*}}: f32[2,5,3], {{.*}}: f32[5,3,4]) -> f32[5,2,4] {
; CHECK-DAG:     [[P0:%[^ ]+]] = f32[2,5,3]{2,1,0} parameter(0)
; CHECK-DAG:     [[P1:%[^ ]+]] = f32[5,3,4]{2,1,0} parameter(1)
; CHECK:         [[GEMM:%[^ ]+]] = {{.*}} custom-call([[P0]], [[P1]]),
; CHECK:           custom_call_target="__cublas$lt$matmul",
; CHECK:           backend_config={
; CHECK-DAG:         "alpha_real":1
; CHECK-DAG:         "alpha_imag":0
; CHECK-DAG:         "beta":0
; CHECK-DAG:         "dot_dimension_numbers":{
; CHECK-DAG:           "lhs_contracting_dimensions":["2"]
; CHECK-DAG:           "rhs_contracting_dimensions":["1"]
; CHECK-DAG:           "lhs_batch_dimensions":["1"]
; CHECK-DAG:           "rhs_batch_dimensions":["0"]
; CHECK-DAG:         }
; CHECK-DAG:         "precision_config":{
; CHECK-DAG:           "operand_precision":["DEFAULT","DEFAULT"]
; CHECK-DAG:         }
; CHECK-DAG:         "epilogue":"DEFAULT"
; CHECK:           }
)");
}

TEST_F(ParameterizedGemmRewriteTest, BatchFromMinorDimTransposeIsNotFolded) {
  const char* hlo_text = R"(
HloModule BatchFromMinorDimTransposeDoesntFold

ENTRY AddDotsFunc {
  x = f32[3,2,5] parameter(0)
  y = f32[5,3,4] parameter(1)
  x_transposed = f32[5,2,3] transpose(x), dimensions={2, 1, 0}
  ROOT dot_a = f32[5,2,4] dot(x_transposed, y), lhs_contracting_dims={2}, rhs_contracting_dims={1}, lhs_batch_dims={0}, rhs_batch_dims={0}
}

)";

  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{2.5e-5, 1e-5}));
  MatchOptimizedHlo(hlo_text,
                    R"(
; CHECK-LABEL: ENTRY %{{.*}} ({{.*}}: f32[3,2,5], {{.*}}: f32[5,3,4]) -> f32[5,2,4] {
; CHECK-DAG:     [[P0:%[^ ]+]] = f32[3,2,5]{2,1,0} parameter(0)
; CHECK-DAG:     [[FUSION:%[^ ]+]] = f32[5,2,3]{2,1,0} fusion([[P0]])
; CHECK-DAG:     [[P1:%[^ ]+]] = f32[5,3,4]{2,1,0} parameter(1)
; CHECK:         {{[^ ]+}} = {{.*}} custom-call([[FUSION]], [[P1]]), custom_call_target="__cublas$lt$matmul",
; CHECK:           backend_config={
; CHECK-DAG:         "alpha_real":1
; CHECK-DAG:         "alpha_imag":0
; CHECK-DAG:         "beta":0
; CHECK-DAG:         "dot_dimension_numbers":{
; CHECK-DAG:           "lhs_contracting_dimensions":["2"]
; CHECK-DAG:           "rhs_contracting_dimensions":["1"]
; CHECK-DAG:           "lhs_batch_dimensions":["0"]
; CHECK-DAG:           "rhs_batch_dimensions":["0"]
; CHECK-DAG:         }
; CHECK-DAG:         "precision_config":{
; CHECK-DAG:           "operand_precision":["DEFAULT","DEFAULT"]
; CHECK-DAG:         }
; CHECK-DAG:         "epilogue":"DEFAULT"
; CHECK:           }
)");
}

TEST_F(ParameterizedGemmRewriteTest, InstrTransposeFoldCheck) {
  const char* hlo_text = R"(
HloModule InstrTransposeFoldGemm

ENTRY AddDotsFunc {
  x = f32[2,3] parameter(0)
  y = f32[3,4] parameter(1)
  dot_a = f32[2,4] dot(x, y), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  ROOT out = f32[4,2] transpose(dot_a), dimensions={1, 0}
}

)";

  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{1e-5, 1e-5}));
  MatchOptimizedHlo(hlo_text,
                    R"(
; CHECK-LABEL: ENTRY %{{.*}} ({{.*}}: f32[2,3], {{.*}}: f32[3,4]) -> f32[4,2] {
; CHECK-DAG:     [[P1:%[^ ]+]] = f32[3,4]{1,0} parameter(1)
; CHECK-DAG:     [[P0:%[^ ]+]] = f32[2,3]{1,0} parameter(0)
; CHECK:         [[GEMM:%[^ ]+]] = {{.*}} custom-call([[P1]], [[P0]]),
; CHECK:           custom_call_target="__cublas$lt$matmul",
; CHECK:           backend_config={
; CHECK-DAG:         "alpha_real":1
; CHECK-DAG:         "alpha_imag":0
; CHECK-DAG:         "beta":0
; CHECK-DAG:         "dot_dimension_numbers":{
; CHECK-DAG:           "lhs_contracting_dimensions":["0"]
; CHECK-DAG:           "rhs_contracting_dimensions":["1"]
; CHECK-DAG:           "lhs_batch_dimensions":[]
; CHECK-DAG:           "rhs_batch_dimensions":[]
; CHECK-DAG:         }
; CHECK-DAG:         "precision_config":{
; CHECK-DAG:           "operand_precision":["DEFAULT","DEFAULT"]
; CHECK-DAG:         }
; CHECK-DAG:         "epilogue":"DEFAULT"
; CHECK:           }
)");
}

TEST_F(ParameterizedGemmRewriteTest, BatchedInstrLayoutTransposed) {
  const char* hlo_text = R"(
HloModule BatchedInstrLayoutCheck

ENTRY AddDotsFunc {
  x = f32[5,2,3] parameter(0)
  y = f32[5,3,4] parameter(1)
  dot_a = f32[5,2,4] dot(x, y), lhs_contracting_dims={2}, rhs_contracting_dims={1}, lhs_batch_dims={0}, rhs_batch_dims={0}
  ROOT out = f32[2,5,4] transpose(dot_a), dimensions={1, 0, 2}
}

)";

  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{2.5e-5, 1e-5}));
  MatchOptimizedHlo(hlo_text,
                    R"(
; CHECK-LABEL: ENTRY %{{.*}} ({{.*}}: f32[5,2,3], {{.*}}: f32[5,3,4]) -> f32[2,5,4] {
; CHECK-DAG:     [[P0:%[^ ]+]] = f32[5,2,3]{2,1,0} parameter(0)
; CHECK-DAG:     [[P1:%[^ ]+]] = f32[5,3,4]{2,1,0} parameter(1)
; CHECK:         [[GEMM:%[^ ]+]] = {{.*}} custom-call([[P0]], [[P1]]),
; CHECK:           custom_call_target="__cublas$lt$matmul",
; CHECK:           backend_config={
; CHECK-DAG:         "alpha_real":1
; CHECK-DAG:         "alpha_imag":0
; CHECK-DAG:         "beta":0
; CHECK-DAG:         "dot_dimension_numbers":{
; CHECK-DAG:           "lhs_contracting_dimensions":["2"]
; CHECK-DAG:           "rhs_contracting_dimensions":["1"]
; CHECK-DAG:           "lhs_batch_dimensions":["0"]
; CHECK-DAG:           "rhs_batch_dimensions":["0"]
; CHECK-DAG:         }
; CHECK-DAG:         "precision_config":{
; CHECK-DAG:           "operand_precision":["DEFAULT","DEFAULT"]
; CHECK-DAG:         }
; CHECK-DAG:         "epilogue":"DEFAULT"
; CHECK:           }
; CHECK:         ROOT [[OUT:%[^ ]+]] = f32[2,5,4]{2,1,0} bitcast
)");
}

TEST_F(ParameterizedGemmRewriteTest, BatchedInstrLayoutBatchNotInMinorDim) {
  const char* hlo_text = R"(
HloModule BatchedInstrLayoutBatchNotInMinorDim

ENTRY AddDotsFunc {
  x = f32[5,2,3] parameter(0)
  y = f32[5,3,4] parameter(1)
  dot_a = f32[5,2,4] dot(x, y), lhs_contracting_dims={2}, rhs_contracting_dims={1}, lhs_batch_dims={0}, rhs_batch_dims={0}
  ROOT out = f32[2,4,5] transpose(dot_a), dimensions={1, 2, 0}
}

)";

  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{2.5e-5, 1e-5}));
  MatchOptimizedHlo(hlo_text,
                    R"(
; CHECK-LABEL: ENTRY %{{.*}} ({{.*}}: f32[5,2,3], {{.*}}: f32[5,3,4]) -> f32[2,4,5] {
; CHECK-DAG:     [[P0:%[^ ]+]] = f32[5,2,3]{2,1,0} parameter(0)
; CHECK-DAG:     [[P1:%[^ ]+]] = f32[5,3,4]{2,1,0} parameter(1)
; CHECK:         [[GEMM:%[^ ]+]] = {{.*}} custom-call([[P0]], [[P1]]),
; CHECK:           custom_call_target="__cublas$lt$matmul",
; CHECK:           backend_config={
; CHECK-DAG:         "alpha_real":1
; CHECK-DAG:         "alpha_imag":0
; CHECK-DAG:         "beta":0
; CHECK-DAG:         "dot_dimension_numbers":{
; CHECK-DAG:           "lhs_contracting_dimensions":["2"]
; CHECK-DAG:           "rhs_contracting_dimensions":["1"]
; CHECK-DAG:           "lhs_batch_dimensions":["0"]
; CHECK-DAG:           "rhs_batch_dimensions":["0"]
; CHECK-DAG:         }
; CHECK-DAG:         "precision_config":{
; CHECK-DAG:           "operand_precision":["DEFAULT","DEFAULT"]
; CHECK-DAG:         }
; CHECK-DAG:         "epilogue":"DEFAULT"
; CHECK:           }
; CHECK:         ROOT [[OUT:%[^ ]+]] = f32[2,4,5]{2,1,0} [[OP:[^ ]+]]
)");
}

TEST_F(ParameterizedGemmRewriteTest, AlphaSimpleRewrite) {
  const char* hlo_text = R"(
HloModule AlphaSimpleRewrite

ENTRY AddDotsFunc {
  x = f32[2,2] parameter(0)
  y = f32[2,2] parameter(1)
  k = f32[] constant(3.0)
  k_broadcast = f32[2, 2] broadcast(k), dimensions={}
  dot_a = f32[2,2] dot(x, y), lhs_contracting_dims={1}, rhs_contracting_dims={0}, operand_precision={highest,highest}
  ROOT dot_a_multiplied = f32[2, 2] multiply(dot_a, k_broadcast)
}

)";

  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{1e-5, 1e-5}));
  MatchOptimizedHlo(hlo_text,
                    R"(
; CHECK-LABEL: ENTRY %{{.*}} ({{.*}}: f32[2,2], {{.*}}: f32[2,2]) -> f32[2,2] {
; CHECK-DAG:     [[P0:%[^ ]+]] = f32[2,2]{1,0} parameter(0)
; CHECK-DAG:     [[P1:%[^ ]+]] = f32[2,2]{1,0} parameter(1)
; CHECK:         [[GEMM:%[^ ]+]] = {{.*}} custom-call([[P0]], [[P1]]),
; CHECK:           custom_call_target="__cublas$lt$matmul",
; CHECK:           backend_config={
; CHECK-DAG:         "alpha_real":3
; CHECK-DAG:         "alpha_imag":0
; CHECK-DAG:         "beta":0
; CHECK-DAG:         "dot_dimension_numbers":{
; CHECK-DAG:           "lhs_contracting_dimensions":["1"]
; CHECK-DAG:           "rhs_contracting_dimensions":["0"]
; CHECK-DAG:           "lhs_batch_dimensions":[]
; CHECK-DAG:           "rhs_batch_dimensions":[]
; CHECK-DAG:         }
; CHECK-DAG:         "precision_config":{
; CHECK-DAG:           "operand_precision":["HIGHEST","HIGHEST"]
; CHECK-DAG:         }
; CHECK-DAG:         "epilogue":"DEFAULT"
; CHECK:           }
)");
}

TEST_F(ParameterizedGemmRewriteTest, F64C64_CublasLtSupportTest) {
  // This test should fail if gemm rewriter does not correctly rewrite
  // F64/C64 dots to cublas-lt or legacy cublas calls
  {
    const char* hlo_text = R"(
HloModule F64_rewrite

ENTRY AddDotsFunc {
  x = f64[2,2] parameter(0)
  y = f64[2,2] parameter(1)
  k = f64[] constant(3.0)
  k_broadcast = f64[2, 2] broadcast(k), dimensions={}
  dot_a = f64[2,2] dot(x, y), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  ROOT dot_a_multiplied = f64[2, 2] multiply(dot_a, k_broadcast)
}
)";
    EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{1e-4, 1e-5}));
  }
  {
    const char* hlo_text = R"(
HloModule C64_rewrite

ENTRY AddDotsFunc {
  x = c64[2,2] parameter(0)
  y = c64[2,2] parameter(1)
  k = c64[] constant((3.0, 3.0))
  k_broadcast = c64[2, 2] broadcast(k), dimensions={}
  dot_a = c64[2,2] dot(x, y), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  ROOT dot_a_multiplied = c64[2, 2] multiply(dot_a, k_broadcast)
}
)";
    EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{1e-4, 1e-5}));
  }
}

TEST_F(ParameterizedGemmRewriteTest, ComplexAlphaSimpleRewrite) {
  if (IsRocm() && GetDebugOptionsForTest().xla_gpu_enable_cublaslt()) {
    GTEST_SKIP() << "TODO: Unsupported C64 gpublas-lt datatype on ROCM";
  }
  const char* hlo_text = R"(
HloModule ComplexAlphaSimpleRewrite

ENTRY AddDotsFunc {
  x = c64[2,2] parameter(0)
  y = c64[2,2] parameter(1)
  k = c64[] constant((3.0, 3.0))
  k_broadcast = c64[2, 2] broadcast(k), dimensions={}
  dot_a = c64[2,2] dot(x, y), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  ROOT dot_a_multiplied = c64[2, 2] multiply(dot_a, k_broadcast)
}

)";

  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{1e-4, 1e-5}));
  MatchOptimizedHlo(hlo_text,
                    R"(
; CHECK-LABEL: ENTRY %{{.*}} ({{.*}}: c64[2,2], {{.*}}: c64[2,2]) -> c64[2,2] {
; CHECK-DAG:     [[P0:%[^ ]+]] = c64[2,2]{1,0} parameter(0)
; CHECK-DAG:     [[P1:%[^ ]+]] = c64[2,2]{1,0} parameter(1)
; CHECK:         [[GEMM:%[^ ]+]] = {{.*}} custom-call([[P0]], [[P1]]),
; CHECK:           custom_call_target="__cublas$lt$matmul",
; CHECK:           backend_config={
; CHECK-DAG:         "alpha_real":3
; CHECK-DAG:         "alpha_imag":3
; CHECK-DAG:         "beta":0
; CHECK-DAG:         "dot_dimension_numbers":{
; CHECK-DAG:           "lhs_contracting_dimensions":["1"]
; CHECK-DAG:           "rhs_contracting_dimensions":["0"]
; CHECK-DAG:           "lhs_batch_dimensions":[]
; CHECK-DAG:           "rhs_batch_dimensions":[]
; CHECK-DAG:         }
; CHECK-DAG:         "precision_config":{
; CHECK-DAG:           "operand_precision":["DEFAULT","DEFAULT"]
; CHECK-DAG:         }
; CHECK-DAG:         "epilogue":"DEFAULT"
; CHECK:           }
)");
}

TEST_F(ParameterizedGemmRewriteTest, AlphaMultipleUsersNoRewrite) {
  const char* hlo_text = R"(
HloModule AlphaMultipleUsersNoRewrite

ENTRY AddDotsFunc {
  x = f32[2,2] parameter(0)
  y = f32[2,2] parameter(1)
  k = f32[] constant(3.0)
  k_broadcast = f32[2, 2] broadcast(k), dimensions={}
  dot_a = f32[2,2] dot(x, y), lhs_contracting_dims={1}, rhs_contracting_dims={0}, operand_precision={highest,highest}
  dot_a_multiplied = f32[2, 2] multiply(dot_a, k_broadcast)
  ROOT out = f32[2,2] add(dot_a_multiplied, dot_a)
}

)";

  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{1e-5, 1e-5}));
  MatchOptimizedHlo(hlo_text,
                    R"(
; CHECK:    {{[^ ]+}} = {{.*}} custom-call({{[^,]+}}, {{[^)]+}}),
; CHECK:           custom_call_target="__cublas$lt$matmul",
; CHECK:           backend_config={
; CHECK-DAG:         "alpha_real":1
; CHECK-DAG:         "alpha_imag":0
; CHECK-DAG:         "beta":0
; CHECK-DAG:         "dot_dimension_numbers":{
; CHECK-DAG:           "lhs_contracting_dimensions":["1"]
; CHECK-DAG:           "rhs_contracting_dimensions":["0"]
; CHECK-DAG:           "lhs_batch_dimensions":[]
; CHECK-DAG:           "rhs_batch_dimensions":[]
; CHECK-DAG:         }
; CHECK-DAG:         "precision_config":{
; CHECK-DAG:           "operand_precision":["HIGHEST","HIGHEST"]
; CHECK-DAG:         }
; CHECK-DAG:         "epilogue":"DEFAULT"
; CHECK:           }
)");
}

TEST_F(ParameterizedGemmRewriteTest, AlphaVectorNoRewrite) {
  const char* hlo_text = R"(
HloModule AlphaVectorNoRewrite

ENTRY AddDotsFunc {
  x = f32[2,2] parameter(0)
  y = f32[2,2] parameter(1)
  alpha = f32[2] constant({1, 2})
  alpha_broadcast = f32[2,2] broadcast(alpha), dimensions={1}
  dot = f32[2,2] dot(x, y), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  ROOT dot_a_multiplied = f32[2, 2] multiply(dot, alpha_broadcast)
}
)";

  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{1e-5, 1e-5}));
  MatchOptimizedHlo(hlo_text,
                    R"(
; CHECK-LABEL: ENTRY %{{.*}} ({{.*}}: f32[2,2], {{.*}}: f32[2,2]) -> f32[2,2] {
; CHECK-DAG:     [[P0:%[^ ]+]] = f32[2,2]{1,0} parameter(0)
; CHECK-DAG:     [[P1:%[^ ]+]] = f32[2,2]{1,0} parameter(1)
; CHECK:         [[GEMM:%[^ ]+]] = {{.*}} custom-call([[P0]], [[P1]]),
; CHECK:           custom_call_target="__cublas$lt$matmul",
; CHECK:           backend_config={
; CHECK-DAG:         "alpha_real":1
; CHECK-DAG:         "alpha_imag":0
; CHECK-DAG:         "beta":0
; CHECK-DAG:         "dot_dimension_numbers":{
; CHECK-DAG:           "lhs_contracting_dimensions":["1"]
; CHECK-DAG:           "rhs_contracting_dimensions":["0"]
; CHECK-DAG:           "lhs_batch_dimensions":[]
; CHECK-DAG:           "rhs_batch_dimensions":[]
; CHECK-DAG:         }
; CHECK-DAG:         "precision_config":{
; CHECK-DAG:           "operand_precision":["DEFAULT","DEFAULT"]
; CHECK-DAG:         }
; CHECK-DAG:         "epilogue":"DEFAULT"
; CHECK:           }
)");
}

TEST_F(ParameterizedGemmRewriteTest, BF16Gemm) {
  const char* hlo_text = R"(
HloModule bf16gemm

ENTRY bf16gemm {
  %parameter.1 = bf16[12,4]{1,0} parameter(0)
  %parameter.2 = bf16[4,8]{1,0} parameter(1)
  ROOT %dot.8 = bf16[12,8] dot(bf16[12,4] %parameter.1, bf16[4,8] %parameter.2), lhs_contracting_dims={1}, rhs_contracting_dims={0}
}
  )";
  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{1e-5, 1e-5}));

  if (IsRocm()) {
    MatchOptimizedHlo(hlo_text,
                      R"(
; CHECK: {{.*}} custom-call(bf16[12,4]{1,0} {{.*}}, bf16[4,8]{1,0} {{.*}}), custom_call_target="__cublas$lt$matmul"
  )",
                      /*print_operand_shape=*/true);
  } else if (HasCudaComputeCapability(se::CudaComputeCapability::Ampere())) {
    MatchOptimizedHlo(hlo_text,
                      R"(
; CHECK: {{.*}} custom-call(bf16[16,8]{1,0} {{.*}}, bf16[8,8]{1,0} {{.*}}), custom_call_target="__cublas$lt$matmul"
  )",
                      /*print_operand_shape=*/true);
  } else {
    GTEST_SKIP() << "Pre-Ampere casts up bf16 to fp32";
  }
}

TEST_F(ParameterizedGemmRewriteTest, BF16GemmStrided) {
  const char* hlo_text = R"(
HloModule bf16gemm

ENTRY bf16gemm {
  %parameter.1 = bf16[3,3,4] parameter(0)
  %parameter.2 = bf16[3,3,2] parameter(1)
  ROOT %dot.3 = bf16[3,4,2]{2,1,0} dot(bf16[3,3,4]{2,1,0} %parameter.1, bf16[3,3,2]{2,1,0} %parameter.2), lhs_batch_dims={0}, lhs_contracting_dims={1}, rhs_batch_dims={0}, rhs_contracting_dims={1}, operand_precision={highest,highest}
}

  )";
  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{1e-5, 1e-5}));

  if (IsRocm()) {
    MatchOptimizedHlo(hlo_text,
                      R"(
    ; CHECK: {{.*}} custom-call(bf16[3,3,4]{2,1,0} {{.*}}, bf16[3,3,2]{2,1,0} {{.*}}), custom_call_target="__cublas$lt$matmul"
    )",
                      /*print_operand_shape=*/true);
  } else if (HasCudaComputeCapability(se::CudaComputeCapability::Ampere())) {
    MatchOptimizedHlo(hlo_text,
                      R"(
    ; CHECK: {{.*}} custom-call(bf16[3,8,8]{2,1,0} {{.*}}, bf16[3,8,8]{2,1,0} {{.*}}), custom_call_target="__cublas$lt$matmul"
    )",
                      /*print_operand_shape=*/true);
  } else {
    GTEST_SKIP() << "Pre-Ampere casts up bf16 to fp32";
  }
}

TEST_F(ParameterizedGemmRewriteTest, Int8Gemm) {
  const char* hlo_text = R"(
HloModule int8gemm

ENTRY int8gemm {
  %parameter.1 = s8[12,4]{1,0} parameter(0)
  %parameter.2 = s8[4,8]{1,0} parameter(1)
  ROOT %dot.8 = s32[12,8] dot(s8[12,4] %parameter.1, s8[4,8] %parameter.2), lhs_contracting_dims={1}, rhs_contracting_dims={0}
}
  )";
  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{1e-5, 1e-5}));

  if (IsRocm() ||
      HasCudaComputeCapability(se::CudaComputeCapability::Volta())) {
    MatchOptimizedHlo(hlo_text,
                      R"(
; CHECK: {{.*}} custom-call(s8[12,4]{1,0} [[A:%[^ ]+]], s8[4,8]{0,1} [[B:%[^ ]+]]), custom_call_target="__cublas$lt$matmul"
  )",
                      /*print_operand_shape=*/true);
  } else {
    MatchOptimizedHlo(hlo_text,
                      R"(
; CHECK: {{.*}} dot(s32[12,4]{1,0} [[A:%[^ ]+]], s32[4,8]{1,0} [[B:%[^ ]+]]), lhs_contracting_dims={1}, rhs_contracting_dims={0}

  )",
                      /*print_operand_shape=*/true);
  }
}

TEST_F(GemmRewriteTest, Int8GemmRankGreaterThanTwo) {
  const char* hlo_text = R"(
HloModule int8gemm

ENTRY main.4 {
  Arg_0.1 = s8[1,8,2]{2,1,0} parameter(0)
  Arg_1.2 = s8[2,4]{1,0} parameter(1)
  ROOT dot.3 = s32[1,8,4]{2,1,0} dot(Arg_0.1, Arg_1.2),
  lhs_contracting_dims={2}, rhs_contracting_dims={0}
}
  )";

  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{1e-5, 1e-5}));

  DebugOptions debug_options = GetDebugOptionsForTest();
  std::string custom_call_target = debug_options.xla_gpu_enable_cublaslt()
                                       ? "__cublas$lt$matmul"
                                       : "__cublas$gemm";

  if (IsRocm()) {
    // ROCm does not pad Int8 GEMM operands to multiples of 4.
    MatchOptimizedHlo(hlo_text,
                      absl::StrReplaceAll(
                          R"(
; CHECK: {{.*}} custom-call(s8[8,2]{1,0} %{{.*}}, s8[2,4]{0,1} %{{.*}}), custom_call_target="$0"
  )",
                          {{"$0", custom_call_target}}),
                      /*print_operand_shape=*/true);
  } else if (IsCuda()) {
    if (HasCudaComputeCapability(se::CudaComputeCapability::Volta())) {
      MatchOptimizedHlo(hlo_text,
                        absl::StrReplaceAll(
                            R"(
; CHECK: {{.*}} custom-call(s8[8,4]{1,0} %{{.*}}, s8[4,4]{0,1} %{{.*}}), custom_call_target="$0"
  )",
                            {{"$0", custom_call_target}}),
                        /*print_operand_shape=*/true);
    }
  }
}

TEST_F(ParameterizedGemmRewriteTest, Int8GemmNoAlphaRewrite) {
  const char* hlo_text = R"(
HloModule int8gemm

ENTRY int8gemm {
  %parameter.1 = s8[12,4]{1,0} parameter(0)
  %parameter.2 = s8[4,8]{1,0} parameter(1)
  k = s32[] constant(2)
  k_broadcast = s32[12,8] broadcast(k), dimensions={}
  %dot.8 = s32[12,8] dot(s8[12,4] %parameter.1, s8[4,8] %parameter.2), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  ROOT dot_multiplied = s32[12,8] multiply(%dot.8, k_broadcast)
}
  )";
  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{1e-5, 1e-5}));

  if (IsRocm() ||
      HasCudaComputeCapability(se::CudaComputeCapability::Volta())) {
    MatchOptimizedHlo(hlo_text,
                      R"(
; CHECK: {{.*}} custom-call(s8[12,4]{1,0} [[A:%[^ ]+]], s8[4,8]{0,1} [[B:%[^ ]+]]),
; CHECK:           custom_call_target="__cublas$lt$matmul",
; CHECK:           backend_config={
; CHECK-DAG:       "alpha_real":1
; CHECK-DAG:       "alpha_imag":0
  )",
                      /*print_operand_shape=*/true);
  } else {
    MatchOptimizedHlo(hlo_text,
                      R"(
; CHECK: {{.*}} dot(s32[12,4]{1,0} [[A:%[^ ]+]], s32[4,8]{1,0} [[B:%[^ ]+]]), lhs_contracting_dims={1}, rhs_contracting_dims={0}

  )",
                      /*print_operand_shape=*/true);
  }
}

TEST_F(ParameterizedGemmRewriteTest, Int8GemmNoBetaRewrite) {
  const char* hlo_text = R"(
HloModule int8gemm

ENTRY int8gemm {
  %parameter.1 = s8[12,4]{1,0} parameter(0)
  %parameter.2 = s8[4,8]{1,0} parameter(1)
  bias = s32[12,8] parameter(2)
  %dot.8 = s32[12,8] dot(s8[12,4] %parameter.1, s8[4,8] %parameter.2), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  ROOT out = s32[12,8] add(%dot.8, bias)
}
  )";
  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{1e-5, 1e-5}));

  if (IsRocm() ||
      HasCudaComputeCapability(se::CudaComputeCapability::Volta())) {
    MatchOptimizedHlo(hlo_text,
                      R"(
; CHECK: {{.*}} custom-call(s8[12,4]{1,0} [[A:%[^ ]+]], s8[4,8]{0,1} [[B:%[^ ]+]]),
; CHECK:           custom_call_target="__cublas$lt$matmul",
; CHECK:           backend_config={
; CHECK-DAG:       "alpha_real":1
; CHECK-DAG:       "alpha_imag":0
; CHECK-DAG:       "beta":0
  )",
                      /*print_operand_shape=*/true);
  } else {
    MatchOptimizedHlo(hlo_text,
                      R"(
; CHECK: {{.*}} dot(s32[12,4]{1,0} [[A:%[^ ]+]], s32[4,8]{1,0} [[B:%[^ ]+]]), lhs_contracting_dims={1}, rhs_contracting_dims={0}

  )",
                      /*print_operand_shape=*/true);
  }
}

TEST_F(ParameterizedGemmRewriteTest, Int8GemmNotMultipleOfFour) {
  const char* hlo_text = R"(
HloModule int8gemm

ENTRY int8gemm {
  %parameter.1 = s8[13,4]{1,0} parameter(0)
  %parameter.2 = s8[4,9]{1,0} parameter(1)
  ROOT %dot.9 = s32[13,9] dot(s8[13,4] %parameter.1, s8[4,9] %parameter.2), lhs_contracting_dims={1}, rhs_contracting_dims={0}
}
  )";
  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{1e-5, 1e-5}));

  DebugOptions debug_options = GetDebugOptionsForTest();
  std::string custom_call_target = debug_options.xla_gpu_enable_cublaslt()
                                       ? "__cublas$lt$matmul"
                                       : "__cublas$gemm";

  if (IsRocm()) {
    // ROCm does not pad Int8 GEMM operands to multiples of 4.
    MatchOptimizedHlo(hlo_text,
                      absl::StrReplaceAll(
                          R"(
; CHECK: {{.*}} custom-call(s8[13,4]{1,0} [[A:%[^ ]+]], s8[4,9]{0,1} [[B:%[^ ]+]]), custom_call_target="$0"
  )",
                          {{"$0", custom_call_target}}),
                      /*print_operand_shape=*/true);
  } else if (IsCuda()) {
    if (HasCudaComputeCapability(se::CudaComputeCapability::Volta())) {
      MatchOptimizedHlo(hlo_text,
                        absl::StrReplaceAll(
                            R"(
; CHECK: {{.*}} custom-call(s8[16,4]{1,0} [[A:%[^ ]+]], s8[4,12]{0,1} [[B:%[^ ]+]]), custom_call_target="$0"
  )",
                            {{"$0", custom_call_target}}),
                        /*print_operand_shape=*/true);
    } else {
      MatchOptimizedHlo(hlo_text,
                        R"(
; CHECK: {{.*}} dot(s32[13,4]{1,0} [[A:%[^ ]+]], s32[4,9]{1,0} [[B:%[^ ]+]]), lhs_contracting_dims={1}, rhs_contracting_dims={0}

  )",
                        /*print_operand_shape=*/true);
    }
  }
}

TEST_F(ParameterizedGemmRewriteTest, GemmTypeCombinationCheck) {
  std::vector<std::tuple<absl::string_view, absl::string_view, bool>>
      type_combinations = {{"s8", "s8", true},
                           {"s32", "s32", true},
                           {"bf16", "bf16", true},
                           {"f16", "f16", true},
                           {"f32", "f32", true},
                           {"f64", "f64", true},
                           {"c64", "c64", true},
                           {"c128", "c128", true},
                           // mix type gemm
                           {"s8", "s32", true},
                           {"f16", "f32", true},
                           {"bf16", "f32", true}};

  if (IsCuda()) {
    // cuBLAS and cuBLASLt both support s8 x s8 -> f32 GEMM.
    type_combinations.push_back({"s8", "f32", true});
  } else if (IsRocm()) {
    // Neither rocBLAS nor hipblasLt supports s8 x s8 -> f32 GEMM.
    type_combinations.push_back({"s8", "f32", false});
  }

  if (IsRocm() ||
      HasCudaComputeCapability(se::CudaComputeCapability::Ampere())) {
    // For compute capabilities before Ampere, we may do upcasting, so it
    // would be impossible for this test to fail. That is why we only add these
    // cases when the compute capability is at least Volta.
    std::vector<std::tuple<absl::string_view, absl::string_view, bool>>
        more_type_combinations = {
            {"s8", "bf16", false},  {"s8", "f16", false},
            {"s8", "f64", false},   {"s8", "c64", false},
            {"s8", "c128", false},

            {"s32", "f32", false},  {"s32", "f64", false},
            {"s32", "c64", false},  {"s32", "c128", false},

            {"f16", "bf16", false}, {"f16", "f64", false},
            {"f16", "c64", false},  {"f16", "c128", false},

            {"bf16", "f16", false}, {"bf16", "f64", false},
            {"bf16", "c64", false}, {"bf16", "c128", false},

            {"f32", "f64", false},  {"f32", "c64", false},
            {"f32", "c128", false},

            {"f64", "c64", false},  {"f64", "c128", false},
        };
    type_combinations.insert(type_combinations.end(),
                             more_type_combinations.begin(),
                             more_type_combinations.end());
  }

  for (const auto& type_combination : type_combinations) {
    absl::flat_hash_map<absl::string_view, absl::string_view> replacements;
    replacements["<<ABType>>"] = std::get<0>(type_combination);
    replacements["<<DType>>"] = std::get<1>(type_combination);
    const char* hlo_template = R"(
  HloModule type_combo

  ENTRY type_combo {
    %parameter.1 = <<ABType>>[4,4]{1,0} parameter(0)
    %parameter.2 = <<ABType>>[4,4]{1,0} parameter(1)
    ROOT %dot = <<DType>>[4,4] dot(%parameter.1, %parameter.2), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  }
    )";
    const auto hlo_text = absl::StrReplaceAll(hlo_template, replacements);
    if (std::get<2>(type_combination)) {
      EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{1e-3, 1e-3}));
    } else {
      EXPECT_FALSE(RunAndCompare(hlo_text, ErrorSpec{1e-3, 1e-3}));
    }
  }
}

TEST_F(ParameterizedGemmRewriteTest, DoNotUpconvertOutput) {
  const char* hlo_text = R"(
HloModule test

ENTRY main {
  param_0 = f16[240,88]{1,0} parameter(0)
  param_1 = f16[88,4]{1,0} parameter(1)
  dot = f16[240,4]{1,0} dot(param_0, param_1), lhs_contracting_dims={1}, rhs_contracting_dims={0}, operand_precision={highest,highest}
  constant_255 = f16[] constant(255)
  broadcast = f16[240,4]{1,0} broadcast(constant_255), dimensions={}
  multiply = f16[240,4]{1,0} multiply(dot, broadcast)
  ROOT result = f32[240,4]{1,0} convert(multiply)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_text));
  GemmRewriterOptions options;
  options.enable_cublaslt = true;
  GemmRewriter pass(Capability(), GetToolkitVersion(), options);
  TF_ASSERT_OK_AND_ASSIGN(bool changed, this->RunHloPass(&pass, module.get()));
  EXPECT_TRUE(changed);

  // input fp16 and output fp32 combination is supported by legacy cublas and
  // cublasLt, expect GemmRewriter to fuse the convert into gemm.
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::Convert(
                  m::GetTupleElement(m::CustomCall({CustomCallTarget()}), 0))));
}

TEST_F(ParameterizedGemmRewriteTest, UnsupportedMixTypeGemm) {
  const char* hlo_text = R"(
HloModule test

ENTRY main {
  param_0 = f32[240,88]{1,0} parameter(0)
  param_1 = f32[88,4]{1,0} parameter(1)
  dot = f32[240,4]{1,0} dot(param_0, param_1), lhs_contracting_dims={1}, rhs_contracting_dims={0}, operand_precision={highest,highest}
  constant_255 = f32[] constant(255)
  broadcast = f32[240,4]{1,0} broadcast(constant_255), dimensions={}
  multiply = f32[240,4]{1,0} multiply(dot, broadcast)
  ROOT result = u8[240,4]{1,0} convert(multiply)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_text));
  GemmRewriterOptions options;
  options.enable_cublaslt = true;
  GemmRewriter pass(Capability(), GetToolkitVersion(), options);
  TF_ASSERT_OK_AND_ASSIGN(bool changed, this->RunHloPass(&pass, module.get()));
  EXPECT_TRUE(changed);

  // u8 is not supported by legacy cublas and cublasLt, expect
  // GemmRewriter to not fuse the convert into gemm.
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::Convert(
                  m::GetTupleElement(m::CustomCall({CustomCallTarget()}), 0))));
}

TEST_F(ParameterizedGemmRewriteTest, CheckIsGemmAliasedBeforeFusion) {
  const char* hlo_text = R"(
HloModule test

ENTRY main {
  Arg_0.1 = f16[8,16]{1,0} parameter(0)
  Arg_1.2 = f16[16,32]{1,0} parameter(1)
  dot.8 = f16[8,32]{1,0} dot(Arg_0.1, Arg_1.2), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  Arg_2.3 = f16[8,32]{1,0} parameter(2)
  constant.5 = f16[] constant(1)
  broadcast.6 = f16[8,32]{1,0} broadcast(constant.5), dimensions={}
  add.7 = f16[8,32]{1,0} add(Arg_2.3, broadcast.6)
  add.9 = f16[8,32]{1,0} add(dot.8, add.7)
  convert.10 = f32[8,32]{1,0} convert(add.9)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_text));
  GemmRewriterOptions options;
  options.enable_cublaslt = true;
  GemmRewriter pass(Capability(), GetToolkitVersion(), options);
  TF_ASSERT_OK_AND_ASSIGN(bool changed, this->RunHloPass(&pass, module.get()));
  EXPECT_TRUE(changed);

  // input fp16 and output fp32 combination is supported by legacy cublas and
  // cublasLt, but gemm output is already aliased with one of the input expect
  // GemmRewriter to not fuse the convert into gemm.
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::Convert(
                  m::GetTupleElement(m::CustomCall({CustomCallTarget()}), 0))));
}

class SmallDotGemmRewriteTest : public GemmRewriteTest {
 public:
  DebugOptions GetDebugOptionsForTest() const override {
    DebugOptions debug_options = GemmRewriteTest::GetDebugOptionsForTest();
    debug_options.set_xla_gpu_gemm_rewrite_size_threshold(100);
    return debug_options;
  }
};

TEST_F(SmallDotGemmRewriteTest, SkipSmallMatrixMultiplicationRewrite) {
  const char* hlo_text = R"(
HloModule SkipSmallMatrixRewrite

ENTRY DotFunc {
  x = f32[3,3] parameter(0)
  y = f32[3,3] parameter(1)
  ROOT out = f32[3,3] dot(x, y), lhs_contracting_dims={1}, rhs_contracting_dims={0}
}
)";

  MatchOptimizedHlo(hlo_text,
                    R"(
; CHECK-LABEL: ENTRY %{{.*}} ({{.*}}: f32[3,3], {{.*}}: f32[3,3]) -> f32[3,3] {
; CHECK-DAG:     [[P0:%[^ ]+]] = f32[3,3]{1,0} parameter(0)
; CHECK-DAG:     [[P1:%[^ ]+]] = f32[3,3]{1,0} parameter(1)
; CHECK:         ROOT {{[^ ]+}} = f32[3,3]{1,0} fusion([[P0]], [[P1]]), kind=kLoop
)");
}

TEST_F(SmallDotGemmRewriteTest, LargeMatrixMultiplicationIsRewritten) {
  const char* hlo_text = R"(
HloModule SkipSmallMatrixRewrite

ENTRY DotFunc {
  x = f32[8,8] parameter(0)
  y = f32[8,8] parameter(1)
  ROOT out = f32[8,8] dot(x, y), lhs_contracting_dims={1}, rhs_contracting_dims={0}
}
)";

  MatchOptimizedHlo(hlo_text,
                    R"(
; CHECK-LABEL: ENTRY %{{.*}} ({{.*}}: f32[8,8], {{.*}}: f32[8,8]) -> f32[8,8] {
; CHECK-DAG:     [[P0:%[^ ]+]] = f32[8,8]{1,0} parameter(0)
; CHECK-DAG:     [[P1:%[^ ]+]] = f32[8,8]{1,0} parameter(1)
; CHECK:         {{[^ ]+}} = {{.*}} custom-call([[P0]], [[P1]])
)");
}

TEST_F(SmallDotGemmRewriteTest, RewriteForALG_BF16_BF16_F32) {
  if (!HasCudaComputeCapability(se::CudaComputeCapability::Ampere())) {
    GTEST_SKIP()
        << "There is no autotuning starting with the Nvidia Ampere generation";
  }

  const char* hlo_text = R"(
    HloModule RewriteForALG_BF16_BF16_F32

    ENTRY DotFunc {
      x = f32[1024,1024] parameter(0)
      y = f32[1024,1024] parameter(1)
      ROOT out = f32[1024,1024] dot(x, y),
        algorithm=dot_bf16_bf16_f32,
        lhs_contracting_dims={1},
        rhs_contracting_dims={0}
    }
  )";

  MatchOptimizedHlo(hlo_text,
                    R"(
; CHECK-LABEL: ENTRY %{{.*}} ({{.*}}: f32[1024,1024], {{.*}}: f32[1024,1024]) -> f32[1024,1024] {
; CHECK-DAG:     [[P0:%[^ ]+]] = f32[1024,1024]{1,0} parameter(0)
; CHECK-DAG:     [[P1:%[^ ]+]] = f32[1024,1024]{1,0} parameter(1)
; CHECK:        [[GEMM:%[^ ]+]] = {{.*}} custom-call({{.*}}), custom_call_target="__cublas${{gemm|lt\$matmul}}", {{.*}},"algorithm":"ALG_UNSET"
)");
}

TEST_F(GemmRewriteTest, RewriterReportsChangeWhenAddingWorkspace) {
  const char* hlo_text = R"(
HloModule module

ENTRY main {
  p0 = f32[2,2] parameter(0)
  p1 = f32[2,2] parameter(1)
  ROOT gemm = f32[2,2] custom-call(p0, p1),
      custom_call_target="__cublas$gemm",
      backend_config="{\"gemm_backend_config\":{\"alpha_real\":1,\"alpha_imag\":0,\"beta\":0,\"dot_dimension_numbers\":{\"lhs_contracting_dimensions\":[\"1\"],\"rhs_contracting_dimensions\":[\"0\"],\"lhs_batch_dimensions\":[],\"rhs_batch_dimensions\":[]},\"precision_config\":{\"operand_precision\":[\"DEFAULT\",\"DEFAULT\"]},\"epilogue\":\"DEFAULT\"}}"
}
)";

  auto module_status = ParseAndReturnVerifiedModule(hlo_text);
  ASSERT_TRUE(module_status.ok());
  auto module = std::move(module_status.value());
  GemmRewriter pass(se::CudaComputeCapability{},
                    stream_executor::SemanticVersion{0, 0, 0});
  auto changed_status = pass.Run(module.get());
  ASSERT_TRUE(changed_status.ok());
  EXPECT_TRUE(changed_status.value());

  EXPECT_TRUE(RunFileCheck(module->ToString(), R"(
    // CHECK: %[[CC:.*]] = (f32[2,2]{1,0}, s8[{{[0-9]+}}]{0}) custom-call
    // CHECK: ROOT %{{.*}} = f32[2,2]{1,0} get-tuple-element(%[[CC]]), index=0
  )")
                  .value());
}

TEST_F(GemmRewriteTest, SkipNonFusibleComputations) {
  const char* hlo_text = R"(
HloModule module

reducer {
  p0 = f32[] parameter(0)
  p1 = f32[] parameter(1)
  lhs = f32[2,2] broadcast(p0), dimensions={}
  rhs = f32[2,2] broadcast(p1), dimensions={}
  dot = f32[2,2] dot(lhs, rhs), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  slice = f32[1,1] slice(dot), slice={[0:1:1], [0:1:1]}
  ROOT reshape = f32[] reshape(slice)
}

ENTRY main {
  p0 = f32[10] parameter(0)
  zero = f32[] constant(0.0)
  ROOT reduce = f32[] reduce(p0, zero), dimensions={0}, to_apply=reducer
}
)";
  // Expect no change.
  RunAndFilecheckHloRewrite(
      hlo_text,
      GemmRewriter(se::CudaComputeCapability{},
                   stream_executor::SemanticVersion{0, 0, 0}),
      std::nullopt);
}

TEST_F(GemmRewriteTest, UnsupportedGemmFusionF16) {
  const char* hlo_text = R"(
HloModule module

ENTRY main {
  p0 = f16[100,200,300,512]{1,3,2,0} parameter(0)
  p1 = f16[100,400,300,512]{1,3,2,0} parameter(1)
  ROOT dot = f16[100,300,200,400]{3,2,1,0} dot(p0, p1),
      lhs_batch_dims={0,2}, lhs_contracting_dims={3},
      rhs_batch_dims={0,2}, rhs_contracting_dims={3}
}
)";

  auto module_status = ParseAndReturnVerifiedModule(hlo_text);
  ASSERT_TRUE(module_status.ok());
  auto module = std::move(module_status.value());

  GemmRewriter pass(se::CudaComputeCapability{},
                    stream_executor::SemanticVersion{0, 0, 0});
  auto changed_status = pass.Run(module.get());
  ASSERT_TRUE(changed_status.ok());
  EXPECT_TRUE(changed_status.value());

  HloVerifier verifier(/*layout_sensitive=*/false,
                       /*allow_mixed_precision=*/false);
  auto verify_status = verifier.Run(module.get());
  EXPECT_TRUE(verify_status.ok());
}

}  // namespace
}  // namespace gpu
}  // namespace xla

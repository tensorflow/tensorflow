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

#include "xla/service/gpu/transforms/gemm_rewriter.h"

#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_replace.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/error_spec.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/testlib/filecheck.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/executable.h"
#include "xla/service/gpu/gpu_executable.h"
#include "xla/service/gpu/tests/gpu_codegen_test.h"
#include "xla/service/gpu/transforms/gemm_rewriter_test_lib.h"
#include "xla/service/hlo_module_config.h"
#include "xla/service/pattern_matcher.h"
#include "xla/service/pattern_matcher_gmock.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/device_memory_allocator.h"
#include "xla/stream_executor/semantic_version.h"
#include "xla/stream_executor/stream_executor_memory_allocator.h"
#include "xla/test.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/xla.pb.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace gpu {

namespace {

namespace m = ::xla::match;

using GemmRewriteTest = GemmRewriteTestBase;

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

TEST_F(GemmRewriteTest, TestBatchedAutotuning) {
  if (HasCudaComputeCapability(se::CudaComputeCapability::Ampere())) {
    GTEST_SKIP()
        << "There is no autotuning starting with the Nvidia Ampere generation";
  }

  const char* hlo_text = R"(
HloModule ComplexDotMultipleNonContracting

ENTRY %test {
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

  ErrorSpec error_spec = [&] {
    DebugOptions debug_options = GetDebugOptionsForTest();
    if (debug_options.xla_gpu_enable_cublaslt()) {
      return ErrorSpec{1e-3, 1e-3};
    } else {
      return ErrorSpec{1e-3, 1e-3};
    }
  }();

  auto get_module = [&]() {
    HloModuleConfig config;
    DebugOptions debug_options = GetDebugOptionsForTest();
    debug_options.set_xla_gpu_exclude_nondeterministic_ops(true);
    config.set_debug_options(debug_options);
    return ParseAndReturnVerifiedModule(hlo_text, config);
  };

  se::StreamExecutorMemoryAllocator allocator(
      backend().default_stream_executor());
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HloModule> optimized_module,
      backend().compiler()->RunHloPasses(
          *get_module(), backend().default_stream_executor(), &allocator));

  absl::StatusOr<bool> filecheck_result =
      RunFileCheck(optimized_module->ToString(),
                   R"(
; CHECK:    custom_call_target="__cublas${{(lt\$matmul|gemm)}}"
    )");
  TF_ASSERT_OK(filecheck_result.status());
  EXPECT_TRUE(filecheck_result.value());
  EXPECT_TRUE(RunAndCompare(*get_module(), error_spec));
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
  // CHECK: custom_call_target="__cublas$gemm"
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

using ParameterizedGemmRewriteTest = ParameterizedGemmRewriteTestBase;

TEST_P(ParameterizedGemmRewriteTest, Simple) {
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
; CHECK-LABEL: ENTRY %test ({{.*}}: f32[2,3], {{.*}}: f32[3,4]) -> f32[2,4] {
; CHECK-NEXT:    [[P0:%[^ ]+]] = f32[2,3]{1,0} parameter(0)
; CHECK-NEXT:    [[P1:%[^ ]+]] = f32[3,4]{1,0} parameter(1)
; CHECK-NEXT:    [[GEMM:%[^ ]+]] = {{.*}} custom-call([[P0]], [[P1]]),
; CHECK:           custom_call_target="<<CUBLAS_CUSTOM_CALL_TARGET_PLACEHOLDER>>",
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

TEST_P(ParameterizedGemmRewriteTest, SimpleRewrite) {
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
; CHECK-LABEL: ENTRY %AddDotsFunc ({{.*}}: f32[2,3], {{.*}}: f32[3,4]) -> f32[2,4] {
; CHECK-NEXT:    [[P0:%[^ ]+]] = f32[2,3]{1,0} parameter(0)
; CHECK-NEXT:    [[P1:%[^ ]+]] = f32[3,4]{1,0} parameter(1)
; CHECK-NEXT:    [[GEMM:%[^ ]+]] = {{.*}} custom-call([[P0]], [[P1]]),
; CHECK:           custom_call_target="<<CUBLAS_CUSTOM_CALL_TARGET_PLACEHOLDER>>",
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

TEST_P(ParameterizedGemmRewriteTest, MultipleContractingDims) {
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
; CHECK-LABEL: ENTRY %AddDotsFunc ({{.*}}: f32[3,4,2], {{.*}}: f32[3,4,5]) -> f32[2,5] {
; CHECK-NEXT:    [[P0:%[^ ]+]] = f32[3,4,2]{2,1,0} parameter(0)
; CHECK-DAG:     [[P1:%[^ ]+]] = f32[3,4,5]{2,1,0} parameter(1)
; CHECK-DAG:     [[BITCAST0:%[^ ]+]] = f32[2,12]{0,1} bitcast([[P0]])
; CHECK-DAG:     [[BITCAST1:%[^ ]+]] = f32[12,5]{1,0} bitcast([[P1]])
; CHECK-NEXT:    [[GEMM:%[^ ]+]] = {{.*}} custom-call([[BITCAST0]], [[BITCAST1]]),
; CHECK:           custom_call_target="<<CUBLAS_CUSTOM_CALL_TARGET_PLACEHOLDER>>",
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

TEST_P(ParameterizedGemmRewriteTest, ArgTransposeFoldCheck) {
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
; CHECK-LABEL: ENTRY %AddDotsFunc ({{.*}}: f32[3,2], {{.*}}: f32[3,4]) -> f32[2,4] {
; CHECK-NEXT:    [[P0:%[^ ]+]] = f32[3,2]{1,0} parameter(0)
; CHECK-NEXT:    [[P1:%[^ ]+]] = f32[3,4]{1,0} parameter(1)
; CHECK-NEXT:    [[GEMM:%[^ ]+]] = {{.*}} custom-call([[P0]], [[P1]]),
; CHECK:           custom_call_target="<<CUBLAS_CUSTOM_CALL_TARGET_PLACEHOLDER>>",
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

TEST_P(ParameterizedGemmRewriteTest, BatchedArgRowColTransposeFoldCheck) {
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
; CHECK-LABEL: ENTRY %AddDotsFunc ({{.*}}: f32[5,3,2], {{.*}}: f32[5,3,4]) -> f32[5,2,4] {
; CHECK-NEXT:    [[P0:%[^ ]+]] = f32[5,3,2]{2,1,0} parameter(0)
; CHECK-NEXT:    [[P1:%[^ ]+]] = f32[5,3,4]{2,1,0} parameter(1)
; CHECK-NEXT:    [[GEMM:%[^ ]+]] = {{.*}} custom-call([[P0]], [[P1]]),
; CHECK:           custom_call_target="<<CUBLAS_CUSTOM_CALL_TARGET_PLACEHOLDER>>",
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

TEST_P(ParameterizedGemmRewriteTest, BatchRowTransposeFoldCheck) {
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
; CHECK-LABEL: ENTRY %AddDotsFunc ({{.*}}: f32[2,5,3], {{.*}}: f32[5,3,4]) -> f32[5,2,4] {
; CHECK-NEXT:    [[P0:%[^ ]+]] = f32[2,5,3]{2,1,0} parameter(0)
; CHECK-NEXT:    [[P1:%[^ ]+]] = f32[5,3,4]{2,1,0} parameter(1)
; CHECK-NEXT:    [[GEMM:%[^ ]+]] = {{.*}} custom-call([[P0]], [[P1]]),
; CHECK:           custom_call_target="<<CUBLAS_CUSTOM_CALL_TARGET_PLACEHOLDER>>",
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

TEST_P(ParameterizedGemmRewriteTest, BatchFromMinorDimTransposeIsNotFolded) {
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
; CHECK-LABEL: ENTRY %AddDotsFunc ({{.*}}: f32[3,2,5], {{.*}}: f32[5,3,4]) -> f32[5,2,4] {
; CHECK-NEXT:    [[P0:%[^ ]+]] = f32[3,2,5]{2,1,0} parameter(0)
; CHECK-DAG:     [[P1:%[^ ]+]] = f32[5,3,4]{2,1,0} parameter(1)
; CHECK-DAG:     [[FUSION:%[^ ]+]] = f32[5,2,3]{2,1,0} transpose([[P0]])
; CHECK-NEXT:    [[GEMM:%[^ ]+]] = {{.*}} custom-call([[FUSION]], [[P1]]),
; CHECK:           custom_call_target="<<CUBLAS_CUSTOM_CALL_TARGET_PLACEHOLDER>>",
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

TEST_P(ParameterizedGemmRewriteTest, LargeBatch) {
  const char* hlo_text = R"(
HloModule BatchedArgRowColTransposeFoldGemm

ENTRY AddDotsFunc {
  x = f32[20000,4,3,2] parameter(0)
  y = f32[20000,4,3,4] parameter(1)
  ROOT dot_a = f32[20000,4,2,4] dot(x, y), lhs_contracting_dims={2}, rhs_contracting_dims={2}, lhs_batch_dims={0,1}, rhs_batch_dims={0,1}
}

)";

  // Batch sizes larger than 2^16-1 are not supported by cublasLt. Ensure that
  // the custom_call_target is __cublas$gemm.
  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{1e-3, 1e-3}));
  MatchOptimizedHlo(hlo_text,
                    R"(
; CHECK-LABEL: ENTRY %AddDotsFunc ({{.*}}: f32[20000,4,3,2], {{.*}}: f32[20000,4,3,4]) -> f32[20000,4,2,4] {
; CHECK:    [[P0:%[^ ]+]] = f32[20000,4,3,2]{3,2,1,0} parameter(0)
; CHECK:    [[BC0:%[^ ]+]] = f32[80000,3,2]{2,1,0} bitcast([[P0]])
; CHECK:    [[P1:%[^ ]+]] = f32[20000,4,3,4]{3,2,1,0} parameter(1)
; CHECK:    [[BC1:%[^ ]+]] = f32[80000,3,4]{2,1,0} bitcast([[P1]])
; CHECK:    [[GEMM:%[^ ]+]] = (f32[80000,2,4]{2,1,0}, s8[{{[0-9]+}}]{0}) custom-call([[BC0]], [[BC1]]),
; CHECK:           custom_call_target="__cublas$gemm",
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
; CHECK:           }
; CHECK:   [[OUT:%[^ ]+]] = f32[80000,2,4]{2,1,0} get-tuple-element([[GEMM]]), index=0
; CHECK:   ROOT {{[^ ]+}} = f32[20000,4,2,4]{3,2,1,0} bitcast([[OUT]])
)");
}

TEST_P(ParameterizedGemmRewriteTest, InstrTransposeFoldCheck) {
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
; CHECK-LABEL: ENTRY %AddDotsFunc ({{.*}}: f32[2,3], {{.*}}: f32[3,4]) -> f32[4,2] {
; CHECK-NEXT:    [[P1:%[^ ]+]] = f32[3,4]{1,0} parameter(1)
; CHECK-NEXT:    [[P0:%[^ ]+]] = f32[2,3]{1,0} parameter(0)
; CHECK-NEXT:    [[GEMM:%[^ ]+]] = {{.*}} custom-call([[P1]], [[P0]]),
; CHECK:           custom_call_target="<<CUBLAS_CUSTOM_CALL_TARGET_PLACEHOLDER>>",
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

TEST_P(ParameterizedGemmRewriteTest, BatchedInstrLayoutTransposed) {
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
; CHECK-LABEL: ENTRY %AddDotsFunc ({{.*}}: f32[5,2,3], {{.*}}: f32[5,3,4]) -> f32[2,5,4] {
; CHECK-NEXT:    [[P0:%[^ ]+]] = f32[5,2,3]{2,1,0} parameter(0)
; CHECK-NEXT:    [[P1:%[^ ]+]] = f32[5,3,4]{2,1,0} parameter(1)
; CHECK-NEXT:    [[GEMM:%[^ ]+]] = {{.*}} custom-call([[P0]], [[P1]]),
; CHECK:           custom_call_target="<<CUBLAS_CUSTOM_CALL_TARGET_PLACEHOLDER>>",
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

TEST_P(ParameterizedGemmRewriteTest, BatchedInstrLayoutBatchNotInMinorDim) {
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
; CHECK-LABEL: ENTRY %AddDotsFunc ({{.*}}: f32[5,2,3], {{.*}}: f32[5,3,4]) -> f32[2,4,5] {
; CHECK-NEXT:    [[P0:%[^ ]+]] = f32[5,2,3]{2,1,0} parameter(0)
; CHECK-NEXT:    [[P1:%[^ ]+]] = f32[5,3,4]{2,1,0} parameter(1)
; CHECK-NEXT:    [[GEMM:%[^ ]+]] = {{.*}} custom-call([[P0]], [[P1]]),
; CHECK:           custom_call_target="<<CUBLAS_CUSTOM_CALL_TARGET_PLACEHOLDER>>",
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

TEST_P(ParameterizedGemmRewriteTest, AlphaSimpleRewrite) {
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
; CHECK-LABEL: ENTRY %AddDotsFunc ({{.*}}: f32[2,2], {{.*}}: f32[2,2]) -> f32[2,2] {
; CHECK-NEXT:    [[P0:%[^ ]+]] = f32[2,2]{1,0} parameter(0)
; CHECK-NEXT:    [[P1:%[^ ]+]] = f32[2,2]{1,0} parameter(1)
; CHECK-NEXT:    [[GEMM:%[^ ]+]] = {{.*}} custom-call([[P0]], [[P1]]),
; CHECK:           custom_call_target="<<CUBLAS_CUSTOM_CALL_TARGET_PLACEHOLDER>>",
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

TEST_P(ParameterizedGemmRewriteTest, F64C64_CublasLtSupportTest) {
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

TEST_P(ParameterizedGemmRewriteTest, ComplexAlphaSimpleRewrite) {
  if (!IsCuda() && GetDebugOptionsForTest().xla_gpu_enable_cublaslt()) {
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
; CHECK-LABEL: ENTRY %AddDotsFunc ({{.*}}: c64[2,2], {{.*}}: c64[2,2]) -> c64[2,2] {
; CHECK-NEXT:    [[P0:%[^ ]+]] = c64[2,2]{1,0} parameter(0)
; CHECK-NEXT:    [[P1:%[^ ]+]] = c64[2,2]{1,0} parameter(1)
; CHECK-NEXT:    [[GEMM:%[^ ]+]] = {{.*}} custom-call([[P0]], [[P1]]),
; CHECK:           custom_call_target="<<CUBLAS_CUSTOM_CALL_TARGET_PLACEHOLDER>>",
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

TEST_P(ParameterizedGemmRewriteTest, AlphaMultipleUsersNoRewrite) {
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
; CHECK:           custom_call_target="<<CUBLAS_CUSTOM_CALL_TARGET_PLACEHOLDER>>",
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

TEST_P(ParameterizedGemmRewriteTest, AlphaVectorNoRewrite) {
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
; CHECK-LABEL: ENTRY %AddDotsFunc ({{.*}}: f32[2,2], {{.*}}: f32[2,2]) -> f32[2,2] {
; CHECK-NEXT:    [[P0:%[^ ]+]] = f32[2,2]{1,0} parameter(0)
; CHECK-NEXT:    [[P1:%[^ ]+]] = f32[2,2]{1,0} parameter(1)
; CHECK-NEXT:    [[GEMM:%[^ ]+]] = {{.*}} custom-call([[P0]], [[P1]]),
; CHECK:           custom_call_target="<<CUBLAS_CUSTOM_CALL_TARGET_PLACEHOLDER>>",
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

TEST_P(ParameterizedGemmRewriteTest, BF16Gemm) {
  const char* hlo_text = R"(
HloModule bf16gemm

ENTRY bf16gemm {
  %parameter.1 = bf16[12,4]{1,0} parameter(0)
  %parameter.2 = bf16[4,8]{1,0} parameter(1)
  ROOT %dot.8 = bf16[12,8] dot(bf16[12,4] %parameter.1, bf16[4,8] %parameter.2), lhs_contracting_dims={1}, rhs_contracting_dims={0}
}
  )";
  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{1e-5, 1e-5}));

  if (!IsCuda() ||
      HasCudaComputeCapability(se::CudaComputeCapability::Ampere())) {
    MatchOptimizedHlo(hlo_text,
                      R"(
; CHECK: {{.*}} custom-call(bf16[16,8]{1,0} {{.*}}, bf16[8,8]{1,0} {{.*}}), custom_call_target="<<CUBLAS_CUSTOM_CALL_TARGET_PLACEHOLDER>>"
  )",
                      /*print_operand_shape=*/true);
  } else {
    GTEST_SKIP() << "Pre-Ampere casts up bf16 to fp32";
  }
}

TEST_P(ParameterizedGemmRewriteTest, BF16GemmStrided) {
  const char* hlo_text = R"(
HloModule bf16gemm

ENTRY bf16gemm {
  %parameter.1 = bf16[3,3,4] parameter(0)
  %parameter.2 = bf16[3,3,2] parameter(1)
  ROOT %dot.3 = bf16[3,4,2]{2,1,0} dot(bf16[3,3,4]{2,1,0} %parameter.1, bf16[3,3,2]{2,1,0} %parameter.2), lhs_batch_dims={0}, lhs_contracting_dims={1}, rhs_batch_dims={0}, rhs_contracting_dims={1}, operand_precision={highest,highest}
}

  )";
  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{1e-5, 1e-5}));

  if (!IsCuda() ||
      HasCudaComputeCapability(se::CudaComputeCapability::Ampere())) {
    MatchOptimizedHlo(hlo_text,
                      R"(
    ; CHECK: {{.*}} custom-call(bf16[3,8,8]{2,1,0} {{.*}}, bf16[3,8,8]{2,1,0} {{.*}}), custom_call_target="<<CUBLAS_CUSTOM_CALL_TARGET_PLACEHOLDER>>"
    )",
                      /*print_operand_shape=*/true);
  } else {
    GTEST_SKIP() << "Pre-Ampere casts up bf16 to fp32";
  }
}

TEST_P(ParameterizedGemmRewriteTest, Int8Gemm) {
  const char* hlo_text = R"(
HloModule int8gemm

ENTRY int8gemm {
  %parameter.1 = s8[12,4]{1,0} parameter(0)
  %parameter.2 = s8[4,8]{1,0} parameter(1)
  ROOT %dot.8 = s32[12,8] dot(s8[12,4] %parameter.1, s8[4,8] %parameter.2), lhs_contracting_dims={1}, rhs_contracting_dims={0}
}
  )";
  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{1e-5, 1e-5}));

  if (!IsCuda() ||
      HasCudaComputeCapability(se::CudaComputeCapability::Volta())) {
    MatchOptimizedHlo(hlo_text,
                      R"(
; CHECK: {{.*}} custom-call(s8[12,4]{1,0} [[A:%[^ ]+]], s8[4,8]{0,1} [[B:%[^ ]+]]), custom_call_target="__cublas$gemm"
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
  if (!IsCuda()) {
    GTEST_SKIP() << "DoBlasGemmWithAlgorithm is not yet implemented on ROCm";
  }

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

  if (!IsCuda() ||
      HasCudaComputeCapability(se::CudaComputeCapability::Volta())) {
    MatchOptimizedHlo(hlo_text,
                      R"(
; CHECK: [[GEMM:%[^ ]+]] = (s32[8,4]{1,0}, s8[{{[0-9]+}}]{0}) custom-call(s8[8,4]{1,0} %{{.*}}, s8[4,4]{0,1} %{{.*}}), custom_call_target="__cublas$gemm",
  )",
                      /*print_operand_shape=*/true);
  }
}

TEST_P(ParameterizedGemmRewriteTest, Int8GemmNoAlphaRewrite) {
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

  if (!IsCuda() ||
      HasCudaComputeCapability(se::CudaComputeCapability::Volta())) {
    MatchOptimizedHlo(hlo_text,
                      R"(
; CHECK: {{.*}} custom-call(s8[12,4]{1,0} [[A:%[^ ]+]], s8[4,8]{0,1} [[B:%[^ ]+]]),
; CHECK:           custom_call_target="__cublas$gemm",
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

TEST_P(ParameterizedGemmRewriteTest, Int8GemmNoBetaRewrite) {
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

  if (!IsCuda() ||
      HasCudaComputeCapability(se::CudaComputeCapability::Volta())) {
    MatchOptimizedHlo(hlo_text,
                      R"(
; CHECK: {{.*}} custom-call(s8[12,4]{1,0} [[A:%[^ ]+]], s8[4,8]{0,1} [[B:%[^ ]+]]),
; CHECK:           custom_call_target="__cublas$gemm",
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

TEST_P(ParameterizedGemmRewriteTest, Int8GemmNotMultipleOfFour) {
  if (!IsCuda()) {
    GTEST_SKIP() << "DoBlasGemmWithAlgorithm is not yet implemented on ROCm";
  }

  const char* hlo_text = R"(
HloModule int8gemm

ENTRY int8gemm {
  %parameter.1 = s8[13,4]{1,0} parameter(0)
  %parameter.2 = s8[4,9]{1,0} parameter(1)
  ROOT %dot.9 = s32[13,9] dot(s8[13,4] %parameter.1, s8[4,9] %parameter.2), lhs_contracting_dims={1}, rhs_contracting_dims={0}
}
  )";
  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{1e-5, 1e-5}));

  if (!IsCuda() ||
      HasCudaComputeCapability(se::CudaComputeCapability::Volta())) {
    MatchOptimizedHlo(hlo_text,
                      R"(
; CHECK: {{.*}} custom-call(s8[16,4]{1,0} [[A:%[^ ]+]], s8[4,12]{0,1} [[B:%[^ ]+]]), custom_call_target="__cublas$gemm"
  )",
                      /*print_operand_shape=*/true);
  } else {
    MatchOptimizedHlo(hlo_text,
                      R"(
; CHECK: {{.*}} dot(s32[13,4]{1,0} [[A:%[^ ]+]], s32[4,9]{1,0} [[B:%[^ ]+]]), lhs_contracting_dims={1}, rhs_contracting_dims={0}

  )",
                      /*print_operand_shape=*/true);
  }
}

TEST_P(ParameterizedGemmRewriteTest, GemmTypeCombinationCheck) {
  if (!IsCuda()) {
    GTEST_SKIP() << "DoBlasGemmWithAlgorithm is not yet implemented on ROCm";
  }

  std::vector<std::tuple<absl::string_view, absl::string_view, bool>>
      type_combinations = {{"s8", "s8", true},
                           {"s32", "s32", true},
                           {"bf16", "bf16", true},
                           {"f16", "f16", true},
                           {"f32", "f32", true},
                           {"f64", "f64", true},
                           {"c64", "c64", true},
                           {"c128", "c128", true},
                           // add mix type gemm
                           {"s8", "s32", true},
                           {"s8", "f32", true},
                           {"f16", "f32", true},
                           {"bf16", "f32", true}};

  if (!IsCuda() ||
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

TEST_P(ParameterizedGemmRewriteTest, UpcastingBf16ToF64) {
  const char* hlo_text = R"(
HloModule test

ENTRY test {
  Arg_0.1 = bf16[4,3]{1,0} parameter(0)
  Arg_1.2 = bf16[3,6]{1,0} parameter(1)
  ROOT dot.3 = f64[4,6]{1,0} dot(Arg_0.1, Arg_1.2), lhs_contracting_dims={1}, rhs_contracting_dims={0}
}
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_text));
  GemmRewriter pass(Capability(), GetToolkitVersion());
  TF_ASSERT_OK_AND_ASSIGN(bool changed, this->RunHloPass(&pass, module.get()));
  EXPECT_TRUE(changed);

  // This is a type combination which is not supported by cublasLt, expect
  // GemmRewriter to choose legacy cublas.
  EXPECT_THAT(
      module->entry_computation()->root_instruction(),
      GmockMatch(m::GetTupleElement(m::CustomCall({"__cublas$gemm"}), 0)));
}

TEST_P(ParameterizedGemmRewriteTest, UpcastingC64ToC128) {
  const char* hlo_text = R"(
HloModule test

ENTRY test {
  Arg_0.1 = c64[4,3]{1,0} parameter(0)
  Arg_1.2 = c64[3,6]{1,0} parameter(1)
  ROOT dot.3 = c128[4,6]{1,0} dot(Arg_0.1, Arg_1.2), lhs_contracting_dims={1}, rhs_contracting_dims={0}
}
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_text));
  GemmRewriter pass(Capability(), GetToolkitVersion());
  TF_ASSERT_OK_AND_ASSIGN(bool changed, this->RunHloPass(&pass, module.get()));
  EXPECT_TRUE(changed);

  // This is a type combination which is not supported by cublasLt, expect
  // GemmRewriter to choose legacy cublas.
  EXPECT_THAT(
      module->entry_computation()->root_instruction(),
      GmockMatch(m::GetTupleElement(m::CustomCall({"__cublas$gemm"}), 0)));
}

TEST_P(ParameterizedGemmRewriteTest, UpcastingF16ToF32) {
  const char* hlo_text = R"(
HloModule test

ENTRY test {
  Arg_0.1 = f16[4,3]{1,0} parameter(0)
  Arg_1.2 = f16[3,6]{1,0} parameter(1)
  ROOT dot.3 = f32[4,6]{1,0} dot(Arg_0.1, Arg_1.2), lhs_contracting_dims={1}, rhs_contracting_dims={0}, operand_precision={highest, highest}
}
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_text));
  GemmRewriter pass(Capability(), GetToolkitVersion());
  TF_ASSERT_OK_AND_ASSIGN(bool changed, this->RunHloPass(&pass, module.get()));
  EXPECT_TRUE(changed);
  EXPECT_THAT(
      module->entry_computation()->root_instruction(),
      GmockMatch(m::GetTupleElement(m::CustomCall({CustomCallTarget()}), 0)));
}

TEST_P(ParameterizedGemmRewriteTest, UpcastingF16ToF64) {
  const char* hlo_text = R"(
HloModule test

ENTRY test {
  Arg_0.1 = f16[4,3]{1,0} parameter(0)
  Arg_1.2 = f16[3,6]{1,0} parameter(1)
  ROOT dot.3 = f64[4,6]{1,0} dot(Arg_0.1, Arg_1.2), lhs_contracting_dims={1}, rhs_contracting_dims={0}
}
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_text));
  GemmRewriter pass(Capability(), GetToolkitVersion());
  TF_ASSERT_OK_AND_ASSIGN(bool changed, this->RunHloPass(&pass, module.get()));
  EXPECT_TRUE(changed);

  // This is a type combination which is not supported by cublasLt, expect
  // GemmRewriter to choose legacy cublas.
  EXPECT_THAT(
      module->entry_computation()->root_instruction(),
      GmockMatch(m::GetTupleElement(m::CustomCall({"__cublas$gemm"}), 0)));
}

TEST_P(ParameterizedGemmRewriteTest, UpcastingF32ToF64) {
  const char* hlo_text = R"(
HloModule test

ENTRY test {
  Arg_0.1 = f32[4,3]{1,0} parameter(0)
  Arg_1.2 = f32[3,6]{1,0} parameter(1)
  ROOT dot.3 = f64[4,6]{1,0} dot(Arg_0.1, Arg_1.2), lhs_contracting_dims={1}, rhs_contracting_dims={0}
}
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_text));
  GemmRewriter pass(Capability(), GetToolkitVersion());
  TF_ASSERT_OK_AND_ASSIGN(bool changed, this->RunHloPass(&pass, module.get()));
  EXPECT_TRUE(changed);

  // This is a type combination which is not supported by cublasLt, expect
  // GemmRewriter to choose legacy cublas.
  EXPECT_THAT(
      module->entry_computation()->root_instruction(),
      GmockMatch(m::GetTupleElement(m::CustomCall({"__cublas$gemm"}), 0)));
}

TEST_P(ParameterizedGemmRewriteTest, DoNotUpconvertOutput) {
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
  GemmRewriter pass(Capability(), GetToolkitVersion());
  TF_ASSERT_OK_AND_ASSIGN(bool changed, this->RunHloPass(&pass, module.get()));
  EXPECT_TRUE(changed);

  // input fp16 and output fp32 combination is supported by legacy cublas and
  // cublasLt, expect GemmRewriter to fuse the convert into gemm.
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::Convert(
                  m::GetTupleElement(m::CustomCall({CustomCallTarget()}), 0))));
}

TEST_P(ParameterizedGemmRewriteTest, UnsupportedMixTypeGemm) {
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
  GemmRewriter pass(Capability(), GetToolkitVersion());
  TF_ASSERT_OK_AND_ASSIGN(bool changed, this->RunHloPass(&pass, module.get()));
  EXPECT_TRUE(changed);

  // u8 is not supported by legacy cublas and cublasLt, expect
  // GemmRewriter to not fuse the convert into gemm.
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::Convert(
                  m::GetTupleElement(m::CustomCall({CustomCallTarget()}), 0))));
}

TEST_P(ParameterizedGemmRewriteTest, CheckIsGemmAliasedBeforeFusion) {
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
  GemmRewriter pass(Capability(), GetToolkitVersion());
  TF_ASSERT_OK_AND_ASSIGN(bool changed, this->RunHloPass(&pass, module.get()));
  EXPECT_TRUE(changed);

  // input fp16 and output fp32 combination is supported by legacy cublas and
  // cublasLt, but gemm output is already aliased with one of the input expect
  // GemmRewriter to not fuse the convert into gemm.
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::Convert(
                  m::GetTupleElement(m::CustomCall({CustomCallTarget()}), 0))));
}

INSTANTIATE_TEST_SUITE_P(CublasTestsBothLegacyAndLt,
                         ParameterizedGemmRewriteTest, ::testing::Bool());

class GemmRewriteAllocationTest : public GpuCodegenTest {
 public:
  void CheckNumberOfAllocations(const std::string& hlo,
                                int expected_number_of_allocations) {
    TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> optimized_module,
                            GetOptimizedModule(hlo));
    if (allocator_ == nullptr) {
      allocator_ = std::make_unique<se::StreamExecutorMemoryAllocator>(
          backend().default_stream_executor());
    }
    TF_ASSERT_OK_AND_ASSIGN(
        std::unique_ptr<Executable> executable,
        backend().compiler()->RunBackend(std::move(optimized_module),
                                         backend().default_stream_executor(),
                                         allocator_.get()));
    GpuExecutable* gpu_executable =
        static_cast<GpuExecutable*>(executable.get());
    absl::Span<const BufferAllocation> allocations =
        gpu_executable->GetAllocations();
    ASSERT_EQ(allocations.size(), expected_number_of_allocations);
  }

  DebugOptions GetDebugOptionsForTest() const override {
    DebugOptions debug_options = GpuCodegenTest::GetDebugOptionsForTest();
    // Make sure the rewriter does not skip the rewrite for being too small.
    debug_options.set_xla_gpu_gemm_rewrite_size_threshold(0);
    debug_options.set_xla_gpu_enable_triton_gemm(false);
    return debug_options;
  }

 private:
  std::unique_ptr<se::DeviceMemoryAllocator> allocator_;
};

TEST_F(GemmRewriteAllocationTest, SharedBufferAssignment) {
  const char* hlo_text = R"(
HloModule SharedBufferAssignment

ENTRY AddDotsFunc {
  x = f32[2,2] parameter(0)
  y = f32[2,2] parameter(1)
  bias = f32[2,2] add(x, y)
  dot = f32[2,2] dot(x, y), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  ROOT out = f32[2,2] add(dot, bias)
}

)";

  // Bias should be fused into the multiplication.
  CheckNumberOfAllocations(hlo_text, 4);
  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{1e-5, 1e-5}));
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
; CHECK-LABEL: ENTRY %DotFunc ({{.*}}: f32[3,3], {{.*}}: f32[3,3]) -> f32[3,3] {
; CHECK-NEXT:    [[P0:%[^ ]+]] = f32[3,3]{1,0} parameter(0)
; CHECK-NEXT:    [[P1:%[^ ]+]] = f32[3,3]{1,0} parameter(1)
; CHECK-NEXT:    [[GEMM:%[^ ]+]] = {{.*}} dot([[P0]], [[P1]]),
; CHECK:           lhs_contracting_dims={1}, rhs_contracting_dims={0}
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
; CHECK-LABEL: ENTRY %DotFunc ({{.*}}: f32[8,8], {{.*}}: f32[8,8]) -> f32[8,8] {
; CHECK-NEXT:    [[P0:%[^ ]+]] = f32[8,8]{1,0} parameter(0)
; CHECK-NEXT:    [[P1:%[^ ]+]] = f32[8,8]{1,0} parameter(1)
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
; CHECK-LABEL: ENTRY %DotFunc ({{.*}}: f32[1024,1024], {{.*}}: f32[1024,1024]) -> f32[1024,1024] {
; CHECK-NEXT:    [[P0:%[^ ]+]] = f32[1024,1024]{1,0} parameter(0)
; CHECK-NEXT:    [[P1:%[^ ]+]] = f32[1024,1024]{1,0} parameter(1)
; CHECK:        [[GEMM:%[^ ]+]] = {{.*}} custom-call({{.*}}), custom_call_target="__cublas$gemm", {{.*}},"algorithm":"ALG_UNSET"
)");
}

}  // namespace
}  // namespace gpu
}  // namespace xla

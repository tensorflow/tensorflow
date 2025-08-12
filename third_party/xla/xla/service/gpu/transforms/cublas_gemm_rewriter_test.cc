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

#include <memory>
#include <optional>
#include <tuple>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/str_replace.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/error_spec.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/testlib/pattern_matcher_gmock.h"
#include "xla/hlo/testlib/test.h"
#include "xla/service/gpu/transforms/gemm_rewriter.h"
#include "xla/service/gpu/transforms/gemm_rewriter_test_lib.h"
#include "xla/service/pattern_matcher.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/semantic_version.h"
#include "xla/xla.pb.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace gpu {

namespace {

namespace m = ::xla::match;

// A test fixture class for tests which are specific to legacy cublas
class LegacyCublasGemmRewriteTest : public GemmRewriteTestBase {
 public:
  DebugOptions GetDebugOptionsForTest() const override {
    DebugOptions debug_options = GemmRewriteTestBase::GetDebugOptionsForTest();
    debug_options.set_xla_gpu_enable_triton_gemm(false);
    debug_options.set_xla_gpu_enable_cublaslt(false);
    return debug_options;
  }
};

TEST_F(LegacyCublasGemmRewriteTest, MatrixVectorMultiplication) {
  const char* hlo_text = R"(
HloModule m

ENTRY e {
  p0 = f32[2048] parameter(0)
  p1 = f32[2048, 16384] parameter(1)
  ROOT d = f32[16384] dot(p0, p1),
    lhs_contracting_dims={0}, rhs_contracting_dims={0}
})";

  RunAndFilecheckHloRewrite(
      hlo_text,
      GemmRewriter(
          se::CudaComputeCapability{se::CudaComputeCapability::kAmpere, 0},
          /*toolkit_version=*/stream_executor::SemanticVersion{12, 4, 0}),
      R"(
; CHECK:  %[[P0:.+]] = f32[2048]{0} parameter(0)
; CHECK:  %[[P1:.+]] = f32[2048,16384]{1,0} parameter(1)
; CHECK:  %[[CUSTOM_CALL:.+]] = (f32[16384]{0}, s8[4194304]{0}) custom-call(%[[P0]], %[[P1]]), custom_call_target="__cublas$gemm"
)");
}

TEST_F(LegacyCublasGemmRewriteTest, MatrixVectorMultiplicationWithBatch) {
  const char* hlo_text = R"(
HloModule m

ENTRY e {
  p0 = f32[10, 10, 2048] parameter(0)
  p1 = f32[10, 10, 2048, 16384] parameter(1)
  ROOT d = f32[10, 10, 16384] dot(p0, p1),
   lhs_batch_dims={0, 1}, rhs_batch_dims={0, 1},
   lhs_contracting_dims={2}, rhs_contracting_dims={2}
})";

  RunAndFilecheckHloRewrite(
      hlo_text,
      GemmRewriter(
          se::CudaComputeCapability{se::CudaComputeCapability::kAmpere, 0},
          /*toolkit_version=*/stream_executor::SemanticVersion{12, 4, 0}),
      R"(
; CHECK:  %[[P0:.+]] = f32[10,10,2048]{2,1,0} parameter(0)
; CHECK:  %[[P1:.+]] = f32[10,10,2048,16384]{3,2,1,0} parameter(1)
; CHECK:  %[[CUSTOM_CALL:.+]] = (f32[10,10,16384]{2,1,0}, s8[4194304]{0}) custom-call(%[[P0]], %[[P1]]), custom_call_target="__cublas$gemm"
)");
}

TEST_F(LegacyCublasGemmRewriteTest, SparseDotNotSupported) {
  const char* hlo_text = R"(
HloModule test

ENTRY main {
  lhs = f16[5,16] parameter(0)
  rhs = f16[32,10] parameter(1)
  meta = u16[5,2] parameter(2)
  ROOT dot = f32[5,10] dot(lhs, rhs, meta),
      lhs_contracting_dims={1}, rhs_contracting_dims={0}, sparsity=L.1@2:4
})";
  auto hlo_pass = GemmRewriter(
      se::CudaComputeCapability{se::CudaComputeCapability::kAmpere, 0},
      /*toolkit_version=*/stream_executor::SemanticVersion{12, 4, 0});
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_text));
  TF_ASSERT_OK_AND_ASSIGN(bool changed, RunHloPass(&hlo_pass, module.get()));
  EXPECT_FALSE(changed);
}

// Test that the alpha and beta fields of the GemmBackendConfig are updated.
// A bias must be present for the beta value to be set.
// In order to have a bias add fused, the bias term must be overwritable.
// We assume that we may not overwrite parameters of a computation. Hence, we
// use the third parameter to create a new value which can be overwritten and
// will be used as the bias. This negate(param_2) has no semantic use, it simply
// exists so that bias may be overwritten.
TEST_F(LegacyCublasGemmRewriteTest, AlphaBetaRewrite) {
  const char* hlo_text = R"(
HloModule NonZeroAlphaBeta

ENTRY AddDotsFunc {
  x = f32[2,2] parameter(0)
  y = f32[2,2] parameter(1)
  param_2 = f32[2,2] parameter(2)
  bias = f32[2,2] negate(param_2)
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
; CHECK-LABEL: ENTRY %AddDotsFunc ({{.*}}: f32[2,2], {{.*}}: f32[2,2], {{.*}}: f32[2,2]) -> f32[2,2] {
; CHECK-DAG:     [[X:%[^ ]+]] = f32[2,2]{1,0} parameter(0)
; CHECK-DAG:     [[Y:%[^ ]+]] = f32[2,2]{1,0} parameter(1)
; CHECK:         [[O:%[^ ]+]] = (f32[2,2]{1,0}, s8[{{[0-9]+}}]{0}) custom-call([[X]], [[Y]], {{[^,)]+}}),
; CHECK:           custom_call_target="__cublas$gemm",
; CHECK:           output_to_operand_aliasing={
; CHECK-SAME:        {0}: (2, {})
; CHECK-SAME:      }
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
; CHECK:         ROOT [[OUT:%[^ ]+]] = f32[2,2]{1,0} get-tuple-element([[O]]), index=0
)");
}

TEST_F(LegacyCublasGemmRewriteTest, BiasMultipleUsersNoOverwrite) {
  const char* hlo_text = R"(
HloModule BiasMultipleUsersNoOverwrite

ENTRY AddDotsFunc {
  x = f32[2,2] parameter(0)
  y = f32[2,2] parameter(1)
  bias = f32[2,2] parameter(2)
  k = f32[] constant(3.0)
  k_broadcast = f32[2, 2] broadcast(k), dimensions={}
  dot_a = f32[2,2] dot(x, y), lhs_contracting_dims={1}, rhs_contracting_dims={0}, operand_precision={highest,highest}
  dot_a_multiplied = f32[2, 2] multiply(dot_a, k_broadcast)
  biased_out = f32[2,2] add(dot_a_multiplied, bias)
  ROOT out = f32[2,2] add(biased_out, bias)
}
)";

  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{1e-5, 1e-5}));
  MatchOptimizedHlo(hlo_text,
                    R"(
; CHECK-LABEL: ENTRY %AddDotsFunc ({{.*}}: f32[2,2], {{.*}}: f32[2,2], {{.*}}: f32[2,2]) -> f32[2,2] {
; CHECK-DAG:     [[P0:%[^ ]+]] = f32[2,2]{1,0} parameter(0)
; CHECK-DAG:     [[P1:%[^ ]+]] = f32[2,2]{1,0} parameter(1)
; CHECK-NEXT:    [[CUSTOM_CALL:%[^ ]+]] = (f32[2,2]{1,0}, s8[{{[0-9]+}}]{0}) custom-call([[P0]], [[P1]]),
; CHECK:           custom_call_target="__cublas$gemm",
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

TEST_F(LegacyCublasGemmRewriteTest, BiasParameterNoOverwrite) {
  const char* hlo_text = R"(
HloModule BiasParameterNoOverwrite

ENTRY AddDotsFunc {
  x = f32[2,2] parameter(0)
  y = f32[2,2] parameter(1)
  bias = f32[2,2] parameter(2)
  dot_a = f32[2,2] dot(x, y), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  ROOT out = f32[2,2] add(dot_a, bias)
}
)";

  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{1e-5, 1e-5}));
  MatchOptimizedHlo(hlo_text,
                    R"(
; CHECK-LABEL: ENTRY %AddDotsFunc ({{.*}}: f32[2,2], {{.*}}: f32[2,2], {{.*}}: f32[2,2]) -> f32[2,2] {
; CHECK-DAG:     [[P0:%[^ ]+]] = f32[2,2]{1,0} parameter(0)
; CHECK-DAG:     [[P1:%[^ ]+]] = f32[2,2]{1,0} parameter(1)
; CHECK-NEXT:    [[GEMM:%[^ ]+]] = (f32[2,2]{1,0}, s8[{{[0-9]+}}]{0}) custom-call([[P0]], [[P1]]),
; CHECK:           custom_call_target="__cublas$gemm",
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

TEST_F(LegacyCublasGemmRewriteTest, BiasTupleParameterOverwrite) {
  const char* hlo_text = R"(
HloModule BiasTupleParameterOverwrite

ENTRY AddDotsFunc {
  x = f32[2,2] parameter(0)
  y = f32[2,2] parameter(1)
  param_2 = (f32[2,2], f32[3,3]) parameter(2)
  bias = f32[2,2] get-tuple-element(param_2), index=0
  dot_a = f32[2,2] dot(x, y), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  ROOT out = f32[2,2] add(dot_a, bias)
}
)";

  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{1e-5, 1e-5}));
  MatchOptimizedHlo(hlo_text,
                    R"(
; CHECK-LABEL: ENTRY %AddDotsFunc ({{.*}}: f32[2,2], {{.*}}: f32[2,2], {{.*}}: (f32[2,2], f32[3,3])) -> f32[2,2] {
; CHECK-DAG:     [[P0:%[^ ]+]] = f32[2,2]{1,0} parameter(0)
; CHECK-DAG:     [[P1:%[^ ]+]] = f32[2,2]{1,0} parameter(1)
; CHECK-DAG:     [[P2:%[^ ]+]] = (f32[2,2]{1,0}, f32[3,3]{1,0}) parameter(2)
; CHECK-DAG:     [[BIAS:%[^ ]+]] = f32[2,2]{1,0} get-tuple-element([[P2]]), index=0
; CHECK-DAG:     [[BIAS_COPY:%[^ ]+]] = f32[2,2]{1,0} copy([[BIAS]])
; CHECK-NEXT:    [[GEMM:%[^ ]+]] = (f32[2,2]{1,0}, s8[{{[0-9]+}}]{0}) custom-call([[P0]], [[P1]], [[BIAS_COPY]]),
; CHECK:           custom_call_target="__cublas$gemm",
; CHECK:           output_to_operand_aliasing={
; CHECK-SAME:        {0}: (2, {})
; CHECK-SAME:      }
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

TEST_F(LegacyCublasGemmRewriteTest, AliasedBiasOverwrite) {
  const char* hlo_text = R"(
HloModule AliasedBiasOverwrite, input_output_alias={ {}: (2, {}, must-alias) }

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
; CHECK-LABEL: ENTRY %AddDotsFunc ({{.*}}: f32[2,2], {{.*}}: f32[2,2], {{.*}}: f32[2,2]) -> f32[2,2] {
; CHECK-DAG:     [[X:%[^ ]+]] = f32[2,2]{1,0} parameter(0)
; CHECK-DAG:     [[Y:%[^ ]+]] = f32[2,2]{1,0} parameter(1)
; CHECK-DAG:     [[BIAS:%[^ ]+]] = f32[2,2]{1,0} parameter(2)
; CHECK:         [[GEMM:%[^ ]+]] = (f32[2,2]{1,0}, s8[{{[0-9]+}}]{0}) custom-call([[X]], [[Y]], [[BIAS]]),
; CHECK:           custom_call_target="__cublas$gemm",
; CHECK:           output_to_operand_aliasing={
; CHECK-SAME:        {0}: (2, {})
; CHECK-SAME:      }
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
)");
}

TEST_F(LegacyCublasGemmRewriteTest, LargerBiasMultipleUsersNoRewrite) {
  const char* hlo_text = R"(
HloModule LargerBiasMultipleUsersNoRewrite

ENTRY AddDotsFunc {
  x = f32[1024,1024] parameter(0)
  y = f32[1024,1024] parameter(1)
  bias = f32[1024,1024] parameter(2)
  dot_a = f32[1024,1024] dot(x, y), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  biased_out = f32[1024,1024] add(dot_a, bias)
  ROOT out = f32[1024,1024] add(biased_out, bias)
}

)";

  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{1e-3, 1e-3}));
  MatchOptimizedHlo(hlo_text,
                    R"(
; CHECK-LABEL: ENTRY %AddDotsFunc ({{.*}}: f32[1024,1024], {{.*}}: f32[1024,1024], {{.*}}: f32[1024,1024]) -> f32[1024,1024] {
; CHECK-DAG:     [[P0:%[^ ]+]] = f32[1024,1024]{1,0} parameter(0)
; CHECK-DAG:     [[P1:%[^ ]+]] = f32[1024,1024]{1,0} parameter(1)
; CHECK-NEXT:    [[GEMM:%[^ ]+]] = (f32[1024,1024]{1,0}, s8[{{[0-9]+}}]{0}) custom-call([[P0]], [[P1]]),
; CHECK:           custom_call_target="__cublas$gemm",
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

// In order to have a bias add fused, the bias term must be overwritable.
// We assume that we may not overwrite parameters of a computation. Hence, we
// use the third parameter to create a new value which can be overwritten and
// will be used as the bias. This negate(param_2) has no semantic use, it simply
// exists so that bias may be overwritten.
TEST_F(LegacyCublasGemmRewriteTest, BF16GemmWithBias) {
  const char* hlo_text = R"(
HloModule BF16GemmWithBias

ENTRY BF16GemmWithBias {
  x = bf16[8,8]{1,0} parameter(0)
  y = bf16[8,8]{1,0} parameter(1)
  dot.5 = bf16[8,8]{1,0} dot(x, y), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  param_2 = bf16[8,8]{1,0} parameter(2)
  bias = bf16[8,8]{1,0} negate(param_2)
  ROOT add.6 = bf16[8,8]{1,0} add(dot.5, bias)
}
  )";

  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{2e-3, 2e-3}));

  if (IsCuda() &&
      !HasCudaComputeCapability(se::CudaComputeCapability::Ampere())) {
    GTEST_SKIP() << "Pre-Ampere casts up bf16 to fp32";
  }

  MatchOptimizedHlo(hlo_text,
                    R"(
; CHECK-LABEL: ENTRY %BF16GemmWithBias ({{.*}}: bf16[8,8], {{.*}}: bf16[8,8], {{.*}}: bf16[8,8]) -> bf16[8,8] {
; CHECK-DAG:    [[X:%[^ ]+]] = bf16[8,8]{1,0} parameter(0)
; CHECK-DAG:    [[Y:%[^ ]+]] = bf16[8,8]{1,0} parameter(1)
; CHECK:        [[GEMM:%[^ ]+]] = (bf16[8,8]{1,0}, s8[{{[0-9]+}}]{0}) custom-call([[X]], [[Y]], {{[^,)]+}}),
; CHECK:           custom_call_target="__cublas$gemm",
; CHECK:           output_to_operand_aliasing={
; CHECK-SAME:        {0}: (2, {})
; CHECK-SAME:      }
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

// In order to have a bias add fused, the bias term must be overwritable.
// We assume that we may not overwrite parameters of a computation. Hence, we
// use the third parameter to create a new value which can be overwritten and
// will be used as the bias. This negate(param_2) has no semantic use, it simply
// exists so that bias may be overwritten.
TEST_F(LegacyCublasGemmRewriteTest, MatrixBias) {
  const char* hlo_text = R"(
HloModule test

ENTRY test {
  x = f32[2,3] parameter(0)
  y = f32[3,4] parameter(1)
  param_2 = f32[2,4] parameter(2)
  bias = f32[2,4] negate(param_2)
  dot_a = f32[2,4] dot(x, y), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  ROOT out = f32[2,4] add(dot_a, bias)
}

)";

  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{1e-5, 1e-5}));
  MatchOptimizedHlo(hlo_text,
                    R"(
; CHECK-LABEL: ENTRY %test ({{.*}}: f32[2,3], {{.*}}: f32[3,4], {{.*}}: f32[2,4]) -> f32[2,4] {
; CHECK-DAG:     [[P0:%[^ ]+]] = f32[2,3]{1,0} parameter(0)
; CHECK-DAG:     [[P1:%[^ ]+]] = f32[3,4]{1,0} parameter(1)
; CHECK:         [[GEMM:%[^ ]+]] = (f32[2,4]{1,0}, s8[{{[0-9]+}}]{0}) custom-call([[P0]], [[P1]], {{[^,)]+}}),
; CHECK:           custom_call_target="__cublas$gemm",
; CHECK:           output_to_operand_aliasing={
; CHECK-SAME:        {0}: (2, {})
; CHECK-SAME:      }
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

TEST_F(LegacyCublasGemmRewriteTest, MatrixBiasWhereBiasIsNotAParameter) {
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
; CHECK-LABEL: ENTRY %test ({{.*}}: f32[2,3], {{.*}}: f32[3,4], {{.*}}: f32[2,3], {{.*}}: f32[3,4]) -> f32[2,4] {
; CHECK-DAG:     [[P0:%[^ ]+]] = f32[2,3]{1,0} parameter(0)
; CHECK-DAG:     [[P1:%[^ ]+]] = f32[3,4]{1,0} parameter(1)
; CHECK-DAG:     [[P2:%[^ ]+]] = f32[2,3]{1,0} parameter(2)
; CHECK-DAG:     [[P3:%[^ ]+]] = f32[3,4]{1,0} parameter(3)
; CHECK-NEXT:    [[FIRST_GEMM:%[^ ]+]] = (f32[2,4]{1,0}, s8[{{[0-9]+}}]{0}) custom-call([[P0]], [[P1]]),
; CHECK:           custom_call_target="__cublas$gemm",
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
; CHECK:         [[FIRST_GEMM_OUT:%[^ ]+]] = f32[2,4]{1,0} get-tuple-element([[FIRST_GEMM]]), index=0
; CHECK-NEXT:    [[SECOND_GEMM:%[^ ]+]] = (f32[2,4]{1,0}, s8[{{[0-9]+}}]{0}) custom-call([[P2]], [[P3]], [[FIRST_GEMM_OUT]]),
; CHECK:           custom_call_target="__cublas$gemm",
; CHECK:           output_to_operand_aliasing={
; CHECK-SAME:        {0}: (2, {})
; CHECK-SAME:      }
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

// Test gemm matrix bias add fusion with mix type
TEST_F(LegacyCublasGemmRewriteTest, MatrixBiasMixType) {
  std::vector<std::tuple<absl::string_view, absl::string_view>>
      type_combinations = {
          {"f16", "f32"},
          {"bf16", "f32"},
      };

  const char* hlo_text_template = R"(
HloModule test

ENTRY test {
  x = <<ABType>>[16,32] parameter(0)
  y = <<ABType>>[32,16] parameter(1)
  z = <<DType>>[16,16] parameter(2)
  dot_a = <<ABType>>[16,16] dot(x, y), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  bias = <<DType>>[16,16] negate(z)
  convert = <<DType>>[16,16] convert(dot_a)
  ROOT out = <<DType>>[16,16] add(convert, bias)
}

)";
  for (const auto& type_combination : type_combinations) {
    absl::flat_hash_map<absl::string_view, absl::string_view> replacements;
    replacements["<<ABType>>"] = std::get<0>(type_combination);
    replacements["<<DType>>"] = std::get<1>(type_combination);
    const auto hlo_text = absl::StrReplaceAll(hlo_text_template, replacements);
    EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{1e-3, 1e-3}));

    if (std::get<0>(type_combination) == "bf16" && IsCuda() &&
        !HasCudaComputeCapability(se::CudaComputeCapability::Ampere())) {
      continue;
    }

    TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> optimized_module,
                            GetOptimizedModule(hlo_text));
    EXPECT_THAT(optimized_module->entry_computation()->root_instruction(),
                GmockMatch(m::GetTupleElement(
                    m::CustomCall(m::Parameter(0), m::Parameter(1),
                                  m::Negate(m::Parameter(2))),
                    0)));
  }
}

// Test batch gemm matrix bias add fusion with mix type
TEST_F(LegacyCublasGemmRewriteTest, MatrixBiasMixTypeBatched) {
  std::vector<std::tuple<absl::string_view, absl::string_view>>
      type_combinations = {
          {"f16", "f32"},
          {"bf16", "f32"},
      };

  const char* hlo_text_template = R"(
HloModule test

ENTRY test {
  x = <<ABType>>[4,16,32] parameter(0)
  y = <<ABType>>[4,32,16] parameter(1)
  z = <<DType>>[4,16,16] parameter(2)
  dot_a = <<ABType>>[4,16,16] dot(x, y), lhs_contracting_dims={2}, rhs_contracting_dims={1}, lhs_batch_dims={0}, rhs_batch_dims={0}
  bias = <<DType>>[4,16,16] negate(z)
  convert = <<DType>>[4,16,16] convert(dot_a)
  ROOT out = <<DType>>[4,16,16] add(convert, bias)
})";
  for (const auto& type_combination : type_combinations) {
    absl::flat_hash_map<absl::string_view, absl::string_view> replacements;
    replacements["<<ABType>>"] = std::get<0>(type_combination);
    replacements["<<DType>>"] = std::get<1>(type_combination);
    const auto hlo_text = absl::StrReplaceAll(hlo_text_template, replacements);
    EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{1e-3, 1e-3}));

    if (std::get<0>(type_combination) == "bf16" && IsCuda() &&
        !HasCudaComputeCapability(se::CudaComputeCapability::Ampere())) {
      continue;
    }

    TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> optimized_module,
                            GetOptimizedModule(hlo_text));
    EXPECT_THAT(optimized_module->entry_computation()->root_instruction(),
                GmockMatch(m::GetTupleElement(
                    m::CustomCall(m::Parameter(0), m::Parameter(1),
                                  m::Negate(m::Parameter(2))),
                    0)));
  }
}

// Test batch gemm matrix bias add fusion with mix type that is not supported.
TEST_F(LegacyCublasGemmRewriteTest, MatrixBiasMixTypeNotSupported) {
  if (IsCuda() &&
      !HasCudaComputeCapability(se::CudaComputeCapability::Ampere())) {
    GTEST_SKIP()
        << "Pre-Ampere rewrites to cutlass_gemm_with_upcast instead of cublas.";
  }

  const char* hlo_text = R"(
HloModule test

ENTRY test {
  x = bf16[16,32] parameter(0)
  y = bf16[32,16] parameter(1)
  z = f64[16,16] parameter(2)
  dot_a = bf16[16,16] dot(x, y), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  bias = f64[16,16] negate(z)
  convert = f64[16,16] convert(dot_a)
  ROOT out = f64[16,16] add(convert, bias)
}

)";

  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{1e-3, 1e-3}));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> optimized_module,
                          GetOptimizedModule(hlo_text));
  MatchOptimizedHlo(hlo_text, R"(
; CHECK:        %[[custom_call:.*]] = {{.*}} custom-call{{.*}}__cublas$gemm
; CHECK:        %[[gte:.*]] = {{.*}} get-tuple-element{{.*}}%[[custom_call]]
; CHECK:        ROOT {{.*}} fusion({{.*}}%[[gte]]
)");
}

// Test batch gemm matrix bias add fusion with mix type that is not supported
// because there are consumers of bias add.
TEST_F(LegacyCublasGemmRewriteTest, MatrixBiasMixTypeAddWithMoreConsumers) {
  const char* hlo_text = R"(
HloModule test

ENTRY test {
  x = bf16[16,32] parameter(0)
  y = bf16[32,16] parameter(1)
  z = f32[16,16] parameter(2)
  dot_a = bf16[16,16] dot(x, y), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  bias = f32[16,16] negate(z)
  convert = f32[16,16] convert(dot_a)
  add_bias = f32[16,16] add(convert, bias)
  ROOT out = f32[16,16] negate(add_bias)
}

)";

  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{1e-3, 1e-3}));

  if (IsCuda() &&
      !HasCudaComputeCapability(se::CudaComputeCapability::Ampere())) {
    GTEST_SKIP() << "Pre-Ampere casts up bf16 to fp32";
  }

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> optimized_module,
                          GetOptimizedModule(hlo_text));
  MatchOptimizedHlo(hlo_text, R"(
; CHECK:        %[[custom_call:.*]] = {{.*}} custom-call{{.*}}__cublas$gemm
; CHECK:        %[[gte:.*]] = {{.*}} get-tuple-element{{.*}}%[[custom_call]]
; CHECK:        ROOT {{.*}} fusion({{.*}}%[[gte]]
)");
}

TEST_F(LegacyCublasGemmRewriteTest, MergeBitcastAndAdd) {
  const char* hlo_text = R"(
HloModule test
ENTRY test {
  x = f32[2,2] parameter(0)
  y = f32[2,2] parameter(1)
  bias = f32[4] parameter(2)
  dot = f32[2,2] dot(x, y), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  ROOT out = f32[4] add(f32[4] bitcast(dot), bias)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_text));
  GemmRewriter pass(Capability(), GetToolkitVersion());
  TF_ASSERT_OK_AND_ASSIGN(bool changed, this->RunHloPass(&pass, module.get()));
  EXPECT_TRUE(changed);

  EXPECT_THAT(
      module->entry_computation()->root_instruction(),
      GmockMatch(
          m::Bitcast(
              m::GetTupleElement(
                  m::CustomCall(
                      {"__cublas$gemm"}, m::Parameter(0), m::Parameter(1),
                      m::Bitcast(m::Parameter(2)).WithShape(F32, {2, 2})),
                  0))
              .WithShape(F32, {4})));
}

// In order to have a bias add fused, the bias term must be overwritable.
// We assume that we may not overwrite parameters of a computation. Hence, we
// use the third parameter to create a new value which can be overwritten and
// will be used as the bias. This negate(param_2) has no semantic use, it simply
// exists so that bias may be overwritten.
TEST_F(LegacyCublasGemmRewriteTest, FoldConstantBias) {
  const char* hlo_text = R"(
HloModule test
ENTRY test {
  x = f32[2,2] parameter(0)
  y = f32[2,2] parameter(1)
  bias = f32[2,2] broadcast(f32[2] constant({0, 0})), dimensions={0}

  dot1 = f32[2,2] dot(x, y), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  param_2 = f32[2,2] parameter(2)
  bias1 = f32[2,2] negate(param_2)
  sum1 = add(dot1, bias1)

  dot2 = f32[2,2] dot(x, y), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  sum2 = add(dot2, f32[2,2] reshape(bias))

  dot3 = f32[2,2] dot(x, y), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  bias3 = f32[2,2] transpose(bias), dimensions={1,0}
  sum3 = add(dot3, bias3)

  dot4 = f32[2,2] dot(x, y), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  sum4 = add(dot4, f32[2,2] bitcast(bias))

  ROOT root = tuple(sum1, sum2, sum3, sum4)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_text));
  GemmRewriter pass(Capability(), GetToolkitVersion());
  TF_ASSERT_OK_AND_ASSIGN(bool changed, this->RunHloPass(&pass, module.get()));
  SCOPED_TRACE(module->ToString());
  EXPECT_TRUE(changed);

  EXPECT_THAT(
      module->entry_computation()->root_instruction(),
      GmockMatch(m::Tuple(
          m::GetTupleElement(m::CustomCall(m::Parameter(0), m::Parameter(1),
                                           m::Negate(m::Parameter(2))),
                             0),
          m::GetTupleElement(
              m::CustomCall(m::Parameter(0), m::Parameter(1), m::Constant()),
              0),
          m::GetTupleElement(
              m::CustomCall(m::Parameter(0), m::Parameter(1), m::Constant()),
              0),
          m::GetTupleElement(
              m::CustomCall(m::Parameter(0), m::Parameter(1), m::Constant()),
              0))));
}

// A test fixture class for tests which are specific to cublasLt
class CublasLtGemmRewriteTest : public GemmRewriteTestBase {
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
; CHECK-LABEL: ENTRY %AddDotsFunc ({{.*}}: f32[2,2], {{.*}}: f32[2,2], {{.*}}: f32[2,2]) -> f32[2,2] {
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

TEST_F(CublasLtGemmRewriteTest, BiasMultipleUsersNoOverwrite) {
  const char* hlo_text = R"(
HloModule BiasMultipleUsersNoOverwrite

ENTRY AddDotsFunc {
  x = f32[2,2] parameter(0)
  y = f32[2,2] parameter(1)
  bias = f32[2,2] parameter(2)
  k = f32[] constant(3.0)
  k_broadcast = f32[2, 2] broadcast(k), dimensions={}
  dot_a = f32[2,2] dot(x, y), lhs_contracting_dims={1}, rhs_contracting_dims={0}, operand_precision={highest,highest}
  dot_a_multiplied = f32[2, 2] multiply(dot_a, k_broadcast)
  biased_out = f32[2,2] add(dot_a_multiplied, bias)
  ROOT out = f32[2,2] add(biased_out, bias)
}
)";

  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{1e-5, 1e-5}));
  MatchOptimizedHlo(hlo_text,
                    R"(
; CHECK-LABEL: ENTRY %AddDotsFunc ({{.*}}: f32[2,2], {{.*}}: f32[2,2], {{.*}}: f32[2,2]) -> f32[2,2] {
; CHECK-DAG:     [[P0:%[^ ]+]] = f32[2,2]{1,0} parameter(0)
; CHECK-DAG:     [[P1:%[^ ]+]] = f32[2,2]{1,0} parameter(1)
; CHECK-DAG:     [[BIAS:%[^ ]+]] = f32[2,2]{1,0} parameter(2)
; CHECK-NEXT:    [[GEMM:%[^ ]+]] = (f32[2,2]{1,0}, s8[{{[0-9]+}}]{0}) custom-call([[P0]], [[P1]], [[BIAS]]),
; CHECK:           custom_call_target="__cublas$lt$matmul",
; CHECK-NOT:       output_to_operand_aliasing
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
)");
}

TEST_F(CublasLtGemmRewriteTest, LargerBiasMultipleUsersNoRewrite) {
  const char* hlo_text = R"(
HloModule LargerBiasMultipleUsersNoRewrite

ENTRY AddDotsFunc {
  x = f32[1024,1024] parameter(0)
  y = f32[1024,1024] parameter(1)
  bias = f32[1024,1024] parameter(2)
  dot_a = f32[1024,1024] dot(x, y), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  biased_out = f32[1024,1024] add(dot_a, bias)
  ROOT out = f32[1024,1024] add(biased_out, bias)
}

)";

  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{1e-3, 1e-3}));
  MatchOptimizedHlo(hlo_text,
                    R"(
; CHECK-LABEL: ENTRY %AddDotsFunc ({{.*}}: f32[1024,1024], {{.*}}: f32[1024,1024], {{.*}}: f32[1024,1024]) -> f32[1024,1024] {
; CHECK-DAG:     [[P0:%[^ ]+]] = f32[1024,1024]{1,0} parameter(0)
; CHECK-DAG:     [[P1:%[^ ]+]] = f32[1024,1024]{1,0} parameter(1)
; CHECK-DAG:     [[BIAS:%[^ ]+]] = f32[1024,1024]{1,0} parameter(2)
; CHECK-NEXT:    [[GEMM_TUPLE:%[^ ]+]] = (f32[1024,1024]{1,0}, s8[{{[0-9]+}}]{0}) custom-call([[P0]], [[P1]], [[BIAS]]),
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
; CHECK-NEXT:  [[GEMM:%[^ ]+]] = f32[1024,1024]{1,0} get-tuple-element([[GEMM_TUPLE]]), index=0
; CHECK-NEXT:  ROOT [[OUT:%[^ ]+]] = f32[1024,1024]{1,0} add([[GEMM]], [[BIAS]])
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
; CHECK-LABEL: ENTRY %BF16GemmWithBias ({{.*}}: bf16[8,8], {{.*}}: bf16[8,8], {{.*}}: bf16[8,8]) -> bf16[8,8] {
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
; CHECK-LABEL: ENTRY %test ({{.*}}: f32[2,3], {{.*}}: f32[3,4], {{.*}}: f32[2,4]) -> f32[2,4] {
; CHECK-NEXT:    [[P0:%[^ ]+]] = f32[2,3]{1,0} parameter(0)
; CHECK-NEXT:    [[P1:%[^ ]+]] = f32[3,4]{1,0} parameter(1)
; CHECK-NEXT:    [[P2:%[^ ]+]] = f32[2,4]{1,0} parameter(2)
; CHECK-NEXT:    [[GEMM:%[^ ]+]] = (f32[2,4]{1,0}, s8[{{[0-9]+}}]{0}) custom-call([[P0]], [[P1]], [[P2]]),
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
; CHECK-LABEL: ENTRY %test ({{.*}}: f32[2,3], {{.*}}: f32[3,4], {{.*}}: f32[2,3], {{.*}}: f32[3,4]) -> f32[2,4] {
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
; CHECK-LABEL: ENTRY %test ({{.*}}: f32[2,3], {{.*}}: f32[3,4], {{.*}}: f32[4]) -> f32[2,4] {
; CHECK-NEXT:    [[P0:%[^ ]+]] = f32[2,3]{1,0} parameter(0)
; CHECK-NEXT:    [[P1:%[^ ]+]] = f32[3,4]{1,0} parameter(1)
; CHECK-NEXT:    [[P2:%[^ ]+]] = f32[4]{0} parameter(2)
; CHECK-NEXT:    [[OUT:%[^ ]+]] = (f32[2,4]{1,0}, s8[{{[0-9]+}}]{0}) custom-call([[P0]], [[P1]], [[P2]]),
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

// Epilogue Fusion disabled when GEMM has multiple users.
TEST_F(CublasLtGemmRewriteTest, VectorBiasMultipleUsers) {
  const char* hlo_text = R"(
HloModule test

ENTRY test {
  x = f32[4,4] parameter(0)
  y = f32[4,4] parameter(1)
  z = f32[4] parameter(2)
  c = f32[] constant(5)
  dot_a = f32[4,4] dot(x, y), lhs_contracting_dims={1}, rhs_contracting_dims={0}, operand_precision={highest,highest}
  z_bcast = f32[4,4] broadcast(z), dimensions={1}
  add_a = f32[4,4] add(dot_a, z_bcast)
  c_bcast = f32[4,4] broadcast(c), dimensions={}
  dot_b = f32[4,4] dot(dot_a, c_bcast), lhs_contracting_dims={1}, rhs_contracting_dims={0}, operand_precision={highest,highest}
  ROOT out = f32[4,4] dot(add_a, dot_b), lhs_contracting_dims={1}, rhs_contracting_dims={0}, operand_precision={highest,highest}
}

)";

  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{1e-5, 1e-5}));
  MatchOptimizedHlo(hlo_text,
                    R"(

; CHECK:        [[FUSED_COMPUTATION:%[^ ]+]] ([[DUMMY0:[^ ]+]]: f32[4,4], [[DUMMY1:[^ ]+]]: f32[4]) -> f32[4,4] {
; CHECK-NEXT:    [[P0:%[^ ]+]] = f32[4,4]{1,0} parameter(0)
; CHECK-NEXT:    [[P1:%[^ ]+]] = f32[4]{0} parameter(1)
; CHECK-NEXT:    [[P2:%[^ ]+]] = f32[4,4]{1,0} broadcast([[P1]]), dimensions={1}
; CHECK-NEXT:    ROOT [[OUT:%[^ ]+]] = f32[4,4]{1,0} add([[P0]], [[P2]])
}

; CHECK-LABEL: ENTRY %test ({{.*}}: f32[4,4], {{.*}}: f32[4,4], {{.*}}: f32[4]) -> f32[4,4] {
; CHECK-NEXT:    [[P0:%[^ ]+]] = f32[4,4]{1,0} parameter(0)
; CHECK-NEXT:    [[P1:%[^ ]+]] = f32[4,4]{1,0} parameter(1)
; CHECK-NEXT:    [[MATMUL0_TUPLE:%[^ ]+]] = (f32[4,4]{1,0}, s8[{{[0-9]+}}]{0}) custom-call([[P0]], [[P1]]),
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
; CHECK-NEXT:    [[MATMUL0:%[^ ]+]] = f32[4,4]{1,0} get-tuple-element([[MATMUL0_TUPLE]]), index=0
; CHECK-NEXT:    [[P2:%[^ ]+]] = f32[4]{0} parameter(2)
; CHECK-NEXT:    [[FUSION:%[^ ]+]] = f32[4,4]{1,0} fusion([[MATMUL0]], [[P2]]), kind=kLoop, calls=[[FUSED_COMPUTATION]]
; CHECK:         [[MATMUL1_TUPLE:%[^ ]+]] = (f32[4,4]{1,0}, s8[{{[0-9]+}}]{0}) custom-call([[MATMUL0]]
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
; CHECK-NEXT:    [[MATMUL1:%[^ ]+]] = f32[4,4]{1,0} get-tuple-element([[MATMUL1_TUPLE]]), index=0
; CHECK-NEXT:    [[OUT:%[^ ]+]] = (f32[4,4]{1,0}, s8[{{[0-9]+}}]{0}) custom-call([[FUSION]], [[MATMUL1]]),
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

; CHECK-LABEL: ENTRY %test ({{.*}}: f32[2,3,4], {{.*}}: f32[4,5,6], {{.*}}: f32[3,5,6]) -> f32[2,3,5,6] {
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

; CHECK-LABEL: ENTRY %test ({{.*}}: f32[2,3,4], {{.*}}: f32[4,5,6], {{.*}}: f32[6]) -> f32[2,3,5,6] {
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
; CHECK-LABEL: ENTRY %test ({{.*}}: f32[2,3], {{.*}}: f32[3,4], {{.*}}: f32[2]) -> f32[4,2] {
; CHECK-NEXT:    [[P0:%[^ ]+]] = f32[2,3]{1,0} parameter(0)
; CHECK-NEXT:    [[P1:%[^ ]+]] = f32[3,4]{1,0} parameter(1)
; CHECK-NEXT:    [[P2:%[^ ]+]] = f32[2]{0} parameter(2)
; CHECK-NEXT:    [[MATMUL_TUPLE:%[^ ]+]] = (f32[2,4]{0,1}, s8[{{[0-9]+}}]{0}) custom-call([[P0]], [[P1]], [[P2]]),
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

; CHECK-LABEL: ENTRY %test ({{.*}}: f32[4,3], {{.*}}: f32[3,4], {{.*}}: f32[3]) -> f32[2,3] {
; CHECK-NEXT:    [[P0:%[^ ]+]] = f32[4,3]{1,0} parameter(0)
; CHECK-NEXT:    [[P1:%[^ ]+]] = f32[3,4]{1,0} parameter(1)
; CHECK-NEXT:    [[P2:%[^ ]+]] = f32[3]{0} parameter(2)
; CHECK-NEXT:    [[MATMUL:%[^ ]+]] = (f32[4,4]{1,0}, s8[{{[0-9]+}}]{0}) custom-call([[P0]], [[P1]], [[P2]]),
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
; CHECK-NEXT:    ROOT [[OUT:%[^ ]+]] = f32[2,3]{1,0} slice([[GETTUPLE]]), slice={[0:2], [0:3]}
      )");
}

// Epilogue Fusion disabled when slice has multiple users.
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
; CHECK-LABEL: ENTRY %test ({{.*}}: f32[2,3], {{.*}}: f32[3,4], {{.*}}: f32[2]) -> f32[2,2] {
; CHECK-DAG:     [[P0:%[^ ]+]] = f32[2,3]{1,0} parameter(0)
; CHECK-DAG:     [[P1:%[^ ]+]] = f32[3,4]{1,0} parameter(1)
; CHECK-DAG:     [[P2:%[^ ]+]] = f32[2]{0} parameter(2)
; CHECK-NEXT:    [[MATMUL0_TUPLE:%[^ ]+]] = (f32[2,4]{1,0}, s8[{{[0-9]+}}]{0}) custom-call([[P0]], [[P1]]),
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
; CHECK:    [[P0:%[^ ]+]] = f32[2,3]{1,0} parameter(0)
; CHECK-NEXT:    [[P1:%[^ ]+]] = f32[3,4]{1,0} parameter(1)
; CHECK-NEXT:    [[P2_BCAST:%[^ ]+]] = f32[2,4]{1,0} parameter(3)
; CHECK-NEXT:    [[OUT:%[^ ]+]] = (f32[2,4]{1,0}, s8[{{[0-9]+}}]{0}) custom-call([[P0]], [[P1]], [[P2_BCAST]]),
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
; CHECK-LABEL: ENTRY %test ({{.*}}: f32[2,3], {{.*}}: f32[3,4], {{.*}}: f32[4], {{.*}}: f32[2,4]) -> f32[2,4] {
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

; CHECK-LABEL: ENTRY %test ({{.*}}: bf16[16,24], {{.*}}: bf16[24,32], {{.*}}: bf16[32]) -> bf16[16,32] {
; CHECK-NEXT:    [[P0:%[^ ]+]] = bf16[16,24]{1,0} parameter(0)
; CHECK-NEXT:    [[P1:%[^ ]+]] = bf16[24,32]{1,0} parameter(1)
; CHECK-NEXT:    [[P2:%[^ ]+]] = bf16[32]{0} parameter(2)
; CHECK-NEXT:    [[OUT:%[^ ]+]] = (bf16[16,32]{1,0}, s8[{{[0-9]+}}]{0}) custom-call([[P0]], [[P1]], [[P2]]),
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
  MatchOptimizedHlo(hlo_text, R"(
; CHECK-DAG: ENTRY %test ({{.*}}: bf16[2,3], {{.*}}: bf16[3,4], {{.*}}: bf16[4]) -> bf16[2,4] {
; CHECK-DAG:    bf16[8,8]{1,0} pad({{.*}}), padding=0_6x0_5
; CHECK-DAG:    bf16[8,8]{1,0} pad({{.*}}), padding=0_5x0_4
      )");
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

; CHECK-LABEL: ENTRY %test ({{.*}}: f32[2,3], {{.*}}: f32[3,4]) -> f32[2,4] {
; CHECK-NEXT:    [[P0:%[^ ]+]] = f32[2,3]{1,0} parameter(0)
; CHECK-NEXT:    [[P1:%[^ ]+]] = f32[3,4]{1,0} parameter(1)
; CHECK-NEXT:    [[OUT:%[^ ]+]] = (f32[2,4]{1,0}, s8[{{[0-9]+}}]{0}) custom-call([[P0]], [[P1]]),
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

; CHECK-LABEL: ENTRY %test ({{.*}}: f32[2,3,4], {{.*}}: f32[4,5,6]) -> f32[2,3,5,6] {
; CHECK-NEXT:    [[P0:%[^ ]+]] = f32[2,3,4]{2,1,0} parameter(0)
; CHECK-NEXT:    [[P0_BITCAST:%[^ ]+]] = f32[6,4]{1,0} bitcast([[P0]])
; CHECK-NEXT:    [[P1:%[^ ]+]] = f32[4,5,6]{2,1,0} parameter(1)
; CHECK-NEXT:    [[P1_BITCAST:%[^ ]+]] = f32[4,30]{1,0}
; CHECK-NEXT:    [[MATMUL_TUPLE:%[^ ]+]] = (f32[6,30]{1,0}, s8[{{[0-9]+}}]{0}) custom-call([[P0_BITCAST]], [[P1_BITCAST]]),
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

; CHECK-LABEL: ENTRY %test ({{.*}}: f32[2,3], {{.*}}: f32[3,4]) -> f32[2,2] {
; CHECK-NEXT:    [[P0:%[^ ]+]] = f32[2,3]{1,0} parameter(0)
; CHECK-NEXT:    [[P1:%[^ ]+]] = f32[3,4]{1,0} parameter(1)
; CHECK-NEXT:    [[MATMUL_TUPLE:%[^ ]+]] = (f32[2,4]{1,0}, s8[{{[0-9]+}}]{0}) custom-call([[P0]], [[P1]]),
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
; CHECK-NEXT:    ROOT [[OUT:%[^ ]+]] = f32[2,2]{1,0} slice([[MATMUL]]), slice={[0:2], [0:2]}
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

; CHECK-LABEL: ENTRY %test ({{.*}}: f32[2,3], {{.*}}: f32[3,4], {{.*}}: f32[2,4]) -> f32[2,4] {
; CHECK-NEXT:    [[P0:%[^ ]+]] = f32[2,3]{1,0} parameter(0)
; CHECK-NEXT:    [[P1:%[^ ]+]] = f32[3,4]{1,0} parameter(1)
; CHECK-NEXT:    [[P2:%[^ ]+]] = f32[2,4]{1,0} parameter(2)
; CHECK-NEXT:    [[OUT:%[^ ]+]] = (f32[2,4]{1,0}, s8[{{[0-9]+}}]{0}) custom-call([[P0]], [[P1]], [[P2]]),
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

; CHECK-LABEL: ENTRY %test ({{.*}}: f32[4,4], {{.*}}: f32[4,4], {{.*}}: f32[4,4]) -> f32[4,4] {
; CHECK-NEXT:    [[P0:%[^ ]+]] = f32[4,4]{1,0} parameter(0)
; CHECK-NEXT:    [[P1:%[^ ]+]] = f32[4,4]{1,0} parameter(1)
; CHECK-NEXT:    [[P2:%[^ ]+]] = f32[4,4]{1,0} parameter(2)
; CHECK-NEXT:    [[OUT:%[^ ]+]] = (f32[4,4]{1,0}, s8[{{[0-9]+}}]{0}) custom-call([[P0]], [[P1]], [[P2]]),
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

; CHECK-LABEL: ENTRY %test ({{.*}}: f32[2,3], {{.*}}: f32[3,4], {{.*}}: f32[4]) -> f32[2,4] {
; CHECK-NEXT:    [[P0:%[^ ]+]] = f32[2,3]{1,0} parameter(0)
; CHECK-NEXT:    [[P1:%[^ ]+]] = f32[3,4]{1,0} parameter(1)
; CHECK-NEXT:    [[P2:%[^ ]+]] = f32[4]{0} parameter(2)
; CHECK-NEXT:    [[OUT:%[^ ]+]] = (f32[2,4]{1,0}, s8[{{[0-9]+}}]{0}) custom-call([[P0]], [[P1]], [[P2]]),
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

; CHECK-LABEL: ENTRY %test ({{.*}}: f32[2,3,4], {{.*}}: f32[4,5,6], {{.*}}: f32[3,5,6]) -> f32[2,3,5,6] {
; CHECK:         [[MATMUL_TUPLE:%[^ ]+]] = (f32[6,30]{1,0}, s8[{{[0-9]+}}]{0}) custom-call(
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
; CHECK-NEXT:    [[MATMUL:%[^ ]+]] = f32[6,30]{1,0} get-tuple-element([[MATMUL_TUPLE]]), index=0
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

; CHECK-LABEL: ENTRY %test ({{.*}}: f32[2,3], {{.*}}: f32[3,4], {{.*}}: f32[2]) -> f32[4,2] {
; CHECK-NEXT:    [[P0:%[^ ]+]] = f32[2,3]{1,0} parameter(0)
; CHECK-NEXT:    [[P1:%[^ ]+]] = f32[3,4]{1,0} parameter(1)
; CHECK-NEXT:    [[P2:%[^ ]+]] = f32[2]{0} parameter(2)
; CHECK-NEXT:    [[MATMUL_TUPLE:%[^ ]+]] = (f32[2,4]{0,1}, s8[{{[0-9]+}}]{0}) custom-call([[P0]], [[P1]], [[P2]]),
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

; CHECK-LABEL: ENTRY %test ({{.*}}: f32[2,3], {{.*}}: f32[3,4], {{.*}}: f32[4], {{.*}}: f32[2,4]) -> f32[2,4] {
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

; CHECK-LABEL: ENTRY %test ({{.*}}: f32[2,3], {{.*}}: f32[3,4]) -> f32[2,4] {
; CHECK-NEXT:    [[P0:%[^ ]+]] = f32[2,3]{1,0} parameter(0)
; CHECK-NEXT:    [[P1:%[^ ]+]] = f32[3,4]{1,0} parameter(1)
; CHECK-NEXT:    [[OUT:%[^ ]+]] = (f32[2,4]{1,0}, s8[{{[0-9]+}}]{0}) custom-call([[P0]], [[P1]]),
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
  auto runtime_version = GetRuntimeVersion();
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

; CHECK-LABEL: ENTRY %test ({{.*}}: f32[2,3], {{.*}}: f32[3,4]) -> f32[2,4] {
; CHECK-NEXT:    [[P0:%[^ ]+]] = f32[2,3]{1,0} parameter(0)
; CHECK-NEXT:    [[P1:%[^ ]+]] = f32[3,4]{1,0} parameter(1)
; CHECK-NEXT:    [[OUT:%[^ ]+]] = (f32[2,4]{1,0}, s8[{{[0-9]+}}]{0}) custom-call([[P0]], [[P1]]),
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

TEST_F(CublasLtGemmRewriteTest, VectorBiasThenApproxGeluActivation) {
  auto runtime_version = GetRuntimeVersion();
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

; CHECK-LABEL: ENTRY %test ({{.*}}: f32[2,3], {{.*}}: f32[3,4], {{.*}}: f32[4]) -> f32[2,4] {
; CHECK-NEXT:    [[P0:%[^ ]+]] = f32[2,3]{1,0} parameter(0)
; CHECK-NEXT:    [[P1:%[^ ]+]] = f32[3,4]{1,0} parameter(1)
; CHECK-NEXT:    [[P2:%[^ ]+]] = f32[4]{0} parameter(2)
; CHECK-NEXT:    [[OUT:%[^ ]+]] = (f32[2,4]{1,0}, s8[{{[0-9]+}}]{0}) custom-call([[P0]], [[P1]], [[P2]]),
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

; CHECK-LABEL: ENTRY %test ({{.*}}: f32[2,3], {{.*}}: f32[3,4]) -> (f32[2,4], f32[2,4]) {
; CHECK-NEXT:    [[P0:%[^ ]+]] = f32[2,3]{1,0} parameter(0)
; CHECK-NEXT:    [[P1:%[^ ]+]] = f32[3,4]{1,0} parameter(1)
; CHECK-NEXT:    [[OUT:%[^ ]+]] = (f32[2,4]{1,0}, f32[2,4]{1,0}, s8[{{[0-9]+}}]{0}) custom-call([[P0]], [[P1]]),
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

; CHECK-LABEL: ENTRY %test ({{.*}}: f32[2,3], {{.*}}: f32[3,4], {{.*}}: f32[4]) -> (f32[2,4], f32[2,4]) {
; CHECK-NEXT:    [[P0:%[^ ]+]] = f32[2,3]{1,0} parameter(0)
; CHECK-NEXT:    [[P1:%[^ ]+]] = f32[3,4]{1,0} parameter(1)
; CHECK-NEXT:    [[P2:%[^ ]+]] = f32[4]{0} parameter(2)
; CHECK-NEXT:    [[OUT:%[^ ]+]] = (f32[2,4]{1,0}, f32[2,4]{1,0}, s8[{{[0-9]+}}]{0}) custom-call([[P0]], [[P1]], [[P2]]),
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
  MatchOptimizedHlo(hlo_text, R"(
; CHECK-DAG: ENTRY %test ({{.*}}: bf16[2,3], {{.*}}: bf16[3,4]) -> bf16[2,4] {
; CHECK-DAG:    bf16[8,8]{1,0} pad({{.*}}), padding=0_6x0_5
; CHECK-DAG:    bf16[8,8]{1,0} pad({{.*}}), padding=0_5x0_4
      )");
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
  GemmRewriter pass(Capability(), GetToolkitVersion());
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

// For F16, the sizes of all dimensions of the operands are required to be
// multiples of 8 to allow matrix bias fusion.
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

; CHECK-LABEL: ENTRY %test ({{.*}}: f16[8,16], {{.*}}: f16[16,8], {{.*}}: f16[8,8]) -> f16[8,8] {
; CHECK-NEXT:    [[P0:%[^ ]+]] = f16[8,16]{1,0} parameter(0)
; CHECK-NEXT:    [[P1:%[^ ]+]] = f16[16,8]{1,0} parameter(1)
; CHECK-NEXT:    [[P2:%[^ ]+]] = f16[8,8]{1,0} parameter(2)
; CHECK-NEXT:    [[OUT:%[^ ]+]] = (f16[8,8]{1,0}, s8[{{[0-9]+}}]{0}) custom-call([[P0]], [[P1]], [[P2]]),
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

TEST_F(CublasLtGemmRewriteTest, VectorBiasF32UnpaddedWithBitcast) {
  const char* hlo_text = R"(
HloModule test

ENTRY test {
  x = f32[2,3]{1,0} parameter(0)
  y = f32[3,4]{1,0} parameter(1)
  z = f32[2]{0} parameter(2)
  dot_a = f32[2,4]{0,1} dot(x, y), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  bitc = f32[4,2]{1,0} bitcast(f32[2,4]{0,1} dot_a)
  z_bcast = f32[4,2] broadcast(z), dimensions={1}
  ROOT add = f32[4,2]{1,0} add(bitc, z_bcast)
}

)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_text));
  GemmRewriter pass(Capability(), GetToolkitVersion());
  TF_ASSERT_OK_AND_ASSIGN(bool changed, this->RunHloPass(&pass, module.get()));
  EXPECT_TRUE(changed);

  EXPECT_THAT(
      module->entry_computation()->root_instruction(),
      GmockMatch(
          m::Bitcast(m::GetTupleElement(
                         m::CustomCall({"__cublas$lt$matmul"}, m::Parameter(0),
                                       m::Parameter(1),
                                       m::Parameter(2).WithShape(F32, {2})),
                         0)
                         .WithShape(F32, {2, 4}))
              .WithShape(F32, {4, 2})));
}

// For F16, the operands are padded on GPUs with Tensor Cores (i.e. Volta and
// newer architectures) so that the sizes of all dimensions are multiples of 8.
TEST_F(CublasLtGemmRewriteTest, VectorBiasF16Unpadded) {
  const char* hlo_text = R"(
HloModule test

ENTRY test {
  x = f16[8,16] parameter(0)
  y = f16[16,8] parameter(1)
  z = f16[8] parameter(2)
  dot_a = f16[8,8] dot(x, y), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  z_bcast = f16[8,8] broadcast(z), dimensions={1}
  ROOT add = f16[8,8] add(dot_a, z_bcast)
})";

  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{8e-3, 2e-3}));
  MatchOptimizedHlo(hlo_text, R"(
; CHECK-NOT:  pad("
; CHECK:      custom-call
; CHECK-SAME: custom_call_target="__cublas$lt$matmul"
      )");
}

TEST_F(CublasLtGemmRewriteTest, VectorBiasF16Padded) {
  if (IsCuda() &&
      !HasCudaComputeCapability(se::CudaComputeCapability::Volta())) {
    GTEST_SKIP() << "Padding of GEMM operands only implemented on "
                    "architectures with Tensor Cores.";
  }
  const char* hlo_text = R"(
HloModule test

ENTRY test {
  x = f16[6,12] parameter(0)
  y = f16[12,6] parameter(1)
  z = f16[6] parameter(2)
  dot_a = f16[6,6] dot(x, y), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  z_bcast = f16[6,6] broadcast(z), dimensions={1}
  ROOT add = f16[6,6] add(dot_a, z_bcast)
})";

  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{1e-3, 1e-3}));
  MatchOptimizedHlo(hlo_text,
                    R"(

; CHECK-DAG: ENTRY %test ({{.*}}: f16[6,12], {{.*}}: f16[12,6], {{.*}}: f16[6]) -> f16[6,6] {
; CHECK-DAG:    f16[8,16]{1,0} pad({{.*}}), padding=0_2x0_4
; CHECK-DAG:    f16[16,8]{1,0} pad({{.*}}), padding=0_4x0_2
      )");
}

// For F16, the operands are padded on GPUs with Tensor Cores (i.e. Volta and
// newer architectures) so that the sizes of all dimensions are multiples of 8.
TEST_F(CublasLtGemmRewriteTest, ReluActivationF16Unpadded) {
  const char* hlo_text = R"(
HloModule test

ENTRY test {
  x = f16[8,16] parameter(0)
  y = f16[16,8] parameter(1)
  dot_a = f16[8,8] dot(x, y), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  c = f16[] constant(0)
  c_bcast = f16[8,8] broadcast(c), dimensions={}
  ROOT out = f16[8,8] maximum(dot_a, c_bcast)
})";

  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{1e-3, 1e-3}));
  MatchOptimizedHlo(hlo_text, R"(
; CHECK-NOT:  pad("
; CHECK:      custom-call
; CHECK-SAME: custom_call_target="__cublas$lt$matmul"
      )");
}

TEST_F(CublasLtGemmRewriteTest, ReluActivationF16Padded) {
  if (IsCuda() &&
      !HasCudaComputeCapability(se::CudaComputeCapability::Volta())) {
    GTEST_SKIP() << "Padding of GEMM operands only implemented on "
                    "architectures with Tensor Cores.";
  }
  const char* hlo_text = R"(
HloModule test

ENTRY test {
  x = f16[6,12] parameter(0)
  y = f16[12,6] parameter(1)
  dot_a = f16[6,6] dot(x, y), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  c = f16[] constant(0)
  c_bcast = f16[6,6] broadcast(c), dimensions={}
  ROOT out = f16[6,6] maximum(dot_a, c_bcast)
})";

  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{1e-5, 1e-5}));
  MatchOptimizedHlo(hlo_text, R"(
; CHECK-DAG: ENTRY %test ({{.*}}: f16[6,12], {{.*}}: f16[12,6]) -> f16[6,6] {
; CHECK-DAG:    f16[8,16]{1,0} pad({{.*}}), padding=0_2x0_4
; CHECK-DAG:    f16[16,8]{1,0} pad({{.*}}), padding=0_4x0_2
      )");
}

TEST_F(CublasLtGemmRewriteTest, MatrixBiasReluActivationF16) {
  const char* hlo_text = R"(
HloModule test

ENTRY test {
  x = f16[8,16] parameter(0)
  y = f16[16,8] parameter(1)
  z = f16[8,8] parameter(2)
  dot_a = f16[8,8] dot(x, y), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  add = f16[8,8] add(dot_a, z)
  c = f16[] constant(0)
  c_bcast = f16[8,8] broadcast(c), dimensions={}
  ROOT out = f16[8,8] maximum(add, c_bcast)
}

)";

  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{1e-3, 1e-3}));
  MatchOptimizedHlo(hlo_text,
                    R"(

; CHECK-LABEL: ENTRY %test ({{.*}}: f16[8,16], {{.*}}: f16[16,8], {{.*}}: f16[8,8]) -> f16[8,8] {
; CHECK-NEXT:    [[P0:%[^ ]+]] = f16[8,16]{1,0} parameter(0)
; CHECK-NEXT:    [[P1:%[^ ]+]] = f16[16,8]{1,0} parameter(1)
; CHECK-NEXT:    [[P2:%[^ ]+]] = f16[8,8]{1,0} parameter(2)
; CHECK-NEXT:    [[OUT:%[^ ]+]] = (f16[8,8]{1,0}, s8[{{[0-9]+}}]{0}) custom-call([[P0]], [[P1]], [[P2]]),
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

// For F16, the operands are padded on GPUs with Tensor Cores (i.e. Volta and
// newer architectures) so that the sizes of all dimensions are multiples of 8.
TEST_F(CublasLtGemmRewriteTest, VectorBiasReluActivationF16Unpadded) {
  const char* hlo_text = R"(
HloModule test

ENTRY test {
  x = f16[8,16] parameter(0)
  y = f16[16,8] parameter(1)
  z = f16[8] parameter(2)
  dot_a = f16[8,8] dot(x, y), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  z_bcast = f16[8,8] broadcast(z), dimensions={1}
  add = f16[8,8] add(dot_a, z_bcast)
  c = f16[] constant(0)
  c_bcast = f16[8,8] broadcast(c), dimensions={}
  ROOT out = f16[8,8] maximum(add, c_bcast)
})";

  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{1e-3, 1e-3}));
  MatchOptimizedHlo(hlo_text, R"(
; CHECK-NOT:  pad("
; CHECK:      custom-call
; CHECK-SAME: custom_call_target="__cublas$lt$matmul"
)");
}

TEST_F(CublasLtGemmRewriteTest, VectorBiasReluActivationF16Padded) {
  if (IsCuda() &&
      !HasCudaComputeCapability(se::CudaComputeCapability::Volta())) {
    GTEST_SKIP() << "Padding of GEMM operands only implemented on "
                    "architectures with Tensor Cores.";
  }
  const char* hlo_text = R"(
HloModule test

ENTRY test {
  x = f16[6,12] parameter(0)
  y = f16[12,6] parameter(1)
  z = f16[6] parameter(2)
  dot_a = f16[6,6] dot(x, y), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  z_bcast = f16[6,6] broadcast(z), dimensions={1}
  add = f16[6,6] add(dot_a, z_bcast)
  c = f16[] constant(0)
  c_bcast = f16[6,6] broadcast(c), dimensions={}
  ROOT out = f16[6,6] maximum(add, c_bcast)
})";

  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{1e-3, 1e-3}));
  MatchOptimizedHlo(hlo_text, R"(
; CHECK-DAG: ENTRY %test ({{.*}}: f16[6,12], {{.*}}: f16[12,6], {{.*}}: f16[6]) -> f16[6,6] {
; CHECK-DAG:   f16[8,16]{1,0} pad({{.*}}), padding=0_2x0_4
; CHECK-DAG:   f16[16,8]{1,0} pad({{.*}}), padding=0_4x0_2
      )");
}

// For bfloat16, the sizes of all dimensions of the operands are required to be
// multiples of 8 to allow matrix bias fusion.
TEST_F(CublasLtGemmRewriteTest, MatrixBiasBF16) {
  const char* hlo_text = R"(
HloModule test

ENTRY test {
  x = bf16[8,16] parameter(0)
  y = bf16[16,8] parameter(1)
  z = bf16[8,8] parameter(2)
  dot_a = bf16[8,8] dot(x, y), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  ROOT out = bf16[8,8] add(dot_a, z)
}

)";

  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{1e-3, 1e-3}));

  if (IsCuda() &&
      !HasCudaComputeCapability(se::CudaComputeCapability::Ampere())) {
    GTEST_SKIP() << "Pre-Ampere casts up bf16 to fp32";
  }

  MatchOptimizedHlo(hlo_text,
                    R"(

; CHECK-LABEL: ENTRY %test ({{.*}}: bf16[8,16], {{.*}}: bf16[16,8], {{.*}}: bf16[8,8]) -> bf16[8,8] {
; CHECK-DAG:     [[P0:%[^ ]+]] = bf16[8,16]{1,0} parameter(0)
; CHECK-DAG:     [[P1:%[^ ]+]] = bf16[16,8]{1,0} parameter(1)
; CHECK-DAG:     [[P2:%[^ ]+]] = bf16[8,8]{1,0} parameter(2)
; CHECK-NEXT:    [[OUT:%[^ ]+]] = (bf16[8,8]{1,0}, s8[{{[0-9]+}}]{0}) custom-call([[P0]], [[P1]], [[P2]]),
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

TEST_F(CublasLtGemmRewriteTest, MatrixBiasBitcastBF16) {
  const char* hlo_text = R"(
HloModule test

ENTRY test {
  x = bf16[8,16] parameter(0)
  y = bf16[16,8] parameter(1)
  bias = bf16[2,4,8] parameter(2)
  dot = bf16[8,8] dot(x, y), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  bitcast = bf16[2,4,8] bitcast(dot)
  ROOT out = bf16[2,4,8] add(bitcast, bias)
}

)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_text));
  GemmRewriter pass(Capability(), GetToolkitVersion());
  TF_ASSERT_OK_AND_ASSIGN(bool changed, this->RunHloPass(&pass, module.get()));
  EXPECT_TRUE(changed);

  EXPECT_THAT(
      module->entry_computation()->root_instruction(),
      GmockMatch(
          m::Bitcast(
              m::GetTupleElement(
                  m::CustomCall(
                      {"__cublas$lt$matmul"},
                      m::Parameter(0).WithShape(BF16, {8, 16}),
                      m::Parameter(1).WithShape(BF16, {16, 8}),
                      m::Bitcast(m::Parameter(2)).WithShape(BF16, {8, 8})),
                  0))
              .WithShape(BF16, {2, 4, 8})));
}

// For bfloat16, the operands are padded if necessary on Ampere and newer
// architectures so that the sizes of all dimensions are multiples of 8.
TEST_F(CublasLtGemmRewriteTest, VectorBiasBF16Unpadded) {
  if (IsCuda() &&
      !HasCudaComputeCapability(se::CudaComputeCapability::Ampere())) {
    GTEST_SKIP()
        << "Pre-Ampere rewrites to cutlass_gemm_with_upcast instead of cublas.";
  }

  const char* hlo_text = R"(
HloModule test

ENTRY test {
  x = bf16[8,16] parameter(0)
  y = bf16[16,8] parameter(1)
  z = bf16[8] parameter(2)
  dot_a = bf16[8,8] dot(x, y), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  z_bcast = bf16[8,8] broadcast(z), dimensions={1}
  ROOT add = bf16[8,8] add(dot_a, z_bcast)
})";

  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{8e-3, 2e-3}));
  MatchOptimizedHlo(hlo_text, R"(
; CHECK-NOT:  pad("
; CHECK:      custom-call
; CHECK-SAME: custom_call_target="__cublas$lt$matmul"
      )");
}

TEST_F(CublasLtGemmRewriteTest, VectorBiasBF16Padded) {
  if (IsCuda() &&
      !HasCudaComputeCapability(se::CudaComputeCapability::Ampere())) {
    GTEST_SKIP() << "Padding of GEMM operands in bfloat16 only implemented on "
                    "Ampere and newer architectures.";
  }
  const char* hlo_text = R"(
HloModule test

ENTRY test {
  x = bf16[6,12] parameter(0)
  y = bf16[12,6] parameter(1)
  z = bf16[6] parameter(2)
  dot_a = bf16[6,6] dot(x, y), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  z_bcast = bf16[6,6] broadcast(z), dimensions={1}
  ROOT add = bf16[6,6] add(dot_a, z_bcast)
})";

  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{1e-3, 1e-3}));
  MatchOptimizedHlo(hlo_text, R"(
; CHECK-DAG:  ENTRY %test ({{.*}}: bf16[6,12], {{.*}}: bf16[12,6], {{.*}}: bf16[6]) -> bf16[6,6] {
; CHECK-DAG:    bf16[8,16]{1,0} pad({{.*}}), padding=0_2x0_4
; CHECK-DAG:    bf16[16,8]{1,0} pad({{.*}}), padding=0_4x0_2
      )");
}

// For bfloat16, the operands are padded if necessary on Ampere and newer
// architectures so that the sizes of all dimensions are multiples of 8.
TEST_F(CublasLtGemmRewriteTest, ReluActivationBF16Unpadded) {
  if (IsCuda() &&
      !HasCudaComputeCapability(se::CudaComputeCapability::Ampere())) {
    GTEST_SKIP()
        << "Pre-Ampere rewrites to cutlass_gemm_with_upcast instead of cublas.";
  }

  const char* hlo_text = R"(
HloModule test

ENTRY test {
  x = bf16[8,16] parameter(0)
  y = bf16[16,8] parameter(1)
  dot_a = bf16[8,8] dot(x, y), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  c = bf16[] constant(0)
  c_bcast = bf16[8,8] broadcast(c), dimensions={}
  ROOT out = bf16[8,8] maximum(dot_a, c_bcast)
}

)";
  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{1e-3, 1e-3}));
  MatchOptimizedHlo(hlo_text, R"(
; CHECK-NOT:  pad("
; CHECK:      custom-call
; CHECK-SAME: custom_call_target="__cublas$lt$matmul"
      )");
}

TEST_F(CublasLtGemmRewriteTest, ReluActivationBF16Padded) {
  if (IsCuda() &&
      !HasCudaComputeCapability(se::CudaComputeCapability::Ampere())) {
    GTEST_SKIP() << "Padding of GEMM operands in bfloat16 only implemented on "
                    "Ampere and newer architectures.";
  }
  const char* hlo_text = R"(
HloModule test

ENTRY test {
  x = bf16[6,12] parameter(0)
  y = bf16[12,6] parameter(1)
  dot_a = bf16[6,6] dot(x, y), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  c = bf16[] constant(0)
  c_bcast = bf16[6,6] broadcast(c), dimensions={}
  ROOT out = bf16[6,6] maximum(dot_a, c_bcast)
})";

  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{1e-5, 1e-5}));
  MatchOptimizedHlo(hlo_text, R"(
; CHECK-DAG: ENTRY %test ({{.*}}: bf16[6,12], {{.*}}: bf16[12,6]) -> bf16[6,6] {
; CHECK-DAG:     bf16[8,16]{1,0} pad({{.*}}), padding=0_2x0_4
; CHECK-DAG:     bf16[16,8]{1,0} pad({{.*}}), padding=0_4x0_2
      )");
}

// For bfloat16, the operands are padded if necessary on Ampere and newer
// architectures so that the sizes of all dimensions are multiples of 8.
TEST_F(CublasLtGemmRewriteTest, VectorBiasReluActivationBF16Unpadded) {
  if (IsCuda() &&
      !HasCudaComputeCapability(se::CudaComputeCapability::Ampere())) {
    GTEST_SKIP()
        << "Pre-Ampere rewrites to cutlass_gemm_with_upcast instead of cublas.";
  }

  const char* hlo_text = R"(
HloModule test

ENTRY test {
  x = bf16[8,16] parameter(0)
  y = bf16[16,8] parameter(1)
  z = bf16[8] parameter(2)
  dot_a = bf16[8,8] dot(x, y), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  z_bcast = bf16[8,8] broadcast(z), dimensions={1}
  add = bf16[8,8] add(dot_a, z_bcast)
  c = bf16[] constant(0)
  c_bcast = bf16[8,8] broadcast(c), dimensions={}
  ROOT out = bf16[8,8] maximum(add, c_bcast)
})";

  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{8e-3, 2e-3}));
  MatchOptimizedHlo(hlo_text,
                    R"(
; CHECK-NOT:  pad("
; CHECK:      custom-call
; CHECK-SAME: custom_call_target="__cublas$lt$matmul"
      )");
}

TEST_F(CublasLtGemmRewriteTest, VectorBiasReluActivationBF16Padded) {
  if (IsCuda() &&
      !HasCudaComputeCapability(se::CudaComputeCapability::Ampere())) {
    GTEST_SKIP() << "Padding of GEMM operands in bfloat16 only implemented on "
                    "Ampere and newer architectures.";
  }
  const char* hlo_text = R"(
HloModule test

ENTRY test {
  x = bf16[6,12] parameter(0)
  y = bf16[12,6] parameter(1)
  z = bf16[6] parameter(2)
  dot_a = bf16[6,6] dot(x, y), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  z_bcast = bf16[6,6] broadcast(z), dimensions={1}
  add = bf16[6,6] add(dot_a, z_bcast)
  c = bf16[] constant(0)
  c_bcast = bf16[6,6] broadcast(c), dimensions={}
  ROOT out = bf16[6,6] maximum(add, c_bcast)
}

)";
  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{1e-3, 1e-3}));
  MatchOptimizedHlo(hlo_text, R"(
; CHECK-DAG: ENTRY %test ({{.*}}: bf16[6,12], {{.*}}: bf16[12,6], {{.*}}: bf16[6]) -> bf16[6,6] {
; CHECK-DAG:     bf16[8,16]{1,0} pad({{.*}}), padding=0_2x0_4
; CHECK-DAG:     bf16[16,8]{1,0} pad({{.*}}), padding=0_4x0_2
      )");
}

TEST_F(CublasLtGemmRewriteTest, VectorBiasReluActivationF64) {
  if (IsRocm()) {
    GTEST_SKIP() << "TODO: Unsupported blas-lt F64 datatype on ROCM";
  }
  const char* hlo_text = R"(
HloModule test

ENTRY test {
  x = f64[2,3] parameter(0)
  y = f64[3,4] parameter(1)
  z = f64[4] parameter(2)
  dot_a = f64[2,4] dot(x, y), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  z_bcast = f64[2,4] broadcast(z), dimensions={1}
  add = f64[2,4] add(dot_a, z_bcast)
  c = f64[] constant(0)
  c_bcast = f64[2,4] broadcast(c), dimensions={}
  ROOT out = f64[2,4] maximum(add, c_bcast)
}

)";

  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{1e-10, 1e-10}));
  MatchOptimizedHlo(hlo_text,
                    R"(

; CHECK-LABEL: ENTRY %test ({{.*}}: f64[2,3], {{.*}}: f64[3,4], {{.*}}: f64[4]) -> f64[2,4] {
; CHECK-NEXT:    [[P0:%[^ ]+]] = f64[2,3]{1,0} parameter(0)
; CHECK-NEXT:    [[P1:%[^ ]+]] = f64[3,4]{1,0} parameter(1)
; CHECK-NEXT:    [[P2:%[^ ]+]] = f64[4]{0} parameter(2)
; CHECK-NEXT:    [[OUT:%[^ ]+]] = (f64[2,4]{1,0}, s8[{{[0-9]+}}]{0}) custom-call([[P0]], [[P1]], [[P2]]),
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

TEST_F(CublasLtGemmRewriteTest, AlphaSimpleRewriteBiasAddActivation) {
  const char* hlo_text = R"(
HloModule test

ENTRY test {
  x = f32[2,3] parameter(0)
  y = f32[3,4] parameter(1)
  z = f32[4] parameter(2)
  k = f32[] constant(3.0)
  k_bcast = f32[2,4] broadcast(k), dimensions={}
  dot_a = f32[2,4] dot(x, y), lhs_contracting_dims={1}, rhs_contracting_dims={0}, operand_precision={highest,highest}
  dot_a_multiplied = f32[2, 4] multiply(dot_a, k_bcast)
  z_bcast = f32[2,4] broadcast(z), dimensions={1}
  add = f32[2,4] add(dot_a_multiplied, z_bcast)
  c = f32[] constant(0)
  c_bcast = f32[2,4] broadcast(c), dimensions={}
  ROOT out = f32[2,4] maximum(add, c_bcast)
}

)";

  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{1e-5, 1e-5}));
  MatchOptimizedHlo(hlo_text,
                    R"(

; CHECK-LABEL: ENTRY %test ({{.*}}: f32[2,3], {{.*}}: f32[3,4], {{.*}}: f32[4]) -> f32[2,4] {
; CHECK-NEXT:    [[P0:%[^ ]+]] = f32[2,3]{1,0} parameter(0)
; CHECK-NEXT:    [[P1:%[^ ]+]] = f32[3,4]{1,0} parameter(1)
; CHECK-NEXT:    [[P2:%[^ ]+]] = f32[4]{0} parameter(2)
; CHECK-NEXT:    [[OUT:%[^ ]+]] = (f32[2,4]{1,0}, s8[{{[0-9]+}}]{0}) custom-call([[P0]], [[P1]], [[P2]]),
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
; CHECK-DAG:         "epilogue":"BIAS_RELU"
; CHECK:           }
      )");
}

TEST_F(CublasLtGemmRewriteTest, FoldConstantBias) {
  const char* hlo_text = R"(
HloModule test
ENTRY test {
  x = f32[2,2] parameter(0)
  y = f32[2,2] parameter(1)
  bias = f32[2,2] broadcast(f32[2] constant({0, 0})), dimensions={0}

  dot1 = f32[2,2] dot(x, y), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  bias1 = f32[2,2] parameter(2)
  sum1 = add(dot1, bias1)

  dot2 = f32[2,2] dot(x, y), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  sum2 = add(dot2, f32[2,2] reshape(bias))

  dot3 = f32[2,2] dot(x, y), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  bias3 = f32[2,2] transpose(bias), dimensions={1,0}
  sum3 = add(dot3, bias3)

  dot4 = f32[2,2] dot(x, y), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  sum4 = add(dot4, f32[2,2] bitcast(bias))

  ROOT root = tuple(sum1, sum2, sum3, sum4)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_text));
  GemmRewriter pass(Capability(), GetToolkitVersion());
  TF_ASSERT_OK_AND_ASSIGN(bool changed, this->RunHloPass(&pass, module.get()));
  SCOPED_TRACE(module->ToString());
  EXPECT_TRUE(changed);

  EXPECT_THAT(
      module->entry_computation()->root_instruction(),
      GmockMatch(m::Tuple(
          m::GetTupleElement(
              m::CustomCall(m::Parameter(0), m::Parameter(1), m::Parameter()),
              0),
          m::GetTupleElement(
              m::CustomCall(m::Parameter(0), m::Parameter(1), m::Constant()),
              0),
          m::GetTupleElement(
              m::CustomCall(m::Parameter(0), m::Parameter(1), m::Constant()),
              0),
          m::GetTupleElement(
              m::CustomCall(m::Parameter(0), m::Parameter(1), m::Constant()),
              0))));
}

TEST_F(CublasLtGemmRewriteTest, MultipleMaximumUsers) {
  const char* hlo_text = R"(
HloModule multiple_maximum_users

relu {
  Arg_0 = f32[3,896,54]{2,1,0} parameter(0)
  constant = f32[] constant(0)
  broadcast = f32[3,896,54]{2,1,0} broadcast(constant), dimensions={}
  ROOT maximum = f32[3,896,54]{2,1,0} maximum(Arg_0, broadcast)
}

ENTRY main {
  constant = f32[] constant(1)
  broadcast_1 = f32[3,896,1024]{2,1,0} broadcast(constant), dimensions={}
  Arg_2 = f32[1024,54]{1,0} parameter(2)
  dot = f32[3,896,54]{2,1,0} dot(broadcast_1, Arg_2), lhs_contracting_dims={2}, rhs_contracting_dims={0}
  Arg_1 = f32[54]{0} parameter(1)
  broadcast_2 = f32[3,896,54]{2,1,0} broadcast(Arg_1), dimensions={2}
  add = f32[3,896,54]{2,1,0} add(dot, broadcast_2)
  call = f32[3,896,54]{2,1,0} call(add), to_apply=relu
  Arg_0 = f32[1]{0} parameter(0)
  reshape_1 = f32[1,1,1]{2,1,0} reshape(Arg_0)
  broadcast_3 = f32[1,1,1]{2,1,0} broadcast(reshape_1), dimensions={0,1,2}
  reshape_2 = f32[] reshape(broadcast_3)
  broadcast_4 = f32[3,896,54]{2,1,0} broadcast(reshape_2), dimensions={}
  multiply = f32[3,896,54]{2,1,0} multiply(call, broadcast_4)
  ROOT tuple = (f32[3,896,54]{2,1,0}, f32[3,896,54]{2,1,0}) tuple(multiply, call)
}
)";

  // TODO(cjfj): Why do we need to relax the error constraint here?!
  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{1e-5, 1e-4}));
  MatchOptimizedHlo(hlo_text,
                    R"(
; CHECK:           custom_call_target="__cublas$lt$matmul",
      )");
}

// Test gemm matrix bias add fusion with mix type and out of place update(C !=
// D)
TEST_F(CublasLtGemmRewriteTest, MatrixBiasMixTypeOutOfPlace) {
  if (IsRocm()) {
    GTEST_SKIP() << "TODO: Unsupported mixed datatypes on ROCM";
  }
  std::vector<std::tuple<absl::string_view, absl::string_view>>
      type_combinations = {
          {"f16", "f32"},
          {"bf16", "f32"},
      };

  const char* hlo_text_template = R"(
HloModule test

ENTRY test {
  x = <<ABType>>[16,32] parameter(0)
  y = <<ABType>>[32,16] parameter(1)
  z = <<DType>>[16,16] parameter(2)
  dot_a = <<ABType>>[16,16] dot(x, y), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  convert = <<DType>>[16,16] convert(dot_a)
  ROOT out = <<DType>>[16,16] add(convert, z)
})";
  for (const auto& type_combination : type_combinations) {
    absl::flat_hash_map<absl::string_view, absl::string_view> replacements;
    replacements["<<ABType>>"] = std::get<0>(type_combination);
    replacements["<<DType>>"] = std::get<1>(type_combination);
    const auto hlo_text = absl::StrReplaceAll(hlo_text_template, replacements);
    EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{1e-3, 1e-3}));

    if (std::get<0>(type_combination) == "bf16" && IsCuda() &&
        !HasCudaComputeCapability(se::CudaComputeCapability::Ampere())) {
      continue;
    }

    TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> optimized_module,
                            GetOptimizedModule(hlo_text));
    EXPECT_THAT(
        optimized_module->entry_computation()->root_instruction(),
        GmockMatch(m::GetTupleElement(
            m::CustomCall(m::Parameter(0), m::Parameter(1), m::Parameter(2)),
            0)));
  }
}

// Test batch gemm matrix bias add fusion with mix type and out of place
// update(C != D)
TEST_F(CublasLtGemmRewriteTest, MatrixBiasMixTypeOutOfPlaceBatched) {
  if (IsRocm()) {
    GTEST_SKIP() << "TODO: Unsupported mixed datatypes on ROCM";
  }
  std::vector<std::tuple<absl::string_view, absl::string_view>>
      type_combinations = {
          {"f16", "f32"},
          {"bf16", "f32"},
      };

  const char* hlo_text_template = R"(
HloModule test

ENTRY test {
  x = <<ABType>>[4,16,32] parameter(0)
  y = <<ABType>>[4,32,16] parameter(1)
  z = <<DType>>[4,16,16] parameter(2)
  dot_a = <<ABType>>[4,16,16] dot(x, y), lhs_contracting_dims={2}, rhs_contracting_dims={1}, lhs_batch_dims={0}, rhs_batch_dims={0}
  convert = <<DType>>[4,16,16] convert(dot_a)
  ROOT out = <<DType>>[4,16,16] add(convert, z)
})";
  for (const auto& type_combination : type_combinations) {
    absl::flat_hash_map<absl::string_view, absl::string_view> replacements;
    replacements["<<ABType>>"] = std::get<0>(type_combination);
    replacements["<<DType>>"] = std::get<1>(type_combination);
    const auto hlo_text = absl::StrReplaceAll(hlo_text_template, replacements);
    EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{1e-3, 1e-3}));

    if (std::get<0>(type_combination) == "bf16" && IsCuda() &&
        !HasCudaComputeCapability(se::CudaComputeCapability::Ampere())) {
      continue;
    }

    TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> optimized_module,
                            GetOptimizedModule(hlo_text));
    EXPECT_THAT(
        optimized_module->entry_computation()->root_instruction(),
        GmockMatch(m::GetTupleElement(
            m::CustomCall(m::Parameter(0), m::Parameter(1), m::Parameter(2)),
            0)));
  }
}

// Test gemm matrix bias add fusion with mix type and in place update(C = D)
TEST_F(CublasLtGemmRewriteTest, MatrixBiasMixTypeInPlace) {
  if (IsRocm()) {
    GTEST_SKIP() << "TODO: Unsupported mixed datatypes on ROCM";
  }
  std::vector<std::tuple<absl::string_view, absl::string_view>>
      type_combinations = {
          {"f16", "f32"},
          {"bf16", "f32"},
      };
  const char* hlo_text_template = R"(
HloModule test

ENTRY test {
  x = <<ABType>>[16,32] parameter(0)
  y = <<ABType>>[32,16] parameter(1)
  z = <<DType>>[16,16] parameter(2)
  dot_a = <<ABType>>[16,16] dot(x, y), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  bias = <<DType>>[16,16] negate(z)
  convert = <<DType>>[16,16] convert(dot_a)
  ROOT out = <<DType>>[16,16] add(convert, bias)
})";

  for (const auto& type_combination : type_combinations) {
    absl::flat_hash_map<absl::string_view, absl::string_view> replacements;
    replacements["<<ABType>>"] = std::get<0>(type_combination);
    replacements["<<DType>>"] = std::get<1>(type_combination);
    const auto hlo_text = absl::StrReplaceAll(hlo_text_template, replacements);
    EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{1e-3, 1e-3}));

    if (std::get<0>(type_combination) == "bf16" && IsCuda() &&
        !HasCudaComputeCapability(se::CudaComputeCapability::Ampere())) {
      continue;
    }

    TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> optimized_module,
                            GetOptimizedModule(hlo_text));
    EXPECT_THAT(optimized_module->entry_computation()->root_instruction(),
                GmockMatch(m::GetTupleElement(
                    m::CustomCall(m::Parameter(0), m::Parameter(1),
                                  m::Negate(m::Parameter(2))),
                    0)));
  }
}

// Test gemm matrix bias add fusion with mix type that is not supported
TEST_F(CublasLtGemmRewriteTest, MatrixBiasMixTypeNotSupported) {
  const char* hlo_text = R"(
HloModule test

ENTRY test {
  x = bf16[16,32] parameter(0)
  y = bf16[32,16] parameter(1)
  z = f64[16,16] parameter(2)
  dot_a = bf16[16,16] dot(x, y), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  bias = f64[16,16] negate(z)
  convert = f64[16,16] convert(dot_a)
  ROOT out = f64[16,16] add(convert, bias)
}

)";

  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{1e-3, 1e-3}));

  if (IsCuda() &&
      !HasCudaComputeCapability(se::CudaComputeCapability::Ampere())) {
    GTEST_SKIP() << "Pre-Ampere casts up bf16 to fp32";
  }

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> optimized_module,
                          GetOptimizedModule(hlo_text));
  MatchOptimizedHlo(hlo_text, R"(
; CHECK:        %[[custom_call:.*]] = {{.*}} custom-call{{.*}}__cublas$lt$matmul
; CHECK:        %[[tuple:.*]] = bf16[16,16]{1,0} get-tuple-element(%[[custom_call]]), index=0
; CHECK:        ROOT {{.*}} fusion({{.*}}%[[tuple]]
)");
}

TEST_F(CublasLtGemmRewriteTest, CublasLtFullyContractingRhsWithBias) {
  const char* hlo_text = R"(
HloModule test

ENTRY test {
  param_0 = bf16[10240,1024]{1,0} parameter(0)
  param_1 = bf16[1024,1]{1,0} parameter(1)
  dot = bf16[10240,1]{1,0} dot(param_0, param_1), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  transpose = bf16[10240,1]{1,0} transpose(dot), dimensions={0,1}
  param_2 = bf16[1]{0} parameter(2)
  reshape = bf16[1]{0} reshape(param_2)
  broadcast = bf16[10240,1]{1,0} broadcast(reshape), dimensions={1}
  ROOT out = bf16[10240,1]{1,0} add(transpose, broadcast)
}
)";

  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{1e-2, 1e-2}));

  if (IsCuda() &&
      !HasCudaComputeCapability(se::CudaComputeCapability::Ampere())) {
    GTEST_SKIP() << "Pre-Ampere casts up bf16 to fp32";
  }

  MatchOptimizedHlo(hlo_text, R"(
; CHECK-DAG: [[LHS:%[^ ]+]] = bf16[10240,1024]{1,0} parameter(0)
; CHECK-DAG: [[P_1:%[^ ]+]] = bf16[1024,1]{1,0} parameter(1)
; CHECK-DAG: [[P_2:%[^ ]+]] = bf16[1]{0} parameter(2)
; CHECK-DAG: [[RHS:%[^ ]+]] = bf16[1024]{0} {{.+}}([[P_1]])
; CHECK-DAG: [[BIAS:%[^ ]+]] = bf16[] {{.+}}([[P_2]])
; CHECK: custom-call([[LHS]], [[RHS]], [[BIAS]]), custom_call_target="__cublas$lt$matmul"
)");
}

}  // namespace
}  // namespace gpu
}  // namespace xla

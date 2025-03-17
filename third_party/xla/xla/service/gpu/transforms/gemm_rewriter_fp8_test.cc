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

#include <array>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>

#include <gtest/gtest.h>
#include "absl/container/flat_hash_map.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_replace.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/error_spec.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/pass/hlo_pass_interface.h"
#include "xla/hlo/testlib/pattern_matcher_gmock.h"
#include "xla/hlo/testlib/test.h"
#include "xla/hlo/testlib/verified_hlo_module.h"
#include "xla/service/computation_placer.h"
#include "xla/service/gpu/transforms/gemm_rewriter.h"
#include "xla/service/gpu/transforms/gemm_rewriter_test_lib.h"
#include "xla/service/hlo_module_config.h"
#include "xla/service/pattern_matcher.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/semantic_version.h"
#include "xla/tests/hlo_runner_agnostic_test_base.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/xla.pb.h"

namespace xla {
namespace gpu {

namespace {

namespace m = ::xla::match;

class ParameterizedFp8GemmRewriteTest
    : public ParameterizedGemmRewriteTestBase {
 public:
  ParameterizedFp8GemmRewriteTest() {
    replacements_[kF8E4M3DatatypePlaceholder] =
        IsCuda() ? "f8e4m3fn" : "f8e4m3fnuz";
    replacements_[kF8E5M2DatatypePlaceholder] =
        IsCuda() ? "f8e5m2" : "f8e5m2fnuz";
    replacements_[kF8E4M3AmaxPlaceholder] = IsCuda() ? "448." : "240.";
  }

  void SetUp() override {
    if (IsCuda() && GetToolkitVersion() < se::SemanticVersion{12, 0, 0}) {
      GTEST_SKIP() << "F8 gemm rewrite is only supported in CUDA 12 and above.";
    }

    if (IsRocm() && GetToolkitVersion() < se::SemanticVersion{6, 0, 0}) {
      GTEST_SKIP()
          << "F8 gemm rewrite is only supported in ROCm 6.0 and above.";
    }
  }

 protected:
  // Check the HLO runs and has an FP8 cuBLAS LT custom call on supported
  // architectures (Ada, Hopper, and later).
  void CheckFp8IfSupported(absl::string_view hlo_text,
                           ErrorSpec error_spec = ErrorSpec{1e-2, 1e-2}) {
    if (!HasFp8Support()) {
      return;
    }
    std::string replaced_hlo_text =
        absl::StrReplaceAll(hlo_text, replacements_);
    EXPECT_TRUE(RunAndCompare(absl::StrReplaceAll(hlo_text, replacements_),
                              error_spec));

    // Most FP8 tests directly create a GemmRewriter and check the output.
    // Here, also run the entire HLO pass pipeline to ensure no other passes
    // interfere with GemmRewriter's pattern matching.
    TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> optimized_module,
                            GetOptimizedModule(replaced_hlo_text));
    const HloInstruction* call =
        FindInstruction(optimized_module.get(), HloOpcode::kCustomCall);
    ASSERT_NE(call, nullptr);
    EXPECT_EQ(call->custom_call_target(), "__cublas$lt$matmul$f8");
  }

  void MatchOptimizedHlo(absl::string_view hlo, const absl::string_view pattern,
                         bool print_operand_shape = false) {
    GemmRewriteTestBase::MatchOptimizedHlo(
        absl::StrReplaceAll(hlo, replacements_),
        absl::StrReplaceAll(pattern, replacements_), print_operand_shape);
  }

  void RunAndFilecheckHloRewrite(
      absl::string_view hlo, HloPassInterface&& hlo_pass,
      std::optional<absl::string_view> expected,
      std::function<void(HloModule*)> after_pass_checks = nullptr,
      const HloModuleConfig* config = nullptr) {
    if (expected.has_value()) {
      std::string replaced_pattern =
          absl::StrReplaceAll(expected.value(), replacements_);
      GemmRewriteTestBase::RunAndFilecheckHloRewrite(
          absl::StrReplaceAll(hlo, replacements_), std::move(hlo_pass),
          replaced_pattern, after_pass_checks, config);
    }
  }

  using ParameterizedGemmRewriteTestBase::ParseAndReturnVerifiedModule;

  absl::StatusOr<std::unique_ptr<VerifiedHloModule>>
  ParseAndReturnVerifiedModule(
      absl::string_view hlo_text, int64_t replica_count = 1,
      int64_t num_partitions = 1,
      std::optional<DeviceAssignment> device_assignment = std::nullopt) const {
    return GemmRewriteTestBase::ParseAndReturnVerifiedModule(
        absl::StrReplaceAll(hlo_text, replacements_), replica_count,
        num_partitions, device_assignment);
  }

 private:
  static constexpr const char* kF8E4M3DatatypePlaceholder{"<<F8E4M3>>"};
  static constexpr const char* kF8E5M2DatatypePlaceholder{"<<F8E5M2>>"};
  static constexpr const char* kF8E4M3AmaxPlaceholder{"<<F8E4M3_AMAX>>"};
};

TEST_P(ParameterizedFp8GemmRewriteTest, SupportsF8NonMajorBatchDim) {
  const char* hlo_text = R"(
HloModule t

ENTRY main {
  %bitcast.73421 = f8e4m3fn[16,8,160]{2,1,0} parameter(0)
  %parameter_1.5 = f8e4m3fn[8,160,1280]{2,1,0} parameter(1)
  %parameter_2 = f8e4m3fn[8,160,1280]{2,1,0} parameter(2)
  %concatenate.2145 = f8e4m3fn[8,160,2560]{2,1,0} concatenate(
      f8e4m3fn[8,160,1280]{2,1,0} %parameter_1.5,
      f8e4m3fn[8,160,1280]{2,1,0} %parameter_2),
        dimensions={2}
  %dot.6237 = f32[8,16,2560]{2,1,0} dot(
      f8e4m3fn[16,8,160]{2,1,0} %bitcast.73421,
      f8e4m3fn[8,160,2560]{2,1,0} %concatenate.2145),
        lhs_batch_dims={1},
        lhs_contracting_dims={2},
        rhs_batch_dims={0},
        rhs_contracting_dims={1}
  ROOT %convert.20480 = bf16[8,16,2560]{2,1,0} convert(
      f32[8,16,2560]{2,1,0} %dot.6237)
})";
  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{1e-2, 1e-2}));
}

TEST_P(ParameterizedFp8GemmRewriteTest, DoNotRewriteToF8OnPreAda) {
  if (!IsCuda()) {
    GTEST_SKIP() << "FP8 Rewrite pattern is different on ROCM-6.2 ";
  }
  if (HasFp8Support()) {
    GTEST_SKIP() << "Test requires a pre-Ada GPU";
  }
  const char* hlo_text = R"(
    HloModule test

    ENTRY PreAdaTest {
      x = <<F8E4M3>>[16,32] parameter(0)
      y = <<F8E4M3>>[32,16] parameter(1)
      ROOT out = <<F8E4M3>>[16,16] dot(x, y), lhs_contracting_dims={1}, rhs_contracting_dims={0}
          }

)";

  EXPECT_TRUE(RunAndCompare(absl::StrReplaceAll(hlo_text, replacements_),
                            ErrorSpec{1e-2, 1e-2}));
  MatchOptimizedHlo(hlo_text,
                    R"(
; CHECK-LABEL: ENTRY %PreAdaTest ({{.*}}: <<F8E4M3>>[16,32], {{.*}}: <<F8E4M3>>[32,16]) -> <<F8E4M3>>[16,16] {
; CHECK:    {{.*}} = {{.*}} custom-call({{.*}}, {{.*}})
; CHECK-DAG:  custom_call_target="<<CUBLAS_CUSTOM_CALL_TARGET_PLACEHOLDER>>"
          )");
}

TEST_P(ParameterizedFp8GemmRewriteTest, DoNotRewriteOnPreAdaWithF32Output) {
  if (HasFp8Support()) {
    GTEST_SKIP() << "Test requires a pre-Ada GPU or an AMD GPU prior to MI300.";
  }
  const char* hlo_text = R"(
    HloModule test

    ENTRY PreAdaTest {
      x = <<F8E4M3>>[16,32] parameter(0)
      y = <<F8E4M3>>[32,16] parameter(1)
      ROOT out = f32[16,16] dot(x, y), lhs_contracting_dims={1}, rhs_contracting_dims={0}
          }

)";

  EXPECT_TRUE(RunAndCompare(absl::StrReplaceAll(hlo_text, replacements_),
                            ErrorSpec{1e-2, 1e-2}));
  MatchOptimizedHlo(hlo_text,
                    R"(
; CHECK-LABEL: ENTRY %PreAdaTest ({{.*}}: <<F8E4M3>>[16,32], {{.*}}: <<F8E4M3>>[32,16]) -> f32[16,16] {
; CHECK:    {{.*}} = {{.*}} custom-call({{.*}}, {{.*}})
; CHECK-DAG:  custom_call_target="<<CUBLAS_CUSTOM_CALL_TARGET_PLACEHOLDER>>"
          )");
}

TEST_P(ParameterizedFp8GemmRewriteTest, UnsupportedTypesF8) {
  // Test with types unsupported by cuBLAS LT when FP8 is used. cuBLAS LT with
  // FP8 requires one of the operands to be F8E4M3FN.
  const char* hlo_text = R"(
    HloModule test

    ENTRY unsupported_types {
      x = <<F8E5M2>>[16,16] parameter(0)
      y = <<F8E5M2>>[16,16] parameter(1)
      ROOT out = <<F8E5M2>>[16,16] dot(x, y), lhs_contracting_dims={1}, rhs_contracting_dims={0}
          }
)";
  EXPECT_TRUE(RunAndCompare(absl::StrReplaceAll(hlo_text, replacements_),
                            ErrorSpec{1e-2, 1e-2}));
  RunAndFilecheckHloRewrite(
      hlo_text,
      GemmRewriter(Capability(), GetToolkitVersion(),
                   GemmRewriterOptions{GemmRewriterOptions::DType::kFp8Only}),
      R"(
; CHECK-LABEL: ENTRY %unsupported_types ({{.*}}: <<F8E5M2>>[16,16], {{.*}}: <<F8E5M2>>[16,16]) -> <<F8E5M2>>[16,16] {
; CHECK-NEXT:    [[P0:%[^ ]+]] = <<F8E5M2>>[16,16]{1,0} parameter(0)
; CHECK-NEXT:    [[P0_CONVERT:%[^ ]+]] = f16[16,16]{1,0} convert([[P0]])
; CHECK-NEXT:    [[P1:%[^ ]+]] = <<F8E5M2>>[16,16]{1,0} parameter(1)
; CHECK-NEXT:    [[P1_CONVERT:%[^ ]+]] = f16[16,16]{1,0} convert([[P1]])
; CHECK-NEXT:    [[DOT:%[^ ]+]] = f16[16,16]{1,0} dot([[P0_CONVERT]], [[P1_CONVERT]]), lhs_contracting_dims={1}, rhs_contracting_dims={0}
; CHECK-NEXT:    ROOT [[OUT:%[^ ]+]] = <<F8E5M2>>[16,16]{1,0} convert([[DOT]])
      )");
}

TEST_P(ParameterizedFp8GemmRewriteTest, UnscaledABUnscaledDF8) {
  const char* hlo_text = R"(
    HloModule test

    ENTRY test {
      x = <<F8E4M3>>[16,32] parameter(0)
      y = <<F8E4M3>>[32,16] parameter(1)
      ROOT out = <<F8E4M3>>[16,16] dot(x, y), lhs_contracting_dims={1}, rhs_contracting_dims={0}
          }

)";

  CheckFp8IfSupported(hlo_text);
  std::string checks = R"(
; CHECK-LABEL: ENTRY %test ({{.*}}: <<F8E4M3>>[16,32], {{.*}}: <<F8E4M3>>[32,16]) -> <<F8E4M3>>[16,16] {
; CHECK-NEXT:    [[P0:%[^ ]+]] = <<F8E4M3>>[16,32]{1,0} parameter(0)
; CHECK-NEXT:    [[P1:%[^ ]+]] = <<F8E4M3>>[32,16]{1,0} parameter(1)
; CHECK-NEXT:    [[P1_TRANSPOSE:%[^ ]+]] = <<F8E4M3>>[16,32]{1,0} transpose([[P1]]), dimensions={1,0}
; CHECK-NEXT:    [[C1:[^ ]+]] = f32[] constant(1)
)";
  if (IsRocm() && GetToolkitVersion() < se::SemanticVersion{6, 2, 0}) {
    checks.append(
        R"(; CHECK-GCN-NEXT:    [[OUT:%[^ ]+]] = (f32[16,16]{1,0}, s8[{{[0-9]+}}]{0}) custom-call([[P0]], [[P1_TRANSPOSE]], [[C1]], [[C1]]),
)");
  } else {
    checks.append(
        R"(; CHECK-NEXT:    [[OUT:%[^ ]+]] = (<<F8E4M3>>[16,16]{1,0}, s8[{{[0-9]+}}]{0}) custom-call([[P0]], [[P1_TRANSPOSE]], [[C1]], [[C1]]),
)");
  }
  checks.append(
      R"(; CHECK:           custom_call_target="__cublas$lt$matmul$f8",
; CHECK:           backend_config={
; CHECK-DAG:         "alpha_real":1
; CHECK-DAG:         "alpha_imag":0
; CHECK-DAG:         "beta":0
; CHECK-DAG:         "dot_dimension_numbers":{
; CHECK-DAG:           "lhs_contracting_dimensions":["1"]
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

  RunAndFilecheckHloRewrite(
      hlo_text,
      GemmRewriter(CudaHopperOrRocmMI300(), GetToolkitVersion(),
                   GemmRewriterOptions{GemmRewriterOptions::DType::kFp8Only}),
      checks);
}

// Do not fuse FP8 matrix bias.
TEST_P(ParameterizedFp8GemmRewriteTest, UnscaledABUnscaledDMatrixBiasF8) {
  const char* hlo_text = R"(
    HloModule test

    ENTRY test {
      x = <<F8E4M3>>[16,32] parameter(0)
      y = <<F8E4M3>>[32,16] parameter(1)
      dot_a = <<F8E4M3>>[16,16] dot(x, y), lhs_contracting_dims={1}, rhs_contracting_dims={0}
      b = <<F8E4M3>>[16,16] parameter(2)
      ROOT out = <<F8E4M3>>[16,16] add(dot_a, b)
          }

)";

  CheckFp8IfSupported(hlo_text);
  RunAndFilecheckHloRewrite(
      hlo_text,
      GemmRewriter(CudaHopperOrRocmMI300(), GetToolkitVersion(),
                   GemmRewriterOptions{GemmRewriterOptions::DType::kFp8Only}),
      R"(
; CHECK-LABEL: ENTRY %test ({{.*}}: <<F8E4M3>>[16,32], {{.*}}: <<F8E4M3>>[32,16], {{.*}}: <<F8E4M3>>[16,16]) -> <<F8E4M3>>[16,16] {
; CHECK-NEXT:    [[P0:%[^ ]+]] = <<F8E4M3>>[16,32]{1,0} parameter(0)
; CHECK-NEXT:    [[P1:%[^ ]+]] = <<F8E4M3>>[32,16]{1,0} parameter(1)
; CHECK-NEXT:    [[P1_TRANSPOSE:%[^ ]+]] = <<F8E4M3>>[16,32]{1,0} transpose([[P1]]), dimensions={1,0}
; CHECK-NEXT:    [[C1:[^ ]+]] = f32[] constant(1)
; CHECK-NEXT:    [[DOT_TUPLE:%[^ ]+]] = (<<F8E4M3>>[16,16]{1,0}, s8[{{[0-9]+}}]{0}) custom-call([[P0]], [[P1_TRANSPOSE]], [[C1]], [[C1]]),
; CHECK:           custom_call_target="__cublas$lt$matmul$f8",
; CHECK:           backend_config={
; CHECK-DAG:         "alpha_real":1
; CHECK-DAG:         "alpha_imag":0
; CHECK-DAG:         "beta":0
; CHECK-DAG:         "dot_dimension_numbers":{
; CHECK-DAG:           "lhs_contracting_dimensions":["1"]
; CHECK-DAG:           "rhs_contracting_dimensions":["1"]
; CHECK-DAG:           "lhs_batch_dimensions":[]
; CHECK-DAG:           "rhs_batch_dimensions":[]
; CHECK-DAG:         }
; CHECK-DAG:         "precision_config":{
; CHECK-DAG:           "operand_precision":["DEFAULT","DEFAULT"]
; CHECK-DAG:         }
; CHECK-DAG:         "epilogue":"DEFAULT"
; CHECK:           }
; CHECK:         [[DOT:%[^ ]+]] = <<F8E4M3>>[16,16]{1,0} get-tuple-element([[DOT_TUPLE]]), index=0
; CHECK-NEXT:    [[P2:%[^ ]+]] = <<F8E4M3>>[16,16]{1,0} parameter(2)
; CHECK-NEXT:    ROOT [[OUT:%[^ ]+]] = <<F8E4M3>>[16,16]{1,0} add([[DOT]], [[P2]])
      )");
}

TEST_P(ParameterizedFp8GemmRewriteTest, ScaledABUnscaledDColMajorLhsF8) {
  const char* hlo_text = R"(
HloModule test
    ENTRY test {
      x = <<F8E4M3>>[2,64,32]{1,2,0} parameter(0)
      y = <<F8E4M3>>[2,32,16]{2,1,0} parameter(1)
      x_scale = f32[] parameter(2)
      y_scale = f32[] parameter(3)
      dq_scale = f32[] multiply(x_scale, y_scale)
      dq_scale_bcast = f32[2,64,16] broadcast(dq_scale), dimensions={}
      out.0 = f32[2,64,16] dot(x, y), lhs_contracting_dims={2}, rhs_contracting_dims={1}, lhs_batch_dims={0}, rhs_batch_dims={0}
      ROOT out = f32[2,64,16] multiply(out.0, dq_scale_bcast)
          }
)";

  CheckFp8IfSupported(hlo_text);
  RunAndFilecheckHloRewrite(
      hlo_text,
      GemmRewriter(CudaHopperOrRocmMI300(), GetToolkitVersion(),
                   GemmRewriterOptions{GemmRewriterOptions::DType::kFp8Only}),
      R"(
; CHECK-LABEL: ENTRY %test ({{.*}}: <<F8E4M3>>[2,64,32], {{.*}}: <<F8E4M3>>[2,32,16], {{.*}}: f32[], {{.*}}: f32[]) -> f32[2,64,16] {
; CHECK-NEXT:    [[P0:%[^ ]+]] = <<F8E4M3>>[2,64,32]{1,2,0} parameter(0)
; CHECK-NEXT:    [[P0_BT:%[^ ]+]] = <<F8E4M3>>[2,32,64]{2,1,0} bitcast([[P0]])
; CHECK-NEXT:    [[P0_TR:%[^ ]+]] = <<F8E4M3>>[2,64,32]{2,1,0} transpose([[P0_BT]]), dimensions={0,2,1}
; CHECK-NEXT:    [[P0_BT1:%[^ ]+]] = <<F8E4M3>>[2,32,64]{1,2,0} bitcast([[P0_TR]])
; CHECK-NEXT:    [[P1:%[^ ]+]] = <<F8E4M3>>[2,32,16]{2,1,0} parameter(1)
; CHECK-NEXT:    [[P1_TRANSPOSE:%[^ ]+]] = <<F8E4M3>>[2,16,32]{2,1,0} transpose([[P1]]), dimensions={0,2,1}
; CHECK-NEXT:    [[P2:%[^ ]+]] = f32[] parameter(2)
; CHECK-NEXT:    [[P3:%[^ ]+]] = f32[] parameter(3)
; CHECK-NEXT:    [[DQ:%[^ ]+]] = f32[] multiply([[P2]], [[P3]])
; CHECK-NEXT:    [[C1:%[^ ]+]] = f32[] constant(1)
; CHECK-NEXT:    [[OUT:%[^ ]+]] = (f32[2,64,16]{2,1,0}, s8[{{[0-9]+}}]{0}) custom-call([[P0_BT1]], [[P1_TRANSPOSE]], [[DQ]], [[C1]]),
; CHECK:           custom_call_target="__cublas$lt$matmul$f8",
; CHECK:           backend_config={
; CHECK-DAG:         "alpha_real":1
; CHECK-DAG:         "alpha_imag":0
; CHECK-DAG:         "beta":0
; CHECK-DAG:         "dot_dimension_numbers":{
; CHECK-DAG:           "lhs_contracting_dimensions":["1"]
; CHECK-DAG:           "rhs_contracting_dimensions":["2"]
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

TEST_P(ParameterizedFp8GemmRewriteTest, ScaledABUnscaledDF8) {
  const char* hlo_text = R"(
    HloModule test

    ENTRY test {
      x = <<F8E4M3>>[16,32] parameter(0)
      y = <<F8E4M3>>[32,16] parameter(1)
      x_f32 = f32[16,32] convert(x)
      y_f32 = f32[32,16] convert(y)
      x_scale = f32[] parameter(2)
      y_scale = f32[] parameter(3)
      x_scale_bcast = f32[16,32] broadcast(x_scale), dimensions={}
      y_scale_bcast = f32[32,16] broadcast(y_scale), dimensions={}
      x_unscaled = f32[16,32] multiply(x_f32, x_scale_bcast)
      y_unscaled = f32[32,16] multiply(y_f32, y_scale_bcast)
      ROOT out = f32[16,16] dot(x_unscaled, y_unscaled), lhs_contracting_dims={1}, rhs_contracting_dims={0}
          }

)";

  CheckFp8IfSupported(hlo_text);
  RunAndFilecheckHloRewrite(
      hlo_text,
      GemmRewriter(CudaHopperOrRocmMI300(), GetToolkitVersion(),
                   GemmRewriterOptions{GemmRewriterOptions::DType::kFp8Only}),
      R"(
; CHECK-LABEL: ENTRY %test ({{.*}}: <<F8E4M3>>[16,32], {{.*}}: <<F8E4M3>>[32,16], {{.*}}: f32[], {{.*}}: f32[]) -> f32[16,16] {
; CHECK-NEXT:    [[P0:%[^ ]+]] = <<F8E4M3>>[16,32]{1,0} parameter(0)
; CHECK-NEXT:    [[P1:%[^ ]+]] = <<F8E4M3>>[32,16]{1,0} parameter(1)
; CHECK-NEXT:    [[P1_TRANSPOSE:%[^ ]+]] = <<F8E4M3>>[16,32]{1,0} transpose([[P1]]), dimensions={1,0}
; CHECK-NEXT:    [[P2:%[^ ]+]] = f32[] parameter(2)
; CHECK-NEXT:    [[P3:%[^ ]+]] = f32[] parameter(3)
; CHECK-NEXT:    [[OUT:%[^ ]+]] = (f32[16,16]{1,0}, s8[{{[0-9]+}}]{0}) custom-call([[P0]], [[P1_TRANSPOSE]], [[P2]], [[P3]]),
; CHECK:           custom_call_target="__cublas$lt$matmul$f8",
; CHECK:           backend_config={
; CHECK-DAG:         "alpha_real":1
; CHECK-DAG:         "alpha_imag":0
; CHECK-DAG:         "beta":0
; CHECK-DAG:         "dot_dimension_numbers":{
; CHECK-DAG:           "lhs_contracting_dimensions":["1"]
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

TEST_P(ParameterizedFp8GemmRewriteTest, ScaledABUnscaledDPaddedF8) {
  const char* hlo_text = R"(
    HloModule test

    ENTRY test {
      x = <<F8E4M3>>[13,17] parameter(0)
      y = <<F8E4M3>>[17,31] parameter(1)
      x_f32 = f32[13,17] convert(x)
      y_f32 = f32[17,31] convert(y)
      x_scale = f32[] parameter(2)
      y_scale = f32[] parameter(3)
      x_scale_bcast = f32[13,17] broadcast(x_scale), dimensions={}
      y_scale_bcast = f32[17,31] broadcast(y_scale), dimensions={}
      x_unscaled = f32[13,17] multiply(x_f32, x_scale_bcast)
      y_unscaled = f32[17,31] multiply(y_f32, y_scale_bcast)
      ROOT out = f32[13,31] dot(x_unscaled, y_unscaled), lhs_contracting_dims={1}, rhs_contracting_dims={0}
          }

)";

  CheckFp8IfSupported(hlo_text);
  RunAndFilecheckHloRewrite(
      hlo_text,
      GemmRewriter(CudaHopperOrRocmMI300(), GetToolkitVersion(),
                   GemmRewriterOptions{GemmRewriterOptions::DType::kFp8Only}),
      R"(
; CHECK-LABEL: ENTRY %test ({{.*}}: <<F8E4M3>>[13,17], {{.*}}: <<F8E4M3>>[17,31], {{.*}}: f32[], {{.*}}: f32[]) -> f32[13,31] {
; CHECK-NEXT:    [[P0:%[^ ]+]] = <<F8E4M3>>[13,17]{1,0} parameter(0)
; CHECK-NEXT:    [[C0:%[^ ]+]] = <<F8E4M3>>[] constant(0)
; CHECK-NEXT:    [[P0_PADDED:%[^ ]+]] = <<F8E4M3>>[16,32]{1,0} pad([[P0]], [[C0]]), padding=0_3x0_15
; CHECK-NEXT:    [[P1:%[^ ]+]] = <<F8E4M3>>[17,31]{1,0} parameter(1)
; CHECK-NEXT:    [[P1_TRANSPOSE:%[^ ]+]] = <<F8E4M3>>[31,17]{1,0} transpose([[P1]]), dimensions={1,0}
; CHECK-NEXT:    [[C1:%[^ ]+]] = <<F8E4M3>>[] constant(0)
; CHECK-NEXT:    [[P1_TRANSPOSE_PADDED:%[^ ]+]] = <<F8E4M3>>[32,32]{1,0} pad([[P1_TRANSPOSE]], [[C1]])
; CHECK-NEXT:    [[P2:%[^ ]+]] = f32[] parameter(2)
; CHECK-NEXT:    [[P3:%[^ ]+]] = f32[] parameter(3)
; CHECK-NEXT:    [[DOT_TUPLE:%[^ ]+]] = (f32[16,32]{1,0}, s8[{{[0-9]+}}]{0}) custom-call([[P0_PADDED]], [[P1_TRANSPOSE_PADDED]], [[P2]], [[P3]]),
; CHECK:           custom_call_target="__cublas$lt$matmul$f8",
; CHECK:           backend_config={
; CHECK-DAG:         "alpha_real":1
; CHECK-DAG:         "alpha_imag":0
; CHECK-DAG:         "beta":0
; CHECK-DAG:         "dot_dimension_numbers":{
; CHECK-DAG:           "lhs_contracting_dimensions":["1"]
; CHECK-DAG:           "rhs_contracting_dimensions":["1"]
; CHECK-DAG:           "lhs_batch_dimensions":[]
; CHECK-DAG:           "rhs_batch_dimensions":[]
; CHECK-DAG:         }
; CHECK-DAG:         "precision_config":{
; CHECK-DAG:           "operand_precision":["DEFAULT","DEFAULT"]
; CHECK-DAG:         }
; CHECK-DAG:         "epilogue":"DEFAULT"
; CHECK:           }
; CHECK-NEXT:  [[DOT:%[^ ]+]] = f32[16,32]{1,0} get-tuple-element([[DOT_TUPLE]]), index=0
; CHECK-NEXT: ROOT [[OUT:%[^ ]+]] = f32[13,31]{1,0} slice([[DOT]]), slice={[0:13], [0:31]}
      )");
}

TEST_P(ParameterizedFp8GemmRewriteTest, ScaledABUnscaledDBitcastF8) {
  const char* hlo_text = R"(
    HloModule test

    ENTRY test {
      x = <<F8E4M3>>[2,8,16] parameter(0)
      y = <<F8E4M3>>[16,16] parameter(1)
      x_f32 = f32[2,8,16] convert(x)
      y_f32 = f32[16,16] convert(y)
      x_scale = f32[] parameter(2)
      y_scale = f32[] parameter(3)
      x_scale_bcast = f32[2,8,16] broadcast(x_scale), dimensions={}
      y_scale_bcast = f32[16,16] broadcast(y_scale), dimensions={}
      x_unscaled = f32[2,8,16] multiply(x_f32, x_scale_bcast)
      y_unscaled = f32[16,16] multiply(y_f32, y_scale_bcast)
      x_bitcast = f32[16,16] bitcast(x_unscaled)
      ROOT out = f32[16,16] dot(x_bitcast, y_unscaled), lhs_contracting_dims={1}, rhs_contracting_dims={0}
          }

)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_text));
  GemmRewriter pass(CudaHopperOrRocmMI300(), GetToolkitVersion(),
                    GemmRewriterOptions{GemmRewriterOptions::DType::kFp8Only});
  TF_ASSERT_OK_AND_ASSIGN(bool changed, this->RunHloPass(&pass, module.get()));
  EXPECT_TRUE(changed);

  EXPECT_THAT(
      module->entry_computation()->root_instruction(),
      GmockMatch(m::GetTupleElement(m::CustomCall({"__cublas$lt$matmul$f8"}), 0)
                     .WithShape(F32, {16, 16})));
}

// Test case where F8 inputs are converted to F32 before the dot, but without
// any scaling.
TEST_P(ParameterizedFp8GemmRewriteTest, UnscaledABUnscaledDWithConvertF8) {
  const char* hlo_text = R"(
    HloModule test

    ENTRY test {
      x = <<F8E4M3>>[16,32] parameter(0)
      y = <<F8E4M3>>[32,16] parameter(1)
      x_f32 = f32[16,32] convert(x)
      y_f32 = f32[32,16] convert(y)
      ROOT out = f32[16,16] dot(x_f32, y_f32), lhs_contracting_dims={1}, rhs_contracting_dims={0}
          }

)";

  CheckFp8IfSupported(hlo_text);
  RunAndFilecheckHloRewrite(
      hlo_text,
      GemmRewriter(CudaHopperOrRocmMI300(), GetToolkitVersion(),
                   GemmRewriterOptions{GemmRewriterOptions::DType::kFp8Only}),
      R"(
; CHECK-LABEL: ENTRY %test ({{.*}}: <<F8E4M3>>[16,32], {{.*}}: <<F8E4M3>>[32,16]) -> f32[16,16] {
; CHECK-NEXT:    [[P0:%[^ ]+]] = <<F8E4M3>>[16,32]{1,0} parameter(0)
; CHECK-NEXT:    [[P1:%[^ ]+]] = <<F8E4M3>>[32,16]{1,0} parameter(1)
; CHECK-NEXT:    [[P1_TRANSPOSE:%[^ ]+]] = <<F8E4M3>>[16,32]{1,0} transpose([[P1]]), dimensions={1,0}
; CHECK-NEXT:    [[C1:%[^ ]+]] = f32[] constant(1)
; CHECK-NEXT:    [[OUT:%[^ ]+]] = (f32[16,16]{1,0}, s8[{{[0-9]+}}]{0}) custom-call([[P0]], [[P1_TRANSPOSE]], [[C1]], [[C1]]),
; CHECK:           custom_call_target="__cublas$lt$matmul$f8",
; CHECK:           backend_config={
; CHECK-DAG:         "alpha_real":1
; CHECK-DAG:         "alpha_imag":0
; CHECK-DAG:         "beta":0
; CHECK-DAG:         "dot_dimension_numbers":{
; CHECK-DAG:           "lhs_contracting_dimensions":["1"]
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

TEST_P(ParameterizedFp8GemmRewriteTest, ScaledABUnscaledDUnaryOpsF8) {
  const char* hlo_text = R"(
    HloModule test

    ENTRY test {
      x = <<F8E4M3>>[3] parameter(0)
      y = <<F8E4M3>>[32,16] parameter(1)
      x_f32 = f32[3] convert(x)
      y_f32 = f32[32,16] convert(y)
      x_scale = f32[] parameter(2)
      y_scale = f32[] parameter(3)
      x_scale_bcast = f32[3] broadcast(x_scale), dimensions={}
      y_scale_bcast = f32[32,16] broadcast(y_scale), dimensions={}
      x_unscaled = f32[3] multiply(x_f32, x_scale_bcast)
      zero = f32[] constant(0)
      x_unscaled_padded = f32[30] pad(x_unscaled, zero), padding=0_27
      x_unscaled_padded_bcast = f32[30,8,5] broadcast(x_unscaled_padded), dimensions={0}
      x_unscaled_padded_bcast_sliced = f32[16,8,4] slice(x_unscaled_padded_bcast), slice={[2:18], [0:8], [0:4]}
      x_unscaled_padded_bcast_sliced_reshaped = f32[16,32] reshape(x_unscaled_padded_bcast_sliced)
      y_unscaled = f32[32,16] multiply(y_f32, y_scale_bcast)
      ROOT out = f32[16,16] dot(x_unscaled_padded_bcast_sliced_reshaped, y_unscaled), lhs_contracting_dims={1}, rhs_contracting_dims={0}
          }

)";

  CheckFp8IfSupported(hlo_text);
  RunAndFilecheckHloRewrite(
      hlo_text,
      GemmRewriter(CudaHopperOrRocmMI300(), GetToolkitVersion(),
                   GemmRewriterOptions{GemmRewriterOptions::DType::kFp8Only}),
      R"(

; CHECK-LABEL: ENTRY %test ({{.*}}: <<F8E4M3>>[3], {{.*}}: <<F8E4M3>>[32,16], {{.*}}: f32[], {{.*}}: f32[]) -> f32[16,16] {
; CHECK-NEXT:    [[P0:%[^ ]+]] = <<F8E4M3>>[3]{0} parameter(0)
; CHECK-NEXT:    [[C0:%[^ ]+]] = f32[] constant(0)
; CHECK-NEXT:    [[C0_CONVERT:%[^ ]+]] = <<F8E4M3>>[] convert([[C0]])
; CHECK-NEXT:    [[P0_U0:%[^ ]+]] = <<F8E4M3>>[30]{0} pad([[P0]], [[C0_CONVERT]]), padding=0_27
; CHECK-NEXT:    [[P0_U1:%[^ ]+]] = <<F8E4M3>>[30,8,5]{2,1,0} broadcast([[P0_U0]]), dimensions={0}
; CHECK-NEXT:    [[P0_U2:%[^ ]+]] = <<F8E4M3>>[16,8,4]{2,1,0} slice([[P0_U1]]), slice={[2:18], [0:8], [0:4]}
; CHECK-NEXT:    [[P0_U3:%[^ ]+]] = <<F8E4M3>>[16,32]{1,0} reshape([[P0_U2]])
; CHECK-NEXT:    [[P1:%[^ ]+]] = <<F8E4M3>>[32,16]{1,0} parameter(1)
; CHECK-NEXT:    [[P1_TRANSPOSE:%[^ ]+]] = <<F8E4M3>>[16,32]{1,0} transpose([[P1]]), dimensions={1,0}
; CHECK-NEXT:    [[P2:%[^ ]+]] = f32[] parameter(2)
; CHECK-NEXT:    [[P3:%[^ ]+]] = f32[] parameter(3)
; CHECK-NEXT:    [[OUT:%[^ ]+]] = (f32[16,16]{1,0}, s8[{{[0-9]+}}]{0}) custom-call([[P0_U3]], [[P1_TRANSPOSE]], [[P2]], [[P3]]),
; CHECK:           custom_call_target="__cublas$lt$matmul$f8",
; CHECK:           backend_config={
; CHECK-DAG:         "alpha_real":1
; CHECK-DAG:         "alpha_imag":0
; CHECK-DAG:         "beta":0
; CHECK-DAG:         "dot_dimension_numbers":{
; CHECK-DAG:           "lhs_contracting_dimensions":["1"]
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

TEST_P(ParameterizedFp8GemmRewriteTest,
       UnscaledABUnscaledDUnaryOpsWithConvertF8) {
  const char* hlo_text = R"(
    HloModule test

    ENTRY test {
      x = <<F8E4M3>>[3] parameter(0)
      y = <<F8E4M3>>[32,16] parameter(1)
      x_f32 = f32[3] convert(x)
      y_f32 = f32[32,16] convert(y)
      zero = f32[] constant(0)
      x_padded = f32[30] pad(x_f32, zero), padding=0_27
      x_padded_bcast = f32[30,8,5] broadcast(x_padded), dimensions={0}
      x_padded_bcast_sliced = f32[16,8,4] slice(x_padded_bcast), slice={[2:18], [0:8], [0:4]}
      x_padded_bcast_sliced_reshaped = f32[16,32] reshape(x_padded_bcast_sliced)
      ROOT out = f32[16,16] dot(x_padded_bcast_sliced_reshaped, y_f32), lhs_contracting_dims={1}, rhs_contracting_dims={0}
          }

)";

  CheckFp8IfSupported(hlo_text);
  RunAndFilecheckHloRewrite(
      hlo_text,
      GemmRewriter(CudaHopperOrRocmMI300(), GetToolkitVersion(),
                   GemmRewriterOptions{GemmRewriterOptions::DType::kFp8Only}),
      R"(

; CHECK-LABEL: ENTRY %test ({{.*}}: <<F8E4M3>>[3], {{.*}}: <<F8E4M3>>[32,16]) -> f32[16,16] {
; CHECK-NEXT:    [[P0:%[^ ]+]] = <<F8E4M3>>[3]{0} parameter(0)
; CHECK-NEXT:    [[C0:%[^ ]+]] = f32[] constant(0)
; CHECK-NEXT:    [[C0_CONVERT:%[^ ]+]] = <<F8E4M3>>[] convert([[C0]])
; CHECK-NEXT:    [[P0_U0:%[^ ]+]] = <<F8E4M3>>[30]{0} pad([[P0]], [[C0_CONVERT]]), padding=0_27
; CHECK-NEXT:    [[P0_U1:%[^ ]+]] = <<F8E4M3>>[30,8,5]{2,1,0} broadcast([[P0_U0]]), dimensions={0}
; CHECK-NEXT:    [[P0_U2:%[^ ]+]] = <<F8E4M3>>[16,8,4]{2,1,0} slice([[P0_U1]]), slice={[2:18], [0:8], [0:4]}
; CHECK-NEXT:    [[P0_U3:%[^ ]+]] = <<F8E4M3>>[16,32]{1,0} reshape([[P0_U2]])
; CHECK-NEXT:    [[P1:%[^ ]+]] = <<F8E4M3>>[32,16]{1,0} parameter(1)
; CHECK-NEXT:    [[P1_TRANSPOSE:%[^ ]+]] = <<F8E4M3>>[16,32]{1,0} transpose([[P1]]), dimensions={1,0}
; CHECK-NEXT:    [[C2:%[^ ]+]] = f32[] constant(1)
; CHECK-NEXT:    [[OUT:%[^ ]+]] = (f32[16,16]{1,0}, s8[{{[0-9]+}}]{0}) custom-call([[P0_U3]], [[P1_TRANSPOSE]], [[C2]], [[C2]]),
; CHECK:           custom_call_target="__cublas$lt$matmul$f8",
; CHECK:           backend_config={
; CHECK-DAG:         "alpha_real":1
; CHECK-DAG:         "alpha_imag":0
; CHECK-DAG:         "beta":0
; CHECK-DAG:         "dot_dimension_numbers":{
; CHECK-DAG:           "lhs_contracting_dimensions":["1"]
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

TEST_P(ParameterizedFp8GemmRewriteTest, ScaledABUnscaledDDynamicSliceF8) {
  const char* hlo_text = R"(
    HloModule test

    ENTRY test {
      x = <<F8E4M3>>[32,32] parameter(0)
      y = <<F8E4M3>>[16,32] parameter(1)
      zero = s32[] constant(0)
      x_f32 = f32[32,32] convert(x)
      y_f32 = f32[16,32] convert(y)
      x_scale = f32[] parameter(2)
      y_scale = f32[] parameter(3)
      x_scale_bcast = f32[32,32] broadcast(x_scale), dimensions={}
      y_scale_bcast = f32[16,32] broadcast(y_scale), dimensions={}
      x_unscaled = f32[32,32] multiply(x_f32, x_scale_bcast)
      y_unscaled = f32[16,32] multiply(y_f32, y_scale_bcast)
      dyn_slice = f32[16,32]{1,0} dynamic-slice(x_unscaled, zero, zero), dynamic_slice_sizes={16,32}
      ROOT dot_a = f32[16,16] dot(dyn_slice, y_unscaled), lhs_contracting_dims={1}, rhs_contracting_dims={1}
          }
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_text));
  GemmRewriter pass(CudaHopperOrRocmMI300(), GetToolkitVersion(),
                    GemmRewriterOptions{GemmRewriterOptions::DType::kFp8Only});
  TF_ASSERT_OK_AND_ASSIGN(bool changed, this->RunHloPass(&pass, module.get()));
  EXPECT_TRUE(changed);

  CheckFp8IfSupported(hlo_text);
  RunAndFilecheckHloRewrite(
      hlo_text,
      GemmRewriter(CudaHopperOrRocmMI300(), GetToolkitVersion(),
                   GemmRewriterOptions{GemmRewriterOptions::DType::kFp8Only}),
      R"(
; CHECK-LABEL: ENTRY %test ({{.*}}: <<F8E4M3>>[32,32], {{.*}}: <<F8E4M3>>[16,32], {{.*}}: f32[], {{.*}}: f32[]) -> f32[16,16] {
; CHECK-NEXT:    [[P0:%[^ ]+]] = <<F8E4M3>>[32,32]{1,0} parameter(0)
; CHECK-NEXT:    [[C0:%[^ ]+]] = s32[] constant(0)
; CHECK-NEXT:    [[DYN_SLICE:%[^ ]+]] = <<F8E4M3>>[16,32]{1,0} dynamic-slice([[P0]], [[C0]], [[C0]]), dynamic_slice_sizes={16,32}
; CHECK-NEXT:    [[P1:%[^ ]+]] = <<F8E4M3>>[16,32]{1,0} parameter(1)
; CHECK-NEXT:    [[P2:%[^ ]+]] = f32[] parameter(2)
; CHECK-NEXT:    [[P3:%[^ ]+]] = f32[] parameter(3)
; CHECK-NEXT:    [[OUT:%[^ ]+]] = (f32[16,16]{1,0}, s8[{{[0-9]+}}]{0}) custom-call([[DYN_SLICE]], [[P1]], [[P2]], [[P3]]),
; CHECK:           custom_call_target="__cublas$lt$matmul$f8",
; CHECK:           backend_config={
; CHECK-DAG:         "alpha_real":1
; CHECK-DAG:         "alpha_imag":0
; CHECK-DAG:         "beta":0
; CHECK-DAG:         "dot_dimension_numbers":{
; CHECK-DAG:           "lhs_contracting_dimensions":["1"]
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

TEST_P(ParameterizedFp8GemmRewriteTest, ScaledABUnscaledDSelectF8) {
  const char* hlo_text = R"(
    HloModule test

    ENTRY test {
      x = <<F8E4M3>>[16,32] parameter(0)
      y = <<F8E4M3>>[16,32] parameter(1)
      x_f32 = f32[16,32] convert(x)
      y_f32 = f32[16,32] convert(y)
      x_scale = f32[] parameter(2)
      y_scale = f32[] parameter(3)
      x_scale_bcast = f32[16,32] broadcast(x_scale), dimensions={}
      y_scale_bcast = f32[16,32] broadcast(y_scale), dimensions={}
      x_unscaled = f32[16,32] multiply(x_f32, x_scale_bcast)
      y_unscaled = f32[16,32] multiply(y_f32, y_scale_bcast)
      k = pred[16,32] parameter(4)
      c = f32[] constant(0)
      c_bcast = f32[16,32] broadcast(c), dimensions={}
      select_a = f32[16,32] select(k, y_unscaled, c_bcast)
      ROOT dot_a = f32[16,16] dot(x_unscaled, select_a), lhs_contracting_dims={1}, rhs_contracting_dims={1}
          }
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_text));
  GemmRewriter pass(CudaHopperOrRocmMI300(), GetToolkitVersion(),
                    GemmRewriterOptions{GemmRewriterOptions::DType::kFp8Only});
  TF_ASSERT_OK_AND_ASSIGN(bool changed, this->RunHloPass(&pass, module.get()));
  EXPECT_TRUE(changed);

  CheckFp8IfSupported(hlo_text);
  RunAndFilecheckHloRewrite(
      hlo_text,
      GemmRewriter(CudaHopperOrRocmMI300(), GetToolkitVersion(),
                   GemmRewriterOptions{GemmRewriterOptions::DType::kFp8Only}),
      R"(
; CHECK-LABEL: ENTRY %test ({{.*}}: <<F8E4M3>>[16,32], {{.*}}: <<F8E4M3>>[16,32], {{.*}}: f32[], {{.*}}: f32[], {{.*}}: pred[16,32]) -> f32[16,16] {
; CHECK-NEXT:    [[P0:%[^ ]+]] = <<F8E4M3>>[16,32]{1,0} parameter(0)
; CHECK-NEXT:    [[P4:%[^ ]+]] = pred[16,32]{1,0} parameter(4)
; CHECK-NEXT:    [[P1:%[^ ]+]] = <<F8E4M3>>[16,32]{1,0} parameter(1)
; CHECK-NEXT:    [[C0:%[^ ]+]] = f32[] constant(0)
; CHECK-NEXT:    [[C0_BCAST:%[^ ]+]] = f32[16,32]{1,0} broadcast([[C0]]), dimensions={}
; CHECK-NEXT:    [[C0_CONVERT:%[^ ]+]] = <<F8E4M3>>[16,32]{1,0} convert([[C0_BCAST]])
; CHECK-NEXT:    [[SELECT:%[^ ]+]] = <<F8E4M3>>[16,32]{1,0} select([[P4]], [[P1]], [[C0_CONVERT]])
; CHECK-NEXT:    [[P2:%[^ ]+]] = f32[] parameter(2)
; CHECK-NEXT:    [[P3:%[^ ]+]] = f32[] parameter(3)
; CHECK-NEXT:    [[OUT:%[^ ]+]] = (f32[16,16]{1,0}, s8[{{[0-9]+}}]{0}) custom-call([[P0]], [[SELECT]], [[P2]], [[P3]]),
; CHECK:           custom_call_target="__cublas$lt$matmul$f8",
; CHECK:           backend_config={
; CHECK-DAG:         "alpha_real":1
; CHECK-DAG:         "alpha_imag":0
; CHECK-DAG:         "beta":0
; CHECK-DAG:         "dot_dimension_numbers":{
; CHECK-DAG:           "lhs_contracting_dimensions":["1"]
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

TEST_P(ParameterizedFp8GemmRewriteTest,
       ScaledABUnscaledDSelectNonzeroConstantF8) {
  const char* hlo_text = R"(
    HloModule test

    ENTRY test {
      x = <<F8E4M3>>[16,32] parameter(0)
      y = <<F8E4M3>>[16,32] parameter(1)
      x_f32 = f32[16,32] convert(x)
      y_f32 = f32[16,32] convert(y)
      x_scale = f32[] parameter(2)
      y_scale = f32[] parameter(3)
      x_scale_bcast = f32[16,32] broadcast(x_scale), dimensions={}
      y_scale_bcast = f32[16,32] broadcast(y_scale), dimensions={}
      x_unscaled = f32[16,32] multiply(x_f32, x_scale_bcast)
      y_unscaled = f32[16,32] multiply(y_f32, y_scale_bcast)
      k = pred[16,32] parameter(4)
      c = f32[] constant(1)
      c_bcast = f32[16,32] broadcast(c), dimensions={}
      select_a = f32[16,32] select(k, y_unscaled, c_bcast)
      ROOT dot_a = f32[16,16] dot(x_unscaled, select_a), lhs_contracting_dims={1}, rhs_contracting_dims={1}
          }
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_text));
  GemmRewriter pass(CudaHopperOrRocmMI300(), GetToolkitVersion(),
                    GemmRewriterOptions{GemmRewriterOptions::DType::kFp8Only});
  TF_ASSERT_OK_AND_ASSIGN(bool changed, this->RunHloPass(&pass, module.get()));
  EXPECT_FALSE(changed);
}

TEST_P(ParameterizedFp8GemmRewriteTest, BatchedScaledABUnscaledDF8) {
  const char* hlo_text = R"(
    HloModule test

    ENTRY test {
      x = <<F8E4M3>>[10,16,32] parameter(0)
      y = <<F8E4M3>>[10,32,16] parameter(1)
      x_f32 = f32[10,16,32] convert(x)
      y_f32 = f32[10,32,16] convert(y)
      x_scale = f32[] parameter(2)
      y_scale = f32[] parameter(3)
      x_scale_bcast = f32[10,16,32] broadcast(x_scale), dimensions={}
      y_scale_bcast = f32[10,32,16] broadcast(y_scale), dimensions={}
      x_unscaled = f32[10,16,32] multiply(x_f32, x_scale_bcast)
      y_unscaled = f32[10,32,16] multiply(y_f32, y_scale_bcast)
      ROOT out = f32[10,16,16] dot(x_unscaled, y_unscaled), lhs_contracting_dims={2}, rhs_contracting_dims={1}, lhs_batch_dims={0}, rhs_batch_dims={0}
          }

)";

  CheckFp8IfSupported(hlo_text);
  RunAndFilecheckHloRewrite(
      hlo_text,
      GemmRewriter(CudaHopperOrRocmMI300(), GetToolkitVersion(),
                   GemmRewriterOptions{GemmRewriterOptions::DType::kFp8Only}),
      R"(
; CHECK-LABEL: ENTRY %test ({{.*}}: <<F8E4M3>>[10,16,32], {{.*}}: <<F8E4M3>>[10,32,16], {{.*}}: f32[], {{.*}}: f32[]) -> f32[10,16,16] {
; CHECK-NEXT:    [[P0:%[^ ]+]] = <<F8E4M3>>[10,16,32]{2,1,0} parameter(0)
; CHECK-NEXT:    [[P1:%[^ ]+]] = <<F8E4M3>>[10,32,16]{2,1,0} parameter(1)
; CHECK-NEXT:    [[P1_TRANSPOSE:%[^ ]+]] = <<F8E4M3>>[10,16,32]{2,1,0} transpose([[P1]]), dimensions={0,2,1}
; CHECK-NEXT:    [[P2:%[^ ]+]] = f32[] parameter(2)
; CHECK-NEXT:    [[P3:%[^ ]+]] = f32[] parameter(3)
; CHECK-NEXT:    [[OUT:%[^ ]+]] = (f32[10,16,16]{2,1,0}, s8[{{[0-9]+}}]{0}) custom-call([[P0]], [[P1_TRANSPOSE]], [[P2]], [[P3]]),
; CHECK:           custom_call_target="__cublas$lt$matmul$f8",
; CHECK:           backend_config={
; CHECK-DAG:         "alpha_real":1
; CHECK-DAG:         "alpha_imag":0
; CHECK-DAG:         "beta":0
; CHECK-DAG:         "dot_dimension_numbers":{
; CHECK-DAG:           "lhs_contracting_dimensions":["2"]
; CHECK-DAG:           "rhs_contracting_dimensions":["2"]
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

TEST_P(ParameterizedFp8GemmRewriteTest, ScaledABAlphaDF8) {
  const char* hlo_text = R"(
    HloModule test

    ENTRY test {
      x = <<F8E4M3>>[16,32] parameter(0)
      y = <<F8E4M3>>[32,16] parameter(1)
      x_f32 = f32[16,32] convert(x)
      y_f32 = f32[32,16] convert(y)
      x_scale = f32[] parameter(2)
      y_scale = f32[] parameter(3)
      x_scale_bcast = f32[16,32] broadcast(x_scale), dimensions={}
      y_scale_bcast = f32[32,16] broadcast(y_scale), dimensions={}
      x_unscaled = f32[16,32] multiply(x_f32, x_scale_bcast)
      y_unscaled = f32[32,16] multiply(y_f32, y_scale_bcast)
      k = f32[] constant(3.0)
      k_bcast = f32[16,16] broadcast(k), dimensions={}
      dot_a = f32[16,16] dot(x_unscaled, y_unscaled), lhs_contracting_dims={1}, rhs_contracting_dims={0}
      ROOT out = f32[16,16] multiply(dot_a, k_bcast)
          }

)";

  CheckFp8IfSupported(hlo_text);
  RunAndFilecheckHloRewrite(
      hlo_text,
      GemmRewriter(CudaHopperOrRocmMI300(), GetToolkitVersion(),
                   GemmRewriterOptions{GemmRewriterOptions::DType::kFp8Only}),
      R"(

; CHECK-LABEL: ENTRY %test ({{.*}}: <<F8E4M3>>[16,32], {{.*}}: <<F8E4M3>>[32,16], {{.*}}: f32[], {{.*}}: f32[]) -> f32[16,16] {
; CHECK-NEXT:    [[P0:%[^ ]+]] = <<F8E4M3>>[16,32]{1,0} parameter(0)
; CHECK-NEXT:    [[P1:%[^ ]+]] = <<F8E4M3>>[32,16]{1,0} parameter(1)
; CHECK-NEXT:    [[P1_TRANSPOSE:%[^ ]+]] = <<F8E4M3>>[16,32]{1,0} transpose([[P1]]), dimensions={1,0}
; CHECK-NEXT:    [[P2:%[^ ]+]] = f32[] parameter(2)
; CHECK-NEXT:    [[P3:%[^ ]+]] = f32[] parameter(3)
; CHECK-NEXT:    [[OUT:%[^ ]+]] = (f32[16,16]{1,0}, s8[{{[0-9]+}}]{0}) custom-call([[P0]], [[P1_TRANSPOSE]], [[P2]], [[P3]]),
; CHECK:           custom_call_target="__cublas$lt$matmul$f8",
; CHECK:           backend_config={
; CHECK-DAG:         "alpha_real":3
; CHECK-DAG:         "alpha_imag":0
; CHECK-DAG:         "beta":0
; CHECK-DAG:         "dot_dimension_numbers":{
; CHECK-DAG:           "lhs_contracting_dimensions":["1"]
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

TEST_P(ParameterizedFp8GemmRewriteTest, ScaledABUnscaledDReluActivationF8) {
  const char* hlo_text = R"(
    HloModule test

    ENTRY test {
      x = <<F8E4M3>>[16,32] parameter(0)
      y = <<F8E4M3>>[32,16] parameter(1)
      x_f32 = f32[16,32] convert(x)
      y_f32 = f32[32,16] convert(y)
      x_scale = f32[] parameter(2)
      y_scale = f32[] parameter(3)
      x_scale_bcast = f32[16,32] broadcast(x_scale), dimensions={}
      y_scale_bcast = f32[32,16] broadcast(y_scale), dimensions={}
      x_unscaled = f32[16,32] multiply(x_f32, x_scale_bcast)
      y_unscaled = f32[32,16] multiply(y_f32, y_scale_bcast)
      dot_a = f32[16,16] dot(x_unscaled, y_unscaled), lhs_contracting_dims={1}, rhs_contracting_dims={0}
      c = f32[] constant(0)
      c_bcast = f32[16,16] broadcast(c), dimensions={}
      ROOT out = f32[16,16] maximum(dot_a, c_bcast)
          }

)";

  CheckFp8IfSupported(hlo_text);
  RunAndFilecheckHloRewrite(
      hlo_text,
      GemmRewriter(CudaHopperOrRocmMI300(), GetToolkitVersion(),
                   GemmRewriterOptions{GemmRewriterOptions::DType::kFp8Only}),
      R"(

; CHECK-LABEL: ENTRY %test ({{.*}}: <<F8E4M3>>[16,32], {{.*}}: <<F8E4M3>>[32,16], {{.*}}: f32[], {{.*}}: f32[]) -> f32[16,16] {
; CHECK-NEXT:    [[P0:%[^ ]+]] = <<F8E4M3>>[16,32]{1,0} parameter(0)
; CHECK-NEXT:    [[P1:%[^ ]+]] = <<F8E4M3>>[32,16]{1,0} parameter(1)
; CHECK-NEXT:    [[P1_TRANSPOSE:%[^ ]+]] = <<F8E4M3>>[16,32]{1,0} transpose([[P1]]), dimensions={1,0}
; CHECK-NEXT:    [[P2:%[^ ]+]] = f32[] parameter(2)
; CHECK-NEXT:    [[P3:%[^ ]+]] = f32[] parameter(3)
; CHECK-NEXT:    [[OUT:%[^ ]+]] = (f32[16,16]{1,0}, s8[{{[0-9]+}}]{0}) custom-call([[P0]], [[P1_TRANSPOSE]], [[P2]], [[P3]]),
; CHECK:           custom_call_target="__cublas$lt$matmul$f8",
; CHECK:           backend_config={
; CHECK-DAG:         "alpha_real":1
; CHECK-DAG:         "alpha_imag":0
; CHECK-DAG:         "beta":0
; CHECK-DAG:         "dot_dimension_numbers":{
; CHECK-DAG:           "lhs_contracting_dimensions":["1"]
; CHECK-DAG:           "rhs_contracting_dimensions":["1"]
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

TEST_P(ParameterizedFp8GemmRewriteTest,
       ScaledABUnscaledDVectorBiasThenApproxGeluActivationF8) {
  const char* hlo_text = R"(
    HloModule test
    ENTRY test {
      x = <<F8E4M3>>[16,32] parameter(0)
      y = <<F8E4M3>>[32,16] parameter(1)
      x_bf16 = bf16[16,32] convert(x)
      y_bf16 = bf16[32,16] convert(y)
      x_scale = bf16[] parameter(2)
      y_scale = bf16[] parameter(3)
      bias = bf16[16] parameter(4)
      x_scale_bcast = bf16[16,32] broadcast(x_scale), dimensions={}
      y_scale_bcast = bf16[32,16] broadcast(y_scale), dimensions={}
      x_unscaled = bf16[16,32] multiply(x_bf16, x_scale_bcast)
      y_unscaled = bf16[32,16] multiply(y_bf16, y_scale_bcast)
      dot1 = bf16[16,16] dot(x_unscaled, y_unscaled), lhs_contracting_dims={1}, rhs_contracting_dims={0}
      b_bcast = bf16[16,16] broadcast(bias), dimensions={1}
      dot = bf16[16,16] add(dot1, b_bcast)
      mul.0 = bf16[16,16] multiply(dot, dot)
      mul.1 = bf16[16,16] multiply(dot, mul.0)
      const.0 = bf16[] constant(0.044715)
      bcast.0 = bf16[16,16] broadcast(const.0), dimensions={}
      mul.2 = bf16[16,16] multiply(mul.1, bcast.0)
      add.0 = bf16[16,16] add(dot, mul.2)
      const.1 = bf16[] constant(0.797884583)
      bcast.1 = bf16[16,16] broadcast(const.1), dimensions={}
      mul.3 = bf16[16,16] multiply(add.0, bcast.1)
      tanh = bf16[16,16] tanh(mul.3)
      const.2 = bf16[] constant(1)
      bcast.2 = bf16[16,16] broadcast(const.2), dimensions={}
      add.2 = bf16[16,16] add(tanh, bcast.2)
      const.3 = bf16[] constant(0.5)
      bcast.3 = bf16[16,16] broadcast(const.3), dimensions={}
      mul.4 = bf16[16,16] multiply(add.2, bcast.3)
      ROOT out = bf16[16,16] multiply(dot, mul.4)
          }
)";

  CheckFp8IfSupported(hlo_text);

  // Fusing gelu into FP8 cublas matmuls is disabled on CUDA versions less
  // than 12.4.
  if ((IsCuda() && GetToolkitVersion() >= se::SemanticVersion{12, 4, 0}) ||
      IsRocm()) {
    std::string checks = R"(
; CHECK-LABEL: ENTRY %test ({{.*}}: <<F8E4M3>>[16,32], {{.*}}: <<F8E4M3>>[32,16], {{.*}}: bf16[], {{.*}}: bf16[], {{.*}}: bf16[16]) -> bf16[16,16] {
; CHECK-NEXT:    [[P0:%[^ ]+]] = <<F8E4M3>>[16,32]{1,0} parameter(0)
; CHECK-NEXT:    [[P1:%[^ ]+]] = <<F8E4M3>>[32,16]{1,0} parameter(1)
; CHECK-NEXT:    [[P1_TRANSPOSE:%[^ ]+]] = <<F8E4M3>>[16,32]{1,0} transpose([[P1]]), dimensions={1,0}
; CHECK-NEXT:    [[P2:%[^ ]+]] = bf16[] parameter(2)
; CHECK-NEXT:    [[XS:%[^ ]+]] = f32[] convert([[P2]])
; CHECK-NEXT:    [[P3:%[^ ]+]] = bf16[] parameter(3)
; CHECK-NEXT:    [[XS1:%[^ ]+]] = f32[] convert([[P3]])
)";
    if (IsRocm() && GetToolkitVersion() < se::SemanticVersion{6, 2, 0}) {
      checks +=
          R"(; CHECK-GCN-NEXT:    [[OUT:%[^ ]+]] = (f32[16,16]{1,0}, s8[{{[0-9]+}}]{0}) custom-call([[P0]], [[P1_TRANSPOSE]], [[XS]], [[XS1]]),
)";
    } else {
      checks += R"(; CHECK-NEXT:    [[B:%[^ ]+]] = bf16[16]{0} parameter(4)
; CHECK-NEXT:    [[OUT:%[^ ]+]] = (bf16[16,16]{1,0}, s8[{{[0-9]+}}]{0}) custom-call([[P0]], [[P1_TRANSPOSE]], [[XS]], [[XS1]], [[B]]),
)";
    }
    checks += R"(; CHECK:           custom_call_target="__cublas$lt$matmul$f8",
; CHECK:           backend_config={
; CHECK-DAG:         "alpha_real":1
; CHECK-DAG:         "alpha_imag":0
; CHECK-DAG:         "beta":0
; CHECK-DAG:         "dot_dimension_numbers":{
; CHECK-DAG:           "lhs_contracting_dimensions":["1"]
; CHECK-DAG:           "rhs_contracting_dimensions":["1"]
; CHECK-DAG:           "lhs_batch_dimensions":[]
; CHECK-DAG:           "rhs_batch_dimensions":[]
; CHECK-DAG:         }
; CHECK-DAG:         "precision_config":{
; CHECK-DAG:           "operand_precision":["DEFAULT","DEFAULT"]
; CHECK-DAG:         }
)";
    if (IsRocm() && GetToolkitVersion() < se::SemanticVersion{6, 2, 0}) {
      checks +=
          R"(; CHECK-GCN-DAG:         "epilogue":"DEFAULT"
)";
    } else {
      checks +=
          R"(; CHECK-DAG:         "epilogue":"BIAS_GELU"
)";
    }
    checks += R"(; CHECK:           }
      )";

    RunAndFilecheckHloRewrite(
        hlo_text,
        GemmRewriter(CudaHopperOrRocmMI300(), GetToolkitVersion(),
                     GemmRewriterOptions{GemmRewriterOptions::DType::kFp8Only}),
        checks);
  }
}

TEST_P(ParameterizedFp8GemmRewriteTest,
       ScaledABUnscaledDApproxGeluActivationF8) {
  const char* hlo_text = R"(
    HloModule test
    ENTRY test {
      x = <<F8E4M3>>[16,32] parameter(0)
      y = <<F8E4M3>>[32,16] parameter(1)
      x_bf16 = bf16[16,32] convert(x)
      y_bf16 = bf16[32,16] convert(y)
      x_scale = bf16[] parameter(2)
      y_scale = bf16[] parameter(3)
      x_scale_bcast = bf16[16,32] broadcast(x_scale), dimensions={}
      y_scale_bcast = bf16[32,16] broadcast(y_scale), dimensions={}
      x_unscaled = bf16[16,32] multiply(x_bf16, x_scale_bcast)
      y_unscaled = bf16[32,16] multiply(y_bf16, y_scale_bcast)
      dot = bf16[16,16] dot(x_unscaled, y_unscaled), lhs_contracting_dims={1}, rhs_contracting_dims={0}
      mul.0 = bf16[16,16] multiply(dot, dot)
      mul.1 = bf16[16,16] multiply(dot, mul.0)
      const.0 = bf16[] constant(0.044715)
      bcast.0 = bf16[16,16] broadcast(const.0), dimensions={}
      mul.2 = bf16[16,16] multiply(mul.1, bcast.0)
      add.0 = bf16[16,16] add(dot, mul.2)
      const.1 = bf16[] constant(0.797884583)
      bcast.1 = bf16[16,16] broadcast(const.1), dimensions={}
      mul.3 = bf16[16,16] multiply(add.0, bcast.1)
      tanh = bf16[16,16] tanh(mul.3)
      const.2 = bf16[] constant(1)
      bcast.2 = bf16[16,16] broadcast(const.2), dimensions={}
      add.2 = bf16[16,16] add(tanh, bcast.2)
      const.3 = bf16[] constant(0.5)
      bcast.3 = bf16[16,16] broadcast(const.3), dimensions={}
      mul.4 = bf16[16,16] multiply(add.2, bcast.3)
      ROOT out = bf16[16,16] multiply(dot, mul.4)
          }
)";

  CheckFp8IfSupported(hlo_text);

  // Fusing gelu into FP8 cublas matmuls is disabled on CUDA versions less
  // than 12.4.
  if ((IsCuda() && GetToolkitVersion() >= se::SemanticVersion{12, 4, 0}) ||
      IsRocm()) {
    // Currently, hipBlasLt does not support output datatype bf16 for fp8
    // matmul. And no fusion was done for such cases.
    std::string checks =
        R"(
; CHECK-LABEL: ENTRY %test ({{.*}}: <<F8E4M3>>[16,32], {{.*}}: <<F8E4M3>>[32,16], {{.*}}: bf16[], {{.*}}: bf16[]) -> bf16[16,16] {
; CHECK-NEXT:    [[P0:%[^ ]+]] = <<F8E4M3>>[16,32]{1,0} parameter(0)
; CHECK-NEXT:    [[P1:%[^ ]+]] = <<F8E4M3>>[32,16]{1,0} parameter(1)
; CHECK-NEXT:    [[P1_TRANSPOSE:%[^ ]+]] = <<F8E4M3>>[16,32]{1,0} transpose([[P1]]), dimensions={1,0}
; CHECK-NEXT:    [[P2:%[^ ]+]] = bf16[] parameter(2)
; CHECK-NEXT:    [[XS:%[^ ]+]] = f32[] convert([[P2]])
; CHECK-NEXT:    [[P3:%[^ ]+]] = bf16[] parameter(3)
; CHECK-NEXT:    [[XS1:%[^ ]+]] = f32[] convert([[P3]])
)";
    if (IsRocm() && GetToolkitVersion() < se::SemanticVersion{6, 2, 0}) {
      checks +=
          R"(; CHECK-GCN-NEXT:    [[OUT:%[^ ]+]] = (f32[16,16]{1,0}, s8[{{[0-9]+}}]{0}) custom-call([[P0]], [[P1_TRANSPOSE]], [[XS]], [[XS1]]),
)";
    } else {
      checks +=
          R"(; CHECK-NEXT:    [[OUT:%[^ ]+]] = (bf16[16,16]{1,0}, s8[{{[0-9]+}}]{0}) custom-call([[P0]], [[P1_TRANSPOSE]], [[XS]], [[XS1]]),
)";
    }
    checks += R"(; CHECK:           custom_call_target="__cublas$lt$matmul$f8",
; CHECK:           backend_config={
; CHECK-DAG:         "alpha_real":1
; CHECK-DAG:         "alpha_imag":0
; CHECK-DAG:         "beta":0
; CHECK-DAG:         "dot_dimension_numbers":{
; CHECK-DAG:           "lhs_contracting_dimensions":["1"]
; CHECK-DAG:           "rhs_contracting_dimensions":["1"]
; CHECK-DAG:           "lhs_batch_dimensions":[]
; CHECK-DAG:           "rhs_batch_dimensions":[]
; CHECK-DAG:         }
; CHECK-DAG:         "precision_config":{
; CHECK-DAG:           "operand_precision":["DEFAULT","DEFAULT"]
; CHECK-DAG:         }
)";
    if (IsRocm() && GetToolkitVersion() < se::SemanticVersion{6, 2, 0}) {
      checks += R"(; CHECK-GCN-DAG:         "epilogue":"DEFAULT"
)";
    } else {
      checks += R"(; CHECK-DAG:         "epilogue":"GELU"
)";
    }
    checks += R"(; CHECK:           }
      )";
    RunAndFilecheckHloRewrite(
        hlo_text,
        GemmRewriter(CudaHopperOrRocmMI300(), GetToolkitVersion(),
                     GemmRewriterOptions{GemmRewriterOptions::DType::kFp8Only}),
        checks);
  }
}

TEST_P(ParameterizedFp8GemmRewriteTest, InvScaledABUnscaledDF8) {
  const char* hlo_text = R"(
    HloModule test

    ENTRY test {
      x = <<F8E4M3>>[16,32] parameter(0)
      y = <<F8E4M3>>[32,16] parameter(1)
      x_f32 = f32[16,32] convert(x)
      y_f32 = f32[32,16] convert(y)
      x_scale = f32[] parameter(2)
      y_scale = f32[] parameter(3)
      x_scale_bcast = f32[16,32] broadcast(x_scale), dimensions={}
      y_scale_bcast = f32[32,16] broadcast(y_scale), dimensions={}
      x_unscaled = f32[16,32] divide(x_f32, x_scale_bcast)
      y_unscaled = f32[32,16] divide(y_f32, y_scale_bcast)
      ROOT out = f32[16,16] dot(x_unscaled, y_unscaled), lhs_contracting_dims={1}, rhs_contracting_dims={0}
          }

)";

  CheckFp8IfSupported(hlo_text);
  RunAndFilecheckHloRewrite(
      hlo_text,
      GemmRewriter(CudaHopperOrRocmMI300(), GetToolkitVersion(),
                   GemmRewriterOptions{GemmRewriterOptions::DType::kFp8Only}),
      R"(
; CHECK:           custom_call_target="__cublas$lt$matmul$f8",
      )");
}

TEST_P(ParameterizedFp8GemmRewriteTest, ScaledABUnscaledDMatrixBiasF8) {
  const char* hlo_text = R"(
    HloModule test

    ENTRY test {
      x = <<F8E4M3>>[16,32] parameter(0)
      y = <<F8E4M3>>[32,16] parameter(1)
      b = f32[16,16] parameter(2)
      one = f32[] constant(1)
      ones = f32[16,16] broadcast(one), dimensions={}
      b_ones = f32[16,16] add(b, ones)
      x_f32 = f32[16,32] convert(x)
      y_f32 = f32[32,16] convert(y)
      x_scale = f32[] parameter(3)
      y_scale = f32[] parameter(4)
      x_scale_bcast = f32[16,32] broadcast(x_scale), dimensions={}
      y_scale_bcast = f32[32,16] broadcast(y_scale), dimensions={}
      x_unscaled = f32[16,32] multiply(x_f32, x_scale_bcast)
      y_unscaled = f32[32,16] multiply(y_f32, y_scale_bcast)
      dot_a = f32[16,16] dot(x_unscaled, y_unscaled), lhs_contracting_dims={1}, rhs_contracting_dims={0}
      ROOT out = add(dot_a, b_ones)
          }

)";

  CheckFp8IfSupported(hlo_text);
  RunAndFilecheckHloRewrite(
      hlo_text,
      GemmRewriter(CudaHopperOrRocmMI300(), GetToolkitVersion(),
                   GemmRewriterOptions{GemmRewriterOptions::DType::kFp8Only}),
      R"(

; CHECK-LABEL: ENTRY %test ({{.*}}: <<F8E4M3>>[16,32], {{.*}}: <<F8E4M3>>[32,16], {{.*}}: f32[16,16], {{.*}}: f32[], {{.*}}: f32[]) -> f32[16,16] {
; CHECK-NEXT:    [[P0:%[^ ]+]] = <<F8E4M3>>[16,32]{1,0} parameter(0)
; CHECK-NEXT:    [[P1:%[^ ]+]] = <<F8E4M3>>[32,16]{1,0} parameter(1)
; CHECK-NEXT:    [[P1_TRANSPOSE:%[^ ]+]] = <<F8E4M3>>[16,32]{1,0} transpose([[P1]]), dimensions={1,0}
; CHECK:         [[C0:%[^ ]+]] = f32[16,16]{1,0} add({{.*}})
; CHECK-NEXT:    [[P2:%[^ ]+]] = f32[] parameter(3)
; CHECK-NEXT:    [[P3:%[^ ]+]] = f32[] parameter(4)
; CHECK-NEXT:    [[OUT:%[^ ]+]] = (f32[16,16]{1,0}, s8[{{[0-9]+}}]{0}) custom-call([[P0]], [[P1_TRANSPOSE]], [[C0]], [[P2]], [[P3]]),
; CHECK:           custom_call_target="__cublas$lt$matmul$f8",
; CHECK:           output_to_operand_aliasing={
; CHECK-SAME:        {0}: (2, {})
; CHECK-SAME:      }
; CHECK:           backend_config={
; CHECK-DAG:         "alpha_real":1
; CHECK-DAG:         "alpha_imag":0
; CHECK-DAG:         "beta":1
; CHECK-DAG:         "dot_dimension_numbers":{
; CHECK-DAG:           "lhs_contracting_dimensions":["1"]
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

TEST_P(ParameterizedFp8GemmRewriteTest, ScaledABUnscaledDMatrixBiasPaddedF8) {
  const char* hlo_text = R"(
    HloModule test

    ENTRY test {
      x = <<F8E4M3>>[14,31] parameter(0)
      y = <<F8E4M3>>[31,14] parameter(1)
      b = f32[14,14] parameter(2)
      x_f32 = f32[14,31] convert(x)
      y_f32 = f32[31,14] convert(y)
      x_scale = f32[] parameter(3)
      y_scale = f32[] parameter(4)
      x_scale_bcast = f32[14,31] broadcast(x_scale), dimensions={}
      y_scale_bcast = f32[31,14] broadcast(y_scale), dimensions={}
      x_unscaled = f32[14,31] multiply(x_f32, x_scale_bcast)
      y_unscaled = f32[31,14] multiply(y_f32, y_scale_bcast)
      dot_a = f32[14,14] dot(x_unscaled, y_unscaled), lhs_contracting_dims={1}, rhs_contracting_dims={0}
      ROOT out = add(dot_a, b)
          }

)";

  CheckFp8IfSupported(hlo_text);
  RunAndFilecheckHloRewrite(
      hlo_text,
      GemmRewriter(CudaHopperOrRocmMI300(), GetToolkitVersion(),
                   GemmRewriterOptions{GemmRewriterOptions::DType::kFp8Only}),
      R"(

; CHECK-LABEL: ENTRY %test ({{.*}}: <<F8E4M3>>[14,31], {{.*}}: <<F8E4M3>>[31,14], {{.*}}: f32[14,14], {{.*}}: f32[], {{.*}}: f32[]) -> f32[14,14] {
; CHECK-NEXT:    [[P0:%[^ ]+]] = <<F8E4M3>>[14,31]{1,0} parameter(0)
; CHECK-NEXT:    [[C0:%[^ ]+]] = <<F8E4M3>>[] constant(0)
; CHECK-NEXT:    [[P0_PADDED:%[^ ]+]] = <<F8E4M3>>[16,32]{1,0} pad([[P0]], [[C0]]), padding=0_2x0_1
; CHECK-NEXT:    [[P1:%[^ ]+]] = <<F8E4M3>>[31,14]{1,0} parameter(1)
; CHECK-NEXT:    [[P1_TRANSPOSE:%[^ ]+]] = <<F8E4M3>>[14,31]{1,0} transpose([[P1]]), dimensions={1,0}
; CHECK-NEXT:    [[C1:%[^ ]+]] = <<F8E4M3>>[] constant(0)
; CHECK-NEXT:    [[P1_TRANSPOSE_PADDED:%[^ ]+]] = <<F8E4M3>>[16,32]{1,0} pad([[P1_TRANSPOSE]], [[C1]]), padding=0_2x0_1
; CHECK-NEXT:    [[P2:%[^ ]+]] = f32[14,14]{1,0} parameter(2)
; CHECK-NEXT:    [[C2:%[^ ]+]] = f32[] constant(0)
; CHECK-NEXT:    [[P2_PADDED:%[^ ]+]] = f32[16,16]{1,0} pad([[P2]], [[C2]]), padding=0_2x0_2
; CHECK-NEXT:    [[P3:%[^ ]+]] = f32[] parameter(3)
; CHECK-NEXT:    [[P4:%[^ ]+]] = f32[] parameter(4)
; CHECK-NEXT:    [[DOT_TUPLE:%[^ ]+]] = (f32[16,16]{1,0}, s8[{{[0-9]+}}]{0}) custom-call([[P0_PADDED]], [[P1_TRANSPOSE_PADDED]], [[P2_PADDED]], [[P3]], [[P4]]),
; CHECK:           custom_call_target="__cublas$lt$matmul$f8",
; CHECK:           backend_config={
; CHECK-DAG:         "alpha_real":1
; CHECK-DAG:         "alpha_imag":0
; CHECK-DAG:         "beta":1
; CHECK-DAG:         "dot_dimension_numbers":{
; CHECK-DAG:           "lhs_contracting_dimensions":["1"]
; CHECK-DAG:           "rhs_contracting_dimensions":["1"]
; CHECK-DAG:           "lhs_batch_dimensions":[]
; CHECK-DAG:           "rhs_batch_dimensions":[]
; CHECK-DAG:         }
; CHECK-DAG:         "precision_config":{
; CHECK-DAG:           "operand_precision":["DEFAULT","DEFAULT"]
; CHECK-DAG:         }
; CHECK-DAG:         "epilogue":"DEFAULT"
; CHECK:           }
; CHECK:      [[DOT:%[^ ]+]] = f32[16,16]{1,0} get-tuple-element([[DOT_TUPLE]]), index=0
; CHECK-NEXT: ROOT [[OUT:%[^ ]+]] = f32[14,14]{1,0} slice([[DOT]]), slice={[0:14], [0:14]}
      )");
}

TEST_P(ParameterizedFp8GemmRewriteTest, UnscaledABScaledDF8) {
  const char* hlo_text = R"(
    HloModule test

    ENTRY test {
      x = <<F8E4M3>>[16,32] parameter(0)
      y = <<F8E4M3>>[32,16] parameter(1)
      z_scale = f32[] parameter(2)
      z_scale_bcast = f32[16,16] broadcast(z_scale), dimensions={}
      dot_a = f32[16,16] dot(x, y), lhs_contracting_dims={1}, rhs_contracting_dims={0}
      dot_a_scaled = f32[16,16] divide(dot_a, z_scale_bcast)
      c1 = f32[] constant(-448.)
      c1_bcast = f32[16,16] broadcast(c1), dimensions={}
      c2 = f32[] constant(448.)
      c2_bcast = f32[16,16] broadcast(c2), dimensions={}
      dot_a_clamped = f32[16,16] clamp(c1_bcast, dot_a_scaled, c2_bcast)
      ROOT dot_a_f8 = <<F8E4M3>>[16,16] convert(dot_a_clamped)
          }

)";

  CheckFp8IfSupported(hlo_text, ErrorSpec{1e-2, 1e-1});
  RunAndFilecheckHloRewrite(
      hlo_text,
      GemmRewriter(CudaHopperOrRocmMI300(), GetToolkitVersion(),
                   GemmRewriterOptions{GemmRewriterOptions::DType::kFp8Only}),
      R"(
; CHECK-LABEL: ENTRY %test ({{.*}}: <<F8E4M3>>[16,32], {{.*}}: <<F8E4M3>>[32,16], {{.*}}: f32[]) -> <<F8E4M3>>[16,16] {
; CHECK:         [[P0:%[^ ]+]] = <<F8E4M3>>[16,32]{1,0} parameter(0)
; CHECK-NEXT:    [[P1:%[^ ]+]] = <<F8E4M3>>[32,16]{1,0} parameter(1)
; CHECK-NEXT:    [[P1_TRANSPOSE:%[^ ]+]] = <<F8E4M3>>[16,32]{1,0} transpose([[P1]]), dimensions={1,0}
; CHECK-NEXT:    [[C0:%[^ ]+]] = f32[] constant(1)
; CHECK-NEXT:    [[P2:%[^ ]+]] = f32[] parameter(2)
; CHECK-NEXT:    [[P2_INV:%[^ ]+]] = f32[] divide([[C0]], [[P2]])
; CHECK-NEXT:    [[C1:%[^ ]+]] = f32[] constant(1)
; CHECK-PTX-NEXT:    [[C2:%[^ ]+]] = f32[] constant(1)
; CHECK-PTX-NEXT:    [[OUT:%[^ ]+]] = (<<F8E4M3>>[16,16]{1,0}, s8[{{[0-9]+}}]{0}) custom-call([[P0]], [[P1_TRANSPOSE]], [[P2_INV]], [[C1]], [[C2]]),
; CHECK-GCN-NEXT:    [[OUT:%[^ ]+]] = (f32[16,16]{1,0}, s8[{{[0-9]+}}]{0}) custom-call([[P0]], [[P1_TRANSPOSE]], [[P2_INV]], [[C1]]),
; CHECK:           custom_call_target="__cublas$lt$matmul$f8",
; CHECK:           backend_config={
; CHECK-DAG:         "alpha_real":1
; CHECK-DAG:         "alpha_imag":0
; CHECK-DAG:         "beta":0
; CHECK-DAG:         "dot_dimension_numbers":{
; CHECK-DAG:           "lhs_contracting_dimensions":["1"]
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

TEST_P(ParameterizedFp8GemmRewriteTest, UnscaledABScaledF32DF8) {
  const char* hlo_text = R"(
    HloModule test

    ENTRY test {
      x = <<F8E4M3>>[16,32] parameter(0)
      y = <<F8E4M3>>[32,16] parameter(1)
      z_scale = f32[] parameter(2)
      z_scale_bcast = f32[16,16] broadcast(z_scale), dimensions={}
      dot_a = f32[16,16] dot(x, y), lhs_contracting_dims={1}, rhs_contracting_dims={0}
      ROOT dot_a_scaled = f32[16,16] divide(dot_a, z_scale_bcast)
          }

)";

  CheckFp8IfSupported(hlo_text, ErrorSpec{1e-2, 1e-1});
  RunAndFilecheckHloRewrite(
      hlo_text,
      GemmRewriter(CudaHopperOrRocmMI300(), GetToolkitVersion(),
                   GemmRewriterOptions{GemmRewriterOptions::DType::kFp8Only}),
      R"(
; CHECK-LABEL: ENTRY %test ({{.*}}: <<F8E4M3>>[16,32], {{.*}}: <<F8E4M3>>[32,16], {{.*}}: f32[]) -> f32[16,16] {
; CHECK:         [[P0:%[^ ]+]] = <<F8E4M3>>[16,32]{1,0} parameter(0)
; CHECK-NEXT:    [[P1:%[^ ]+]] = <<F8E4M3>>[32,16]{1,0} parameter(1)
; CHECK-NEXT:    [[P1_TRANSPOSE:%[^ ]+]] = <<F8E4M3>>[16,32]{1,0} transpose([[P1]]), dimensions={1,0}
; CHECK-NEXT:    [[C0:%[^ ]+]] = f32[] constant(1)
; CHECK-NEXT:    [[P2:%[^ ]+]] = f32[] parameter(2)
; CHECK-NEXT:    [[P2_INV:%[^ ]+]] = f32[] divide([[C0]], [[P2]])
; CHECK-NEXT:    [[C1:%[^ ]+]] = f32[] constant(1)
; CHECK-NEXT:    [[OUT:%[^ ]+]] = (f32[16,16]{1,0}, s8[{{[0-9]+}}]{0}) custom-call([[P0]], [[P1_TRANSPOSE]], [[P2_INV]], [[C1]]),
; CHECK:           custom_call_target="__cublas$lt$matmul$f8",
; CHECK:           backend_config={
; CHECK-DAG:         "alpha_real":1
; CHECK-DAG:         "alpha_imag":0
; CHECK-DAG:         "beta":0
; CHECK-DAG:         "dot_dimension_numbers":{
; CHECK-DAG:           "lhs_contracting_dimensions":["1"]
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

TEST_P(ParameterizedFp8GemmRewriteTest, UnscaledABInvScaledF32DF8) {
  const char* hlo_text = R"(
    HloModule test

    ENTRY test {
      x = <<F8E4M3>>[16,32] parameter(0)
      y = <<F8E4M3>>[32,16] parameter(1)
      z_scale = f32[] parameter(2)
      z_scale_bcast = f32[16,16] broadcast(z_scale), dimensions={}
      dot_a = f32[16,16] dot(x, y), lhs_contracting_dims={1}, rhs_contracting_dims={0}
      ROOT dot_a_scaled = f32[16,16] multiply(dot_a, z_scale_bcast)
          }

)";

  CheckFp8IfSupported(hlo_text, ErrorSpec{1e-2, 1e-1});
  RunAndFilecheckHloRewrite(
      hlo_text,
      GemmRewriter(CudaHopperOrRocmMI300(), GetToolkitVersion(),
                   GemmRewriterOptions{GemmRewriterOptions::DType::kFp8Only}),
      R"(
; CHECK-LABEL: ENTRY %test ({{.*}}: <<F8E4M3>>[16,32], {{.*}}: <<F8E4M3>>[32,16], {{.*}}: f32[]) -> f32[16,16] {
; CHECK:         [[P0:%[^ ]+]] = <<F8E4M3>>[16,32]{1,0} parameter(0)
; CHECK-NEXT:    [[P1:%[^ ]+]] = <<F8E4M3>>[32,16]{1,0} parameter(1)
; CHECK-NEXT:    [[P1_TRANSPOSE:%[^ ]+]] = <<F8E4M3>>[16,32]{1,0} transpose([[P1]]), dimensions={1,0}
; CHECK-NEXT:    [[P2:%[^ ]+]] = f32[] parameter(2)
; CHECK-NEXT:    [[C0:%[^ ]+]] = f32[] constant(1)
; CHECK-NEXT:    [[OUT:%[^ ]+]] = (f32[16,16]{1,0}, s8[{{[0-9]+}}]{0}) custom-call([[P0]], [[P1_TRANSPOSE]], [[P2]], [[C0]]),
; CHECK:           custom_call_target="__cublas$lt$matmul$f8",
; CHECK:           backend_config={
; CHECK-DAG:         "alpha_real":1
; CHECK-DAG:         "alpha_imag":0
; CHECK-DAG:         "beta":0
; CHECK-DAG:         "dot_dimension_numbers":{
; CHECK-DAG:           "lhs_contracting_dimensions":["1"]
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

// Do not fuse output scaling without type conversion when a matrix bias was
// fused.
TEST_P(ParameterizedFp8GemmRewriteTest, UnscaledABScaledF32DMatrixBiasF8) {
  const char* hlo_text = R"(
    HloModule test

    ENTRY test {
      x = <<F8E4M3>>[16,32] parameter(0)
      y = <<F8E4M3>>[32,16] parameter(1)
      b = f32[16,16] parameter(2)
      z_scale = f32[] parameter(3)
      z_scale_bcast = f32[16,16] broadcast(z_scale), dimensions={}
      dot_a = f32[16,16] dot(x, y), lhs_contracting_dims={1}, rhs_contracting_dims={0}
      dot_a_bias = f32[16,16] add(dot_a, b)
      ROOT dot_a_scaled = f32[16,16] divide(dot_a_bias, z_scale_bcast)
          }

)";

  CheckFp8IfSupported(hlo_text, ErrorSpec{1e-2, 1e-1});
  RunAndFilecheckHloRewrite(
      hlo_text,
      GemmRewriter(CudaHopperOrRocmMI300(), GetToolkitVersion(),
                   GemmRewriterOptions{GemmRewriterOptions::DType::kFp8Only}),
      R"(
; CHECK-LABEL: ENTRY %test ({{.*}}: <<F8E4M3>>[16,32], {{.*}}: <<F8E4M3>>[32,16], {{.*}}: f32[]) -> f32[16,16] {
; CHECK:         [[P0:%[^ ]+]] = <<F8E4M3>>[16,32]{1,0} parameter(0)
; CHECK-NEXT:    [[P1:%[^ ]+]] = <<F8E4M3>>[32,16]{1,0} parameter(1)
; CHECK-NEXT:    [[P1_TRANSPOSE:%[^ ]+]] = <<F8E4M3>>[16,32]{1,0} transpose([[P1]]), dimensions={1,0}
; CHECK-NEXT:    [[P2:%[^ ]+]] = f32[16,16]{1,0} parameter(2)
; CHECK-NEXT:    [[C0:%[^ ]+]] = f32[] constant(1)
; CHECK-NEXT:    [[GEMM_TUPLE:%[^ ]+]] = (f32[16,16]{1,0}, s8[{{[0-9]+}}]{0}) custom-call([[P0]], [[P1_TRANSPOSE]], [[P2]], [[C0]], [[C0]]),
; CHECK:           custom_call_target="__cublas$lt$matmul$f8",
; CHECK:           backend_config={
; CHECK-DAG:         "alpha_real":1
; CHECK-DAG:         "alpha_imag":0
; CHECK-DAG:         "beta":1
; CHECK-DAG:         "dot_dimension_numbers":{
; CHECK-DAG:           "lhs_contracting_dimensions":["1"]
; CHECK-DAG:           "rhs_contracting_dimensions":["1"]
; CHECK-DAG:           "lhs_batch_dimensions":[]
; CHECK-DAG:           "rhs_batch_dimensions":[]
; CHECK-DAG:         }
; CHECK-DAG:         "precision_config":{
; CHECK-DAG:           "operand_precision":["DEFAULT","DEFAULT"]
; CHECK-DAG:         }
; CHECK-DAG:         "epilogue":"DEFAULT"
; CHECK-PTX-NEXT:    [[GEMM:%[^ ]+]] = f32[16,16]{1,0} get-tuple-element([[GEMM_TUPLE]]), index=0
; CHECK-PTX-NEXT:    [[P3:%[^ ]+]] = f32[] parameter(3)
; CHECK-PTX-NEXT:    [[P3_BCAST:%[^ ]+]] = f32[16,16]{1,0} broadcast([[P3]]), dimensions={}
; CHECK-PTX-NEXT:    ROOT [[OUT:%[^ ]+]] = f32[16,16]{1,0} divide([[GEMM]], [[P3_BCAST]])
; CHECK:           }
      )");
}

TEST_P(ParameterizedFp8GemmRewriteTest, ScaledABScaledDF8) {
  const char* hlo_text = R"(
    HloModule test

    ENTRY test {
      x = <<F8E4M3>>[16,32] parameter(0)
      y = <<F8E4M3>>[32,16] parameter(1)
      x_f32 = f32[16,32] convert(x)
      y_f32 = f32[32,16] convert(y)
      x_scale = f32[] parameter(2)
      y_scale = f32[] parameter(3)
      z_scale = f32[] parameter(4)
      x_scale_bcast = f32[16,32] broadcast(x_scale), dimensions={}
      y_scale_bcast = f32[32,16] broadcast(y_scale), dimensions={}
      z_scale_bcast = f32[16,16] broadcast(z_scale), dimensions={}
      x_unscaled = f32[16,32] multiply(x_f32, x_scale_bcast)
      y_unscaled = f32[32,16] multiply(y_f32, y_scale_bcast)
      dot_a = f32[16,16] dot(x_unscaled, y_unscaled), lhs_contracting_dims={1}, rhs_contracting_dims={0}
      dot_a_scaled = f32[16,16] divide(dot_a, z_scale_bcast)
      c1 = f32[] constant(-<<F8E4M3_AMAX>>)
      c1_bcast = f32[16,16] broadcast(c1), dimensions={}
      c2 = f32[] constant(<<F8E4M3_AMAX>>)
      c2_bcast = f32[16,16] broadcast(c2), dimensions={}
      dot_a_clamped = f32[16,16] clamp(c1_bcast, dot_a_scaled, c2_bcast)
      ROOT dot_a_f8 = <<F8E4M3>>[16,16] convert(dot_a_clamped)
          }

)";

  CheckFp8IfSupported(hlo_text);
  RunAndFilecheckHloRewrite(
      hlo_text,
      GemmRewriter(CudaHopperOrRocmMI300(), GetToolkitVersion(),
                   GemmRewriterOptions{GemmRewriterOptions::DType::kFp8Only}),
      R"(
; CHECK-LABEL: ENTRY %test ({{.*}}: <<F8E4M3>>[16,32], {{.*}}: <<F8E4M3>>[32,16], {{.*}}: f32[], {{.*}}: f32[], {{.*}}: f32[]) -> <<F8E4M3>>[16,16] {
; CHECK:         [[P0:%[^ ]+]] = <<F8E4M3>>[16,32]{1,0} parameter(0)
; CHECK-NEXT:    [[P1:%[^ ]+]] = <<F8E4M3>>[32,16]{1,0} parameter(1)
; CHECK-NEXT:    [[P1_TRANSPOSE:%[^ ]+]] = <<F8E4M3>>[16,32]{1,0} transpose([[P1]]), dimensions={1,0}
; CHECK-NEXT:    [[P2:%[^ ]+]] = f32[] parameter(2)
; CHECK-NEXT:    [[P3:%[^ ]+]] = f32[] parameter(3)
; CHECK-PTX-NEXT:    [[C2:%[^ ]+]] = f32[] constant(1)
; CHECK-PTX-NEXT:    [[P4:%[^ ]+]] = f32[] parameter(4)
; CHECK-PTX-NEXT:    [[P4_INV:%[^ ]+]] = f32[] divide([[C2]], [[P4]])
; CHECK-PTX-NEXT:    [[OUT:%[^ ]+]] = (<<F8E4M3>>[16,16]{1,0}, s8[{{[0-9]+}}]{0}) custom-call([[P0]], [[P1_TRANSPOSE]], [[P2]], [[P3]], [[P4_INV]]),
; CHECK-GCN-NEXT:    [[OUT:%[^ ]+]] = (f32[16,16]{1,0}, s8[{{[0-9]+}}]{0}) custom-call([[P0]], [[P1_TRANSPOSE]], [[P2]], [[P3]]),
; CHECK:           custom_call_target="__cublas$lt$matmul$f8",
; CHECK:           backend_config={
; CHECK-DAG:         "alpha_real":1
; CHECK-DAG:         "alpha_imag":0
; CHECK-DAG:         "beta":0
; CHECK-DAG:         "dot_dimension_numbers":{
; CHECK-DAG:           "lhs_contracting_dimensions":["1"]
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

TEST_P(ParameterizedFp8GemmRewriteTest, ScaledABInvScaledDF8) {
  const char* hlo_text = R"(
    HloModule test

    ENTRY test {
      x = <<F8E4M3>>[16,32] parameter(0)
      y = <<F8E4M3>>[32,16] parameter(1)
      x_f32 = f32[16,32] convert(x)
      y_f32 = f32[32,16] convert(y)
      x_scale = f32[] parameter(2)
      y_scale = f32[] parameter(3)
      z_scale = f32[] parameter(4)
      x_scale_bcast = f32[16,32] broadcast(x_scale), dimensions={}
      y_scale_bcast = f32[32,16] broadcast(y_scale), dimensions={}
      z_scale_bcast = f32[16,16] broadcast(z_scale), dimensions={}
      x_unscaled = f32[16,32] multiply(x_f32, x_scale_bcast)
      y_unscaled = f32[32,16] multiply(y_f32, y_scale_bcast)
      dot_a = f32[16,16] dot(x_unscaled, y_unscaled), lhs_contracting_dims={1}, rhs_contracting_dims={0}
      dot_a_scaled = f32[16,16] multiply(dot_a, z_scale_bcast)
      c1 = f32[] constant(-<<F8E4M3_AMAX>>)
      c1_bcast = f32[16,16] broadcast(c1), dimensions={}
      c2 = f32[] constant(<<F8E4M3_AMAX>>)
      c2_bcast = f32[16,16] broadcast(c2), dimensions={}
      dot_a_clamped = f32[16,16] clamp(c1_bcast, dot_a_scaled, c2_bcast)
      ROOT dot_a_f8 = <<F8E4M3>>[16,16] convert(dot_a_clamped)
          }

)";

  CheckFp8IfSupported(hlo_text);
  RunAndFilecheckHloRewrite(
      hlo_text,
      GemmRewriter(CudaHopperOrRocmMI300(), GetToolkitVersion(),
                   GemmRewriterOptions{GemmRewriterOptions::DType::kFp8Only}),
      R"(

; CHECK-NOT:     divide

; CHECK:           custom_call_target="__cublas$lt$matmul$f8",

      )");
}

TEST_P(ParameterizedFp8GemmRewriteTest, ScaledABScaledDReluActivationF8) {
  const char* hlo_text = R"(
    HloModule test
    ENTRY test {
      x = <<F8E4M3>>[16,32] parameter(0)
      y = <<F8E4M3>>[32,16] parameter(1)
      x_f32 = f32[16,32] convert(x)
      y_f32 = f32[32,16] convert(y)
      x_scale = f32[] parameter(2)
      y_scale = f32[] parameter(3)
      z_scale = f32[] parameter(4)
      x_scale_bcast = f32[16,32] broadcast(x_scale), dimensions={}
      y_scale_bcast = f32[32,16] broadcast(y_scale), dimensions={}
      z_scale_bcast = f32[16,16] broadcast(z_scale), dimensions={}
      x_unscaled = f32[16,32] multiply(x_f32, x_scale_bcast)
      y_unscaled = f32[32,16] multiply(y_f32, y_scale_bcast)
      c = f32[] constant(0)
      c_bcast = f32[16,16] broadcast(c), dimensions={}
      dot_a = f32[16,16] dot(x_unscaled, y_unscaled), lhs_contracting_dims={1}, rhs_contracting_dims={0}
      relu_a = f32[16,16] maximum(dot_a, c_bcast)
      relu_a_scaled = f32[16,16] divide(relu_a, z_scale_bcast)
      c1 = f32[] constant(-<<F8E4M3_AMAX>>)
      c1_bcast = f32[16,16] broadcast(c1), dimensions={}
      c2 = f32[] constant(<<F8E4M3_AMAX>>)
      c2_bcast = f32[16,16] broadcast(c2), dimensions={}
      relu_a_clamped = f32[16,16] clamp(c1_bcast, relu_a_scaled, c2_bcast)
      ROOT out = <<F8E4M3>>[16,16] convert(relu_a_clamped)
          }
)";

  CheckFp8IfSupported(hlo_text);
  RunAndFilecheckHloRewrite(
      hlo_text,
      GemmRewriter(CudaHopperOrRocmMI300(), GetToolkitVersion(),
                   GemmRewriterOptions{GemmRewriterOptions::DType::kFp8Only}),
      R"(
; CHECK-LABEL: ENTRY %test ({{.*}}: <<F8E4M3>>[16,32], {{.*}}: <<F8E4M3>>[32,16], {{.*}}: f32[], {{.*}}: f32[], {{.*}}: f32[]) -> <<F8E4M3>>[16,16] {
; CHECK:         [[P0:%[^ ]+]] = <<F8E4M3>>[16,32]{1,0} parameter(0)
; CHECK-NEXT:    [[P1:%[^ ]+]] = <<F8E4M3>>[32,16]{1,0} parameter(1)
; CHECK-NEXT:    [[P1_TRANSPOSE:%[^ ]+]] = <<F8E4M3>>[16,32]{1,0} transpose([[P1]]), dimensions={1,0}
; CHECK-NEXT:    [[P2:%[^ ]+]] = f32[] parameter(2)
; CHECK-NEXT:    [[P3:%[^ ]+]] = f32[] parameter(3)
; CHECK-PTX-NEXT:    [[C2:%[^ ]+]] = f32[] constant(1)
; CHECK-PTX-NEXT:    [[P4:%[^ ]+]] = f32[] parameter(4)
; CHECK-PTX-NEXT:    [[P4_INV:%[^ ]+]] = f32[] divide([[C2]], [[P4]])
; CHECK-PTX-NEXT:    [[OUT:%[^ ]+]] = (<<F8E4M3>>[16,16]{1,0}, s8[{{[0-9]+}}]{0}) custom-call([[P0]], [[P1_TRANSPOSE]], [[P2]], [[P3]], [[P4_INV]]),
; CHECK-GCN-NEXT:    [[OUT:%[^ ]+]] = (f32[16,16]{1,0}, s8[{{[0-9]+}}]{0}) custom-call([[P0]], [[P1_TRANSPOSE]], [[P2]], [[P3]]),
; CHECK:           custom_call_target="__cublas$lt$matmul$f8",
; CHECK:           backend_config={
; CHECK-DAG:         "alpha_real":1
; CHECK-DAG:         "alpha_imag":0
; CHECK-DAG:         "beta":0
; CHECK-DAG:         "dot_dimension_numbers":{
; CHECK-DAG:           "lhs_contracting_dimensions":["1"]
; CHECK-DAG:           "rhs_contracting_dimensions":["1"]
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

TEST_P(ParameterizedFp8GemmRewriteTest, ScaledABScaledDMatrixBiasWithDAmaxF8) {
  const char* hlo_text = R"(
    HloModule test

    apply {
      a = f16[] parameter(0)
      b = f16[] parameter(1)
      ROOT c = f16[] maximum(a, b)
    }

    ENTRY test {
      x = <<F8E4M3>>[16,32] parameter(0)
      y = <<F8E4M3>>[32,16] parameter(1)
      x_f16 = f16[16,32] convert(x)
      y_f16 = f16[32,16] convert(y)
      b = f16[16,16] parameter(2)
      one = f16[] constant(1)
      ones = f16[16,16] broadcast(one), dimensions={}
      b_ones = f16[16,16] add(b, ones)
      x_scale = f16[] parameter(3)
      y_scale = f16[] parameter(4)
      z_scale = f16[] parameter(5)
      x_scale_bcast = f16[16,32] broadcast(x_scale), dimensions={}
      y_scale_bcast = f16[32,16] broadcast(y_scale), dimensions={}
      z_scale_bcast = f16[16,16] broadcast(z_scale), dimensions={}
      x_unscaled = f16[16,32] multiply(x_f16, x_scale_bcast)
      y_unscaled = f16[32,16] multiply(y_f16, y_scale_bcast)
      dot_a = f16[16,16] dot(x_unscaled, y_unscaled), lhs_contracting_dims={1}, rhs_contracting_dims={0}
      dot_a_bias = f16[16,16] add(dot_a, b_ones)
      abs_dot_a = f16[16,16] abs(dot_a_bias)
      c0 = f16[] constant(-inf)
      amax = f16[] reduce(abs_dot_a, c0), dimensions={0,1}, to_apply=apply
      dot_a_scaled = f16[16,16] divide(dot_a_bias, z_scale_bcast)
      c1 = f16[] constant(-<<F8E4M3_AMAX>>)
      c1_bcast = f16[16,16] broadcast(c1), dimensions={}
      c2 = f16[] constant(<<F8E4M3_AMAX>>)
      c2_bcast = f16[16,16] broadcast(c2), dimensions={}
      dot_a_clamped = f16[16,16] clamp(c1_bcast, dot_a_scaled, c2_bcast)
      dot_a_f8 = <<F8E4M3>>[16,16] convert(dot_a_clamped)
      ROOT result = (<<F8E4M3>>[16,16], f16[]) tuple(dot_a_f8, amax)
    }
)";

  CheckFp8IfSupported(hlo_text, ErrorSpec{0.1, 0.1});
  RunAndFilecheckHloRewrite(
      hlo_text,
      GemmRewriter(CudaHopperOrRocmMI300(), GetToolkitVersion(),
                   GemmRewriterOptions{GemmRewriterOptions::DType::kFp8Only}),
      R"(

; CHECK-LABEL: ENTRY %test ({{.*}}: <<F8E4M3>>[16,32], {{.*}}: <<F8E4M3>>[32,16], {{.*}}: f16[16,16], {{.*}}: f16[], {{.*}}: f16[], {{.*}}: f16[]) -> (<<F8E4M3>>[16,16], f16[]) {
; CHECK:    [[P0:%[^ ]+]] = <<F8E4M3>>[16,32]{1,0} parameter(0)
; CHECK-NEXT:    [[P1:%[^ ]+]] = <<F8E4M3>>[32,16]{1,0} parameter(1)
; CHECK-NEXT:    [[P1_TRANSPOSE:%[^ ]+]] = <<F8E4M3>>[16,32]{1,0} transpose([[P1]]), dimensions={1,0}
; CHECK:         [[C0:%[^ ]+]] = f16[16,16]{1,0} add({{.*}})
; CHECK-NEXT:    [[P2:%[^ ]+]] = f16[] parameter(3)
; CHECK:         [[P3:%[^ ]+]] = f16[] parameter(4)
; CHECK-PTX:         [[P4:%[^ ]+]] = f16[] parameter(5)
; CHECK-PTX:       [[OUT:%[^ ]+]] = (<<F8E4M3>>[16,16]{1,0}, f32[], s8[{{[0-9]+}}]{0}) custom-call([[P0]], [[P1_TRANSPOSE]], [[C0]], [[DUMMY0:%[^ ]+]], [[DUMMY1:%[^ ]+]], /*index=5*/[[DUMMY2:%[^ ]+]]),
; CHECK-NOT:       output_to_operand_aliasing
; CHECK-GCN:       [[OUT:%[^ ]+]] = (f16[16,16]{1,0}, s8[{{[0-9]+}}]{0}) custom-call([[P0]], [[P1_TRANSPOSE]], [[C0]], [[DUMMY0:%[^ ]+]], [[DUMMY1:%[^ ]+]]),
; CHECK:           custom_call_target="__cublas$lt$matmul$f8",
; CHECK:           backend_config={
; CHECK-DAG:         "alpha_real":1
; CHECK-DAG:         "alpha_imag":0
; CHECK-DAG:         "beta":1
; CHECK-DAG:         "dot_dimension_numbers":{
; CHECK-DAG:           "lhs_contracting_dimensions":["1"]
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

TEST_P(ParameterizedFp8GemmRewriteTest, ScaledABScaledDVectorBiasF8) {
  const char* hlo_text = R"(
    HloModule test

    ENTRY test {
      x = <<F8E4M3>>[16,32] parameter(0)
      y = <<F8E4M3>>[32,16] parameter(1)
      x_f16 = f16[16,32] convert(x)
      y_f16 = f16[32,16] convert(y)
      b = f16[16] parameter(2)
      b_bcast = f16[16,16] broadcast(b), dimensions={1}
      x_scale = f16[] parameter(3)
      y_scale = f16[] parameter(4)
      z_scale = f16[] parameter(5)
      x_scale_bcast = f16[16,32] broadcast(x_scale), dimensions={}
      y_scale_bcast = f16[32,16] broadcast(y_scale), dimensions={}
      z_scale_bcast = f16[16,16] broadcast(z_scale), dimensions={}
      x_unscaled = f16[16,32] multiply(x_f16, x_scale_bcast)
      y_unscaled = f16[32,16] multiply(y_f16, y_scale_bcast)
      dot_a = f16[16,16] dot(x_unscaled, y_unscaled), lhs_contracting_dims={1}, rhs_contracting_dims={0}
      dot_a_bias = f16[16,16] add(dot_a, b_bcast)
      dot_a_scaled = f16[16,16] divide(dot_a_bias, z_scale_bcast)
      c1 = f16[] constant(-<<F8E4M3_AMAX>>)
      c1_bcast = f16[16,16] broadcast(c1), dimensions={}
      c2 = f16[] constant(<<F8E4M3_AMAX>>)
      c2_bcast = f16[16,16] broadcast(c2), dimensions={}
      dot_a_clamped = f16[16,16] clamp(c1_bcast, dot_a_scaled, c2_bcast)
      ROOT dot_a_f8 = <<F8E4M3>>[16,16] convert(dot_a_clamped)
          }

)";

  CheckFp8IfSupported(hlo_text, ErrorSpec{0.1, 0.1});
  RunAndFilecheckHloRewrite(
      hlo_text,
      GemmRewriter(CudaHopperOrRocmMI300(), GetToolkitVersion(),
                   GemmRewriterOptions{GemmRewriterOptions::DType::kFp8Only}),
      R"(

; CHECK-LABEL: ENTRY %test ({{.*}}: <<F8E4M3>>[16,32], {{.*}}: <<F8E4M3>>[32,16], {{.*}}: f16[16], {{.*}}: f16[], {{.*}}: f16[], {{.*}}: f16[]) -> <<F8E4M3>>[16,16] {
; CHECK:    [[P0:%[^ ]+]] = <<F8E4M3>>[16,32]{1,0} parameter(0)
; CHECK-NEXT:    [[P1:%[^ ]+]] = <<F8E4M3>>[32,16]{1,0} parameter(1)
; CHECK-NEXT:    [[P1_TRANSPOSE:%[^ ]+]] = <<F8E4M3>>[16,32]{1,0} transpose([[P1]]), dimensions={1,0}
; CHECK-NEXT:    [[P2:%[^ ]+]] = f16[] parameter(3)
; CHECK-NEXT:    [[CV:%[^ ]+]] = f32[] convert([[P2]])
; CHECK-NEXT:    [[P3:%[^ ]+]] = f16[] parameter(4)
; CHECK-NEXT:    [[CV1:%[^ ]+]] = f32[] convert([[P3]])
; CHECK-NEXT:    [[VB:%[^ ]+]] = f16[16]{0} parameter(2)
; CHECK-PTX-NEXT:    [[C2:%[^ ]+]] = f16[] constant(1)
; CHECK-PTX-NEXT:    [[P4:%[^ ]+]] = f16[] parameter(5)
; CHECK-PTX-NEXT:    [[DV:%[^ ]+]] = f16[] divide([[C2]], [[P4]])
; CHECK-PTX-NEXT:    [[CV2:%[^ ]+]] = f32[] convert([[DV]])
; CHECK-PTX:     [[OUT:%[^ ]+]] = (<<F8E4M3>>[16,16]{1,0}, s8[{{[0-9]+}}]{0}) custom-call([[P0]], [[P1_TRANSPOSE]], [[CV]], [[CV1]], [[VB]], /*index=5*/[[CV2]]),
; CHECK-GCN:     [[OUT:%[^ ]+]] = (f16[16,16]{1,0}, s8[{{[0-9]+}}]{0}) custom-call([[P0]], [[P1_TRANSPOSE]], [[CV]], [[CV1]], [[VB]]),
; CHECK:           custom_call_target="__cublas$lt$matmul$f8",
; CHECK:           backend_config={
; CHECK-DAG:         "alpha_real":1
; CHECK-DAG:         "alpha_imag":0
; CHECK-DAG:         "beta":0
; CHECK-DAG:         "dot_dimension_numbers":{
; CHECK-DAG:           "lhs_contracting_dimensions":["1"]
; CHECK-DAG:           "rhs_contracting_dimensions":["1"]
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

TEST_P(ParameterizedFp8GemmRewriteTest, ScaledABUnscaledDF32VectorBiasF8) {
  const char* hlo_text = R"(
    HloModule test

    ENTRY test {
      x = <<F8E4M3>>[16,32] parameter(0)
      y = <<F8E4M3>>[32,16] parameter(1)
      x_f32 = f32[16,32] convert(x)
      y_f32 = f32[32,16] convert(y)
      b = f32[16] parameter(2)
      b_bf16 = bf16[16] convert(b)
      b_f32 = f32[16] convert(b_bf16)
      b_bcast = f32[16,16] broadcast(b_f32), dimensions={1}
      x_scale = f32[] parameter(3)
      y_scale = f32[] parameter(4)
      x_scale_bcast = f32[16,32] broadcast(x_scale), dimensions={}
      y_scale_bcast = f32[32,16] broadcast(y_scale), dimensions={}
      x_unscaled = f32[16,32] multiply(x_f32, x_scale_bcast)
      y_unscaled = f32[32,16] multiply(y_f32, y_scale_bcast)
      dot_a = f32[16,16] dot(x_unscaled, y_unscaled), lhs_contracting_dims={1}, rhs_contracting_dims={0}
      ROOT out = f32[16,16] add(dot_a, b_bcast)
           }

)";

  CheckFp8IfSupported(hlo_text);
  RunAndFilecheckHloRewrite(
      hlo_text,
      GemmRewriter(CudaHopperOrRocmMI300(), GetToolkitVersion(),
                   GemmRewriterOptions{GemmRewriterOptions::DType::kFp8Only}),
      R"(
; CHECK-LABEL: ENTRY %test ({{.*}}: <<F8E4M3>>[16,32], {{.*}}: <<F8E4M3>>[32,16], {{.*}}: f32[16], {{.*}}: f32[], {{.*}}: f32[]) -> f32[16,16] {
; CHECK:         [[P0:%[^ ]+]] = <<F8E4M3>>[16,32]{1,0} parameter(0)
; CHECK-NEXT:    [[P1:%[^ ]+]] = <<F8E4M3>>[32,16]{1,0} parameter(1)
; CHECK-NEXT:    [[P1_TRANSPOSE:%[^ ]+]] = <<F8E4M3>>[16,32]{1,0} transpose([[P1]]), dimensions={1,0}
; CHECK-NEXT:    [[P2:%[^ ]+]] = f32[] parameter(3)
; CHECK-NEXT:    [[P3:%[^ ]+]] = f32[] parameter(4)
; CHECK-NEXT:    [[VB:%[^ ]+]] = f32[16]{0} parameter(2)
; CHECK-NEXT:    [[VBC:%[^ ]+]] = bf16[16]{0} convert([[VB]])
; CHECK:         [[OUT:%[^ ]+]] = (f32[16,16]{1,0}, s8[{{[0-9]+}}]{0}) custom-call([[P0]], [[P1_TRANSPOSE]], [[P2]], [[P3]], [[VBC]]),
; CHECK:           custom_call_target="__cublas$lt$matmul$f8",
; CHECK:           backend_config={
; CHECK-DAG:         "alpha_real":1
; CHECK-DAG:         "alpha_imag":0
; CHECK-DAG:         "beta":0
; CHECK-DAG:         "dot_dimension_numbers":{
; CHECK-DAG:           "lhs_contracting_dimensions":["1"]
; CHECK-DAG:           "rhs_contracting_dimensions":["1"]
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

TEST_P(ParameterizedFp8GemmRewriteTest,
       ScaledABUnscaledDVectorBiasThenReluActivationF8) {
  const char* hlo_text = R"(
    HloModule test

    ENTRY test {
      x = <<F8E4M3>>[16,32] parameter(0)
      y = <<F8E4M3>>[32,16] parameter(1)
      b = f16[16] parameter(2)
      b_bcast = f16[16,16] broadcast(b), dimensions={1}
      x_f32 = f16[16,32] convert(x)
      y_f32 = f16[32,16] convert(y)
      x_scale = f16[] parameter(3)
      y_scale = f16[] parameter(4)
      x_scale_bcast = f16[16,32] broadcast(x_scale), dimensions={}
      y_scale_bcast = f16[32,16] broadcast(y_scale), dimensions={}
      x_unscaled = f16[16,32] multiply(x_f32, x_scale_bcast)
      y_unscaled = f16[32,16] multiply(y_f32, y_scale_bcast)
      c = f16[] constant(0)
      c_bcast = f16[16,16] broadcast(c), dimensions={}
      dot_a0 = f16[16,16] dot(x_unscaled, y_unscaled), lhs_contracting_dims={1}, rhs_contracting_dims={0}
      dot_a = f16[16,16] add(dot_a0, b_bcast)
      ROOT out = f16[16,16] maximum(dot_a, c_bcast)
          }
)";

  CheckFp8IfSupported(hlo_text, ErrorSpec{2e-3, 0.});
  RunAndFilecheckHloRewrite(
      hlo_text,
      GemmRewriter(CudaHopperOrRocmMI300(), GetToolkitVersion(),
                   GemmRewriterOptions{GemmRewriterOptions::DType::kFp8Only}),
      R"(
; CHECK-LABEL: ENTRY %test ({{.*}}: <<F8E4M3>>[16,32], {{.*}}: <<F8E4M3>>[32,16], {{.*}}: f16[16], {{.*}}: f16[], {{.*}}: f16[]) -> f16[16,16] {
; CHECK-NEXT:    [[P0:%[^ ]+]] = <<F8E4M3>>[16,32]{1,0} parameter(0)
; CHECK-NEXT:    [[P1:%[^ ]+]] = <<F8E4M3>>[32,16]{1,0} parameter(1)
; CHECK-NEXT:    [[P1_TRANSPOSE:%[^ ]+]] = <<F8E4M3>>[16,32]{1,0} transpose([[P1]]), dimensions={1,0}
; CHECK-NEXT:    [[P2:%[^ ]+]] = f16[] parameter(3)
; CHECK-NEXT:    [[CV:%[^ ]+]] = f32[] convert([[P2]])
; CHECK-NEXT:    [[P3:%[^ ]+]] = f16[] parameter(4)
; CHECK-NEXT:    [[CV1:%[^ ]+]] = f32[] convert([[P3]])
; CHECK-NEXT:    [[VB:%[^ ]+]] = f16[16]{0} parameter(2)
; CHECK     :    ROOT [[OUT:%[^ ]+]] = f16[16,16]{1,0} custom-call([[P0]], [[P1_TRANSPOSE]], [[CV]], [[CV1]], [[VB]]),
; CHECK:           custom_call_target="__cublas$lt$matmul$f8",
; CHECK:           backend_config={
; CHECK-DAG:         "alpha_real":1
; CHECK-DAG:         "alpha_imag":0
; CHECK-DAG:         "beta":0
; CHECK-DAG:         "dot_dimension_numbers":{
; CHECK-DAG:           "lhs_contracting_dimensions":["1"]
; CHECK-DAG:           "rhs_contracting_dimensions":["1"]
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

TEST_P(ParameterizedFp8GemmRewriteTest, Rank3ScaledABUnscaledDVectorBiasF8) {
  const char* hlo_text = R"(
    HloModule test
    ENTRY test {
      x = <<F8E4M3>>[4,16,16] parameter(0)
      y = <<F8E4M3>>[16,32] parameter(1)
      b = f32[32] parameter(2)
      b_f16 = f16[32] convert(b)
      b_bcast = f16[4,16,32] broadcast(b_f16), dimensions={2}
      x_f16 = f16[4,16,16] convert(x)
      y_f16 = f16[16,32] convert(y)
      x_scale = f16[] parameter(3)
      y_scale = f16[] parameter(4)
      x_scale_bcast = f16[4,16,16] broadcast(x_scale), dimensions={}
      y_scale_bcast = f16[16,32] broadcast(y_scale), dimensions={}
      x_unscaled = f16[4,16,16] multiply(x_f16, x_scale_bcast)
      x_unscaled_bitcast = f16[64,16] bitcast(x_unscaled)
      y_unscaled = f16[16,32] multiply(y_f16, y_scale_bcast)
      dot_a = f16[64,32] dot(x_unscaled_bitcast, y_unscaled), lhs_contracting_dims={1}, rhs_contracting_dims={0}
      dot_a_bitcast = f16[4,16,32]{2,1,0} bitcast(dot_a)
      ROOT out = f16[4,16,32] add(dot_a_bitcast, b_bcast)
          }
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_text));
  GemmRewriter pass(CudaHopperOrRocmMI300(), GetToolkitVersion(),
                    GemmRewriterOptions{GemmRewriterOptions::DType::kFp8Only});
  TF_ASSERT_OK_AND_ASSIGN(bool changed, this->RunHloPass(&pass, module.get()));
  EXPECT_TRUE(changed);

  EXPECT_THAT(
      module->entry_computation()->root_instruction(),
      GmockMatch(m::Bitcast(m::GetTupleElement(
                                m::CustomCall({"__cublas$lt$matmul$f8"}), 0)
                                .WithShape(F16, {64, 32}))
                     .WithShape(F16, {4, 16, 32})));

  RunAndFilecheckHloRewrite(
      hlo_text,
      GemmRewriter(CudaHopperOrRocmMI300(), GetToolkitVersion(),
                   GemmRewriterOptions{GemmRewriterOptions::DType::kFp8Only}),
      R"(
; CHECK-LABEL: ENTRY %test ({{.*}}: <<F8E4M3>>[4,16,16], {{.*}}: <<F8E4M3>>[16,32], {{.*}}: f32[32], {{.*}}: f16[], {{.*}}: f16[]) -> f16[4,16,32] {
; CHECK-NEXT:    [[P0:%[^ ]+]] = <<F8E4M3>>[4,16,16]{2,1,0} parameter(0)
; CHECK-NEXT:    [[P0_BITCAST:%[^ ]+]] = <<F8E4M3>>[64,16]{1,0} bitcast([[P0]])
; CHECK-NEXT:    [[P1:%[^ ]+]] = <<F8E4M3>>[16,32]{1,0} parameter(1)
; CHECK-NEXT:    [[P1_TRANSPOSE:%[^ ]+]] = <<F8E4M3>>[32,16]{1,0} transpose([[P1]]), dimensions={1,0}
; CHECK-NEXT:    [[P2:%[^ ]+]] = f16[] parameter(3)
; CHECK-NEXT:    [[P2_CV:%[^ ]+]] = f32[] convert([[P2]])
; CHECK-NEXT:    [[P3:%[^ ]+]] = f16[] parameter(4)
; CHECK-NEXT:    [[P3_CV:%[^ ]+]] = f32[] convert([[P3]])
; CHECK-NEXT:    [[B:%[^ ]+]] = f32[32]{0} parameter(2)
; CHECK-NEXT:    [[B_F16:%[^ ]+]] = f16[32]{0} convert([[B]])
; CHECK-NEXT:    [[GEMM_TUPLE:%[^ ]+]] = (f16[64,32]{1,0}, s8[{{[0-9]+}}]{0}) custom-call([[P0_BITCAST]], [[P1_TRANSPOSE]], [[P2_CV]], [[P3_CV]], [[B_F16]]),
; CHECK:           custom_call_target="__cublas$lt$matmul$f8",
; CHECK:           backend_config={
; CHECK-DAG:         "alpha_real":1
; CHECK-DAG:         "alpha_imag":0
; CHECK-DAG:         "beta":0
; CHECK-DAG:         "dot_dimension_numbers":{
; CHECK-DAG:           "lhs_contracting_dimensions":["1"]
; CHECK-DAG:           "rhs_contracting_dimensions":["1"]
; CHECK-DAG:           "lhs_batch_dimensions":[]
; CHECK-DAG:           "rhs_batch_dimensions":[]
; CHECK-DAG:         }
; CHECK-DAG:         "precision_config":{
; CHECK-DAG:           "operand_precision":["DEFAULT","DEFAULT"]
; CHECK-DAG:         }
; CHECK-DAG:         "epilogue":"BIAS"
; CHECK:           }
; CHECK:         [[GEMM:%[^ ]+]] = f16[64,32]{1,0} get-tuple-element([[GEMM_TUPLE]]), index=0
; CHECK:         ROOT [[OUT:%[^ ]+]] = f16[4,16,32]{2,1,0} bitcast([[GEMM]])
      )");
}

TEST_P(ParameterizedFp8GemmRewriteTest,
       Rank3ScaledABUnscaledDVectorBiasPaddedF8) {
  const char* hlo_text = R"(
    HloModule test
    ENTRY test {
      x = <<F8E4M3>>[4,15,15] parameter(0)
      y = <<F8E4M3>>[15,31] parameter(1)
      b = f32[31] parameter(2)
      b_f16 = f16[31] convert(b)
      b_bcast = f16[4,15,31] broadcast(b_f16), dimensions={2}
      x_f16 = f16[4,15,15] convert(x)
      y_f16 = f16[15,31] convert(y)
      x_scale = f16[] parameter(3)
      y_scale = f16[] parameter(4)
      x_scale_bcast = f16[4,15,15] broadcast(x_scale), dimensions={}
      y_scale_bcast = f16[15,31] broadcast(y_scale), dimensions={}
      x_unscaled = f16[4,15,15] multiply(x_f16, x_scale_bcast)
      x_unscaled_bitcast = f16[60,15] bitcast(x_unscaled)
      y_unscaled = f16[15,31] multiply(y_f16, y_scale_bcast)
      dot_a = f16[60,31] dot(x_unscaled_bitcast, y_unscaled), lhs_contracting_dims={1}, rhs_contracting_dims={0}
      dot_a_bitcast = f16[4,15,31]{2,1,0} bitcast(dot_a)
      ROOT out = f16[4,15,31] add(dot_a_bitcast, b_bcast)
          }
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_text));
  GemmRewriter pass(CudaHopperOrRocmMI300(), GetToolkitVersion(),
                    GemmRewriterOptions{GemmRewriterOptions::DType::kFp8Only});
  TF_ASSERT_OK_AND_ASSIGN(bool changed, this->RunHloPass(&pass, module.get()));
  EXPECT_TRUE(changed);

  EXPECT_THAT(
      module->entry_computation()->root_instruction(),
      GmockMatch(
          m::Bitcast(m::Slice(m::GetTupleElement(
                                  m::CustomCall({"__cublas$lt$matmul$f8"}), 0)
                                  .WithShape(F16, {64, 32}))
                         .WithShape(F16, {60, 31}))
              .WithShape(F16, {4, 15, 31})));

  RunAndFilecheckHloRewrite(
      hlo_text,
      GemmRewriter(CudaHopperOrRocmMI300(), GetToolkitVersion(),
                   GemmRewriterOptions{GemmRewriterOptions::DType::kFp8Only}),
      R"(
; CHECK-LABEL: ENTRY %test ({{.*}}: <<F8E4M3>>[4,15,15], {{.*}}: <<F8E4M3>>[15,31], {{.*}}: f32[31], {{.*}}: f16[], {{.*}}: f16[]) -> f16[4,15,31] {
; CHECK-NEXT:    [[P0:%[^ ]+]] = <<F8E4M3>>[4,15,15]{2,1,0} parameter(0)
; CHECK-NEXT:    [[P0_BITCAST:%[^ ]+]] = <<F8E4M3>>[60,15]{1,0} bitcast([[P0]])
; CHECK-NEXT:    [[C1:%[^ ]+]] = <<F8E4M3>>[] constant(0)
; CHECK-NEXT:    [[P0_PAD:%[^ ]+]] = <<F8E4M3>>[64,16]{1,0} pad([[P0_BITCAST]], [[C1]]), padding=0_4x0_1
; CHECK-NEXT:    [[P1:%[^ ]+]] = <<F8E4M3>>[15,31]{1,0} parameter(1)
; CHECK-NEXT:    [[P1_TRANSPOSE:%[^ ]+]] = <<F8E4M3>>[31,15]{1,0} transpose([[P1]]), dimensions={1,0}
; CHECK-NEXT:    [[C2:%[^ ]+]] = <<F8E4M3>>[] constant(0)
; CHECK-NEXT:    [[P1_PAD:%[^ ]+]] = <<F8E4M3>>[32,16]{1,0} pad([[P1_TRANSPOSE]], [[C2]]), padding=0_1x0_1
; CHECK-NEXT:    [[P2:%[^ ]+]] = f16[] parameter(3)
; CHECK-NEXT:    [[P2_CV:%[^ ]+]] = f32[] convert([[P2]])
; CHECK-NEXT:    [[P3:%[^ ]+]] = f16[] parameter(4)
; CHECK-NEXT:    [[P3_CV:%[^ ]+]] = f32[] convert([[P3]])
; CHECK-NEXT:    [[B:%[^ ]+]] = f32[31]{0} parameter(2)
; CHECK-NEXT:    [[B_F16:%[^ ]+]] = f16[31]{0} convert([[B]])
; CHECK-NEXT:    [[C3:%[^ ]+]] = f16[] constant(0)
; CHECK-NEXT:    [[P2_PAD:%[^ ]+]] = f16[32]{0} pad([[B_F16]], [[C3]]), padding=0_1
; CHECK-NEXT:    [[GEMM_TUPLE:%[^ ]+]] = (f16[64,32]{1,0}, s8[{{[0-9]+}}]{0}) custom-call([[P0_PAD]], [[P1_PAD]], [[P2_CV]], [[P3_CV]], [[P2_PAD]]),
; CHECK:           custom_call_target="__cublas$lt$matmul$f8",
; CHECK:           backend_config={
; CHECK-DAG:         "alpha_real":1
; CHECK-DAG:         "alpha_imag":0
; CHECK-DAG:         "beta":0
; CHECK-DAG:         "dot_dimension_numbers":{
; CHECK-DAG:           "lhs_contracting_dimensions":["1"]
; CHECK-DAG:           "rhs_contracting_dimensions":["1"]
; CHECK-DAG:           "lhs_batch_dimensions":[]
; CHECK-DAG:           "rhs_batch_dimensions":[]
; CHECK-DAG:         }
; CHECK-DAG:         "precision_config":{
; CHECK-DAG:           "operand_precision":["DEFAULT","DEFAULT"]
; CHECK-DAG:         }
; CHECK-DAG:         "epilogue":"BIAS"
; CHECK:           }
; CHECK:          [[GEMM:%[^ ]+]] = f16[64,32]{1,0} get-tuple-element([[GEMM_TUPLE]]), index=0
; CHECK-NEXT:     [[SLICE:%[^ ]+]] = f16[60,31]{1,0} slice([[GEMM]]), slice={[0:60], [0:31]}
; CHECK-NEXT:     ROOT [[OUT:%[^ ]+]] = f16[4,15,31]{2,1,0} bitcast([[SLICE]])
      )");
}

TEST_P(ParameterizedFp8GemmRewriteTest, Rank3ScaledABUnscaledDMatrixBiasF8) {
  const char* hlo_text = R"(
    HloModule test
    ENTRY test {
      x = <<F8E4M3>>[4,16,16] parameter(0)
      y = <<F8E4M3>>[16,32] parameter(1)
      b = f32[4,16,32] parameter(2)
      x_f32 = f32[4,16,16] convert(x)
      y_f32 = f32[16,32] convert(y)
      x_scale = f32[] parameter(3)
      y_scale = f32[] parameter(4)
      x_scale_bcast = f32[4,16,16] broadcast(x_scale), dimensions={}
      y_scale_bcast = f32[16,32] broadcast(y_scale), dimensions={}
      x_unscaled = f32[4,16,16] multiply(x_f32, x_scale_bcast)
      x_unscaled_bitcast = f32[64,16] bitcast(x_unscaled)
      y_unscaled = f32[16,32] multiply(y_f32, y_scale_bcast)
      dot_a = f32[64,32] dot(x_unscaled_bitcast, y_unscaled), lhs_contracting_dims={1}, rhs_contracting_dims={0}
      dot_a_bitcast = f32[4,16,32]{2,1,0} bitcast(dot_a)
      ROOT out = f32[4,16,32] add(dot_a_bitcast, b)
          }
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_text));
  GemmRewriter pass(CudaHopperOrRocmMI300(), GetToolkitVersion(),
                    GemmRewriterOptions{GemmRewriterOptions::DType::kFp8Only});
  TF_ASSERT_OK_AND_ASSIGN(bool changed, this->RunHloPass(&pass, module.get()));
  EXPECT_TRUE(changed);

  EXPECT_THAT(
      module->entry_computation()->root_instruction(),
      GmockMatch(m::Bitcast(m::GetTupleElement(
                                m::CustomCall({"__cublas$lt$matmul$f8"}), 0)
                                .WithShape(F32, {64, 32}))
                     .WithShape(F32, {4, 16, 32})));

  RunAndFilecheckHloRewrite(
      hlo_text,
      GemmRewriter(CudaHopperOrRocmMI300(), GetToolkitVersion(),
                   GemmRewriterOptions{GemmRewriterOptions::DType::kFp8Only}),
      R"(
; CHECK-LABEL: ENTRY %test ({{.*}}: <<F8E4M3>>[4,16,16], {{.*}}: <<F8E4M3>>[16,32], {{.*}}: f32[4,16,32], {{.*}}: f32[], {{.*}}: f32[]) -> f32[4,16,32] {
; CHECK-NEXT:    [[P0:%[^ ]+]] = <<F8E4M3>>[4,16,16]{2,1,0} parameter(0)
; CHECK-NEXT:    [[P0_BITCAST:%[^ ]+]] = <<F8E4M3>>[64,16]{1,0} bitcast([[P0]])
; CHECK-NEXT:    [[P1:%[^ ]+]] = <<F8E4M3>>[16,32]{1,0} parameter(1)
; CHECK-NEXT:    [[P1_TRANSPOSE:%[^ ]+]] = <<F8E4M3>>[32,16]{1,0} transpose([[P1]]), dimensions={1,0}
; CHECK-NEXT:    [[B:%[^ ]+]] = f32[4,16,32]{2,1,0} parameter(2)
; CHECK-NEXT:    [[B_BITCAST:%[^ ]+]] = f32[64,32]{1,0} bitcast([[B]])
; CHECK-NEXT:    [[P2:%[^ ]+]] = f32[] parameter(3)
; CHECK-NEXT:    [[P3:%[^ ]+]] = f32[] parameter(4)
; CHECK-NEXT:    [[GEMM_TUPLE:%[^ ]+]] = (f32[64,32]{1,0}, s8[{{[0-9]+}}]{0}) custom-call([[P0_BITCAST]], [[P1_TRANSPOSE]], [[B_BITCAST]], [[P2]], [[P3]]),
; CHECK:           custom_call_target="__cublas$lt$matmul$f8",
; CHECK:           backend_config={
; CHECK-DAG:         "alpha_real":1
; CHECK-DAG:         "alpha_imag":0
; CHECK-DAG:         "beta":1
; CHECK-DAG:         "dot_dimension_numbers":{
; CHECK-DAG:           "lhs_contracting_dimensions":["1"]
; CHECK-DAG:           "rhs_contracting_dimensions":["1"]
; CHECK-DAG:           "lhs_batch_dimensions":[]
; CHECK-DAG:           "rhs_batch_dimensions":[]
; CHECK-DAG:         }
; CHECK-DAG:         "precision_config":{
; CHECK-DAG:           "operand_precision":["DEFAULT","DEFAULT"]
; CHECK-DAG:         }
; CHECK-DAG:         "epilogue":"DEFAULT"
; CHECK:           }
; CHECK:         [[GEMM:%[^ ]+]] = f32[64,32]{1,0} get-tuple-element([[GEMM_TUPLE]]), index=0
; CHECK:         ROOT [[OUT:%[^ ]+]] = f32[4,16,32]{2,1,0} bitcast([[GEMM]])
      )");
}

TEST_P(ParameterizedFp8GemmRewriteTest,
       Rank3ScaledABUnscaledDMatrixBiasPaddedF8) {
  const char* hlo_text = R"(
    HloModule test
    ENTRY test {
      x = <<F8E4M3>>[3,15,15] parameter(0)
      y = <<F8E4M3>>[15,31] parameter(1)
      b = f32[3,15,31] parameter(2)
      x_f32 = f32[3,15,15] convert(x)
      y_f32 = f32[15,31] convert(y)
      x_scale = f32[] parameter(3)
      y_scale = f32[] parameter(4)
      x_scale_bcast = f32[3,15,15] broadcast(x_scale), dimensions={}
      y_scale_bcast = f32[15,31] broadcast(y_scale), dimensions={}
      x_unscaled = f32[3,15,15] multiply(x_f32, x_scale_bcast)
      x_unscaled_bitcast = f32[45,15] bitcast(x_unscaled)
      y_unscaled = f32[15,31] multiply(y_f32, y_scale_bcast)
      dot_a = f32[45,31] dot(x_unscaled_bitcast, y_unscaled), lhs_contracting_dims={1}, rhs_contracting_dims={0}
      dot_a_bitcast = f32[3,15,31]{2,1,0} bitcast(dot_a)
      ROOT out = f32[3,15,31] add(dot_a_bitcast, b)
          }
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_text));
  GemmRewriter pass(CudaHopperOrRocmMI300(), GetToolkitVersion(),
                    GemmRewriterOptions{GemmRewriterOptions::DType::kFp8Only});
  TF_ASSERT_OK_AND_ASSIGN(bool changed, this->RunHloPass(&pass, module.get()));
  EXPECT_TRUE(changed);

  EXPECT_THAT(
      module->entry_computation()->root_instruction(),
      GmockMatch(
          m::Bitcast(m::Slice(m::GetTupleElement(
                                  m::CustomCall({"__cublas$lt$matmul$f8"}), 0)
                                  .WithShape(F32, {48, 32}))
                         .WithShape(F32, {45, 31}))
              .WithShape(F32, {3, 15, 31})));

  RunAndFilecheckHloRewrite(
      hlo_text,
      GemmRewriter(CudaHopperOrRocmMI300(), GetToolkitVersion(),
                   GemmRewriterOptions{GemmRewriterOptions::DType::kFp8Only}),
      R"(
; CHECK-LABEL: ENTRY %test ({{.*}}: <<F8E4M3>>[3,15,15], {{.*}}: <<F8E4M3>>[15,31], {{.*}}: f32[3,15,31], {{.*}}: f32[], {{.*}}: f32[]) -> f32[3,15,31] {
; CHECK-NEXT:    [[P0:%[^ ]+]] = <<F8E4M3>>[3,15,15]{2,1,0} parameter(0)
; CHECK-NEXT:    [[P0_BITCAST:%[^ ]+]] = <<F8E4M3>>[45,15]{1,0} bitcast([[P0]])
; CHECK-NEXT:    [[C1:%[^ ]+]] = <<F8E4M3>>[] constant(0)
; CHECK-NEXT:    [[P0_PADDED:%[^ ]+]] = <<F8E4M3>>[48,16]{1,0} pad([[P0_BITCAST]], [[C1]]), padding=0_3x0_1
; CHECK-NEXT:    [[P1:%[^ ]+]] = <<F8E4M3>>[15,31]{1,0} parameter(1)
; CHECK-NEXT:    [[P1_TRANSPOSE:%[^ ]+]] = <<F8E4M3>>[31,15]{1,0} transpose([[P1]]), dimensions={1,0}
; CHECK-NEXT:    [[C2:%[^ ]+]] = <<F8E4M3>>[] constant(0)
; CHECK-NEXT:    [[P1_PADDED:%[^ ]+]] = <<F8E4M3>>[32,16]{1,0} pad([[P1_TRANSPOSE]], [[C2]]), padding=0_1x0_1
; CHECK-NEXT:    [[B:%[^ ]+]] = f32[3,15,31]{2,1,0} parameter(2)
; CHECK-NEXT:    [[B_BITCAST:%[^ ]+]] = f32[45,31]{1,0} bitcast([[B]])
; CHECK-NEXT:    [[C3:%[^ ]+]] = f32[] constant(0)
; CHECK-NEXT:    [[P2_PADDED:%[^ ]+]] = f32[48,32]{1,0} pad([[B_BITCAST]], [[C3]]), padding=0_3x0_1
; CHECK-NEXT:    [[P2:%[^ ]+]] = f32[] parameter(3)
; CHECK-NEXT:    [[P3:%[^ ]+]] = f32[] parameter(4)
; CHECK-NEXT:    [[GEMM_TUPLE:%[^ ]+]] = (f32[48,32]{1,0}, s8[{{[0-9]+}}]{0}) custom-call([[P0_PADDED]], [[P1_PADDED]], [[P2_PADDED]], [[P2]], [[P3]]),
; CHECK:           custom_call_target="__cublas$lt$matmul$f8",
; CHECK:           backend_config={
; CHECK-DAG:         "alpha_real":1
; CHECK-DAG:         "alpha_imag":0
; CHECK-DAG:         "beta":1
; CHECK-DAG:         "dot_dimension_numbers":{
; CHECK-DAG:           "lhs_contracting_dimensions":["1"]
; CHECK-DAG:           "rhs_contracting_dimensions":["1"]
; CHECK-DAG:           "lhs_batch_dimensions":[]
; CHECK-DAG:           "rhs_batch_dimensions":[]
; CHECK-DAG:         }
; CHECK-DAG:         "precision_config":{
; CHECK-DAG:           "operand_precision":["DEFAULT","DEFAULT"]
; CHECK-DAG:         }
; CHECK-DAG:         "epilogue":"DEFAULT"
; CHECK:           }
; CHECK-NEXT:      [[GEMM:%[^ ]+]] = f32[48,32]{1,0} get-tuple-element([[GEMM_TUPLE]]), index=0
; CHECK-NEXT:      [[SLICE:%[^ ]+]] = f32[45,31]{1,0} slice([[GEMM]]), slice={[0:45], [0:31]}
; CHECK-NEXT:      ROOT [[OUT:%[^ ]+]] = f32[3,15,31]{2,1,0} bitcast([[SLICE]])
      )");
}

// Do not fuse matrix bias When there is a slice that does not chop off the ends
// of dimensions.
TEST_P(ParameterizedFp8GemmRewriteTest,
       ScaledABUnscaledDMatrixBiasWithSliceF8) {
  const char* hlo_text = R"(
    HloModule test
    ENTRY test {
      x = <<F8E4M3>>[48,16] parameter(0)
      y = <<F8E4M3>>[16,32] parameter(1)
      b = f32[32,16] parameter(2)
      x_f32 = f32[48,16] convert(x)
      y_f32 = f32[16,32] convert(y)
      x_scale = f32[] parameter(3)
      y_scale = f32[] parameter(4)
      x_scale_bcast = f32[48,16] broadcast(x_scale), dimensions={}
      y_scale_bcast = f32[16,32] broadcast(y_scale), dimensions={}
      x_unscaled = f32[48,16] multiply(x_f32, x_scale_bcast)
      y_unscaled = f32[16,32] multiply(y_f32, y_scale_bcast)
      dot_a = f32[48,32] dot(x_unscaled, y_unscaled), lhs_contracting_dims={1}, rhs_contracting_dims={0}
      dot_a_sliced = f32[32,16] slice(dot_a), slice={[16:48], [16:32]}
      ROOT out = f32[32,16] add(dot_a_sliced, b)
          }
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_text));
  GemmRewriter pass(CudaHopperOrRocmMI300(), GetToolkitVersion(),
                    GemmRewriterOptions{GemmRewriterOptions::DType::kFp8Only});
  TF_ASSERT_OK_AND_ASSIGN(bool changed, this->RunHloPass(&pass, module.get()));
  EXPECT_TRUE(changed);

  RunAndFilecheckHloRewrite(
      hlo_text,
      GemmRewriter(CudaHopperOrRocmMI300(), GetToolkitVersion(),
                   GemmRewriterOptions{GemmRewriterOptions::DType::kFp8Only}),
      R"(
; CHECK-LABEL: ENTRY %test ({{.*}}: <<F8E4M3>>[48,16], {{.*}}: <<F8E4M3>>[16,32], {{.*}}: f32[32,16], {{.*}}: f32[], {{.*}}: f32[]) -> f32[32,16] {
; CHECK-NEXT:    [[P0:%[^ ]+]] = <<F8E4M3>>[48,16]{1,0} parameter(0)
; CHECK-NEXT:    [[P1:%[^ ]+]] = <<F8E4M3>>[16,32]{1,0} parameter(1)
; CHECK-NEXT:    [[P1_TRANSPOSE:%[^ ]+]] = <<F8E4M3>>[32,16]{1,0} transpose([[P1]]), dimensions={1,0}
; CHECK-NEXT:    [[P2:%[^ ]+]] = f32[] parameter(3)
; CHECK-NEXT:    [[P3:%[^ ]+]] = f32[] parameter(4)
; CHECK-NEXT:    [[GEMM_TUPLE:%[^ ]+]] = (f32[48,32]{1,0}, s8[{{[0-9]+}}]{0}) custom-call([[P0]], [[P1_TRANSPOSE]], [[P2]], [[P3]]),
; CHECK:           custom_call_target="__cublas$lt$matmul$f8",
; CHECK:           backend_config={
; CHECK-DAG:         "alpha_real":1
; CHECK-DAG:         "alpha_imag":0
; CHECK-DAG:         "beta":0
; CHECK-DAG:         "dot_dimension_numbers":{
; CHECK-DAG:           "lhs_contracting_dimensions":["1"]
; CHECK-DAG:           "rhs_contracting_dimensions":["1"]
; CHECK-DAG:           "lhs_batch_dimensions":[]
; CHECK-DAG:           "rhs_batch_dimensions":[]
; CHECK-DAG:         }
; CHECK-DAG:         "precision_config":{
; CHECK-DAG:           "operand_precision":["DEFAULT","DEFAULT"]
; CHECK-DAG:         }
; CHECK-DAG:         "epilogue":"DEFAULT"
; CHECK:           }
; CHECK:           [[GEMM:%[^_]+]] = f32[48,32]{1,0} get-tuple-element([[GEMM_TUPLE]]), index=0
; CHECK-NEXT:      [[SLICE:%[^ ]+]] = f32[32,16]{1,0} slice([[GEMM]]), slice={[16:48], [16:32]}
; CHECK-NEXT:      [[B:%[^ ]+]] = f32[32,16]{1,0} parameter(2)
; CHECK-NEXT:      ROOT [[OUT:%[^ ]+]] = f32[32,16]{1,0} add([[SLICE]], [[B]])
      )");
}

TEST_P(ParameterizedFp8GemmRewriteTest,
       ScaledABUnscaledDMatrixBiasThenVectorBiasF8) {
  const char* hlo_text = R"(
    HloModule test

    ENTRY test {
      x = <<F8E4M3>>[16,32] parameter(0)
      y = <<F8E4M3>>[32,16] parameter(1)
      x_f16 = f16[16,32] convert(x)
      y_f16 = f16[32,16] convert(y)
      b = f16[16] parameter(2)
      b_bcast = f16[16,16] broadcast(b), dimensions={1}
      b2 = f16[16,16] parameter(3)
      x_scale = f16[] parameter(4)
      y_scale = f16[] parameter(5)
      x_scale_bcast = f16[16,32] broadcast(x_scale), dimensions={}
      y_scale_bcast = f16[32,16] broadcast(y_scale), dimensions={}
      x_unscaled = f16[16,32] multiply(x_f16, x_scale_bcast)
      y_unscaled = f16[32,16] multiply(y_f16, y_scale_bcast)
      dot_a = f16[16,16] dot(x_unscaled, y_unscaled), lhs_contracting_dims={1}, rhs_contracting_dims={0}
      dot_a_bias1 = f16[16,16] add(dot_a, b2)
      ROOT dot_a_bias = f16[16,16] add(dot_a_bias1, b_bcast)
          }

)";

  CheckFp8IfSupported(hlo_text, ErrorSpec{2e-3, 0.});
  RunAndFilecheckHloRewrite(
      hlo_text,
      GemmRewriter(CudaHopperOrRocmMI300(), GetToolkitVersion(),
                   GemmRewriterOptions{GemmRewriterOptions::DType::kFp8Only}),
      R"(
; CHECK-LABEL:   ENTRY %test ({{.*}}: <<F8E4M3>>[16,32], {{.*}}: <<F8E4M3>>[32,16], {{.*}}: f16[16], {{.*}}: f16[16,16], {{.*}}: f16[], {{.*}}: f16[]) -> f16[16,16] {
; CHECK-DAG:     [[P0:%[^ ]+]] = <<F8E4M3>>[16,32]{1,0} parameter(0)
; CHECK-NEXT:    [[P1:%[^ ]+]] = <<F8E4M3>>[32,16]{1,0} parameter(1)
; CHECK-NEXT:    [[P1_TRANSPOSE:%[^ ]+]] = <<F8E4M3>>[16,32]{1,0} transpose([[P1]]), dimensions={1,0}
; CHECK-NEXT:    [[MB:%[^ ]+]] = f16[16,16]{1,0} parameter(3)
; CHECK-NEXT:    [[P2:%[^ ]+]] = f16[] parameter(4)
; CHECK-NEXT:    [[CV0:%[^ ]+]] = f32[] convert([[P2]])
; CHECK-NEXT:    [[P3:%[^ ]+]] = f16[] parameter(5)
; CHECK-NEXT:    [[CV1:%[^ ]+]] = f32[] convert([[P3]])
; CHECK:         [[GEMMOUT_TUPLE:%[^ ]+]] = (f16[16,16]{1,0}, s8[{{[0-9]+}}]{0}) custom-call([[P0]], [[P1_TRANSPOSE]], [[MB]], [[CV0]], [[CV1]]),
; CHECK:           custom_call_target="__cublas$lt$matmul$f8",
; CHECK:           backend_config={
; CHECK-DAG:         "alpha_real":1
; CHECK-DAG:         "alpha_imag":0
; CHECK-DAG:         "beta":1
; CHECK-DAG:         "dot_dimension_numbers":{
; CHECK-DAG:           "lhs_contracting_dimensions":["1"]
; CHECK-DAG:           "rhs_contracting_dimensions":["1"]
; CHECK-DAG:           "lhs_batch_dimensions":[]
; CHECK-DAG:           "rhs_batch_dimensions":[]
; CHECK-DAG:         }
; CHECK-DAG:         "precision_config":{
; CHECK-DAG:           "operand_precision":["DEFAULT","DEFAULT"]
; CHECK-DAG:         }
; CHECK-DAG:         "epilogue":"DEFAULT"
; CHECK:           }
; CHECK:         [[GEMMOUT:%[^ ]+]] = f16[16,16]{1,0} get-tuple-element([[GEMMOUT_TUPLE]]), index=0
; CHECK:         [[VB:%[^ ]+]] = f16[16]{0} parameter(2)
; CHECK:         [[VBC:%[^ ]+]] = f16[16,16]{1,0} broadcast([[VB]]), dimensions={1}
; CHECK:         ROOT [[OUT:%[^ ]+]] = f16[16,16]{1,0} add([[GEMMOUT]], [[VBC]])
      )");
}

TEST_P(ParameterizedFp8GemmRewriteTest, ScaledABScaledDWithDAmaxF8) {
  const char* hlo_text = R"(
    HloModule test

    apply {
      a = f32[] parameter(0)
      b = f32[] parameter(1)
      ROOT c = f32[] maximum(a, b)
    }

    ENTRY test {
      x = <<F8E4M3>>[16,32] parameter(0)
      y = <<F8E4M3>>[32,16] parameter(1)
      x_f32 = f32[16,32] convert(x)
      y_f32 = f32[32,16] convert(y)
      x_scale = f32[] parameter(2)
      y_scale = f32[] parameter(3)
      z_scale = f32[] parameter(4)
      x_scale_bcast = f32[16,32] broadcast(x_scale), dimensions={}
      y_scale_bcast = f32[32,16] broadcast(y_scale), dimensions={}
      z_scale_bcast = f32[16,16] broadcast(z_scale), dimensions={}
      x_unscaled = f32[16,32] multiply(x_f32, x_scale_bcast)
      y_unscaled = f32[32,16] multiply(y_f32, y_scale_bcast)
      dot_a = f32[16,16] dot(x_unscaled, y_unscaled), lhs_contracting_dims={1}, rhs_contracting_dims={0}
      abs_dot_a = f32[16,16] abs(dot_a)
      c0 = f32[] constant(-inf)
      amax = f32[] reduce(abs_dot_a, c0), dimensions={0,1}, to_apply=apply
      dot_a_scaled = f32[16,16] divide(dot_a, z_scale_bcast)
      c1 = f32[] constant(-<<F8E4M3_AMAX>>)
      c1_bcast = f32[16,16] broadcast(c1), dimensions={}
      c2 = f32[] constant(<<F8E4M3_AMAX>>)
      c2_bcast = f32[16,16] broadcast(c2), dimensions={}
      dot_a_clamped = f32[16,16] clamp(c1_bcast, dot_a_scaled, c2_bcast)
      dot_a_f8 = <<F8E4M3>>[16,16] convert(dot_a_clamped)
      ROOT out = (<<F8E4M3>>[16,16], f32[]) tuple(dot_a_f8, amax)
          }

)";

  CheckFp8IfSupported(hlo_text);
  RunAndFilecheckHloRewrite(
      hlo_text,
      GemmRewriter(CudaHopperOrRocmMI300(), GetToolkitVersion(),
                   GemmRewriterOptions{GemmRewriterOptions::DType::kFp8Only}),
      R"(
; CHECK-LABEL: ENTRY %test ({{.*}}: <<F8E4M3>>[16,32], {{.*}}: <<F8E4M3>>[32,16], {{.*}}: f32[], {{.*}}: f32[], {{.*}}: f32[]) -> (<<F8E4M3>>[16,16], f32[]) {
; CHECK:    [[P0:%[^ ]+]] = <<F8E4M3>>[16,32]{1,0} parameter(0)
; CHECK-NEXT:    [[P1:%[^ ]+]] = <<F8E4M3>>[32,16]{1,0} parameter(1)
; CHECK-NEXT:    [[P1_TRANSPOSE:%[^ ]+]] = <<F8E4M3>>[16,32]{1,0} transpose([[P1]])
; CHECK-NEXT:    [[P2:%[^ ]+]] = f32[] parameter(2)
; CHECK-NEXT:    [[P3:%[^ ]+]] = f32[] parameter(3)
; CHECK-PTX-NEXT:    [[C2:%[^ ]+]] = f32[] constant(1)
; CHECK-PTX-NEXT:    [[P4:%[^ ]+]] = f32[] parameter(4)
; CHECK-PTX-NEXT:    [[P4_INV:%[^ ]+]] = f32[] divide([[C2]], [[P4]])
; CHECK-PTX-NEXT:    [[OUT:%[^ ]+]] = (<<F8E4M3>>[16,16]{1,0}, f32[], s8[{{[0-9]+}}]{0}) custom-call([[P0]], [[P1_TRANSPOSE]], [[P2]], [[P3]], [[P4_INV]]),
; CHECK-GCN-NEXT:    [[OUT:%[^ ]+]] = (f32[16,16]{1,0}, s8[{{[0-9]+}}]{0}) custom-call([[P0]], [[P1_TRANSPOSE]], [[P2]], [[P3]]),
; CHECK:           custom_call_target="__cublas$lt$matmul$f8",
; CHECK:           backend_config={
; CHECK-DAG:         "alpha_real":1
; CHECK-DAG:         "alpha_imag":0
; CHECK-DAG:         "beta":0
; CHECK-DAG:         "dot_dimension_numbers":{
; CHECK-DAG:           "lhs_contracting_dimensions":["1"]
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

TEST_P(ParameterizedFp8GemmRewriteTest,
       ScaledABScaledDWithDAmaxF8WithF16Intermediates) {
  // This is the same as ScaledABScaledDWithDAmaxF8, but uses F16 intermediate
  // values instead of F32 intermediate values.
  const char* hlo_text = R"(
    HloModule test

    apply {
      a = f16[] parameter(0)
      b = f16[] parameter(1)
      ROOT c = f16[] maximum(a, b)
    }

    ENTRY test {
      x = <<F8E4M3>>[16,32] parameter(0)
      y = <<F8E4M3>>[32,16] parameter(1)
      x_f16 = f16[16,32] convert(x)
      y_f16 = f16[32,16] convert(y)
      x_scale = f16[] parameter(2)
      y_scale = f16[] parameter(3)
      z_scale = f16[] parameter(4)
      x_scale_bcast = f16[16,32] broadcast(x_scale), dimensions={}
      y_scale_bcast = f16[32,16] broadcast(y_scale), dimensions={}
      z_scale_bcast = f16[16,16] broadcast(z_scale), dimensions={}
      x_unscaled = f16[16,32] multiply(x_f16, x_scale_bcast)
      y_unscaled = f16[32,16] multiply(y_f16, y_scale_bcast)
      dot_a = f16[16,16] dot(x_unscaled, y_unscaled), lhs_contracting_dims={1}, rhs_contracting_dims={0}
      abs_dot_a = f16[16,16] abs(dot_a)
      c0 = f16[] constant(-inf)
      amax = f16[] reduce(abs_dot_a, c0), dimensions={0,1}, to_apply=apply
      dot_a_scaled = f16[16,16] divide(dot_a, z_scale_bcast)
      c1 = f16[] constant(-<<F8E4M3_AMAX>>)
      c1_bcast = f16[16,16] broadcast(c1), dimensions={}
      c2 = f16[] constant(<<F8E4M3_AMAX>>)
      c2_bcast = f16[16,16] broadcast(c2), dimensions={}
      dot_a_clamped = f16[16,16] clamp(c1_bcast, dot_a_scaled, c2_bcast)
      dot_a_f8 = <<F8E4M3>>[16,16] convert(dot_a_clamped)
      ROOT out = (<<F8E4M3>>[16,16], f16[]) tuple(dot_a_f8, amax)
          }

)";

  CheckFp8IfSupported(hlo_text);
  RunAndFilecheckHloRewrite(
      hlo_text,
      GemmRewriter(CudaHopperOrRocmMI300(), GetToolkitVersion(),
                   GemmRewriterOptions{GemmRewriterOptions::DType::kFp8Only}),
      R"(
; CHECK-LABEL: ENTRY %test ({{.*}}: <<F8E4M3>>[16,32], {{.*}}: <<F8E4M3>>[32,16], {{.*}}: f16[], {{.*}}: f16[], {{.*}}: f16[]) -> (<<F8E4M3>>[16,16], f16[]) {
; CHECK:    [[P0:%[^ ]+]] = <<F8E4M3>>[16,32]{1,0} parameter(0)
; CHECK-NEXT:    [[P1:%[^ ]+]] = <<F8E4M3>>[32,16]{1,0} parameter(1)
; CHECK-NEXT:    [[P1_TRANSPOSE:%[^ ]+]] = <<F8E4M3>>[16,32]{1,0} transpose([[P1]])
; CHECK-NEXT:    [[P2:%[^ ]+]] = f16[] parameter(2)
; CHECK-NEXT:    [[P2_CONVERT:%[^ ]+]] = f32[] convert([[P2]])
; CHECK-NEXT:    [[P3:%[^ ]+]] = f16[] parameter(3)
; CHECK-NEXT:    [[P3_CONVERT:%[^ ]+]] = f32[] convert([[P3]])
; CHECK-PTX-NEXT:    [[C2:%[^ ]+]] = f16[] constant(1)
; CHECK-PTX-NEXT:    [[P4:%[^ ]+]] = f16[] parameter(4)
; CHECK-PTX-NEXT:    [[P4_INV:%[^ ]+]] = f16[] divide([[C2]], [[P4]])
; CHECK-PTX-NEXT:    [[P4_INV_CONVERT:%[^ ]+]] = f32[] convert([[P4_INV]])
; CHECK-PTX-NEXT:    [[OUT:%[^ ]+]] = (<<F8E4M3>>[16,16]{1,0}, f32[], s8[{{[0-9]+}}]{0}) custom-call([[P0]], [[P1_TRANSPOSE]], [[P2_CONVERT]], [[P3_CONVERT]], [[P4_INV_CONVERT]]),
; CHECK-GCN-NEXT:    [[OUT:%[^ ]+]] = (f16[16,16]{1,0}, s8[{{[0-9]+}}]{0}) custom-call([[P0]], [[P1_TRANSPOSE]], [[P2_CONVERT]], [[P3_CONVERT]]),
; CHECK:           custom_call_target="__cublas$lt$matmul$f8",
; CHECK:           backend_config={
; CHECK-DAG:         "alpha_real":1
; CHECK-DAG:         "alpha_imag":0
; CHECK-DAG:         "beta":0
; CHECK-DAG:         "dot_dimension_numbers":{
; CHECK-DAG:           "lhs_contracting_dimensions":["1"]
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

TEST_P(ParameterizedFp8GemmRewriteTest,
       ScaledABScaledDReluActivationWithDAmaxF8) {
  const char* hlo_text = R"(
    HloModule test

    apply {
      a = f32[] parameter(0)
      b = f32[] parameter(1)
      ROOT c = f32[] maximum(a, b)
    }

    ENTRY test {
      x = <<F8E4M3>>[16,32] parameter(0)
      y = <<F8E4M3>>[32,16] parameter(1)
      x_f32 = f32[16,32] convert(x)
      y_f32 = f32[32,16] convert(y)
      x_scale = f32[] parameter(2)
      y_scale = f32[] parameter(3)
      z_scale = f32[] parameter(4)
      x_scale_bcast = f32[16,32] broadcast(x_scale), dimensions={}
      y_scale_bcast = f32[32,16] broadcast(y_scale), dimensions={}
      z_scale_bcast = f32[16,16] broadcast(z_scale), dimensions={}
      x_unscaled = f32[16,32] multiply(x_f32, x_scale_bcast)
      y_unscaled = f32[32,16] multiply(y_f32, y_scale_bcast)
      dot_a = f32[16,16] dot(x_unscaled, y_unscaled), lhs_contracting_dims={1}, rhs_contracting_dims={0}
      czero = f32[] constant(0)
      czero_bcast = f32[16,16] broadcast(czero), dimensions={}
      dot_a_relu = f32[16,16] maximum(dot_a, czero_bcast)
      c0 = f32[] constant(-inf)
      amax = f32[] reduce(dot_a_relu, c0), dimensions={0,1}, to_apply=apply
      dot_a_scaled = f32[16,16] divide(dot_a_relu, z_scale_bcast)
      c1 = f32[] constant(-<<F8E4M3_AMAX>>)
      c1_bcast = f32[16,16] broadcast(c1), dimensions={}
      c2 = f32[] constant(<<F8E4M3_AMAX>>)
      c2_bcast = f32[16,16] broadcast(c2), dimensions={}
      dot_a_clamped = f32[16,16] clamp(c1_bcast, dot_a_scaled, c2_bcast)
      dot_a_f8 = <<F8E4M3>>[16,16] convert(dot_a_clamped)
      ROOT out = (<<F8E4M3>>[16,16], f32[]) tuple(dot_a_f8, amax)
          }

)";

  CheckFp8IfSupported(hlo_text);
  RunAndFilecheckHloRewrite(
      hlo_text,
      GemmRewriter(CudaHopperOrRocmMI300(), GetToolkitVersion(),
                   GemmRewriterOptions{GemmRewriterOptions::DType::kFp8Only}),
      R"(
; CHECK-LABEL: ENTRY %test ({{.*}}: <<F8E4M3>>[16,32], {{.*}}: <<F8E4M3>>[32,16], {{.*}}: f32[], {{.*}}: f32[], {{.*}}: f32[]) -> (<<F8E4M3>>[16,16], f32[]) {
; CHECK:    [[P0:%[^ ]+]] = <<F8E4M3>>[16,32]{1,0} parameter(0)
; CHECK-NEXT:    [[P1:%[^ ]+]] = <<F8E4M3>>[32,16]{1,0} parameter(1)
; CHECK-NEXT:    [[P1_TRANSPOSE:%[^ ]+]] = <<F8E4M3>>[16,32]{1,0} transpose([[P1]])
; CHECK-NEXT:    [[P2:%[^ ]+]] = f32[] parameter(2)
; CHECK-NEXT:    [[P3:%[^ ]+]] = f32[] parameter(3)
; CHECK-PTX-NEXT:    [[C2:%[^ ]+]] = f32[] constant(1)
; CHECK-PTX-NEXT:    [[P4:%[^ ]+]] = f32[] parameter(4)
; CHECK-PTX-NEXT:    [[P4_INV:%[^ ]+]] = f32[] divide([[C2]], [[P4]])
; CHECK-PTX-NEXT:    [[OUT:%[^ ]+]] = (<<F8E4M3>>[16,16]{1,0}, f32[], s8[{{[0-9]+}}]{0}) custom-call([[P0]], [[P1_TRANSPOSE]], [[P2]], [[P3]], [[P4_INV]]),
; CHECK-GCN-NEXT:    [[OUT:%[^ ]+]] = (f32[16,16]{1,0}, s8[{{[0-9]+}}]{0}) custom-call([[P0]], [[P1_TRANSPOSE]], [[P2]], [[P3]]),
; CHECK:           custom_call_target="__cublas$lt$matmul$f8",
; CHECK:           backend_config={
; CHECK-DAG:         "alpha_real":1
; CHECK-DAG:         "alpha_imag":0
; CHECK-DAG:         "beta":0
; CHECK-DAG:         "dot_dimension_numbers":{
; CHECK-DAG:           "lhs_contracting_dimensions":["1"]
; CHECK-DAG:           "rhs_contracting_dimensions":["1"]
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

TEST_P(ParameterizedFp8GemmRewriteTest, UnscaledABUnscaledDPrecisionF8) {
  const char* raw_hlo_template = R"(
    HloModule test

    ENTRY test {
      x = <<F8E4M3>>[1600,3200] parameter(0)
      y = <<F8E4M3>>[3200,1600] parameter(1)
      x_f32 = f32[1600,3200] convert(x)
      y_f32 = f32[3200,1600] convert(y)
      x_scale = f32[] parameter(2)
      y_scale = f32[] parameter(3)
      x_scale_bcast = f32[1600,3200] broadcast(x_scale), dimensions={}
      y_scale_bcast = f32[3200,1600] broadcast(y_scale), dimensions={}
      x_unscaled = f32[1600,3200] multiply(x_f32, x_scale_bcast)
      y_unscaled = f32[3200,1600] multiply(y_f32, y_scale_bcast)
      ROOT out = f32[1600,1600] dot(x_unscaled, y_unscaled), lhs_contracting_dims={1}, rhs_contracting_dims={0}, operand_precision={<<precision>>,<<precision>>}
          }
)";

  std::string hlo_template =
      absl::StrReplaceAll(raw_hlo_template, replacements_);

  absl::flat_hash_map<absl::string_view, absl::string_view> replacements;
  replacements["<<precision>>"] = "default";
  const auto hlo_text_default = absl::StrReplaceAll(hlo_template, replacements);
  EXPECT_TRUE(RunAndCompare(hlo_text_default, ErrorSpec{1e-3, 1e-3}));

  replacements["<<precision>>"] = "highest";
  const auto hlo_text_highest = absl::StrReplaceAll(hlo_template, replacements);
  EXPECT_TRUE(RunAndCompare(hlo_text_highest, ErrorSpec{1e-4, 1e-4}));
}

TEST_P(ParameterizedFp8GemmRewriteTest, ScaledABUnscaledDF8Parameterized) {
  std::array<std::array<absl::string_view, 7>, 32> combinations;
  int i = 0;

  for (bool d_is_col : {false, true}) {
    for (bool a_is_col : {false, true}) {
      for (bool b_is_col : {false, true}) {
        for (int lhs_contracting_dim : {0, 1}) {
          for (int rhs_contracting_dim : {0, 1}) {
            const absl::string_view lcd =
                lhs_contracting_dim == 1 ? "{1}" : "{0}";
            const absl::string_view rcd =
                rhs_contracting_dim == 1 ? "{1}" : "{0}";
            const absl::string_view a_shape =
                lhs_contracting_dim == 1 ? "[64,32]" : "[32,64]";
            const absl::string_view b_shape =
                rhs_contracting_dim == 0 ? "[32,16]" : "[16,32]";
            const absl::string_view a_layout = a_is_col ? "{0,1}" : "{1,0}";
            const absl::string_view b_layout = b_is_col ? "{0,1}" : "{1,0}";
            const absl::string_view output_layout =
                d_is_col ? "{0,1}" : "{1,0}";
            combinations[i++] = std::array{
                lcd, rcd, a_shape, b_shape, a_layout, b_layout, output_layout};
          }
        }
      }
    }
  }

  const char* hlo_template = R"(
      HloModule test
    ENTRY test {
      x = <<F8E4M3>><<Ashape>><<Alayout>> parameter(0)
      x_f32 = f32<<Ashape>><<Alayout>> convert(x)
      x_scale = f32[] parameter(2)
      x_scale_bcast = f32<<Ashape>> broadcast(x_scale), dimensions={}
      x_unscaled = f32<<Ashape>> multiply(x_f32, x_scale_bcast)
      y = <<F8E4M3>><<Bshape>><<Blayout>> parameter(1)
      y_f32 = f32<<Bshape>><<Blayout>> convert(y)
      y_scale = f32[] parameter(3)
      y_scale_bcast = f32<<Bshape>> broadcast(y_scale), dimensions={}
      y_unscaled = f32<<Bshape>> multiply(y_f32, y_scale_bcast)
      ROOT out = f32[64,16]<<Olayout>> dot(x_unscaled, y_unscaled), lhs_contracting_dims=<<Lcd>>, rhs_contracting_dims=<<Rcd>>
    }
      )";
  for (const auto& combination : combinations) {
    absl::flat_hash_map<absl::string_view, absl::string_view> replacements;
    replacements["<<Lcd>>"] = std::get<0>(combination);
    replacements["<<Rcd>>"] = std::get<1>(combination);
    replacements["<<Ashape>>"] = std::get<2>(combination);
    replacements["<<Bshape>>"] = std::get<3>(combination);
    replacements["<<Alayout>>"] = std::get<4>(combination);
    replacements["<<Blayout>>"] = std::get<5>(combination);
    replacements["<<Olayout>>"] = std::get<6>(combination);
    const auto hlo_text = absl::StrReplaceAll(hlo_template, replacements);
    CheckFp8IfSupported(hlo_text);

    RunAndFilecheckHloRewrite(
        hlo_text,
        GemmRewriter(CudaHopperOrRocmMI300(), GetToolkitVersion(),
                     GemmRewriterOptions{GemmRewriterOptions::DType::kFp8Only}),
        R"(
    ; CHECK:           custom_call_target="__cublas$lt$matmul$f8",
          )");
  }
}

TEST_P(ParameterizedFp8GemmRewriteTest,
       ScaledABUnscaledDF8ParameterizedBatched) {
  // TODO(wenscarl): For batched matmul, not all combinations of A, B and
  // output layouts get pattern matched successfully to FP8 custom call. Only
  // a handful of cases are tested here.
  std::array<std::array<std::string, 7>, 32> combinations;
  std::string lcd, rcd, a_shape, b_shape, a_layout, b_layout, o_layout;
  int i = 0;
  for (bool o_is_col : {false, true}) {
    for (int lhs_contracting_dim : {2, 1}) {
      for (int rhs_contracting_dim : {2, 1}) {
        lcd = lhs_contracting_dim == 2 ? "{2}" : "{1}";
        rcd = rhs_contracting_dim == 2 ? "{2}" : "{1}";
        a_shape = lhs_contracting_dim == 2 ? "[2,64,32]" : "[2,32,64]";
        b_shape = rhs_contracting_dim == 1 ? "[2,32,16]" : "[2,16,32]";
        o_layout = o_is_col ? "{2, 0, 1}" : "{2, 1, 0}";
        for (std::string a_layout : {"{2,1,0}", "{1,2,0}"}) {
          for (std::string b_layout : {"{2,1,0}", "{1,2,0}"}) {
            combinations[i++] = std::array{lcd,      rcd,      a_shape, b_shape,
                                           a_layout, b_layout, o_layout};
          }
        }
      }
    }
  }

  const char* hlo_template = R"(
      HloModule m
ENTRY f {
  x_q = <<F8E4M3>><<Ashape>><<Alayout>> parameter(0)
  x_scale = f32[] parameter(2)
  x_scale_broadcast = f32<<Ashape>><<Alayout>> broadcast(x_scale), dimensions={}
  x_q_convert = f32<<Ashape>><<Alayout>> convert(x_q)
  x_qdq = f32<<Ashape>><<Alayout>> multiply(x_q_convert, x_scale_broadcast)

  y_q = <<F8E4M3>><<Bshape>><<Blayout>> parameter(1)
  y_scale = f32[] parameter(3)
  y_scale_broadcast = f32<<Bshape>><<Blayout>> broadcast(y_scale), dimensions={}
  y_q_convert = f32<<Bshape>><<Blayout>> convert(y_q)
  y_qdq = f32<<Bshape>><<Blayout>> multiply(y_q_convert, y_scale_broadcast)

  ROOT out = f32[2,64,16]<<Olayout>> dot(x_qdq, y_qdq), lhs_batch_dims={0}, lhs_contracting_dims=<<Lcd>>, rhs_batch_dims={0}, rhs_contracting_dims=<<Rcd>>
}
     )";

  for (const auto& combination : combinations) {
    absl::flat_hash_map<std::string, std::string> replacements;
    replacements["<<Lcd>>"] = std::get<0>(combination);
    replacements["<<Rcd>>"] = std::get<1>(combination);
    replacements["<<Ashape>>"] = std::get<2>(combination);
    replacements["<<Bshape>>"] = std::get<3>(combination);
    replacements["<<Alayout>>"] = std::get<4>(combination);
    replacements["<<Blayout>>"] = std::get<5>(combination);
    replacements["<<Olayout>>"] = std::get<6>(combination);

    const auto hlo_text = absl::StrReplaceAll(hlo_template, replacements);
    CheckFp8IfSupported(hlo_text);

    RunAndFilecheckHloRewrite(
        hlo_text,
        GemmRewriter(CudaHopperOrRocmMI300(), GetToolkitVersion(),
                     GemmRewriterOptions{GemmRewriterOptions::DType::kFp8Only}),
        R"(
    ; CHECK:           custom_call_target="__cublas$lt$matmul$f8",
          )");
  }
}

TEST_P(ParameterizedFp8GemmRewriteTest, ScaledABUnscaledDF8TF32E5M2) {
  const char* hlo_text = R"(
    HloModule test

    ENTRY test {
      x = <<F8E4M3>>[16,32] parameter(0)
      y = <<F8E5M2>>[32,16] parameter(1)
      x_f32 = f32[16,32] convert(x)
      y_f32 = f32[32,16] convert(y)
      x_scale = f32[] parameter(2)
      y_scale = f32[] parameter(3)
      x_scale_bcast = f32[16,32] broadcast(x_scale), dimensions={}
      y_scale_bcast = f32[32,16] broadcast(y_scale), dimensions={}
      x_unscaled = f32[16,32] multiply(x_f32, x_scale_bcast)
      y_unscaled = f32[32,16] multiply(y_f32, y_scale_bcast)
      ROOT out = f32[16,16] dot(x_unscaled, y_unscaled), lhs_contracting_dims={1}, rhs_contracting_dims={0}
          }

)";

  CheckFp8IfSupported(hlo_text);
  RunAndFilecheckHloRewrite(
      hlo_text,
      GemmRewriter(CudaHopperOrRocmMI300(), GetToolkitVersion(),
                   GemmRewriterOptions{GemmRewriterOptions::DType::kFp8Only}),
      R"(
    ; CHECK:           custom_call_target="__cublas$lt$matmul$f8",
          )");
}

TEST_P(ParameterizedFp8GemmRewriteTest, FnuzTypeF8) {
  // Test that FNUZ FP8 gemms are not rewritten, as cuBLAS does not support them
  const char* hlo_text = R"(
    HloModule test

    ENTRY test {
      x = f8e4m3fnuz[16,32] parameter(0)
      y = f8e4m3fnuz[32,16] parameter(1)
      x_f32 = f32[16,32] convert(x)
      y_f32 = f32[32,16] convert(y)
      x_scale = f32[] parameter(2)
      y_scale = f32[] parameter(3)
      x_scale_bcast = f32[16,32] broadcast(x_scale), dimensions={}
      y_scale_bcast = f32[32,16] broadcast(y_scale), dimensions={}
      x_unscaled = f32[16,32] multiply(x_f32, x_scale_bcast)
      y_unscaled = f32[32,16] multiply(y_f32, y_scale_bcast)
      ROOT out = f32[16,16] dot(x_unscaled, y_unscaled), lhs_contracting_dims={1}, rhs_contracting_dims={0}
          }
)";
  if (IsCuda()) {
    TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                            ParseAndReturnVerifiedModule(hlo_text));
    GemmRewriter pass(
        CudaHopperOrRocmMI300(), GetToolkitVersion(),
        GemmRewriterOptions{GemmRewriterOptions::DType::kFp8Only});
    TF_ASSERT_OK_AND_ASSIGN(bool changed,
                            this->RunHloPass(&pass, module.get()));
    EXPECT_FALSE(changed);
    return;
  }
  if (IsRocm()) {
    EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{1e-2, 1e-2}));
    RunAndFilecheckHloRewrite(
        hlo_text,
        GemmRewriter(CudaHopperOrRocmMI300(), GetToolkitVersion(),
                     GemmRewriterOptions{GemmRewriterOptions::DType::kFp8Only}),
        R"(
; CHECK-LABEL: ENTRY %test ({{.*}}: f8e4m3fnuz[16,32], {{.*}}: f8e4m3fnuz[32,16], {{.*}}: f32[], {{.*}}: f32[]) -> f32[16,16] {
; CHECK-NEXT:    [[P0:%[^ ]+]] = f8e4m3fnuz[16,32]{1,0} parameter(0)
; CHECK-PTX-NEXT:    [[P0_CV:%[^ ]+]] = f32[16,32]{1,0} convert([[P0]])
; CHECK-PTX-NEXT:    [[P2:%[^ ]+]] = f32[] parameter(2)
; CHECK-PTX-NEXT:    [[P2_B:%[^ ]+]] = f32[16,32]{1,0} broadcast([[P2]]), dimensions={}
; CHECK-PTX-NEXT:    [[P0_UNSCALED:%[^ ]+]] = f32[16,32]{1,0} multiply([[P0_CV]], [[P2_B]])
; CHECK-PTX-NEXT:    [[P1:%[^ ]+]] = f8e4m3fnuz[32,16]{1,0} parameter(1)
; CHECK-PTX-NEXT:    [[P1_CV:%[^ ]+]] = f32[32,16]{1,0} convert([[P1]])
; CHECK-PTX-NEXT:    [[P3:%[^ ]+]] = f32[] parameter(3)
; CHECK-PTX-NEXT:    [[P3_B:%[^ ]+]] = f32[32,16]{1,0} broadcast([[P3]]), dimensions={}
; CHECK-PTX-NEXT:    [[P1_UNSCALED:%[^ ]+]] = f32[32,16]{1,0} multiply([[P1_CV]], [[P3_B]])
; CHECK-PTX-NEXT:    [[GEMM:%[^ ]+]] = {{.*}} custom-call([[P0_UNSCALED]], [[P1_UNSCALED]]),
; CHECK-GCN-NEXT:    [[P1:%[^ ]+]] = f8e4m3fnuz[32,16]{1,0} parameter(1)
; CHECK-GCN-NEXT:    [[P1_TRANSPOSE:%[^ ]+]] = <<F8E4M3>>[16,32]{1,0} transpose([[P1]])
; CHECK-GCN-NEXT:    [[P2:%[^ ]+]] = f32[] parameter(2)
; CHECK-GCN-NEXT:    [[P3:%[^ ]+]] = f32[] parameter(3)
; CHECK-PTX:           custom_call_target="<<CUBLAS_CUSTOM_CALL_TARGET_PLACEHOLDER>>",
; CHECK-GCN:           custom_call_target="__cublas$lt$matmul$f8",
; CHECK:           backend_config={
; CHECK-DAG:         "alpha_real":1
; CHECK-DAG:         "alpha_imag":0
; CHECK-DAG:         "beta":0
; CHECK-DAG:         "dot_dimension_numbers":{
; CHECK-DAG:           "lhs_contracting_dimensions":["1"]
; CHECK-PTX-DAG:           "rhs_contracting_dimensions":["0"]
; CHECK-GCN-DAG:           "rhs_contracting_dimensions":["1"]
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
}

TEST_P(ParameterizedFp8GemmRewriteTest, NoTransposeOnBlackwellF8) {
  if (!IsBlackwell()) {
    GTEST_SKIP() << "Test requires a Blackwell GPU.";
  }
  const char* hlo_text = R"(
    HloModule test

    ENTRY test {
      x = <<F8E4M3>>[32,16] parameter(0)
      y = <<F8E4M3>>[32,16] parameter(1)
      ROOT out = <<F8E4M3>>[16,16] dot(x, y), lhs_contracting_dims={0}, rhs_contracting_dims={0}
          }

)";

  EXPECT_TRUE(RunAndCompare(absl::StrReplaceAll(hlo_text, replacements_),
                            ErrorSpec{1e-2, 1e-2}));
  MatchOptimizedHlo(hlo_text,
                    R"(
; CHECK-LABEL: ENTRY %test ({{.*}}: <<F8E4M3>>[32,16], {{.*}}: <<F8E4M3>>[32,16]) -> <<F8E4M3>>[16,16] {
; CHECK-NEXT:    [[P0:%[^ ]+]] = <<F8E4M3>>[32,16]{1,0} parameter(0)
; CHECK-NEXT:    [[P1:%[^ ]+]] = <<F8E4M3>>[32,16]{1,0} parameter(1)
; CHECK-NEXT:    [[C1:[^ ]+]] = f32[] constant(1)
; CHECK-NEXT:    [[OUT:%[^ ]+]] = (<<F8E4M3>>[16,16]{1,0}, s8[{{[0-9]+}}]{0}) custom-call([[P0]], [[P1]], [[C1]], [[C1]]),
; CHECK:           custom_call_target="__cublas$lt$matmul$f8",
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

INSTANTIATE_TEST_SUITE_P(Fp8CublasTestsBothLegacyAndLt,
                         ParameterizedFp8GemmRewriteTest, ::testing::Bool());

}  // namespace
}  // namespace gpu
}  // namespace xla

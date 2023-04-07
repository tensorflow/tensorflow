/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/strings/str_replace.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_instruction.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_module.h"
#include "tensorflow/compiler/xla/service/gpu/gemm_rewriter.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_executable.h"
#include "tensorflow/compiler/xla/service/gpu/stream_executor_util.h"
#include "tensorflow/compiler/xla/service/gpu/tests/gpu_codegen_test.h"
#include "tensorflow/compiler/xla/service/hlo_module_config.h"
#include "tensorflow/compiler/xla/service/hlo_parser.h"
#include "tensorflow/compiler/xla/service/pattern_matcher.h"
#include "tensorflow/compiler/xla/service/pattern_matcher_gmock.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/tests/filecheck.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/compiler/xla/xla.pb.h"
#include "tensorflow/tsl/lib/core/status_test_util.h"
#include "tensorflow/tsl/platform/tensor_float_32_utils.h"
#include "tensorflow/tsl/platform/test.h"

namespace xla {
namespace gpu {

namespace {

namespace m = ::xla::match;

class GemmRewriteTest : public GpuCodegenTest {
 public:
  se::CudaComputeCapability GetCudaComputeCapability() {
    return backend()
        .default_stream_executor()
        ->GetDeviceDescription()
        .cuda_compute_capability();
  }
  void SetUp() override {
    tf32_state_ = tsl::tensor_float_32_execution_enabled();
    tsl::enable_tensor_float_32_execution(false);
  }
  void TearDown() override {
    tsl::enable_tensor_float_32_execution(tf32_state_);
  }

 private:
  bool tf32_state_;
};

TEST_F(GemmRewriteTest, CheckCustomCallTarget) {
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
  if (GetCudaComputeCapability().IsAtLeast(se::CudaComputeCapability::AMPERE)) {
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
    debug_options.set_xla_gpu_deterministic_ops(true);
    config.set_debug_options(debug_options);
    return ParseAndReturnVerifiedModule(hlo_text, config);
  };

  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HloModule> optimized_module,
      backend().compiler()->RunHloPasses(
          *get_module(), backend().default_stream_executor(),
          backend().default_stream_executor()->GetAllocator()));

  StatusOr<bool> filecheck_result = RunFileCheck(optimized_module->ToString(),
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

  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{1e-5, 1e-5}));
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

// A test fixture class for tests which should have similar results with legacy
// cublas and cublasLt
class ParameterizedGemmRewriteTest
    : public GemmRewriteTest,
      public ::testing::WithParamInterface<bool> {
 public:
  ParameterizedGemmRewriteTest() {
    const bool kUsingCublasLt = GetParam();
    replacements_[kCustomCallTargetPlaceholder] =
        kUsingCublasLt ? "__cublas$lt$matmul" : "__cublas$gemm";
  }
  DebugOptions GetDebugOptionsForTest() override {
    DebugOptions debug_options = GemmRewriteTest::GetDebugOptionsForTest();
    debug_options.set_xla_gpu_enable_cublaslt(GetParam());
    return debug_options;
  }
  void MatchOptimizedHlo(absl::string_view hlo, const absl::string_view pattern,
                         bool print_operand_shape = false) {
    GemmRewriteTest::MatchOptimizedHlo(
        hlo, absl::StrReplaceAll(pattern, replacements_), print_operand_shape);
  }
  absl::string_view CustomCallTarget() {
    return replacements_[kCustomCallTargetPlaceholder];
  }

 protected:
  absl::flat_hash_map<absl::string_view, absl::string_view> replacements_;

 private:
  static constexpr const char* kCustomCallTargetPlaceholder{
      "<<CUBLAS_CUSTOM_CALL_TARGET_PLACEHOLDER>>"};
};

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
; CHECK-LABEL: ENTRY %test (x: f32[2,3], y: f32[3,4]) -> f32[2,4] {
; CHECK-NEXT:    [[P0:%[^ ]+]] = f32[2,3]{1,0} parameter(0)
; CHECK-NEXT:    [[P1:%[^ ]+]] = f32[3,4]{1,0} parameter(1)
; CHECK-NEXT:    ROOT [[OUT:%[^ ]+]] = f32[2,4]{1,0} custom-call([[P0]], [[P1]]),
; CHECK:           custom_call_target="<<CUBLAS_CUSTOM_CALL_TARGET_PLACEHOLDER>>",
; CHECK:           backend_config="{
; CHECK-DAG:         \"alpha_real\":1
; CHECK-DAG:         \"alpha_imag\":0
; CHECK-DAG:         \"beta\":0
; CHECK-DAG:         \"dot_dimension_numbers\":{
; CHECK-DAG:           \"lhs_contracting_dimensions\":[\"1\"]
; CHECK-DAG:           \"rhs_contracting_dimensions\":[\"0\"]
; CHECK-DAG:           \"lhs_batch_dimensions\":[]
; CHECK-DAG:           \"rhs_batch_dimensions\":[]
; CHECK-DAG:         }
; CHECK-DAG:         \"precision_config\":{
; CHECK-DAG:           \"operand_precision\":[\"DEFAULT\",\"DEFAULT\"]
; CHECK-DAG:         }
; CHECK-DAG:         \"epilogue\":\"DEFAULT\"
; CHECK:           }"
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
; CHECK-LABEL: ENTRY %AddDotsFunc (x: f32[2,3], y: f32[3,4]) -> f32[2,4] {
; CHECK-NEXT:    [[P0:%[^ ]+]] = f32[2,3]{1,0} parameter(0)
; CHECK-NEXT:    [[P1:%[^ ]+]] = f32[3,4]{1,0} parameter(1)
; CHECK-NEXT:    ROOT [[OUT:%[^ ]+]] = f32[2,4]{1,0} custom-call([[P0]], [[P1]]),
; CHECK:           custom_call_target="<<CUBLAS_CUSTOM_CALL_TARGET_PLACEHOLDER>>",
; CHECK:           backend_config="{
; CHECK-DAG:         \"alpha_real\":1
; CHECK-DAG:         \"alpha_imag\":0
; CHECK-DAG:         \"beta\":0
; CHECK-DAG:         \"dot_dimension_numbers\":{
; CHECK-DAG:           \"lhs_contracting_dimensions\":[\"1\"]
; CHECK-DAG:           \"rhs_contracting_dimensions\":[\"0\"]
; CHECK-DAG:           \"lhs_batch_dimensions\":[]
; CHECK-DAG:           \"rhs_batch_dimensions\":[]
; CHECK-DAG:         }
; CHECK-DAG:         \"precision_config\":{
; CHECK-DAG:           \"operand_precision\":[\"DEFAULT\",\"DEFAULT\"]
; CHECK-DAG:         }
; CHECK-DAG:         \"epilogue\":\"DEFAULT\"
; CHECK:           }"
)");
}

TEST_P(ParameterizedGemmRewriteTest, MultipleContractingDims) {
  const char* hlo_text = R"(
HloModule MultipleContractingCheckGemm

ENTRY AddDotsFunc {
  x = f32[3,4,2] parameter(0)
  y = f32[3,4,5] parameter(1)
  ROOT dot_a = f32[2,5] dot(x, y), lhs_contracting_dims={0,1}, rhs_contracting_dims={0,1}
}

)";

  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{1e-5, 1e-5}));
  MatchOptimizedHlo(hlo_text,
                    R"(
; CHECK-NOT:     copy
;
; CHECK-LABEL: ENTRY %AddDotsFunc (x: f32[3,4,2], y: f32[3,4,5]) -> f32[2,5] {
; CHECK-NEXT:    [[P0:%[^ ]+]] = f32[3,4,2]{2,1,0} parameter(0)
; CHECK-DAG:     [[P1:%[^ ]+]] = f32[3,4,5]{2,1,0} parameter(1)
; CHECK-DAG:     [[BITCAST0:%[^ ]+]] = f32[2,12]{0,1} bitcast([[P0]])
; CHECK-DAG:     [[BITCAST1:%[^ ]+]] = f32[12,5]{1,0} bitcast([[P1]])
; CHECK-NEXT:    ROOT [[OUT:%[^ ]+]] = f32[2,5]{1,0} custom-call([[BITCAST0]], [[BITCAST1]]),
; CHECK:           custom_call_target="<<CUBLAS_CUSTOM_CALL_TARGET_PLACEHOLDER>>",
; CHECK:           backend_config="{
; CHECK-DAG:         \"alpha_real\":1
; CHECK-DAG:         \"alpha_imag\":0
; CHECK-DAG:         \"beta\":0
; CHECK-DAG:         \"dot_dimension_numbers\":{
; CHECK-DAG:           \"lhs_contracting_dimensions\":[\"1\"]
; CHECK-DAG:           \"rhs_contracting_dimensions\":[\"0\"]
; CHECK-DAG:           \"lhs_batch_dimensions\":[]
; CHECK-DAG:           \"rhs_batch_dimensions\":[]
; CHECK-DAG:         }
; CHECK-DAG:         \"precision_config\":{
; CHECK-DAG:           \"operand_precision\":[\"DEFAULT\",\"DEFAULT\"]
; CHECK-DAG:         }
; CHECK-DAG:         \"epilogue\":\"DEFAULT\"
; CHECK:           }"
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
; CHECK-LABEL: ENTRY %AddDotsFunc (x: f32[3,2], y: f32[3,4]) -> f32[2,4] {
; CHECK-NEXT:    [[P0:%[^ ]+]] = f32[3,2]{1,0} parameter(0)
; CHECK-NEXT:    [[P1:%[^ ]+]] = f32[3,4]{1,0} parameter(1)
; CHECK-NEXT:    ROOT [[OUT:%[^ ]+]] = f32[2,4]{1,0} custom-call([[P0]], [[P1]]),
; CHECK:           custom_call_target="<<CUBLAS_CUSTOM_CALL_TARGET_PLACEHOLDER>>",
; CHECK:           backend_config="{
; CHECK-DAG:         \"alpha_real\":1
; CHECK-DAG:         \"alpha_imag\":0
; CHECK-DAG:         \"beta\":0
; CHECK-DAG:         \"dot_dimension_numbers\":{
; CHECK-DAG:           \"lhs_contracting_dimensions\":[\"0\"]
; CHECK-DAG:           \"rhs_contracting_dimensions\":[\"0\"]
; CHECK-DAG:           \"lhs_batch_dimensions\":[]
; CHECK-DAG:           \"rhs_batch_dimensions\":[]
; CHECK-DAG:         }
; CHECK-DAG:         \"precision_config\":{
; CHECK-DAG:           \"operand_precision\":[\"DEFAULT\",\"DEFAULT\"]
; CHECK-DAG:         }
; CHECK-DAG:         \"epilogue\":\"DEFAULT\"
; CHECK:           }"
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
; CHECK-LABEL: ENTRY %AddDotsFunc (x: f32[5,3,2], y: f32[5,3,4]) -> f32[5,2,4] {
; CHECK-NEXT:    [[P0:%[^ ]+]] = f32[5,3,2]{2,1,0} parameter(0)
; CHECK-NEXT:    [[P1:%[^ ]+]] = f32[5,3,4]{2,1,0} parameter(1)
; CHECK-NEXT:    ROOT [[OUT:%[^ ]+]] = f32[5,2,4]{2,1,0} custom-call([[P0]], [[P1]]),
; CHECK:           custom_call_target="<<CUBLAS_CUSTOM_CALL_TARGET_PLACEHOLDER>>",
; CHECK:           backend_config="{
; CHECK-DAG:         \"alpha_real\":1
; CHECK-DAG:         \"alpha_imag\":0
; CHECK-DAG:         \"beta\":0
; CHECK-DAG:         \"dot_dimension_numbers\":{
; CHECK-DAG:           \"lhs_contracting_dimensions\":[\"1\"]
; CHECK-DAG:           \"rhs_contracting_dimensions\":[\"1\"]
; CHECK-DAG:           \"lhs_batch_dimensions\":[\"0\"]
; CHECK-DAG:           \"rhs_batch_dimensions\":[\"0\"]
; CHECK-DAG:         }
; CHECK-DAG:         \"precision_config\":{
; CHECK-DAG:           \"operand_precision\":[\"DEFAULT\",\"DEFAULT\"]
; CHECK-DAG:         }
; CHECK-DAG:         \"epilogue\":\"DEFAULT\"
; CHECK:           }"
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

  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{1e-5, 1e-5}));
  MatchOptimizedHlo(hlo_text,
                    R"(
; CHECK-LABEL: ENTRY %AddDotsFunc (x: f32[2,5,3], y: f32[5,3,4]) -> f32[5,2,4] {
; CHECK-NEXT:    [[P0:%[^ ]+]] = f32[2,5,3]{2,1,0} parameter(0)
; CHECK-NEXT:    [[P1:%[^ ]+]] = f32[5,3,4]{2,1,0} parameter(1)
; CHECK-NEXT:    ROOT [[OUT:%[^ ]+]] = f32[5,2,4]{2,1,0} custom-call([[P0]], [[P1]]),
; CHECK:           custom_call_target="<<CUBLAS_CUSTOM_CALL_TARGET_PLACEHOLDER>>",
; CHECK:           backend_config="{
; CHECK-DAG:         \"alpha_real\":1
; CHECK-DAG:         \"alpha_imag\":0
; CHECK-DAG:         \"beta\":0
; CHECK-DAG:         \"dot_dimension_numbers\":{
; CHECK-DAG:           \"lhs_contracting_dimensions\":[\"2\"]
; CHECK-DAG:           \"rhs_contracting_dimensions\":[\"1\"]
; CHECK-DAG:           \"lhs_batch_dimensions\":[\"1\"]
; CHECK-DAG:           \"rhs_batch_dimensions\":[\"0\"]
; CHECK-DAG:         }
; CHECK-DAG:         \"precision_config\":{
; CHECK-DAG:           \"operand_precision\":[\"DEFAULT\",\"DEFAULT\"]
; CHECK-DAG:         }
; CHECK-DAG:         \"epilogue\":\"DEFAULT\"
; CHECK:           }"
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

  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{1e-5, 1e-5}));
  MatchOptimizedHlo(hlo_text,
                    R"(
; CHECK-LABEL: ENTRY %AddDotsFunc (x: f32[3,2,5], y: f32[5,3,4]) -> f32[5,2,4] {
; CHECK-NEXT:    [[P0:%[^ ]+]] = f32[3,2,5]{2,1,0} parameter(0)
; CHECK-DAG:     [[P1:%[^ ]+]] = f32[5,3,4]{2,1,0} parameter(1)
; CHECK-DAG:     [[FUSION:%[^ ]+]] = f32[5,2,3]{2,1,0} transpose([[P0]])
; CHECK-NEXT:    ROOT [[OUT:%[^ ]+]] = f32[5,2,4]{2,1,0} custom-call([[FUSION]], [[P1]]),
; CHECK:           custom_call_target="<<CUBLAS_CUSTOM_CALL_TARGET_PLACEHOLDER>>",
; CHECK:           backend_config="{
; CHECK-DAG:         \"alpha_real\":1
; CHECK-DAG:         \"alpha_imag\":0
; CHECK-DAG:         \"beta\":0
; CHECK-DAG:         \"dot_dimension_numbers\":{
; CHECK-DAG:           \"lhs_contracting_dimensions\":[\"2\"]
; CHECK-DAG:           \"rhs_contracting_dimensions\":[\"1\"]
; CHECK-DAG:           \"lhs_batch_dimensions\":[\"0\"]
; CHECK-DAG:           \"rhs_batch_dimensions\":[\"0\"]
; CHECK-DAG:         }
; CHECK-DAG:         \"precision_config\":{
; CHECK-DAG:           \"operand_precision\":[\"DEFAULT\",\"DEFAULT\"]
; CHECK-DAG:         }
; CHECK-DAG:         \"epilogue\":\"DEFAULT\"
; CHECK:           }"
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
; CHECK-LABEL: ENTRY %AddDotsFunc (x: f32[20000,4,3,2], y: f32[20000,4,3,4]) -> f32[20000,4,2,4] {
; CHECK-NEXT:    [[P0:%[^ ]+]] = f32[20000,4,3,2]{3,2,1,0} parameter(0)
; CHECK-NEXT:    [[P1:%[^ ]+]] = f32[20000,4,3,4]{3,2,1,0} parameter(1)
; CHECK-NEXT:    ROOT [[OUT:%[^ ]+]] = f32[20000,4,2,4]{3,2,1,0} custom-call([[P0]], [[P1]]),
; CHECK:           custom_call_target="__cublas$gemm",
; CHECK:           backend_config="{
; CHECK-DAG:         \"alpha_real\":1
; CHECK-DAG:         \"alpha_imag\":0
; CHECK-DAG:         \"beta\":0
; CHECK-DAG:         \"dot_dimension_numbers\":{
; CHECK-DAG:           \"lhs_contracting_dimensions\":[\"2\"]
; CHECK-DAG:           \"rhs_contracting_dimensions\":[\"2\"]
; CHECK-DAG:           \"lhs_batch_dimensions\":[\"0\",\"1\"]
; CHECK-DAG:           \"rhs_batch_dimensions\":[\"0\",\"1\"]
; CHECK-DAG:         }
; CHECK-DAG:         \"precision_config\":{
; CHECK-DAG:           \"operand_precision\":[\"DEFAULT\",\"DEFAULT\"]
; CHECK-DAG:         }
; CHECK:           }"
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
; CHECK-LABEL: ENTRY %AddDotsFunc (x: f32[2,3], y: f32[3,4]) -> f32[4,2] {
; CHECK-NEXT:    [[P1:%[^ ]+]] = f32[3,4]{1,0} parameter(1)
; CHECK-NEXT:    [[P0:%[^ ]+]] = f32[2,3]{1,0} parameter(0)
; CHECK-NEXT:    ROOT [[OUT:%[^ ]+]] = f32[4,2]{1,0} custom-call([[P1]], [[P0]]),
; CHECK:           custom_call_target="<<CUBLAS_CUSTOM_CALL_TARGET_PLACEHOLDER>>",
; CHECK:           backend_config="{
; CHECK-DAG:         \"alpha_real\":1
; CHECK-DAG:         \"alpha_imag\":0
; CHECK-DAG:         \"beta\":0
; CHECK-DAG:         \"dot_dimension_numbers\":{
; CHECK-DAG:           \"lhs_contracting_dimensions\":[\"0\"]
; CHECK-DAG:           \"rhs_contracting_dimensions\":[\"1\"]
; CHECK-DAG:           \"lhs_batch_dimensions\":[]
; CHECK-DAG:           \"rhs_batch_dimensions\":[]
; CHECK-DAG:         }
; CHECK-DAG:         \"precision_config\":{
; CHECK-DAG:           \"operand_precision\":[\"DEFAULT\",\"DEFAULT\"]
; CHECK-DAG:         }
; CHECK-DAG:         \"epilogue\":\"DEFAULT\"
; CHECK:           }"
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

  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{1e-5, 1e-5}));
  MatchOptimizedHlo(hlo_text,
                    R"(
; CHECK-LABEL: ENTRY %AddDotsFunc (x: f32[5,2,3], y: f32[5,3,4]) -> f32[2,5,4] {
; CHECK-NEXT:    [[P0:%[^ ]+]] = f32[5,2,3]{2,1,0} parameter(0)
; CHECK-NEXT:    [[P1:%[^ ]+]] = f32[5,3,4]{2,1,0} parameter(1)
; CHECK-NEXT:    [[GEMM:%[^ ]+]] = f32[5,2,4]{2,0,1} custom-call([[P0]], [[P1]]),
; CHECK:           custom_call_target="<<CUBLAS_CUSTOM_CALL_TARGET_PLACEHOLDER>>",
; CHECK:           backend_config="{
; CHECK-DAG:         \"alpha_real\":1
; CHECK-DAG:         \"alpha_imag\":0
; CHECK-DAG:         \"beta\":0
; CHECK-DAG:         \"dot_dimension_numbers\":{
; CHECK-DAG:           \"lhs_contracting_dimensions\":[\"2\"]
; CHECK-DAG:           \"rhs_contracting_dimensions\":[\"1\"]
; CHECK-DAG:           \"lhs_batch_dimensions\":[\"0\"]
; CHECK-DAG:           \"rhs_batch_dimensions\":[\"0\"]
; CHECK-DAG:         }
; CHECK-DAG:         \"precision_config\":{
; CHECK-DAG:           \"operand_precision\":[\"DEFAULT\",\"DEFAULT\"]
; CHECK-DAG:         }
; CHECK-DAG:         \"epilogue\":\"DEFAULT\"
; CHECK:           }"
; CHECK-NEXT:    ROOT [[OUT:%[^ ]+]] = f32[2,5,4]{2,1,0} bitcast([[GEMM]])
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

  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{1e-5, 1e-5}));
  MatchOptimizedHlo(hlo_text,
                    R"(
; CHECK-LABEL: ENTRY %AddDotsFunc (x: f32[5,2,3], y: f32[5,3,4]) -> f32[2,4,5] {
; CHECK-NEXT:    [[P0:%[^ ]+]] = f32[5,2,3]{2,1,0} parameter(0)
; CHECK-NEXT:    [[P1:%[^ ]+]] = f32[5,3,4]{2,1,0} parameter(1)
; CHECK-NEXT:    [[GEMM:%[^ ]+]] = f32[5,2,4]{2,1,0} custom-call([[P0]], [[P1]]),
; CHECK:           custom_call_target="<<CUBLAS_CUSTOM_CALL_TARGET_PLACEHOLDER>>",
; CHECK:           backend_config="{
; CHECK-DAG:         \"alpha_real\":1
; CHECK-DAG:         \"alpha_imag\":0
; CHECK-DAG:         \"beta\":0
; CHECK-DAG:         \"dot_dimension_numbers\":{
; CHECK-DAG:           \"lhs_contracting_dimensions\":[\"2\"]
; CHECK-DAG:           \"rhs_contracting_dimensions\":[\"1\"]
; CHECK-DAG:           \"lhs_batch_dimensions\":[\"0\"]
; CHECK-DAG:           \"rhs_batch_dimensions\":[\"0\"]
; CHECK-DAG:         }
; CHECK-DAG:         \"precision_config\":{
; CHECK-DAG:           \"operand_precision\":[\"DEFAULT\",\"DEFAULT\"]
; CHECK-DAG:         }
; CHECK-DAG:         \"epilogue\":\"DEFAULT\"
; CHECK:           }"
; CHECK-NEXT:    ROOT [[OUT:%[^ ]+]] = f32[2,4,5]{2,1,0} [[OP:[^ ]+]]([[GEMM]])
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
  dot_a = f32[2,2] dot(x, y), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  ROOT dot_a_multiplied = f32[2, 2] multiply(dot_a, k_broadcast)
}

)";

  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{1e-5, 1e-5}));
  MatchOptimizedHlo(hlo_text,
                    R"(
; CHECK-LABEL: ENTRY %AddDotsFunc (x: f32[2,2], y: f32[2,2]) -> f32[2,2] {
; CHECK-NEXT:    [[P0:%[^ ]+]] = f32[2,2]{1,0} parameter(0)
; CHECK-NEXT:    [[P1:%[^ ]+]] = f32[2,2]{1,0} parameter(1)
; CHECK-NEXT:    ROOT [[OUT:%[^ ]+]] = f32[2,2]{1,0} custom-call([[P0]], [[P1]]),
; CHECK:           custom_call_target="<<CUBLAS_CUSTOM_CALL_TARGET_PLACEHOLDER>>",
; CHECK:           backend_config="{
; CHECK-DAG:         \"alpha_real\":3
; CHECK-DAG:         \"alpha_imag\":0
; CHECK-DAG:         \"beta\":0
; CHECK-DAG:         \"dot_dimension_numbers\":{
; CHECK-DAG:           \"lhs_contracting_dimensions\":[\"1\"]
; CHECK-DAG:           \"rhs_contracting_dimensions\":[\"0\"]
; CHECK-DAG:           \"lhs_batch_dimensions\":[]
; CHECK-DAG:           \"rhs_batch_dimensions\":[]
; CHECK-DAG:         }
; CHECK-DAG:         \"precision_config\":{
; CHECK-DAG:           \"operand_precision\":[\"DEFAULT\",\"DEFAULT\"]
; CHECK-DAG:         }
; CHECK-DAG:         \"epilogue\":\"DEFAULT\"
; CHECK:           }"
)");
}

TEST_P(ParameterizedGemmRewriteTest, ComplexAlphaSimpleRewrite) {
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
; CHECK-LABEL: ENTRY %AddDotsFunc (x: c64[2,2], y: c64[2,2]) -> c64[2,2] {
; CHECK-NEXT:    [[P0:%[^ ]+]] = c64[2,2]{1,0} parameter(0)
; CHECK-NEXT:    [[P1:%[^ ]+]] = c64[2,2]{1,0} parameter(1)
; CHECK-NEXT:    ROOT [[OUT:%[^ ]+]] = c64[2,2]{1,0} custom-call([[P0]], [[P1]]),
; CHECK:           custom_call_target="<<CUBLAS_CUSTOM_CALL_TARGET_PLACEHOLDER>>",
; CHECK:           backend_config="{
; CHECK-DAG:         \"alpha_real\":3
; CHECK-DAG:         \"alpha_imag\":3
; CHECK-DAG:         \"beta\":0
; CHECK-DAG:         \"dot_dimension_numbers\":{
; CHECK-DAG:           \"lhs_contracting_dimensions\":[\"1\"]
; CHECK-DAG:           \"rhs_contracting_dimensions\":[\"0\"]
; CHECK-DAG:           \"lhs_batch_dimensions\":[]
; CHECK-DAG:           \"rhs_batch_dimensions\":[]
; CHECK-DAG:         }
; CHECK-DAG:         \"precision_config\":{
; CHECK-DAG:           \"operand_precision\":[\"DEFAULT\",\"DEFAULT\"]
; CHECK-DAG:         }
; CHECK-DAG:         \"epilogue\":\"DEFAULT\"
; CHECK:           }"
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
  dot_a = f32[2,2] dot(x, y), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  dot_a_multiplied = f32[2, 2] multiply(dot_a, k_broadcast)
  ROOT out = f32[2,2] add(dot_a_multiplied, dot_a)
}

)";

  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{1e-5, 1e-5}));
  MatchOptimizedHlo(hlo_text,
                    R"(
; CHECK:    {{[^ ]+}} = f32[2,2]{1,0} custom-call({{[^,]+}}, {{[^)]+}}),
; CHECK:           custom_call_target="<<CUBLAS_CUSTOM_CALL_TARGET_PLACEHOLDER>>",
; CHECK:           backend_config="{
; CHECK-DAG:         \"alpha_real\":1
; CHECK-DAG:         \"alpha_imag\":0
; CHECK-DAG:         \"beta\":0
; CHECK-DAG:         \"dot_dimension_numbers\":{
; CHECK-DAG:           \"lhs_contracting_dimensions\":[\"1\"]
; CHECK-DAG:           \"rhs_contracting_dimensions\":[\"0\"]
; CHECK-DAG:           \"lhs_batch_dimensions\":[]
; CHECK-DAG:           \"rhs_batch_dimensions\":[]
; CHECK-DAG:         }
; CHECK-DAG:         \"precision_config\":{
; CHECK-DAG:           \"operand_precision\":[\"DEFAULT\",\"DEFAULT\"]
; CHECK-DAG:         }
; CHECK-DAG:         \"epilogue\":\"DEFAULT\"
; CHECK:           }"
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
; CHECK-LABEL: ENTRY %AddDotsFunc (x: f32[2,2], y: f32[2,2]) -> f32[2,2] {
; CHECK-NEXT:    [[P0:%[^ ]+]] = f32[2,2]{1,0} parameter(0)
; CHECK-NEXT:    [[P1:%[^ ]+]] = f32[2,2]{1,0} parameter(1)
; CHECK-NEXT:    [[OUT:%[^ ]+]] = f32[2,2]{1,0} custom-call([[P0]], [[P1]]),
; CHECK:           custom_call_target="<<CUBLAS_CUSTOM_CALL_TARGET_PLACEHOLDER>>",
; CHECK:           backend_config="{
; CHECK-DAG:         \"alpha_real\":1
; CHECK-DAG:         \"alpha_imag\":0
; CHECK-DAG:         \"beta\":0
; CHECK-DAG:         \"dot_dimension_numbers\":{
; CHECK-DAG:           \"lhs_contracting_dimensions\":[\"1\"]
; CHECK-DAG:           \"rhs_contracting_dimensions\":[\"0\"]
; CHECK-DAG:           \"lhs_batch_dimensions\":[]
; CHECK-DAG:           \"rhs_batch_dimensions\":[]
; CHECK-DAG:         }
; CHECK-DAG:         \"precision_config\":{
; CHECK-DAG:           \"operand_precision\":[\"DEFAULT\",\"DEFAULT\"]
; CHECK-DAG:         }
; CHECK-DAG:         \"epilogue\":\"DEFAULT\"
; CHECK:           }"
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

  if (GetCudaComputeCapability().IsAtLeast(se::CudaComputeCapability::AMPERE)) {
    MatchOptimizedHlo(hlo_text,
                      R"(
; CHECK: bf16[16,8]{1,0} custom-call(bf16[16,8]{1,0} {{.*}}, bf16[8,8]{1,0} {{.*}}), custom_call_target="<<CUBLAS_CUSTOM_CALL_TARGET_PLACEHOLDER>>"
  )",
                      /*print_operand_shape=*/true);
  } else {
    MatchOptimizedHlo(hlo_text,
                      R"(
; CHECK: bf16[12,8]{1,0} custom-call(bf16[12,4]{1,0} [[P0:%[^ ]+]], bf16[4,8]{1,0} [[P1:%[^ ]+]]), custom_call_target="<<CUBLAS_CUSTOM_CALL_TARGET_PLACEHOLDER>>"
  )",
                      /*print_operand_shape=*/true);
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

  if (GetCudaComputeCapability().IsAtLeast(se::CudaComputeCapability::AMPERE)) {
    MatchOptimizedHlo(hlo_text,
                      R"(
    ; CHECK: bf16[3,8,8]{2,1,0} custom-call(bf16[3,8,8]{2,1,0} {{.*}}, bf16[3,8,8]{2,1,0} {{.*}}), custom_call_target="<<CUBLAS_CUSTOM_CALL_TARGET_PLACEHOLDER>>"
    )",
                      /*print_operand_shape=*/true);
  } else {
    MatchOptimizedHlo(hlo_text,
                      R"(
    ; CHECK: ROOT [[OUT:%[^ ]+]] = bf16[3,4,2]{2,1,0} custom-call(bf16[3,3,4]{2,1,0} [[A:%[^ ]+]], bf16[3,3,2]{2,1,0} [[B:%[^ ]+]]), custom_call_target="<<CUBLAS_CUSTOM_CALL_TARGET_PLACEHOLDER>>"
    )",
                      /*print_operand_shape=*/true);
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

  if (GetCudaComputeCapability().IsAtLeast(se::CudaComputeCapability::VOLTA)) {
    MatchOptimizedHlo(hlo_text,
                      R"(
; CHECK: s32[12,8]{1,0} custom-call(s8[12,4]{1,0} [[A:%[^ ]+]], s8[4,8]{0,1} [[B:%[^ ]+]]), custom_call_target="__cublas$gemm"
  )",
                      /*print_operand_shape=*/true);
  } else {
    MatchOptimizedHlo(hlo_text,
                      R"(
; CHECK: s32[12,8]{1,0} dot(s32[12,4]{1,0} [[A:%[^ ]+]], s32[4,8]{1,0} [[B:%[^ ]+]]), lhs_contracting_dims={1}, rhs_contracting_dims={0}

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

  if (GetCudaComputeCapability().IsAtLeast(se::CudaComputeCapability::VOLTA)) {
    MatchOptimizedHlo(hlo_text,
                      R"(
; CHECK: s32[12,8]{1,0} custom-call(s8[12,4]{1,0} [[A:%[^ ]+]], s8[4,8]{0,1} [[B:%[^ ]+]]),
; CHECK:           custom_call_target="__cublas$gemm",
; CHECK:           backend_config="{
; CHECK-DAG:       \"alpha_real\":1
; CHECK-DAG:       \"alpha_imag\":0
  )",
                      /*print_operand_shape=*/true);
  } else {
    MatchOptimizedHlo(hlo_text,
                      R"(
; CHECK: s32[12,8]{1,0} dot(s32[12,4]{1,0} [[A:%[^ ]+]], s32[4,8]{1,0} [[B:%[^ ]+]]), lhs_contracting_dims={1}, rhs_contracting_dims={0}

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

  if (GetCudaComputeCapability().IsAtLeast(se::CudaComputeCapability::VOLTA)) {
    MatchOptimizedHlo(hlo_text,
                      R"(
; CHECK: s32[12,8]{1,0} custom-call(s8[12,4]{1,0} [[A:%[^ ]+]], s8[4,8]{0,1} [[B:%[^ ]+]]),
; CHECK:           custom_call_target="__cublas$gemm",
; CHECK:           backend_config="{
; CHECK-DAG:       \"alpha_real\":1
; CHECK-DAG:       \"alpha_imag\":0
; CHECK-DAG:       \"beta\":0
  )",
                      /*print_operand_shape=*/true);
  } else {
    MatchOptimizedHlo(hlo_text,
                      R"(
; CHECK: s32[12,8]{1,0} dot(s32[12,4]{1,0} [[A:%[^ ]+]], s32[4,8]{1,0} [[B:%[^ ]+]]), lhs_contracting_dims={1}, rhs_contracting_dims={0}

  )",
                      /*print_operand_shape=*/true);
  }
}

TEST_P(ParameterizedGemmRewriteTest, Int8GemmNotMultipleOfFour) {
  const char* hlo_text = R"(
HloModule int8gemm

ENTRY int8gemm {
  %parameter.1 = s8[13,4]{1,0} parameter(0)
  %parameter.2 = s8[4,9]{1,0} parameter(1)
  ROOT %dot.9 = s32[13,9] dot(s8[13,4] %parameter.1, s8[4,9] %parameter.2), lhs_contracting_dims={1}, rhs_contracting_dims={0}
}
  )";
  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{1e-5, 1e-5}));

  if (GetCudaComputeCapability().IsAtLeast(se::CudaComputeCapability::VOLTA)) {
    MatchOptimizedHlo(hlo_text,
                      R"(
; CHECK: s32[16,12]{1,0} custom-call(s8[16,4]{1,0} [[A:%[^ ]+]], s8[4,12]{0,1} [[B:%[^ ]+]]), custom_call_target="__cublas$gemm"
  )",
                      /*print_operand_shape=*/true);
  } else {
    MatchOptimizedHlo(hlo_text,
                      R"(
; CHECK: s32[13,9]{1,0} dot(s32[13,4]{1,0} [[A:%[^ ]+]], s32[4,9]{1,0} [[B:%[^ ]+]]), lhs_contracting_dims={1}, rhs_contracting_dims={0}

  )",
                      /*print_operand_shape=*/true);
  }
}

TEST_P(ParameterizedGemmRewriteTest, GemmTypeCombinationCheck) {
  std::vector<std::tuple<absl::string_view, absl::string_view, bool>>
      type_combinations = {
          {"s8", "s32", true},    {"s8", "s8", true},   {"s32", "s32", true},
          {"bf16", "bf16", true}, {"f16", "f16", true}, {"f32", "f32", true},
          {"f64", "f64", true},   {"c64", "c64", true}, {"c128", "c128", true},
      };

  if (GetCudaComputeCapability().IsAtLeast(se::CudaComputeCapability::VOLTA)) {
    // For compute capabilities before volta, we always do upcasting, so it
    // would be impossible for this test to fail. That is why we only add these
    // cases when the compute capabilit is at least Volta.
    std::vector<std::tuple<absl::string_view, absl::string_view, bool>>
        more_type_combinations = {
            {"s8", "bf16", false},  {"s8", "f16", false},
            {"s8", "f32", false},   {"s8", "f64", false},
            {"s8", "c64", false},   {"s8", "c128", false},

            {"s32", "f32", false},  {"s32", "f64", false},
            {"s32", "c64", false},  {"s32", "c128", false},

            {"f16", "bf16", false}, {"f16", "f32", false},
            {"f16", "f64", false},  {"f16", "c64", false},
            {"f16", "c128", false},

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
  GemmRewriter pass(GetCudaComputeCapability());
  TF_ASSERT_OK_AND_ASSIGN(bool changed, this->RunHloPass(&pass, module.get()));
  EXPECT_TRUE(changed);

  // This is a type combination which is not supported by cublasLt, expect
  // GemmRewriter to choose legacy cublas.
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::CustomCall({"__cublas$gemm"})));
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
  GemmRewriter pass(GetCudaComputeCapability());
  TF_ASSERT_OK_AND_ASSIGN(bool changed, this->RunHloPass(&pass, module.get()));
  EXPECT_TRUE(changed);

  // This is a type combination which is not supported by cublasLt, expect
  // GemmRewriter to choose legacy cublas.
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::CustomCall({"__cublas$gemm"})));
}

TEST_P(ParameterizedGemmRewriteTest, UpcastingF16ToF32) {
  const char* hlo_text = R"(
HloModule test

ENTRY test {
  Arg_0.1 = f16[4,3]{1,0} parameter(0)
  Arg_1.2 = f16[3,6]{1,0} parameter(1)
  ROOT dot.3 = f32[4,6]{1,0} dot(Arg_0.1, Arg_1.2), lhs_contracting_dims={1}, rhs_contracting_dims={0}
}
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_text));
  GemmRewriter pass(GetCudaComputeCapability());
  TF_ASSERT_OK_AND_ASSIGN(bool changed, this->RunHloPass(&pass, module.get()));
  EXPECT_TRUE(changed);

  EXPECT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::CustomCall({CustomCallTarget()})));
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
  GemmRewriter pass(GetCudaComputeCapability());
  TF_ASSERT_OK_AND_ASSIGN(bool changed, this->RunHloPass(&pass, module.get()));
  EXPECT_TRUE(changed);

  // This is a type combination which is not supported by cublasLt, expect
  // GemmRewriter to choose legacy cublas.
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::CustomCall({"__cublas$gemm"})));
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
  GemmRewriter pass(GetCudaComputeCapability());
  TF_ASSERT_OK_AND_ASSIGN(bool changed, this->RunHloPass(&pass, module.get()));
  EXPECT_TRUE(changed);

  // This is a type combination which is not supported by cublasLt, expect
  // GemmRewriter to choose legacy cublas.
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::CustomCall({"__cublas$gemm"})));
}

INSTANTIATE_TEST_SUITE_P(CublasTestsBothLegacyAndLt,
                         ParameterizedGemmRewriteTest, ::testing::Bool());

// A test fixture class for tests which are specific to legacy cublas
class LegacyCublasGemmRewriteTest : public GemmRewriteTest {
 public:
  DebugOptions GetDebugOptionsForTest() override {
    DebugOptions debug_options = GemmRewriteTest::GetDebugOptionsForTest();
    debug_options.set_xla_gpu_enable_cublaslt(false);
    return debug_options;
  }
};

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
  dot_a = f32[2,2] dot(x, y), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  dot_a_multiplied = f32[2, 2] multiply(dot_a, k_broadcast)
  ROOT out = f32[2,2] add(dot_a_multiplied, bias)
}

)";

  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{1e-5, 1e-5}));
  MatchOptimizedHlo(hlo_text,
                    R"(
; CHECK-LABEL: ENTRY %AddDotsFunc (x: f32[2,2], y: f32[2,2], param_2: f32[2,2]) -> f32[2,2] {
; CHECK-DAG:     [[X:%[^ ]+]] = f32[2,2]{1,0} parameter(0)
; CHECK-DAG:     [[Y:%[^ ]+]] = f32[2,2]{1,0} parameter(1)
; CHECK:         ROOT [[OUT:%[^ ]+]] = f32[2,2]{1,0} custom-call([[X]], [[Y]], {{[^,)]+}}),
; CHECK:           custom_call_target="__cublas$gemm",
; CHECK:           output_to_operand_aliasing={{{{}: \(2, {}\)}}},
; CHECK:           backend_config="{
; CHECK-DAG:         \"alpha_real\":3
; CHECK-DAG:         \"alpha_imag\":0
; CHECK-DAG:         \"beta\":1
; CHECK-DAG:         \"dot_dimension_numbers\":{
; CHECK-DAG:           \"lhs_contracting_dimensions\":[\"1\"]
; CHECK-DAG:           \"rhs_contracting_dimensions\":[\"0\"]
; CHECK-DAG:           \"lhs_batch_dimensions\":[]
; CHECK-DAG:           \"rhs_batch_dimensions\":[]
; CHECK-DAG:         }
; CHECK-DAG:         \"precision_config\":{
; CHECK-DAG:           \"operand_precision\":[\"DEFAULT\",\"DEFAULT\"]
; CHECK-DAG:         }
; CHECK-DAG:         \"epilogue\":\"DEFAULT\"
; CHECK:           }"
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
  dot_a = f32[2,2] dot(x, y), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  dot_a_multiplied = f32[2, 2] multiply(dot_a, k_broadcast)
  biased_out = f32[2,2] add(dot_a_multiplied, bias)
  ROOT out = f32[2,2] add(biased_out, bias)
}
)";

  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{1e-5, 1e-5}));
  MatchOptimizedHlo(hlo_text,
                    R"(
; CHECK-LABEL: ENTRY %AddDotsFunc (x: f32[2,2], y: f32[2,2], bias: f32[2,2]) -> f32[2,2] {
; CHECK-DAG:     [[P0:%[^ ]+]] = f32[2,2]{1,0} parameter(0)
; CHECK-DAG:     [[P1:%[^ ]+]] = f32[2,2]{1,0} parameter(1)
; CHECK-NEXT:    [[GEMM:%[^ ]+]] = f32[2,2]{1,0} custom-call([[P0]], [[P1]]),
; CHECK:           custom_call_target="__cublas$gemm",
; CHECK:           backend_config="{
; CHECK-DAG:         \"alpha_real\":3
; CHECK-DAG:         \"alpha_imag\":0
; CHECK-DAG:         \"beta\":0
; CHECK-DAG:         \"dot_dimension_numbers\":{
; CHECK-DAG:           \"lhs_contracting_dimensions\":[\"1\"]
; CHECK-DAG:           \"rhs_contracting_dimensions\":[\"0\"]
; CHECK-DAG:           \"lhs_batch_dimensions\":[]
; CHECK-DAG:           \"rhs_batch_dimensions\":[]
; CHECK-DAG:         }
; CHECK-DAG:         \"precision_config\":{
; CHECK-DAG:           \"operand_precision\":[\"DEFAULT\",\"DEFAULT\"]
; CHECK-DAG:         }
; CHECK-DAG:         \"epilogue\":\"DEFAULT\"
; CHECK:           }"
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
; CHECK-LABEL: ENTRY %AddDotsFunc (x: f32[2,2], y: f32[2,2], bias: f32[2,2]) -> f32[2,2] {
; CHECK-DAG:     [[P0:%[^ ]+]] = f32[2,2]{1,0} parameter(0)
; CHECK-DAG:     [[P1:%[^ ]+]] = f32[2,2]{1,0} parameter(1)
; CHECK-NEXT:    [[GEMM:%[^ ]+]] = f32[2,2]{1,0} custom-call([[P0]], [[P1]]),
; CHECK:           custom_call_target="__cublas$gemm",
; CHECK:           backend_config="{
; CHECK-DAG:         \"alpha_real\":1
; CHECK-DAG:         \"alpha_imag\":0
; CHECK-DAG:         \"beta\":0
; CHECK-DAG:         \"dot_dimension_numbers\":{
; CHECK-DAG:           \"lhs_contracting_dimensions\":[\"1\"]
; CHECK-DAG:           \"rhs_contracting_dimensions\":[\"0\"]
; CHECK-DAG:           \"lhs_batch_dimensions\":[]
; CHECK-DAG:           \"rhs_batch_dimensions\":[]
; CHECK-DAG:         }
; CHECK-DAG:         \"precision_config\":{
; CHECK-DAG:           \"operand_precision\":[\"DEFAULT\",\"DEFAULT\"]
; CHECK-DAG:         }
; CHECK-DAG:         \"epilogue\":\"DEFAULT\"
; CHECK:           }"
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
; CHECK-LABEL: ENTRY %AddDotsFunc (x: f32[2,2], y: f32[2,2], param_2: (f32[2,2], f32[3,3])) -> f32[2,2] {
; CHECK-DAG:     [[P0:%[^ ]+]] = f32[2,2]{1,0} parameter(0)
; CHECK-DAG:     [[P1:%[^ ]+]] = f32[2,2]{1,0} parameter(1)
; CHECK-DAG:     [[P2:%[^ ]+]] = (f32[2,2]{1,0}, f32[3,3]{1,0}) parameter(2)
; CHECK-DAG:     [[BIAS:%[^ ]+]] = f32[2,2]{1,0} get-tuple-element([[P2]]), index=0
; CHECK-DAG:     [[BIAS_COPY:%[^ ]+]] = f32[2,2]{1,0} copy([[BIAS]])
; CHECK-NEXT:    [[GEMM:%[^ ]+]] = f32[2,2]{1,0} custom-call([[P0]], [[P1]], [[BIAS_COPY]]),
; CHECK:           custom_call_target="__cublas$gemm",
; CHECK:           output_to_operand_aliasing={{{{}: \(2, {}\)}}},
; CHECK:           backend_config="{
; CHECK-DAG:         \"alpha_real\":1
; CHECK-DAG:         \"alpha_imag\":0
; CHECK-DAG:         \"beta\":1
; CHECK-DAG:         \"dot_dimension_numbers\":{
; CHECK-DAG:           \"lhs_contracting_dimensions\":[\"1\"]
; CHECK-DAG:           \"rhs_contracting_dimensions\":[\"0\"]
; CHECK-DAG:           \"lhs_batch_dimensions\":[]
; CHECK-DAG:           \"rhs_batch_dimensions\":[]
; CHECK-DAG:         }
; CHECK-DAG:         \"precision_config\":{
; CHECK-DAG:           \"operand_precision\":[\"DEFAULT\",\"DEFAULT\"]
; CHECK-DAG:         }
; CHECK-DAG:         \"epilogue\":\"DEFAULT\"
; CHECK:           }"
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
  dot_a = f32[2,2] dot(x, y), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  dot_a_multiplied = f32[2, 2] multiply(dot_a, k_broadcast)
  ROOT out = f32[2,2] add(dot_a_multiplied, bias)
}

)";

  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{1e-5, 1e-5}));
  MatchOptimizedHlo(hlo_text,
                    R"(
; CHECK-LABEL: ENTRY %AddDotsFunc (x: f32[2,2], y: f32[2,2], bias: f32[2,2]) -> f32[2,2] {
; CHECK-DAG:     [[X:%[^ ]+]] = f32[2,2]{1,0} parameter(0)
; CHECK-DAG:     [[Y:%[^ ]+]] = f32[2,2]{1,0} parameter(1)
; CHECK-DAG:     [[BIAS:%[^ ]+]] = f32[2,2]{1,0} parameter(2)
; CHECK:         ROOT [[OUT:%[^ ]+]] = f32[2,2]{1,0} custom-call([[X]], [[Y]], [[BIAS]]),
; CHECK:           custom_call_target="__cublas$gemm",
; CHECK:           output_to_operand_aliasing={{{{}: \(2, {}\)}}},
; CHECK:           backend_config="{
; CHECK-DAG:         \"alpha_real\":3
; CHECK-DAG:         \"alpha_imag\":0
; CHECK-DAG:         \"beta\":1
; CHECK-DAG:         \"dot_dimension_numbers\":{
; CHECK-DAG:           \"lhs_contracting_dimensions\":[\"1\"]
; CHECK-DAG:           \"rhs_contracting_dimensions\":[\"0\"]
; CHECK-DAG:           \"lhs_batch_dimensions\":[]
; CHECK-DAG:           \"rhs_batch_dimensions\":[]
; CHECK-DAG:         }
; CHECK-DAG:         \"precision_config\":{
; CHECK-DAG:           \"operand_precision\":[\"DEFAULT\",\"DEFAULT\"]
; CHECK-DAG:         }
; CHECK-DAG:         \"epilogue\":\"DEFAULT\"
; CHECK:           }"
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
; CHECK-LABEL: ENTRY %AddDotsFunc (x: f32[1024,1024], y: f32[1024,1024], bias: f32[1024,1024]) -> f32[1024,1024] {
; CHECK-DAG:     [[P0:%[^ ]+]] = f32[1024,1024]{1,0} parameter(0)
; CHECK-DAG:     [[P1:%[^ ]+]] = f32[1024,1024]{1,0} parameter(1)
; CHECK-NEXT:    [[GEMM:%[^ ]+]] = f32[1024,1024]{1,0} custom-call([[P0]], [[P1]]),
; CHECK:           custom_call_target="__cublas$gemm",
; CHECK:           backend_config="{
; CHECK-DAG:         \"alpha_real\":1
; CHECK-DAG:         \"alpha_imag\":0
; CHECK-DAG:         \"beta\":0
; CHECK-DAG:         \"dot_dimension_numbers\":{
; CHECK-DAG:           \"lhs_contracting_dimensions\":[\"1\"]
; CHECK-DAG:           \"rhs_contracting_dimensions\":[\"0\"]
; CHECK-DAG:           \"lhs_batch_dimensions\":[]
; CHECK-DAG:           \"rhs_batch_dimensions\":[]
; CHECK-DAG:         }
; CHECK-DAG:         \"precision_config\":{
; CHECK-DAG:           \"operand_precision\":[\"DEFAULT\",\"DEFAULT\"]
; CHECK-DAG:         }
; CHECK-DAG:         \"epilogue\":\"DEFAULT\"
; CHECK:           }"
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
  MatchOptimizedHlo(hlo_text,
                    R"(
; CHECK-LABEL: ENTRY %BF16GemmWithBias (x: bf16[8,8], y: bf16[8,8], param_2: bf16[8,8]) -> bf16[8,8] {
; CHECK-DAG:    [[X:%[^ ]+]] = bf16[8,8]{1,0} parameter(0)
; CHECK-DAG:    [[Y:%[^ ]+]] = bf16[8,8]{1,0} parameter(1)
; CHECK:        ROOT [[GEMM:%[^ ]+]] = bf16[8,8]{1,0} custom-call([[X]], [[Y]], {{[^,)]+}}),
; CHECK:           custom_call_target="__cublas$gemm",
; CHECK:           output_to_operand_aliasing={{{{}: \(2, {}\)}}},
; CHECK:           backend_config="{
; CHECK-DAG:         \"alpha_real\":1
; CHECK-DAG:         \"alpha_imag\":0
; CHECK-DAG:         \"beta\":1
; CHECK-DAG:         \"dot_dimension_numbers\":{
; CHECK-DAG:           \"lhs_contracting_dimensions\":[\"1\"]
; CHECK-DAG:           \"rhs_contracting_dimensions\":[\"0\"]
; CHECK-DAG:           \"lhs_batch_dimensions\":[]
; CHECK-DAG:           \"rhs_batch_dimensions\":[]
; CHECK-DAG:         }
; CHECK-DAG:         \"precision_config\":{
; CHECK-DAG:           \"operand_precision\":[\"DEFAULT\",\"DEFAULT\"]
; CHECK-DAG:         }
; CHECK-DAG:         \"epilogue\":\"DEFAULT\"
; CHECK:           }"
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
; CHECK-LABEL: ENTRY %test (x: f32[2,3], y: f32[3,4], param_2: f32[2,4]) -> f32[2,4] {
; CHECK-DAG:     [[P0:%[^ ]+]] = f32[2,3]{1,0} parameter(0)
; CHECK-DAG:     [[P1:%[^ ]+]] = f32[3,4]{1,0} parameter(1)
; CHECK:         ROOT [[GEMM:%[^ ]+]] = f32[2,4]{1,0} custom-call([[P0]], [[P1]], {{[^,)]+}}),
; CHECK:           custom_call_target="__cublas$gemm",
; CHECK:           output_to_operand_aliasing={{{{}: \(2, {}\)}}},
; CHECK:           backend_config="{
; CHECK-DAG:         \"alpha_real\":1
; CHECK-DAG:         \"alpha_imag\":0
; CHECK-DAG:         \"beta\":1
; CHECK-DAG:         \"dot_dimension_numbers\":{
; CHECK-DAG:           \"lhs_contracting_dimensions\":[\"1\"]
; CHECK-DAG:           \"rhs_contracting_dimensions\":[\"0\"]
; CHECK-DAG:           \"lhs_batch_dimensions\":[]
; CHECK-DAG:           \"rhs_batch_dimensions\":[]
; CHECK-DAG:         }
; CHECK-DAG:         \"precision_config\":{
; CHECK-DAG:           \"operand_precision\":[\"DEFAULT\",\"DEFAULT\"]
; CHECK-DAG:         }
; CHECK-DAG:         \"epilogue\":\"DEFAULT\"
; CHECK:           }"
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
; CHECK-LABEL: ENTRY %test (w: f32[2,3], x: f32[3,4], y: f32[2,3], z: f32[3,4]) -> f32[2,4] {
; CHECK-DAG:     [[P0:%[^ ]+]] = f32[2,3]{1,0} parameter(0)
; CHECK-DAG:     [[P1:%[^ ]+]] = f32[3,4]{1,0} parameter(1)
; CHECK-DAG:     [[P2:%[^ ]+]] = f32[2,3]{1,0} parameter(2)
; CHECK-DAG:     [[P3:%[^ ]+]] = f32[3,4]{1,0} parameter(3)
; CHECK-NEXT:    [[FIRST_GEMM:%[^ ]+]] = f32[2,4]{1,0} custom-call([[P0]], [[P1]]),
; CHECK:           custom_call_target="__cublas$gemm",
; CHECK:           backend_config="{
; CHECK-DAG:         \"alpha_real\":1
; CHECK-DAG:         \"alpha_imag\":0
; CHECK-DAG:         \"beta\":0
; CHECK-DAG:         \"dot_dimension_numbers\":{
; CHECK-DAG:           \"lhs_contracting_dimensions\":[\"1\"]
; CHECK-DAG:           \"rhs_contracting_dimensions\":[\"0\"]
; CHECK-DAG:           \"lhs_batch_dimensions\":[]
; CHECK-DAG:           \"rhs_batch_dimensions\":[]
; CHECK-DAG:         }
; CHECK-DAG:         \"precision_config\":{
; CHECK-DAG:           \"operand_precision\":[\"DEFAULT\",\"DEFAULT\"]
; CHECK-DAG:         }
; CHECK-DAG:         \"epilogue\":\"DEFAULT\"
; CHECK:           }"
; CHECK-NEXT:    ROOT [[SECOND_GEMM:%[^ ]+]] = f32[2,4]{1,0} custom-call([[P2]], [[P3]], [[FIRST_GEMM]]),
; CHECK:           custom_call_target="__cublas$gemm",
; CHECK:           output_to_operand_aliasing={{{{}: \(2, {}\)}}},
; CHECK:           backend_config="{
; CHECK-DAG:         \"alpha_real\":1
; CHECK-DAG:         \"alpha_imag\":0
; CHECK-DAG:         \"beta\":1
; CHECK-DAG:         \"dot_dimension_numbers\":{
; CHECK-DAG:           \"lhs_contracting_dimensions\":[\"1\"]
; CHECK-DAG:           \"rhs_contracting_dimensions\":[\"0\"]
; CHECK-DAG:           \"lhs_batch_dimensions\":[]
; CHECK-DAG:           \"rhs_batch_dimensions\":[]
; CHECK-DAG:         }
; CHECK-DAG:         \"precision_config\":{
; CHECK-DAG:           \"operand_precision\":[\"DEFAULT\",\"DEFAULT\"]
; CHECK-DAG:         }
; CHECK-DAG:         \"epilogue\":\"DEFAULT\"
; CHECK:           }"
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
  GemmRewriter pass(GetCudaComputeCapability());
  TF_ASSERT_OK_AND_ASSIGN(bool changed, this->RunHloPass(&pass, module.get()));
  EXPECT_TRUE(changed);

  EXPECT_THAT(
      module->entry_computation()->root_instruction(),
      GmockMatch(
          m::Bitcast(
              m::CustomCall({"__cublas$gemm"}, m::Parameter(0), m::Parameter(1),
                            m::Bitcast(m::Parameter(2)).WithShape(F32, {2, 2})))
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
  GemmRewriter pass(GetCudaComputeCapability());
  TF_ASSERT_OK_AND_ASSIGN(bool changed, this->RunHloPass(&pass, module.get()));
  SCOPED_TRACE(module->ToString());
  EXPECT_TRUE(changed);

  EXPECT_THAT(
      module->entry_computation()->root_instruction(),
      GmockMatch(m::Tuple(
          m::CustomCall(m::Parameter(0), m::Parameter(1),
                        m::Negate(m::Parameter(2))),
          m::CustomCall(m::Parameter(0), m::Parameter(1), m::Constant()),
          m::CustomCall(m::Parameter(0), m::Parameter(1), m::Constant()),
          m::CustomCall(m::Parameter(0), m::Parameter(1), m::Constant()))));
}

// A test fixture class for tests which are specific to cublasLt
class CublasLtGemmRewriteTest : public GemmRewriteTest {
 public:
  DebugOptions GetDebugOptionsForTest() override {
    DebugOptions debug_options = GemmRewriteTest::GetDebugOptionsForTest();
    debug_options.set_xla_gpu_enable_cublaslt(true);
    return debug_options;
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
  dot_a = f32[2,2] dot(x, y), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  dot_a_multiplied = f32[2, 2] multiply(dot_a, k_broadcast)
  ROOT out = f32[2,2] add(dot_a_multiplied, bias)
}

)";

  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{1e-5, 1e-5}));
  MatchOptimizedHlo(hlo_text,
                    R"(
; CHECK-LABEL: ENTRY %AddDotsFunc (x: f32[2,2], y: f32[2,2], bias: f32[2,2]) -> f32[2,2] {
; CHECK-DAG:     [[X:%[^ ]+]] = f32[2,2]{1,0} parameter(0)
; CHECK-DAG:     [[Y:%[^ ]+]] = f32[2,2]{1,0} parameter(1)
; CHECK-DAG:     [[BIAS:%[^ ]+]] = f32[2,2]{1,0} parameter(2)
; CHECK-NEXT:    ROOT [[OUT:%[^ ]+]] = f32[2,2]{1,0} custom-call([[X]], [[Y]], [[BIAS]]),
; CHECK:           custom_call_target="__cublas$lt$matmul",
; CHECK:           backend_config="{
; CHECK-DAG:         \"alpha_real\":3
; CHECK-DAG:         \"alpha_imag\":0
; CHECK-DAG:         \"beta\":1
; CHECK-DAG:         \"dot_dimension_numbers\":{
; CHECK-DAG:           \"lhs_contracting_dimensions\":[\"1\"]
; CHECK-DAG:           \"rhs_contracting_dimensions\":[\"0\"]
; CHECK-DAG:           \"lhs_batch_dimensions\":[]
; CHECK-DAG:           \"rhs_batch_dimensions\":[]
; CHECK-DAG:         }
; CHECK-DAG:         \"precision_config\":{
; CHECK-DAG:           \"operand_precision\":[\"DEFAULT\",\"DEFAULT\"]
; CHECK-DAG:         }
; CHECK-DAG:         \"epilogue\":\"DEFAULT\"
; CHECK:           }"
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
  dot_a = f32[2,2] dot(x, y), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  dot_a_multiplied = f32[2, 2] multiply(dot_a, k_broadcast)
  biased_out = f32[2,2] add(dot_a_multiplied, bias)
  ROOT out = f32[2,2] add(biased_out, bias)
}
)";

  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{1e-5, 1e-5}));
  MatchOptimizedHlo(hlo_text,
                    R"(
; CHECK-LABEL: ENTRY %AddDotsFunc (x: f32[2,2], y: f32[2,2], bias: f32[2,2]) -> f32[2,2] {
; CHECK-DAG:     [[P0:%[^ ]+]] = f32[2,2]{1,0} parameter(0)
; CHECK-DAG:     [[P1:%[^ ]+]] = f32[2,2]{1,0} parameter(1)
; CHECK-DAG:     [[BIAS:%[^ ]+]] = f32[2,2]{1,0} parameter(2)
; CHECK-NEXT:    [[GEMM:%[^ ]+]] = f32[2,2]{1,0} custom-call([[P0]], [[P1]], [[BIAS]]),
; CHECK:           custom_call_target="__cublas$lt$matmul",
; CHECK-NOT:       output_to_operand_aliasing
; CHECK:           backend_config="{
; CHECK-DAG:         \"alpha_real\":3
; CHECK-DAG:         \"alpha_imag\":0
; CHECK-DAG:         \"beta\":1
; CHECK-DAG:         \"dot_dimension_numbers\":{
; CHECK-DAG:           \"lhs_contracting_dimensions\":[\"1\"]
; CHECK-DAG:           \"rhs_contracting_dimensions\":[\"0\"]
; CHECK-DAG:           \"lhs_batch_dimensions\":[]
; CHECK-DAG:           \"rhs_batch_dimensions\":[]
; CHECK-DAG:         }
; CHECK-DAG:         \"precision_config\":{
; CHECK-DAG:           \"operand_precision\":[\"DEFAULT\",\"DEFAULT\"]
; CHECK-DAG:         }
; CHECK-DAG:         \"epilogue\":\"DEFAULT\"
; CHECK:           }"
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
; CHECK-LABEL: ENTRY %AddDotsFunc (x: f32[1024,1024], y: f32[1024,1024], bias: f32[1024,1024]) -> f32[1024,1024] {
; CHECK-DAG:     [[P0:%[^ ]+]] = f32[1024,1024]{1,0} parameter(0)
; CHECK-DAG:     [[P1:%[^ ]+]] = f32[1024,1024]{1,0} parameter(1)
; CHECK-DAG:     [[BIAS:%[^ ]+]] = f32[1024,1024]{1,0} parameter(2)
; CHECK-NEXT:    [[GEMM:%[^ ]+]] = f32[1024,1024]{1,0} custom-call([[P0]], [[P1]], [[BIAS]]),
; CHECK:           custom_call_target="__cublas$lt$matmul",
; CHECK:           backend_config="{
; CHECK-DAG:         \"alpha_real\":1
; CHECK-DAG:         \"alpha_imag\":0
; CHECK-DAG:         \"beta\":1
; CHECK-DAG:         \"dot_dimension_numbers\":{
; CHECK-DAG:           \"lhs_contracting_dimensions\":[\"1\"]
; CHECK-DAG:           \"rhs_contracting_dimensions\":[\"0\"]
; CHECK-DAG:           \"lhs_batch_dimensions\":[]
; CHECK-DAG:           \"rhs_batch_dimensions\":[]
; CHECK-DAG:         }
; CHECK-DAG:         \"precision_config\":{
; CHECK-DAG:           \"operand_precision\":[\"DEFAULT\",\"DEFAULT\"]
; CHECK-DAG:         }
; CHECK-DAG:         \"epilogue\":\"DEFAULT\"
; CHECK:           }"
; CHECK-NEXT:    ROOT [[OUT:%[^ ]+]] = f32[1024,1024]{1,0} add([[GEMM]], [[BIAS]])
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
  MatchOptimizedHlo(hlo_text,
                    R"(
; CHECK-LABEL: ENTRY %BF16GemmWithBias (x: bf16[8,8], y: bf16[8,8], bias: bf16[8,8]) -> bf16[8,8] {
; CHECK-DAG:    [[X:%[^ ]+]] = bf16[8,8]{1,0} parameter(0)
; CHECK-DAG:    [[Y:%[^ ]+]] = bf16[8,8]{1,0} parameter(1)
; CHECK-DAG:    [[BIAS:%[^ ]+]] = bf16[8,8]{1,0} parameter(2)
; CHECK-NEXT:   ROOT [[GEMM:%[^ ]+]] = bf16[8,8]{1,0} custom-call([[X]], [[Y]], [[BIAS]]),
; CHECK:           custom_call_target="__cublas$lt$matmul",
; CHECK:           backend_config="{
; CHECK-DAG:         \"alpha_real\":1
; CHECK-DAG:         \"alpha_imag\":0
; CHECK-DAG:         \"beta\":1
; CHECK-DAG:         \"dot_dimension_numbers\":{
; CHECK-DAG:           \"lhs_contracting_dimensions\":[\"1\"]
; CHECK-DAG:           \"rhs_contracting_dimensions\":[\"0\"]
; CHECK-DAG:           \"lhs_batch_dimensions\":[]
; CHECK-DAG:           \"rhs_batch_dimensions\":[]
; CHECK-DAG:         }
; CHECK-DAG:         \"precision_config\":{
; CHECK-DAG:           \"operand_precision\":[\"DEFAULT\",\"DEFAULT\"]
; CHECK-DAG:         }
; CHECK-DAG:         \"epilogue\":\"DEFAULT\"
; CHECK:           }"
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
; CHECK-LABEL: ENTRY %test (x: f32[2,3], y: f32[3,4], z: f32[2,4]) -> f32[2,4] {
; CHECK-NEXT:    [[P0:%[^ ]+]] = f32[2,3]{1,0} parameter(0)
; CHECK-NEXT:    [[P1:%[^ ]+]] = f32[3,4]{1,0} parameter(1)
; CHECK-NEXT:    [[P2:%[^ ]+]] = f32[2,4]{1,0} parameter(2)
; CHECK-NEXT:    ROOT [[GEMM:%[^ ]+]] = f32[2,4]{1,0} custom-call([[P0]], [[P1]], [[P2]]),
; CHECK:           custom_call_target="__cublas$lt$matmul",
; CHECK:           backend_config="{
; CHECK-DAG:         \"alpha_real\":1
; CHECK-DAG:         \"alpha_imag\":0
; CHECK-DAG:         \"beta\":1
; CHECK-DAG:         \"dot_dimension_numbers\":{
; CHECK-DAG:           \"lhs_contracting_dimensions\":[\"1\"]
; CHECK-DAG:           \"rhs_contracting_dimensions\":[\"0\"]
; CHECK-DAG:           \"lhs_batch_dimensions\":[]
; CHECK-DAG:           \"rhs_batch_dimensions\":[]
; CHECK-DAG:         }
; CHECK-DAG:         \"precision_config\":{
; CHECK-DAG:           \"operand_precision\":[\"DEFAULT\",\"DEFAULT\"]
; CHECK-DAG:         }
; CHECK-DAG:         \"epilogue\":\"DEFAULT\"
; CHECK:           }"
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
; CHECK-LABEL: ENTRY %test (w: f32[2,3], x: f32[3,4], y: f32[2,3], z: f32[3,4]) -> f32[2,4] {
; CHECK-DAG:     [[P0:%[^ ]+]] = f32[2,3]{1,0} parameter(0)
; CHECK-DAG:     [[P1:%[^ ]+]] = f32[3,4]{1,0} parameter(1)
; CHECK-DAG:     [[P2:%[^ ]+]] = f32[2,3]{1,0} parameter(2)
; CHECK-DAG:     [[P3:%[^ ]+]] = f32[3,4]{1,0} parameter(3)
; CHECK-NEXT:    [[FIRST_GEMM:%[^ ]+]] = f32[2,4]{1,0} custom-call([[P0]], [[P1]]),
; CHECK:           custom_call_target="__cublas$lt$matmul",
; CHECK:           backend_config="{
; CHECK-DAG:         \"alpha_real\":1
; CHECK-DAG:         \"alpha_imag\":0
; CHECK-DAG:         \"beta\":0
; CHECK-DAG:         \"dot_dimension_numbers\":{
; CHECK-DAG:           \"lhs_contracting_dimensions\":[\"1\"]
; CHECK-DAG:           \"rhs_contracting_dimensions\":[\"0\"]
; CHECK-DAG:           \"lhs_batch_dimensions\":[]
; CHECK-DAG:           \"rhs_batch_dimensions\":[]
; CHECK-DAG:         }
; CHECK-DAG:         \"precision_config\":{
; CHECK-DAG:           \"operand_precision\":[\"DEFAULT\",\"DEFAULT\"]
; CHECK-DAG:         }
; CHECK-DAG:         \"epilogue\":\"DEFAULT\"
; CHECK:           }"
; CHECK-NEXT:    ROOT [[SECOND_GEMM:%[^ ]+]] = f32[2,4]{1,0} custom-call([[P2]], [[P3]], [[FIRST_GEMM]]),
; CHECK:           custom_call_target="__cublas$lt$matmul",
; CHECK:           output_to_operand_aliasing={{{{}: \(2, {}\)}}},
; CHECK:           backend_config="{
; CHECK-DAG:         \"alpha_real\":1
; CHECK-DAG:         \"alpha_imag\":0
; CHECK-DAG:         \"beta\":1
; CHECK-DAG:         \"dot_dimension_numbers\":{
; CHECK-DAG:           \"lhs_contracting_dimensions\":[\"1\"]
; CHECK-DAG:           \"rhs_contracting_dimensions\":[\"0\"]
; CHECK-DAG:           \"lhs_batch_dimensions\":[]
; CHECK-DAG:           \"rhs_batch_dimensions\":[]
; CHECK-DAG:         }
; CHECK-DAG:         \"precision_config\":{
; CHECK-DAG:           \"operand_precision\":[\"DEFAULT\",\"DEFAULT\"]
; CHECK-DAG:         }
; CHECK-DAG:         \"epilogue\":\"DEFAULT\"
; CHECK:           }"
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
; CHECK-LABEL: ENTRY %test (x: f32[2,3], y: f32[3,4], z: f32[4]) -> f32[2,4] {
; CHECK-NEXT:    [[P0:%[^ ]+]] = f32[2,3]{1,0} parameter(0)
; CHECK-NEXT:    [[P1:%[^ ]+]] = f32[3,4]{1,0} parameter(1)
; CHECK-NEXT:    [[P2:%[^ ]+]] = f32[4]{0} parameter(2)
; CHECK-NEXT:    ROOT [[OUT:%[^ ]+]] = f32[2,4]{1,0} custom-call([[P0]], [[P1]], [[P2]]),
; CHECK:           custom_call_target="__cublas$lt$matmul",
; CHECK:           backend_config="{
; CHECK-DAG:         \"alpha_real\":1
; CHECK-DAG:         \"alpha_imag\":0
; CHECK-DAG:         \"beta\":0
; CHECK-DAG:         \"dot_dimension_numbers\":{
; CHECK-DAG:           \"lhs_contracting_dimensions\":[\"1\"]
; CHECK-DAG:           \"rhs_contracting_dimensions\":[\"0\"]
; CHECK-DAG:           \"lhs_batch_dimensions\":[]
; CHECK-DAG:           \"rhs_batch_dimensions\":[]
; CHECK-DAG:         }
; CHECK-DAG:         \"precision_config\":{
; CHECK-DAG:           \"operand_precision\":[\"DEFAULT\",\"DEFAULT\"]
; CHECK-DAG:         }
; CHECK-DAG:         \"epilogue\":\"BIAS\"
; CHECK:           }"
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
  dot_a = f32[4,4] dot(x, y), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  z_bcast = f32[4,4] broadcast(z), dimensions={1}
  add_a = f32[4,4] add(dot_a, z_bcast)
  c_bcast = f32[4,4] broadcast(c), dimensions={}
  dot_b = f32[4,4] dot(dot_a, c_bcast), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  ROOT out = f32[4,4] dot(add_a, dot_b), lhs_contracting_dims={1}, rhs_contracting_dims={0}
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

; CHECK-LABEL: ENTRY %test (x: f32[4,4], y: f32[4,4], z: f32[4]) -> f32[4,4] {
; CHECK-NEXT:    [[P0:%[^ ]+]] = f32[4,4]{1,0} parameter(0)
; CHECK-NEXT:    [[P1:%[^ ]+]] = f32[4,4]{1,0} parameter(1)
; CHECK-NEXT:    [[MATMUL0:%[^ ]+]] = f32[4,4]{1,0} custom-call([[P0]], [[P1]]),
; CHECK:           custom_call_target="__cublas$lt$matmul",
; CHECK:           backend_config="{
; CHECK-DAG:         \"alpha_real\":1
; CHECK-DAG:         \"alpha_imag\":0
; CHECK-DAG:         \"beta\":0
; CHECK-DAG:         \"dot_dimension_numbers\":{
; CHECK-DAG:           \"lhs_contracting_dimensions\":[\"1\"]
; CHECK-DAG:           \"rhs_contracting_dimensions\":[\"0\"]
; CHECK-DAG:           \"lhs_batch_dimensions\":[]
; CHECK-DAG:           \"rhs_batch_dimensions\":[]
; CHECK-DAG:         }
; CHECK-DAG:         \"precision_config\":{
; CHECK-DAG:           \"operand_precision\":[\"DEFAULT\",\"DEFAULT\"]
; CHECK-DAG:         }
; CHECK-DAG:         \"epilogue\":\"DEFAULT\"
; CHECK:           }"
; CHECK-NEXT:    [[P2:%[^ ]+]] = f32[4]{0} parameter(2)
; CHECK-NEXT:    [[FUSION:%[^ ]+]] = f32[4,4]{1,0} fusion([[MATMUL0]], [[P2]]), kind=kLoop, calls=[[FUSED_COMPUTATION]]
; CHECK-NEXT:    [[C0:%[^ ]+]] = f32[] constant(5)
; CHECK-NEXT:    [[C0_BCAST:%[^ ]+]] = f32[4,4]{1,0} broadcast([[C0]]), dimensions={}
; CHECK-NEXT:    [[MATMUL1:%[^ ]+]] = f32[4,4]{1,0} custom-call([[MATMUL0]], [[C0_BCAST]]),
; CHECK:           custom_call_target="__cublas$lt$matmul",
; CHECK:           backend_config="{
; CHECK-DAG:         \"alpha_real\":1
; CHECK-DAG:         \"alpha_imag\":0
; CHECK-DAG:         \"beta\":0
; CHECK-DAG:         \"dot_dimension_numbers\":{
; CHECK-DAG:           \"lhs_contracting_dimensions\":[\"1\"]
; CHECK-DAG:           \"rhs_contracting_dimensions\":[\"0\"]
; CHECK-DAG:           \"lhs_batch_dimensions\":[]
; CHECK-DAG:           \"rhs_batch_dimensions\":[]
; CHECK-DAG:         }
; CHECK-DAG:         \"precision_config\":{
; CHECK-DAG:           \"operand_precision\":[\"DEFAULT\",\"DEFAULT\"]
; CHECK-DAG:         }
; CHECK-DAG:         \"epilogue\":\"DEFAULT\"
; CHECK:           }"
; CHECK-NEXT:    ROOT [[OUT:%[^ ]+]] = f32[4,4]{1,0} custom-call([[FUSION]], [[MATMUL1]]),
; CHECK:           custom_call_target="__cublas$lt$matmul",
; CHECK:           backend_config="{
; CHECK-DAG:         \"alpha_real\":1
; CHECK-DAG:         \"alpha_imag\":0
; CHECK-DAG:         \"beta\":0
; CHECK-DAG:         \"dot_dimension_numbers\":{
; CHECK-DAG:           \"lhs_contracting_dimensions\":[\"1\"]
; CHECK-DAG:           \"rhs_contracting_dimensions\":[\"0\"]
; CHECK-DAG:           \"lhs_batch_dimensions\":[]
; CHECK-DAG:           \"rhs_batch_dimensions\":[]
; CHECK-DAG:         }
; CHECK-DAG:         \"precision_config\":{
; CHECK-DAG:           \"operand_precision\":[\"DEFAULT\",\"DEFAULT\"]
; CHECK-DAG:         }
; CHECK-DAG:         \"epilogue\":\"DEFAULT\"
; CHECK:           }"
      )");
}

TEST_F(CublasLtGemmRewriteTest, BatchedVectorBias) {
  const char* hlo_text = R"(
HloModule test

ENTRY test {
  x = f32[2,3,4] parameter(0)
  y = f32[4,5,6] parameter(1)
  z = f32[3,5,6] parameter(2)
  dot_a = f32[2,3,5,6] dot(x, y), lhs_contracting_dims={2}, rhs_contracting_dims={0}
  z_bcast = f32[2,3,5,6] broadcast(z), dimensions={1,2,3}
  ROOT out = f32[2,3,5,6] add(dot_a, z_bcast)
}

)";

  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{1e-5, 1e-5}));
  MatchOptimizedHlo(hlo_text,
                    R"(

; CHECK:        [[FUSED_COMPUTATION:%[^ ]+]] ([[DUMMY0:[^ ]+]]: f32[3,5,6]) -> f32[6,30] {
; CHECK-NEXT:    [[P0:%[^ ]+]] = f32[3,5,6]{2,1,0} parameter(0)
; CHECK-NEXT:    [[P0_BCAST:%[^ ]+]] = f32[2,3,5,6]{3,2,1,0} broadcast([[P0]]), dimensions={1,2,3}
; CHECK-NEXT:    ROOT [[OUT:%[^ ]+]] = f32[6,30]{1,0} bitcast([[P0_BCAST]])
}

; CHECK-LABEL: ENTRY %test (x: f32[2,3,4], y: f32[4,5,6], z: f32[3,5,6]) -> f32[2,3,5,6] {
; CHECK-NEXT:    [[P0:%[^ ]+]] = f32[2,3,4]{2,1,0} parameter(0)
; CHECK-NEXT:    [[P0_BITCAST:%[^ ]+]] = f32[6,4]{1,0} bitcast([[P0]])
; CHECK-NEXT:    [[P1:%[^ ]+]] = f32[4,5,6]{2,1,0} parameter(1)
; CHECK-NEXT:    [[P1_BITCAST:%[^ ]+]] = f32[4,30]{1,0}
; CHECK-NEXT:    [[P2:%[^ ]+]] = f32[3,5,6]{2,1,0} parameter(2)
; CHECK-NEXT:    [[FUSION:%[^ ]+]] = f32[6,30]{1,0} fusion([[P2]]), kind=kLoop, calls=[[FUSED_COMPUTATION]]
; CHECK-NEXT:    [[MATMUL:%[^ ]+]] = f32[6,30]{1,0} custom-call([[P0_BITCAST]], [[P1_BITCAST]], [[FUSION]]),
; CHECK:           custom_call_target="__cublas$lt$matmul",
; CHECK:           output_to_operand_aliasing={{[{][{]}}}: (2, {})},
; CHECK:           backend_config="{
; CHECK-DAG:         \"alpha_real\":1
; CHECK-DAG:         \"alpha_imag\":0
; CHECK-DAG:         \"beta\":1
; CHECK-DAG:         \"dot_dimension_numbers\":{
; CHECK-DAG:           \"lhs_contracting_dimensions\":[\"1\"]
; CHECK-DAG:           \"rhs_contracting_dimensions\":[\"0\"]
; CHECK-DAG:           \"lhs_batch_dimensions\":[]
; CHECK-DAG:           \"rhs_batch_dimensions\":[]
; CHECK-DAG:         }
; CHECK-DAG:         \"precision_config\":{
; CHECK-DAG:           \"operand_precision\":[\"DEFAULT\",\"DEFAULT\"]
; CHECK-DAG:         }
; CHECK-DAG:         \"epilogue\":\"DEFAULT\"
; CHECK:           }"
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
  dot_a = f32[2,3,5,6] dot(x, y), lhs_contracting_dims={2}, rhs_contracting_dims={0}
  z_bcast = f32[2,3,5,6] broadcast(z), dimensions={3}
  ROOT out = f32[2,3,5,6] add(dot_a, z_bcast)
}

)";

  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{1e-5, 1e-5}));
  MatchOptimizedHlo(hlo_text,
                    R"(

; CHECK:        [[FUSED_COMPUTATION:%[^ ]+]] ([[DUMMY0:[^ ]+]]: f32[6]) -> f32[6,30] {
; CHECK-NEXT:    [[P0:%[^ ]+]] = f32[6]{0} parameter(0)
; CHECK-NEXT:    [[P0_BCAST:%[^ ]+]] = f32[2,3,5,6]{3,2,1,0} broadcast([[P0]]), dimensions={3}
; CHECK-NEXT:    ROOT [[OUT:%[^ ]+]] = f32[6,30]{1,0} bitcast([[P0_BCAST]])
}

; CHECK-LABEL: ENTRY %test (x: f32[2,3,4], y: f32[4,5,6], z: f32[6]) -> f32[2,3,5,6] {
; CHECK-NEXT:    [[P0:%[^ ]+]] = f32[2,3,4]{2,1,0} parameter(0)
; CHECK-NEXT:    [[P0_BITCAST:%[^ ]+]] = f32[6,4]{1,0} bitcast([[P0]])
; CHECK-NEXT:    [[P1:%[^ ]+]] = f32[4,5,6]{2,1,0} parameter(1)
; CHECK-NEXT:    [[P1_BITCAST:%[^ ]+]] = f32[4,30]{1,0}
; CHECK-NEXT:    [[P2:%[^ ]+]] = f32[6]{0} parameter(2)
; CHECK-NEXT:    [[FUSION:%[^ ]+]] = f32[6,30]{1,0} fusion([[P2]]), kind=kLoop, calls=[[FUSED_COMPUTATION]]
; CHECK-NEXT:    [[MATMUL:%[^ ]+]] = f32[6,30]{1,0} custom-call([[P0_BITCAST]], [[P1_BITCAST]], [[FUSION]]),
; CHECK:           custom_call_target="__cublas$lt$matmul",
; CHECK:           output_to_operand_aliasing={{[{][{]}}}: (2, {})},
; CHECK:           backend_config="{
; CHECK-DAG:         \"alpha_real\":1
; CHECK-DAG:         \"alpha_imag\":0
; CHECK-DAG:         \"beta\":1
; CHECK-DAG:         \"dot_dimension_numbers\":{
; CHECK-DAG:           \"lhs_contracting_dimensions\":[\"1\"]
; CHECK-DAG:           \"rhs_contracting_dimensions\":[\"0\"]
; CHECK-DAG:           \"lhs_batch_dimensions\":[]
; CHECK-DAG:           \"rhs_batch_dimensions\":[]
; CHECK-DAG:         }
; CHECK-DAG:         \"precision_config\":{
; CHECK-DAG:           \"operand_precision\":[\"DEFAULT\",\"DEFAULT\"]
; CHECK-DAG:         }
; CHECK-DAG:         \"epilogue\":\"DEFAULT\"
; CHECK:           }"
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
; CHECK-LABEL: ENTRY %test (x: f32[2,3], y: f32[3,4], z: f32[2]) -> f32[4,2] {
; CHECK-NEXT:    [[P0:%[^ ]+]] = f32[2,3]{1,0} parameter(0)
; CHECK-NEXT:    [[P1:%[^ ]+]] = f32[3,4]{1,0} parameter(1)
; CHECK-NEXT:    [[P2:%[^ ]+]] = f32[2]{0} parameter(2)
; CHECK-NEXT:    [[MATMUL:%[^ ]+]] = f32[2,4]{0,1} custom-call([[P0]], [[P1]], [[P2]]),
; CHECK:           custom_call_target="__cublas$lt$matmul",
; CHECK:           backend_config="{
; CHECK-DAG:         \"alpha_real\":1
; CHECK-DAG:         \"alpha_imag\":0
; CHECK-DAG:         \"beta\":0
; CHECK-DAG:         \"dot_dimension_numbers\":{
; CHECK-DAG:           \"lhs_contracting_dimensions\":[\"1\"]
; CHECK-DAG:           \"rhs_contracting_dimensions\":[\"0\"]
; CHECK-DAG:           \"lhs_batch_dimensions\":[]
; CHECK-DAG:           \"rhs_batch_dimensions\":[]
; CHECK-DAG:         }
; CHECK-DAG:         \"precision_config\":{
; CHECK-DAG:           \"operand_precision\":[\"DEFAULT\",\"DEFAULT\"]
; CHECK-DAG:         }
; CHECK-DAG:         \"epilogue\":\"BIAS\"
; CHECK:           }"
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

; CHECK-LABEL: ENTRY %test (x: f32[4,3], y: f32[3,4], z: f32[3]) -> f32[2,3] {
; CHECK-NEXT:    [[P0:%[^ ]+]] = f32[4,3]{1,0} parameter(0)
; CHECK-NEXT:    [[P1:%[^ ]+]] = f32[3,4]{1,0} parameter(1)
; CHECK-NEXT:    [[P2:%[^ ]+]] = f32[3]{0} parameter(2)
; CHECK-NEXT:    [[MATMUL:%[^ ]+]] = f32[4,4]{1,0} custom-call([[P0]], [[P1]], [[P2]]),
; CHECK:           custom_call_target="__cublas$lt$matmul",
; CHECK:           backend_config="{
; CHECK-DAG:         \"alpha_real\":1
; CHECK-DAG:         \"alpha_imag\":0
; CHECK-DAG:         \"beta\":0
; CHECK-DAG:         \"dot_dimension_numbers\":{
; CHECK-DAG:           \"lhs_contracting_dimensions\":[\"1\"]
; CHECK-DAG:           \"rhs_contracting_dimensions\":[\"0\"]
; CHECK-DAG:           \"lhs_batch_dimensions\":[]
; CHECK-DAG:           \"rhs_batch_dimensions\":[]
; CHECK-DAG:         }
; CHECK-DAG:         \"precision_config\":{
; CHECK-DAG:           \"operand_precision\":[\"DEFAULT\",\"DEFAULT\"]
; CHECK-DAG:         }
; CHECK-DAG:         \"epilogue\":\"BIAS\"
; CHECK:           }"
; CHECK-NEXT:    ROOT [[OUT:%[^ ]+]] = f32[2,3]{1,0} slice([[MATMUL]]), slice={[0:2], [0:3]}
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

; CHECK:        [[FUSED_COMPUTATION:%[^ ]+]] ([[DUMMY0:[^ ]+]]: f32[2], [[DUMMY1:[^ ]+]]: f32[2,4]) -> f32[2,2] {
; CHECK-DAG:     [[P0:%[^ ]+]] = f32[2]{0} parameter(0)
; CHECK-DAG:     [[P1:%[^ ]+]] = f32[2,4]{1,0} parameter(1)
; CHECK-DAG:     [[SLICE:%[^ ]+]] = f32[2,2]{1,0} slice([[P1]]), slice={[0:2], [0:2]}
; CHECK-NEXT:    [[P0_BCAST:%[^ ]+]] = f32[2,2]{1,0} broadcast([[P0]]), dimensions={1}
; CHECK-NEXT:    ROOT [[OUT:%[^ ]+]] = f32[2,2]{1,0} add([[SLICE]], [[P0_BCAST]])
}

; CHECK-LABEL: ENTRY %test (x: f32[2,3], y: f32[3,4], z: f32[2]) -> f32[2,2] {
; CHECK-DAG:     [[P0:%[^ ]+]] = f32[2,3]{1,0} parameter(0)
; CHECK-DAG:     [[P1:%[^ ]+]] = f32[3,4]{1,0} parameter(1)
; CHECK-DAG:     [[P2:%[^ ]+]] = f32[2]{0} parameter(2)
; CHECK-NEXT:    [[MATMUL0:%[^ ]+]] = f32[2,4]{1,0} custom-call([[P0]], [[P1]]),
; CHECK:           custom_call_target="__cublas$lt$matmul",
; CHECK:           backend_config="{
; CHECK-DAG:         \"alpha_real\":1
; CHECK-DAG:         \"alpha_imag\":0
; CHECK-DAG:         \"beta\":0
; CHECK-DAG:         \"dot_dimension_numbers\":{
; CHECK-DAG:           \"lhs_contracting_dimensions\":[\"1\"]
; CHECK-DAG:           \"rhs_contracting_dimensions\":[\"0\"]
; CHECK-DAG:           \"lhs_batch_dimensions\":[]
; CHECK-DAG:           \"rhs_batch_dimensions\":[]
; CHECK-DAG:         }
; CHECK-DAG:         \"precision_config\":{
; CHECK-DAG:           \"operand_precision\":[\"DEFAULT\",\"DEFAULT\"]
; CHECK-DAG:         }
; CHECK-DAG:         \"epilogue\":\"DEFAULT\"
; CHECK:           }"
; CHECK-NEXT:    [[FUSION:%[^ ]+]] = f32[2,2]{1,0} fusion([[P2]], [[MATMUL0]]), kind=kLoop, calls=[[FUSED_COMPUTATION]]
; CHECK-NEXT:    [[SLICE:%[^ ]+]] = f32[2,2]{1,0} slice([[MATMUL0]]), slice={[0:2], [0:2]}
; CHECK-NEXT:    [[C0:%[^ ]+]] = f32[] constant(5)
; CHECK-NEXT:    [[C0_BCAST:%[^ ]+]] = f32[2,2]{1,0} broadcast([[C0]]), dimensions={}
; CHECK-NEXT:    [[MATMUL1:%[^ ]+]] = f32[2,2]{1,0} custom-call([[SLICE]], [[C0_BCAST]]),
; CHECK:           custom_call_target="__cublas$lt$matmul",
; CHECK:           backend_config="{
; CHECK-DAG:         \"alpha_real\":1
; CHECK-DAG:         \"alpha_imag\":0
; CHECK-DAG:         \"beta\":0
; CHECK-DAG:         \"dot_dimension_numbers\":{
; CHECK-DAG:           \"lhs_contracting_dimensions\":[\"1\"]
; CHECK-DAG:           \"rhs_contracting_dimensions\":[\"0\"]
; CHECK-DAG:           \"lhs_batch_dimensions\":[]
; CHECK-DAG:           \"rhs_batch_dimensions\":[]
; CHECK-DAG:         }
; CHECK-DAG:         \"precision_config\":{
; CHECK-DAG:           \"operand_precision\":[\"DEFAULT\",\"DEFAULT\"]
; CHECK-DAG:         }
; CHECK-DAG:         \"epilogue\":\"DEFAULT\"
; CHECK:           }"
; CHECK-NEXT:    ROOT [[OUT:%[^ ]+]] = f32[2,2]{1,0} custom-call([[FUSION]], [[MATMUL1]]),
; CHECK:           custom_call_target="__cublas$lt$matmul",
; CHECK:           backend_config="{
; CHECK-DAG:         \"alpha_real\":1
; CHECK-DAG:         \"alpha_imag\":0
; CHECK-DAG:         \"beta\":0
; CHECK-DAG:         \"dot_dimension_numbers\":{
; CHECK-DAG:           \"lhs_contracting_dimensions\":[\"1\"]
; CHECK-DAG:           \"rhs_contracting_dimensions\":[\"0\"]
; CHECK-DAG:           \"lhs_batch_dimensions\":[]
; CHECK-DAG:           \"rhs_batch_dimensions\":[]
; CHECK-DAG:         }
; CHECK-DAG:         \"precision_config\":{
; CHECK-DAG:           \"operand_precision\":[\"DEFAULT\",\"DEFAULT\"]
; CHECK-DAG:         }
; CHECK-DAG:         \"epilogue\":\"DEFAULT\"
; CHECK:           }"
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
; CHECK-NEXT:    ROOT [[OUT:%[^ ]+]] = f32[2,4]{1,0} custom-call([[P0]], [[P1]], [[P2_BCAST]]),
; CHECK:           custom_call_target="__cublas$lt$matmul",
; CHECK:           backend_config="{
; CHECK-DAG:         \"alpha_real\":1
; CHECK-DAG:         \"alpha_imag\":0
; CHECK-DAG:         \"beta\":1
; CHECK-DAG:         \"dot_dimension_numbers\":{
; CHECK-DAG:           \"lhs_contracting_dimensions\":[\"1\"]
; CHECK-DAG:           \"rhs_contracting_dimensions\":[\"0\"]
; CHECK-DAG:           \"lhs_batch_dimensions\":[]
; CHECK-DAG:           \"rhs_batch_dimensions\":[]
; CHECK-DAG:         }
; CHECK-DAG:         \"precision_config\":{
; CHECK-DAG:           \"operand_precision\":[\"DEFAULT\",\"DEFAULT\"]
; CHECK-DAG:         }
; CHECK-DAG:         \"epilogue\":\"DEFAULT\"
; CHECK:           }"
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
; CHECK-LABEL: ENTRY %test (x: f32[2,3], y: f32[3,4], z: f32[4], z2: f32[2,4]) -> f32[2,4] {
; CHECK-DAG:     [[P0:%[^ ]+]] = f32[2,3]{1,0} parameter(0)
; CHECK-DAG:     [[P1:%[^ ]+]] = f32[3,4]{1,0} parameter(1)
; CHECK-DAG:     [[VECTOR_BIAS:%[^ ]+]] = f32[4]{0} parameter(2)
; CHECK-DAG:     [[MATRIX_BIAS:%[^ ]+]] = f32[2,4]{1,0} parameter(3)
; CHECK-NEXT:    ROOT [[OUT:%[^ ]+]] = f32[2,4]{1,0} custom-call([[P0]], [[P1]], [[MATRIX_BIAS]], [[VECTOR_BIAS]]),
; CHECK:           custom_call_target="__cublas$lt$matmul",
; CHECK:           backend_config="{
; CHECK-DAG:         \"alpha_real\":1
; CHECK-DAG:         \"alpha_imag\":0
; CHECK-DAG:         \"beta\":1
; CHECK-DAG:         \"dot_dimension_numbers\":{
; CHECK-DAG:           \"lhs_contracting_dimensions\":[\"1\"]
; CHECK-DAG:           \"rhs_contracting_dimensions\":[\"0\"]
; CHECK-DAG:           \"lhs_batch_dimensions\":[]
; CHECK-DAG:           \"rhs_batch_dimensions\":[]
; CHECK-DAG:         }
; CHECK-DAG:         \"precision_config\":{
; CHECK-DAG:           \"operand_precision\":[\"DEFAULT\",\"DEFAULT\"]
; CHECK-DAG:         }
; CHECK-DAG:         \"epilogue\":\"BIAS\"
; CHECK:           }"
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
  MatchOptimizedHlo(hlo_text,
                    R"(

; CHECK-LABEL: ENTRY %test (x: bf16[16,24], y: bf16[24,32], z: bf16[32]) -> bf16[16,32] {
; CHECK-NEXT:    [[P0:%[^ ]+]] = bf16[16,24]{1,0} parameter(0)
; CHECK-NEXT:    [[P1:%[^ ]+]] = bf16[24,32]{1,0} parameter(1)
; CHECK-NEXT:    [[P2:%[^ ]+]] = bf16[32]{0} parameter(2)
; CHECK-NEXT:    ROOT [[OUT:%[^ ]+]] = bf16[16,32]{1,0} custom-call([[P0]], [[P1]], [[P2]]),
; CHECK:           custom_call_target="__cublas$lt$matmul",
; CHECK:           backend_config="{
; CHECK-DAG:         \"alpha_real\":1
; CHECK-DAG:         \"alpha_imag\":0
; CHECK-DAG:         \"beta\":0
; CHECK-DAG:         \"dot_dimension_numbers\":{
; CHECK-DAG:           \"lhs_contracting_dimensions\":[\"1\"]
; CHECK-DAG:           \"rhs_contracting_dimensions\":[\"0\"]
; CHECK-DAG:           \"lhs_batch_dimensions\":[]
; CHECK-DAG:           \"rhs_batch_dimensions\":[]
; CHECK-DAG:         }
; CHECK-DAG:         \"precision_config\":{
; CHECK-DAG:           \"operand_precision\":[\"DEFAULT\",\"DEFAULT\"]
; CHECK-DAG:         }
; CHECK-DAG:         \"epilogue\":\"BIAS\"
      )");
}

TEST_F(CublasLtGemmRewriteTest, BF16VectorBiasPadded) {
  if (!GetCudaComputeCapability().IsAtLeast(
          se::CudaComputeCapability::AMPERE)) {
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
}

)";

  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{1e-3, 1e-3}));
  MatchOptimizedHlo(hlo_text,
                    R"(

; CHECK-LABEL: ENTRY %test (x: bf16[2,3], y: bf16[3,4], z: bf16[4]) -> bf16[2,4] {
; CHECK-NEXT:    [[P0:%[^ ]+]] = bf16[2,3]{1,0} parameter(0)
; CHECK-NEXT:    [[C0:%[^ ]+]] = bf16[] constant(0)
; CHECK-NEXT:    [[P0_PADDED:%[^ ]+]] = bf16[8,8]{1,0} pad([[P0]], [[C0]]), padding=0_6x0_5
; CHECK-NEXT:    [[P1:%[^ ]+]] = bf16[3,4]{1,0} parameter(1)
; CHECK-NEXT:    [[P1_PADDED:%[^ ]+]] = bf16[8,8]{1,0} pad([[P1]], [[C0]]), padding=0_5x0_4
; CHECK-NEXT:    [[P2:%[^ ]+]] = bf16[4]{0} parameter(2)
; CHECK-NEXT:    [[MATMUL:%[^ ]+]] = bf16[8,8]{1,0} custom-call([[P0_PADDED]], [[P1_PADDED]], [[P2]]),
; CHECK:           custom_call_target="__cublas$lt$matmul",
; CHECK:           backend_config="{
; CHECK-DAG:         \"alpha_real\":1
; CHECK-DAG:         \"alpha_imag\":0
; CHECK-DAG:         \"beta\":0
; CHECK-DAG:         \"dot_dimension_numbers\":{
; CHECK-DAG:           \"lhs_contracting_dimensions\":[\"1\"]
; CHECK-DAG:           \"rhs_contracting_dimensions\":[\"0\"]
; CHECK-DAG:           \"lhs_batch_dimensions\":[]
; CHECK-DAG:           \"rhs_batch_dimensions\":[]
; CHECK-DAG:         }
; CHECK-DAG:         \"precision_config\":{
; CHECK-DAG:           \"operand_precision\":[\"DEFAULT\",\"DEFAULT\"]
; CHECK-DAG:         }
; CHECK-DAG:         \"epilogue\":\"BIAS\"
; CHECK-NEXT:    ROOT [[OUT:%[^ ]+]] = bf16[2,4]{1,0} slice([[MATMUL]]), slice={[0:2], [0:4]}
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

; CHECK-LABEL: ENTRY %test (x: f32[2,3], y: f32[3,4]) -> f32[2,4] {
; CHECK-NEXT:    [[P0:%[^ ]+]] = f32[2,3]{1,0} parameter(0)
; CHECK-NEXT:    [[P1:%[^ ]+]] = f32[3,4]{1,0} parameter(1)
; CHECK-NEXT:    ROOT [[OUT:%[^ ]+]] = f32[2,4]{1,0} custom-call([[P0]], [[P1]]),
; CHECK:           custom_call_target="__cublas$lt$matmul",
; CHECK:           backend_config="{
; CHECK-DAG:         \"alpha_real\":1
; CHECK-DAG:         \"alpha_imag\":0
; CHECK-DAG:         \"beta\":0
; CHECK-DAG:         \"dot_dimension_numbers\":{
; CHECK-DAG:           \"lhs_contracting_dimensions\":[\"1\"]
; CHECK-DAG:           \"rhs_contracting_dimensions\":[\"0\"]
; CHECK-DAG:           \"lhs_batch_dimensions\":[]
; CHECK-DAG:           \"rhs_batch_dimensions\":[]
; CHECK-DAG:         }
; CHECK-DAG:         \"precision_config\":{
; CHECK-DAG:           \"operand_precision\":[\"DEFAULT\",\"DEFAULT\"]
; CHECK-DAG:         }
; CHECK-DAG:         \"epilogue\":\"RELU\"
; CHECK:           }"
      )");
}

TEST_F(CublasLtGemmRewriteTest, BatchedReluActivation) {
  const char* hlo_text = R"(
HloModule test

ENTRY test {
  x = f32[2,3,4] parameter(0)
  y = f32[4,5,6] parameter(1)
  dot_a = f32[2,3,5,6] dot(x, y), lhs_contracting_dims={2}, rhs_contracting_dims={0}
  c = f32[] constant(0)
  c_bcast = f32[2,3,5,6] broadcast(c), dimensions={}
  ROOT out = f32[2,3,5,6] maximum(dot_a, c_bcast)
}

)";

  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{1e-5, 1e-5}));
  MatchOptimizedHlo(hlo_text,
                    R"(

; CHECK-LABEL: ENTRY %test (x: f32[2,3,4], y: f32[4,5,6]) -> f32[2,3,5,6] {
; CHECK-NEXT:    [[P0:%[^ ]+]] = f32[2,3,4]{2,1,0} parameter(0)
; CHECK-NEXT:    [[P0_BITCAST:%[^ ]+]] = f32[6,4]{1,0} bitcast([[P0]])
; CHECK-NEXT:    [[P1:%[^ ]+]] = f32[4,5,6]{2,1,0} parameter(1)
; CHECK-NEXT:    [[P1_BITCAST:%[^ ]+]] = f32[4,30]{1,0}
; CHECK-NEXT:    [[MATMUL:%[^ ]+]] = f32[6,30]{1,0} custom-call([[P0_BITCAST]], [[P1_BITCAST]]),
; CHECK:           custom_call_target="__cublas$lt$matmul",
; CHECK:           backend_config="{
; CHECK-DAG:         \"alpha_real\":1
; CHECK-DAG:         \"alpha_imag\":0
; CHECK-DAG:         \"beta\":0
; CHECK-DAG:         \"dot_dimension_numbers\":{
; CHECK-DAG:           \"lhs_contracting_dimensions\":[\"1\"]
; CHECK-DAG:           \"rhs_contracting_dimensions\":[\"0\"]
; CHECK-DAG:           \"lhs_batch_dimensions\":[]
; CHECK-DAG:           \"rhs_batch_dimensions\":[]
; CHECK-DAG:         }
; CHECK-DAG:         \"precision_config\":{
; CHECK-DAG:           \"operand_precision\":[\"DEFAULT\",\"DEFAULT\"]
; CHECK-DAG:         }
; CHECK-DAG:         \"epilogue\":\"RELU\"
; CHECK:           }"
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

; CHECK-LABEL: ENTRY %test (x: f32[2,3], y: f32[3,4]) -> f32[2,2] {
; CHECK-NEXT:    [[P0:%[^ ]+]] = f32[2,3]{1,0} parameter(0)
; CHECK-NEXT:    [[P1:%[^ ]+]] = f32[3,4]{1,0} parameter(1)
; CHECK-NEXT:    [[MATMUL:%[^ ]+]] = f32[2,4]{1,0} custom-call([[P0]], [[P1]]),
; CHECK:           custom_call_target="__cublas$lt$matmul",
; CHECK:           backend_config="{
; CHECK-DAG:         \"alpha_real\":1
; CHECK-DAG:         \"alpha_imag\":0
; CHECK-DAG:         \"beta\":0
; CHECK-DAG:         \"dot_dimension_numbers\":{
; CHECK-DAG:           \"lhs_contracting_dimensions\":[\"1\"]
; CHECK-DAG:           \"rhs_contracting_dimensions\":[\"0\"]
; CHECK-DAG:           \"lhs_batch_dimensions\":[]
; CHECK-DAG:           \"rhs_batch_dimensions\":[]
; CHECK-DAG:         }
; CHECK-DAG:         \"precision_config\":{
; CHECK-DAG:           \"operand_precision\":[\"DEFAULT\",\"DEFAULT\"]
; CHECK-DAG:         }
; CHECK-DAG:         \"epilogue\":\"RELU\"
; CHECK:           }"
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

; CHECK-LABEL: ENTRY %test (x: f32[2,3], y: f32[3,4], z: f32[2,4]) -> f32[2,4] {
; CHECK-NEXT:    [[P0:%[^ ]+]] = f32[2,3]{1,0} parameter(0)
; CHECK-NEXT:    [[P1:%[^ ]+]] = f32[3,4]{1,0} parameter(1)
; CHECK-NEXT:    [[P2:%[^ ]+]] = f32[2,4]{1,0} parameter(2)
; CHECK-NEXT:    ROOT [[OUT:%[^ ]+]] = f32[2,4]{1,0} custom-call([[P0]], [[P1]], [[P2]]),
; CHECK:           custom_call_target="__cublas$lt$matmul",
; CHECK:           backend_config="{
; CHECK-DAG:         \"alpha_real\":1
; CHECK-DAG:         \"alpha_imag\":0
; CHECK-DAG:         \"beta\":1
; CHECK-DAG:         \"dot_dimension_numbers\":{
; CHECK-DAG:           \"lhs_contracting_dimensions\":[\"1\"]
; CHECK-DAG:           \"rhs_contracting_dimensions\":[\"0\"]
; CHECK-DAG:           \"lhs_batch_dimensions\":[]
; CHECK-DAG:           \"rhs_batch_dimensions\":[]
; CHECK-DAG:         }
; CHECK-DAG:         \"precision_config\":{
; CHECK-DAG:           \"operand_precision\":[\"DEFAULT\",\"DEFAULT\"]
; CHECK-DAG:         }
; CHECK-DAG:         \"epilogue\":\"RELU\"
; CHECK:           }"
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

; CHECK-LABEL: ENTRY %test (x: f32[4,4], y: f32[4,4], z: f32[4,4]) -> f32[4,4] {
; CHECK-NEXT:    [[P0:%[^ ]+]] = f32[4,4]{1,0} parameter(0)
; CHECK-NEXT:    [[P1:%[^ ]+]] = f32[4,4]{1,0} parameter(1)
; CHECK-NEXT:    [[P2:%[^ ]+]] = f32[4,4]{1,0} parameter(2)
; CHECK-NEXT:    ROOT [[OUT:%[^ ]+]] = f32[4,4]{1,0} custom-call([[P0]], [[P1]], [[P2]]),
; CHECK:           custom_call_target="__cublas$lt$matmul",
; CHECK:           backend_config="{
; CHECK-DAG:         \"alpha_real\":1
; CHECK-DAG:         \"alpha_imag\":0
; CHECK-DAG:         \"beta\":1
; CHECK-DAG:         \"dot_dimension_numbers\":{
; CHECK-DAG:           \"lhs_contracting_dimensions\":[\"1\"]
; CHECK-DAG:           \"rhs_contracting_dimensions\":[\"0\"]
; CHECK-DAG:           \"lhs_batch_dimensions\":[]
; CHECK-DAG:           \"rhs_batch_dimensions\":[]
; CHECK-DAG:         }
; CHECK-DAG:         \"precision_config\":{
; CHECK-DAG:           \"operand_precision\":[\"DEFAULT\",\"DEFAULT\"]
; CHECK-DAG:         }
; CHECK-DAG:         \"epilogue\":\"RELU\"
; CHECK:           }"
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

; CHECK-LABEL: ENTRY %test (x: f32[2,3], y: f32[3,4], z: f32[4]) -> f32[2,4] {
; CHECK-NEXT:    [[P0:%[^ ]+]] = f32[2,3]{1,0} parameter(0)
; CHECK-NEXT:    [[P1:%[^ ]+]] = f32[3,4]{1,0} parameter(1)
; CHECK-NEXT:    [[P2:%[^ ]+]] = f32[4]{0} parameter(2)
; CHECK-NEXT:    ROOT [[OUT:%[^ ]+]] = f32[2,4]{1,0} custom-call([[P0]], [[P1]], [[P2]]),
; CHECK:           custom_call_target="__cublas$lt$matmul",
; CHECK:           backend_config="{
; CHECK-DAG:         \"alpha_real\":1
; CHECK-DAG:         \"alpha_imag\":0
; CHECK-DAG:         \"beta\":0
; CHECK-DAG:         \"dot_dimension_numbers\":{
; CHECK-DAG:           \"lhs_contracting_dimensions\":[\"1\"]
; CHECK-DAG:           \"rhs_contracting_dimensions\":[\"0\"]
; CHECK-DAG:           \"lhs_batch_dimensions\":[]
; CHECK-DAG:           \"rhs_batch_dimensions\":[]
; CHECK-DAG:         }
; CHECK-DAG:         \"precision_config\":{
; CHECK-DAG:           \"operand_precision\":[\"DEFAULT\",\"DEFAULT\"]
; CHECK-DAG:         }
; CHECK-DAG:         \"epilogue\":\"BIAS_RELU\"
; CHECK:           }"
      )");
}

TEST_F(CublasLtGemmRewriteTest, BatchedVectorBiasReluActivation) {
  const char* hlo_text = R"(
HloModule test

ENTRY test {
  x = f32[2,3,4] parameter(0)
  y = f32[4,5,6] parameter(1)
  z = f32[3,5,6] parameter(2)
  dot_a = f32[2,3,5,6] dot(x, y), lhs_contracting_dims={2}, rhs_contracting_dims={0}
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

; CHECK:        [[FUSED_COMPUTATION:%[^ ]+]] ([[DUMMY0:[^ ]+]]: f32[3,5,6]) -> f32[6,30] {
; CHECK-NEXT:    [[P0:%[^ ]+]] = f32[3,5,6]{2,1,0} parameter(0)
; CHECK-NEXT:    [[P0_BCAST:%[^ ]+]] = f32[2,3,5,6]{3,2,1,0} broadcast([[P0]]), dimensions={1,2,3}
; CHECK-NEXT:    ROOT [[OUT:%[^ ]+]] = f32[6,30]{1,0} bitcast([[P0_BCAST]])
}

; CHECK-LABEL: ENTRY %test (x: f32[2,3,4], y: f32[4,5,6], z: f32[3,5,6]) -> f32[2,3,5,6] {
; CHECK-NEXT:    [[P0:%[^ ]+]] = f32[2,3,4]{2,1,0} parameter(0)
; CHECK-NEXT:    [[P0_BITCAST:%[^ ]+]] = f32[6,4]{1,0} bitcast([[P0]])
; CHECK-NEXT:    [[P1:%[^ ]+]] = f32[4,5,6]{2,1,0} parameter(1)
; CHECK-NEXT:    [[P1_BITCAST:%[^ ]+]] = f32[4,30]{1,0}
; CHECK-NEXT:    [[P2:%[^ ]+]] = f32[3,5,6]{2,1,0} parameter(2)
; CHECK-NEXT:    [[FUSION:%[^ ]+]] = f32[6,30]{1,0} fusion([[P2]]), kind=kLoop, calls=[[FUSED_COMPUTATION]]
; CHECK-NEXT:    [[MATMUL:%[^ ]+]] = f32[6,30]{1,0} custom-call([[P0_BITCAST]], [[P1_BITCAST]], [[FUSION]]),
; CHECK:           custom_call_target="__cublas$lt$matmul",
; CHECK:           backend_config="{
; CHECK-DAG:         \"alpha_real\":1
; CHECK-DAG:         \"alpha_imag\":0
; CHECK-DAG:         \"beta\":1
; CHECK-DAG:         \"dot_dimension_numbers\":{
; CHECK-DAG:           \"lhs_contracting_dimensions\":[\"1\"]
; CHECK-DAG:           \"rhs_contracting_dimensions\":[\"0\"]
; CHECK-DAG:           \"lhs_batch_dimensions\":[]
; CHECK-DAG:           \"rhs_batch_dimensions\":[]
; CHECK-DAG:         }
; CHECK-DAG:         \"precision_config\":{
; CHECK-DAG:           \"operand_precision\":[\"DEFAULT\",\"DEFAULT\"]
; CHECK-DAG:         }
; CHECK-DAG:         \"epilogue\":\"RELU\"
; CHECK:           }"
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

; CHECK-LABEL: ENTRY %test (x: f32[2,3], y: f32[3,4], z: f32[2]) -> f32[4,2] {
; CHECK-NEXT:    [[P0:%[^ ]+]] = f32[2,3]{1,0} parameter(0)
; CHECK-NEXT:    [[P1:%[^ ]+]] = f32[3,4]{1,0} parameter(1)
; CHECK-NEXT:    [[P2:%[^ ]+]] = f32[2]{0} parameter(2)
; CHECK-NEXT:    [[MATMUL:%[^ ]+]] = f32[2,4]{0,1} custom-call([[P0]], [[P1]], [[P2]]),
; CHECK:           custom_call_target="__cublas$lt$matmul",
; CHECK:           backend_config="{
; CHECK-DAG:       \"alpha_real\":1
; CHECK-DAG:       \"alpha_imag\":0
; CHECK-DAG:       \"beta\":0
; CHECK-DAG:         \"dot_dimension_numbers\":{
; CHECK-DAG:           \"lhs_contracting_dimensions\":[\"1\"]
; CHECK-DAG:           \"rhs_contracting_dimensions\":[\"0\"]
; CHECK-DAG:           \"lhs_batch_dimensions\":[]
; CHECK-DAG:           \"rhs_batch_dimensions\":[]
; CHECK-DAG:         }
; CHECK-DAG:         \"precision_config\":{
; CHECK-DAG:           \"operand_precision\":[\"DEFAULT\",\"DEFAULT\"]
; CHECK-DAG:         }
; CHECK-DAG:         \"epilogue\":\"BIAS_RELU\"
; CHECK:           }"
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

; CHECK-LABEL: ENTRY %test (x: f32[2,3], y: f32[3,4], z_vec: f32[4], z_matrix: f32[2,4]) -> f32[2,4] {
; CHECK-DAG:     [[P0:%[^ ]+]] = f32[2,3]{1,0} parameter(0)
; CHECK-DAG:     [[P1:%[^ ]+]] = f32[3,4]{1,0} parameter(1)
; CHECK-DAG:     [[P2:%[^ ]+]] = f32[4]{0} parameter(2)
; CHECK-DAG:     [[P3:%[^ ]+]] = f32[2,4]{1,0} parameter(3)
; CHECK-NEXT:    ROOT [[OUT:%[^ ]+]] = f32[2,4]{1,0} custom-call([[P0]], [[P1]], [[P3]], [[P2]]),
; CHECK:           custom_call_target="__cublas$lt$matmul",
; CHECK:           backend_config="{
; CHECK-DAG:         \"alpha_real\":1
; CHECK-DAG:         \"alpha_imag\":0
; CHECK-DAG:         \"beta\":1
; CHECK-DAG:         \"dot_dimension_numbers\":{
; CHECK-DAG:           \"lhs_contracting_dimensions\":[\"1\"]
; CHECK-DAG:           \"rhs_contracting_dimensions\":[\"0\"]
; CHECK-DAG:           \"lhs_batch_dimensions\":[]
; CHECK-DAG:           \"rhs_batch_dimensions\":[]
; CHECK-DAG:         }
; CHECK-DAG:         \"precision_config\":{
; CHECK-DAG:           \"operand_precision\":[\"DEFAULT\",\"DEFAULT\"]
; CHECK-DAG:         }
; CHECK-DAG:         \"epilogue\":\"BIAS_RELU\"
; CHECK:           }"
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

; CHECK-LABEL: ENTRY %test (x: f32[2,3], y: f32[3,4]) -> f32[2,4] {
; CHECK-NEXT:    [[P0:%[^ ]+]] = f32[2,3]{1,0} parameter(0)
; CHECK-NEXT:    [[P1:%[^ ]+]] = f32[3,4]{1,0} parameter(1)
; CHECK-NEXT:    ROOT [[OUT:%[^ ]+]] = f32[2,4]{1,0} custom-call([[P0]], [[P1]]),
; CHECK:           custom_call_target="__cublas$lt$matmul",
; CHECK:           backend_config="{
; CHECK-DAG:         \"alpha_real\":1
; CHECK-DAG:         \"alpha_imag\":0
; CHECK-DAG:         \"beta\":0
; CHECK-DAG:         \"dot_dimension_numbers\":{
; CHECK-DAG:           \"lhs_contracting_dimensions\":[\"1\"]
; CHECK-DAG:           \"rhs_contracting_dimensions\":[\"0\"]
; CHECK-DAG:           \"lhs_batch_dimensions\":[]
; CHECK-DAG:           \"rhs_batch_dimensions\":[]
; CHECK-DAG:         }
; CHECK-DAG:         \"precision_config\":{
; CHECK-DAG:           \"operand_precision\":[\"DEFAULT\",\"DEFAULT\"]
; CHECK-DAG:         }
; CHECK-DAG:         \"epilogue\":\"GELU\"
; CHECK:           }"
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

TEST_F(CublasLtGemmRewriteTest, VectorBiasThenApproxGeluActivation) {
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

; CHECK-LABEL: ENTRY %test (x: f32[2,3], y: f32[3,4], z: f32[4]) -> f32[2,4] {
; CHECK-NEXT:    [[P0:%[^ ]+]] = f32[2,3]{1,0} parameter(0)
; CHECK-NEXT:    [[P1:%[^ ]+]] = f32[3,4]{1,0} parameter(1)
; CHECK-NEXT:    [[P2:%[^ ]+]] = f32[4]{0} parameter(2)
; CHECK-NEXT:    ROOT [[OUT:%[^ ]+]] = f32[2,4]{1,0} custom-call([[P0]], [[P1]], [[P2]]),
; CHECK:           custom_call_target="__cublas$lt$matmul",
; CHECK:           backend_config="{
; CHECK-DAG:         \"alpha_real\":1
; CHECK-DAG:         \"alpha_imag\":0
; CHECK-DAG:         \"beta\":0
; CHECK-DAG:         \"dot_dimension_numbers\":{
; CHECK-DAG:           \"lhs_contracting_dimensions\":[\"1\"]
; CHECK-DAG:           \"rhs_contracting_dimensions\":[\"0\"]
; CHECK-DAG:           \"lhs_batch_dimensions\":[]
; CHECK-DAG:           \"rhs_batch_dimensions\":[]
; CHECK-DAG:         }
; CHECK-DAG:         \"precision_config\":{
; CHECK-DAG:           \"operand_precision\":[\"DEFAULT\",\"DEFAULT\"]
; CHECK-DAG:         }
; CHECK-DAG:         \"epilogue\":\"BIAS_GELU\"
; CHECK:           }"
      )");
}

TEST_F(CublasLtGemmRewriteTest, ApproxGeluActivationWithAux) {
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

; CHECK-LABEL: ENTRY %test (x: f32[2,3], y: f32[3,4]) -> (f32[2,4], f32[2,4]) {
; CHECK-NEXT:    [[P0:%[^ ]+]] = f32[2,3]{1,0} parameter(0)
; CHECK-NEXT:    [[P1:%[^ ]+]] = f32[3,4]{1,0} parameter(1)
; CHECK-NEXT:    ROOT [[OUT:%[^ ]+]] = (f32[2,4]{1,0}, f32[2,4]{1,0}) custom-call([[P0]], [[P1]]),
; CHECK:           custom_call_target="__cublas$lt$matmul",
; CHECK:           backend_config="{
; CHECK-DAG:         \"alpha_real\":1
; CHECK-DAG:         \"alpha_imag\":0
; CHECK-DAG:         \"beta\":0
; CHECK-DAG:         \"dot_dimension_numbers\":{
; CHECK-DAG:           \"lhs_contracting_dimensions\":[\"1\"]
; CHECK-DAG:           \"rhs_contracting_dimensions\":[\"0\"]
; CHECK-DAG:           \"lhs_batch_dimensions\":[]
; CHECK-DAG:           \"rhs_batch_dimensions\":[]
; CHECK-DAG:         }
; CHECK-DAG:         \"precision_config\":{
; CHECK-DAG:           \"operand_precision\":[\"DEFAULT\",\"DEFAULT\"]
; CHECK-DAG:         }
; CHECK-DAG:         \"epilogue\":\"GELU_AUX\"
; CHECK:           }"
      )");
}

TEST_F(CublasLtGemmRewriteTest, VectorBiasThenApproxGeluActivationWithAux) {
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

; CHECK-LABEL: ENTRY %test (x: f32[2,3], y: f32[3,4], z: f32[4]) -> (f32[2,4], f32[2,4]) {
; CHECK-NEXT:    [[P0:%[^ ]+]] = f32[2,3]{1,0} parameter(0)
; CHECK-NEXT:    [[P1:%[^ ]+]] = f32[3,4]{1,0} parameter(1)
; CHECK-NEXT:    [[P2:%[^ ]+]] = f32[4]{0} parameter(2)
; CHECK-NEXT:    ROOT [[OUT:%[^ ]+]] = (f32[2,4]{1,0}, f32[2,4]{1,0}) custom-call([[P0]], [[P1]], [[P2]]),
; CHECK:           custom_call_target="__cublas$lt$matmul",
; CHECK:           backend_config="{
; CHECK-DAG:         \"alpha_real\":1
; CHECK-DAG:         \"alpha_imag\":0
; CHECK-DAG:         \"beta\":0
; CHECK-DAG:         \"dot_dimension_numbers\":{
; CHECK-DAG:           \"lhs_contracting_dimensions\":[\"1\"]
; CHECK-DAG:           \"rhs_contracting_dimensions\":[\"0\"]
; CHECK-DAG:           \"lhs_batch_dimensions\":[]
; CHECK-DAG:           \"rhs_batch_dimensions\":[]
; CHECK-DAG:         }
; CHECK-DAG:         \"precision_config\":{
; CHECK-DAG:           \"operand_precision\":[\"DEFAULT\",\"DEFAULT\"]
; CHECK-DAG:         }
; CHECK-DAG:         \"epilogue\":\"BIAS_GELU_AUX\"
; CHECK:           }"
      )");
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

; CHECK-LABEL: ENTRY %test (x: f16[8,16], y: f16[16,8], z: f16[8,8]) -> f16[8,8] {
; CHECK-NEXT:    [[P0:%[^ ]+]] = f16[8,16]{1,0} parameter(0)
; CHECK-NEXT:    [[P1:%[^ ]+]] = f16[16,8]{1,0} parameter(1)
; CHECK-NEXT:    [[P2:%[^ ]+]] = f16[8,8]{1,0} parameter(2)
; CHECK-NEXT:    ROOT [[OUT:%[^ ]+]] = f16[8,8]{1,0} custom-call([[P0]], [[P1]], [[P2]]),
; CHECK:           custom_call_target="__cublas$lt$matmul",
; CHECK:           backend_config="{
; CHECK-DAG:         \"alpha_real\":1
; CHECK-DAG:         \"alpha_imag\":0
; CHECK-DAG:         \"beta\":1
; CHECK-DAG:         \"dot_dimension_numbers\":{
; CHECK-DAG:           \"lhs_contracting_dimensions\":[\"1\"]
; CHECK-DAG:           \"rhs_contracting_dimensions\":[\"0\"]
; CHECK-DAG:           \"lhs_batch_dimensions\":[]
; CHECK-DAG:           \"rhs_batch_dimensions\":[]
; CHECK-DAG:         }
; CHECK-DAG:         \"precision_config\":{
; CHECK-DAG:           \"operand_precision\":[\"DEFAULT\",\"DEFAULT\"]
; CHECK-DAG:         }
; CHECK-DAG:         \"epilogue\":\"DEFAULT\"
; CHECK:           }"
      )");
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
}

)";
  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{8e-3, 2e-3}));
  MatchOptimizedHlo(hlo_text,
                    R"(

; CHECK-LABEL: ENTRY %test (x: f16[8,16], y: f16[16,8], z: f16[8]) -> f16[8,8] {
; CHECK-NEXT:    [[P0:%[^ ]+]] = f16[8,16]{1,0} parameter(0)
; CHECK-NEXT:    [[P1:%[^ ]+]] = f16[16,8]{1,0} parameter(1)
; CHECK-NEXT:    [[P2:%[^ ]+]] = f16[8]{0} parameter(2)
; CHECK-NEXT:    ROOT [[OUT:%[^ ]+]] = f16[8,8]{1,0} custom-call([[P0]], [[P1]], [[P2]]),
; CHECK:           custom_call_target="__cublas$lt$matmul",
; CHECK:           backend_config="{
; CHECK-DAG:         \"alpha_real\":1
; CHECK-DAG:         \"alpha_imag\":0
; CHECK-DAG:         \"beta\":0
; CHECK-DAG:         \"dot_dimension_numbers\":{
; CHECK-DAG:           \"lhs_contracting_dimensions\":[\"1\"]
; CHECK-DAG:           \"rhs_contracting_dimensions\":[\"0\"]
; CHECK-DAG:           \"lhs_batch_dimensions\":[]
; CHECK-DAG:           \"rhs_batch_dimensions\":[]
; CHECK-DAG:         }
; CHECK-DAG:         \"precision_config\":{
; CHECK-DAG:           \"operand_precision\":[\"DEFAULT\",\"DEFAULT\"]
; CHECK-DAG:         }
; CHECK-DAG:         \"epilogue\":\"BIAS\"
; CHECK:           }"
      )");
}

TEST_F(CublasLtGemmRewriteTest, VectorBiasF16Padded) {
  if (!GetCudaComputeCapability().IsAtLeast(se::CudaComputeCapability::VOLTA)) {
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
}

)";
  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{1e-3, 1e-3}));
  MatchOptimizedHlo(hlo_text,
                    R"(

; CHECK-LABEL: ENTRY %test (x: f16[6,12], y: f16[12,6], z: f16[6]) -> f16[6,6] {
; CHECK-NEXT:    [[P0:%[^ ]+]] = f16[6,12]{1,0} parameter(0)
; CHECK-NEXT:    [[C0:%[^ ]+]] = f16[] constant(0)
; CHECK-NEXT:    [[P0_PADDED:%[^ ]+]] = f16[8,16]{1,0} pad([[P0]], [[C0]]), padding=0_2x0_4
; CHECK-NEXT:    [[P1:%[^ ]+]] = f16[12,6]{1,0} parameter(1)
; CHECK-NEXT:    [[P1_PADDED:%[^ ]+]] = f16[16,8]{1,0} pad([[P1]], [[C0]]), padding=0_4x0_2
; CHECK-NEXT:    [[P2:%[^ ]+]] = f16[6]{0} parameter(2)
; CHECK-NEXT:    [[MATMUL:%[^ ]+]] = f16[8,8]{1,0} custom-call([[P0_PADDED]], [[P1_PADDED]], [[P2]]),
; CHECK:           custom_call_target="__cublas$lt$matmul",
; CHECK:           backend_config="{
; CHECK-DAG:         \"alpha_real\":1
; CHECK-DAG:         \"alpha_imag\":0
; CHECK-DAG:         \"beta\":0
; CHECK-DAG:         \"dot_dimension_numbers\":{
; CHECK-DAG:           \"lhs_contracting_dimensions\":[\"1\"]
; CHECK-DAG:           \"rhs_contracting_dimensions\":[\"0\"]
; CHECK-DAG:           \"lhs_batch_dimensions\":[]
; CHECK-DAG:           \"rhs_batch_dimensions\":[]
; CHECK-DAG:         }
; CHECK-DAG:         \"precision_config\":{
; CHECK-DAG:           \"operand_precision\":[\"DEFAULT\",\"DEFAULT\"]
; CHECK-DAG:         }
; CHECK-DAG:         \"epilogue\":\"BIAS\"
; CHECK:           }"
; CHECK-NEXT:    [[OUT:%[^ ]+]] = f16[6,6]{1,0} slice([[MATMUL]]), slice={[0:6], [0:6]}
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
}

)";
  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{1e-3, 1e-3}));
  MatchOptimizedHlo(hlo_text,
                    R"(

; CHECK-LABEL: ENTRY %test (x: f16[8,16], y: f16[16,8]) -> f16[8,8] {
; CHECK-NEXT:    [[P0:%[^ ]+]] = f16[8,16]{1,0} parameter(0)
; CHECK-NEXT:    [[P1:%[^ ]+]] = f16[16,8]{1,0} parameter(1)
; CHECK-NEXT:    ROOT [[OUT:%[^ ]+]] = f16[8,8]{1,0} custom-call([[P0]], [[P1]]),
; CHECK:           custom_call_target="__cublas$lt$matmul",
; CHECK:           backend_config="{
; CHECK-DAG:         \"alpha_real\":1
; CHECK-DAG:         \"alpha_imag\":0
; CHECK-DAG:         \"beta\":0
; CHECK-DAG:         \"dot_dimension_numbers\":{
; CHECK-DAG:           \"lhs_contracting_dimensions\":[\"1\"]
; CHECK-DAG:           \"rhs_contracting_dimensions\":[\"0\"]
; CHECK-DAG:           \"lhs_batch_dimensions\":[]
; CHECK-DAG:           \"rhs_batch_dimensions\":[]
; CHECK-DAG:         }
; CHECK-DAG:         \"precision_config\":{
; CHECK-DAG:           \"operand_precision\":[\"DEFAULT\",\"DEFAULT\"]
; CHECK-DAG:         }
; CHECK-DAG:         \"epilogue\":\"RELU\"
; CHECK:           }"
      )");
}

TEST_F(CublasLtGemmRewriteTest, ReluActivationF16Padded) {
  if (!GetCudaComputeCapability().IsAtLeast(se::CudaComputeCapability::VOLTA)) {
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
}

)";
  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{1e-5, 1e-5}));
  MatchOptimizedHlo(hlo_text,
                    R"(

; CHECK-LABEL: ENTRY %test (x: f16[6,12], y: f16[12,6]) -> f16[6,6] {
; CHECK-NEXT:    [[P0:%[^ ]+]] = f16[6,12]{1,0} parameter(0)
; CHECK-NEXT:    [[C0:%[^ ]+]] = f16[] constant(0)
; CHECK-NEXT:    [[P0_PADDED:%[^ ]+]] = f16[8,16]{1,0} pad([[P0]], [[C0]]), padding=0_2x0_4
; CHECK-NEXT:    [[P1:%[^ ]+]] = f16[12,6]{1,0} parameter(1)
; CHECK-NEXT:    [[P1_PADDED:%[^ ]+]] = f16[16,8]{1,0} pad([[P1]], [[C0]]), padding=0_4x0_2
; CHECK-NEXT:    [[MATMUL:%[^ ]+]] = f16[8,8]{1,0} custom-call([[P0_PADDED]], [[P1_PADDED]]),
; CHECK:           custom_call_target="__cublas$lt$matmul",
; CHECK:           backend_config="{
; CHECK-DAG:         \"alpha_real\":1
; CHECK-DAG:         \"alpha_imag\":0
; CHECK-DAG:         \"beta\":0
; CHECK-DAG:         \"dot_dimension_numbers\":{
; CHECK-DAG:           \"lhs_contracting_dimensions\":[\"1\"]
; CHECK-DAG:           \"rhs_contracting_dimensions\":[\"0\"]
; CHECK-DAG:           \"lhs_batch_dimensions\":[]
; CHECK-DAG:           \"rhs_batch_dimensions\":[]
; CHECK-DAG:         }
; CHECK-DAG:         \"precision_config\":{
; CHECK-DAG:           \"operand_precision\":[\"DEFAULT\",\"DEFAULT\"]
; CHECK-DAG:         }
; CHECK-DAG:         \"epilogue\":\"RELU\"
; CHECK:           }"
; CHECK-NEXT:    ROOT [[OUT:%[^ ]+]] = f16[6,6]{1,0} slice([[MATMUL]]), slice={[0:6], [0:6]}
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

; CHECK-LABEL: ENTRY %test (x: f16[8,16], y: f16[16,8], z: f16[8,8]) -> f16[8,8] {
; CHECK-NEXT:    [[P0:%[^ ]+]] = f16[8,16]{1,0} parameter(0)
; CHECK-NEXT:    [[P1:%[^ ]+]] = f16[16,8]{1,0} parameter(1)
; CHECK-NEXT:    [[P2:%[^ ]+]] = f16[8,8]{1,0} parameter(2)
; CHECK-NEXT:    ROOT [[OUT:%[^ ]+]] = f16[8,8]{1,0} custom-call([[P0]], [[P1]], [[P2]]),
; CHECK:           custom_call_target="__cublas$lt$matmul",
; CHECK:           backend_config="{
; CHECK-DAG:         \"alpha_real\":1
; CHECK-DAG:         \"alpha_imag\":0
; CHECK-DAG:         \"beta\":1
; CHECK-DAG:         \"dot_dimension_numbers\":{
; CHECK-DAG:           \"lhs_contracting_dimensions\":[\"1\"]
; CHECK-DAG:           \"rhs_contracting_dimensions\":[\"0\"]
; CHECK-DAG:           \"lhs_batch_dimensions\":[]
; CHECK-DAG:           \"rhs_batch_dimensions\":[]
; CHECK-DAG:         }
; CHECK-DAG:         \"precision_config\":{
; CHECK-DAG:           \"operand_precision\":[\"DEFAULT\",\"DEFAULT\"]
; CHECK-DAG:         }
; CHECK-DAG:         \"epilogue\":\"RELU\"
; CHECK:           }"
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
}

)";
  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{1e-3, 1e-3}));
  MatchOptimizedHlo(hlo_text,
                    R"(

; CHECK-LABEL: ENTRY %test (x: f16[8,16], y: f16[16,8], z: f16[8]) -> f16[8,8] {
; CHECK-NEXT:    [[P0:%[^ ]+]] = f16[8,16]{1,0} parameter(0)
; CHECK-NEXT:    [[P1:%[^ ]+]] = f16[16,8]{1,0} parameter(1)
; CHECK-NEXT:    [[P2:%[^ ]+]] = f16[8]{0} parameter(2)
; CHECK-NEXT:    ROOT [[OUT:%[^ ]+]] = f16[8,8]{1,0} custom-call([[P0]], [[P1]], [[P2]]),
; CHECK:           custom_call_target="__cublas$lt$matmul",
; CHECK:           backend_config="{
; CHECK-DAG:         \"alpha_real\":1
; CHECK-DAG:         \"alpha_imag\":0
; CHECK-DAG:         \"beta\":0
; CHECK-DAG:         \"dot_dimension_numbers\":{
; CHECK-DAG:           \"lhs_contracting_dimensions\":[\"1\"]
; CHECK-DAG:           \"rhs_contracting_dimensions\":[\"0\"]
; CHECK-DAG:           \"lhs_batch_dimensions\":[]
; CHECK-DAG:           \"rhs_batch_dimensions\":[]
; CHECK-DAG:         }
; CHECK-DAG:         \"precision_config\":{
; CHECK-DAG:           \"operand_precision\":[\"DEFAULT\",\"DEFAULT\"]
; CHECK-DAG:         }
; CHECK-DAG:         \"epilogue\":\"BIAS_RELU\"
; CHECK:           }"
      )");
}

TEST_F(CublasLtGemmRewriteTest, VectorBiasReluActivationF16Padded) {
  if (!GetCudaComputeCapability().IsAtLeast(se::CudaComputeCapability::VOLTA)) {
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
}

)";
  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{1e-3, 1e-3}));
  MatchOptimizedHlo(hlo_text,
                    R"(

; CHECK-LABEL: ENTRY %test (x: f16[6,12], y: f16[12,6], z: f16[6]) -> f16[6,6] {
; CHECK-NEXT:    [[P0:%[^ ]+]] = f16[6,12]{1,0} parameter(0)
; CHECK-NEXT:    [[C0:%[^ ]+]] = f16[] constant(0)
; CHECK-NEXT:    [[P0_PADDED:%[^ ]+]] = f16[8,16]{1,0} pad([[P0]], [[C0]]), padding=0_2x0_4
; CHECK-NEXT:    [[P1:%[^ ]+]] = f16[12,6]{1,0} parameter(1)
; CHECK-NEXT:    [[P1_PADDED:%[^ ]+]] = f16[16,8]{1,0} pad([[P1]], [[C0]]), padding=0_4x0_2
; CHECK-NEXT:    [[P2:%[^ ]+]] = f16[6]{0} parameter(2)
; CHECK-NEXT:    [[MATMUL:%[^ ]+]] = f16[8,8]{1,0} custom-call([[P0_PADDED]], [[P1_PADDED]], [[P2]]),
; CHECK:           custom_call_target="__cublas$lt$matmul",
; CHECK:           backend_config="{
; CHECK-DAG:         \"alpha_real\":1
; CHECK-DAG:         \"alpha_imag\":0
; CHECK-DAG:         "beta\":0
; CHECK-DAG:         \"dot_dimension_numbers\":{
; CHECK-DAG:           \"lhs_contracting_dimensions\":[\"1\"]
; CHECK-DAG:           \"rhs_contracting_dimensions\":[\"0\"]
; CHECK-DAG:           \"lhs_batch_dimensions\":[]
; CHECK-DAG:           \"rhs_batch_dimensions\":[]
; CHECK-DAG:         }
; CHECK-DAG:         \"precision_config\":{
; CHECK-DAG:           \"operand_precision\":[\"DEFAULT\",\"DEFAULT\"]
; CHECK-DAG:         }
; CHECK-DAG:         \"epilogue\":\"BIAS_RELU\"
; CHECK:           }"
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
  MatchOptimizedHlo(hlo_text,
                    R"(

; CHECK-LABEL: ENTRY %test (x: bf16[8,16], y: bf16[16,8], z: bf16[8,8]) -> bf16[8,8] {
; CHECK-DAG:     [[P0:%[^ ]+]] = bf16[8,16]{1,0} parameter(0)
; CHECK-DAG:     [[P1:%[^ ]+]] = bf16[16,8]{1,0} parameter(1)
; CHECK-DAG:     [[P2:%[^ ]+]] = bf16[8,8]{1,0} parameter(2)
; CHECK-NEXT:    ROOT [[OUT:%[^ ]+]] = bf16[8,8]{1,0} custom-call([[P0]], [[P1]], [[P2]]),
; CHECK:           custom_call_target="__cublas$lt$matmul",
; CHECK:           backend_config="{
; CHECK-DAG:         \"alpha_real\":1
; CHECK-DAG:         \"alpha_imag\":0
; CHECK-DAG:         \"beta\":1
; CHECK-DAG:         \"dot_dimension_numbers\":{
; CHECK-DAG:           \"lhs_contracting_dimensions\":[\"1\"]
; CHECK-DAG:           \"rhs_contracting_dimensions\":[\"0\"]
; CHECK-DAG:           \"lhs_batch_dimensions\":[]
; CHECK-DAG:           \"rhs_batch_dimensions\":[]
; CHECK-DAG:         }
; CHECK-DAG:         \"precision_config\":{
; CHECK-DAG:           \"operand_precision\":[\"DEFAULT\",\"DEFAULT\"]
; CHECK-DAG:         }
; CHECK-DAG:         \"epilogue\":\"DEFAULT\"
; CHECK:           }"
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
  GemmRewriter pass(GetCudaComputeCapability());
  TF_ASSERT_OK_AND_ASSIGN(bool changed, this->RunHloPass(&pass, module.get()));
  EXPECT_TRUE(changed);

  EXPECT_THAT(
      module->entry_computation()->root_instruction(),
      GmockMatch(
          m::Bitcast(m::CustomCall(
                         {"__cublas$lt$matmul"},
                         m::Parameter(0).WithShape(BF16, {8, 16}),
                         m::Parameter(1).WithShape(BF16, {16, 8}),
                         m::Bitcast(m::Parameter(2)).WithShape(BF16, {8, 8})))
              .WithShape(BF16, {2, 4, 8})));
}

// For bfloat16, the operands are padded if necessary on Ampere and newer
// architectures so that the sizes of all dimensions are multiples of 8.
TEST_F(CublasLtGemmRewriteTest, VectorBiasBF16Unpadded) {
  const char* hlo_text = R"(
HloModule test

ENTRY test {
  x = bf16[8,16] parameter(0)
  y = bf16[16,8] parameter(1)
  z = bf16[8] parameter(2)
  dot_a = bf16[8,8] dot(x, y), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  z_bcast = bf16[8,8] broadcast(z), dimensions={1}
  ROOT add = bf16[8,8] add(dot_a, z_bcast)
}

)";
  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{8e-3, 2e-3}));
  MatchOptimizedHlo(hlo_text,
                    R"(

; CHECK-LABEL: ENTRY %test (x: bf16[8,16], y: bf16[16,8], z: bf16[8]) -> bf16[8,8] {
; CHECK-DAG:     [[P0:%[^ ]+]] = bf16[8,16]{1,0} parameter(0)
; CHECK-DAG:     [[P1:%[^ ]+]] = bf16[16,8]{1,0} parameter(1)
; CHECK-DAG:     [[P2:%[^ ]+]] = bf16[8]{0} parameter(2)
; CHECK-NEXT:    ROOT [[OUT:%[^ ]+]] = bf16[8,8]{1,0} custom-call([[P0]], [[P1]], [[P2]]),
; CHECK:           custom_call_target="__cublas$lt$matmul",
; CHECK:           backend_config="{
; CHECK-DAG:         \"alpha_real\":1
; CHECK-DAG:         \"alpha_imag\":0
; CHECK-DAG:         \"beta\":0
; CHECK-DAG:         \"dot_dimension_numbers\":{
; CHECK-DAG:           \"lhs_contracting_dimensions\":[\"1\"]
; CHECK-DAG:           \"rhs_contracting_dimensions\":[\"0\"]
; CHECK-DAG:           \"lhs_batch_dimensions\":[]
; CHECK-DAG:           \"rhs_batch_dimensions\":[]
; CHECK-DAG:         }
; CHECK-DAG:         \"precision_config\":{
; CHECK-DAG:           \"operand_precision\":[\"DEFAULT\",\"DEFAULT\"]
; CHECK-DAG:         }
; CHECK-DAG:         \"epilogue\":\"BIAS\"
; CHECK:           }"
      )");
}

TEST_F(CublasLtGemmRewriteTest, VectorBiasBF16Padded) {
  if (!GetCudaComputeCapability().IsAtLeast(
          se::CudaComputeCapability::AMPERE)) {
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
}

)";
  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{1e-3, 1e-3}));
  MatchOptimizedHlo(hlo_text,
                    R"(

; CHECK-LABEL: ENTRY %test (x: bf16[6,12], y: bf16[12,6], z: bf16[6]) -> bf16[6,6] {
; CHECK-DAG:     [[P0:%[^ ]+]] = bf16[6,12]{1,0} parameter(0)
; CHECK-DAG:     [[C0:%[^ ]+]] = bf16[] constant(0)
; CHECK-DAG:     [[P0_PADDED:%[^ ]+]] = bf16[8,16]{1,0} pad([[P0]], [[C0]]), padding=0_2x0_4
; CHECK-DAG:     [[P1:%[^ ]+]] = bf16[12,6]{1,0} parameter(1)
; CHECK-DAG:     [[P1_PADDED:%[^ ]+]] = bf16[16,8]{1,0} pad([[P1]], [[C0]]), padding=0_4x0_2
; CHECK-DAG:     [[P2:%[^ ]+]] = bf16[6]{0} parameter(2)
; CHECK-NEXT:    [[MATMUL:%[^ ]+]] = bf16[8,8]{1,0} custom-call([[P0_PADDED]], [[P1_PADDED]], [[P2]]),
; CHECK:           custom_call_target="__cublas$lt$matmul",
; CHECK:           backend_config="{
; CHECK-DAG:         \"alpha_real\":1
; CHECK-DAG:         \"alpha_imag\":0
; CHECK-DAG:         \"beta\":0
; CHECK-DAG:         \"dot_dimension_numbers\":{
; CHECK-DAG:           \"lhs_contracting_dimensions\":[\"1\"]
; CHECK-DAG:           \"rhs_contracting_dimensions\":[\"0\"]
; CHECK-DAG:           \"lhs_batch_dimensions\":[]
; CHECK-DAG:           \"rhs_batch_dimensions\":[]
; CHECK-DAG:         }
; CHECK-DAG:         \"precision_config\":{
; CHECK-DAG:           \"operand_precision\":[\"DEFAULT\",\"DEFAULT\"]
; CHECK-DAG:         }
; CHECK-DAG:         \"epilogue\":\"BIAS\"
; CHECK:           }"
; CHECK-NEXT:    [[OUT:%[^ ]+]] = bf16[6,6]{1,0} slice([[MATMUL]]), slice={[0:6], [0:6]}
      )");
}

// For bfloat16, the operands are padded if necessary on Ampere and newer
// architectures so that the sizes of all dimensions are multiples of 8.
TEST_F(CublasLtGemmRewriteTest, ReluActivationBF16Unpadded) {
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
  MatchOptimizedHlo(hlo_text,
                    R"(

; CHECK-LABEL: ENTRY %test (x: bf16[8,16], y: bf16[16,8]) -> bf16[8,8] {
; CHECK-DAG:     [[P0:%[^ ]+]] = bf16[8,16]{1,0} parameter(0)
; CHECK-DAG:     [[P1:%[^ ]+]] = bf16[16,8]{1,0} parameter(1)
; CHECK-NEXT:    ROOT [[OUT:%[^ ]+]] = bf16[8,8]{1,0} custom-call([[P0]], [[P1]]),
; CHECK:           custom_call_target="__cublas$lt$matmul",
; CHECK:           backend_config="{
; CHECK-DAG:         \"alpha_real\":1
; CHECK-DAG:         \"alpha_imag\":0
; CHECK-DAG:         \"beta\":0
; CHECK-DAG:         \"dot_dimension_numbers\":{
; CHECK-DAG:           \"lhs_contracting_dimensions\":[\"1\"]
; CHECK-DAG:           \"rhs_contracting_dimensions\":[\"0\"]
; CHECK-DAG:           \"lhs_batch_dimensions\":[]
; CHECK-DAG:           \"rhs_batch_dimensions\":[]
; CHECK-DAG:         }
; CHECK-DAG:         \"precision_config\":{
; CHECK-DAG:           \"operand_precision\":[\"DEFAULT\",\"DEFAULT\"]
; CHECK-DAG:         }
; CHECK-DAG:         \"epilogue\":\"RELU\"
; CHECK:           }"
      )");
}

TEST_F(CublasLtGemmRewriteTest, ReluActivationBF16Padded) {
  if (!GetCudaComputeCapability().IsAtLeast(
          se::CudaComputeCapability::AMPERE)) {
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
}

)";
  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{1e-5, 1e-5}));
  MatchOptimizedHlo(hlo_text,
                    R"(

; CHECK-LABEL: ENTRY %test (x: bf16[6,12], y: bf16[12,6]) -> bf16[6,6] {
; CHECK-DAG:     [[P0:%[^ ]+]] = bf16[6,12]{1,0} parameter(0)
; CHECK-DAG:     [[C0:%[^ ]+]] = bf16[] constant(0)
; CHECK-DAG:     [[P0_PADDED:%[^ ]+]] = bf16[8,16]{1,0} pad([[P0]], [[C0]]), padding=0_2x0_4
; CHECK-DAG:     [[P1:%[^ ]+]] = bf16[12,6]{1,0} parameter(1)
; CHECK-DAG:     [[P1_PADDED:%[^ ]+]] = bf16[16,8]{1,0} pad([[P1]], [[C0]]), padding=0_4x0_2
; CHECK-NEXT:    [[MATMUL:%[^ ]+]] = bf16[8,8]{1,0} custom-call([[P0_PADDED]], [[P1_PADDED]]),
; CHECK:           custom_call_target="__cublas$lt$matmul",
; CHECK:           backend_config="{
; CHECK-DAG:         \"alpha_real\":1
; CHECK-DAG:         \"alpha_imag\":0
; CHECK-DAG:         \"beta\":0
; CHECK-DAG:         \"dot_dimension_numbers\":{
; CHECK-DAG:           \"lhs_contracting_dimensions\":[\"1\"]
; CHECK-DAG:           \"rhs_contracting_dimensions\":[\"0\"]
; CHECK-DAG:           \"lhs_batch_dimensions\":[]
; CHECK-DAG:           \"rhs_batch_dimensions\":[]
; CHECK-DAG:         }
; CHECK-DAG:         \"precision_config\":{
; CHECK-DAG:           \"operand_precision\":[\"DEFAULT\",\"DEFAULT\"]
; CHECK-DAG:         }
; CHECK-DAG:         \"epilogue\":\"RELU\"
; CHECK:           }"
; CHECK-NEXT:    ROOT [[OUT:%[^ ]+]] = bf16[6,6]{1,0} slice([[MATMUL]]), slice={[0:6], [0:6]}
      )");
}

// For bfloat16, the operands are padded if necessary on Ampere and newer
// architectures so that the sizes of all dimensions are multiples of 8.
TEST_F(CublasLtGemmRewriteTest, VectorBiasReluActivationBF16Unpadded) {
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
}

)";
  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{8e-3, 2e-3}));
  MatchOptimizedHlo(hlo_text,
                    R"(

; CHECK-LABEL: ENTRY %test (x: bf16[8,16], y: bf16[16,8], z: bf16[8]) -> bf16[8,8] {
; CHECK-DAG:     [[P0:%[^ ]+]] = bf16[8,16]{1,0} parameter(0)
; CHECK-DAG:     [[P1:%[^ ]+]] = bf16[16,8]{1,0} parameter(1)
; CHECK-DAG:     [[P2:%[^ ]+]] = bf16[8]{0} parameter(2)
; CHECK-NEXT:    ROOT [[OUT:%[^ ]+]] = bf16[8,8]{1,0} custom-call([[P0]], [[P1]], [[P2]]),
; CHECK:           custom_call_target="__cublas$lt$matmul",
; CHECK:           backend_config="{
; CHECK-DAG:         \"alpha_real\":1
; CHECK-DAG:         \"alpha_imag\":0
; CHECK-DAG:         \"beta\":0
; CHECK-DAG:         \"dot_dimension_numbers\":{
; CHECK-DAG:           \"lhs_contracting_dimensions\":[\"1\"]
; CHECK-DAG:           \"rhs_contracting_dimensions\":[\"0\"]
; CHECK-DAG:           \"lhs_batch_dimensions\":[]
; CHECK-DAG:           \"rhs_batch_dimensions\":[]
; CHECK-DAG:         }
; CHECK-DAG:         \"precision_config\":{
; CHECK-DAG:           \"operand_precision\":[\"DEFAULT\",\"DEFAULT\"]
; CHECK-DAG:         }
; CHECK-DAG:         \"epilogue\":\"BIAS_RELU\"
; CHECK:           }"
      )");
}

TEST_F(CublasLtGemmRewriteTest, VectorBiasReluActivationBF16Padded) {
  if (!GetCudaComputeCapability().IsAtLeast(
          se::CudaComputeCapability::AMPERE)) {
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
  MatchOptimizedHlo(hlo_text,
                    R"(

; CHECK-LABEL: ENTRY %test (x: bf16[6,12], y: bf16[12,6], z: bf16[6]) -> bf16[6,6] {
; CHECK-DAG:     [[P0:%[^ ]+]] = bf16[6,12]{1,0} parameter(0)
; CHECK-DAG:     [[C0:%[^ ]+]] = bf16[] constant(0)
; CHECK-DAG:     [[P0_PADDED:%[^ ]+]] = bf16[8,16]{1,0} pad([[P0]], [[C0]]), padding=0_2x0_4
; CHECK-DAG:     [[P1:%[^ ]+]] = bf16[12,6]{1,0} parameter(1)
; CHECK-DAG:     [[P1_PADDED:%[^ ]+]] = bf16[16,8]{1,0} pad([[P1]], [[C0]]), padding=0_4x0_2
; CHECK-DAG:     [[P2:%[^ ]+]] = bf16[6]{0} parameter(2)
; CHECK-NEXT:    [[MATMUL:%[^ ]+]] = bf16[8,8]{1,0} custom-call([[P0_PADDED]], [[P1_PADDED]], [[P2]]),
; CHECK:           custom_call_target="__cublas$lt$matmul",
; CHECK:           backend_config="{
; CHECK-DAG:         \"alpha_real\":1
; CHECK-DAG:         \"alpha_imag\":0
; CHECK-DAG:         \"beta\":0
; CHECK-DAG:         \"dot_dimension_numbers\":{
; CHECK-DAG:           \"lhs_contracting_dimensions\":[\"1\"]
; CHECK-DAG:           \"rhs_contracting_dimensions\":[\"0\"]
; CHECK-DAG:           \"lhs_batch_dimensions\":[]
; CHECK-DAG:           \"rhs_batch_dimensions\":[]
; CHECK-DAG:         }
; CHECK-DAG:         \"precision_config\":{
; CHECK-DAG:           \"operand_precision\":[\"DEFAULT\",\"DEFAULT\"]
; CHECK-DAG:         }
; CHECK-DAG:         \"epilogue\":\"BIAS_RELU\"
; CHECK:           }"
; CHECK-NEXT:    ROOT [[OUT:%[^ ]+]] = bf16[6,6]{1,0} slice([[MATMUL]]), slice={[0:6], [0:6]}
      )");
}

TEST_F(CublasLtGemmRewriteTest, VectorBiasReluActivationF64) {
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

; CHECK-LABEL: ENTRY %test (x: f64[2,3], y: f64[3,4], z: f64[4]) -> f64[2,4] {
; CHECK-NEXT:    [[P0:%[^ ]+]] = f64[2,3]{1,0} parameter(0)
; CHECK-NEXT:    [[P1:%[^ ]+]] = f64[3,4]{1,0} parameter(1)
; CHECK-NEXT:    [[P2:%[^ ]+]] = f64[4]{0} parameter(2)
; CHECK-NEXT:    ROOT [[OUT:%[^ ]+]] = f64[2,4]{1,0} custom-call([[P0]], [[P1]], [[P2]]),
; CHECK:           custom_call_target="__cublas$lt$matmul",
; CHECK:           backend_config="{
; CHECK-DAG:         \"alpha_real\":1
; CHECK-DAG:         \"alpha_imag\":0
; CHECK-DAG:         \"beta\":0
; CHECK-DAG:         \"dot_dimension_numbers\":{
; CHECK-DAG:           \"lhs_contracting_dimensions\":[\"1\"]
; CHECK-DAG:           \"rhs_contracting_dimensions\":[\"0\"]
; CHECK-DAG:           \"lhs_batch_dimensions\":[]
; CHECK-DAG:           \"rhs_batch_dimensions\":[]
; CHECK-DAG:         }
; CHECK-DAG:         \"precision_config\":{
; CHECK-DAG:           \"operand_precision\":[\"DEFAULT\",\"DEFAULT\"]
; CHECK-DAG:         }
; CHECK-DAG:         \"epilogue\":\"BIAS_RELU\"
; CHECK:           }"
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
  dot_a = f32[2,4] dot(x, y), lhs_contracting_dims={1}, rhs_contracting_dims={0}
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

; CHECK-LABEL: ENTRY %test (x: f32[2,3], y: f32[3,4], z: f32[4]) -> f32[2,4] {
; CHECK-NEXT:    [[P0:%[^ ]+]] = f32[2,3]{1,0} parameter(0)
; CHECK-NEXT:    [[P1:%[^ ]+]] = f32[3,4]{1,0} parameter(1)
; CHECK-NEXT:    [[P2:%[^ ]+]] = f32[4]{0} parameter(2)
; CHECK-NEXT:    ROOT [[OUT:%[^ ]+]] = f32[2,4]{1,0} custom-call([[P0]], [[P1]], [[P2]]),
; CHECK:           custom_call_target="__cublas$lt$matmul",
; CHECK:           backend_config="{
; CHECK-DAG:         \"alpha_real\":3
; CHECK-DAG:         \"alpha_imag\":0
; CHECK-DAG:         \"beta\":0
; CHECK-DAG:         \"dot_dimension_numbers\":{
; CHECK-DAG:           \"lhs_contracting_dimensions\":[\"1\"]
; CHECK-DAG:           \"rhs_contracting_dimensions\":[\"0\"]
; CHECK-DAG:           \"lhs_batch_dimensions\":[]
; CHECK-DAG:           \"rhs_batch_dimensions\":[]
; CHECK-DAG:         }
; CHECK-DAG:         \"precision_config\":{
; CHECK-DAG:           \"operand_precision\":[\"DEFAULT\",\"DEFAULT\"]
; CHECK-DAG:         }
; CHECK-DAG:         \"epilogue\":\"BIAS_RELU\"
; CHECK:           }"
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
  GemmRewriter pass(GetCudaComputeCapability());
  TF_ASSERT_OK_AND_ASSIGN(bool changed, this->RunHloPass(&pass, module.get()));
  SCOPED_TRACE(module->ToString());
  EXPECT_TRUE(changed);

  EXPECT_THAT(
      module->entry_computation()->root_instruction(),
      GmockMatch(m::Tuple(
          m::CustomCall(m::Parameter(0), m::Parameter(1), m::Parameter()),
          m::CustomCall(m::Parameter(0), m::Parameter(1), m::Constant()),
          m::CustomCall(m::Parameter(0), m::Parameter(1), m::Constant()),
          m::CustomCall(m::Parameter(0), m::Parameter(1), m::Constant()))));
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

class ParameterizedFp8GemmRewriteTest : public ParameterizedGemmRewriteTest {
 public:
  DebugOptions GetDebugOptionsForTest() override {
    DebugOptions debug_options =
        ParameterizedGemmRewriteTest::GetDebugOptionsForTest();
    debug_options.set_xla_gpu_enable_xla_runtime_executable(false);
    return debug_options;
  }

 protected:
  void CheckFp8IfOnHopper(absl::string_view hlo_text,
                          ErrorSpec error_spec = ErrorSpec{1e-2, 1e-2}) {
    if (!GetCudaComputeCapability().IsAtLeast(
            se::CudaComputeCapability::HOPPER)) {
      return;
    }
    EXPECT_TRUE(RunAndCompare(hlo_text, error_spec));

    // Most FP8 tests directly create a GemmRewriter and check the output.
    // Here, also run the entire HLO pass pipeline to ensure no other passes
    // interfere with GemmRewriter's pattern matching.
    TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> optimized_module,
                            GetOptimizedModule(hlo_text));
    const HloInstruction* call =
        FindInstruction(optimized_module.get(), HloOpcode::kCustomCall);
    ASSERT_NE(call, nullptr);
    EXPECT_EQ(call->custom_call_target(), "__cublas$lt$matmul$f8");
  }
};

TEST_P(ParameterizedFp8GemmRewriteTest, DoNotRewriteToF8OnPreHopper) {
  if (GetCudaComputeCapability().IsAtLeast(se::CudaComputeCapability::HOPPER)) {
    GTEST_SKIP() << "Test requires a pre-Hopper GPU.";
  }
  const char* hlo_text = R"(
    HloModule test

    ENTRY PreHopperTest {
      x = f8e4m3fn[16,32] parameter(0)
      y = f8e4m3fn[32,16] parameter(1)
      ROOT out = f8e4m3fn[16,16] dot(x, y), lhs_contracting_dims={1}, rhs_contracting_dims={0}
          }

)";

  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{1e-2, 1e-2}));
  MatchOptimizedHlo(hlo_text,
                    R"(
; CHECK-LABEL: ENTRY %PreHopperTest (x: f8e4m3fn[16,32], y: f8e4m3fn[32,16]) -> f8e4m3fn[16,16] {
; CHECK:    {{.*}} = f16[16,16]{1,0} custom-call({{.*}}, {{.*}})
; CHECK-DAG:  custom_call_target="<<CUBLAS_CUSTOM_CALL_TARGET_PLACEHOLDER>>"
          )");
}

TEST_P(ParameterizedFp8GemmRewriteTest, UnsupportedTypesF8) {
  // Test with types unsupported by cuBLAS LT when FP8 is used. cuBLAS LT with
  // FP8 requires one of the operands to be F8E4M3FN.
  const char* hlo_text = R"(
    HloModule test

    ENTRY unsupported_types {
      x = f8e5m2[16,16] parameter(0)
      y = f8e5m2[16,16] parameter(1)
      ROOT out = f8e5m2[16,16] dot(x, y), lhs_contracting_dims={1}, rhs_contracting_dims={0}
          }
)";
  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{1e-2, 1e-2}));
  RunAndFilecheckHloRewrite(hlo_text, GemmRewriter(GetCudaComputeCapability()),
                            absl::StrReplaceAll(R"(
; CHECK-LABEL: ENTRY %unsupported_types (x: f8e5m2[16,16], y: f8e5m2[16,16]) -> f8e5m2[16,16] {
; CHECK-NEXT:    [[P0:%[^ ]+]] = f8e5m2[16,16]{1,0} parameter(0)
; CHECK-NEXT:    [[P0_CONVERT:%[^ ]+]] = f16[16,16]{1,0} convert([[P0]])
; CHECK-NEXT:    [[P1:%[^ ]+]] = f8e5m2[16,16]{1,0} parameter(1)
; CHECK-NEXT:    [[P1_CONVERT:%[^ ]+]] = f16[16,16]{1,0} convert([[P1]])
; CHECK-NEXT:    [[DOT:%[^ ]+]] = f16[16,16]{1,0} custom-call([[P0_CONVERT]], [[P1_CONVERT]]),
; CHECK:           custom_call_target="<<CUBLAS_CUSTOM_CALL_TARGET_PLACEHOLDER>>",
; CHECK:           backend_config="{
; CHECK-DAG:         \"alpha_real\":1
; CHECK-DAG:         \"alpha_imag\":0
; CHECK-DAG:         \"beta\":0
; CHECK-DAG:         \"dot_dimension_numbers\":{
; CHECK-DAG:           \"lhs_contracting_dimensions\":[\"1\"]
; CHECK-DAG:           \"rhs_contracting_dimensions\":[\"0\"]
; CHECK-DAG:           \"lhs_batch_dimensions\":[]
; CHECK-DAG:           \"rhs_batch_dimensions\":[]
; CHECK-DAG:         }
; CHECK-DAG:         \"precision_config\":{
; CHECK-DAG:           \"operand_precision\":[\"DEFAULT\",\"DEFAULT\"]
; CHECK-DAG:         }
; CHECK-DAG:         \"epilogue\":\"DEFAULT\"
; CHECK:           }"
; CHECK-NEXT:    ROOT [[OUT:%[^ ]+]] = f8e5m2[16,16]{1,0} convert([[DOT]])
      )",
                                                replacements_));
}

TEST_P(ParameterizedFp8GemmRewriteTest, UnscaledABUnscaledDF8) {
  const char* hlo_text = R"(
    HloModule test

    ENTRY test {
      x = f8e4m3fn[16,32] parameter(0)
      y = f8e4m3fn[32,16] parameter(1)
      ROOT out = f8e4m3fn[16,16] dot(x, y), lhs_contracting_dims={1}, rhs_contracting_dims={0}
          }

)";

  CheckFp8IfOnHopper(hlo_text);
  RunAndFilecheckHloRewrite(hlo_text,
                            GemmRewriter(se::CudaComputeCapability{
                                se::CudaComputeCapability::HOPPER, 0}),
                            R"(
; CHECK-LABEL: ENTRY %test (x: f8e4m3fn[16,32], y: f8e4m3fn[32,16]) -> f8e4m3fn[16,16] {
; CHECK-NEXT:    [[P0:%[^ ]+]] = f8e4m3fn[16,32]{1,0} parameter(0)
; CHECK-NEXT:    [[P1:%[^ ]+]] = f8e4m3fn[32,16]{1,0} parameter(1)
; CHECK-NEXT:    [[P1_TRANSPOSE:%[^ ]+]] = f8e4m3fn[16,32]{1,0} transpose([[P1]]), dimensions={1,0}
; CHECK-NEXT:    [[C0:[^ ]+]] = bf16[] constant(0)
; CHECK-NEXT:    [[C0_BCAST:[^ ]+]] = bf16[16,16]{1,0} broadcast([[C0]]), dimensions={}
; CHECK-NEXT:    [[C1:[^ ]+]] = f32[] constant(1)
; CHECK-NEXT:    ROOT [[OUT:%[^ ]+]] = f8e4m3fn[16,16]{1,0} custom-call([[P0]], [[P1_TRANSPOSE]], [[C0_BCAST]], [[C1]], [[C1]], /*index=5*/[[C1]], [[C1]]),
; CHECK:           custom_call_target="__cublas$lt$matmul$f8",
; CHECK:           backend_config="{
; CHECK-DAG:         \"alpha_real\":1
; CHECK-DAG:         \"alpha_imag\":0
; CHECK-DAG:         \"beta\":0
; CHECK-DAG:         \"dot_dimension_numbers\":{
; CHECK-DAG:           \"lhs_contracting_dimensions\":[\"1\"]
; CHECK-DAG:           \"rhs_contracting_dimensions\":[\"1\"]
; CHECK-DAG:           \"lhs_batch_dimensions\":[]
; CHECK-DAG:           \"rhs_batch_dimensions\":[]
; CHECK-DAG:         }
; CHECK-DAG:         \"precision_config\":{
; CHECK-DAG:           \"operand_precision\":[\"DEFAULT\",\"DEFAULT\"]
; CHECK-DAG:         }
; CHECK-DAG:         \"epilogue\":\"DEFAULT\"
; CHECK:           }"
      )");
}

TEST_P(ParameterizedFp8GemmRewriteTest, ScaledABUnscaledDF8) {
  const char* hlo_text = R"(
    HloModule test

    ENTRY test {
      x = f8e4m3fn[16,32] parameter(0)
      y = f8e4m3fn[32,16] parameter(1)
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

  CheckFp8IfOnHopper(hlo_text);
  RunAndFilecheckHloRewrite(hlo_text,
                            GemmRewriter(se::CudaComputeCapability{
                                se::CudaComputeCapability::HOPPER, 0}),
                            R"(
; CHECK-LABEL: ENTRY %test (x: f8e4m3fn[16,32], y: f8e4m3fn[32,16], x_scale: f32[], y_scale: f32[]) -> f32[16,16] {
; CHECK-NEXT:    [[P0:%[^ ]+]] = f8e4m3fn[16,32]{1,0} parameter(0)
; CHECK-NEXT:    [[P1:%[^ ]+]] = f8e4m3fn[32,16]{1,0} parameter(1)
; CHECK-NEXT:    [[P1_TRANSPOSE:%[^ ]+]] = f8e4m3fn[16,32]{1,0} transpose([[P1]]), dimensions={1,0}
; CHECK-NEXT:    [[C0:%[^ ]+]] = f32[] constant(0)
; CHECK-NEXT:    [[C0_BCAST:%[^ ]+]] = f32[16,16]{1,0} broadcast([[C0]]), dimensions={}
; CHECK-NEXT:    [[P2:%[^ ]+]] = f32[] parameter(2)
; CHECK-NEXT:    [[P3:%[^ ]+]] = f32[] parameter(3)
; CHECK-NEXT:    [[C1:%[^ ]+]] = f32[] constant(1)
; CHECK-NEXT:    ROOT [[OUT:%[^ ]+]] = f32[16,16]{1,0} custom-call([[P0]], [[P1_TRANSPOSE]], [[C0_BCAST]], [[P2]], [[P3]], /*index=5*/[[C1]], [[C1]]),
; CHECK:           custom_call_target="__cublas$lt$matmul$f8",
; CHECK:           backend_config="{
; CHECK-DAG:         \"alpha_real\":1
; CHECK-DAG:         \"alpha_imag\":0
; CHECK-DAG:         \"beta\":0
; CHECK-DAG:         \"dot_dimension_numbers\":{
; CHECK-DAG:           \"lhs_contracting_dimensions\":[\"1\"]
; CHECK-DAG:           \"rhs_contracting_dimensions\":[\"1\"]
; CHECK-DAG:           \"lhs_batch_dimensions\":[]
; CHECK-DAG:           \"rhs_batch_dimensions\":[]
; CHECK-DAG:         }
; CHECK-DAG:         \"precision_config\":{
; CHECK-DAG:           \"operand_precision\":[\"DEFAULT\",\"DEFAULT\"]
; CHECK-DAG:         }
; CHECK-DAG:         \"epilogue\":\"DEFAULT\"
; CHECK:           }"
      )");
}

TEST_P(ParameterizedFp8GemmRewriteTest, ScaledABUnscaledDPaddedF8) {
  const char* hlo_text = R"(
    HloModule test

    ENTRY test {
      x = f8e4m3fn[13,17] parameter(0)
      y = f8e4m3fn[17,31] parameter(1)
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

  CheckFp8IfOnHopper(hlo_text);
  RunAndFilecheckHloRewrite(hlo_text,
                            GemmRewriter(se::CudaComputeCapability{
                                se::CudaComputeCapability::HOPPER, 0}),
                            R"(
; CHECK-LABEL: ENTRY %test (x: f8e4m3fn[13,17], y: f8e4m3fn[17,31], x_scale: f32[], y_scale: f32[]) -> f32[13,31] {
; CHECK-NEXT:    [[P0:%[^ ]+]] = f8e4m3fn[13,17]{1,0} parameter(0)
; CHECK-NEXT:    [[C0:%[^ ]+]] = f8e4m3fn[] constant(0)
; CHECK-NEXT:    [[P0_PADDED:%[^ ]+]] = f8e4m3fn[16,32]{1,0} pad([[P0]], [[C0]]), padding=0_3x0_15
; CHECK-NEXT:    [[P1:%[^ ]+]] = f8e4m3fn[17,31]{1,0} parameter(1)
; CHECK-NEXT:    [[P1_TRANSPOSE:%[^ ]+]] = f8e4m3fn[31,17]{1,0} transpose([[P1]]), dimensions={1,0}
; CHECK-NEXT:    [[C1:%[^ ]+]] = f8e4m3fn[] constant(0)
; CHECK-NEXT:    [[P1_TRANSPOSE_PADDED:%[^ ]+]] = f8e4m3fn[32,32]{1,0} pad([[P1_TRANSPOSE]], [[C1]])
; CHECK-NEXT:    [[C2:%[^ ]+]] = f32[] constant(0)
; CHECK-NEXT:    [[C2_BCAST:%[^ ]+]] = f32[13,31]{1,0} broadcast([[C2]]), dimensions={}
; CHECK-NEXT:    [[C3:%[^ ]+]] = f32[] constant(0)
; CHECK-NEXT:    [[C2_BCAST_PADDED:%[^ ]+]] = f32[16,32]{1,0} pad([[C2_BCAST]], [[C3]]), padding=0_3x0_1
; CHECK-NEXT:    [[P2:%[^ ]+]] = f32[] parameter(2)
; CHECK-NEXT:    [[P3:%[^ ]+]] = f32[] parameter(3)
; CHECK-NEXT:    [[C4:%[^ ]+]] = f32[] constant(1)
; CHECK-NEXT:    [[DOT:%[^ ]+]] = f32[16,32]{1,0} custom-call([[P0_PADDED]], [[P1_TRANSPOSE_PADDED]], [[C2_BCAST_PADDED]], [[P2]], [[P3]], /*index=5*/[[C4]], [[C4]]),
; CHECK:           custom_call_target="__cublas$lt$matmul$f8",
; CHECK:           backend_config="{
; CHECK-DAG:         \"alpha_real\":1
; CHECK-DAG:         \"alpha_imag\":0
; CHECK-DAG:         \"beta\":0
; CHECK-DAG:         \"dot_dimension_numbers\":{
; CHECK-DAG:           \"lhs_contracting_dimensions\":[\"1\"]
; CHECK-DAG:           \"rhs_contracting_dimensions\":[\"1\"]
; CHECK-DAG:           \"lhs_batch_dimensions\":[]
; CHECK-DAG:           \"rhs_batch_dimensions\":[]
; CHECK-DAG:         }
; CHECK-DAG:         \"precision_config\":{
; CHECK-DAG:           \"operand_precision\":[\"DEFAULT\",\"DEFAULT\"]
; CHECK-DAG:         }
; CHECK-DAG:         \"epilogue\":\"DEFAULT\"
; CHECK:           }"
; CHECK-NEXT: ROOT [[OUT:%[^ ]+]] = f32[13,31]{1,0} slice([[DOT]]), slice={[0:13], [0:31]}
      )");
}

TEST_P(ParameterizedFp8GemmRewriteTest, ScaledABUnscaledDBitcastF8) {
  const char* hlo_text = R"(
    HloModule test

    ENTRY test {
      x = f8e4m3fn[2,8,16] parameter(0)
      y = f8e4m3fn[16,16] parameter(1)
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
  GemmRewriter pass(
      se::CudaComputeCapability{se::CudaComputeCapability::HOPPER, 0});
  TF_ASSERT_OK_AND_ASSIGN(bool changed, this->RunHloPass(&pass, module.get()));
  EXPECT_TRUE(changed);

  EXPECT_THAT(
      module->entry_computation()->root_instruction(),
      GmockMatch(
          m::CustomCall({"__cublas$lt$matmul$f8"}).WithShape(F32, {16, 16})));
}

TEST_P(ParameterizedFp8GemmRewriteTest, ScaledABUnscaledDUnaryOpsF8) {
  const char* hlo_text = R"(
    HloModule test

    ENTRY test {
      x = f8e4m3fn[3] parameter(0)
      y = f8e4m3fn[32,16] parameter(1)
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
  CheckFp8IfOnHopper(hlo_text);
  RunAndFilecheckHloRewrite(hlo_text,
                            GemmRewriter(se::CudaComputeCapability{
                                se::CudaComputeCapability::HOPPER, 0}),
                            R"(

; CHECK-LABEL: ENTRY %test (x: f8e4m3fn[3], y: f8e4m3fn[32,16], x_scale: f32[], y_scale: f32[]) -> f32[16,16] {
; CHECK-NEXT:    [[P0:%[^ ]+]] = f8e4m3fn[3]{0} parameter(0)
; CHECK-NEXT:    [[C0:%[^ ]+]] = f32[] constant(0)
; CHECK-NEXT:    [[C0_CONVERT:%[^ ]+]] = f8e4m3fn[] convert([[C0]])
; CHECK-NEXT:    [[P0_U0:%[^ ]+]] = f8e4m3fn[30]{0} pad([[P0]], [[C0_CONVERT]]), padding=0_27
; CHECK-NEXT:    [[P0_U1:%[^ ]+]] = f8e4m3fn[30,8,5]{2,1,0} broadcast([[P0_U0]]), dimensions={0}
; CHECK-NEXT:    [[P0_U2:%[^ ]+]] = f8e4m3fn[16,8,4]{2,1,0} slice([[P0_U1]]), slice={[2:18], [0:8], [0:4]}
; CHECK-NEXT:    [[P0_U3:%[^ ]+]] = f8e4m3fn[16,32]{1,0} reshape([[P0_U2]])
; CHECK-NEXT:    [[P1:%[^ ]+]] = f8e4m3fn[32,16]{1,0} parameter(1)
; CHECK-NEXT:    [[P1_TRANSPOSE:%[^ ]+]] = f8e4m3fn[16,32]{1,0} transpose([[P1]]), dimensions={1,0}
; CHECK-NEXT:    [[C1:%[^ ]+]] = f32[] constant(0)
; CHECK-NEXT:    [[C1_BCAST:%[^ ]+]] = f32[16,16]{1,0} broadcast([[C1]]), dimensions={}
; CHECK-NEXT:    [[P2:%[^ ]+]] = f32[] parameter(2)
; CHECK-NEXT:    [[P3:%[^ ]+]] = f32[] parameter(3)
; CHECK-NEXT:    [[C2:%[^ ]+]] = f32[] constant(1)
; CHECK-NEXT:    ROOT [[OUT:%[^ ]+]] = f32[16,16]{1,0} custom-call([[P0_U3]], [[P1_TRANSPOSE]], [[C1_BCAST]], [[P2]], [[P3]], /*index=5*/[[C2]], [[C2]]),
; CHECK:           custom_call_target="__cublas$lt$matmul$f8",
; CHECK:           backend_config="{
; CHECK-DAG:         \"alpha_real\":1
; CHECK-DAG:         \"alpha_imag\":0
; CHECK-DAG:         \"beta\":0
; CHECK-DAG:         \"dot_dimension_numbers\":{
; CHECK-DAG:           \"lhs_contracting_dimensions\":[\"1\"]
; CHECK-DAG:           \"rhs_contracting_dimensions\":[\"1\"]
; CHECK-DAG:           \"lhs_batch_dimensions\":[]
; CHECK-DAG:           \"rhs_batch_dimensions\":[]
; CHECK-DAG:         }
; CHECK-DAG:         \"precision_config\":{
; CHECK-DAG:           \"operand_precision\":[\"DEFAULT\",\"DEFAULT\"]
; CHECK-DAG:         }
; CHECK-DAG:         \"epilogue\":\"DEFAULT\"
; CHECK:           }"
      )");
}

TEST_P(ParameterizedFp8GemmRewriteTest, BatchedScaledABUnscaledDF8) {
  const char* hlo_text = R"(
    HloModule test

    ENTRY test {
      x = f8e4m3fn[10,16,32] parameter(0)
      y = f8e4m3fn[10,32,16] parameter(1)
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

  CheckFp8IfOnHopper(hlo_text);
  RunAndFilecheckHloRewrite(hlo_text,
                            GemmRewriter(se::CudaComputeCapability{
                                se::CudaComputeCapability::HOPPER, 0}),
                            R"(
; CHECK-LABEL: ENTRY %test (x: f8e4m3fn[10,16,32], y: f8e4m3fn[10,32,16], x_scale: f32[], y_scale: f32[]) -> f32[10,16,16] {
; CHECK-NEXT:    [[P0:%[^ ]+]] = f8e4m3fn[10,16,32]{2,1,0} parameter(0)
; CHECK-NEXT:    [[P1:%[^ ]+]] = f8e4m3fn[10,32,16]{2,1,0} parameter(1)
; CHECK-NEXT:    [[P1_TRANSPOSE:%[^ ]+]] = f8e4m3fn[10,16,32]{2,1,0} transpose([[P1]]), dimensions={0,2,1}
; CHECK-NEXT:    [[C0:%[^ ]+]] = f32[] constant(0)
; CHECK-NEXT:    [[C0_BCAST:%[^ ]+]] = f32[10,16,16]{2,1,0} broadcast([[C0]]), dimensions={}
; CHECK-NEXT:    [[P2:%[^ ]+]] = f32[] parameter(2)
; CHECK-NEXT:    [[P3:%[^ ]+]] = f32[] parameter(3)
; CHECK-NEXT:    [[C1:%[^ ]+]] = f32[] constant(1)
; CHECK-NEXT:    ROOT [[OUT:%[^ ]+]] = f32[10,16,16]{2,1,0} custom-call([[P0]], [[P1_TRANSPOSE]], [[C0_BCAST]], [[P2]], [[P3]], /*index=5*/[[C1]], [[C1]]),
; CHECK:           custom_call_target="__cublas$lt$matmul$f8",
; CHECK:           backend_config="{
; CHECK-DAG:         \"alpha_real\":1
; CHECK-DAG:         \"alpha_imag\":0
; CHECK-DAG:         \"beta\":0
; CHECK-DAG:         \"dot_dimension_numbers\":{
; CHECK-DAG:           \"lhs_contracting_dimensions\":[\"2\"]
; CHECK-DAG:           \"rhs_contracting_dimensions\":[\"2\"]
; CHECK-DAG:           \"lhs_batch_dimensions\":[\"0\"]
; CHECK-DAG:           \"rhs_batch_dimensions\":[\"0\"]
; CHECK-DAG:         }
; CHECK-DAG:         \"precision_config\":{
; CHECK-DAG:           \"operand_precision\":[\"DEFAULT\",\"DEFAULT\"]
; CHECK-DAG:         }
; CHECK-DAG:         \"epilogue\":\"DEFAULT\"
; CHECK:           }"
      )");
}

TEST_P(ParameterizedFp8GemmRewriteTest, ScaledABAlphaDF8) {
  const char* hlo_text = R"(
    HloModule test

    ENTRY test {
      x = f8e4m3fn[16,32] parameter(0)
      y = f8e4m3fn[32,16] parameter(1)
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

  CheckFp8IfOnHopper(hlo_text);
  RunAndFilecheckHloRewrite(hlo_text,
                            GemmRewriter(se::CudaComputeCapability{
                                se::CudaComputeCapability::HOPPER, 0}),
                            R"(

; CHECK-LABEL: ENTRY %test (x: f8e4m3fn[16,32], y: f8e4m3fn[32,16], x_scale: f32[], y_scale: f32[]) -> f32[16,16] {
; CHECK-NEXT:    [[P0:%[^ ]+]] = f8e4m3fn[16,32]{1,0} parameter(0)
; CHECK-NEXT:    [[P1:%[^ ]+]] = f8e4m3fn[32,16]{1,0} parameter(1)
; CHECK-NEXT:    [[P1_TRANSPOSE:%[^ ]+]] = f8e4m3fn[16,32]{1,0} transpose([[P1]]), dimensions={1,0}
; CHECK-NEXT:    [[C0:%[^ ]+]] = f32[] constant(0)
; CHECK-NEXT:    [[C0_BCAST:%[^ ]+]] = f32[16,16]{1,0} broadcast([[C0]]), dimensions={}
; CHECK-NEXT:    [[P2:%[^ ]+]] = f32[] parameter(2)
; CHECK-NEXT:    [[P3:%[^ ]+]] = f32[] parameter(3)
; CHECK-NEXT:    [[C1:%[^ ]+]] = f32[] constant(1)
; CHECK-NEXT:    ROOT [[OUT:%[^ ]+]] = f32[16,16]{1,0} custom-call([[P0]], [[P1_TRANSPOSE]], [[C0_BCAST]], [[P2]], [[P3]], /*index=5*/[[C1]], [[C1]]),
; CHECK:           custom_call_target="__cublas$lt$matmul$f8",
; CHECK:           backend_config="{
; CHECK-DAG:         \"alpha_real\":3
; CHECK-DAG:         \"alpha_imag\":0
; CHECK-DAG:         \"beta\":0
; CHECK-DAG:         \"dot_dimension_numbers\":{
; CHECK-DAG:           \"lhs_contracting_dimensions\":[\"1\"]
; CHECK-DAG:           \"rhs_contracting_dimensions\":[\"1\"]
; CHECK-DAG:           \"lhs_batch_dimensions\":[]
; CHECK-DAG:           \"rhs_batch_dimensions\":[]
; CHECK-DAG:         }
; CHECK-DAG:         \"precision_config\":{
; CHECK-DAG:           \"operand_precision\":[\"DEFAULT\",\"DEFAULT\"]
; CHECK-DAG:         }
; CHECK-DAG:         \"epilogue\":\"DEFAULT\"
; CHECK:           }"
      )");
}

TEST_P(ParameterizedFp8GemmRewriteTest, ScaledABUnscaledDReluActivationF8) {
  const char* hlo_text = R"(
    HloModule test

    ENTRY test {
      x = f8e4m3fn[16,32] parameter(0)
      y = f8e4m3fn[32,16] parameter(1)
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

  CheckFp8IfOnHopper(hlo_text);
  RunAndFilecheckHloRewrite(hlo_text,
                            GemmRewriter(se::CudaComputeCapability{
                                se::CudaComputeCapability::HOPPER, 0}),
                            R"(

; CHECK-LABEL: ENTRY %test (x: f8e4m3fn[16,32], y: f8e4m3fn[32,16], x_scale: f32[], y_scale: f32[]) -> f32[16,16] {
; CHECK-NEXT:    [[P0:%[^ ]+]] = f8e4m3fn[16,32]{1,0} parameter(0)
; CHECK-NEXT:    [[P1:%[^ ]+]] = f8e4m3fn[32,16]{1,0} parameter(1)
; CHECK-NEXT:    [[P1_TRANSPOSE:%[^ ]+]] = f8e4m3fn[16,32]{1,0} transpose([[P1]]), dimensions={1,0}
; CHECK-NEXT:    [[C0:%[^ ]+]] = f32[] constant(0)
; CHECK-NEXT:    [[C0_BCAST:%[^ ]+]] = f32[16,16]{1,0} broadcast([[C0]]), dimensions={}
; CHECK-NEXT:    [[P2:%[^ ]+]] = f32[] parameter(2)
; CHECK-NEXT:    [[P3:%[^ ]+]] = f32[] parameter(3)
; CHECK-NEXT:    [[C1:%[^ ]+]] = f32[] constant(1)
; CHECK-NEXT:    ROOT [[OUT:%[^ ]+]] = f32[16,16]{1,0} custom-call([[P0]], [[P1_TRANSPOSE]], [[C0_BCAST]], [[P2]], [[P3]], /*index=5*/[[C1]], [[C1]]),
; CHECK:           custom_call_target="__cublas$lt$matmul$f8",
; CHECK:           backend_config="{
; CHECK-DAG:         \"alpha_real\":1
; CHECK-DAG:         \"alpha_imag\":0
; CHECK-DAG:         \"beta\":0
; CHECK-DAG:         \"dot_dimension_numbers\":{
; CHECK-DAG:           \"lhs_contracting_dimensions\":[\"1\"]
; CHECK-DAG:           \"rhs_contracting_dimensions\":[\"1\"]
; CHECK-DAG:           \"lhs_batch_dimensions\":[]
; CHECK-DAG:           \"rhs_batch_dimensions\":[]
; CHECK-DAG:         }
; CHECK-DAG:         \"precision_config\":{
; CHECK-DAG:           \"operand_precision\":[\"DEFAULT\",\"DEFAULT\"]
; CHECK-DAG:         }
; CHECK-DAG:         \"epilogue\":\"RELU\"
; CHECK:           }"
      )");
}

TEST_P(ParameterizedFp8GemmRewriteTest, InvScaledABUnscaledDF8) {
  const char* hlo_text = R"(
    HloModule test

    ENTRY test {
      x = f8e4m3fn[16,32] parameter(0)
      y = f8e4m3fn[32,16] parameter(1)
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

  CheckFp8IfOnHopper(hlo_text);
  RunAndFilecheckHloRewrite(hlo_text,
                            GemmRewriter(se::CudaComputeCapability{
                                se::CudaComputeCapability::HOPPER, 0}),
                            R"(
; CHECK:           custom_call_target="__cublas$lt$matmul$f8",
      )");
}

TEST_P(ParameterizedFp8GemmRewriteTest, ScaledABUnscaledDMatrixBiasF8) {
#if CUDA_VERSION < 12000
  GTEST_SKIP() << "A matrix bias on a matmul is only supported in CUDA 12";
#endif
  const char* hlo_text = R"(
    HloModule test

    ENTRY test {
      x = f8e4m3fn[16,32] parameter(0)
      y = f8e4m3fn[32,16] parameter(1)
      b = f32[16,16] parameter(2)
      x_f32 = f32[16,32] convert(x)
      y_f32 = f32[32,16] convert(y)
      x_scale = f32[] parameter(3)
      y_scale = f32[] parameter(4)
      x_scale_bcast = f32[16,32] broadcast(x_scale), dimensions={}
      y_scale_bcast = f32[32,16] broadcast(y_scale), dimensions={}
      x_unscaled = f32[16,32] multiply(x_f32, x_scale_bcast)
      y_unscaled = f32[32,16] multiply(y_f32, y_scale_bcast)
      dot_a = f32[16,16] dot(x_unscaled, y_unscaled), lhs_contracting_dims={1}, rhs_contracting_dims={0}
      ROOT out = add(dot_a, b)
          }

)";

  CheckFp8IfOnHopper(hlo_text);
  RunAndFilecheckHloRewrite(hlo_text,
                            GemmRewriter(se::CudaComputeCapability{
                                se::CudaComputeCapability::HOPPER, 0}),
                            R"(

; CHECK-LABEL: ENTRY %test (x: f8e4m3fn[16,32], y: f8e4m3fn[32,16], b: f32[16,16], x_scale: f32[], y_scale: f32[]) -> f32[16,16] {
; CHECK-NEXT:    [[P0:%[^ ]+]] = f8e4m3fn[16,32]{1,0} parameter(0)
; CHECK-NEXT:    [[P1:%[^ ]+]] = f8e4m3fn[32,16]{1,0} parameter(1)
; CHECK-NEXT:    [[P1_TRANSPOSE:%[^ ]+]] = f8e4m3fn[16,32]{1,0} transpose([[P1]]), dimensions={1,0}
; CHECK-NEXT:    [[C0:%[^ ]+]] = f32[16,16]{1,0} parameter(2)
; CHECK-NEXT:    [[P2:%[^ ]+]] = f32[] parameter(3)
; CHECK-NEXT:    [[P3:%[^ ]+]] = f32[] parameter(4)
; CHECK-NEXT:    [[C1:%[^ ]+]] = f32[] constant(1)
; CHECK-NEXT:    ROOT [[OUT:%[^ ]+]] = f32[16,16]{1,0} custom-call([[P0]], [[P1_TRANSPOSE]], [[C0]], [[P2]], [[P3]], /*index=5*/[[C1]], [[C1]]),
; CHECK:           custom_call_target="__cublas$lt$matmul$f8",
; CHECK:           backend_config="{
; CHECK-DAG:         \"alpha_real\":1
; CHECK-DAG:         \"alpha_imag\":0
; CHECK-DAG:         \"beta\":1
; CHECK-DAG:         \"dot_dimension_numbers\":{
; CHECK-DAG:           \"lhs_contracting_dimensions\":[\"1\"]
; CHECK-DAG:           \"rhs_contracting_dimensions\":[\"1\"]
; CHECK-DAG:           \"lhs_batch_dimensions\":[]
; CHECK-DAG:           \"rhs_batch_dimensions\":[]
; CHECK-DAG:         }
; CHECK-DAG:         \"precision_config\":{
; CHECK-DAG:           \"operand_precision\":[\"DEFAULT\",\"DEFAULT\"]
; CHECK-DAG:         }
; CHECK-DAG:         \"epilogue\":\"DEFAULT\"
; CHECK:           }"
      )");
}

TEST_P(ParameterizedFp8GemmRewriteTest, ScaledABScaledDF8) {
  const char* hlo_text = R"(
    HloModule test

    ENTRY test {
      x = f8e4m3fn[16,32] parameter(0)
      y = f8e4m3fn[32,16] parameter(1)
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
      c1 = f32[] constant(-448.)
      c1_bcast = f32[16,16] broadcast(c1), dimensions={}
      c2 = f32[] constant(448.)
      c2_bcast = f32[16,16] broadcast(c2), dimensions={}
      dot_a_clamped = f32[16,16] clamp(c1_bcast, dot_a_scaled, c2_bcast)
      ROOT dot_a_f8 = f8e4m3fn[16,16] convert(dot_a_clamped)
          }

)";

  CheckFp8IfOnHopper(hlo_text);
  RunAndFilecheckHloRewrite(hlo_text,
                            GemmRewriter(se::CudaComputeCapability{
                                se::CudaComputeCapability::HOPPER, 0}),
                            R"(
; CHECK-LABEL: ENTRY %test (x: f8e4m3fn[16,32], y: f8e4m3fn[32,16], x_scale: f32[], y_scale: f32[], z_scale: f32[]) -> f8e4m3fn[16,16] {
; CHECK:         [[P0:%[^ ]+]] = f8e4m3fn[16,32]{1,0} parameter(0)
; CHECK-NEXT:    [[P1:%[^ ]+]] = f8e4m3fn[32,16]{1,0} parameter(1)
; CHECK-NEXT:    [[P1_TRANSPOSE:%[^ ]+]] = f8e4m3fn[16,32]{1,0} transpose([[P1]]), dimensions={1,0}
; CHECK-NEXT:    [[C0:%[^ ]+]] = bf16[] constant(0)
; CHECK-NEXT:    [[C0_BCAST:%[^ ]+]] = bf16[16,16]{1,0} broadcast([[C0]]), dimensions={}
; CHECK-NEXT:    [[P2:%[^ ]+]] = f32[] parameter(2)
; CHECK-NEXT:    [[P3:%[^ ]+]] = f32[] parameter(3)
; CHECK-NEXT:    [[C1:%[^ ]+]] = f32[] constant(1)
; CHECK-NEXT:    [[C2:%[^ ]+]] = f32[] constant(1)
; CHECK-NEXT:    [[P4:%[^ ]+]] = f32[] parameter(4)
; CHECK-NEXT:    [[P4_INV:%[^ ]+]] = f32[] divide([[C2]], [[P4]])
; CHECK-NEXT:    ROOT [[OUT:%[^ ]+]] = f8e4m3fn[16,16]{1,0} custom-call([[P0]], [[P1_TRANSPOSE]], [[C0_BCAST]], [[P2]], [[P3]], /*index=5*/[[C1]], [[P4_INV]]),
; CHECK:           custom_call_target="__cublas$lt$matmul$f8",
; CHECK:           backend_config="{
; CHECK-DAG:         \"alpha_real\":1
; CHECK-DAG:         \"alpha_imag\":0
; CHECK-DAG:         \"beta\":0
; CHECK-DAG:         \"dot_dimension_numbers\":{
; CHECK-DAG:           \"lhs_contracting_dimensions\":[\"1\"]
; CHECK-DAG:           \"rhs_contracting_dimensions\":[\"1\"]
; CHECK-DAG:           \"lhs_batch_dimensions\":[]
; CHECK-DAG:           \"rhs_batch_dimensions\":[]
; CHECK-DAG:         }
; CHECK-DAG:         \"precision_config\":{
; CHECK-DAG:           \"operand_precision\":[\"DEFAULT\",\"DEFAULT\"]
; CHECK-DAG:         }
; CHECK-DAG:         \"epilogue\":\"DEFAULT\"
; CHECK:           }"
      )");
}

TEST_P(ParameterizedFp8GemmRewriteTest, ScaledABInvScaledDF8) {
  const char* hlo_text = R"(
    HloModule test

    ENTRY test {
      x = f8e4m3fn[16,32] parameter(0)
      y = f8e4m3fn[32,16] parameter(1)
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
      c1 = f32[] constant(-448.)
      c1_bcast = f32[16,16] broadcast(c1), dimensions={}
      c2 = f32[] constant(448.)
      c2_bcast = f32[16,16] broadcast(c2), dimensions={}
      dot_a_clamped = f32[16,16] clamp(c1_bcast, dot_a_scaled, c2_bcast)
      ROOT dot_a_f8 = f8e4m3fn[16,16] convert(dot_a_clamped)
          }

)";

  CheckFp8IfOnHopper(hlo_text);
  RunAndFilecheckHloRewrite(hlo_text,
                            GemmRewriter(se::CudaComputeCapability{
                                se::CudaComputeCapability::HOPPER, 0}),
                            R"(

; CHECK-NOT:     divide

; CHECK:           custom_call_target="__cublas$lt$matmul$f8",

      )");
}

TEST_P(ParameterizedFp8GemmRewriteTest, ScaledABScaledDReluActivationF8) {
  const char* hlo_text = R"(
    HloModule test
    ENTRY test {
      x = f8e4m3fn[16,32] parameter(0)
      y = f8e4m3fn[32,16] parameter(1)
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
      c1 = f32[] constant(-448.)
      c1_bcast = f32[16,16] broadcast(c1), dimensions={}
      c2 = f32[] constant(448.)
      c2_bcast = f32[16,16] broadcast(c2), dimensions={}
      relu_a_clamped = f32[16,16] clamp(c1_bcast, relu_a_scaled, c2_bcast)
      ROOT out = f8e4m3fn[16,16] convert(relu_a_clamped)
          }
)";

  CheckFp8IfOnHopper(hlo_text);
  RunAndFilecheckHloRewrite(hlo_text,
                            GemmRewriter(se::CudaComputeCapability{
                                se::CudaComputeCapability::HOPPER, 0}),
                            R"(
; CHECK-LABEL: ENTRY %test (x: f8e4m3fn[16,32], y: f8e4m3fn[32,16], x_scale: f32[], y_scale: f32[], z_scale: f32[]) -> f8e4m3fn[16,16] {
; CHECK:         [[P0:%[^ ]+]] = f8e4m3fn[16,32]{1,0} parameter(0)
; CHECK-NEXT:    [[P1:%[^ ]+]] = f8e4m3fn[32,16]{1,0} parameter(1)
; CHECK-NEXT:    [[P1_TRANSPOSE:%[^ ]+]] = f8e4m3fn[16,32]{1,0} transpose([[P1]]), dimensions={1,0}
; CHECK-NEXT:    [[C0:%[^ ]+]] = bf16[] constant(0)
; CHECK-NEXT:    [[C0_BCAST:%[^ ]+]] = bf16[16,16]{1,0} broadcast([[C0]]), dimensions={}
; CHECK-NEXT:    [[P2:%[^ ]+]] = f32[] parameter(2)
; CHECK-NEXT:    [[P3:%[^ ]+]] = f32[] parameter(3)
; CHECK-NEXT:    [[C1:%[^ ]+]] = f32[] constant(1)
; CHECK-NEXT:    [[C2:%[^ ]+]] = f32[] constant(1)
; CHECK-NEXT:    [[P4:%[^ ]+]] = f32[] parameter(4)
; CHECK-NEXT:    [[P4_INV:%[^ ]+]] = f32[] divide([[C2]], [[P4]])
; CHECK-NEXT:    ROOT [[OUT:%[^ ]+]] = f8e4m3fn[16,16]{1,0} custom-call([[P0]], [[P1_TRANSPOSE]], [[C0_BCAST]], [[P2]], [[P3]], /*index=5*/[[C1]], [[P4_INV]]),
; CHECK:           custom_call_target="__cublas$lt$matmul$f8",
; CHECK:           backend_config="{
; CHECK-DAG:         \"alpha_real\":1
; CHECK-DAG:         \"alpha_imag\":0
; CHECK-DAG:         \"beta\":0
; CHECK-DAG:         \"dot_dimension_numbers\":{
; CHECK-DAG:           \"lhs_contracting_dimensions\":[\"1\"]
; CHECK-DAG:           \"rhs_contracting_dimensions\":[\"1\"]
; CHECK-DAG:           \"lhs_batch_dimensions\":[]
; CHECK-DAG:           \"rhs_batch_dimensions\":[]
; CHECK-DAG:         }
; CHECK-DAG:         \"precision_config\":{
; CHECK-DAG:           \"operand_precision\":[\"DEFAULT\",\"DEFAULT\"]
; CHECK-DAG:         }
; CHECK-DAG:         \"epilogue\":\"RELU\"
; CHECK:           }"
      )");
}

TEST_P(ParameterizedFp8GemmRewriteTest, ScaledABScaledDMatrixBiasF8) {
#if CUDA_VERSION < 12000
  GTEST_SKIP() << "A matrix bias on a matmul is only supported in CUDA 12";
#endif
  const char* hlo_text = R"(
    HloModule test

    ENTRY test {
      x = f8e4m3fn[16,32] parameter(0)
      y = f8e4m3fn[32,16] parameter(1)
      x_f16 = f16[16,32] convert(x)
      y_f16 = f16[32,16] convert(y)
      b = f16[16,16] parameter(2)
      x_scale = f16[] parameter(3)
      y_scale = f16[] parameter(4)
      z_scale = f16[] parameter(5)
      x_scale_bcast = f16[16,32] broadcast(x_scale), dimensions={}
      y_scale_bcast = f16[32,16] broadcast(y_scale), dimensions={}
      z_scale_bcast = f16[16,16] broadcast(z_scale), dimensions={}
      x_unscaled = f16[16,32] multiply(x_f16, x_scale_bcast)
      y_unscaled = f16[32,16] multiply(y_f16, y_scale_bcast)
      dot_a = f16[16,16] dot(x_unscaled, y_unscaled), lhs_contracting_dims={1}, rhs_contracting_dims={0}
      dot_a_bias = f16[16,16] add(dot_a, b)
      dot_a_scaled = f16[16,16] divide(dot_a_bias, z_scale_bcast)
      c1 = f16[] constant(-448.)
      c1_bcast = f16[16,16] broadcast(c1), dimensions={}
      c2 = f16[] constant(448.)
      c2_bcast = f16[16,16] broadcast(c2), dimensions={}
      dot_a_clamped = f16[16,16] clamp(c1_bcast, dot_a_scaled, c2_bcast)
      ROOT dot_a_f8 = f8e4m3fn[16,16] convert(dot_a_clamped)
          }

)";

  CheckFp8IfOnHopper(hlo_text, ErrorSpec{0.1, 0.1});
  RunAndFilecheckHloRewrite(hlo_text,
                            GemmRewriter(se::CudaComputeCapability{
                                se::CudaComputeCapability::HOPPER, 0}),
                            R"(

; CHECK-LABEL: ENTRY %test (x: f8e4m3fn[16,32], y: f8e4m3fn[32,16], b: f16[16,16], x_scale: f16[], y_scale: f16[], z_scale: f16[]) -> f8e4m3fn[16,16] {
; CHECK-NEXT:    [[P0:%[^ ]+]] = f8e4m3fn[16,32]{1,0} parameter(0)
; CHECK-NEXT:    [[P1:%[^ ]+]] = f8e4m3fn[32,16]{1,0} parameter(1)
; CHECK-NEXT:    [[P1_TRANSPOSE:%[^ ]+]] = f8e4m3fn[16,32]{1,0} transpose([[P1]]), dimensions={1,0}
; CHECK-NEXT:    [[C0:%[^ ]+]] = f16[16,16]{1,0} parameter(2)
; CHECK-NEXT:    [[P2:%[^ ]+]] = f16[] parameter(3)
; CHECK:         [[P3:%[^ ]+]] = f16[] parameter(4)
; CHECK:         [[C1:%[^ ]+]] = f32[] constant(1)
; CHECK:         [[P4:%[^ ]+]] = f16[] parameter(5)
; CHECK:       ROOT [[OUT:%[^ ]+]] = f8e4m3fn[16,16]{1,0} custom-call([[P0]], [[P1_TRANSPOSE]], [[C0]], [[DUMMY0:%[^ ]+]], [[DUMMY1:%[^ ]+]], /*index=5*/[[C1]], [[DUMMY2:%[^ ]+]]),
; CHECK:           custom_call_target="__cublas$lt$matmul$f8",
; CHECK:           backend_config="{
; CHECK-DAG:         \"alpha_real\":1
; CHECK-DAG:         \"alpha_imag\":0
; CHECK-DAG:         \"beta\":1
; CHECK-DAG:         \"dot_dimension_numbers\":{
; CHECK-DAG:           \"lhs_contracting_dimensions\":[\"1\"]
; CHECK-DAG:           \"rhs_contracting_dimensions\":[\"1\"]
; CHECK-DAG:           \"lhs_batch_dimensions\":[]
; CHECK-DAG:           \"rhs_batch_dimensions\":[]
; CHECK-DAG:         }
; CHECK-DAG:         \"precision_config\":{
; CHECK-DAG:           \"operand_precision\":[\"DEFAULT\",\"DEFAULT\"]
; CHECK-DAG:         }
; CHECK-DAG:         \"epilogue\":\"DEFAULT\"
; CHECK:           }"
      )");
}

TEST_P(ParameterizedFp8GemmRewriteTest, ScaledABScaledDVectorBiasF8) {
  const char* hlo_text = R"(
    HloModule test

    ENTRY test {
      x = f8e4m3fn[16,32] parameter(0)
      y = f8e4m3fn[32,16] parameter(1)
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
      c1 = f16[] constant(-448.)
      c1_bcast = f16[16,16] broadcast(c1), dimensions={}
      c2 = f16[] constant(448.)
      c2_bcast = f16[16,16] broadcast(c2), dimensions={}
      dot_a_clamped = f16[16,16] clamp(c1_bcast, dot_a_scaled, c2_bcast)
      ROOT dot_a_f8 = f8e4m3fn[16,16] convert(dot_a_clamped)
          }

)";

  CheckFp8IfOnHopper(hlo_text, ErrorSpec{0.1, 0.1});
  RunAndFilecheckHloRewrite(hlo_text,
                            GemmRewriter(se::CudaComputeCapability{
                                se::CudaComputeCapability::HOPPER, 0}),
                            R"(

; CHECK-LABEL: ENTRY %test (x: f8e4m3fn[16,32], y: f8e4m3fn[32,16], b: f16[16], x_scale: f16[], y_scale: f16[], z_scale: f16[]) -> f8e4m3fn[16,16] {
; CHECK-NEXT:    [[P0:%[^ ]+]] = f8e4m3fn[16,32]{1,0} parameter(0)
; CHECK-NEXT:    [[P1:%[^ ]+]] = f8e4m3fn[32,16]{1,0} parameter(1)
; CHECK-NEXT:    [[P1_TRANSPOSE:%[^ ]+]] = f8e4m3fn[16,32]{1,0} transpose([[P1]]), dimensions={1,0}
; CHECK-NEXT:    [[C1:%[^ ]+]] = f16[] constant(0)
; CHECK-NEXT:    [[BC:%[^ ]+]] = f16[16,16]{1,0} broadcast([[C1]]), dimensions={}
; CHECK-NEXT:    [[P2:%[^ ]+]] = f16[] parameter(3)
; CHECK-NEXT:    [[CV:%[^ ]+]] = f32[] convert([[P2]])
; CHECK-NEXT:    [[P3:%[^ ]+]] = f16[] parameter(4)
; CHECK-NEXT:    [[CV1:%[^ ]+]] = f32[] convert([[P3]])
; CHECK-NEXT:    [[C:%[^ ]+]] = f32[] constant(1)
; CHECK-NEXT:    [[C2:%[^ ]+]] = f16[] constant(1)
; CHECK-NEXT:    [[P4:%[^ ]+]] = f16[] parameter(5)
; CHECK-NEXT:    [[DV:%[^ ]+]] = f16[] divide([[C2]], [[P4]])
; CHECK-NEXT:    [[CV2:%[^ ]+]] = f32[] convert([[DV]])
; CHECK-NEXT:    [[VB:%[^ ]+]] = f16[16]{0} parameter(2)
; CHECK:         ROOT [[OUT:%[^ ]+]] = f8e4m3fn[16,16]{1,0} custom-call([[P0]], [[P1_TRANSPOSE]], [[BC]], [[CV]], [[CV1]], /*index=5*/[[C]], [[CV2]], [[VB]]),
; CHECK:           custom_call_target="__cublas$lt$matmul$f8",
; CHECK:           backend_config="{
; CHECK-DAG:         \"alpha_real\":1
; CHECK-DAG:         \"alpha_imag\":0
; CHECK-DAG:         \"beta\":0
; CHECK-DAG:         \"dot_dimension_numbers\":{ 
; CHECK-DAG:           \"lhs_contracting_dimensions\":[\"1\"]
; CHECK-DAG:           \"rhs_contracting_dimensions\":[\"1\"]
; CHECK-DAG:           \"lhs_batch_dimensions\":[]
; CHECK-DAG:           \"rhs_batch_dimensions\":[] 
; CHECK-DAG:         }
; CHECK-DAG:         \"precision_config\":{
; CHECK-DAG:           \"operand_precision\":[\"DEFAULT\",\"DEFAULT\"]
; CHECK-DAG:         }
; CHECK-DAG:         \"epilogue\":\"BIAS\"
; CHECK:           }"
      )");
}

TEST_P(ParameterizedFp8GemmRewriteTest,
       ScaledABUnscaledDVectorBiasThenReluActivationF8) {
  const char* hlo_text = R"(
    HloModule test

    ENTRY test {
      x = f8e4m3fn[16,32] parameter(0)
      y = f8e4m3fn[32,16] parameter(1)
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

  CheckFp8IfOnHopper(hlo_text, ErrorSpec{2e-3, 0.});
  RunAndFilecheckHloRewrite(hlo_text,
                            GemmRewriter(se::CudaComputeCapability{
                                se::CudaComputeCapability::HOPPER, 0}),
                            R"(
; CHECK-LABEL: ENTRY %test (x: f8e4m3fn[16,32], y: f8e4m3fn[32,16], b: f16[16], x_scale: f16[], y_scale: f16[]) -> f16[16,16] {
; CHECK-NEXT:    [[P0:%[^ ]+]] = f8e4m3fn[16,32]{1,0} parameter(0)
; CHECK-NEXT:    [[P1:%[^ ]+]] = f8e4m3fn[32,16]{1,0} parameter(1)
; CHECK-NEXT:    [[P1_TRANSPOSE:%[^ ]+]] = f8e4m3fn[16,32]{1,0} transpose([[P1]]), dimensions={1,0}
; CHECK-NEXT:    [[C1:%[^ ]+]] = f16[] constant(0)
; CHECK-NEXT:    [[BC:%[^ ]+]] = f16[16,16]{1,0} broadcast([[C1]]), dimensions={}
; CHECK-NEXT:    [[P2:%[^ ]+]] = f16[] parameter(3)
; CHECK-NEXT:    [[CV:%[^ ]+]] = f32[] convert([[P2]])
; CHECK-NEXT:    [[P3:%[^ ]+]] = f16[] parameter(4)
; CHECK-NEXT:    [[CV1:%[^ ]+]] = f32[] convert([[P3]])
; CHECK-NEXT:    [[C:%[^ ]+]] = f32[] constant(1)
; CHECK-NEXT:    [[VB:%[^ ]+]] = f16[16]{0} parameter(2)
; CHECK     :    ROOT [[OUT:%[^ ]+]] = f16[16,16]{1,0} custom-call([[P0]], [[P1_TRANSPOSE]], [[BC]], [[CV]], [[CV1]], /*index=5*/[[C]], [[C]], [[VB]]),
; CHECK:           custom_call_target="__cublas$lt$matmul$f8",
; CHECK:           backend_config="{
; CHECK-DAG:         \"alpha_real\":1
; CHECK-DAG:         \"alpha_imag\":0
; CHECK-DAG:         \"beta\":0
; CHECK-DAG:         \"dot_dimension_numbers\":{
; CHECK-DAG:           \"lhs_contracting_dimensions\":[\"1\"]
; CHECK-DAG:           \"rhs_contracting_dimensions\":[\"1\"]
; CHECK-DAG:           \"lhs_batch_dimensions\":[]
; CHECK-DAG:           \"rhs_batch_dimensions\":[]
; CHECK-DAG:         }
; CHECK-DAG:         \"precision_config\":{
; CHECK-DAG:           \"operand_precision\":[\"DEFAULT\",\"DEFAULT\"]
; CHECK-DAG:         }
; CHECK-DAG:         \"epilogue\":\"BIAS_RELU\"
; CHECK:           }"
      )");
}

TEST_P(ParameterizedFp8GemmRewriteTest,
       ScaledABUnscaledDMatrixBiasThenVectorBiasF8) {
#if CUDA_VERSION < 12000
  GTEST_SKIP() << "A matrix bias on a matmul is only supported in CUDA 12";
#endif
  const char* hlo_text = R"(
    HloModule test

    ENTRY test {
      x = f8e4m3fn[16,32] parameter(0)
      y = f8e4m3fn[32,16] parameter(1)
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
  CheckFp8IfOnHopper(hlo_text, ErrorSpec{2e-3, 0.});
  RunAndFilecheckHloRewrite(hlo_text,
                            GemmRewriter(se::CudaComputeCapability{
                                se::CudaComputeCapability::HOPPER, 0}),
                            R"(
; CHECK-LABEL:   ENTRY %test (x: f8e4m3fn[16,32], y: f8e4m3fn[32,16], b: f16[16], b2: f16[16,16], x_scale: f16[], y_scale: f16[]) -> f16[16,16] {
; CHECK-DAG:     [[P0:%[^ ]+]] = f8e4m3fn[16,32]{1,0} parameter(0)
; CHECK-NEXT:    [[P1:%[^ ]+]] = f8e4m3fn[32,16]{1,0} parameter(1)
; CHECK-NEXT:    [[P1_TRANSPOSE:%[^ ]+]] = f8e4m3fn[16,32]{1,0} transpose([[P1]]), dimensions={1,0}
; CHECK-NEXT:    [[MB:%[^ ]+]] = f16[16,16]{1,0} parameter(3)
; CHECK-NEXT:    [[P2:%[^ ]+]] = f16[] parameter(4)
; CHECK-NEXT:    [[CV0:%[^ ]+]] = f32[] convert([[P2]])
; CHECK-NEXT:    [[P3:%[^ ]+]] = f16[] parameter(5)
; CHECK-NEXT:    [[CV1:%[^ ]+]] = f32[] convert([[P3]])
; CHECK:         [[C1:%[^ ]+]] = f32[] constant(1)
; CHECK:         [[GEMMOUT:%[^ ]+]] = f16[16,16]{1,0} custom-call([[P0]], [[P1_TRANSPOSE]], [[MB]], [[CV0]], [[CV1]], /*index=5*/[[C1]], [[C1]]),
; CHECK:           custom_call_target="__cublas$lt$matmul$f8",
; CHECK:           backend_config="{
; CHECK-DAG:         \"alpha_real\":1
; CHECK-DAG:         \"alpha_imag\":0
; CHECK-DAG:         \"beta\":1
; CHECK-DAG:         \"dot_dimension_numbers\":{ 
; CHECK-DAG:           \"lhs_contracting_dimensions\":[\"1\"]
; CHECK-DAG:           \"rhs_contracting_dimensions\":[\"1\"]
; CHECK-DAG:           \"lhs_batch_dimensions\":[]
; CHECK-DAG:           \"rhs_batch_dimensions\":[] 
; CHECK-DAG:         }
; CHECK-DAG:         \"precision_config\":{
; CHECK-DAG:           \"operand_precision\":[\"DEFAULT\",\"DEFAULT\"]
; CHECK-DAG:         }
; CHECK-DAG:         \"epilogue\":\"DEFAULT\"
; CHECK:           }"
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
      x = f8e4m3fn[16,32] parameter(0)
      y = f8e4m3fn[32,16] parameter(1)
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
      c1 = f32[] constant(-448.)
      c1_bcast = f32[16,16] broadcast(c1), dimensions={}
      c2 = f32[] constant(448.)
      c2_bcast = f32[16,16] broadcast(c2), dimensions={}
      dot_a_clamped = f32[16,16] clamp(c1_bcast, dot_a_scaled, c2_bcast)
      dot_a_f8 = f8e4m3fn[16,16] convert(dot_a_clamped)
      ROOT out = (f8e4m3fn[16,16], f32[]) tuple(dot_a_f8, amax)
          }

)";

  CheckFp8IfOnHopper(hlo_text);
  RunAndFilecheckHloRewrite(hlo_text,
                            GemmRewriter(se::CudaComputeCapability{
                                se::CudaComputeCapability::HOPPER, 0}),
                            R"(
; CHECK-LABEL: ENTRY %test (x: f8e4m3fn[16,32], y: f8e4m3fn[32,16], x_scale: f32[], y_scale: f32[], z_scale: f32[]) -> (f8e4m3fn[16,16], f32[]) {
; CHECK-NEXT:    [[P0:%[^ ]+]] = f8e4m3fn[16,32]{1,0} parameter(0)
; CHECK-NEXT:    [[P1:%[^ ]+]] = f8e4m3fn[32,16]{1,0} parameter(1)
; CHECK-NEXT:    [[P1_TRANSPOSE:%[^ ]+]] = f8e4m3fn[16,32]{1,0} transpose([[P1]])
; CHECK-NEXT:    [[C0:%[^ ]+]] = bf16[] constant(0)
; CHECK-NEXT:    [[C0_BCAST:%[^ ]+]] = bf16[16,16]{1,0} broadcast([[C0]]), dimensions={}
; CHECK-NEXT:    [[P2:%[^ ]+]] = f32[] parameter(2)
; CHECK-NEXT:    [[P3:%[^ ]+]] = f32[] parameter(3)
; CHECK-NEXT:    [[C1:%[^ ]+]] = f32[] constant(1)
; CHECK-NEXT:    [[C2:%[^ ]+]] = f32[] constant(1)
; CHECK-NEXT:    [[P4:%[^ ]+]] = f32[] parameter(4)
; CHECK-NEXT:    [[P4_INV:%[^ ]+]] = f32[] divide([[C2]], [[P4]])
; CHECK-NEXT:    [[OUT:%[^ ]+]] = (f8e4m3fn[16,16]{1,0}, f32[]) custom-call([[P0]], [[P1_TRANSPOSE]], [[C0_BCAST]], [[P2]], [[P3]], /*index=5*/[[C1]], [[P4_INV]]),
; CHECK:           custom_call_target="__cublas$lt$matmul$f8",
; CHECK:           backend_config="{
; CHECK-DAG:         \"alpha_real\":1
; CHECK-DAG:         \"alpha_imag\":0
; CHECK-DAG:         \"beta\":0
; CHECK-DAG:         \"dot_dimension_numbers\":{
; CHECK-DAG:           \"lhs_contracting_dimensions\":[\"1\"]
; CHECK-DAG:           \"rhs_contracting_dimensions\":[\"1\"]
; CHECK-DAG:           \"lhs_batch_dimensions\":[]
; CHECK-DAG:           \"rhs_batch_dimensions\":[]
; CHECK-DAG:         }
; CHECK-DAG:         \"precision_config\":{
; CHECK-DAG:           \"operand_precision\":[\"DEFAULT\",\"DEFAULT\"]
; CHECK-DAG:         }
; CHECK-DAG:         \"epilogue\":\"DEFAULT\"
; CHECK:           }"
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
      x = f8e4m3fn[16,32] parameter(0)
      y = f8e4m3fn[32,16] parameter(1)
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
      c1 = f16[] constant(-448.)
      c1_bcast = f16[16,16] broadcast(c1), dimensions={}
      c2 = f16[] constant(448.)
      c2_bcast = f16[16,16] broadcast(c2), dimensions={}
      dot_a_clamped = f16[16,16] clamp(c1_bcast, dot_a_scaled, c2_bcast)
      dot_a_f8 = f8e4m3fn[16,16] convert(dot_a_clamped)
      ROOT out = (f8e4m3fn[16,16], f16[]) tuple(dot_a_f8, amax)
          }
)";

  CheckFp8IfOnHopper(hlo_text);
  RunAndFilecheckHloRewrite(hlo_text,
                            GemmRewriter(se::CudaComputeCapability{
                                se::CudaComputeCapability::HOPPER, 0}),
                            R"(
; CHECK-LABEL: ENTRY %test (x: f8e4m3fn[16,32], y: f8e4m3fn[32,16], x_scale: f16[], y_scale: f16[], z_scale: f16[]) -> (f8e4m3fn[16,16], f16[]) {
; CHECK-NEXT:    [[P0:%[^ ]+]] = f8e4m3fn[16,32]{1,0} parameter(0)
; CHECK-NEXT:    [[P1:%[^ ]+]] = f8e4m3fn[32,16]{1,0} parameter(1)
; CHECK-NEXT:    [[P1_TRANSPOSE:%[^ ]+]] = f8e4m3fn[16,32]{1,0} transpose([[P1]])
; CHECK-NEXT:    [[C0:%[^ ]+]] = f16[] constant(0)
; CHECK-NEXT:    [[C0_BCAST:%[^ ]+]] = f16[16,16]{1,0} broadcast([[C0]]), dimensions={}
; CHECK-NEXT:    [[P2:%[^ ]+]] = f16[] parameter(2)
; CHECK-NEXT:    [[P2_CONVERT:%[^ ]+]] = f32[] convert([[P2]])
; CHECK-NEXT:    [[P3:%[^ ]+]] = f16[] parameter(3)
; CHECK-NEXT:    [[P3_CONVERT:%[^ ]+]] = f32[] convert([[P3]])
; CHECK-NEXT:    [[C1:%[^ ]+]] = f32[] constant(1)
; CHECK-NEXT:    [[C2:%[^ ]+]] = f16[] constant(1)
; CHECK-NEXT:    [[P4:%[^ ]+]] = f16[] parameter(4)
; CHECK-NEXT:    [[P4_INV:%[^ ]+]] = f16[] divide([[C2]], [[P4]])
; CHECK-NEXT:    [[P4_INV_CONVERT:%[^ ]+]] = f32[] convert([[P4_INV]])
; CHECK-NEXT:    [[OUT:%[^ ]+]] = (f8e4m3fn[16,16]{1,0}, f32[]) custom-call([[P0]], [[P1_TRANSPOSE]], [[C0_BCAST]], [[P2_CONVERT]], [[P3_CONVERT]], /*index=5*/[[C1]], [[P4_INV_CONVERT]]),
; CHECK:           custom_call_target="__cublas$lt$matmul$f8",
; CHECK:           backend_config="{
; CHECK-DAG:         \"alpha_real\":1
; CHECK-DAG:         \"alpha_imag\":0
; CHECK-DAG:         \"beta\":0
; CHECK-DAG:         \"dot_dimension_numbers\":{
; CHECK-DAG:           \"lhs_contracting_dimensions\":[\"1\"]
; CHECK-DAG:           \"rhs_contracting_dimensions\":[\"1\"]
; CHECK-DAG:           \"lhs_batch_dimensions\":[]
; CHECK-DAG:           \"rhs_batch_dimensions\":[]
; CHECK-DAG:         }
; CHECK-DAG:         \"precision_config\":{
; CHECK-DAG:           \"operand_precision\":[\"DEFAULT\",\"DEFAULT\"]
; CHECK-DAG:         }
; CHECK-DAG:         \"epilogue\":\"DEFAULT\"
; CHECK:           }"
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
      x = f8e4m3fn[16,32] parameter(0)
      y = f8e4m3fn[32,16] parameter(1)
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
      c1 = f32[] constant(-448.)
      c1_bcast = f32[16,16] broadcast(c1), dimensions={}
      c2 = f32[] constant(448.)
      c2_bcast = f32[16,16] broadcast(c2), dimensions={}
      dot_a_clamped = f32[16,16] clamp(c1_bcast, dot_a_scaled, c2_bcast)
      dot_a_f8 = f8e4m3fn[16,16] convert(dot_a_clamped)
      ROOT out = (f8e4m3fn[16,16], f32[]) tuple(dot_a_f8, amax)
          }

)";

  CheckFp8IfOnHopper(hlo_text);
  RunAndFilecheckHloRewrite(hlo_text,
                            GemmRewriter(se::CudaComputeCapability{
                                se::CudaComputeCapability::HOPPER, 0}),
                            R"(
; CHECK-LABEL: ENTRY %test (x: f8e4m3fn[16,32], y: f8e4m3fn[32,16], x_scale: f32[], y_scale: f32[], z_scale: f32[]) -> (f8e4m3fn[16,16], f32[]) {
; CHECK-NEXT:    [[P0:%[^ ]+]] = f8e4m3fn[16,32]{1,0} parameter(0)
; CHECK-NEXT:    [[P1:%[^ ]+]] = f8e4m3fn[32,16]{1,0} parameter(1)
; CHECK-NEXT:    [[P1_TRANSPOSE:%[^ ]+]] = f8e4m3fn[16,32]{1,0} transpose([[P1]])
; CHECK-NEXT:    [[C0:%[^ ]+]] = bf16[] constant(0)
; CHECK-NEXT:    [[C0_BCAST:%[^ ]+]] = bf16[16,16]{1,0} broadcast([[C0]]), dimensions={}
; CHECK-NEXT:    [[P2:%[^ ]+]] = f32[] parameter(2)
; CHECK-NEXT:    [[P3:%[^ ]+]] = f32[] parameter(3)
; CHECK-NEXT:    [[C1:%[^ ]+]] = f32[] constant(1)
; CHECK-NEXT:    [[C2:%[^ ]+]] = f32[] constant(1)
; CHECK-NEXT:    [[P4:%[^ ]+]] = f32[] parameter(4)
; CHECK-NEXT:    [[P4_INV:%[^ ]+]] = f32[] divide([[C2]], [[P4]])
; CHECK-NEXT:    [[OUT:%[^ ]+]] = (f8e4m3fn[16,16]{1,0}, f32[]) custom-call([[P0]], [[P1_TRANSPOSE]], [[C0_BCAST]], [[P2]], [[P3]], /*index=5*/[[C1]], [[P4_INV]]),
; CHECK:           custom_call_target="__cublas$lt$matmul$f8",
; CHECK:           backend_config="{
; CHECK-DAG:         \"alpha_real\":1
; CHECK-DAG:         \"alpha_imag\":0
; CHECK-DAG:         \"beta\":0
; CHECK-DAG:         \"dot_dimension_numbers\":{
; CHECK-DAG:           \"lhs_contracting_dimensions\":[\"1\"]
; CHECK-DAG:           \"rhs_contracting_dimensions\":[\"1\"]
; CHECK-DAG:           \"lhs_batch_dimensions\":[]
; CHECK-DAG:           \"rhs_batch_dimensions\":[]
; CHECK-DAG:         }
; CHECK-DAG:         \"precision_config\":{
; CHECK-DAG:           \"operand_precision\":[\"DEFAULT\",\"DEFAULT\"]
; CHECK-DAG:         }
; CHECK-DAG:         \"epilogue\":\"RELU\"
; CHECK:           }"
      )");
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
      x = f8e4m3fn<<Ashape>><<Alayout>> parameter(0)
      x_f32 = f32<<Ashape>><<Alayout>> convert(x)
      x_scale = f32[] parameter(2)
      x_scale_bcast = f32<<Ashape>> broadcast(x_scale), dimensions={}
      x_unscaled = f32<<Ashape>> multiply(x_f32, x_scale_bcast)
      y = f8e4m3fn<<Bshape>><<Blayout>> parameter(1)
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
    CheckFp8IfOnHopper(hlo_text);

    RunAndFilecheckHloRewrite(hlo_text,
                              GemmRewriter(se::CudaComputeCapability{
                                  se::CudaComputeCapability::HOPPER, 0}),
                              R"(
    ; CHECK:           custom_call_target="__cublas$lt$matmul$f8",
          )");
  }
}

TEST_P(ParameterizedFp8GemmRewriteTest,
       ScaledABUnscaledDF8ParameterizedBatched) {
  // TODO(wenscarl): For batched matmaul, not all combinations of A, B and
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
  x_q = f8e4m3fn<<Ashape>><<Alayout>> parameter(0)
  x_scale = f32[] parameter(2)
  x_scale_broadcast = f32<<Ashape>><<Alayout>> broadcast(x_scale), dimensions={}
  x_q_convert = f32<<Ashape>><<Alayout>> convert(x_q)
  x_qdq = f32<<Ashape>><<Alayout>> multiply(x_q_convert, x_scale_broadcast)

  y_q = f8e4m3fn<<Bshape>><<Blayout>> parameter(1)
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
    CheckFp8IfOnHopper(hlo_text);

    RunAndFilecheckHloRewrite(hlo_text,
                              GemmRewriter(se::CudaComputeCapability{
                                  se::CudaComputeCapability::HOPPER, 0}),
                              R"(
    ; CHECK:           custom_call_target="__cublas$lt$matmul$f8",
          )");
  }
}

TEST_P(ParameterizedFp8GemmRewriteTest, ScaledABUnscaledDF8TF32E5M2) {
  const char* hlo_text = R"(
    HloModule test

    ENTRY test {
      x = f8e4m3fn[16,32] parameter(0)
      y = f8e5m2[32,16] parameter(1)
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
  bool tf32_state_ = tsl::tensor_float_32_execution_enabled();
  tsl::enable_tensor_float_32_execution(true);

  CheckFp8IfOnHopper(hlo_text);
  RunAndFilecheckHloRewrite(hlo_text,
                            GemmRewriter(se::CudaComputeCapability{
                                se::CudaComputeCapability::HOPPER, 0}),
                            R"(
    ; CHECK:           custom_call_target="__cublas$lt$matmul$f8",
          )");
  tsl::enable_tensor_float_32_execution(tf32_state_);
}

INSTANTIATE_TEST_SUITE_P(Fp8CublasTestsBothLegacyAndLt,
                         ParameterizedFp8GemmRewriteTest, ::testing::Bool());

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
// CHECK: \"beta\":0
  )");
}

class GemmRewriteAllocationTest : public GpuCodegenTest {
 public:
  void CheckNumberOfAllocations(const std::string& hlo,
                                int expected_number_of_allocations) {
    TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> optimized_module,
                            GetOptimizedModule(hlo));
    TF_ASSERT_OK_AND_ASSIGN(
        std::unique_ptr<Executable> executable,
        backend().compiler()->RunBackend(
            std::move(optimized_module), backend().default_stream_executor(),
            backend().default_stream_executor()->GetAllocator()));
    GpuExecutable* gpu_executable =
        static_cast<GpuExecutable*>(executable.get());
    absl::Span<const BufferAllocation> allocations =
        gpu_executable->GetAllocations();
    CHECK_EQ(allocations.size(), expected_number_of_allocations);
  }
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
  CheckNumberOfAllocations(hlo_text, 3);
  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{1e-5, 1e-5}));
}

}  // namespace
}  // namespace gpu
}  // namespace xla

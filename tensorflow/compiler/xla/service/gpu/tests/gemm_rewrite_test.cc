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

#include <utility>

#include "tensorflow/compiler/xla/service/gpu/tests/gpu_codegen_test.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_module_config.h"
#include "tensorflow/compiler/xla/service/hlo_parser.h"
#include "tensorflow/compiler/xla/tests/filecheck.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/stream_executor/lib/statusor.h"

namespace xla {
namespace gpu {

namespace {

class GemmRewriteTest : public GpuCodegenTest {
 public:
  void MatchOptimizedHlo(const std::string& hlo, const std::string& pattern) {
    HloModuleConfig config;
    TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                            ParseHloString(hlo, config));
    TF_ASSERT_OK_AND_ASSIGN(
        std::unique_ptr<HloModule> optimized_module,
        backend().compiler()->RunHloPasses(
            std::move(module), backend().default_stream_executor(),
            /*device_allocator=*/
            backend().default_stream_executor()->GetAllocator()));

    HloPrintOptions print_opts;
    print_opts.set_print_operand_shape(false);
    StatusOr<bool> filecheck_result =
        RunFileCheck(optimized_module->ToString(print_opts), pattern);
    TF_ASSERT_OK(filecheck_result.status());
    EXPECT_TRUE(filecheck_result.ValueOrDie());
  }
};

TEST_F(GemmRewriteTest, SimpleRewrite) {
  const char* hlo_text = R"(
HloModule SimpleGemm

ENTRY AddDotsFunc {
  x = f32[2,2] parameter(0)
  y = f32[2,2] parameter(1)
  ROOT dot_a = f32[2,2] dot(x, y), lhs_contracting_dims={1}, rhs_contracting_dims={0}
}

)";

  MatchOptimizedHlo(hlo_text,
                    R"(

; CHECK-LABEL: ENTRY %AddDotsFunc (x: f32[2,2], y: f32[2,2]) -> f32[2,2] {
; CHECK-NEXT:    %x = f32[2,2]{1,0} parameter(0)
; CHECK-NEXT:    %y = f32[2,2]{1,0} parameter(1)
; CHECK-NEXT:    ROOT %custom-call = f32[2,2]{1,0} custom-call(%x, %y), custom_call_target="__cublas$gemm", backend_config="{selected_algorithm:{{[0-9]+}},alpha:1,dot_dimension_numbers:{lhs_contracting_dimensions:[1],rhs_contracting_dimensions:[0],lhs_batch_dimensions:[],rhs_batch_dimensions:[]},batch_size:1}"
      )");
}

TEST_F(GemmRewriteTest, ArgTransposeFoldCheck) {
  const char* hlo_text = R"(
HloModule ArgTransposeFoldGemm

ENTRY AddDotsFunc {
  x = f32[2,2] parameter(0)
  y = f32[2,2] parameter(1)
  x_transposed = f32[2,2] transpose(x), dimensions={1, 0}
  ROOT dot_a = f32[2,2] dot(x_transposed, y), lhs_contracting_dims={1}, rhs_contracting_dims={0}
}

)";

  MatchOptimizedHlo(hlo_text,
                    R"(

; CHECK-LABEL: ENTRY %AddDotsFunc (x: f32[2,2], y: f32[2,2]) -> f32[2,2] {
; CHECK-NEXT:    %x = f32[2,2]{1,0} parameter(0)
; CHECK-NEXT:    %y = f32[2,2]{1,0} parameter(1)
; CHECK-NEXT:    ROOT %custom-call = f32[2,2]{1,0} custom-call(%x, %y), custom_call_target="__cublas$gemm", backend_config="{selected_algorithm:{{[0-9]+}},alpha:1,dot_dimension_numbers:{lhs_contracting_dimensions:[0],rhs_contracting_dimensions:[0],lhs_batch_dimensions:[],rhs_batch_dimensions:[]},batch_size:1}"
      )");
}

TEST_F(GemmRewriteTest, InstrTransposeFoldCheck) {
  const char* hlo_text = R"(
HloModule InstrTransposeFoldGemm

ENTRY AddDotsFunc {
  x = f32[2,2] parameter(0)
  y = f32[2,2] parameter(1)
  dot_a = f32[2,2] dot(x, y), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  ROOT out = f32[2,2] transpose(dot_a), dimensions={1, 0}
}

)";

  MatchOptimizedHlo(hlo_text,
                    R"(

; CHECK-LABEL: ENTRY %AddDotsFunc (x: f32[2,2], y: f32[2,2]) -> f32[2,2] {
; CHECK-NEXT:    %y = f32[2,2]{1,0} parameter(1)
; CHECK-NEXT:    %x = f32[2,2]{1,0} parameter(0)
; CHECK-NEXT:    ROOT %custom-call = f32[2,2]{1,0} custom-call(%y, %x), custom_call_target="__cublas$gemm", backend_config="{selected_algorithm:{{[0-9]+}},alpha:1,dot_dimension_numbers:{lhs_contracting_dimensions:[0],rhs_contracting_dimensions:[1],lhs_batch_dimensions:[],rhs_batch_dimensions:[]},batch_size:1}"
      )");
}

TEST_F(GemmRewriteTest, AlphaSimpleRewrite) {
  const char* hlo_text = R"(
HloModule NonZeroAlpha

ENTRY AddDotsFunc {
  x = f32[2,2] parameter(0)
  y = f32[2,2] parameter(1)
  k = f32[] constant(3.0)
  k_broadcast = f32[2, 2] broadcast(k), dimensions={}
  dot_a = f32[2,2] dot(x, y), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  ROOT dot_a_multiplied = f32[2, 2] multiply(dot_a, k_broadcast)
}

)";

  MatchOptimizedHlo(hlo_text,
                    R"(

; CHECK-LABEL: ENTRY %AddDotsFunc (x: f32[2,2], y: f32[2,2]) -> f32[2,2] {
; CHECK-NEXT:    %x = f32[2,2]{1,0} parameter(0)
; CHECK-NEXT:    %y = f32[2,2]{1,0} parameter(1)
; CHECK-NEXT:    ROOT %custom-call = f32[2,2]{1,0} custom-call(%x, %y), custom_call_target="__cublas$gemm", backend_config="{selected_algorithm:{{[0-9]+}},alpha:3,dot_dimension_numbers:{lhs_contracting_dimensions:[1],rhs_contracting_dimensions:[0],lhs_batch_dimensions:[],rhs_batch_dimensions:[]},batch_size:1}"
      )");
}

TEST_F(GemmRewriteTest, AlphaVectorNoRewrite) {
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

  MatchOptimizedHlo(hlo_text,
                    R"(

; CHECK-LABEL: ENTRY %AddDotsFunc (x: f32[2,2], y: f32[2,2]) -> f32[2,2] {
; CHECK-NEXT:    %x = f32[2,2]{1,0} parameter(0)
; CHECK-NEXT:    %y = f32[2,2]{1,0} parameter(1)
; CHECK-NEXT:    %custom-call = f32[2,2]{1,0} custom-call(%x, %y), custom_call_target="__cublas$gemm", backend_config="{selected_algorithm:{{[0-9]+}},alpha:1,dot_dimension_numbers:{lhs_contracting_dimensions:[1],rhs_contracting_dimensions:[0],lhs_batch_dimensions:[],rhs_batch_dimensions:[]},batch_size:1}"
      )");
}

TEST_F(GemmRewriteTest, AlphaBetaRewrite) {
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

  MatchOptimizedHlo(hlo_text,
                    R"(

; CHECK-LABEL: ENTRY %AddDotsFunc (x: f32[2,2], y: f32[2,2], bias: f32[2,2]) -> f32[2,2] {
; CHECK-NEXT:    %x = f32[2,2]{1,0} parameter(0)
; CHECK-NEXT:    %y = f32[2,2]{1,0} parameter(1)
; CHECK-NEXT:    %bias = f32[2,2]{1,0} parameter(2)
; CHECK-NEXT:    ROOT %custom-call.1 = f32[2,2]{1,0} custom-call(%x, %y, %bias), custom_call_target="__cublas$gemm", backend_config="{selected_algorithm:{{[0-9]+}},alpha:3,beta:1,dot_dimension_numbers:{lhs_contracting_dimensions:[1],rhs_contracting_dimensions:[0],lhs_batch_dimensions:[],rhs_batch_dimensions:[]},batch_size:1}"
      )");
}

TEST_F(GemmRewriteTest, BiasMultipleUsersNoRewrite) {
  const char* hlo_text = R"(
HloModule BiasMultipleUsersNoRewrite

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

  MatchOptimizedHlo(hlo_text,
                    R"(

; CHECK-LABEL: ENTRY %AddDotsFunc (x: f32[2,2], y: f32[2,2], bias: f32[2,2]) -> f32[2,2] {
; CHECK-NEXT:    %bias = f32[2,2]{1,0} parameter(2)
; CHECK-NEXT:    %x = f32[2,2]{1,0} parameter(0)
; CHECK-NEXT:    %y = f32[2,2]{1,0} parameter(1)
; CHECK-NEXT:    %custom-call = f32[2,2]{1,0} custom-call(%x, %y), custom_call_target="__cublas$gemm", backend_config="{selected_algorithm:{{[0-9]+}},alpha:3,dot_dimension_numbers:{lhs_contracting_dimensions:[1],rhs_contracting_dimensions:[0],lhs_batch_dimensions:[],rhs_batch_dimensions:[]},batch_size:1}"
      )");
}

TEST_F(GemmRewriteTest, BiasDifferentLayoutNoRewrite) {
  const char* hlo_text = R"(
HloModule BiasDifferentLayoutNoRewrite

ENTRY AddDotsFunc {
  x = f32[2,2]{1,0} parameter(0)
  y = f32[2,2]{1,0} parameter(1)
  bias = f32[2,2]{0,1} parameter(2)
  dot = f32[2,2] dot(x, y), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  ROOT out = f32[2,2] add(dot, bias)
}

)";

  MatchOptimizedHlo(hlo_text,
                    R"(

; CHECK-LABEL: ENTRY %AddDotsFunc (x: f32[2,2], y: f32[2,2], bias: f32[2,2]) -> f32[2,2] {
; CHECK-NEXT:    %x = f32[2,2]{1,0} parameter(0)
; CHECK-NEXT:    %y = f32[2,2]{1,0} parameter(1)
; CHECK-NEXT:    %custom-call = f32[2,2]{1,0} custom-call(%x, %y), custom_call_target="__cublas$gemm", backend_config="{selected_algorithm:{{[0-9]+}},alpha:1,dot_dimension_numbers:{lhs_contracting_dimensions:[1],rhs_contracting_dimensions:[0],lhs_batch_dimensions:[],rhs_batch_dimensions:[]},batch_size:1}"
      )");
}

}  // namespace
}  // namespace gpu
}  // namespace xla

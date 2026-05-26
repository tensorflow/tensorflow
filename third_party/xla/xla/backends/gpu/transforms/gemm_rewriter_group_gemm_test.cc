/* Copyright 2024 The OpenXLA Authors.

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
#include <utility>

#include <gtest/gtest.h>
#include "xla/backends/gpu/transforms/gemm_rewriter_test_lib.h"
#include "xla/error_spec.h"
#include "xla/service/hlo_module_config.h"
#include "xla/stream_executor/semantic_version.h"
#include "xla/tests/hlo_pjrt_interpreter_reference_mixin.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/xla.pb.h"

namespace xla {
namespace gpu {

namespace {

class GroupedGemmRewriteTest
    : public HloPjRtInterpreterReferenceMixin<GemmRewriteTestBase> {
 public:
  DebugOptions GetDebugOptionsForTest() const override {
    DebugOptions debug_options = GemmRewriteTestBase::GetDebugOptionsForTest();
    debug_options.set_xla_gpu_experimental_use_ragged_dot_grouped_gemm(true);
    debug_options.set_xla_gpu_enable_cublaslt(true);
    // Disable autotuning by default for grouped GEMM tests
    debug_options.set_xla_gpu_autotune_level(0);
    return debug_options;
  }

 protected:
  void SetUp() override {
    if (SkipGroupedGemmTest()) {
      GTEST_SKIP()
          << "Grouped GEMM is only supported on ROCm with hipBLASLt on "
             "gfx942 or gfx950";
    }
  }
};

TEST_F(GroupedGemmRewriteTest, CustomCallTargetGroupedGemm) {
  const char* hlo_text = R"(
HloModule GroupedGemm

ENTRY AddRaggedDotsFunc {
    p0 = f16[64,9]{1,0} parameter(0)
    p1 = f16[2,9,8]{2,1,0} parameter(1)
    p2 = s32[2] constant({16, 48})
    ROOT ragged-dot = f16[64,8]{1,0} ragged-dot(p0, p1, p2),
                      lhs_contracting_dims={1}, rhs_contracting_dims={1},
                      lhs_ragged_dims={0}, rhs_group_dims={0}
}
)";
  MatchOptimizedHlo(hlo_text,
                    R"(
                    ; CHECK: custom_call_target="__cublas$lt$groupedMatmul",
                    ; CHECK-SAME: backend_config={"operation_queue_id":"0",
                    ; CHECK-SAME: "force_earliest_schedule":false,"reification_cost":[],
                    ; CHECK-SAME: "device_type":"DEVICE_TYPE_INVALID",
                    ; CHECK-SAME: "grouped_gemm_backend_config":{
                    ; CHECK-SAME: "gemm_backend_config":{
                    ; CHECK-SAME: "dot_dimension_numbers":{"lhs_contracting_dimensions":["1"],
                    ; CHECK-SAME: "rhs_contracting_dimensions":["1"],"lhs_batch_dimensions":[],
                    ; CHECK-SAME: "rhs_batch_dimensions":[]},"alpha_imag":0,"epilogue":"DEFAULT",
                    ; CHECK-SAME: "grad_x":false,"grad_y":false,
                    ; CHECK-SAME: "damax_output":false,"autotune_workspace_size":"{{[0-9]+}}",
                    ; CHECK-SAME: "ragged_dot_dimension_numbers":{"dot_dimension_numbers":{
                    ; CHECK-SAME: "lhs_contracting_dimensions":["1"],
                    ; CHECK-SAME: "rhs_contracting_dimensions":["1"],
                    ; CHECK-SAME: "lhs_batch_dimensions":[],
                    ; CHECK-SAME: "rhs_batch_dimensions":[]},
                    ; CHECK-SAME: "lhs_ragged_dimensions":["0"],
                    ; CHECK-SAME: "rhs_group_dimensions":["0"]}}})");
  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{1e-4, 1e-5}));
}

TEST_F(GroupedGemmRewriteTest, CustomCallTargetGroupedGemmLargeGroupCount600) {
  // Test with 600 groups (> 256) to test single batch processing in grid-stride
  // loop Sum of group sizes: 600 * 4 = 2400 (matches p0 first dim)
  const char* hlo_text = R"(
HloModule GroupedGemmLarge600

ENTRY AddRaggedDotsFunc {
    p0 = f16[2400,9]{1,0} parameter(0)
    p1 = f16[600,9,8]{2,1,0} parameter(1)
    scalar = s32[] constant(4)
    p2 = s32[600] broadcast(scalar), dimensions={}
    ROOT ragged-dot = f16[2400,8]{1,0} ragged-dot(p0, p1, p2),
                      lhs_contracting_dims={1}, rhs_contracting_dims={1},
                      lhs_ragged_dims={0}, rhs_group_dims={0}
}
)";
  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{1e-2, 1e-2}));
}

TEST_F(GroupedGemmRewriteTest, CustomCallTargetGroupedGemmMulipleGroups) {
  const char* hlo_text = R"(
HloModule GroupedGemm

ENTRY AddRaggedDotsFunc {
    p0 = f16[64,9]{1,0} parameter(0)
    p1 = f16[4,9,8]{2,1,0} parameter(1)
    p2 = s64[4] constant({16, 8, 24, 16})
    ROOT ragged-dot = f16[64,8]{1,0} ragged-dot(p0, p1, p2),
                      lhs_contracting_dims={1}, rhs_contracting_dims={1},
                      lhs_ragged_dims={0}, rhs_group_dims={0}
}
)";
  MatchOptimizedHlo(hlo_text,
                    R"(
                    ; CHECK: custom_call_target="__cublas$lt$groupedMatmul",
                    ; CHECK-SAME: backend_config={"operation_queue_id":"0",
                    ; CHECK-SAME: "force_earliest_schedule":false,"reification_cost":[],
                    ; CHECK-SAME: "device_type":"DEVICE_TYPE_INVALID",
                    ; CHECK-SAME: "grouped_gemm_backend_config":{
                    ; CHECK-SAME: "gemm_backend_config":{
                    ; CHECK-SAME: "dot_dimension_numbers":{"lhs_contracting_dimensions":["1"],
                    ; CHECK-SAME: "rhs_contracting_dimensions":["1"],"lhs_batch_dimensions":[],
                    ; CHECK-SAME: "rhs_batch_dimensions":[]},"alpha_imag":0,"epilogue":"DEFAULT",
                    ; CHECK-SAME: "grad_x":false,"grad_y":false,
                    ; CHECK-SAME: "damax_output":false,"autotune_workspace_size":"{{[0-9]+}}",
                    ; CHECK-SAME: "ragged_dot_dimension_numbers":{"dot_dimension_numbers":{
                    ; CHECK-SAME: "lhs_contracting_dimensions":["1"],
                    ; CHECK-SAME: "rhs_contracting_dimensions":["1"],
                    ; CHECK-SAME: "lhs_batch_dimensions":[],
                    ; CHECK-SAME: "rhs_batch_dimensions":[]},
                    ; CHECK-SAME: "lhs_ragged_dimensions":["0"],
                    ; CHECK-SAME: "rhs_group_dimensions":["0"]}}})");
  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{1e-4, 1e-5}));
}

TEST_F(GroupedGemmRewriteTest,
       CustomCallTargetGroupedGemmMulipleGroupsOutputColumnMajor) {
  const char* hlo_text = R"(
HloModule GroupedGemm

ENTRY AddRaggedDotsFunc {
    p0 = f16[64,9]{1,0} parameter(0)
    p1 = f16[4,9,8]{2,1,0} parameter(1)
    p2 = s64[4] constant({16, 8, 24, 16})
    ROOT ragged-dot = f16[64,8]{0,1} ragged-dot(p0, p1, p2),
                      lhs_contracting_dims={1}, rhs_contracting_dims={1},
                      lhs_ragged_dims={0}, rhs_group_dims={0}
}
)";
  MatchOptimizedHlo(hlo_text,
                    R"(
                    ; CHECK: custom_call_target="__cublas$lt$groupedMatmul",
                    ; CHECK-SAME: backend_config={"operation_queue_id":"0",
                    ; CHECK-SAME: "force_earliest_schedule":false,"reification_cost":[],
                    ; CHECK-SAME: "device_type":"DEVICE_TYPE_INVALID",
                    ; CHECK-SAME: "grouped_gemm_backend_config":{
                    ; CHECK-SAME: "gemm_backend_config":{
                    ; CHECK-SAME: "dot_dimension_numbers":{"lhs_contracting_dimensions":["1"],
                    ; CHECK-SAME: "rhs_contracting_dimensions":["1"],"lhs_batch_dimensions":[],
                    ; CHECK-SAME: "rhs_batch_dimensions":[]},"alpha_imag":0,"epilogue":"DEFAULT",
                    ; CHECK-SAME: "grad_x":false,"grad_y":false,
                    ; CHECK-SAME: "damax_output":false,"autotune_workspace_size":"{{[0-9]+}}",
                    ; CHECK-SAME: "ragged_dot_dimension_numbers":{"dot_dimension_numbers":{
                    ; CHECK-SAME: "lhs_contracting_dimensions":["1"],
                    ; CHECK-SAME: "rhs_contracting_dimensions":["1"],
                    ; CHECK-SAME: "lhs_batch_dimensions":[],
                    ; CHECK-SAME: "rhs_batch_dimensions":[]},
                    ; CHECK-SAME: "lhs_ragged_dimensions":["0"],
                    ; CHECK-SAME: "rhs_group_dimensions":["0"]}}})");
  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{1e-4, 1e-5}));
}

TEST_F(GroupedGemmRewriteTest,
       CustomCallTargetGroupedGemmNonContractingWithBatchDim) {
  const char* hlo_text = R"(
HloModule GroupedGemm

ENTRY AddRaggedDotsFunc {
    p0 = f16[3,16,9]{2,1,0} parameter(0)
    p1 = f16[3,2,9,8]{3,2,1,0} parameter(1)
    p2 = s64[3, 2] constant({{4, 12}, {4, 12}, {4, 12}})
    ROOT ragged-dot = f16[3,16,8]{2,1,0} ragged-dot(p0, p1, p2),
                      lhs_contracting_dims={2}, rhs_contracting_dims={2},
                      lhs_batch_dims={0}, rhs_batch_dims={0},
                      lhs_ragged_dims={1}, rhs_group_dims={1}
}
)";
  MatchOptimizedHlo(hlo_text,
                    R"(
                    ; CHECK: custom_call_target="__cublas$lt$groupedMatmul",
                    ; CHECK-SAME: backend_config={"operation_queue_id":"0",
                    ; CHECK-SAME: "force_earliest_schedule":false,"reification_cost":[],
                    ; CHECK-SAME: "device_type":"DEVICE_TYPE_INVALID",
                    ; CHECK-SAME: "grouped_gemm_backend_config":{
                    ; CHECK-SAME: "gemm_backend_config":{
                    ; CHECK-SAME: "dot_dimension_numbers":{"lhs_contracting_dimensions":["2"],
                    ; CHECK-SAME: "rhs_contracting_dimensions":["2"],"lhs_batch_dimensions":["0"],
                    ; CHECK-SAME: "rhs_batch_dimensions":["0"]},"alpha_imag":0,"epilogue":"DEFAULT",
                    ; CHECK-SAME: "grad_x":false,"grad_y":false,
                    ; CHECK-SAME: "damax_output":false,"autotune_workspace_size":"{{[0-9]+}}",
                    ; CHECK-SAME: "ragged_dot_dimension_numbers":{"dot_dimension_numbers":{
                    ; CHECK-SAME: "lhs_contracting_dimensions":["2"],
                    ; CHECK-SAME: "rhs_contracting_dimensions":["2"],
                    ; CHECK-SAME: "lhs_batch_dimensions":["0"],
                    ; CHECK-SAME: "rhs_batch_dimensions":["0"]},
                    ; CHECK-SAME: "lhs_ragged_dimensions":["1"],
                    ; CHECK-SAME: "rhs_group_dimensions":["1"]}}})");
  // Enable autotuning for RunAndCompare
  DebugOptions debug_options_with_autotune = GetDebugOptionsForTest();
  debug_options_with_autotune.set_xla_gpu_autotune_level(4);
  HloModuleConfig config;
  config.set_debug_options(debug_options_with_autotune);
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_text, config));
  EXPECT_TRUE(RunAndCompare(std::move(module), ErrorSpec{1e-4, 1e-5}));
}

TEST_F(GroupedGemmRewriteTest,
       CustomCallTargetGroupedGemmRaggedDimInContractingDim) {
  const char* hlo_text = R"(
HloModule GroupedGemm

ENTRY AddRaggedDotsFunc {
    p0 = f16[64,9]{1,0} parameter(0)
    p1 = f16[9,8]{1,0} parameter(1)
    p2 = s64[2] constant({4, 5})
    ROOT ragged-dot = f16[2,64,8]{2,1,0} ragged-dot(p0, p1, p2),
                      lhs_contracting_dims={1}, rhs_contracting_dims={0},
                      lhs_ragged_dims={1}
}
)";
  MatchOptimizedHlo(hlo_text,
                    R"(
                    ; CHECK: custom_call_target="__cublas$lt$groupedMatmul",
                    ; CHECK-SAME: backend_config={"operation_queue_id":"0",
                    ; CHECK-SAME: "force_earliest_schedule":false,"reification_cost":[],
                    ; CHECK-SAME: "device_type":"DEVICE_TYPE_INVALID",
                    ; CHECK-SAME: "grouped_gemm_backend_config":{
                    ; CHECK-SAME: "gemm_backend_config":{
                    ; CHECK-SAME: "dot_dimension_numbers":{"lhs_contracting_dimensions":["1"],
                    ; CHECK-SAME: "rhs_contracting_dimensions":["0"],"lhs_batch_dimensions":[],
                    ; CHECK-SAME: "rhs_batch_dimensions":[]},"alpha_imag":0,"epilogue":"DEFAULT",
                    ; CHECK-SAME: "grad_x":false,"grad_y":false,
                    ; CHECK-SAME: "damax_output":false,"autotune_workspace_size":"{{[0-9]+}}",
                    ; CHECK-SAME: "ragged_dot_dimension_numbers":{"dot_dimension_numbers":{
                    ; CHECK-SAME: "lhs_contracting_dimensions":["1"],
                    ; CHECK-SAME: "rhs_contracting_dimensions":["0"],
                    ; CHECK-SAME: "lhs_batch_dimensions":[],
                    ; CHECK-SAME: "rhs_batch_dimensions":[]},
                    ; CHECK-SAME: "lhs_ragged_dimensions":["1"],
                    ; CHECK-SAME: "rhs_group_dimensions":[]}}})");
  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{1e-4, 1e-5}));
}

TEST_F(GroupedGemmRewriteTest,
       CustomCallTargetGroupedGemmRaggedDimInContractingDimMultipleGroups) {
  const char* hlo_text = R"(
HloModule GroupedGemm

ENTRY AddRaggedDotsFunc {
    p0 = f16[64,16]{1,0} parameter(0)
    p1 = f16[16,8]{1,0} parameter(1)
    p2 = s64[4] constant({4, 5, 3, 4})
    ROOT ragged-dot = f16[4,64,8]{2,1,0} ragged-dot(p0, p1, p2),
                      lhs_contracting_dims={1}, rhs_contracting_dims={0},
                      lhs_ragged_dims={1}
}
)";
  MatchOptimizedHlo(hlo_text,
                    R"(
                    ; CHECK: custom_call_target="__cublas$lt$groupedMatmul",
                    ; CHECK-SAME: backend_config={"operation_queue_id":"0",
                    ; CHECK-SAME: "force_earliest_schedule":false,"reification_cost":[],
                    ; CHECK-SAME: "device_type":"DEVICE_TYPE_INVALID",
                    ; CHECK-SAME: "grouped_gemm_backend_config":{
                    ; CHECK-SAME: "gemm_backend_config":{
                    ; CHECK-SAME: "dot_dimension_numbers":{"lhs_contracting_dimensions":["1"],
                    ; CHECK-SAME: "rhs_contracting_dimensions":["0"],"lhs_batch_dimensions":[],
                    ; CHECK-SAME: "rhs_batch_dimensions":[]},"alpha_imag":0,"epilogue":"DEFAULT",
                    ; CHECK-SAME: "grad_x":false,"grad_y":false,
                    ; CHECK-SAME: "damax_output":false,"autotune_workspace_size":"{{[0-9]+}}",
                    ; CHECK-SAME: "ragged_dot_dimension_numbers":{"dot_dimension_numbers":{
                    ; CHECK-SAME: "lhs_contracting_dimensions":["1"],
                    ; CHECK-SAME: "rhs_contracting_dimensions":["0"],
                    ; CHECK-SAME: "lhs_batch_dimensions":[],
                    ; CHECK-SAME: "rhs_batch_dimensions":[]},
                    ; CHECK-SAME: "lhs_ragged_dimensions":["1"],
                    ; CHECK-SAME: "rhs_group_dimensions":[]}}})");
  // Enable autotuning for RunAndCompare
  DebugOptions debug_options_with_autotune = GetDebugOptionsForTest();
  debug_options_with_autotune.set_xla_gpu_autotune_level(4);
  HloModuleConfig config;
  config.set_debug_options(debug_options_with_autotune);
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_text, config));
  EXPECT_TRUE(RunAndCompare(std::move(module), ErrorSpec{1e-4, 1e-5}));
}

TEST_F(
    GroupedGemmRewriteTest,
    CustomCallTargetGroupedGemmRaggedDimInContractingDimMultipleGroupsOutputColumnMajor) {  // NOLINT
  const char* hlo_text = R"(
HloModule GroupedGemm

ENTRY AddRaggedDotsFunc {
    p0 = f16[64,16]{1,0} parameter(0)
    p1 = f16[16,8]{1,0} parameter(1)
    p2 = s64[4] constant({4, 5, 3, 4})
    ROOT ragged-dot = f16[4,64,8]{1,2,0} ragged-dot(p0, p1, p2),
                      lhs_contracting_dims={1}, rhs_contracting_dims={0},
                      lhs_ragged_dims={1}
}
)";
  MatchOptimizedHlo(hlo_text,
                    R"(
                    ; CHECK: custom_call_target="__cublas$lt$groupedMatmul",
                    ; CHECK-SAME: backend_config={"operation_queue_id":"0",
                    ; CHECK-SAME: "force_earliest_schedule":false,"reification_cost":[],
                    ; CHECK-SAME: "device_type":"DEVICE_TYPE_INVALID",
                    ; CHECK-SAME: "grouped_gemm_backend_config":{
                    ; CHECK-SAME: "gemm_backend_config":{
                    ; CHECK-SAME: "dot_dimension_numbers":{"lhs_contracting_dimensions":["1"],
                    ; CHECK-SAME: "rhs_contracting_dimensions":["0"],"lhs_batch_dimensions":[],
                    ; CHECK-SAME: "rhs_batch_dimensions":[]},"alpha_imag":0,"epilogue":"DEFAULT",
                    ; CHECK-SAME: "grad_x":false,"grad_y":false,
                    ; CHECK-SAME: "damax_output":false,"autotune_workspace_size":"{{[0-9]+}}",
                    ; CHECK-SAME: "ragged_dot_dimension_numbers":{"dot_dimension_numbers":{
                    ; CHECK-SAME: "lhs_contracting_dimensions":["1"],
                    ; CHECK-SAME: "rhs_contracting_dimensions":["0"],
                    ; CHECK-SAME: "lhs_batch_dimensions":[],
                    ; CHECK-SAME: "rhs_batch_dimensions":[]},
                    ; CHECK-SAME: "lhs_ragged_dimensions":["1"],
                    ; CHECK-SAME: "rhs_group_dimensions":[]}}})");
  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{1e-4, 1e-5}));
}

TEST_F(GroupedGemmRewriteTest,
       CustomCallTargetGroupedGemmRaggedDimInContractingDimLargeGroupCount600) {
  // Test with 600 groups (> 256) for RaggedDimInContractingDim
  // Sum of group sizes: 600 * 4 = 2400 (matches p1 first dim for contracting)
  const char* hlo_text = R"(
HloModule GroupedGemmRaggedContractingLarge600

ENTRY AddRaggedDotsFunc {
    p0 = f16[64,2400]{1,0} parameter(0)
    p1 = f16[2400,8]{1,0} parameter(1)
    scalar = s64[] constant(4)
    p2 = s64[600] broadcast(scalar), dimensions={}
    ROOT ragged-dot = f16[600,64,8]{2,1,0} ragged-dot(p0, p1, p2),
                      lhs_contracting_dims={1}, rhs_contracting_dims={0},
                      lhs_ragged_dims={1}
}
)";
  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{1e-2, 1e-2}));
}

TEST_F(GroupedGemmRewriteTest,
       CustomCallTargetGroupedGemmRaggedDimInContractingDimWithBatchDim) {
  const char* hlo_text = R"(
HloModule GroupedGemm

ENTRY AddRaggedDotsFunc {
    p0 = f16[3,64,9]{2,1,0} parameter(0)
    p1 = f16[3,9,8]{2,1,0} parameter(1)
    p2 = s64[3,2] constant({{4, 5}, {4, 5}, {4, 5}})
    ROOT ragged-dot = f16[2,3,64,8]{3,2,1,0} ragged-dot(p0, p1, p2),
                      lhs_contracting_dims={2}, rhs_contracting_dims={1},
                      lhs_ragged_dims={2}, lhs_batch_dims={0}, rhs_batch_dims={0}
}
)";
  MatchOptimizedHlo(hlo_text,
                    R"(
                    ; CHECK: custom_call_target="__cublas$lt$groupedMatmul",
                    ; CHECK-SAME: backend_config={"operation_queue_id":"0",
                    ; CHECK-SAME: "force_earliest_schedule":false,"reification_cost":[],
                    ; CHECK-SAME: "device_type":"DEVICE_TYPE_INVALID",
                    ; CHECK-SAME: "grouped_gemm_backend_config":{
                    ; CHECK-SAME: "gemm_backend_config":{
                    ; CHECK-SAME: "dot_dimension_numbers":{"lhs_contracting_dimensions":["2"],
                    ; CHECK-SAME: "rhs_contracting_dimensions":["1"],"lhs_batch_dimensions":["0"],
                    ; CHECK-SAME: "rhs_batch_dimensions":["0"]},"alpha_imag":0,"epilogue":"DEFAULT",
                    ; CHECK-SAME: "grad_x":false,"grad_y":false,
                    ; CHECK-SAME: "damax_output":false,"autotune_workspace_size":"{{[0-9]+}}",
                    ; CHECK-SAME: "ragged_dot_dimension_numbers":{"dot_dimension_numbers":{
                    ; CHECK-SAME: "lhs_contracting_dimensions":["2"],
                    ; CHECK-SAME: "rhs_contracting_dimensions":["1"],
                    ; CHECK-SAME: "lhs_batch_dimensions":["0"],
                    ; CHECK-SAME: "rhs_batch_dimensions":["0"]},
                    ; CHECK-SAME: "lhs_ragged_dimensions":["2"],
                    ; CHECK-SAME: "rhs_group_dimensions":[]}}})");
  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{1e-4, 1e-5}));
}

TEST_F(GroupedGemmRewriteTest, CustomCallTargetGroupedGemmRaggedDimInBatchDim) {
  const char* hlo_text = R"(
HloModule GroupedGemm

ENTRY AddRaggedDotsFunc {
    p0 = f16[5,16,9]{2,1,0} parameter(0)
    p1 = f16[5,9,8]{2,1,0} parameter(1)
    p2 = s64[2] constant({3, 2})
    ROOT ragged-dot = f16[5,16,8]{2,1,0} ragged-dot(p0, p1, p2),
                      lhs_contracting_dims={2}, rhs_contracting_dims={1},
                      lhs_ragged_dims={0}, lhs_batch_dims={0}, rhs_batch_dims={0}
}
)";
  MatchOptimizedHlo(hlo_text,
                    R"(
                    ; CHECK: custom_call_target="__cublas$lt$groupedMatmul",
                    ; CHECK-SAME: backend_config={"operation_queue_id":"0",
                    ; CHECK-SAME: "force_earliest_schedule":false,"reification_cost":[],
                    ; CHECK-SAME: "device_type":"DEVICE_TYPE_INVALID",
                    ; CHECK-SAME: "grouped_gemm_backend_config":{
                    ; CHECK-SAME: "gemm_backend_config":{
                    ; CHECK-SAME: "dot_dimension_numbers":{"lhs_contracting_dimensions":["2"],
                    ; CHECK-SAME: "rhs_contracting_dimensions":["1"],"lhs_batch_dimensions":["0"],
                    ; CHECK-SAME: "rhs_batch_dimensions":["0"]},"alpha_imag":0,"epilogue":"DEFAULT",
                    ; CHECK-SAME: "grad_x":false,"grad_y":false,
                    ; CHECK-SAME: "damax_output":false,"autotune_workspace_size":"{{[0-9]+}}",
                    ; CHECK-SAME: "ragged_dot_dimension_numbers":{"dot_dimension_numbers":{
                    ; CHECK-SAME: "lhs_contracting_dimensions":["2"],
                    ; CHECK-SAME: "rhs_contracting_dimensions":["1"],
                    ; CHECK-SAME: "lhs_batch_dimensions":["0"],
                    ; CHECK-SAME: "rhs_batch_dimensions":["0"]},
                    ; CHECK-SAME: "lhs_ragged_dimensions":["0"],
                    ; CHECK-SAME: "rhs_group_dimensions":[]}}})");
  // Enable autotuning for RunAndCompare
  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{1e-3, 1e-3}));
}

TEST_F(GroupedGemmRewriteTest,
       CustomCallTargetGroupedGemmRaggedDimInBatchDimMultipleGroups) {
  const char* hlo_text = R"(
HloModule GroupedGemm

ENTRY AddRaggedDotsFunc {
    p0 = f16[16,64,9]{2,1,0} parameter(0)
    p1 = f16[16,9,8]{2,1,0} parameter(1)
    p2 = s64[4] constant({4, 2, 6, 4})
    ROOT ragged-dot = f16[16,64,8]{2,1,0} ragged-dot(p0, p1, p2),
                      lhs_contracting_dims={2}, rhs_contracting_dims={1},
                      lhs_ragged_dims={0}, lhs_batch_dims={0}, rhs_batch_dims={0}
}
)";
  MatchOptimizedHlo(hlo_text,
                    R"(
                    ; CHECK: custom_call_target="__cublas$lt$groupedMatmul",
                    ; CHECK-SAME: backend_config={"operation_queue_id":"0",
                    ; CHECK-SAME: "force_earliest_schedule":false,"reification_cost":[],
                    ; CHECK-SAME: "device_type":"DEVICE_TYPE_INVALID",
                    ; CHECK-SAME: "grouped_gemm_backend_config":{
                    ; CHECK-SAME: "gemm_backend_config":{
                    ; CHECK-SAME: "dot_dimension_numbers":{"lhs_contracting_dimensions":["2"],
                    ; CHECK-SAME: "rhs_contracting_dimensions":["1"],"lhs_batch_dimensions":["0"],
                    ; CHECK-SAME: "rhs_batch_dimensions":["0"]},"alpha_imag":0,"epilogue":"DEFAULT",
                    ; CHECK-SAME: "grad_x":false,"grad_y":false,
                    ; CHECK-SAME: "damax_output":false,"autotune_workspace_size":"{{[0-9]+}}",
                    ; CHECK-SAME: "ragged_dot_dimension_numbers":{"dot_dimension_numbers":{
                    ; CHECK-SAME: "lhs_contracting_dimensions":["2"],
                    ; CHECK-SAME: "rhs_contracting_dimensions":["1"],
                    ; CHECK-SAME: "lhs_batch_dimensions":["0"],
                    ; CHECK-SAME: "rhs_batch_dimensions":["0"]},
                    ; CHECK-SAME: "lhs_ragged_dimensions":["0"],
                    ; CHECK-SAME: "rhs_group_dimensions":[]}}})");
  // Enable autotuning for RunAndCompare
  DebugOptions debug_options_with_autotune = GetDebugOptionsForTest();
  debug_options_with_autotune.set_xla_gpu_autotune_level(4);
  HloModuleConfig config;
  config.set_debug_options(debug_options_with_autotune);
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_text, config));
  EXPECT_TRUE(RunAndCompare(std::move(module), ErrorSpec{1e-4, 1e-5}));
}

TEST_F(
    GroupedGemmRewriteTest,
    CustomCallTargetGroupedGemmRaggedDimInBatchDimMultipleGroupsOutputColumnMajor) {  // NOLINT
  const char* hlo_text = R"(
HloModule GroupedGemm

ENTRY AddRaggedDotsFunc {
    p0 = f16[16,64,9]{2,1,0} parameter(0)
    p1 = f16[16,9,8]{2,1,0} parameter(1)
    p2 = s64[4] constant({4, 2, 6, 4})
    ROOT ragged-dot = f16[16,64,8]{1,2,0} ragged-dot(p0, p1, p2),
                      lhs_contracting_dims={2}, rhs_contracting_dims={1},
                      lhs_ragged_dims={0}, lhs_batch_dims={0}, rhs_batch_dims={0}
}
)";
  MatchOptimizedHlo(hlo_text,
                    R"(
                    ; CHECK: custom_call_target="__cublas$lt$groupedMatmul",
                    ; CHECK-SAME: backend_config={"operation_queue_id":"0",
                    ; CHECK-SAME: "force_earliest_schedule":false,"reification_cost":[],
                    ; CHECK-SAME: "device_type":"DEVICE_TYPE_INVALID",
                    ; CHECK-SAME: "grouped_gemm_backend_config":{
                    ; CHECK-SAME: "gemm_backend_config":{
                    ; CHECK-SAME: "dot_dimension_numbers":{"lhs_contracting_dimensions":["2"],
                    ; CHECK-SAME: "rhs_contracting_dimensions":["1"],"lhs_batch_dimensions":["0"],
                    ; CHECK-SAME: "rhs_batch_dimensions":["0"]},"alpha_imag":0,"epilogue":"DEFAULT",
                    ; CHECK-SAME: "grad_x":false,"grad_y":false,
                    ; CHECK-SAME: "damax_output":false,"autotune_workspace_size":"{{[0-9]+}}",
                    ; CHECK-SAME: "ragged_dot_dimension_numbers":{"dot_dimension_numbers":{
                    ; CHECK-SAME: "lhs_contracting_dimensions":["2"],
                    ; CHECK-SAME: "rhs_contracting_dimensions":["1"],
                    ; CHECK-SAME: "lhs_batch_dimensions":["0"],
                    ; CHECK-SAME: "rhs_batch_dimensions":["0"]},
                    ; CHECK-SAME: "lhs_ragged_dimensions":["0"],
                    ; CHECK-SAME: "rhs_group_dimensions":[]}}})");
  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{1e-4, 1e-5}));
}

TEST_F(GroupedGemmRewriteTest,
       CustomCallTargetGroupedGemmRaggedDimInBatchDimLargeGroupCount600) {
  // Test with 600 groups (> 256) for RaggedDimInBatchDim
  // Sum of group sizes: 600 * 4 = 2400 (matches p0 first dim for batch)
  const char* hlo_text = R"(
HloModule GroupedGemmRaggedBatchLarge600

ENTRY AddRaggedDotsFunc {
    p0 = f16[2400,64,9]{2,1,0} parameter(0)
    p1 = f16[2400,9,8]{2,1,0} parameter(1)
    scalar = s64[] constant(4)
    p2 = s64[600] broadcast(scalar), dimensions={}
    ROOT ragged-dot = f16[2400,64,8]{2,1,0} ragged-dot(p0, p1, p2),
                      lhs_contracting_dims={2}, rhs_contracting_dims={1},
                      lhs_ragged_dims={0}, lhs_batch_dims={0}, rhs_batch_dims={0}
}
)";
  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{1e-2, 1e-2}));
}

TEST_F(GroupedGemmRewriteTest,
       CustomCallTargetGroupedGemmRaggedInNonContractingGroupDimNoOuterDim) {
  const char* hlo_text = R"(
HloModule GroupedGemm

ENTRY AddRaggedDotsFunc {
    p0 = f16[128,64]{1,0} parameter(0)
    p1 = f16[8,64,32]{2,0,1} parameter(1)
    p2 = s32[8] constant({16, 16, 16, 16, 16, 16, 16, 16})
    ROOT ragged-dot = f16[128,32]{1,0} ragged-dot(p0, p1, p2),
                      lhs_contracting_dims={1}, rhs_contracting_dims={1},
                      lhs_ragged_dims={0}, rhs_group_dims={0}
}
)";
  MatchOptimizedHlo(hlo_text,
                    R"(
                    ; CHECK: custom_call_target="__cublas$lt$groupedMatmul",
                    ; CHECK-SAME: backend_config={"operation_queue_id":"0",
                    ; CHECK-SAME: "force_earliest_schedule":false,"reification_cost":[],
                    ; CHECK-SAME: "device_type":"DEVICE_TYPE_INVALID",
                    ; CHECK-SAME: "grouped_gemm_backend_config":{
                    ; CHECK-SAME: "gemm_backend_config":{
                    ; CHECK-SAME: "dot_dimension_numbers":{"lhs_contracting_dimensions":["1"],
                    ; CHECK-SAME: "rhs_contracting_dimensions":["1"],"lhs_batch_dimensions":[],
                    ; CHECK-SAME: "rhs_batch_dimensions":[]},"alpha_imag":0,"epilogue":"DEFAULT",
                    ; CHECK-SAME: "grad_x":false,"grad_y":false,
                    ; CHECK-SAME: "damax_output":false,"autotune_workspace_size":"{{[0-9]+}}",
                    ; CHECK-SAME: "ragged_dot_dimension_numbers":{"dot_dimension_numbers":{
                    ; CHECK-SAME: "lhs_contracting_dimensions":["1"],
                    ; CHECK-SAME: "rhs_contracting_dimensions":["1"],
                    ; CHECK-SAME: "lhs_batch_dimensions":[],
                    ; CHECK-SAME: "rhs_batch_dimensions":[]},
                    ; CHECK-SAME: "lhs_ragged_dimensions":["0"],
                    ; CHECK-SAME: "rhs_group_dimensions":["0"]}}})");
  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{1e-3, 1e-3}));
}

TEST_F(GroupedGemmRewriteTest,
       CustomCallTargetGroupedGemmRaggedInContractingGroupDimNoOuterDim) {
  const char* hlo_text = R"(
HloModule GroupedGemm

ENTRY AddRaggedDotsFunc {
    p0 = f16[64,1024]{1,0} parameter(0)
    p1 = f16[1024,256]{1,0} parameter(1)
    p2 = s32[8] constant({128, 128, 128, 128, 128, 128, 128, 128})
    ROOT ragged-dot = f16[8,64,256]{2,0,1} ragged-dot(p0, p1, p2),
                      lhs_contracting_dims={1}, rhs_contracting_dims={0},
                      lhs_ragged_dims={1}
}
)";
  MatchOptimizedHlo(hlo_text,
                    R"(
                    ; CHECK: custom_call_target="__cublas$lt$groupedMatmul",
                    ; CHECK-SAME: backend_config={"operation_queue_id":"0",
                    ; CHECK-SAME: "force_earliest_schedule":false,"reification_cost":[],
                    ; CHECK-SAME: "device_type":"DEVICE_TYPE_INVALID",
                    ; CHECK-SAME: "grouped_gemm_backend_config":{
                    ; CHECK-SAME: "gemm_backend_config":{
                    ; CHECK-SAME: "dot_dimension_numbers":{"lhs_contracting_dimensions":["1"],
                    ; CHECK-SAME: "rhs_contracting_dimensions":["0"],"lhs_batch_dimensions":[],
                    ; CHECK-SAME: "rhs_batch_dimensions":[]},"alpha_imag":0,"epilogue":"DEFAULT",
                    ; CHECK-SAME: "grad_x":false,"grad_y":false,
                    ; CHECK-SAME: "damax_output":false,"autotune_workspace_size":"{{[0-9]+}}",
                    ; CHECK-SAME: "ragged_dot_dimension_numbers":{"dot_dimension_numbers":{
                    ; CHECK-SAME: "lhs_contracting_dimensions":["1"],
                    ; CHECK-SAME: "rhs_contracting_dimensions":["0"],
                    ; CHECK-SAME: "lhs_batch_dimensions":[],
                    ; CHECK-SAME: "rhs_batch_dimensions":[]},
                    ; CHECK-SAME: "lhs_ragged_dimensions":["1"],
                    ; CHECK-SAME: "rhs_group_dimensions":[]}}})");
  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{1e-2, 1e-2}));
}

TEST_F(GroupedGemmRewriteTest,
       CustomCallTargetGroupedGemmRaggedDimInContractingTranspose) {
  const char* hlo_text = R"(
HloModule GroupedGemm

ENTRY AddRaggedDotsFunc {
    p0 = f16[1024,64]{1,0} parameter(0)
    p1 = f16[1024,256]{1,0} parameter(1)
    p2 = s32[8] constant({128, 128, 128, 128, 128, 128, 128, 128})
    ROOT ragged-dot = f16[8,64,256]{2,0,1} ragged-dot(p0, p1, p2),
                      lhs_contracting_dims={0}, rhs_contracting_dims={0},
                      lhs_ragged_dims={0}
}
)";
  MatchOptimizedHlo(hlo_text,
                    R"(
                    ; CHECK: custom_call_target="__cublas$lt$groupedMatmul",
                    ; CHECK-SAME: backend_config={"operation_queue_id":"0",
                    ; CHECK-SAME: "force_earliest_schedule":false,"reification_cost":[],
                    ; CHECK-SAME: "device_type":"DEVICE_TYPE_INVALID",
                    ; CHECK-SAME: "grouped_gemm_backend_config":{
                    ; CHECK-SAME: "gemm_backend_config":{
                    ; CHECK-SAME: "dot_dimension_numbers":{"lhs_contracting_dimensions":["0"],
                    ; CHECK-SAME: "rhs_contracting_dimensions":["0"],"lhs_batch_dimensions":[],
                    ; CHECK-SAME: "rhs_batch_dimensions":[]},"alpha_imag":0,"epilogue":"DEFAULT",
                    ; CHECK-SAME: "grad_x":false,"grad_y":false,
                    ; CHECK-SAME: "damax_output":false,"autotune_workspace_size":"{{[0-9]+}}",
                    ; CHECK-SAME: "ragged_dot_dimension_numbers":{"dot_dimension_numbers":{
                    ; CHECK-SAME: "lhs_contracting_dimensions":["0"],
                    ; CHECK-SAME: "rhs_contracting_dimensions":["0"],
                    ; CHECK-SAME: "lhs_batch_dimensions":[],
                    ; CHECK-SAME: "rhs_batch_dimensions":[]},
                    ; CHECK-SAME: "lhs_ragged_dimensions":["0"],
                    ; CHECK-SAME: "rhs_group_dimensions":[]}}})");
  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{1e-2, 1e-2}));
}

TEST_F(GroupedGemmRewriteTest,
       CustomCallTargetGroupedGemmRaggedNonContractingTranspose) {
  if (IsRocm()) {
    const auto* rocm_cc = Capability().rocm_compute_capability();
    if (rocm_cc && rocm_cc->gfx_version() == "gfx950") {
      GTEST_SKIP()
          << "Ragged non-contracting transpose not supported on gfx950";
    }
  }
  const char* hlo_text = R"(
HloModule GroupedGemm

ENTRY AddRaggedDotsFunc {
    p0 = f16[9,64]{1,0} parameter(0)
    p1 = f16[4,9,8]{2,1,0} parameter(1)
    p2 = s64[4] constant({16, 8, 24, 16})
    ROOT ragged-dot = f16[64,8]{1,0} ragged-dot(p0, p1, p2),
                      lhs_contracting_dims={0}, rhs_contracting_dims={1},
                      lhs_ragged_dims={1}, rhs_group_dims={0}
}
)";
  MatchOptimizedHlo(hlo_text,
                    R"(
                    ; CHECK: custom_call_target="__cublas$lt$groupedMatmul",
                    ; CHECK-SAME: backend_config={"operation_queue_id":"0",
                    ; CHECK-SAME: "force_earliest_schedule":false,"reification_cost":[],
                    ; CHECK-SAME: "device_type":"DEVICE_TYPE_INVALID",
                    ; CHECK-SAME: "grouped_gemm_backend_config":{
                    ; CHECK-SAME: "gemm_backend_config":{
                    ; CHECK-SAME: "dot_dimension_numbers":{"lhs_contracting_dimensions":["0"],
                    ; CHECK-SAME: "rhs_contracting_dimensions":["1"],"lhs_batch_dimensions":[],
                    ; CHECK-SAME: "rhs_batch_dimensions":[]},"alpha_imag":0,"epilogue":"DEFAULT",
                    ; CHECK-SAME: "grad_x":false,"grad_y":false,
                    ; CHECK-SAME: "damax_output":false,"autotune_workspace_size":"{{[0-9]+}}",
                    ; CHECK-SAME: "ragged_dot_dimension_numbers":{"dot_dimension_numbers":{
                    ; CHECK-SAME: "lhs_contracting_dimensions":["0"],
                    ; CHECK-SAME: "rhs_contracting_dimensions":["1"],
                    ; CHECK-SAME: "lhs_batch_dimensions":[],
                    ; CHECK-SAME: "rhs_batch_dimensions":[]},
                    ; CHECK-SAME: "lhs_ragged_dimensions":["1"],
                    ; CHECK-SAME: "rhs_group_dimensions":["0"]}}})");
  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{1e-4, 1e-5}));
}

// Test epilogue fusion for grouped GEMM: Vector Bias
TEST_F(GroupedGemmRewriteTest, GroupedGemmVectorBias) {
  const char* hlo_text = R"(
HloModule GroupedGemmVectorBias

ENTRY test {
  p0 = f16[64,9]{1,0} parameter(0)
  p1 = f16[2,9,8]{2,1,0} parameter(1)
  p2 = s32[2] constant({16, 48})
  bias = f16[8]{0} parameter(2)
  ragged-dot = f16[64,8]{1,0} ragged-dot(p0, p1, p2),
                lhs_contracting_dims={1}, rhs_contracting_dims={1},
                lhs_ragged_dims={0}, rhs_group_dims={0}
  bias_bcast = f16[64,8]{1,0} broadcast(bias), dimensions={1}
  ROOT out = f16[64,8]{1,0} add(ragged-dot, bias_bcast)
}
)";
  MatchOptimizedHlo(hlo_text,
                    R"(
; CHECK: custom_call_target="__cublas$lt$groupedMatmul",
; CHECK-SAME: "epilogue":"BIAS"
)");
  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{1e-3, 1e-3}));
}

// Test epilogue fusion for grouped GEMM: Matrix Bias
TEST_F(GroupedGemmRewriteTest, GroupedGemmMatrixBias) {
  const char* hlo_text = R"(
HloModule GroupedGemmMatrixBias

ENTRY test {
  p0 = f16[64,9]{1,0} parameter(0)
  p1 = f16[2,9,8]{2,1,0} parameter(1)
  p2 = s32[2] constant({16, 48})
  bias = f16[64,8]{1,0} parameter(2)
  ragged-dot = f16[64,8]{1,0} ragged-dot(p0, p1, p2),
                lhs_contracting_dims={1}, rhs_contracting_dims={1},
                lhs_ragged_dims={0}, rhs_group_dims={0}
  ROOT out = f16[64,8]{1,0} add(ragged-dot, bias)
}
)";
  MatchOptimizedHlo(hlo_text,
                    R"(
; CHECK: custom_call_target="__cublas$lt$groupedMatmul",
; CHECK-SAME: "beta":1
; CHECK-SAME: "epilogue":"DEFAULT"
)");
  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{1e-3, 1e-3}));
}

// Test that vector bias is NOT fused for grouped GEMM with batch dimensions
// due to hipBLASLt limitation
TEST_F(GroupedGemmRewriteTest, GroupedGemmVectorBiasNonContractingWithBatch) {
  const char* hlo_text = R"(
HloModule GroupedGemmVectorBiasNonContractingWithBatch

ENTRY test {
  p0 = f16[3,16,9]{2,1,0} parameter(0)
  p1 = f16[3,2,9,8]{3,2,1,0} parameter(1)
  p2 = s64[3,2] constant({{4, 12}, {4, 12}, {4, 12}})
  bias = f16[8]{0} parameter(2)
  ragged-dot = f16[3,16,8]{2,1,0} ragged-dot(p0, p1, p2),
                lhs_contracting_dims={2}, rhs_contracting_dims={2},
                lhs_batch_dims={0}, rhs_batch_dims={0},
                lhs_ragged_dims={1}, rhs_group_dims={1}
  bias_bcast = f16[3,16,8]{2,1,0} broadcast(bias), dimensions={2}
  ROOT out = f16[3,16,8]{2,1,0} add(ragged-dot, bias_bcast)
}
)";
  // Verify that bias is NOT fused (epilogue should be DEFAULT, not BIAS)
  MatchOptimizedHlo(hlo_text,
                    R"(
; CHECK: custom_call_target="__cublas$lt$groupedMatmul",
; CHECK-SAME: "epilogue":"DEFAULT"
; CHECK-NOT: "epilogue":"BIAS"
)");
  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{1e-3, 1e-3}));
}

TEST_F(GroupedGemmRewriteTest, GroupedGemmMatrixBiasNonContractingWithBatch) {
  const char* hlo_text = R"(
HloModule GroupedGemmMatrixBiasNonContractingWithBatch

ENTRY test {
  p0 = f16[3,16,9]{2,1,0} parameter(0)
  p1 = f16[3,2,9,8]{3,2,1,0} parameter(1)
  p2 = s64[3,2] constant({{4, 12}, {4, 12}, {4, 12}})
  bias = f16[3,16,8]{2,1,0} parameter(2)
  ragged-dot = f16[3,16,8]{2,1,0} ragged-dot(p0, p1, p2),
                lhs_contracting_dims={2}, rhs_contracting_dims={2},
                lhs_batch_dims={0}, rhs_batch_dims={0},
                lhs_ragged_dims={1}, rhs_group_dims={1}
  ROOT out = f16[3,16,8]{2,1,0} add(ragged-dot, bias)
}
)";
  MatchOptimizedHlo(hlo_text,
                    R"(
; CHECK: custom_call_target="__cublas$lt$groupedMatmul",
; CHECK-SAME: "beta":1
; CHECK-SAME: "epilogue":"DEFAULT"
)");
  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{1e-3, 1e-3}));
}

// Bias tests for ragged dimension in contracting dim
TEST_F(GroupedGemmRewriteTest, GroupedGemmVectorBiasRaggedInContracting) {
  const char* hlo_text = R"(
HloModule GroupedGemmVectorBiasRaggedInContracting

ENTRY test {
  p0 = f16[64,16]{1,0} parameter(0)
  p1 = f16[16,8]{1,0} parameter(1)
  p2 = s64[4] constant({4, 5, 3, 4})
  bias = f16[8]{0} parameter(2)
  ragged-dot = f16[4,64,8]{2,1,0} ragged-dot(p0, p1, p2),
                lhs_contracting_dims={1}, rhs_contracting_dims={0},
                lhs_ragged_dims={1}
  bias_bcast = f16[4,64,8]{2,1,0} broadcast(bias), dimensions={2}
  ROOT out = f16[4,64,8]{2,1,0} add(ragged-dot, bias_bcast)
}
)";
  MatchOptimizedHlo(hlo_text,
                    R"(
; CHECK: custom_call_target="__cublas$lt$groupedMatmul",
; CHECK-SAME: "epilogue":"BIAS"
)");
  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{1e-3, 1e-3}));
}

TEST_F(GroupedGemmRewriteTest, GroupedGemmMatrixBiasRaggedInContracting) {
  const char* hlo_text = R"(
HloModule GroupedGemmMatrixBiasRaggedInContracting

ENTRY test {
  p0 = f16[64,16]{1,0} parameter(0)
  p1 = f16[16,8]{1,0} parameter(1)
  p2 = s64[4] constant({4, 5, 3, 4})
  bias = f16[4,64,8]{2,1,0} parameter(2)
  ragged-dot = f16[4,64,8]{2,1,0} ragged-dot(p0, p1, p2),
                lhs_contracting_dims={1}, rhs_contracting_dims={0},
                lhs_ragged_dims={1}
  ROOT out = f16[4,64,8]{2,1,0} add(ragged-dot, bias)
}
)";
  MatchOptimizedHlo(hlo_text,
                    R"(
; CHECK: custom_call_target="__cublas$lt$groupedMatmul",
; CHECK-SAME: "beta":1
; CHECK-SAME: "epilogue":"DEFAULT"
)");
  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{1e-3, 1e-3}));
}

// Bias tests for ragged dimension in contracting dim with batch
TEST_F(GroupedGemmRewriteTest,
       GroupedGemmMatrixBiasRaggedInContractingWithBatch) {
  const char* hlo_text = R"(
HloModule GroupedGemmMatrixBiasRaggedInContractingWithBatch

ENTRY test {
  p0 = f16[3,64,9]{2,1,0} parameter(0)
  p1 = f16[3,9,8]{2,1,0} parameter(1)
  p2 = s64[3,2] constant({{4, 5}, {4, 5}, {4, 5}})
  bias = f16[2,3,64,8]{3,2,1,0} parameter(2)
  ragged-dot = f16[2,3,64,8]{3,2,1,0} ragged-dot(p0, p1, p2),
                lhs_contracting_dims={2}, rhs_contracting_dims={1},
                lhs_ragged_dims={2}, lhs_batch_dims={0}, rhs_batch_dims={0}
  ROOT out = f16[2,3,64,8]{3,2,1,0} add(ragged-dot, bias)
}
)";
  MatchOptimizedHlo(hlo_text,
                    R"(
; CHECK: custom_call_target="__cublas$lt$groupedMatmul",
; CHECK-SAME: "beta":1
; CHECK-SAME: "epilogue":"DEFAULT"
)");
  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{1e-3, 1e-3}));
}

// Test that vector bias is NOT fused for grouped GEMM with batch dimensions
// (ragged in batch dim case)
TEST_F(GroupedGemmRewriteTest, GroupedGemmVectorBiasRaggedInBatch) {
  const char* hlo_text = R"(
HloModule GroupedGemmVectorBiasRaggedInBatch

ENTRY test {
  p0 = f16[16,64,9]{2,1,0} parameter(0)
  p1 = f16[16,9,8]{2,1,0} parameter(1)
  p2 = s64[4] constant({4, 2, 6, 4})
  bias = f16[8]{0} parameter(2)
  ragged-dot = f16[16,64,8]{2,1,0} ragged-dot(p0, p1, p2),
                lhs_contracting_dims={2}, rhs_contracting_dims={1},
                lhs_ragged_dims={0}, lhs_batch_dims={0}, rhs_batch_dims={0}
  bias_bcast = f16[16,64,8]{2,1,0} broadcast(bias), dimensions={2}
  ROOT out = f16[16,64,8]{2,1,0} add(ragged-dot, bias_bcast)
}
)";
  // Verify that bias is NOT fused (epilogue should be DEFAULT, not BIAS)
  MatchOptimizedHlo(hlo_text,
                    R"(
; CHECK: custom_call_target="__cublas$lt$groupedMatmul",
; CHECK-SAME: "epilogue":"DEFAULT"
; CHECK-NOT: "epilogue":"BIAS"
)");
  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{1e-3, 1e-3}));
}

TEST_F(GroupedGemmRewriteTest, GroupedGemmMatrixBiasRaggedInBatch) {
  const char* hlo_text = R"(
HloModule GroupedGemmMatrixBiasRaggedInBatch

ENTRY test {
  p0 = f16[16,64,9]{2,1,0} parameter(0)
  p1 = f16[16,9,8]{2,1,0} parameter(1)
  p2 = s64[4] constant({4, 2, 6, 4})
  bias = f16[16,64,8]{2,1,0} parameter(2)
  ragged-dot = f16[16,64,8]{2,1,0} ragged-dot(p0, p1, p2),
                lhs_contracting_dims={2}, rhs_contracting_dims={1},
                lhs_ragged_dims={0}, lhs_batch_dims={0}, rhs_batch_dims={0}
  ROOT out = f16[16,64,8]{2,1,0} add(ragged-dot, bias)
}
)";
  MatchOptimizedHlo(hlo_text,
                    R"(
; CHECK: custom_call_target="__cublas$lt$groupedMatmul",
; CHECK-SAME: "beta":1
; CHECK-SAME: "epilogue":"DEFAULT"
)");
  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{1e-3, 1e-3}));
}

// Bias tests for ragged dimension in contracting transpose

TEST_F(GroupedGemmRewriteTest,
       GroupedGemmMatrixBiasRaggedInContractingTranspose) {
  const char* hlo_text = R"(
HloModule GroupedGemmMatrixBiasRaggedInContractingTranspose

ENTRY test {
  p0 = f16[1024,64]{1,0} parameter(0)
  p1 = f16[1024,256]{1,0} parameter(1)
  p2 = s32[8] constant({128, 128, 128, 128, 128, 128, 128, 128})
  bias = f16[8,64,256]{2,0,1} parameter(2)
  ragged-dot = f16[8,64,256]{2,0,1} ragged-dot(p0, p1, p2),
                lhs_contracting_dims={0}, rhs_contracting_dims={0},
                lhs_ragged_dims={0}
  ROOT out = f16[8,64,256]{2,0,1} add(ragged-dot, bias)
}
)";
  MatchOptimizedHlo(hlo_text,
                    R"(
; CHECK: custom_call_target="__cublas$lt$groupedMatmul",
; CHECK-SAME: "beta":1
; CHECK-SAME: "epilogue":"DEFAULT"
)");
  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{1e-3, 1e-3}));
}

// Bias tests for ragged dimension in non-contracting transpose

TEST_F(GroupedGemmRewriteTest,
       GroupedGemmMatrixBiasRaggedNonContractingTranspose) {
  if (IsRocm()) {
    const auto* rocm_cc = Capability().rocm_compute_capability();
    if (rocm_cc && rocm_cc->gfx_version() == "gfx950") {
      GTEST_SKIP()
          << "Ragged non-contracting transpose not supported on gfx950";
    }
  }
  const char* hlo_text = R"(
HloModule GroupedGemmMatrixBiasRaggedNonContractingTranspose

ENTRY test {
  p0 = f16[9,64]{1,0} parameter(0)
  p1 = f16[4,9,8]{2,1,0} parameter(1)
  p2 = s64[4] constant({16, 8, 24, 16})
  bias = f16[64,8]{1,0} parameter(2)
  ragged-dot = f16[64,8]{1,0} ragged-dot(p0, p1, p2),
                lhs_contracting_dims={0}, rhs_contracting_dims={1},
                lhs_ragged_dims={1}, rhs_group_dims={0}
  ROOT out = f16[64,8]{1,0} add(ragged-dot, bias)
}
)";
  MatchOptimizedHlo(hlo_text,
                    R"(
; CHECK: custom_call_target="__cublas$lt$groupedMatmul",
; CHECK-SAME: "beta":1
; CHECK-SAME: "epilogue":"DEFAULT"
)");
  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{1e-3, 1e-3}));
}

// Test epilogue fusion for grouped GEMM: ReLU Activation
TEST_F(GroupedGemmRewriteTest, GroupedGemmReluActivation) {
  const char* hlo_text = R"(
HloModule GroupedGemmRelu

ENTRY test {
  p0 = f16[64,9]{1,0} parameter(0)
  p1 = f16[2,9,8]{2,1,0} parameter(1)
  p2 = s32[2] constant({16, 48})
  ragged-dot = f16[64,8]{1,0} ragged-dot(p0, p1, p2),
                lhs_contracting_dims={1}, rhs_contracting_dims={1},
                lhs_ragged_dims={0}, rhs_group_dims={0}
  zero = f16[] constant(0)
  zero_bcast = f16[64,8]{1,0} broadcast(zero), dimensions={}
  ROOT out = f16[64,8]{1,0} maximum(ragged-dot, zero_bcast)
}
)";
  MatchOptimizedHlo(hlo_text,
                    R"(
; CHECK: custom_call_target="__cublas$lt$groupedMatmul",
; CHECK-SAME: "epilogue":"RELU"
)");
  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{1e-3, 1e-3}));
}

// Test epilogue fusion for grouped GEMM: Vector Bias + ReLU
TEST_F(GroupedGemmRewriteTest, GroupedGemmVectorBiasRelu) {
  const char* hlo_text = R"(
HloModule GroupedGemmVectorBiasRelu

ENTRY test {
  p0 = f16[64,9]{1,0} parameter(0)
  p1 = f16[2,9,8]{2,1,0} parameter(1)
  p2 = s32[2] constant({16, 48})
  bias = f16[8]{0} parameter(2)
  ragged-dot = f16[64,8]{1,0} ragged-dot(p0, p1, p2),
                lhs_contracting_dims={1}, rhs_contracting_dims={1},
                lhs_ragged_dims={0}, rhs_group_dims={0}
  bias_bcast = f16[64,8]{1,0} broadcast(bias), dimensions={1}
  add = f16[64,8]{1,0} add(ragged-dot, bias_bcast)
  zero = f16[] constant(0)
  zero_bcast = f16[64,8]{1,0} broadcast(zero), dimensions={}
  ROOT out = f16[64,8]{1,0} maximum(add, zero_bcast)
}
)";
  MatchOptimizedHlo(hlo_text,
                    R"(
; CHECK: custom_call_target="__cublas$lt$groupedMatmul",
; CHECK-SAME: "epilogue":"BIAS_RELU"
)");
  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{1e-3, 1e-3}));
}

// Test epilogue fusion for grouped GEMM: GELU Activation
TEST_F(GroupedGemmRewriteTest, GroupedGemmGeluActivation) {
  const char* hlo_text = R"(
HloModule GroupedGemmGelu

ENTRY test {
  p0 = f16[64,9]{1,0} parameter(0)
  p1 = f16[2,9,8]{2,1,0} parameter(1)
  p2 = s32[2] constant({16, 48})
  dot = f16[64,8]{1,0} ragged-dot(p0, p1, p2),
        lhs_contracting_dims={1}, rhs_contracting_dims={1},
        lhs_ragged_dims={0}, rhs_group_dims={0}
  mul.0 = f16[64,8] multiply(dot, dot)
  mul.1 = f16[64,8] multiply(dot, mul.0)
  const.0 = f16[] constant(0.044715)
  bcast.0 = f16[64,8] broadcast(const.0), dimensions={}
  mul.2 = f16[64,8] multiply(mul.1, bcast.0)
  add.0 = f16[64,8] add(dot, mul.2)
  const.1 = f16[] constant(0.797884583)
  bcast.1 = f16[64,8] broadcast(const.1), dimensions={}
  mul.3 = f16[64,8] multiply(add.0, bcast.1)
  tanh = f16[64,8] tanh(mul.3)
  const.2 = f16[] constant(1)
  bcast.2 = f16[64,8] broadcast(const.2), dimensions={}
  add.2 = f16[64,8] add(tanh, bcast.2)
  const.3 = f16[] constant(0.5)
  bcast.3 = f16[64,8] broadcast(const.3), dimensions={}
  mul.4 = f16[64,8] multiply(add.2, bcast.3)
  ROOT out = f16[64,8] multiply(dot, mul.4)
}
)";
  MatchOptimizedHlo(hlo_text,
                    R"(
; CHECK: custom_call_target="__cublas$lt$groupedMatmul",
; CHECK-SAME: "epilogue":"GELU"
)");
  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{1e-3, 1e-3}));
}

// Test epilogue fusion for grouped GEMM: Vector Bias + GELU
TEST_F(GroupedGemmRewriteTest, GroupedGemmVectorBiasGelu) {
  const char* hlo_text = R"(
HloModule GroupedGemmVectorBiasGelu

ENTRY test {
  p0 = f16[64,9]{1,0} parameter(0)
  p1 = f16[2,9,8]{2,1,0} parameter(1)
  p2 = s32[2] constant({16, 48})
  bias = f16[8]{0} parameter(2)
  dot = f16[64,8]{1,0} ragged-dot(p0, p1, p2),
        lhs_contracting_dims={1}, rhs_contracting_dims={1},
        lhs_ragged_dims={0}, rhs_group_dims={0}
  bias_bcast = f16[64,8] broadcast(bias), dimensions={1}
  add = f16[64,8] add(dot, bias_bcast)
  mul.0 = f16[64,8] multiply(add, add)
  mul.1 = f16[64,8] multiply(add, mul.0)
  const.0 = f16[] constant(0.044715)
  bcast.0 = f16[64,8] broadcast(const.0), dimensions={}
  mul.2 = f16[64,8] multiply(mul.1, bcast.0)
  add.0 = f16[64,8] add(add, mul.2)
  const.1 = f16[] constant(0.797884583)
  bcast.1 = f16[64,8] broadcast(const.1), dimensions={}
  mul.3 = f16[64,8] multiply(add.0, bcast.1)
  tanh = f16[64,8] tanh(mul.3)
  const.2 = f16[] constant(1)
  bcast.2 = f16[64,8] broadcast(const.2), dimensions={}
  add.2 = f16[64,8] add(tanh, bcast.2)
  const.3 = f16[] constant(0.5)
  bcast.3 = f16[64,8] broadcast(const.3), dimensions={}
  mul.4 = f16[64,8] multiply(add.2, bcast.3)
  ROOT out = f16[64,8] multiply(add, mul.4)
}
)";
  MatchOptimizedHlo(hlo_text,
                    R"(
; CHECK: custom_call_target="__cublas$lt$groupedMatmul",
; CHECK-SAME: "epilogue":"BIAS_GELU"
)");
  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{1e-3, 1e-3}));
}

// Test epilogue fusion for grouped GEMM: Matrix Bias + ReLU
TEST_F(GroupedGemmRewriteTest, GroupedGemmMatrixBiasRelu) {
  const char* hlo_text = R"(
HloModule GroupedGemmMatrixBiasRelu

ENTRY test {
  p0 = f16[64,9]{1,0} parameter(0)
  p1 = f16[2,9,8]{2,1,0} parameter(1)
  p2 = s32[2] constant({16, 48})
  bias = f16[64,8]{1,0} parameter(2)
  ragged-dot = f16[64,8]{1,0} ragged-dot(p0, p1, p2),
                lhs_contracting_dims={1}, rhs_contracting_dims={1},
                lhs_ragged_dims={0}, rhs_group_dims={0}
  add = f16[64,8]{1,0} add(ragged-dot, bias)
  zero = f16[] constant(0)
  zero_bcast = f16[64,8]{1,0} broadcast(zero), dimensions={}
  ROOT out = f16[64,8]{1,0} maximum(add, zero_bcast)
}
)";
  MatchOptimizedHlo(hlo_text,
                    R"(
; CHECK: custom_call_target="__cublas$lt$groupedMatmul",
; CHECK-SAME: "beta":1
; CHECK-SAME: "epilogue":"RELU"
)");
  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{1e-3, 1e-3}));
}

// Test epilogue fusion for grouped GEMM: Swish/SILU Activation
TEST_F(GroupedGemmRewriteTest, GroupedGemmSwishActivation) {
  auto runtime_version = GetToolkitVersion();
  bool rocm_swish_available =
      IsRocm() &&
      (runtime_version >= stream_executor::SemanticVersion(7, 0, 0));
  if (!rocm_swish_available) {
    GTEST_SKIP() << "Swish/SILU activation fusion only available on ROCm 7.0+";
  }

  const char* hlo_text = R"(
HloModule GroupedGemmSwish

ENTRY test {
  p0 = f16[64,9]{1,0} parameter(0)
  p1 = f16[2,9,8]{2,1,0} parameter(1)
  p2 = s32[2] constant({16, 48})
  dot = f16[64,8]{1,0} ragged-dot(p0, p1, p2),
        lhs_contracting_dims={1}, rhs_contracting_dims={1},
        lhs_ragged_dims={0}, rhs_group_dims={0}
  neg = f16[64,8] negate(dot)
  exp = f16[64,8] exponential(neg)
  one = f16[] constant(1)
  one_bcast = f16[64,8] broadcast(one), dimensions={}
  denom = f16[64,8] add(one_bcast, exp)
  sigmoid = f16[64,8] divide(one_bcast, denom)
  ROOT out = f16[64,8] multiply(dot, sigmoid)
}
)";
  MatchOptimizedHlo(hlo_text,
                    R"(
; CHECK: custom_call_target="__cublas$lt$groupedMatmul",
; CHECK-SAME: "epilogue":"SILU"
)");
  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{1e-3, 1e-3}));
}

// Test that GELU activation is NOT fused for grouped GEMM when aux output is
// required (i.e., when there are users before the activation)
TEST_F(GroupedGemmRewriteTest, GroupedGemmGeluActivationWithAuxNoFusion) {
  const char* hlo_text = R"(
HloModule GroupedGemmGeluWithAux

ENTRY test {
  p0 = f16[64,9]{1,0} parameter(0)
  p1 = f16[2,9,8]{2,1,0} parameter(1)
  p2 = s32[2] constant({16, 48})
  dot = f16[64,8]{1,0} ragged-dot(p0, p1, p2),
        lhs_contracting_dims={1}, rhs_contracting_dims={1},
        lhs_ragged_dims={0}, rhs_group_dims={0}
  mul.0 = f16[64,8] multiply(dot, dot)
  mul.1 = f16[64,8] multiply(dot, mul.0)
  const.0 = f16[] constant(0.044715)
  bcast.0 = f16[64,8] broadcast(const.0), dimensions={}
  mul.2 = f16[64,8] multiply(mul.1, bcast.0)
  add.0 = f16[64,8] add(dot, mul.2)
  const.1 = f16[] constant(0.797884583)
  bcast.1 = f16[64,8] broadcast(const.1), dimensions={}
  mul.3 = f16[64,8] multiply(add.0, bcast.1)
  tanh = f16[64,8] tanh(mul.3)
  const.2 = f16[] constant(1)
  bcast.2 = f16[64,8] broadcast(const.2), dimensions={}
  add.2 = f16[64,8] add(tanh, bcast.2)
  const.3 = f16[] constant(0.5)
  bcast.3 = f16[64,8] broadcast(const.3), dimensions={}
  mul.4 = f16[64,8] multiply(add.2, bcast.3)
  gelu_out = f16[64,8] multiply(dot, mul.4)
  extra_user = f16[64,8] add(dot, dot)
  ROOT out = (f16[64,8], f16[64,8]) tuple(gelu_out, extra_user)
}
)";
  MatchOptimizedHlo(hlo_text,
                    R"(
; CHECK: custom_call_target="__cublas$lt$groupedMatmul",
; CHECK-SAME: "epilogue":"DEFAULT"
; CHECK-NOT: "epilogue":"GELU_AUX"
)");
  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{1e-3, 1e-3}));
}

// Test that Swish activation is NOT fused for grouped GEMM when aux output is
// required (i.e., when there are users before the activation)
TEST_F(GroupedGemmRewriteTest, GroupedGemmSwishActivationWithAuxNoFusion) {
  auto runtime_version = GetToolkitVersion();
  bool rocm_swish_available =
      IsRocm() &&
      (runtime_version >= stream_executor::SemanticVersion(7, 0, 0));
  if (!rocm_swish_available) {
    GTEST_SKIP() << "Swish/SILU activation fusion only available on ROCm 7.0+";
  }

  const char* hlo_text = R"(
HloModule GroupedGemmSwishWithAux

ENTRY test {
  p0 = f16[64,9]{1,0} parameter(0)
  p1 = f16[2,9,8]{2,1,0} parameter(1)
  p2 = s32[2] constant({16, 48})
  dot = f16[64,8]{1,0} ragged-dot(p0, p1, p2),
        lhs_contracting_dims={1}, rhs_contracting_dims={1},
        lhs_ragged_dims={0}, rhs_group_dims={0}
  neg = f16[64,8] negate(dot)
  exp = f16[64,8] exponential(neg)
  one = f16[] constant(1)
  one_bcast = f16[64,8] broadcast(one), dimensions={}
  denom = f16[64,8] add(one_bcast, exp)
  sigmoid = f16[64,8] divide(one_bcast, denom)
  swish_out = f16[64,8] multiply(dot, sigmoid)
  extra_user = f16[64,8] add(dot, dot)
  ROOT out = (f16[64,8], f16[64,8]) tuple(swish_out, extra_user)
}
)";
  MatchOptimizedHlo(hlo_text,
                    R"(
; CHECK: custom_call_target="__cublas$lt$groupedMatmul",
; CHECK-SAME: "epilogue":"DEFAULT"
; CHECK-NOT: "epilogue":"SILU"
)");
  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{1e-3, 1e-3}));
}

// Test transpose bias with non-transposed ragged-dot output
TEST_F(GroupedGemmRewriteTest, GroupedGemmTransposedBiasNonTransposedOutput) {
  const char* hlo_text = R"(
HloModule GroupedGemmTransposeBias

ENTRY test {
  p0 = f16[64,9]{1,0} parameter(0)
  p1 = f16[2,9,8]{2,1,0} parameter(1)
  p2 = s32[2] constant({16, 48})
  bias = f16[8,64]{1,0} parameter(2)
  ragged-dot = f16[64,8]{1,0} ragged-dot(p0, p1, p2),
                lhs_contracting_dims={1}, rhs_contracting_dims={1},
                lhs_ragged_dims={0}, rhs_group_dims={0}
  bias_transposed = f16[64,8]{1,0} transpose(bias), dimensions={1,0}
  ROOT out = f16[64,8]{1,0} add(ragged-dot, bias_transposed)
}
)";
  MatchOptimizedHlo(hlo_text,
                    R"(
; CHECK: custom_call_target="__cublas$lt$groupedMatmul",
; CHECK-SAME: "beta":1
; CHECK-SAME: "epilogue":"DEFAULT"
)");
  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{1e-3, 1e-3}));
}

// Test non-transposed bias with transposed ragged-dot output
TEST_F(GroupedGemmRewriteTest, GroupedGemmNonTransposedBiasTransposedOutput) {
  const char* hlo_text = R"(
HloModule GroupedGemmNonTransposedBiasTransposedOutput

ENTRY test {
  p0 = f16[64,9]{1,0} parameter(0)
  p1 = f16[2,9,8]{2,1,0} parameter(1)
  p2 = s32[2] constant({16, 48})
  bias = f16[64,8]{1,0} parameter(2)
  ragged-dot = f16[64,8]{0,1} ragged-dot(p0, p1, p2),
                lhs_contracting_dims={1}, rhs_contracting_dims={1},
                lhs_ragged_dims={0}, rhs_group_dims={0}
  ROOT out = f16[64,8]{0,1} add(ragged-dot, bias)
}
)";
  MatchOptimizedHlo(hlo_text,
                    R"(
; CHECK: custom_call_target="__cublas$lt$groupedMatmul",
; CHECK-SAME: "beta":1
; CHECK-SAME: "epilogue":"DEFAULT"
)");
  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{1e-3, 1e-3}));
}

// Test transpose bias with non-transposed ragged-dot output (ragged in
// contracting)
TEST_F(GroupedGemmRewriteTest,
       GroupedGemmTransposeBiasNonTransposedOutputRaggedInContracting) {
  const char* hlo_text = R"(
HloModule GroupedGemmTransposeBiasRaggedInContracting

ENTRY test {
  p0 = f16[64,16]{1,0} parameter(0)
  p1 = f16[16,8]{1,0} parameter(1)
  p2 = s64[4] constant({4, 5, 3, 4})
  bias = f16[4,8,64]{2,1,0} parameter(2)
  ragged-dot = f16[4,64,8]{2,1,0} ragged-dot(p0, p1, p2),
                lhs_contracting_dims={1}, rhs_contracting_dims={0},
                lhs_ragged_dims={1}
  bias_transposed = f16[4,64,8]{2,1,0} transpose(bias), dimensions={0,2,1}
  ROOT out = f16[4,64,8]{2,1,0} add(ragged-dot, bias_transposed)
}
)";
  MatchOptimizedHlo(hlo_text,
                    R"(
; CHECK: custom_call_target="__cublas$lt$groupedMatmul",
; CHECK-SAME: "beta":1
; CHECK-SAME: "epilogue":"DEFAULT"
)");
  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{1e-3, 1e-3}));
}

// Test non-transposed bias with transposed ragged-dot output (ragged in
// contracting)
TEST_F(GroupedGemmRewriteTest,
       GroupedGemmNonTransposedBiasTransposedOutputRaggedInContracting) {
  const char* hlo_text = R"(
HloModule GroupedGemmNonTransposedBiasTransposedOutputRaggedInContracting

ENTRY test {
  p0 = f16[64,16]{1,0} parameter(0)
  p1 = f16[16,8]{1,0} parameter(1)
  p2 = s64[4] constant({4, 5, 3, 4})
  bias = f16[4,64,8]{2,1,0} parameter(2)
  ragged-dot = f16[4,64,8]{2,0,1} ragged-dot(p0, p1, p2),
                lhs_contracting_dims={1}, rhs_contracting_dims={0},
                lhs_ragged_dims={1}
  ROOT out = f16[4,64,8]{2,0,1} add(ragged-dot, bias)
}
)";
  MatchOptimizedHlo(hlo_text,
                    R"(
; CHECK: custom_call_target="__cublas$lt$groupedMatmul",
; CHECK-SAME: "beta":1
; CHECK-SAME: "epilogue":"DEFAULT"
)");
  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{1e-3, 1e-3}));
}

// Tests that when both a vector bias and a matrix bias are present they are
// always fused in a deterministic operand order, regardless of which `add`
// appears first in the HLO tree.
//
// The expected custom-call operand layout for grouped GEMM is:
//   [lhs, rhs, group_sizes, matrix_bias, vector_bias]
//
// Case 1: vector bias is the inner add, matrix bias is the outer add.
// Before the fix this produced the reversed order
// (custom-call(..., vector_bias, matrix_bias)).
TEST_F(GroupedGemmRewriteTest, GroupedGemmVectorAndMatrixBiasVectorBiasFirst) {
  const char* hlo_text = R"(
HloModule GroupedGemmVectorAndMatrixBiasVectorBiasFirst

ENTRY test {
  p0 = f16[64,9]{1,0} parameter(0)
  p1 = f16[2,9,8]{2,1,0} parameter(1)
  p2 = s32[2] constant({16, 48})
  vector_bias = f16[8]{0} parameter(2)
  matrix_bias = f16[64,8]{1,0} parameter(3)

  vector_bias_bcast = f16[64,8]{1,0} broadcast(vector_bias), dimensions={1}
  ragged-dot = f16[64,8]{1,0} ragged-dot(p0, p1, p2),
                lhs_contracting_dims={1}, rhs_contracting_dims={1},
                lhs_ragged_dims={0}, rhs_group_dims={0}
  inner_add = f16[64,8]{1,0} add(ragged-dot, vector_bias_bcast)
  ROOT out = f16[64,8]{1,0} add(inner_add, matrix_bias)
}
)";
  // Verify operand order: [lhs, rhs, group_sizes, matrix_bias, vector_bias].
  // The HLO printer preserves instruction names, so matrix_bias appears before
  // vector_bias in the custom-call operand list.
  MatchOptimizedHlo(hlo_text,
                    R"(
; CHECK: custom-call(%p0, %p1, %p2{{[._0-9]*}}, %matrix_bias, %vector_bias),
; CHECK-SAME: custom_call_target="__cublas$lt$groupedMatmul"
; CHECK-SAME: "beta":1
; CHECK-SAME: "epilogue":"BIAS"
)");
  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{1e-3, 1e-3}));
}

// Case 2: matrix bias is the inner add, vector bias is the outer add.
// Before the fix this produced: custom-call(..., matrix_bias, vector_bias)
// which happens to be the correct order, but only by coincidence.
// With the fix, both cases produce identical operand ordering.
TEST_F(GroupedGemmRewriteTest, GroupedGemmVectorAndMatrixBiasMatrixBiasFirst) {
  const char* hlo_text = R"(
HloModule GroupedGemmVectorAndMatrixBiasMatrixBiasFirst

ENTRY test {
  p0 = f16[64,9]{1,0} parameter(0)
  p1 = f16[2,9,8]{2,1,0} parameter(1)
  p2 = s32[2] constant({16, 48})
  vector_bias = f16[8]{0} parameter(2)
  matrix_bias = f16[64,8]{1,0} parameter(3)

  ragged-dot = f16[64,8]{1,0} ragged-dot(p0, p1, p2),
                lhs_contracting_dims={1}, rhs_contracting_dims={1},
                lhs_ragged_dims={0}, rhs_group_dims={0}
  inner_add = f16[64,8]{1,0} add(ragged-dot, matrix_bias)
  vector_bias_bcast = f16[64,8]{1,0} broadcast(vector_bias), dimensions={1}
  ROOT out = f16[64,8]{1,0} add(inner_add, vector_bias_bcast)
}
)";
  // Verify operand order: [lhs, rhs, group_sizes, matrix_bias, vector_bias].
  // Identical CHECK to GroupedGemmVectorAndMatrixBiasVectorBiasFirst above,
  // confirming both HLO orderings produce the same stable custom-call layout.
  MatchOptimizedHlo(hlo_text,
                    R"(
; CHECK: custom-call(%p0, %p1, %p2{{[._0-9]*}}, %matrix_bias, %vector_bias),
; CHECK-SAME: custom_call_target="__cublas$lt$groupedMatmul"
; CHECK-SAME: "beta":1
; CHECK-SAME: "epilogue":"BIAS"
)");
  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{1e-3, 1e-3}));
}

}  // namespace
}  // namespace gpu
}  // namespace xla

/* Copyright 2025 The OpenXLA Authors.

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

#include "xla/service/gpu/transforms/gemm_workspace_rewriter.h"

#include <memory>

#include <gtest/gtest.h>
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/gpu/tests/gpu_codegen_test.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {
namespace gpu {
namespace {

namespace se = ::stream_executor;

class GemmWorkspaceRewriterTest : public GpuCodegenTest {};

// Tests that cuBLASLt calls with a selected algorithm and large workspace
// are rewritten to use a smaller workspace.
TEST_F(GemmWorkspaceRewriterTest,
       CublasLtCallWithSelectedAlgorithmIsRewritten) {
  // This HLO simulates a cuBLASLt matmul after autotuning - it has
  // selected_algorithm set and a conservatively large workspace.
  const char* hlo_text = R"(
HloModule TestModule

ENTRY main {
  lhs = f32[32,64] parameter(0)
  rhs = f32[64,128] parameter(1)
  custom_call = (f32[32,128], s8[4194304]) custom-call(lhs, rhs),
    custom_call_target="__cublas$lt$matmul",
    backend_config={"operation_queue_id":"0","wait_on_operation_queues":[],"gemm_backend_config":{"alpha_real":1,"alpha_imag":0,"beta":0,"dot_dimension_numbers":{"lhs_contracting_dimensions":["1"],"rhs_contracting_dimensions":["0"],"lhs_batch_dimensions":[],"rhs_batch_dimensions":[]},"precision_config":{"operand_precision":["DEFAULT","DEFAULT"]},"epilogue":"DEFAULT","selected_algorithm":"0"}}
  ROOT result = f32[32,128] get-tuple-element(custom_call), index=0
}
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_text));

  se::StreamExecutor* stream_exec = backend().default_stream_executor();
  GemmWorkspaceRewriter pass(
      stream_exec->GetDeviceDescription().gpu_compute_capability(),
      stream_exec);
  TF_ASSERT_OK_AND_ASSIGN(bool changed, pass.Run(module.get()));

  // The pass should reduce the workspace size from 4MB to the algorithm's
  // actual requirement (typically much smaller).
  EXPECT_TRUE(changed);
}

}  // namespace
}  // namespace gpu
}  // namespace xla

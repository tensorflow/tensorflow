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

#include <gtest/gtest.h>
#include "absl/strings/substitute.h"
#include "xla/backends/gpu/tests/hlo_pjrt_gpu_test_base.h"
#include "xla/service/gpu/matmul_utils.h"
#include "xla/stream_executor/sycl/sycl_platform_id.h"

namespace xla {
namespace gpu {
namespace {

using IntelGpuCompilerTest = HloPjRtGpuTestBase;

TEST_F(IntelGpuCompilerTest, CheckCompiler) {
  EXPECT_EQ(stream_executor_platform_id(),
            stream_executor::sycl::kSyclPlatformId);
}

// Checks that SyclGemmWorkspacePass rewrites the cuBLASLt matmul workspace
// from the 4 MiB default to the oneDNN scratchpad size (0 for a tiny matmul).
TEST_F(IntelGpuCompilerTest, ResizesMatmulWorkspace) {
  const char* hlo_text = R"(
    HloModule matmul

    ENTRY main {
      a = bf16[8,8] parameter(0)
      b = bf16[8,8] parameter(1)
      ROOT dot = bf16[8,8] dot(a, b),
        lhs_contracting_dims={1}, rhs_contracting_dims={0}
    }
  )";

  const char* check_pattern = R"(
    CHECK: (bf16[8,8]{{.*}}, s8[0]{0}) custom-call
    CHECK-SAME: custom_call_target="__cublas$lt$matmul"
  )";
  MatchOptimizedHlo(hlo_text, check_pattern);
}

// Complex-typed matmuls are skipped because oneDNN's matmul primitive does
// not support complex element types; the 4 MiB default workspace from
// GemmRewriter is retained.
TEST_F(IntelGpuCompilerTest, SkipsWorkspaceResizeForComplexMatmul) {
  const char* hlo_text = R"(
    HloModule complex_matmul

    ENTRY main {
      a = c64[8,8] parameter(0)
      b = c64[8,8] parameter(1)
      ROOT dot = c64[8,8] dot(a, b),
        lhs_contracting_dims={1}, rhs_contracting_dims={0}
    }
  )";

  const char* check_pattern = R"(
    CHECK: (c64[8,8]{{.*}}, s8[$0]{0}) custom-call
    CHECK-SAME: custom_call_target="__cublas$$lt$$matmul"
  )";
  MatchOptimizedHlo(
      hlo_text, absl::Substitute(check_pattern, GemmConfig::kDefaultWorkspace));
}

}  // namespace
}  // namespace gpu
}  // namespace xla

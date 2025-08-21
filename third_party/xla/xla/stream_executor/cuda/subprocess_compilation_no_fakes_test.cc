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

#include <cstdint>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status_matchers.h"
#include "absl/strings/string_view.h"
#include "xla/stream_executor/cuda/cubin_or_ptx_image.h"
#include "xla/stream_executor/cuda/cuda_compute_capability.h"
#include "xla/stream_executor/cuda/subprocess_compilation.h"
#include "xla/stream_executor/gpu/gpu_asm_opts.h"
#include "xla/tsl/platform/statusor.h"
#include "tsl/platform/path.h"

namespace stream_executor {
namespace {

TEST(SubprocessCompilationTest, BundleGpuAsmUsingFatbinWorks) {
  std::string cuda_dir;
  if (!tsl::io::GetTestWorkspaceDir(&cuda_dir)) {
    GTEST_SKIP() << "No test workspace directory found which means we can't "
                    "run this test. Was this called in a Bazel environment?";
  }

  const absl::string_view ptx = R"(
// A minimal PTX kernel to add two constants: 42 + 100
.version 8.0
.target sm_90
.address_size 64

.visible .entry add_constants()
{
    // Declare three 32-bit registers
    .reg .s32 %r_a, %r_b, %r_result;

    // Move the constant values into registers
    mov.s32 %r_a, 42;
    mov.s32 %r_b, 100;

    // Add the two registers and store in the result register
    add.s32 %r_result, %r_a, %r_b; // %r_result will now hold 142

    // End of kernel
    ret;
}
)";

  GpuAsmOpts opts;
  tensorflow::se::CudaComputeCapability cc{9, 0};
  std::vector<CubinOrPTXImage> images;

  std::vector<uint8_t> bytes(ptx.begin(), ptx.end());
  images.push_back({/*is_ptx=*/true, cc, bytes});

  TF_ASSERT_OK_AND_ASSIGN(auto assembly,
                          CompileGpuAsmUsingPtxAs(cc, ptx, opts, false, false));
  images.push_back({/*is_ptx=*/false, cc, assembly.cubin});

  EXPECT_THAT(BundleGpuAsmUsingFatbin(images, opts), absl_testing::IsOk());
}

}  // namespace
}  // namespace stream_executor

/* Copyright 2026 The OpenXLA Authors.

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
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/base/no_destructor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/stream_executor/cuda/cuda_compute_capability.h"
#include "xla/stream_executor/cuda/ptx_compiler_helpers.h"
#include "xla/stream_executor/gpu/gpu_asm_opts.h"

extern "C" absl::Span<const absl::string_view>
XlaGpuPtxCompilerExtraFlagsToPrepend() {
  static const absl::NoDestructor<std::vector<absl::string_view>> kExtraFlags(
      std::vector<absl::string_view>{"--random_prepend_1",
                                     "--random_prepend_2"});
  return *kExtraFlags;
}

namespace stream_executor {
namespace {

using ::testing::ElementsAre;

TEST(PtxCompilerExtraFlagsTest, AppendPtxCompilerFlagsPrependsExtraFlags) {
  GpuAsmOpts options;
  options.disable_gpuasm_optimizations = true;
  options.extra_flags = {"--user_extra_1"};
  std::vector<std::string> flags;
  AppendPtxCompilerFlags(options, flags);
  EXPECT_THAT(flags, ElementsAre("--random_prepend_1", "--random_prepend_2",
                                 "-O0", "--user_extra_1"));
}

TEST(PtxCompilerExtraFlagsTest,
     AppendArchitectureSpecificPtxCompilerFlagsPrependsExtraFlags) {
  CudaComputeCapability cc = CudaComputeCapability::Ampere();
  GpuAsmOpts options;
  options.disable_gpuasm_optimizations = true;
  options.extra_flags = {"--user_extra_1"};
  std::vector<std::string> flags;
  AppendArchitectureSpecificPtxCompilerFlags(
      cc, options, /*dump_compilation_log=*/true, flags);
  EXPECT_THAT(flags, ElementsAre("-arch=sm_80", "--warn-on-spills", "-v",
                                 "--random_prepend_1", "--random_prepend_2",
                                 "-O0", "--user_extra_1"));
}

TEST(PtxCompilerExtraFlagsTest,
     AppendArchitectureSpecificPtxCompilerFlagsBlackwellPrependsExtraFlags) {
  CudaComputeCapability cc = CudaComputeCapability::B200Accelerated();
  GpuAsmOpts options;
  options.disable_gpuasm_optimizations = true;
  options.extra_flags = {"--user_extra_1"};
  std::vector<std::string> flags;
  AppendArchitectureSpecificPtxCompilerFlags(
      cc, options, /*dump_compilation_log=*/true, flags);
  EXPECT_THAT(flags, ElementsAre("-arch=sm_100a", "--warn-on-spills", "-v",
                                 "--random_prepend_1", "--random_prepend_2",
                                 "-O0", "--user_extra_1"));
}

}  // namespace
}  // namespace stream_executor

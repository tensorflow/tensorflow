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

#include "xla/stream_executor/cuda/subprocess_compilation_provider.h"

#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "xla/stream_executor/cuda/compilation_options.h"
#include "xla/stream_executor/cuda/cuda_compute_capability.h"

namespace stream_executor::cuda {
namespace {

using ::absl_testing::StatusIs;
using ::testing::HasSubstr;

TEST(SubprocessCompilationProviderTest, SupportsCompileAndLinkWithNvlink) {
  SubprocessCompilationProvider provider("/path/to/ptxas", "/path/to/nvlink");
  EXPECT_TRUE(provider.SupportsCompileAndLink());
}

TEST(SubprocessCompilationProviderTest,
     DoesNotSupportCompileAndLinkWithoutNvlink) {
  SubprocessCompilationProvider provider("/path/to/ptxas",
                                         /*path_to_nvlink=*/"");
  EXPECT_FALSE(provider.SupportsCompileAndLink());
}

TEST(SubprocessCompilationProviderTest,
     SupportsCompileToRelocatableModuleWithoutNvlink) {
  SubprocessCompilationProvider provider("/path/to/ptxas",
                                         /*path_to_nvlink=*/"");
  EXPECT_TRUE(provider.SupportsCompileToRelocatableModule());
}

TEST(SubprocessCompilationProviderTest,
     CompileAndLinkFailsCleanlyWithoutNvlink) {
  SubprocessCompilationProvider provider("/path/to/ptxas",
                                         /*path_to_nvlink=*/"");
  EXPECT_THAT(provider.CompileAndLink(CudaComputeCapability{9, 0}, {},
                                      CompilationOptions{}),
              StatusIs(absl::StatusCode::kFailedPrecondition,
                       HasSubstr("no nvlink binary was found")));
}

}  // namespace
}  // namespace stream_executor::cuda

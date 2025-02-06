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

#include "xla/stream_executor/cuda/assemble_compilation_provider.h"

#include <memory>
#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "xla/stream_executor/cuda/compilation_provider.h"
#include "xla/stream_executor/cuda/compilation_provider_options.h"
#include "xla/stream_executor/cuda/nvjitlink_support.h"
#include "xla/stream_executor/cuda/ptx_compiler_support.h"
#include "xla/tsl/platform/status_matchers.h"
#include "xla/tsl/platform/statusor.h"
#include "tsl/platform/cuda_root_path.h"
#include "tsl/platform/path.h"

namespace stream_executor::cuda {

namespace {
using ::testing::AllOf;
using ::testing::HasSubstr;
using ::tsl::testing::StatusIs;

TEST(AssembleCompilationProviderTest,
     ReturnsErrorIfNoCompilationProviderIsAvailable) {
  if (!tsl::CandidateCudaRoots().empty()) {
    GTEST_SKIP() << "With the current API design We can't control whether "
                    "`FindCudaExecutable` will find some ptxas installed on "
                    "the testrunner machine. Therefore we skip this test.";
  }

  CompilationProviderOptions options{
      CompilationProviderOptions::NvJitLinkMode::kDisabled,
      /*enable_libnvptxcompiler=*/false,
      /*enable_llvm_module_compilation_parallelism=*/false,
      /*enable_driver_compilation=*/false,
      /*cuda_data_dir=*/"/does/not/exist",
  };
  EXPECT_THAT(AssembleCompilationProvider(options),
              StatusIs(absl::StatusCode::kUnavailable));
}

TEST(AssembleCompilationProviderTest,
     OffersDriverCompilationIfAllowedAndNothingElseIsAvailable) {
  if (!tsl::CandidateCudaRoots().empty()) {
    GTEST_SKIP() << "With the current API design We can't control whether "
                    "`FindCudaExecutable` will find some ptxas installed on "
                    "the testrunner machine. Therefore we skip this test.";
  }

  CompilationProviderOptions options{
      CompilationProviderOptions::NvJitLinkMode::kDisabled,
      /*enable_libnvptxcompiler=*/false,
      /*enable_llvm_module_compilation_parallelism=*/false,
      /*enable_driver_compilation=*/true,
      /*cuda_data_dir=*/"/does/not/exist",
  };

  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<CompilationProvider> compilation_provider,
      AssembleCompilationProvider(options));

  EXPECT_THAT(compilation_provider->name(),
              HasSubstr("DriverCompilationProvider"));
}

TEST(AssembleCompilationProviderTest,
     OffersSubprocessCompilationIfLibraryCompilationIsDisabled) {
  std::string cuda_dir;
  if (!tsl::io::GetTestWorkspaceDir(&cuda_dir)) {
    GTEST_SKIP() << "No test workspace directory found which means we can't "
                    "run this test. Was this called in a Bazel environment?";
  }

  CompilationProviderOptions options{
      CompilationProviderOptions::NvJitLinkMode::kDisabled,
      /*enable_libnvptxcompiler=*/false,
      /*enable_llvm_module_compilation_parallelism=*/false,
      /*enable_driver_compilation=*/false,
      /*cuda_data_dir=*/cuda_dir,
  };

  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<CompilationProvider> compilation_provider,
      AssembleCompilationProvider(options));

  EXPECT_THAT(compilation_provider->name(),
              HasSubstr("SubprocessCompilationProvider"));
}

TEST(
    AssembleCompilationProviderTest,
    OffersLibNvJitLinkWithParallelCompilationShimIfLibNvPtxCompilerIsDisabled) {
  if (!IsLibNvJitLinkSupported()) {
    GTEST_SKIP() << "LibNvJitLink is not supported in this build.";
  }

  CompilationProviderOptions options{
      CompilationProviderOptions::NvJitLinkMode::kEnabled,
      /*enable_libnvptxcompiler=*/false,
      /*enable_llvm_module_compilation_parallelism=*/false,
      /*enable_driver_compilation=*/false,
      /*cuda_data_dir=*/"/does/not/exist",
  };

  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<CompilationProvider> compilation_provider,
      AssembleCompilationProvider(options));

  EXPECT_THAT(compilation_provider->name(),
              AllOf(HasSubstr("DeferRelocatableCompilation"),
                    HasSubstr("NvJitLinkCompilationProvider")));
}

TEST(AssembleCompilationProviderTest,
     OffersLibNvJitLinkAndLibNvPtxCompilerIfBothAreEnabled) {
  if (!IsLibNvJitLinkSupported()) {
    GTEST_SKIP() << "LibNvJitLink is not supported in this build.";
  }
  if (!IsLibNvPtxCompilerSupported()) {
    GTEST_SKIP() << "LibNvPtxCompiler is not supported in this build.";
  }

  CompilationProviderOptions options{
      CompilationProviderOptions::NvJitLinkMode::kEnabled,
      /*enable_libnvptxcompiler=*/true,
      /*enable_llvm_module_compilation_parallelism=*/false,
      /*enable_driver_compilation=*/false,
      /*cuda_data_dir=*/"/does/not/exist",
  };

  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<CompilationProvider> compilation_provider,
      AssembleCompilationProvider(options));

  EXPECT_THAT(compilation_provider->name(),
              AllOf(HasSubstr("CompositeCompilationProvider"),
                    HasSubstr("NvJitLinkCompilationProvider"),
                    HasSubstr("NvptxcompilerCompilationProvider")));
}

}  // namespace
}  // namespace stream_executor::cuda

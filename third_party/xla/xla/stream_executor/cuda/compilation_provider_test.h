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

#ifndef XLA_STREAM_EXECUTOR_CUDA_COMPILATION_PROVIDER_TEST_H_
#define XLA_STREAM_EXECUTOR_CUDA_COMPILATION_PROVIDER_TEST_H_

#include <memory>
#include <string>

#include <gtest/gtest.h>
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/stream_executor/cuda/compilation_provider.h"

namespace stream_executor::cuda {

inline constexpr absl::string_view kSubprocessCompilationProviderName =
    "subprocess";
inline constexpr absl::string_view kNvJitLinkCompilationProviderName =
    "nvjitlink";
inline constexpr absl::string_view kNvptxcompilerCompilationProviderName =
    "nvptxcompiler";
inline constexpr absl::string_view kDriverCompilationProviderName = "driver";

class CompilationProviderTest
    : public testing::TestWithParam<absl::string_view> {
  absl::StatusOr<std::unique_ptr<CompilationProvider>>
  CreateCompilationProvider(absl::string_view name);

  void SetUp() override;
  std::unique_ptr<CompilationProvider> compilation_provider_;

 protected:
  CompilationProvider* compilation_provider() {
    return compilation_provider_.get();
  }
};

// Prints the test parameter name as is. Needed for gtest instantiation.
struct CompilationProviderTestParamNamePrinter {
  std::string operator()(
      const ::testing::TestParamInfo<absl::string_view>& name) const {
    return std::string(name.param);
  }
};

}  // namespace stream_executor::cuda

#endif  // XLA_STREAM_EXECUTOR_CUDA_COMPILATION_PROVIDER_TEST_H_

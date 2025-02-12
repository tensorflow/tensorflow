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

#ifndef XLA_STREAM_EXECUTOR_CUDA_MOCK_COMPILATION_PROVIDER_H_
#define XLA_STREAM_EXECUTOR_CUDA_MOCK_COMPILATION_PROVIDER_H_

#include <string>

#include <gmock/gmock.h>
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/stream_executor/cuda/compilation_options.h"
#include "xla/stream_executor/cuda/compilation_provider.h"
#include "xla/stream_executor/device_description.h"

namespace stream_executor::cuda {

class MockCompilationProvider : public CompilationProvider {
 public:
  MOCK_METHOD(bool, SupportsCompileToRelocatableModule, (), (const, override));
  MOCK_METHOD(bool, SupportsCompileAndLink, (), (const, override));
  MOCK_METHOD(std::string, name, (), (const, override));
  MOCK_METHOD(absl::StatusOr<Assembly>, Compile,
              (const CudaComputeCapability& cc, absl::string_view ptx,
               const CompilationOptions& options),
              (const, override));
  MOCK_METHOD(absl::StatusOr<RelocatableModule>, CompileToRelocatableModule,
              (const CudaComputeCapability& cc, absl::string_view ptx,
               const CompilationOptions& options),
              (const, override));
  MOCK_METHOD(absl::StatusOr<Assembly>, CompileAndLink,
              (const CudaComputeCapability& cc,
               absl::Span<const RelocatableModuleOrPtx> inputs,
               const CompilationOptions& options),
              (const, override));
};

}  // namespace stream_executor::cuda

#endif  // XLA_STREAM_EXECUTOR_CUDA_MOCK_COMPILATION_PROVIDER_H_

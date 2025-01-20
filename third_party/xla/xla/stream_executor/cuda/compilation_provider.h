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

#ifndef XLA_STREAM_EXECUTOR_CUDA_COMPILATION_PROVIDER_H_
#define XLA_STREAM_EXECUTOR_CUDA_COMPILATION_PROVIDER_H_

#include <cstdint>
#include <string>
#include <variant>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/stream_executor/cuda/compilation_options.h"
#include "xla/stream_executor/device_description.h"

namespace stream_executor::cuda {

// A compiled PTX module in CUBIN format. The module still needs to be linked
// before it can be loaded.
struct RelocatableModule {
  std::vector<uint8_t> cubin;

  friend bool operator==(const RelocatableModule& lhs,
                         const RelocatableModule& rhs) {
    return lhs.cubin == rhs.cubin;
  }

  friend bool operator!=(const RelocatableModule& lhs,
                         const RelocatableModule& rhs) {
    return lhs.cubin != rhs.cubin;
  }
};

// A compiled and linked CUDA program in CUBIN format.
struct Assembly {
  std::vector<uint8_t> cubin;

  friend bool operator==(const Assembly& lhs, const Assembly& rhs) {
    return lhs.cubin == rhs.cubin;
  }

  friend bool operator!=(const Assembly& lhs, const Assembly& rhs) {
    return lhs.cubin != rhs.cubin;
  }
};

// A PTX module in textual assembly format.
struct Ptx {
  std::string ptx;

  friend bool operator==(const Ptx& lhs, const Ptx& rhs) {
    return lhs.ptx == rhs.ptx;
  }

  friend bool operator!=(const Ptx& lhs, const Ptx& rhs) {
    return lhs.ptx != rhs.ptx;
  }
};

// Provides PTX compilation and linking facilities
//
// `Compile` is supported by all compilation providers.
//
// `CompileToRelocatableModule` is not supported by all compilation providers.
// `SupportsCompileToRelocatableModule` can be used to check if this method is
// supported.
//
// `CompileAndLink` is not supported by all compilation providers.
// `SupportsCompileAndLink` can be used to check if this method is supported.
//
// Calling `CompileToRelocatableModule` in parallel from multiple threads and
// then linking all modules in a single CompileAndLink call allows for parallel
// compilation.
//
// The CompilationProvider is thread-compatible and since all methods are
// const, it's safe to call them from multiple threads at the same time.
class CompilationProvider {
 public:
  virtual ~CompilationProvider() = default;

  // Compiles a single PTX module into a CUDA program. This method is supported
  // by all compilation providers.
  virtual absl::StatusOr<Assembly> Compile(
      const CudaComputeCapability& cc, absl::string_view ptx,
      const CompilationOptions& options) const = 0;

  // Compiles the given PTX string into relocatable CUBIN for the given
  // architecture `cc`. This method is not supported by all compilation
  // providers. `SupportsCompileToRelocatableModule` can be used to check if
  // this method is supported.
  virtual absl::StatusOr<RelocatableModule> CompileToRelocatableModule(
      const CudaComputeCapability& cc, absl::string_view ptx,
      const CompilationOptions& options) const = 0;

  // Returns true if 'CompileToRelocatableModule' can be used.
  // Not all compilation providers can produce a relocatable CUBIN. For these
  // providers, this function will return false. Any calls to `Compile` will
  // result in an error. `ComileAndLink` can be used instead, but it doesn't
  // allow for separate (parallel) compilation of multiple modules.
  virtual bool SupportsCompileToRelocatableModule() const = 0;

  // Returns true if 'CompileAndLink' can be used.
  // Not all compilation providers can compile and link multiple modules.
  virtual bool SupportsCompileAndLink() const = 0;

  using RelocatableModuleOrPtx = std::variant<RelocatableModule, Ptx>;

  // Links relocatable CUBINs and PTX strings into a single binary. The PTX are
  // getting compiled using the same compilation provider.
  virtual absl::StatusOr<Assembly> CompileAndLink(
      const CudaComputeCapability& cc,
      absl::Span<const RelocatableModuleOrPtx> inputs,
      const CompilationOptions& options) const = 0;

  // Returns the name of the compilation provider.
  virtual std::string name() const = 0;
};

}  // namespace stream_executor::cuda

#endif  // XLA_STREAM_EXECUTOR_CUDA_COMPILATION_PROVIDER_H_

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

#ifndef XLA_STREAM_EXECUTOR_CUDA_CACHING_COMPILATION_PROVIDER_H_
#define XLA_STREAM_EXECUTOR_CUDA_CACHING_COMPILATION_PROVIDER_H_

#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <variant>

#include "absl/base/thread_annotations.h"
#include "absl/container/node_hash_map.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "xla/stream_executor/cuda/compilation_options.h"
#include "xla/stream_executor/cuda/compilation_provider.h"
#include "xla/stream_executor/device_description.h"

namespace stream_executor::cuda {

// Delegates all CompilationProvider calls to a delegate and caches the results
// to avoid recompilation.
//
// Note that linking step is not cached and compilations happening as part of
// `CompileAndLink` are only cached if the delegate supports
// `CompileToRelocatableModule`.
class CachingCompilationProvider : public CompilationProvider {
 public:
  explicit CachingCompilationProvider(
      std::unique_ptr<CompilationProvider> delegate)
      : delegate_(std::move(delegate)) {}

  std::string name() const override;

  bool SupportsCompileToRelocatableModule() const override;
  bool SupportsCompileAndLink() const override;

  absl::StatusOr<Assembly> Compile(
      const CudaComputeCapability& cc, absl::string_view ptx,
      const CompilationOptions& options) const override;
  absl::StatusOr<RelocatableModule> CompileToRelocatableModule(
      const CudaComputeCapability& cc, absl::string_view ptx,
      const CompilationOptions& options) const override;
  absl::StatusOr<Assembly> CompileAndLink(
      const CudaComputeCapability& cc,
      absl::Span<const RelocatableModuleOrPtx> inputs,
      const CompilationOptions& options) const override;

 private:
  std::unique_ptr<CompilationProvider> delegate_;

  using CacheKey =
      std::tuple<CudaComputeCapability, std::string, CompilationOptions>;
  // Indicates that the compilation is currently in progress on a different
  // thread.
  struct Pending {};

  // We use node_hash_maps to ensure pointer stability of values which is
  // required for the interlock mechanism to work.
  using RelocatableModuleCache = absl::node_hash_map<
      CacheKey, std::variant<Pending, absl::StatusOr<RelocatableModule>>>;
  using AssemblyCache =
      absl::node_hash_map<CacheKey,
                          std::variant<Pending, absl::StatusOr<Assembly>>>;
  mutable absl::Mutex relocatable_module_cache_mutex_;
  mutable RelocatableModuleCache relocatable_module_cache_
      ABSL_GUARDED_BY(relocatable_module_cache_mutex_);

  mutable absl::Mutex assembly_cache_mutex_;
  mutable AssemblyCache assembly_cache_ ABSL_GUARDED_BY(assembly_cache_mutex_);
};

}  // namespace stream_executor::cuda

#endif  // XLA_STREAM_EXECUTOR_CUDA_CACHING_COMPILATION_PROVIDER_H_

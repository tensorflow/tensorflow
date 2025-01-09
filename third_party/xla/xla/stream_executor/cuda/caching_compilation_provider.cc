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

#include "xla/stream_executor/cuda/caching_compilation_provider.h"

#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "xla/stream_executor/cuda/compilation_options.h"
#include "xla/stream_executor/cuda/compilation_provider.h"
#include "xla/stream_executor/device_description.h"
#include "tsl/platform/statusor.h"

namespace stream_executor::cuda {

std::string CachingCompilationProvider::name() const {
  return absl::StrCat("CachingCompilationProvider(", delegate_->name(), ")");
}

bool CachingCompilationProvider::SupportsCompileToRelocatableModule() const {
  return delegate_->SupportsCompileToRelocatableModule();
}

bool CachingCompilationProvider::SupportsCompileAndLink() const {
  return delegate_->SupportsCompileAndLink();
}

absl::StatusOr<Assembly> CachingCompilationProvider::Compile(
    const CudaComputeCapability& cc, absl::string_view ptx,
    const CompilationOptions& options) const {
  CacheKey cache_key{cc, std::string{ptx}, options};
  {
    absl::MutexLock lock(&assembly_cache_mutex_);
    auto it = assembly_cache_.find(cache_key);
    if (it != assembly_cache_.end()) {
      // The iterator will get invalid during the `Await` call if the cache is
      // rehashed. That's why we store the address of the value which is stable
      // across rehashes.
      auto cache_value_ptr = &it->second;
      if (std::holds_alternative<Pending>(*cache_value_ptr)) {
        assembly_cache_mutex_.Await(absl::Condition(
            +[](const std::variant<Pending, absl::StatusOr<Assembly>>* value) {
              return !std::holds_alternative<Pending>(*value);
            },
            cache_value_ptr));
      }
      return std::get<absl::StatusOr<Assembly>>(*cache_value_ptr);
    }
    assembly_cache_.emplace(cache_key, Pending{});
  }

  absl::StatusOr<Assembly> assembly = delegate_->Compile(cc, ptx, options);
  {
    absl::MutexLock lock(&assembly_cache_mutex_);
    assembly_cache_[cache_key] = assembly;
  }
  return assembly;
}

absl::StatusOr<RelocatableModule>
CachingCompilationProvider::CompileToRelocatableModule(
    const CudaComputeCapability& cc, absl::string_view ptx,
    const CompilationOptions& options) const {
  CacheKey cache_key{cc, std::string{ptx}, options};
  {
    absl::MutexLock lock(&relocatable_module_cache_mutex_);
    auto it = relocatable_module_cache_.find(cache_key);
    if (it != relocatable_module_cache_.end()) {
      // The iterator will get invalid during the `Await` call if the cache is
      // rehashed. That's why we store the address of the value which is stable
      // across rehashes.
      auto cache_value_ptr = &it->second;
      if (std::holds_alternative<Pending>(*cache_value_ptr)) {
        relocatable_module_cache_mutex_.Await(absl::Condition(
            +[](const std::variant<Pending, absl::StatusOr<RelocatableModule>>*
                    value) { return !std::holds_alternative<Pending>(*value); },
            cache_value_ptr));
      }
      return std::get<absl::StatusOr<RelocatableModule>>(*cache_value_ptr);
    }
    relocatable_module_cache_.emplace(cache_key, Pending{});
  }

  absl::StatusOr<RelocatableModule> relocatable_module =
      delegate_->CompileToRelocatableModule(cc, ptx, options);
  {
    absl::MutexLock lock(&relocatable_module_cache_mutex_);
    relocatable_module_cache_[cache_key] = relocatable_module;
  }
  return relocatable_module;
}

absl::StatusOr<Assembly> CachingCompilationProvider::CompileAndLink(
    const CudaComputeCapability& cc,
    absl::Span<const RelocatableModuleOrPtx> inputs,
    const CompilationOptions& options) const {
  if (!SupportsCompileToRelocatableModule()) {
    return delegate_->CompileAndLink(cc, inputs, options);
  }

  // If the delegate supports CompileToRelocatableModule, then we will compile
  // all PTX modules first to take advantage of the cache.
  std::vector<RelocatableModuleOrPtx> modules;
  modules.reserve(inputs.size());

  for (const auto& input : inputs) {
    if (std::holds_alternative<RelocatableModule>(input)) {
      modules.push_back(std::get<RelocatableModule>(input));
    } else {
      TF_ASSIGN_OR_RETURN(
          RelocatableModule relocatable_module,
          CompileToRelocatableModule(cc, std::get<Ptx>(input).ptx, options));
      modules.push_back(std::move(relocatable_module));
    }
  }

  return delegate_->CompileAndLink(cc, modules, options);
}

}  // namespace stream_executor::cuda

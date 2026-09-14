/* Copyright 2017 The OpenXLA Authors.

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

#include "xla/service/compiler.h"

#include <functional>
#include <memory>
#include <utility>
#include <vector>

#include "absl/base/const_init.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "google/protobuf/message.h"
#include "xla/debug_options_flags.h"
#include "xla/service/metrics_hook_interface.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/util.h"

namespace xla {

/* static */ absl::Mutex Compiler::platform_compiler_mutex_(absl::kConstInit);

std::vector<std::unique_ptr<tsl::protobuf::Message>>
Compiler::ComputeBackendConfigs(const HloInstruction& hlo,
                                se::StreamExecutor* executor) const {
  CHECK(executor != nullptr);
  return {};
}

std::unique_ptr<tsl::protobuf::Message> Compiler::ComputeDefaultBackendConfig(
    const HloInstruction& hlo, se::StreamExecutor* executor) const {
  CHECK(executor != nullptr);
  return nullptr;
}

// Define a default version where metadata is not used.
absl::StatusOr<std::vector<std::unique_ptr<CompiledModule>>>
Compiler::CompileAheadOfTime(
    std::unique_ptr<HloModule> hlo_module, const AotCompilationOptions& options,
    std::unique_ptr<AotCompilationMetadata>* metadata) {
  if (metadata != nullptr) {
    return Unimplemented(
        "Populating AotCompilationMetadata is not implemented on this "
        "compiler.");
  }
  return CompileAheadOfTime(std::move(hlo_module), options);
}

/* static */ absl::flat_hash_map<se::Platform::Id, Compiler::CompilerFactory>*
Compiler::GetPlatformCompilerFactories() {
  static auto* const r =
      new absl::flat_hash_map<se::Platform::Id, CompilerFactory>;
  return r;
}

/* static */
absl::flat_hash_map<se::Platform::Id, std::unique_ptr<Compiler>>*
Compiler::GetPlatformCompilers() {
  static auto* const r =
      new absl::flat_hash_map<se::Platform::Id, std::unique_ptr<Compiler>>;
  return r;
}

/* static */ void Compiler::RegisterCompilerFactory(
    se::Platform::Id platform_id, CompilerFactory compiler_factory) {
  absl::MutexLock lock(platform_compiler_mutex_);
  auto* factories = GetPlatformCompilerFactories();
  CHECK(factories->find(platform_id) == factories->end())
      << "Compiler factory already registered for platform";
  (*factories)[platform_id] = std::move(compiler_factory);
}

/* static */ absl::StatusOr<std::unique_ptr<Compiler>> Compiler::GetForPlatform(
    se::Platform::Id platform_id) {
  absl::MutexLock lock(platform_compiler_mutex_);

  absl::flat_hash_map<se::Platform::Id, Compiler::CompilerFactory>* factories =
      GetPlatformCompilerFactories();
  auto it = factories->find(platform_id);
  if (it == factories->end()) {
    return NotFound(
        "could not find registered compiler for the platform -- was support "
        "for that platform linked in?");
  }
  return it->second();
}

// Default implementation
// TODO(b/256849421) Replace with non-null instantiation of MetricsHookInterface
// with empty implementations.
std::unique_ptr<MetricsHookInterface> Compiler::CreateMetricsHook(
    absl::string_view filename_prefix,
    absl::string_view hlo_module_name) const {
  return nullptr;
}

AotCompilationOptions::AotCompilationOptions()
    : debug_options_(GetDebugOptionsFromFlags()) {}

}  // namespace xla

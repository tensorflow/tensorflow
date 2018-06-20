/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/compiler.h"

#include <string>
#include <utility>

#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"

namespace xla {

/* static */ tensorflow::mutex Compiler::platform_compiler_mutex_(
    tensorflow::LINKER_INITIALIZED);

std::vector<std::unique_ptr<tensorflow::protobuf::Message>>
Compiler::ComputeBackendConfigs(const HloInstruction& hlo,
                                se::StreamExecutor* executor) const {
  CHECK(executor != nullptr);
  return {};
}

// Define a default version where metadata is not used.
StatusOr<std::vector<std::unique_ptr<AotCompilationResult>>>
Compiler::CompileAheadOfTime(
    std::vector<std::unique_ptr<HloModule>> modules,
    const AotCompilationOptions& options,
    std::unique_ptr<AotCompilationMetadata>* metadata) {
  if (metadata != nullptr) {
    return Unimplemented(
        "Populating AotCompilationMetadata is not implemented on this "
        "compiler.");
  }
  return CompileAheadOfTime(std::move(modules), options);
}

/* static */ std::map<se::Platform::Id, Compiler::CompilerFactory>*
Compiler::GetPlatformCompilerFactories() {
  static auto* r = new std::map<se::Platform::Id, CompilerFactory>;
  return r;
}

/* static */
std::map<se::Platform::Id, std::unique_ptr<Compiler>>*
Compiler::GetPlatformCompilers() {
  static auto* r = new std::map<se::Platform::Id, std::unique_ptr<Compiler>>;
  return r;
}

/* static */ void Compiler::RegisterCompilerFactory(
    se::Platform::Id platform_id,
    std::function<std::unique_ptr<Compiler>()> compiler_factory) {
  tensorflow::mutex_lock lock(platform_compiler_mutex_);
  auto* factories = GetPlatformCompilerFactories();
  CHECK(factories->find(platform_id) == factories->end())
      << "Compiler factory already registered for platform";
  (*factories)[platform_id] = std::move(compiler_factory);
}

/* static */ StatusOr<Compiler*> Compiler::GetForPlatform(
    const se::Platform* platform) {
  tensorflow::mutex_lock lock(platform_compiler_mutex_);

  auto* compilers = GetPlatformCompilers();
  // See if we already instantiated a compiler for this platform.
  {
    auto it = compilers->find(platform->id());
    if (it != compilers->end()) {
      return it->second.get();
    }

    // If not, we just fall through to try to create one with a registered
    // factory.
  }

  auto* factories = GetPlatformCompilerFactories();
  auto it = factories->find(platform->id());
  if (it == factories->end()) {
    return NotFound(
        "could not find registered compiler for platform %s -- check "
        "target linkage",
        platform->Name().c_str());
  }

  // And then we invoke the factory, placing the result into the mapping.
  compilers->insert(std::make_pair(platform->id(), it->second()));
  return compilers->at(platform->id()).get();
}

AotCompilationOptions::AotCompilationOptions()
    : debug_options_(legacy_flags::GetDebugOptionsFromFlags()) {}

}  // namespace xla

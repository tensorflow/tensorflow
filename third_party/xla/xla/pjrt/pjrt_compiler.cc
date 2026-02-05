/* Copyright 2022 The OpenXLA Authors.

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

#include "xla/pjrt/pjrt_compiler.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/attributes.h"
#include "absl/base/const_init.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "mlir/IR/BuiltinOps.h"
#include "xla/hlo/builder/xla_computation.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/pjrt/proto/pjrt_partial_program.pb.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {

ABSL_CONST_INIT absl::Mutex registry_mutex(absl::kConstInit);
absl::flat_hash_map<std::pair<std::string, std::string>,
                    std::unique_ptr<PjRtCompiler>>*
CompilerRegistry() {
  static auto* compiler_registry =
      new absl::flat_hash_map<std::pair<std::string, std::string>,
                              std::unique_ptr<PjRtCompiler>>();
  return compiler_registry;
}

absl::flat_hash_map<std::pair<std::string, std::string>, PjRtCompilerFactory>*
CompilerFactoryRegistry() {
  static auto* compiler_factory_registry =
      new absl::flat_hash_map<std::pair<std::string, std::string>,
                              PjRtCompilerFactory>();
  return compiler_factory_registry;
}

// Internal helper to get or create/register the compiler.
absl::StatusOr<PjRtCompiler*> GetOrCreateCompiler(
    absl::string_view platform_name, absl::string_view variant_name) {
  std::pair<std::string, std::string> key{std::string(platform_name),
                                          std::string(variant_name)};

  // Check if compiler has already existed in the compiler registry.
  auto* compiler_registry = CompilerRegistry();
  auto it = compiler_registry->find(key);
  if (it != compiler_registry->end()) {
    return it->second.get();
  }
  LOG(INFO) << "Compiler is not found in the compiler registry for platform: "
            << platform_name << ", variant: " << variant_name;

  // Check if a factory is registered.
  auto* factories = CompilerFactoryRegistry();
  auto factory_it = factories->find(key);
  if (factory_it == factories->end()) {
    return absl::NotFoundError(
        absl::StrCat("No compiler factory for platform: ", platform_name,
                     ", variant: ", variant_name));
  }

  // Create the compiler using the factory.
  TF_ASSIGN_OR_RETURN(auto compiler, factory_it->second());
  auto* compiler_ptr = compiler.get();
  (*compiler_registry)[key] = std::move(compiler);
  return compiler_ptr;
}

void PjRtRegisterCompilerFactory(absl::string_view platform_name,
                                 absl::string_view variant_name,
                                 PjRtCompilerFactory factory) {
  absl::MutexLock l(registry_mutex);
  std::pair<std::string, std::string> key{std::string(platform_name),
                                          std::string(variant_name)};
  CHECK(!CompilerFactoryRegistry()->contains(key))
      << "Variant already registered";
  (*CompilerFactoryRegistry())[key] = std::move(factory);
}

absl::Status PjRtInitializeCompilerVariant(absl::string_view platform_name,
                                           absl::string_view variant_name) {
  absl::MutexLock l(registry_mutex);
  return GetOrCreateCompiler(platform_name, variant_name).status();
}

void PjRtRegisterDefaultCompiler(absl::string_view platform_name,
                                 std::unique_ptr<PjRtCompiler> compiler) {
  PjRtRegisterCompiler(platform_name, /*compiler_variant=*/"",
                       std::move(compiler));
}

void PjRtRegisterCompiler(absl::string_view platform_name,
                          absl::string_view compiler_variant,
                          std::unique_ptr<PjRtCompiler> compiler) {
  CHECK(compiler != nullptr);
  absl::MutexLock l(registry_mutex);
  auto* compiler_registry = CompilerRegistry();
  std::pair<std::string, std::string> key{std::string(platform_name),
                                          std::string(compiler_variant)};
  CHECK(!compiler_registry->contains(key));
  (*compiler_registry)[key] = std::move(compiler);
}

absl::StatusOr<PjRtCompiler*> GetDefaultPjRtCompiler(
    absl::string_view platform_name) {
  return GetPjRtCompiler(platform_name, /*compiler_variant=*/"");
}

absl::StatusOr<PjRtCompiler*> GetPjRtCompiler(
    absl::string_view platform_name, absl::string_view compiler_variant) {
  absl::ReaderMutexLock l(registry_mutex);
  const auto* compiler_registry = CompilerRegistry();
  std::pair<std::string, std::string> key{std::string(platform_name),
                                          std::string(compiler_variant)};
  auto it = compiler_registry->find(key);
  if (it == compiler_registry->end()) {
    return absl::NotFoundError(
        absl::StrCat("No compiler registered for platform ", platform_name,
                     " and compiler variant ", compiler_variant));
  }
  return it->second.get();
}

absl::StatusOr<std::unique_ptr<PjRtExecutable>> PjRtCompile(
    CompileOptions options, const XlaComputation& computation,
    const PjRtTopologyDescription& topology, PjRtClient* client) {
  auto topology_compiler = topology.compiler();
  if (topology_compiler.has_value()) {
    return (*topology_compiler)
        ->Compile(std::move(options), computation, topology, client);
  }

  auto platform_name = topology.platform_name();
  auto compiler_variant = options.compiler_variant.value_or("");
  std::pair<std::string, std::string> key{std::string(platform_name),
                                          std::string(compiler_variant)};
  // Fast path: Try with reader lock if the compiler is already registered.
  {
    absl::ReaderMutexLock l(registry_mutex);
    auto* compiler_registry = CompilerRegistry();
    auto it = compiler_registry->find(key);
    if (it != compiler_registry->end()) {
      return it->second->Compile(std::move(options), computation, topology,
                                 client);
    }
  }

  // Slow path: Acquire exclusive lock to create and compile the compiler if it
  // doesn't exist.
  absl::MutexLock l(registry_mutex);
  TF_ASSIGN_OR_RETURN(PjRtCompiler * compiler,
                      GetOrCreateCompiler(platform_name, compiler_variant));
  return compiler->Compile(std::move(options), computation, topology, client);
}

absl::StatusOr<std::unique_ptr<PjRtExecutable>> PjRtCompile(
    CompileOptions options, mlir::ModuleOp module,
    const PjRtTopologyDescription& topology, PjRtClient* client) {
  auto topology_compiler = topology.compiler();
  if (topology_compiler.has_value()) {
    return (*topology_compiler)
        ->Compile(std::move(options), module, topology, client);
  }

  auto platform_name = topology.platform_name();
  auto compiler_variant = options.compiler_variant.value_or("");
  std::pair<std::string, std::string> key{std::string(platform_name),
                                          std::string(compiler_variant)};
  // Fast path: Try with reader lock if the compiler is already registered.
  {
    absl::ReaderMutexLock l(registry_mutex);
    auto* compiler_registry = CompilerRegistry();
    auto it = compiler_registry->find(key);
    if (it != compiler_registry->end()) {
      return it->second->Compile(std::move(options), module, topology, client);
    }
  }

  // Slow path: Acquire exclusive lock to create and compile the compiler if it
  // doesn't exist.
  absl::MutexLock l(registry_mutex);
  TF_ASSIGN_OR_RETURN(PjRtCompiler * compiler,
                      GetOrCreateCompiler(platform_name, compiler_variant));
  return compiler->Compile(std::move(options), module, topology, client);
}

absl::Status PjRtPhaseCompiler::RegisterPhase(
    const std::string& phase_name, CompilationPhaseFunctions phase_functions) {
  if (phase_name.empty()) {
    return absl::InvalidArgumentError("Phase name cannot be empty");
  }
  if (!phase_functions.compiler) {
    return absl::InvalidArgumentError("Phase compiler cannot be null");
  }
  if (!phase_functions.validator) {
    return absl::InvalidArgumentError("Phase validator cannot be null");
  }
  if (!phase_map_.insert({phase_name, std::move(phase_functions)}).second) {
    return absl::AlreadyExistsError(
        absl::StrCat("A phase compiler/validator with Phase name \"",
                     phase_name, "\" already exists"));
  }
  phase_names_.push_back(phase_name);
  return absl::OkStatus();
}

absl::StatusOr<std::vector<std::string>> PjRtPhaseCompiler::GetPhaseNames() {
  return phase_names_;
}

absl::StatusOr<std::vector<PjRtPartialProgramProto>>
PjRtPhaseCompiler::RunPhases(
    CompileOptions options,
    const std::vector<PjRtPartialProgramProto>& input_programs,
    const PjRtTopologyDescription& topology,
    const std::vector<std::string>& phases_to_run) {
  std::vector<PjRtPartialProgramProto> programs = input_programs;
  for (const auto& phase_name : phases_to_run) {
    auto it = phase_map_.find(phase_name);
    if (it == phase_map_.end()) {
      return absl::NotFoundError(
          absl::StrCat("No phase compiler/validator registered with phase name "
                       "\"",
                       phase_name, "\""));
    }

    // Validate (plugin specific) the input programs.
    auto validation_status = it->second.validator(programs);
    if (!validation_status.ok()) {
      return validation_status;
    }

    // Run the phase.
    auto out_programs = it->second.compiler(options, programs, topology);
    if (!out_programs.ok()) {
      return out_programs.status();
    }
    programs = *out_programs;
  }

  return programs;
}

}  // namespace xla

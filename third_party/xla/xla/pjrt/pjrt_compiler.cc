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
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/attributes.h"
#include "absl/base/const_init.h"
#include "absl/base/no_destructor.h"
#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "mlir/IR/BuiltinOps.h"
#include "xla/hlo/builder/xla_computation.h"
#include "xla/pjrt/maybe_owning_mlir_module.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/pjrt/proto/pjrt_partial_program.pb.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {

PjRtCompilerRegistry& PjRtCompilerRegistry::Global() {
  static absl::NoDestructor<PjRtCompilerRegistry> global_registry;
  return *global_registry;
}

absl::Status PjRtCompilerRegistry::RegisterFactory(
    absl::string_view platform_name, absl::string_view variant_name,
    PjRtCompilerFactory factory) {
  std::pair<std::string, std::string> key{std::string(platform_name),
                                          std::string(variant_name)};
  absl::MutexLock l(factory_mutex_);
  if (factories_.contains(key)) {
    return absl::AlreadyExistsError(
        absl::StrCat("Factory already registered for platform: ", platform_name,
                     ", variant: ", variant_name));
  }
  factories_[key] = std::move(factory);
  return absl::OkStatus();
}

absl::Status PjRtCompilerRegistry::RegisterCompiler(
    absl::string_view platform_name, absl::string_view variant_name,
    std::unique_ptr<PjRtCompiler> compiler) {
  if (compiler == nullptr) {
    return absl::InvalidArgumentError("Compiler cannot be null");
  }
  std::pair<std::string, std::string> key{std::string(platform_name),
                                          std::string(variant_name)};
  absl::MutexLock l(compiler_mutex_);
  if (compilers_.contains(key)) {
    return absl::AlreadyExistsError(absl::StrCat(
        "Compiler already registered for platform: ", platform_name,
        ", variant: ", variant_name));
  }
  compilers_[key] = std::move(compiler);
  return absl::OkStatus();
}

absl::StatusOr<PjRtCompiler*> PjRtCompilerRegistry::GetOrCreateCompiler(
    absl::string_view platform_name, absl::string_view variant_name)
    ABSL_LOCKS_EXCLUDED(compiler_mutex_, factory_mutex_) {
  std::pair<std::string, std::string> key{std::string(platform_name),
                                          std::string(variant_name)};

  // Check if compiler has already existed in the compiler registry.
  {
    absl::MutexLock l(compiler_mutex_);
    auto it = compilers_.find(key);
    if (it != compilers_.end()) {
      return it->second.get();
    }
  }
  LOG(INFO) << "Compiler is not found in the compiler registry for platform: "
            << platform_name << ", variant: " << variant_name
            << ". Trying to create a new compiler with its compiler factory.";

  // Check if a factory is registered.
  PjRtCompilerFactory factory;
  {
    absl::MutexLock l(factory_mutex_);
    auto factory_it = factories_.find(key);
    if (factory_it == factories_.end()) {
      return absl::NotFoundError(absl::StrCat(
          "No compiler factory for platform: ", platform_name,
          ", variant: ", variant_name,
          ". Please register the compiler factory before using it."));
    }
    factory = factory_it->second;
  }

  // Create the compiler using the factory.
  TF_ASSIGN_OR_RETURN(std::unique_ptr<PjRtCompiler> compiler, factory());
  auto* compiler_ptr = compiler.get();

  {
    absl::MutexLock l(compiler_mutex_);
    auto [it, inserted] = compilers_.try_emplace(key, std::move(compiler));
    if (!inserted) {
      return it->second.get();
    }
  }
  return compiler_ptr;
}

absl::StatusOr<PjRtCompiler*> PjRtCompilerRegistry::GetCompiler(
    absl::string_view platform_name, absl::string_view variant_name) {
  return GetOrCreateCompiler(platform_name, variant_name);
}

absl::Status PjRtCompilerRegistry::InitializeVariant(
    absl::string_view platform_name, absl::string_view variant_name) {
  return GetOrCreateCompiler(platform_name, variant_name).status();
}

absl::Status PjRtCompilerRegistry::InitializeAllVariants() {
  std::vector<std::pair<std::string, std::string>> keys;
  {
    absl::MutexLock l(factory_mutex_);
    for (const auto& [key, factory] : factories_) {
      keys.push_back(key);
    }
  }

  for (const auto& key : keys) {
    TF_RETURN_IF_ERROR(InitializeVariant(key.first, key.second));
  }
  return absl::OkStatus();
}

void PjRtRegisterCompilerFactory(absl::string_view platform_name,
                                 absl::string_view variant_name,
                                 PjRtCompilerFactory factory) {
  CHECK_OK(PjRtCompilerRegistry::Global().RegisterFactory(
      platform_name, variant_name, std::move(factory)));
}

absl::Status PjRtInitializeCompilerVariant(absl::string_view platform_name,
                                           absl::string_view variant_name) {
  return PjRtCompilerRegistry::Global().InitializeVariant(platform_name,
                                                          variant_name);
}

absl::Status PjRtInitializeCompilerVariants() {
  return PjRtCompilerRegistry::Global().InitializeAllVariants();
}

void PjRtRegisterDefaultCompiler(absl::string_view platform_name,
                                 std::unique_ptr<PjRtCompiler> compiler) {
  CHECK_OK(PjRtCompilerRegistry::Global().RegisterCompiler(
      platform_name,
      /*variant_name=*/"", std::move(compiler)));
}

void PjRtRegisterCompiler(absl::string_view platform_name,
                          absl::string_view compiler_variant,
                          std::unique_ptr<PjRtCompiler> compiler) {
  CHECK_OK(PjRtCompilerRegistry::Global().RegisterCompiler(
      platform_name, compiler_variant, std::move(compiler)));
}

absl::StatusOr<PjRtCompiler*> GetDefaultPjRtCompiler(
    absl::string_view platform_name) {
  return PjRtCompilerRegistry::Global().GetCompiler(platform_name,
                                                    /*variant_name=*/"");
}

absl::StatusOr<PjRtCompiler*> GetPjRtCompiler(
    absl::string_view platform_name, absl::string_view compiler_variant) {
  return PjRtCompilerRegistry::Global().GetCompiler(platform_name,
                                                    compiler_variant);
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
  TF_ASSIGN_OR_RETURN(PjRtCompiler * compiler,
                      GetPjRtCompiler(platform_name, compiler_variant));
  return compiler->Compile(std::move(options), computation, topology, client);
}

absl::StatusOr<std::unique_ptr<PjRtExecutable>> PjRtCompile(
    CompileOptions options, MaybeOwningMlirModule module,
    const PjRtTopologyDescription& topology, PjRtClient* client) {
  if (std::optional<PjRtCompiler*> topology_compiler = topology.compiler()) {
    return (*topology_compiler)
        ->Compile(std::move(options), std::move(module), topology, client);
  }
  auto platform_name = topology.platform_name();
  auto compiler_variant = options.compiler_variant.value_or("");
  TF_ASSIGN_OR_RETURN(PjRtCompiler * compiler,
                      GetPjRtCompiler(platform_name, compiler_variant));
  return compiler->Compile(std::move(options), std::move(module), topology,
                           client);
}

absl::StatusOr<std::unique_ptr<PjRtExecutable>> PjRtCompile(
    CompileOptions options, mlir::ModuleOp module,
    const PjRtTopologyDescription& topology, PjRtClient* client) {
  return PjRtCompile(std::move(options),
                     MaybeOwningMlirModule(std::move(module)), topology,
                     client);
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

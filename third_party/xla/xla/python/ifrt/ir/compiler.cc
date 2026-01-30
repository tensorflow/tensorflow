/* Copyright 2026 The OpenXLA Authors.

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

#include "xla/python/ifrt/ir/compiler.h"

#include <memory>
#include <utility>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "llvm/Support/Casting.h"
#include "xla/python/ifrt/compiler.h"
#include "xla/python/ifrt/device.h"
#include "xla/python/ifrt/device_list.h"
#include "xla/python/ifrt/executable.h"
#include "xla/python/ifrt/ir/atom_program_compiler.h"
#include "xla/python/ifrt/ir/compiled_ifrt_ir_program.h"
#include "xla/python/ifrt/ir/ifrt_ir_executable_version.h"
#include "xla/python/ifrt/ir/ifrt_ir_loaded_executable.h"
#include "xla/python/ifrt/ir/ifrt_ir_program.h"
#include "xla/python/ifrt/ir/serialization_utils.h"
#include "xla/python/ifrt/ir/version.h"
#include "xla/python/ifrt/program.h"
#include "xla/python/ifrt/topology.h"
#include "xla/tsl/concurrency/future.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"
#include "tsl/profiler/lib/traceme.h"

namespace xla {
namespace ifrt {

namespace {

std::unique_ptr<xla::ifrt::IfrtIrExecutableVersion>
GetCurrentIfrtIrExecutableVersion() {
  // Only the version number is needed here for compatibility checking.
  return std::make_unique<xla::ifrt::IfrtIrExecutableVersion>(
      Version::getCurrentVersion());
}

}  // namespace

char IfrtIrProgramCompiler::ID = 0;

IfrtIrProgramCompiler::IfrtIrProgramCompiler(
    xla::ifrt::Client* client,
    AtomProgramCompilerFactory atom_program_compiler_factory)
    : client_(client),
      atom_program_compiler_factory_(std::move(atom_program_compiler_factory)) {
}

tsl::Future<xla::ifrt::LoadedExecutableRef>
IfrtIrProgramCompiler::CompileAndLoad(
    std::unique_ptr<xla::ifrt::Program> program,
    std::unique_ptr<xla::ifrt::CompileOptions> options) {
  tsl::profiler::TraceMe traceme("IfrtIrProgramCompiler::CompileAndLoad");

  if (!llvm::isa_and_nonnull<xla::ifrt::IfrtIRProgram>(program.get())) {
    return absl::InvalidArgumentError(
        "IFRT IR compiler requires an IFRT IR program");
  }
  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<xla::ifrt::IfrtIRCompileOptions> ifrt_ir_compile_options,
      xla::ifrt::GetIfrtIRCompileOptions(std::move(options)));
  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<xla::ifrt::AtomProgramCompiler> atom_program_compiler,
      atom_program_compiler_factory_(*ifrt_ir_compile_options));

  auto [promise, future] = tsl::MakePromise<LoadedExecutableRef>();

  tsl::Env::Default()->SchedClosure(
      [client = client_, program = std::move(program),
       ifrt_ir_compile_options = std::move(ifrt_ir_compile_options),
       atom_program_compiler = std::move(atom_program_compiler),
       promise = std::move(promise)]() mutable {
        tsl::Future<std::shared_ptr<CompiledIfrtIrProgram>> compiled_program =
            CompiledIfrtIrProgram::Create(
                xla::unique_ptr_down_cast<xla::ifrt::IfrtIRProgram>(
                    std::move(program)),
                std::move(ifrt_ir_compile_options), client,
                std::move(atom_program_compiler));

        compiled_program.OnReady(
            [promise = std::move(promise), client = client](
                absl::StatusOr<std::shared_ptr<CompiledIfrtIrProgram>>
                    compiled_program) mutable {
              if (compiled_program.ok()) {
                promise.Set(IfrtIrLoadedExecutable::Create(
                    client, *std::move(compiled_program)));
              } else {
                promise.Set(compiled_program.status());
              }
            });
      });
  return future;
}

tsl::Future<xla::ifrt::ExecutableRef> IfrtIrProgramCompiler::Compile(
    std::unique_ptr<xla::ifrt::Program> program,
    const xla::ifrt::Topology& topology,
    std::unique_ptr<xla::ifrt::CompileOptions> options) {
  return absl::UnimplementedError(
      "IFRT IR compiler does not support AOT compilation");
}

absl::Status IfrtIrProgramCompiler::IsExecutableVersionCompatible(
    const xla::ifrt::ExecutableVersion& version,
    const xla::ifrt::DeviceListRef& devices) const {
  const xla::ifrt::IfrtIrExecutableVersion* executable_version =
      llvm::dyn_cast<xla::ifrt::IfrtIrExecutableVersion>(&version);
  if (!executable_version) {
    return absl::InvalidArgumentError(
        "Executable version is an unsupported type");
  }

  absl::Status compatibility =
      GetCurrentIfrtIrExecutableVersion()->IsCompatibleWith(
          *client_, devices, *executable_version);
  if (!compatibility.ok()) {
    tsl::errors::AppendToMessage(
        &compatibility,
        absl::StrCat("Incompatible IFRT IR executable version: current: ",
                     GetCurrentIfrtIrExecutableVersion()->ToString(), " vs ",
                     executable_version->ToString()));
    return compatibility;
  }
  return absl::OkStatus();
}

tsl::Future<xla::ifrt::LoadedExecutableRef>
IfrtIrProgramCompiler::DeserializeLoadedExecutable(
    absl::string_view serialized,
    std::unique_ptr<xla::ifrt::DeserializeExecutableOptions> options) {
  tsl::profiler::TraceMe traceme(
      "IfrtIrProgramCompiler::DeserializeLoadedExecutable");

  std::unique_ptr<xla::ifrt::DeserializeIfrtIRProgramOptions>
      deserialize_options;
  if (options != nullptr) {
    if (llvm::isa_and_nonnull<DeserializeIfrtIRProgramOptions>(options.get())) {
      deserialize_options =
          xla::unique_ptr_down_cast<DeserializeIfrtIRProgramOptions>(
              std::move(options));
    } else {
      return absl::InvalidArgumentError(
          "Options must be of type DeserializeIfrtIRProgramOptions");
    }
  }

  TF_ASSIGN_OR_RETURN(DeserializedIfrtIRProgram deserialized_ifrt_executable,
                      DeserializeIfrtIrExecutable(
                          client_, serialized, std::move(deserialize_options)));

  return CompileAndLoad(
      std::unique_ptr<xla::ifrt::IfrtIRProgram>(
          std::move(deserialized_ifrt_executable.program)),
      std::move(deserialized_ifrt_executable.compile_options));
}

}  // namespace ifrt
}  // namespace xla

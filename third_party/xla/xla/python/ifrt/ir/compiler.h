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
#ifndef XLA_PYTHON_IFRT_IR_COMPILER_H_
#define XLA_PYTHON_IFRT_IR_COMPILER_H_

#include <memory>

#include "absl/functional/any_invocable.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "llvm/Support/ExtensibleRTTI.h"
#include "xla/python/ifrt/client.h"
#include "xla/python/ifrt/compiler.h"
#include "xla/python/ifrt/device_list.h"
#include "xla/python/ifrt/executable.h"
#include "xla/python/ifrt/ir/atom_program_compiler.h"
#include "xla/python/ifrt/ir/ifrt_ir_program.h"
#include "xla/python/ifrt/program.h"
#include "xla/python/ifrt/topology.h"
#include "xla/tsl/concurrency/future.h"

namespace xla {
namespace ifrt {

// Implements the IFRT IR compiler. Accepts `IfrtIRProgram` and returns a
// `LoadedExecutable` that runs the compiled IR program.
class IfrtIrProgramCompiler final
    : public llvm::RTTIExtends<IfrtIrProgramCompiler, Compiler> {
 public:
  static char ID;  // NOLINT

  // Callback that creates an `AtomProgramCompiler` from the given compile
  // options.
  using AtomProgramCompilerFactory =
      absl::AnyInvocable<absl::StatusOr<std::unique_ptr<AtomProgramCompiler>>(
          const IfrtIRCompileOptions&) const>;

  IfrtIrProgramCompiler(
      Client* client, AtomProgramCompilerFactory atom_program_compiler_factory);

  tsl::Future<LoadedExecutableRef> CompileAndLoad(
      std::unique_ptr<Program> program,
      std::unique_ptr<CompileOptions> options) override;

  tsl::Future<ExecutableRef> Compile(
      std::unique_ptr<Program> program, const Topology& topology,
      std::unique_ptr<CompileOptions> options) override;

  absl::Status IsExecutableVersionCompatible(
      const ExecutableVersion& executable_version,
      const DeviceListRef& devices) const override;

  tsl::Future<LoadedExecutableRef> DeserializeLoadedExecutable(
      absl::string_view serialized,
      std::unique_ptr<DeserializeExecutableOptions> options) override;

 private:
  Client* const client_;
  const AtomProgramCompilerFactory atom_program_compiler_factory_;
};

}  // namespace ifrt
}  // namespace xla

#endif  // XLA_PYTHON_IFRT_IR_COMPILER_H_

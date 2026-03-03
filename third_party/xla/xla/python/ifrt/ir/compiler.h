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

// Implements the IFRT IR compiler. Accepts `xla::ifrt::IfrtIRProgram` and
// returns a `LoadedExecutable` that runs the compiled IR program.
class IfrtIrProgramCompiler final
    : public llvm::RTTIExtends<IfrtIrProgramCompiler, xla::ifrt::Compiler> {
 public:
  static char ID;  // NOLINT

  // Callback that creates an `AtomProgramCompiler` from the given compile
  // options.
  using AtomProgramCompilerFactory = absl::AnyInvocable<
      absl::StatusOr<std::unique_ptr<xla::ifrt::AtomProgramCompiler>>(
          const xla::ifrt::IfrtIRCompileOptions&) const>;

  using xla::ifrt::Compiler::Compile;

  IfrtIrProgramCompiler(
      xla::ifrt::Client* client,
      AtomProgramCompilerFactory atom_program_compiler_factory);

  tsl::Future<xla::ifrt::LoadedExecutableRef> CompileAndLoad(
      std::unique_ptr<xla::ifrt::Program> program,
      std::unique_ptr<xla::ifrt::CompileOptions> options) override;

  tsl::Future<xla::ifrt::ExecutableRef> Compile(
      std::unique_ptr<xla::ifrt::Program> program,
      const xla::ifrt::Topology& topology,
      std::unique_ptr<xla::ifrt::CompileOptions> options) override;

  absl::Status IsExecutableVersionCompatible(
      const xla::ifrt::ExecutableVersion& executable_version,
      const xla::ifrt::DeviceListRef& devices) const override;

  tsl::Future<xla::ifrt::LoadedExecutableRef> DeserializeLoadedExecutable(
      absl::string_view serialized,
      std::unique_ptr<xla::ifrt::DeserializeExecutableOptions> options)
      override;

 private:
  xla::ifrt::Client* const client_;
  const AtomProgramCompilerFactory atom_program_compiler_factory_;
};

}  // namespace ifrt
}  // namespace xla

#endif  // XLA_PYTHON_IFRT_IR_COMPILER_H_

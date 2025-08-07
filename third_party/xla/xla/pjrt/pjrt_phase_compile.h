/* Copyright 2025 The OpenXLA Authors.

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

#ifndef XLA_PJRT_PJRT_PHASE_COMPILE_H_
#define XLA_PJRT_PJRT_PHASE_COMPILE_H_

#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "absl/status/statusor.h"
#include "xla/pjrt/c/pjrt_c_api.h"
#include "xla/pjrt/c/pjrt_c_api_phase_compile_extension.h"
#include "xla/pjrt/pjrt_compiler.h"
#include "xla/pjrt/pjrt_executable.h"

// This file defines the C++ convenience wrappers and internal functions for
// the PJRT Phase Compile C API extension. It provides higher-level interfaces
// for interacting with the phase compilation features, bridging between the
// low-level C API and XLA's C++ components.

namespace pjrt {

class CApiPjrtPhaseCompiler {
 public:
  CApiPjrtPhaseCompiler(
      const PJRT_Api* api,
      const PJRT_PhaseCompile_Extension* phase_compile_extension,
      const PJRT_PhaseCompiler* phase_compiler)
      : api_(api),
        phase_compile_extension_(phase_compile_extension),
        phase_compiler_(
            phase_compiler,
            [phase_compile_extension](const PJRT_PhaseCompiler* p_compiler) {
              PJRT_PhaseCompile_Destroy_Compiler_Args destroy_args;
              destroy_args.struct_size =
                  PJRT_PhaseCompile_Destroy_Compiler_Args_STRUCT_SIZE;
              destroy_args.extension_start = nullptr;
              destroy_args.phase_compiler = p_compiler;
              phase_compile_extension->phase_compile_destroy_compiler(
                  &destroy_args);
            }) {}

  // Runs the compilation phase with the given phases 'phases_to_run' on the
  // input programs 'partial_programs_in' and returns the output programs.
  // This function first performs plugin-agnostic validation on the inputs and
  // phase names. Next, it triggers plugin-specific phase validators for input
  // compatibility before invoking the appropriate phase compilers.
  absl::StatusOr<std::vector<xla::PjRtPartialProgramProto>> RunPhases(
      xla::CompileOptions options,
      const std::vector<xla::PjRtPartialProgramProto>& partial_programs_in,
      const xla::PjRtTopologyDescription& topology,
      const std::vector<std::string>& phases_to_run);

  // Returns the names of all the phases following the order of their
  // registration.
  absl::StatusOr<std::vector<std::string>> GetPhaseNames();

  // Returns the reference to the phase compiler.
  const PJRT_PhaseCompiler* GetPhaseCompiler() const {
    return phase_compiler_.get();
  }

 private:
  const PJRT_Api* api_ = nullptr;  // Not owned.
  const PJRT_PhaseCompile_Extension* phase_compile_extension_ =
      nullptr;  // Not owned.
  std::unique_ptr<const PJRT_PhaseCompiler,
                  std::function<void(const PJRT_PhaseCompiler*)>>
      phase_compiler_;
};

// Creates and initializes a `CApiPjrtPhaseCompiler` instance.
absl::StatusOr<std::unique_ptr<CApiPjrtPhaseCompiler>> GetCApiPjrtPhaseCompiler(
    const PJRT_Api* api);

}  // namespace pjrt

#endif  // XLA_PJRT_PJRT_PHASE_COMPILE_H_

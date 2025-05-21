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

#ifndef XLA_PJRT_C_PJRT_C_API_PHASE_COMPILE_INTERNAL_H_
#define XLA_PJRT_C_PJRT_C_API_PHASE_COMPILE_INTERNAL_H_

#include <string>
#include <vector>

#include "absl/status/statusor.h"
#include "xla/pjrt/c/pjrt_c_api.h"
#include "xla/pjrt/c/pjrt_c_api_phase_compile_extension.h"
#include "xla/pjrt/pjrt_compiler.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/pjrt/proto/pjrt_partial_program.pb.h"

namespace pjrt {

// This file defines the C++ convenience wrappers and internal functions for
// the PJRT Phase Compile C API extension. It provides higher-level interfaces
// for interacting with the phase compilation features, bridging between the
// low-level C API and XLA's C++ components.

// Runs the compilation phase with the given phases 'phases_to_run' on the
// input programs 'partial_programs_in' and returns the output programs.
// This function first performs plugin-agnostic validation on the inputs and
// phase names. Next, it triggers plugin-specific phase validators for input
// compatibility before invoking the appropriate phase compilers.
absl::StatusOr<std::vector<xla::PjRtPartialProgramProto>> RunPhases(
    const PJRT_Api* api, xla::CompileOptions options,
    const std::vector<xla::PjRtPartialProgramProto>& partial_programs_in,
    const std::vector<std::string>& phases_to_run);

// Returns the names of all the phases following the order of their
// registration.
absl::StatusOr<std::vector<std::string>> GetPhaseNames(const PJRT_Api* api);

// Creates and initializes a PJRT_PhaseCompile_Extension struct. This function
// is used by plugins to create and chain the phase compilation extension
// into the PJRT C API structure.
PJRT_PhaseCompile_Extension CreatePhaseCompileExtension(
    PJRT_Extension_Base* next);

}  // namespace pjrt

#endif  // XLA_PJRT_C_PJRT_C_API_PHASE_COMPILE_INTERNAL_H_

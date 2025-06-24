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

#ifndef XLA_PJRT_C_PJRT_C_API_PHASE_COMPILE_INTERNAL_H_
#define XLA_PJRT_C_PJRT_C_API_PHASE_COMPILE_INTERNAL_H_

#include "xla/pjrt/c/pjrt_c_api.h"
#include "xla/pjrt/c/pjrt_c_api_phase_compile_extension.h"

namespace pjrt {

// Creates and initializes a PJRT_PhaseCompile_Extension struct. This function
// is used by plugins to create and chain the phase compilation extension
// into the PJRT C API structure.
PJRT_PhaseCompile_Extension CreatePhaseCompileExtension(
    PJRT_Extension_Base* next, PJRT_PhaseCompile_Get_Compiler get_compiler,
    PJRT_PhaseCompile_Destroy_Compiler destroy_compiler);

}  // namespace pjrt

// namespace pjrt

#endif  // XLA_PJRT_C_PJRT_C_API_PHASE_COMPILE_INTERNAL_H_

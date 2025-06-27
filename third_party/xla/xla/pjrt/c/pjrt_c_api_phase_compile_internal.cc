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

#include "xla/pjrt/c/pjrt_c_api_phase_compile_internal.h"

#include <cassert>

#include "xla/pjrt/c/pjrt_c_api.h"
#include "xla/pjrt/c/pjrt_c_api_phase_compile_extension.h"

namespace pjrt {

PJRT_PhaseCompile_Extension CreatePhaseCompileExtension(
    PJRT_Extension_Base* next, PJRT_PhaseCompile_Get_Compiler get_compiler,
    PJRT_PhaseCompile_Destroy_Compiler destroy_compiler) {
  return {
      PJRT_Extension_Base{
          /*struct_size=*/PJRT_PhaseCompile_Extension_STRUCT_SIZE,
          /*type=*/PJRT_Extension_Type_PhaseCompile,
          /*next=*/next,
      },
      /*phase_compile_get_compiler=*/get_compiler,
      /*phase_compile_destroy_compiler=*/destroy_compiler,
      /*phase_compile_run_phases=*/nullptr,
      /*phase_compile_get_phase_names=*/
      nullptr,
  };
}
}  // namespace pjrt

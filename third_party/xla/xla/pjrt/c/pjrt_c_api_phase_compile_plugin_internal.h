/* Copyright 2023 The OpenXLA Authors.

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

#ifndef XLA_PJRT_C_PJRT_C_API_PHASE_COMPILE_PLUGIN_INTERNAL_H_
#define XLA_PJRT_C_PJRT_C_API_PHASE_COMPILE_PLUGIN_INTERNAL_H_

#include "xla/pjrt/c/pjrt_c_api.h"

namespace pjrt {
namespace phase_compile_cpu_plugin {

// This file provides internal utilities and the entry point for a sample
// CPU plugin that demonstrates PJRT's phased compilation capabilities.
// It declares the function responsible for providing the PJRT API for this
// specific plugin.

// Returns a pointer to a statically allocated PJRT API with the phase compile
// extension. The caller does not own the returned pointer and should not free
// it.
const PJRT_Api* GetPhaseCompilePjrtApi();

}  // namespace phase_compile_cpu_plugin
}  // namespace pjrt

#endif  // XLA_PJRT_C_PJRT_C_API_PHASE_COMPILE_PLUGIN_INTERNAL_H_

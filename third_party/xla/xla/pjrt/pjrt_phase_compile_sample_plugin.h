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

#ifndef XLA_PJRT_PJRT_PHASE_COMPILE_SAMPLE_PLUGIN_H_
#define XLA_PJRT_PJRT_PHASE_COMPILE_SAMPLE_PLUGIN_H_

#include "xla/pjrt/c/pjrt_c_api_phase_compile_extension.h"

namespace pjrt {
namespace phase_compile_sample_plugin {

// This file demonstrates the artifacts a plugin developer needs to provide
// to create a phase compile plugin. Specifically, it shows the declaration of
// `PJRT_PhaseCompile_Extension` which contains all the functions that the
// plugin needs to implement.

// Creates a phase compile extension for the sample plugin.
PJRT_PhaseCompile_Extension CreateSamplePhaseCompileExtension();

}  // namespace phase_compile_sample_plugin
}  // namespace pjrt

#endif  // XLA_PJRT_PJRT_PHASE_COMPILE_SAMPLE_PLUGIN_H_

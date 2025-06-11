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
#include "xla/pjrt/c/pjrt_c_api_phase_compile_plugin.h"

#include "xla/pjrt/c/pjrt_c_api.h"
#include "xla/pjrt/c/pjrt_c_api_phase_compile_plugin_internal.h"

const PJRT_Api* GetPjrtApi(bool test_flag_null_phase_extension,
                           bool test_flag_illegal_platform) {
  if (test_flag_null_phase_extension) {
    return pjrt::phase_compile_cpu_plugin::GetPjrtApiWithNullPhaseExtension();
  }
  if (test_flag_illegal_platform) {
    return pjrt::phase_compile_cpu_plugin::GetPjrtApiWithInvalidPlatform();
  }
  return pjrt::phase_compile_cpu_plugin::GetPhaseCompileForCpuPjrtApi();
}

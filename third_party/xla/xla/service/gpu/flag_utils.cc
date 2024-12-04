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

#include "xla/service/gpu/flag_utils.h"

#include "xla/service/hlo_module_config.h"

namespace xla {
namespace gpu {

float ExecTimeOptimizationEffort(const HloModuleConfig& config) {
  float flag_exec_effort =
      config.debug_options().xla_experimental_exec_time_optimization_effort();
  if (flag_exec_effort != 0.0) {
    return flag_exec_effort;
  }
  return config.exec_time_optimization_effort();
}

}  // namespace gpu
}  // namespace xla

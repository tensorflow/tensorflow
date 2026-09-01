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

#ifndef XLA_TOOLS_XLA_CPU_COMPILE_LIB_H_
#define XLA_TOOLS_XLA_CPU_COMPILE_LIB_H_

#include <memory>
#include <optional>
#include <string>

#include "absl/status/statusor.h"
#include "xla/backends/cpu/target_machine_options.h"
#include "xla/hlo/ir/hlo_module.h"

namespace xla {

// Compiles the provided HLO module for CPU using AOT.
absl::StatusOr<std::string> AotCompileCpuExecutable(
    std::unique_ptr<HloModule> hlo_module,
    std::optional<cpu::TargetMachineOptions> target_config);

}  // namespace xla

#endif  // XLA_TOOLS_XLA_CPU_COMPILE_LIB_H_

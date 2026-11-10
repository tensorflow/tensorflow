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

#ifndef XLA_SERVICE_GPU_AUTOTUNING_GPU_DEFAULT_BACKEND_POLICY_H_
#define XLA_SERVICE_GPU_AUTOTUNING_GPU_DEFAULT_BACKEND_POLICY_H_

#include <optional>

#include "xla/backends/autotuner/backends.pb.h"
#include "xla/hlo/ir/hlo_instruction.h"

namespace xla::gpu {

// Returns the backend to try first when `instr` isn't autotuned online, or
// nullopt if there is no preference. The backend may not support `instr`, so
// callers must fall back to other backends.
std::optional<autotuner::Backend> GetDefaultBackend(
    const HloInstruction& instr);

}  // namespace xla::gpu

#endif  // XLA_SERVICE_GPU_AUTOTUNING_GPU_DEFAULT_BACKEND_POLICY_H_

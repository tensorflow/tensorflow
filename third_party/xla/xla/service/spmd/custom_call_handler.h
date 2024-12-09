/* Copyright 2021 The OpenXLA Authors.

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

#ifndef XLA_SERVICE_SPMD_CUSTOM_CALL_HANDLER_H_
#define XLA_SERVICE_SPMD_CUSTOM_CALL_HANDLER_H_

#include <cstdint>
#include <memory>

#include "xla/hlo/ir/hlo_instruction.h"

namespace xla {
namespace spmd {

// Creators of custom ops defined by the partitioner itself.

// Creates a custom op that rotates data along `dim` with the given amount.
std::unique_ptr<HloInstruction> CreateCustomCallSPMDInternal_RotateRight(
    HloInstruction* input, int64_t dim, int64_t amount);

}  // namespace spmd
}  // namespace xla

#endif  // XLA_SERVICE_SPMD_CUSTOM_CALL_HANDLER_H_

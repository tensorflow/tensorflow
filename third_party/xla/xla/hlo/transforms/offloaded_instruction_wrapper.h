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

#ifndef XLA_HLO_TRANSFORMS_OFFLOADED_INSTRUCTION_WRAPPER_H_
#define XLA_HLO_TRANSFORMS_OFFLOADED_INSTRUCTION_WRAPPER_H_

#include <utility>
#include <vector>

#include "absl/functional/function_ref.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"

namespace xla::offloader_util {

absl::Status RecursivelyClearComputeTypeFrontendAttribute(
    HloComputation* computation);

// Attempts to create computations islands out of a connected set of
// instructions that satisfy `should_offload`.
// Returns a vector of pairs, where the first element is the offloaded
// instruction and the second element is its replacement, namely the call to the
// newly created computation that contains the offloaded instruction.
absl::StatusOr<std::vector<std::pair<HloInstruction*, HloCallInstruction*>>>
FindAndWrapOffloadedComputations(
    HloComputation& computation,
    absl::FunctionRef<bool(const HloInstruction*)> should_offload,
    absl::FunctionRef<bool(const HloInstruction&, const HloInstruction&)>
        should_fuse,
    absl::FunctionRef<absl::Status(HloInstruction*)>
        clear_backend_config_device_type,
    absl::string_view new_call_name_prefix);

}  // namespace xla::offloader_util

#endif  // XLA_HLO_TRANSFORMS_OFFLOADED_INSTRUCTION_WRAPPER_H_

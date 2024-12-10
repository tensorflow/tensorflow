/* Copyright 2018 The OpenXLA Authors.

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

#ifndef XLA_SERVICE_LLVM_IR_BUFFER_ASSIGNMENT_UTIL_H_
#define XLA_SERVICE_LLVM_IR_BUFFER_ASSIGNMENT_UTIL_H_

#include <string>

#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/literal.h"
#include "xla/service/buffer_assignment.h"

namespace xla {
namespace llvm_ir {
// Sanitizes the HLO constant instruction name so that it can be used for the
// name of the corresponding constant buffer. In particular, it replaces . and
// - with _.
std::string SanitizeConstantName(const HloInstruction& instr);
std::string SanitizeConstantName(absl::string_view name);

std::string ConstantHloToGlobalName(const HloInstruction& instr);
std::string ConstantNameToGlobalName(absl::string_view name);

// Returns the Literal corresponding to `allocation`, which must be a constant
// allocation.
const Literal& LiteralForConstantAllocation(const BufferAllocation& allocation);
// Returns the constant HloInstruction corresponding to `allocation`, which must
// be a constant allocation.
const HloInstruction& InstrForConstantBufferAllocation(
    const BufferAllocation& allocation);
}  // namespace llvm_ir
}  // namespace xla

#endif  // XLA_SERVICE_LLVM_IR_BUFFER_ASSIGNMENT_UTIL_H_

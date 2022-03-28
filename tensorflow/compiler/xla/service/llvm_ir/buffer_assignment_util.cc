/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/llvm_ir/buffer_assignment_util.h"

#include <algorithm>

#include "absl/strings/str_cat.h"

namespace xla {
namespace llvm_ir {
static const HloInstruction& InstrForConstantBufferAllocation(
    const BufferAllocation& allocation) {
  CHECK(allocation.is_constant());
  HloInstruction* const_instr = nullptr;
  for (const auto& buffer_offset_pair : allocation.assigned_buffers()) {
    const BufferValue* buffer = buffer_offset_pair.first;
    // BufferAssignment may have assigned non-constant instructions to this
    // allocation too so we can't CHECK this condition.  E.g. for
    //
    //   while(init = constant, body = identity, cond = ...)
    //
    // the LogicalBuffer for the kWhile instruction will have the same
    // BufferAllocation as the LogicalBuffer for the (init) constant.
    if (buffer->instruction()->opcode() == HloOpcode::kConstant) {
      CHECK_EQ(const_instr, nullptr)
          << const_instr->ToString() << " " << buffer->ToString();
      const_instr = buffer->instruction();
    }
  }
  CHECK_NE(const_instr, nullptr);
  return *const_instr;
}

std::string SanitizeConstantName(const HloInstruction& instr) {
  CHECK_EQ(instr.opcode(), HloOpcode::kConstant);
  return SanitizeConstantName(instr.name());
}

std::string SanitizeConstantName(absl::string_view name) {
  std::string instr_name(name);
  // Replace characters which would require the identifier to be quoted and
  // would therefore crash the LLVM PTX backend.
  std::replace_if(
      instr_name.begin(), instr_name.end(),
      [](char c) { return c == '.' || c == '-' || c == ';'; }, '_');
  return instr_name;
}

std::string ConstantHloToGlobalName(const HloInstruction& instr) {
  return ConstantNameToGlobalName(instr.name());
}

std::string ConstantNameToGlobalName(absl::string_view name) {
  // Check that names are sanitized and stored in the HLO instructions
  // before constant buffer allocation.
  DCHECK_EQ(name, SanitizeConstantName(name));
  return absl::StrCat("buffer_for_", name);
}

std::string ConstantBufferAllocationToGlobalName(
    const BufferAllocation& allocation) {
  return ConstantNameToGlobalName(
      SanitizeConstantName(InstrForConstantBufferAllocation(allocation)));
}

const Literal& LiteralForConstantAllocation(
    const BufferAllocation& allocation) {
  return InstrForConstantBufferAllocation(allocation).literal();
}
}  // namespace llvm_ir
}  // namespace xla

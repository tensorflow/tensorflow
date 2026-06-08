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

#include "xla/service/io_param.h"

#include <cstdint>
#include <functional>
#include <ostream>
#include <string>
#include <tuple>
#include <vector>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "xla/hlo/analysis/hlo_alias_analysis.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/hlo_buffer.h"
#include "xla/service/hlo_value.h"
#include "xla/shape.h"
#include "xla/shape_util.h"

namespace xla {
namespace {

std::vector<IOParam> GetNonTrivialUsesImpl(const IOParam& position) {
  VLOG(2) << "Enter GetNonTrivialUsesImpl(" << position.ToString() << ")";
  CHECK(position.IsOutput());
  HloInstruction* call_site = position.GetCallSite();

  std::vector<IOParam> non_trivial_position_uses;
  for (HloInstruction* call_site_user : call_site->users()) {
    if (call_site_user->opcode() == HloOpcode::kGetTupleElement) {
      const ShapeIndex& index = position.AsOutput().index;
      CHECK(!index.empty());
      if (index[0] == call_site_user->tuple_index()) {
        ShapeIndex new_index = index;
        new_index.pop_front();
        std::vector<IOParam> users = GetNonTrivialUsesImpl(
            IOParam(HloPosition{call_site_user, new_index}));
        non_trivial_position_uses.insert(non_trivial_position_uses.end(),
                                         users.begin(), users.end());
      } else {
        // The GTE is not for `position`.
        continue;
      }
    } else if (call_site_user->opcode() == HloOpcode::kTuple) {
      for (int64_t operand_num : call_site_user->operand_indices(call_site)) {
        ShapeIndex new_index = position.AsOutput().index;
        new_index.push_front(operand_num);
        std::vector<IOParam> users = GetNonTrivialUsesImpl(
            IOParam(HloPosition{call_site_user, new_index}));
        non_trivial_position_uses.insert(non_trivial_position_uses.end(),
                                         users.begin(), users.end());
      }
    } else if (call_site_user->opcode() == HloOpcode::kBitcast) {
      std::vector<IOParam> users = GetNonTrivialUsesImpl(
          IOParam(HloPosition{call_site_user, position.AsOutput().index}));
      non_trivial_position_uses.insert(non_trivial_position_uses.end(),
                                       users.begin(), users.end());
    } else {
      for (int64_t operand_num : call_site_user->operand_indices(call_site)) {
        non_trivial_position_uses.push_back(IOParam(
            HloUse(call_site_user, operand_num, position.AsOutput().index)));
      }
    }
  }

  VLOG(2) << "GetNonTrivialUsesImpl(" << position.ToString() << "): "
          << absl::StrJoin(non_trivial_position_uses, ", ",
                           [](std::string* out, const IOParam& param) {
                             absl::StrAppend(out, param.ToString());
                           });
  return non_trivial_position_uses;
}

IOParam GetNonTrivialSourcePositionImpl(const IOParam& use) {
  VLOG(2) << "Enter GetNonTrivialSourcePositionImpl(" << use.ToString() << ")";
  CHECK(use.IsInput());
  IOParam source(use.GetSourcePosition());
  VLOG(2) << "  Next step source is " << source.ToString();
  HloInstruction* source_call_site = source.GetCallSite();
  if (source_call_site->opcode() == HloOpcode::kGetTupleElement) {
    ShapeIndex new_index = source.AsOutput().index;
    new_index.push_front(source_call_site->tuple_index());
    return GetNonTrivialSourcePositionImpl(
        IOParam(HloUse(source_call_site, /*operand_number=*/0, new_index)));
  }
  if (source_call_site->opcode() == HloOpcode::kBitcast) {
    return GetNonTrivialSourcePositionImpl(IOParam(HloUse(
        source_call_site, /*operand_number=*/0, source.AsOutput().index)));
  }
  if (source_call_site->opcode() == HloOpcode::kTuple) {
    CHECK(!source.AsOutput().index.empty());
    int64_t tuple_operand_number = source.AsOutput().index.front();
    ShapeIndex new_index = source.AsOutput().index;
    new_index.pop_front();
    return GetNonTrivialSourcePositionImpl(
        IOParam(HloUse(source_call_site,
                       /*operand_number=*/tuple_operand_number, new_index)));
  }

  VLOG(2) << "GetNonTrivialSourcePositionImpl(" << use.ToString()
          << ") returns " << source.ToString();
  return source;
}

}  // namespace

bool IOParam::operator<(const IOParam& other) const {
  return ToTuple() < other.ToTuple();
}

std::ostream& operator<<(std::ostream& out, const IOParam& param) {
  out << param.ToString();
  return out;
}

const HloUse& IOParam::AsInput() const {
  CHECK(IsInput());
  return std::get<HloUse>(value_);
}

const HloPosition& IOParam::AsOutput() const {
  CHECK(IsOutput());
  return std::get<HloPosition>(value_);
}

HloInstruction* IOParam::GetCallSite() const {
  if (IsInput()) {
    return AsInput().instruction;
  }
  return AsOutput().instruction;
}

const HloValue& IOParam::GetHloValue(
    const HloAliasAnalysis& alias_analysis) const {
  HloPosition source_position = GetSourcePosition();
  return alias_analysis.dataflow_analysis().GetUniqueValueAt(
      source_position.instruction, source_position.index);
}

const HloBuffer& IOParam::GetHloBuffer(
    const HloAliasAnalysis& alias_analysis) const {
  HloPosition source_position = GetSourcePosition();
  return alias_analysis.GetUniqueBufferAt(source_position.instruction,
                                          source_position.index);
}

HloPosition IOParam::GetSourcePosition() const {
  if (IsInput()) {
    return HloPosition{
        AsInput().instruction->mutable_operand(AsInput().operand_number),
        AsInput().operand_index};
  }
  return AsOutput();
}

const Shape& IOParam::GetShape() const {
  if (IsInput()) {
    const HloInstruction* operand =
        AsInput().instruction->operand(AsInput().operand_number);
    return ShapeUtil::GetSubshape(operand->shape(), AsInput().operand_index);
  }
  return ShapeUtil::GetSubshape(AsOutput().instruction->shape(),
                                AsOutput().index);
}

absl::StatusOr<std::vector<IOParam>> IOParam::GetNonTrivialUses() const {
  if (!IsOutput()) {
    return absl::InvalidArgumentError("GetNonTrivialUses called on an input");
  }
  if (GetShape().IsTuple()) {
    return absl::InvalidArgumentError(
        "GetNonTrivialUses called on a tuple shape");
  }

  std::vector<IOParam> users = GetNonTrivialUsesImpl(*this);
  VLOG(1) << "GetNonTrivialUsers(" << ToString() << "): "
          << absl::StrJoin(users, ", ",
                           [](std::string* out, const IOParam& param) {
                             absl::StrAppend(out, param.ToString());
                           });
  return users;
}

absl::StatusOr<IOParam> IOParam::GetNonTrivialSourcePosition() const {
  if (!IsInput()) {
    return absl::InvalidArgumentError(
        "GetNonTrivialSourcePosition called on an output");
  }
  if (GetShape().IsTuple()) {
    return absl::InvalidArgumentError(
        "GetNonTrivialSourcePosition called on a tuple shape");
  }

  IOParam source = GetNonTrivialSourcePositionImpl(*this);
  VLOG(1) << "GetNonTrivialSource(" << ToString() << "): " << source.ToString();
  return source;
}

std::string IOParam::ToString() const {
  if (IsInput()) {
    return absl::StrCat("Input(", AsInput().ToString(), ")");
  }
  return absl::StrCat("Output(", AsOutput().ToString(), ")");
}

IOParam::Tuple IOParam::ToTuple() const {
  int64_t instruction_id = -1;
  int operand_number = -1;
  const ShapeIndex* shape_index = nullptr;
  if (IsOutput()) {
    instruction_id = AsOutput().instruction->unique_id();
    shape_index = &AsOutput().index;
  } else {
    instruction_id = AsInput().instruction->unique_id();
    operand_number = AsInput().operand_number;
    shape_index = &AsInput().operand_index;
  }
  return std::make_tuple(instruction_id, IsOutput(), operand_number,
                         std::cref(*shape_index));
}
}  // namespace xla

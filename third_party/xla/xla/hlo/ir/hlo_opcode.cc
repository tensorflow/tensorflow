/* Copyright 2017 The OpenXLA Authors.

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

#include "xla/hlo/ir/hlo_opcode.h"

#include <cstdint>
#include <optional>
#include <ostream>
#include <string>

#include "absl/container/flat_hash_map.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/util.h"

namespace xla {

absl::string_view HloOpcodeString(HloOpcode opcode) {
  switch (opcode) {
#define CASE_OPCODE_STRING(enum_name, opcode_name, ...) \
  case HloOpcode::enum_name:                            \
    return opcode_name;
    HLO_OPCODE_LIST(CASE_OPCODE_STRING)
#undef CASE_OPCODE_STRING
  }
}

absl::StatusOr<HloOpcode> StringToHloOpcode(absl::string_view opcode_name) {
  static auto* const opcode_map =
      new absl::flat_hash_map<std::string, HloOpcode>({
#define STRING_TO_OPCODE_ENTRY(enum_name, opcode_name, ...) \
  {opcode_name, HloOpcode::enum_name},
          HLO_OPCODE_LIST(STRING_TO_OPCODE_ENTRY)
#undef STRING_TO_OPCODE_ENTRY
      });
  auto it = opcode_map->find(opcode_name);
  if (it == opcode_map->end()) {
    return InvalidArgument("Unknown opcode: %s", opcode_name);
  }
  return it->second;
}

std::optional<int8_t> HloOpcodeArity(HloOpcode opcode) {
  switch (opcode) {
#define CASE_ARITY(enum_name, opcode_name, arity, ...)  \
  case HloOpcode::enum_name:                            \
    return arity == kHloOpcodeIsVariadic ? std::nullopt \
                                         : std::make_optional(arity);
    HLO_OPCODE_LIST(CASE_ARITY)
#undef CASE_ARITY
  }
}

std::string CallContextToString(CallContext context) {
  switch (context) {
    case CallContext::kNone:
      return "kNone";
    case CallContext::kControlFlow:
      return "kControlFlow";
    case CallContext::kEmbedded:
      return "kEmbedded";
    case CallContext::kBoth:
      return "kBoth";
  }
}

std::ostream& operator<<(std::ostream& out, const CallContext& context) {
  out << CallContextToString(context);
  return out;
}

CallContext GetInstructionCallContext(HloOpcode opcode) {
  switch (opcode) {
    case HloOpcode::kCall:
    case HloOpcode::kConditional:
    case HloOpcode::kWhile:
    case HloOpcode::kAsyncStart:
    case HloOpcode::kAsyncUpdate:
    case HloOpcode::kAsyncDone:
      return CallContext::kControlFlow;
    case HloOpcode::kAllReduce:
    case HloOpcode::kReduceScatter:
    case HloOpcode::kAllReduceStart:
    case HloOpcode::kMap:
    case HloOpcode::kReduce:
    case HloOpcode::kReduceWindow:
    case HloOpcode::kScatter:
    case HloOpcode::kSelectAndScatter:
    case HloOpcode::kSort:
    case HloOpcode::kFusion:
    case HloOpcode::kCustomCall:
      return CallContext::kEmbedded;
    default:
      return CallContext::kNone;
  }
}
}  // namespace xla

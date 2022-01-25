/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/hlo_opcode.h"

#include "absl/container/flat_hash_map.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/util.h"

namespace xla {

std::string HloOpcodeString(HloOpcode opcode) {
  switch (opcode) {
#define CASE_OPCODE_STRING(enum_name, opcode_name, ...) \
  case HloOpcode::enum_name:                            \
    return opcode_name;
    HLO_OPCODE_LIST(CASE_OPCODE_STRING)
#undef CASE_OPCODE_STRING
  }
}

StatusOr<HloOpcode> StringToHloOpcode(const std::string& opcode_name) {
  static auto* opcode_map = new absl::flat_hash_map<std::string, HloOpcode>({
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

bool HloOpcodeIsComparison(HloOpcode opcode) {
  return opcode == HloOpcode::kCompare;
}

bool HloOpcodeIsVariadic(HloOpcode opcode) {
  switch (opcode) {
#define CASE_IS_VARIADIC(enum_name, opcode_name, arity, ...) \
  case HloOpcode::enum_name:                                 \
    return arity == kHloOpcodeIsVariadic;
    HLO_OPCODE_LIST(CASE_IS_VARIADIC)
#undef CASE_IS_VARIADIC
  }
}

absl::optional<int> HloOpcodeArity(HloOpcode opcode) {
  switch (opcode) {
#define CASE_ARITY(enum_name, opcode_name, arity, ...)   \
  case HloOpcode::enum_name:                             \
    return arity == kHloOpcodeIsVariadic ? absl::nullopt \
                                         : absl::make_optional(arity);
    HLO_OPCODE_LIST(CASE_ARITY)
#undef CASE_ARITY
  }
}

}  // namespace xla

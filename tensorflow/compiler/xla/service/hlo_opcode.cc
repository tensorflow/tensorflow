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
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/lib/gtl/flatmap.h"

namespace xla {

string HloOpcodeString(HloOpcode opcode) {
  switch (opcode) {
#define CASE_OPCODE_STRING(enum_name, opcode_name, ...) \
  case HloOpcode::enum_name:                            \
    return opcode_name;
    HLO_OPCODE_LIST(CASE_OPCODE_STRING)
#undef CASE_OPCODE_STRING
  }
}

StatusOr<HloOpcode> StringToHloOpcode(const string& opcode_name) {
  static auto* opcode_map = new tensorflow::gtl::FlatMap<string, HloOpcode>({
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

#define CHECK_DEFAULT(property_name, opcode_name) false
#define CHECK_PROPERTY(property_name, opcode_name, value) \
  (value & property_name)
#define RESOLVE(_1, _2, target, ...) target
#define HAS_PROPERTY(property, ...) \
  RESOLVE(__VA_ARGS__, CHECK_PROPERTY, CHECK_DEFAULT)(property, __VA_ARGS__)

bool HloOpcodeIsComparison(HloOpcode opcode) {
  switch (opcode) {
#define CASE_IS_COMPARISON(enum_name, ...) \
  case HloOpcode::enum_name:               \
    return HAS_PROPERTY(kHloOpcodeIsComparison, __VA_ARGS__);
    HLO_OPCODE_LIST(CASE_IS_COMPARISON)
#undef CASE_IS_COMPARISON
  }
}

bool HloOpcodeIsVariadic(HloOpcode opcode) {
  switch (opcode) {
#define CASE_IS_VARIADIC(enum_name, ...) \
  case HloOpcode::enum_name:             \
    return HAS_PROPERTY(kHloOpcodeIsVariadic, __VA_ARGS__);
    HLO_OPCODE_LIST(CASE_IS_VARIADIC)
#undef CASE_IS_VARIADIC
  }
}

#undef HAS_PROPERTY
#undef RESOLVE
#undef CHECK_DEFAULT
#undef CHECK_PROPERTY

}  // namespace xla

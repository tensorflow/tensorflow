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

// Downcasting functions for HLO instructions similar to LLVM's.
// Offers nullptr tolerant and dynamic versions.
// All versions rely on HloInstruction::ClassOf instead of
// dynamic_cast's runtime type checks for faster performance.

#ifndef XLA_HLO_IR_HLO_CASTING_UTILS_H_
#define XLA_HLO_IR_HLO_CASTING_UTILS_H_

#include <string>

#include "absl/base/config.h"
#include "absl/log/check.h"
#include "absl/strings/str_format.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/tsl/platform/logging.h"

namespace xla {

namespace cast_internal {

template <typename T>
inline const char* TypeName(T* input = nullptr) {
#ifdef ABSL_INTERNAL_HAS_RTTI
  return (input != nullptr) ? typeid(*input).name() : typeid(T).name();
#else
  return "unknown (no RTTI)";
#endif
}

template <typename T>
inline std::string WrongCastError(const HloInstruction* instr) {
  return absl::StrFormat(
      "HloInstruction '%s' is of type '%s' and cannot be downcasted to '%s.'",
      instr->name(), TypeName(instr), TypeName<T>());
}
}  // namespace cast_internal

// Downcasts a const HloInstruction pointer. Dies if argument is nullptr or
// TargetClass::ClassOf() does not match. Similar to LLVM's cast.
template <typename T>
const T* Cast(const HloInstruction* instr) {
  CHECK(instr != nullptr);
  CHECK(T::ClassOf(instr)) << cast_internal::WrongCastError<T>(instr);
  return tsl::down_cast<const T*>(instr);
}

// Downcasts a non-const HloInstruction pointer. Dies if argument is nullptr or
// TargetClass::ClassOf() does not match. Similar to LLVM's cast.
template <typename T>
T* Cast(HloInstruction* instr) {
  return const_cast<T*>(Cast<T>(const_cast<const HloInstruction*>(instr)));
}

// Downcasts a const HloInstruction pointer or returns nullptr if
// TargetClass::ClassOf() does not match. Dies if argument is nullptr. Similar
// to LLVM's dyn_cast.
template <typename T>
const T* DynCast(const HloInstruction* i) {
  CHECK(i != nullptr);
  return !T::ClassOf(i) ? nullptr : tsl::down_cast<const T*>(i);
}

// Downcasts a non-const HloInstruction pointer or returns nullptr if
// TargetClass::ClassOf() does not match. Dies if argument is nullptr. Similar
// to LLVM's dyn_cast.
template <typename T>
T* DynCast(HloInstruction* i) {
  return const_cast<T*>(DynCast<T>(const_cast<const HloInstruction*>(i)));
}

}  // namespace xla

#endif  // XLA_HLO_IR_HLO_CASTING_UTILS_H_

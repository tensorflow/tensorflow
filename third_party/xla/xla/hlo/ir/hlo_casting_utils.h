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

#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/tsl/platform/logging.h"

namespace xla {

// Downcasts a const HloInstruction pointer. Dies if argument is nullptr or
// TargetClass::ClassOf() does not match. Similar to LLVM's cast.
template <typename T>
const T* Cast(const HloInstruction* instr) {
  CHECK(instr != nullptr);
  CHECK(T::ClassOf(instr));
  return tsl::down_cast<const T*>(instr);
}

// Downcasts a non-const HloInstruction pointer. Dies if argument is nullptr or
// TargetClass::ClassOf() does not match. Similar to LLVM's cast.
template <typename T>
T* Cast(HloInstruction* instr) {
  CHECK(instr != nullptr);
  CHECK(T::ClassOf(instr));
  return tsl::down_cast<T*>(instr);
}

// Downcasts a const HloInstruction pointer or returns nullptr if argument is
// nullptr. Dies if TargetClass::ClassOf() does not match.
template <typename T>
const T* CastOrNull(const HloInstruction* i) {
  if (i == nullptr) {
    return nullptr;
  }
  CHECK(T::ClassOf(i));
  return tsl::down_cast<const T*>(i);
}

// Downcasts a const HloInstruction pointer or returns nullptr if argument is
// nullptr. Dies if TargetClass::ClassOf() does not match.
template <typename T>
T* CastOrNull(HloInstruction* i) {
  if (i == nullptr) {
    return nullptr;
  }
  CHECK(T::ClassOf(i));
  return tsl::down_cast<T*>(i);
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
  CHECK(i != nullptr);
  return !T::ClassOf(i) ? nullptr : tsl::down_cast<T*>(i);
}

// Downcasts a const HloInstruction pointer. Return nullptr if argument is
// nullptr orTargetClass::ClassOf() does not match. Similar to LLVM's
// dyn_cast_or_null.
template <typename T>
const T* DynCastOrNull(const HloInstruction* instruction) {
  if (instruction == nullptr || !T::ClassOf(instruction)) {
    return nullptr;
  }
  return tsl::down_cast<const T*>(instruction);
}

// Downcasts a non-const HloInstruction pointer. Return nullptr if argument is
// nullptr orTargetClass::ClassOf() does not match. Similar to LLVM's
// dyn_cast_or_null.
template <typename T>
T* DynCastOrNull(HloInstruction* instruction) {
  if (instruction == nullptr || !T::ClassOf(instruction)) {
    return nullptr;
  }
  return tsl::down_cast<T*>(instruction);
}

}  // namespace xla

#endif  // XLA_HLO_IR_HLO_CASTING_UTILS_H_

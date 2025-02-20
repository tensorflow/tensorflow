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
// All versions rely on HloInstruction::ClassOf instead of dynamic_cast's
// runtime type checks for faster performance.
//
// In debug mode, we use dynamic_cast to double-check whether the downcast is
// legal (we die if it's not). In normal mode, we do the efficient static_cast
// instead. Thus, it's important to test in debug mode to make sure the cast
// is legal!

#ifndef XLA_HLO_IR_HLO_CASTING_UTILS_H_
#define XLA_HLO_IR_HLO_CASTING_UTILS_H_

#include <type_traits>

#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/tsl/platform/logging.h"

namespace xla {

template <class T>
using EnableIfDerivedFromHlo =
    typename std::enable_if<std::is_base_of<HloInstruction, T>::value>::type;

// Downcasts a const HloInstruction pointer. Dies if argument is nullptr or
// TargetClass::ClassOf() does not match.
//
// Similar to LLVM's cast.
template <class T, EnableIfDerivedFromHlo<T>* = nullptr>
const T* Cast(const HloInstruction* instruction) {
  CHECK(instruction != nullptr);
  CHECK(T::ClassOf(instruction))
      << "Invalid HloInstruction casting. Destination type: "
      << typeid(T).name() << ". Instruction: " << instruction->name();
  const T* casted = static_cast<const T*>(instruction);
#ifndef NDEBUG
  const T* dynamic_casted = dynamic_cast<const T*>(instruction);
  CHECK(dynamic_casted != nullptr)
      << "Invalid HloInstruction casting. Destination type: "
      << typeid(T).name() << ". Instruction: " << instruction->name();
#endif
  return casted;
}

// Downcasts a non-const HloInstruction pointer. Dies if argument is nullptr or
// TargetClass::ClassOf() does not match.
//
// Similar to LLVM's cast.
template <class T, EnableIfDerivedFromHlo<T>* = nullptr>
T* Cast(HloInstruction* instruction) {
  return const_cast<T*>(
      Cast<T>(const_cast<const HloInstruction*>(instruction)));
}

// Downcasts a const HloInstruction pointer or returns nullptr if argument is
// nullptr. Dies if TargetClass::ClassOf() does not match.
//
// Similar to LLVM's cast_or_null.
template <class T, EnableIfDerivedFromHlo<T>* = nullptr>
const T* CastOrNull(const HloInstruction* instruction) {
  return instruction != nullptr ? Cast<T>(instruction) : nullptr;
}

// Downcasts a const HloInstruction pointer or returns nullptr if argument is
// nullptr. Dies if TargetClass::ClassOf() does not match.
//
// Similar to LLVM's cast_or_null.
template <class T, EnableIfDerivedFromHlo<T>* = nullptr>
T* CastOrNull(HloInstruction* instruction) {
  return const_cast<T*>(
      CastOrNull<T>(const_cast<const HloInstruction*>(instruction)));
}

// Downcasts a const HloInstruction pointer or returns nullptr if
// TargetClass::ClassOf() does not match. Dies if argument is nullptr.
//
// Similar to LLVM's dyn_cast.
template <class T, EnableIfDerivedFromHlo<T>* = nullptr>
const T* DynCast(const HloInstruction* instruction) {
  CHECK(instruction != nullptr);
  const T* casted =
      T::ClassOf(instruction) ? static_cast<const T*>(instruction) : nullptr;
#ifndef NDEBUG
  CHECK_EQ(casted, dynamic_cast<const T*>(instruction));
#endif
  return casted;
}

// Downcasts a non-const HloInstruction pointer or returns nullptr if
// TargetClass::ClassOf() does not match. Dies if argument is nullptr.
//
// Similar to LLVM's dyn_cast.
template <class T, EnableIfDerivedFromHlo<T>* = nullptr>
T* DynCast(HloInstruction* instruction) {
  return const_cast<T*>(
      DynCast<T>(const_cast<const HloInstruction*>(instruction)));
}

// Downcasts a const HloInstruction pointer. Return nullptr if argument is
// nullptr orTargetClass::ClassOf() does not match.
//
// Similar to LLVM's dyn_cast_or_null.
template <class T, EnableIfDerivedFromHlo<T>* = nullptr>
const T* DynCastOrNull(const HloInstruction* instruction) {
  return instruction != nullptr ? DynCast<T>(instruction) : nullptr;
}

// Downcasts a non-const HloInstruction pointer. Return nullptr if argument is
// nullptr orTargetClass::ClassOf() does not match.
//
// Similar to LLVM's dyn_cast_or_null.
template <class T, EnableIfDerivedFromHlo<T>* = nullptr>
T* DynCastOrNull(HloInstruction* instruction) {
  return const_cast<T*>(
      DynCastOrNull<T>(const_cast<const HloInstruction*>(instruction)));
}

}  // namespace xla

#endif  // XLA_HLO_IR_HLO_CASTING_UTILS_H_

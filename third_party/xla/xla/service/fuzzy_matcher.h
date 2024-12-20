/* Copyright 2024 The OpenXLA Authors.

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

#ifndef XLA_SERVICE_FUZZY_MATCHER_H_
#define XLA_SERVICE_FUZZY_MATCHER_H_

#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/pattern_matcher.h"

namespace xla {

// Fuzzy matchers for HLOs.
namespace fm {

// Expect that Ignorable is a function callable with Pattern
template <typename Pattern, typename Ignorable>
concept IsCallableWith =
    requires(Pattern pattern, Ignorable ignorable) { ignorable(pattern); };

// Return a pattern that matches either the pattern given or the pattern
// composed with the ignorable wrapper
template <typename Pattern, typename Ignorable>
auto OptUnary(Pattern pattern, Ignorable ignorable)
  requires IsCallableWith<Pattern, Ignorable>
{
  auto shared = match::SharedSubpattern(pattern);

  return match::AnyOf<HloInstruction>(ignorable(shared), shared);
}

#define XLA_FUZZY_NULLOP_PATTERN(NAME)                                       \
  template <typename Ignorable>                                              \
  inline auto NAME(Ignorable ignorable) {                                    \
    return OptUnary(match::Op().WithOpcode(HloOpcode::k##NAME), ignorable);  \
  }                                                                          \
                                                                             \
  template <typename HloInstructionType, typename Ignorable>                 \
  inline auto NAME(Ignorable ignorable, HloInstructionType** matched_inst) { \
    return OptUnary(match::Op(matched_inst).WithOpcode(HloOpcode::k##NAME),  \
                    ignorable);                                              \
  }
XLA_FUZZY_NULLOP_PATTERN(Constant)
XLA_FUZZY_NULLOP_PATTERN(Parameter)
#undef XLA_FUZZY_NULLOP_PATTERN

#define XLA_FUZZY_UNOP_PATTERN(NAME)                                         \
  template <typename HloInstructionType, typename Ignorable>                 \
  inline auto NAME(Ignorable ignorable, HloInstructionType** matched_inst) { \
    return OptUnary(match::Op(matched_inst).WithOpcode(HloOpcode::k##NAME),  \
                    ignorable);                                              \
  }                                                                          \
                                                                             \
  template <typename Arg, typename Ignorable>                                \
  inline auto NAME(Ignorable ignorable, Arg&& arg) {                         \
    return OptUnary(match::Op()                                              \
                        .WithOpcode(HloOpcode::k##NAME)                      \
                        .WithOperand(0, std::forward<Arg>(arg)),             \
                    ignorable);                                              \
  }                                                                          \
                                                                             \
  template <typename HloInstructionType, typename Arg, typename Ignorable>   \
  inline auto NAME(Ignorable ignorable, HloInstructionType** matched_inst,   \
                   Arg&& arg) {                                              \
    return OptUnary(match::Op(matched_inst)                                  \
                        .WithOpcode(HloOpcode::k##NAME)                      \
                        .WithOperand(0, std::forward<Arg>(arg)),             \
                    ignorable);                                              \
  }
XLA_FUZZY_UNOP_PATTERN(Tanh)
XLA_FUZZY_UNOP_PATTERN(Exp)
XLA_FUZZY_UNOP_PATTERN(Broadcast)
#undef XLA_FUZZY_UNOP_PATTERN

#define XLA_FUZZY_BINOP_PATTERN(NAME)                                      \
  template <typename HloInstructionType, typename Lhs, typename Rhs,       \
            typename Ignorable>                                            \
  inline auto NAME(Ignorable ignorable, HloInstructionType** matched_inst, \
                   Lhs&& lhs, Rhs&& rhs) {                                 \
    return OptUnary(match::Op(matched_inst)                                \
                        .WithOpcode(HloOpcode::k##NAME)                    \
                        .WithOperand(0, std::forward<Lhs>(lhs))            \
                        .WithOperand(1, std::forward<Rhs>(rhs)),           \
                    ignorable);                                            \
  }                                                                        \
  template <typename Lhs, typename Rhs, typename Ignorable>                \
  inline auto NAME(Ignorable ignorable, Lhs&& lhs, Rhs&& rhs) {            \
    return OptUnary(match::Op()                                            \
                        .WithOpcode(HloOpcode::k##NAME)                    \
                        .WithOperand(0, std::forward<Lhs>(lhs))            \
                        .WithOperand(1, std::forward<Rhs>(rhs)),           \
                    ignorable);                                            \
  }
XLA_FUZZY_BINOP_PATTERN(Dot)
XLA_FUZZY_BINOP_PATTERN(Divide)
XLA_FUZZY_BINOP_PATTERN(Subtract)
XLA_FUZZY_BINOP_PATTERN(Multiply)
// Currently we only use binary matcher for reduce.
XLA_FUZZY_BINOP_PATTERN(Reduce)
#undef XLA_FUZZY_BINOP_PATTERN

#define XLA_FUZZY_TERNOP_PATTERN(NAME)                                       \
  template <typename Arg0, typename Arg1, typename Arg2, typename Ignorable> \
  inline auto NAME(Ignorable ignorable, Arg0&& arg0, Arg1&& arg1,            \
                   Arg2&& arg2) {                                            \
    return OptUnary(match::Op()                                              \
                        .WithOpcode(HloOpcode::k##NAME)                      \
                        .WithOperand(0, std::forward<Arg0>(arg0))            \
                        .WithOperand(1, std::forward<Arg1>(arg1))            \
                        .WithOperand(2, std::forward<Arg2>(arg2)),           \
                    ignorable);                                              \
  }                                                                          \
                                                                             \
  template <typename HloInstructionType, typename Arg0, typename Arg1,       \
            typename Arg2, typename Ignorable>                               \
  inline auto NAME(Ignorable ignorable, HloInstructionType** matched_inst,   \
                   Arg0&& arg0, Arg1&& arg1, Arg2&& arg2) {                  \
    return OptUnary(match::Op(matched_inst)                                  \
                        .WithOpcode(HloOpcode::k##NAME)                      \
                        .WithOperand(0, std::forward<Arg0>(arg0))            \
                        .WithOperand(1, std::forward<Arg1>(arg1))            \
                        .WithOperand(2, std::forward<Arg2>(arg2)),           \
                    ignorable);                                              \
  }
XLA_FUZZY_TERNOP_PATTERN(Select);
#undef XLA_FUZZY_TERNOP_PATTERN

}  // namespace fm

}  // namespace xla

#endif  // XLA_SERVICE_FUZZY_MATCHER_H_

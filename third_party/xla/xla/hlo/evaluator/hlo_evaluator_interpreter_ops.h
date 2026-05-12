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

#ifndef XLA_HLO_EVALUATOR_HLO_EVALUATOR_INTERPRETER_OPS_H_
#define XLA_HLO_EVALUATOR_HLO_EVALUATOR_INTERPRETER_OPS_H_

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <type_traits>

#include "xla/comparison_util.h"
#include "xla/hlo/evaluator/hlo_evaluator_interpreter.h"

namespace xla {

// LinearizedInterpreter supports a subset of HLO opcodes found in common apply
// functions, which will be linearized and executed by the interpreter in
// batches.

class LinearizedInterpreter::Ops {
 public:
  struct Add {
    template <typename ResT, typename LhsT, typename RhsT>
    static void Execute(const Step* step, void* scratchpad_base);
    static Step::ExecuteFn GetExecuteFn(PrimitiveType res_type,
                                        PrimitiveType lhs_type,
                                        PrimitiveType rhs_type);
  };

  struct Maximum {
    template <typename T>
    static void Execute(const Step* step, void* scratchpad_base);
    static Step::ExecuteFn GetExecuteFn(PrimitiveType res_type,
                                        PrimitiveType lhs_type,
                                        PrimitiveType rhs_type);
  };

  struct Compare {
    template <typename T, ComparisonDirection Direction>
    static void Execute(const Step* step, void* scratchpad_base);
    static Step::ExecuteFn GetExecuteFn(ComparisonDirection direction,
                                        PrimitiveType operand_type);
  };

  struct Or {
    static void Execute(const Step* step, void* scratchpad_base);
    static Step::ExecuteFn GetExecuteFn(PrimitiveType type);
  };

  struct And {
    static void Execute(const Step* step, void* scratchpad_base);
    static Step::ExecuteFn GetExecuteFn(PrimitiveType type);
  };

  struct Select {
    template <typename T>
    static void Execute(const Step* step, void* scratchpad_base);
    static Step::ExecuteFn GetExecuteFn(PrimitiveType type);
  };
};

// Implementations
template <typename ResT, typename LhsT, typename RhsT>
void LinearizedInterpreter::Ops::Add::Execute(const Step* step,
                                              void* scratchpad_base) {
  ResT* result = Scratchpad::GetPointerFromBase<ResT>(scratchpad_base,
                                                      step->result_offset);
  const LhsT* lhs = Scratchpad::GetPointerFromBase<LhsT>(
      scratchpad_base, step->operand_offsets[0]);
  const RhsT* rhs = Scratchpad::GetPointerFromBase<RhsT>(
      scratchpad_base, step->operand_offsets[1]);

  for (size_t i = 0; i < step->element_count; ++i) {
    result[i] = static_cast<ResT>(lhs[i]) + static_cast<ResT>(rhs[i]);
  }
}

template <typename T>
void LinearizedInterpreter::Ops::Maximum::Execute(const Step* step,
                                                  void* scratchpad_base) {
  T* result =
      Scratchpad::GetPointerFromBase<T>(scratchpad_base, step->result_offset);
  const T* lhs = Scratchpad::GetPointerFromBase<T>(scratchpad_base,
                                                   step->operand_offsets[0]);
  const T* rhs = Scratchpad::GetPointerFromBase<T>(scratchpad_base,
                                                   step->operand_offsets[1]);

  for (size_t i = 0; i < step->element_count; ++i) {
    if constexpr (std::is_floating_point_v<T>) {
      if (std::isnan(lhs[i])) {
        result[i] = lhs[i];
      } else if (std::isnan(rhs[i])) {
        result[i] = rhs[i];
      } else {
        result[i] = std::max(lhs[i], rhs[i]);
      }
    } else {
      result[i] = std::max(lhs[i], rhs[i]);
    }
  }
}

template <typename T, ComparisonDirection Direction>
void LinearizedInterpreter::Ops::Compare::Execute(const Step* step,
                                                  void* scratchpad_base) {
  static_assert(Direction == ComparisonDirection::kGt ||
                    Direction == ComparisonDirection::kNe ||
                    Direction == ComparisonDirection::kEq ||
                    Direction == ComparisonDirection::kLt ||
                    Direction == ComparisonDirection::kGe ||
                    Direction == ComparisonDirection::kLe,
                "Unsupported compare direction");

  bool* result = Scratchpad::GetPointerFromBase<bool>(scratchpad_base,
                                                      step->result_offset);
  const T* lhs = Scratchpad::GetPointerFromBase<T>(scratchpad_base,
                                                   step->operand_offsets[0]);
  const T* rhs = Scratchpad::GetPointerFromBase<T>(scratchpad_base,
                                                   step->operand_offsets[1]);

  for (size_t i = 0; i < step->element_count; ++i) {
    if constexpr (Direction == ComparisonDirection::kGt) {
      result[i] = lhs[i] > rhs[i];
    } else if constexpr (Direction == ComparisonDirection::kNe) {
      result[i] = lhs[i] != rhs[i];
    } else if constexpr (Direction == ComparisonDirection::kEq) {
      result[i] = lhs[i] == rhs[i];
    } else if constexpr (Direction == ComparisonDirection::kLt) {
      result[i] = lhs[i] < rhs[i];
    } else if constexpr (Direction == ComparisonDirection::kGe) {
      result[i] = lhs[i] >= rhs[i];
    } else if constexpr (Direction == ComparisonDirection::kLe) {
      result[i] = lhs[i] <= rhs[i];
    }
  }
}

inline void LinearizedInterpreter::Ops::Or::Execute(const Step* step,
                                                    void* scratchpad_base) {
  bool* result = Scratchpad::GetPointerFromBase<bool>(scratchpad_base,
                                                      step->result_offset);
  const bool* lhs = Scratchpad::GetPointerFromBase<bool>(
      scratchpad_base, step->operand_offsets[0]);
  const bool* rhs = Scratchpad::GetPointerFromBase<bool>(
      scratchpad_base, step->operand_offsets[1]);

  for (size_t i = 0; i < step->element_count; ++i) {
    result[i] = lhs[i] || rhs[i];
  }
}

inline LinearizedInterpreter::Step::ExecuteFn
LinearizedInterpreter::Ops::Or::GetExecuteFn(PrimitiveType type) {
  if (type == PRED) {
    return &Execute;
  }
  return nullptr;
}

inline void LinearizedInterpreter::Ops::And::Execute(const Step* step,
                                                     void* scratchpad_base) {
  bool* result = Scratchpad::GetPointerFromBase<bool>(scratchpad_base,
                                                      step->result_offset);
  const bool* lhs = Scratchpad::GetPointerFromBase<bool>(
      scratchpad_base, step->operand_offsets[0]);
  const bool* rhs = Scratchpad::GetPointerFromBase<bool>(
      scratchpad_base, step->operand_offsets[1]);

  for (size_t i = 0; i < step->element_count; ++i) {
    result[i] = lhs[i] && rhs[i];
  }
}

inline LinearizedInterpreter::Step::ExecuteFn
LinearizedInterpreter::Ops::And::GetExecuteFn(PrimitiveType type) {
  if (type == PRED) {
    return &Execute;
  }
  return nullptr;
}

template <typename T>
void LinearizedInterpreter::Ops::Select::Execute(const Step* step,
                                                 void* scratchpad_base) {
  T* result =
      Scratchpad::GetPointerFromBase<T>(scratchpad_base, step->result_offset);
  const bool* cond = Scratchpad::GetPointerFromBase<bool>(
      scratchpad_base, step->operand_offsets[0]);
  const T* lhs = Scratchpad::GetPointerFromBase<T>(scratchpad_base,
                                                   step->operand_offsets[1]);
  const T* rhs = Scratchpad::GetPointerFromBase<T>(scratchpad_base,
                                                   step->operand_offsets[2]);

  for (size_t i = 0; i < step->element_count; ++i) {
    result[i] = cond[i] ? lhs[i] : rhs[i];
  }
}

}  // namespace xla

#endif  // XLA_HLO_EVALUATOR_HLO_EVALUATOR_INTERPRETER_OPS_H_

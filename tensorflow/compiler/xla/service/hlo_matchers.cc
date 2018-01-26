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

#include "tensorflow/compiler/xla/service/hlo_matchers.h"

#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/test.h"

namespace xla {
namespace testing {

bool HloMatcher::MatchAndExplain(
    const HloInstruction* instruction,
    ::testing::MatchResultListener* listener) const {
  // These cases are self-explanatory from the printed value.
  if (!instruction || instruction->opcode() != opcode_) {
    return false;
  }
  // Special case: no operand matchers means don't verify.
  if (operands_.empty()) {
    return true;
  }
  const auto& operands = instruction->operands();
  if (operands.size() != operands_.size()) {
    *listener << "has too "
              << (operands.size() > operands_.size() ? "many" : "few")
              << " operands (got " << operands.size() << ", want "
              << operands_.size() << ")";
    return false;
  }
  for (int index = 0; index < operands.size(); index++) {
    ::testing::StringMatchResultListener inner_listener;
    if (!operands_[index].MatchAndExplain(operands[index], &inner_listener)) {
      if (listener->IsInterested()) {
        *listener << "\noperand " << index << ":\n\t"
                  << operands[index]->ToString()
                  << "\ndoesn't match expected:\n\t";
        operands_[index].DescribeTo(listener->stream());
        string explanation = inner_listener.str();
        if (!explanation.empty()) {
          *listener << ", " << explanation;
        }
      }
      return false;
    }
  }
  return true;
}

void HloMatcher::DescribeTo(::std::ostream* os) const {
  *os << opcode_;
  if (!operands_.empty()) {
    *os << "(";
    for (int i = 0; i < operands_.size(); i++) {
      if (i > 0) {
        *os << ", ";
      }
      operands_[i].DescribeTo(os);
    }
    *os << ")";
  }
}

bool HloParameterMatcher::MatchAndExplain(
    const HloInstruction* instruction,
    ::testing::MatchResultListener* listener) const {
  if (!HloMatcher::MatchAndExplain(instruction, listener)) {
    return false;
  }
  if (instruction->parameter_number() != parameter_number_) {
    *listener << "has wrong parameter number (got "
              << instruction->parameter_number() << ", want "
              << parameter_number_ << ")";
    return false;
  }
  return true;
}

bool HloGetTupleElementMatcher::MatchAndExplain(
    const HloInstruction* instruction,
    ::testing::MatchResultListener* listener) const {
  if (!HloMatcher::MatchAndExplain(instruction, listener)) {
    return false;
  }
  if (instruction->tuple_index() != tuple_index_) {
    *listener << "has wrong tuple index (got " << instruction->tuple_index()
              << ", want " << tuple_index_ << ")";
    return false;
  }
  return true;
}

}  // namespace testing

void PrintTo(const HloInstruction* inst, ::std::ostream* os) {
  *os << (inst ? inst->ToString() : "nullptr");
}

void PrintTo(HloInstruction* inst, ::std::ostream* os) {
  PrintTo(const_cast<const HloInstruction*>(inst), os);
}

}  // namespace xla

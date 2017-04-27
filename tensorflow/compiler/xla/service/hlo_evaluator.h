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

#ifndef THIRD_PARTY_TENSORFLOW_COMPILER_XLA_SERVICE_HLO_EVALUATOR_H_
#define THIRD_PARTY_TENSORFLOW_COMPILER_XLA_SERVICE_HLO_EVALUATOR_H_

#include <memory>

#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"

namespace xla {

// Responsible for evaluating a HLO instruction with constant operands.
class HloEvaluator {
 public:
  // Evaluates a single HLO instruction for constants and return the result as a
  // Literal.
  // Precondition: all operands of the instruction are constants, instruction is
  // valid with corresponding number of operands for the given operator.
  // TODO(b/35950897): implement more ops other than element-wise ops.
  static StatusOr<std::unique_ptr<Literal>> EvaluateOpForLiteral(
      HloInstruction* instruction);
};

}  // namespace xla

#endif  // THIRD_PARTY_TENSORFLOW_COMPILER_XLA_SERVICE_HLO_EVALUATOR_H_

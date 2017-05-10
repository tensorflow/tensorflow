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

#include "tensorflow/compiler/xla/service/dfs_hlo_visitor_with_default.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/lib/gtl/flatmap.h"
#include "tensorflow/core/platform/macros.h"

namespace xla {

// Responsible for evaluating HLO and obtain literal as the evaluation results.
//
// This class is not thread-safe.
class HloEvaluator : public DfsHloVisitorWithDefault {
 public:
  HloEvaluator() {}
  ~HloEvaluator() override {}

  // Evaluates a HLO computation and an array of pointers to literals.
  // Return the evaluated result as literal if successful.
  // Precondition: argument literals are in post-order corresponding to the
  // input instruction's parameters.
  StatusOr<std::unique_ptr<Literal>> Evaluate(
      HloComputation* computation,
      tensorflow::gtl::ArraySlice<const Literal*> arg_literals);

  // Evaluates a single HLO instruction and an array of pointers to literals.
  // Return the evaluated result as literal if successful.
  // Precondition:
  // 1. argument literals are in post-order corresponding to the input
  // instruction's parameters.
  // 2. the instruction's operands must be of either Parameter or Constant type.
  // TODO(b/35950897): implement more ops other than element-wise ops.
  // TODO(b/35950897): handle broadcasts.
  StatusOr<std::unique_ptr<Literal>> Evaluate(
      HloInstruction* instruction,
      tensorflow::gtl::ArraySlice<const Literal*> arg_literals);

 protected:
  // The following methods implement the DfsHloVisitor interface.
  //
  // DefaultAction here handles all non-specificialized (i.e., instruction
  // without corresponding Handle* method) instructions.
  // TODO(b/35950897): it's likely better to refactor the switches here and push
  // up the switch to templated methods instead, likely at DfsHloVisitor level.
  Status DefaultAction(HloInstruction* hlo_instruction) override;

  Status HandleParameter(HloInstruction* parameter) override;
  Status HandleConstant(HloInstruction* constant,
                        const Literal& literal) override;

 private:
  // Evaluates a single HLO instruction return the result as a Literal if
  // successful. A Status will be returned on error.
  StatusOr<std::unique_ptr<Literal>> EvaluateBasedOnType(
      HloInstruction* instruction);

  // Evaluates an element-wise HLO instruction that has the same output literal
  // type as the operands' types.
  template <typename NativeT>
  StatusOr<std::unique_ptr<Literal>> EvaluateSameTypedElementwise(
      HloInstruction* instruction);

  // Returns the already-evaluated literal result for the instruction.
  // Crash with log if the given instruction has not been evaluated previously.
  const Literal& GetEvaluatedLiteralFor(const HloInstruction* hlo) {
    auto it = evaluated_.find(hlo);
    CHECK(it != evaluated_.end())
        << "could not find evaluated value for: " << hlo->ToString();
    return *(it->second);
  }

  // Tracks the HLO instruciton and its evaluated literal result.
  tensorflow::gtl::FlatMap<const HloInstruction*, std::unique_ptr<Literal>>
      evaluated_;
  // Stores input literals, assuming they are in post-order. Literals are not
  // owned by this class, and they must outlive the lifetime of the instance of
  // this class.
  tensorflow::gtl::ArraySlice<const Literal*> arg_literals_;

  TF_DISALLOW_COPY_AND_ASSIGN(HloEvaluator);
};

}  // namespace xla

#endif  // THIRD_PARTY_TENSORFLOW_COMPILER_XLA_SERVICE_HLO_EVALUATOR_H_

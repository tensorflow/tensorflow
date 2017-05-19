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
  HloEvaluator();
  // Evaluates a HLO computation and an array of pointers to literals.
  // Return the evaluated result as literal if successful.
  // Precondition: argument literals are corresponds to the input computation's
  // parameters in their post-ordering. For e.g., consider the following graph:
  //
  //                *
  //            /       \
  //            +     Parameter1
  //        /      \
  //       /        \
  //    Parameter0  Constant
  //
  // The input literals array will have its first literal map to Parameter0 and
  // the second map to Parameter1.
  StatusOr<std::unique_ptr<Literal>> Evaluate(
      HloComputation* computation,
      tensorflow::gtl::ArraySlice<const Literal*> arg_literals);

  // Evaluates a single HLO instruction and an array of pointers to literals.
  // Return the evaluated result as literal if successful.
  // Precondition:
  // 1. argument literals are corresponds to the input instruction's
  // parameters in their post-orderring.
  // 2. the instruction's operands must be of either Parameter or Constant type.
  // TODO(b/35950897): implement more ops other than element-wise ops.
  StatusOr<std::unique_ptr<Literal>> Evaluate(
      HloInstruction* instruction,
      tensorflow::gtl::ArraySlice<const Literal*> arg_literals);

 protected:
  // Templated DfsHloVisitor. Typically ReturnT here indicates the resulting
  // literal type of each evaluated Handle* method of a TypedVisitor. One
  // exception to this is HandleCompare, where the resulting literal type is
  // always boolean.
  // Note the forward declaration here is necessary to enable TypedVisitor to
  // access parent members.
  template <typename ReturnT>
  class TypedVisitor;

  // Wraps around instruction handling to infer types before dispatching to
  // the corresponding typed Visitor.
  Status DefaultAction(HloInstruction* hlo) override {
    return hlo->Visit(typed_visitors_.at(hlo->shape().element_type()).get());
  }

  Status HandleParameter(HloInstruction* parameter) override;

  Status HandleConstant(HloInstruction* constant,
                        const Literal& literal) override;

 private:
  // Returns the already-evaluated literal result for the instruction.
  // Crash with log if the given instruction has not been evaluated previously.
  const Literal& GetEvaluatedLiteralFor(const HloInstruction* hlo) {
    auto it = evaluated_.find(hlo);
    CHECK(it != evaluated_.end())
        << "could not find evaluated value for: " << hlo->ToString();
    return *(it->second);
  }

  // Map from a primitive type to its associated (templated) DfsHloVisitor.
  // Note: the hash function here is only needed because current gcc std::hash
  // does not specialize for enum types. This should however be fixed in the
  // future: https://gcc.gnu.org/bugzilla/show_bug.cgi?id=60970#c5
  tensorflow::gtl::FlatMap<PrimitiveType, std::unique_ptr<DfsHloVisitor>,
                           std::hash<int>>
      typed_visitors_;

  // Tracks the HLO instruciton and its evaluated literal result.
  // TODO(b/35950897): have better memory management here to free instructions
  // that are no longer a parent for any other subsequent instruction in
  // post-orderring.
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

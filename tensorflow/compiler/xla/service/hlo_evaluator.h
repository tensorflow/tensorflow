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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_HLO_EVALUATOR_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_HLO_EVALUATOR_H_

#include <memory>

#include "tensorflow/compiler/xla/ptr_util.h"
#include "tensorflow/compiler/xla/service/dfs_hlo_visitor_with_default.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
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
  // Only evaluate up to max_loop_iterations per while-loop execution if
  // specified.
  explicit HloEvaluator(int64 max_loop_iterations = -1);

  // Evaluates an HLO module and an array of pointers to literals.
  // Returns the evaluated result as a literal if successful.
  // Precondition: The indices of arg_literals correspond to the parameter
  // numbers of the HLO parameters in the computation. See comment below for an
  // example.
  // `LiteralPtr` accepts either std::unique_ptr<Literal> or const Literal*
  // type.
  template <typename LiteralPtr>
  StatusOr<std::unique_ptr<Literal>> Evaluate(
      const HloModule& module,
      tensorflow::gtl::ArraySlice<LiteralPtr> arg_literals);

  // Evaluates an HLO computation and an array of pointers to literals.
  // Returns the evaluated result as a literal if successful.
  // Precondition: The indices of arg_literals correspond to the parameter
  // numbers of the HLO parameters in the computation. For e.g., consider the
  // following graph:
  //
  //                *
  //            /       \
  //            +     Parameter1
  //        /      \
  //       /        \
  //    Parameter0  Constant
  //
  // where Parameter0 has parameter_number 0 and Parameter1 has parameter_number
  // 1 in this computation. The input literals array will then have its first
  // literal map to Parameter0 and the second map to Parameter1.
  // `LiteralPtr` accepts either std::unique_ptr<Literal> or const Literal*
  // type.
  template <typename LiteralPtr>
  StatusOr<std::unique_ptr<Literal>> Evaluate(
      const HloComputation& computation,
      tensorflow::gtl::ArraySlice<LiteralPtr> arg_literals);

  // Evaluates a single HLO instruction and an array of pointers to literals.
  // Return the evaluated result as literal if successful.
  // Precondition:
  // 1. argument literals correspond to the input instruction's parameters in
  // their post-ordering.
  // 2. the instruction's operands must be of either Parameter or Constant type.
  // `LiteralPtr` accepts either std::unique_ptr<Literal> or const Literal*
  // type.
  template <typename LiteralPtr>
  StatusOr<std::unique_ptr<Literal>> Evaluate(
      HloInstruction* instruction,
      tensorflow::gtl::ArraySlice<LiteralPtr> arg_literals);

  // Evaluates a single HLO instruction with constant operands.
  // Returns the evaluated result as literal if successful.
  // Precondition:
  // 1. all operands of the input instruction are constants.
  // 2. the instruction is not a Parameter operation.
  StatusOr<std::unique_ptr<Literal>> Evaluate(HloInstruction* instruction);

  // Same as Evaluate, except returning nullptr on error.
  std::unique_ptr<Literal> TryEvaluate(HloInstruction* instruction);

  // Evaluates a single HLO instruction, substituting the given literals for
  // some of the instruction's operands.
  //
  // For example, given instruction = op(A, B, C) and the map
  // {A = x, C = y}, this evaluates op(x, B, y).
  StatusOr<std::unique_ptr<Literal>> EvaluateWithSubstitutions(
      const HloInstruction* instruction,
      const std::unordered_map<const HloInstruction*, const Literal*>&
          substitutions);

  StatusOr<std::unique_ptr<Literal>> EvaluateElementwiseBinaryOp(
      HloOpcode opcode, const Literal& lhs, const Literal& rhs);

  StatusOr<std::unique_ptr<Literal>> EvaluateElementwiseUnaryOp(
      HloOpcode opcode, const Literal& operand);

 protected:
  // Make HloEvaluatorTypedVisitor a friend because it is logically part of this
  // class.
  //
  // A straightforward implementation would be to make it a nested class
  // declared and defined in hlo_evaluator.cc.  Instead HloEvaluatorTypedVisitor
  // lives as a separate class with its own header because its template gets
  // instantiated many times and we want to use extern templates to shard out
  // the compilation of those instantiations across multiple cc files.
  template <typename ReturnT, typename ElementwiseT>
  friend class HloEvaluatorTypedVisitor;

  // Wraps around instruction handling to infer types before dispatching to
  // the corresponding typed Visitor.
  Status DefaultAction(HloInstruction* hlo) override {
    return hlo->Visit(typed_visitors_.at(hlo->shape().element_type()).get());
  }

  Status Preprocess(HloInstruction* hlo) override;

  Status Postprocess(HloInstruction* hlo) override;

  // Operations that are type-agnostic or always return a specific type, such as
  // HandleIsFinite where boolean is always returned.
  //
  Status HandleParameter(HloInstruction* parameter) override;

  Status HandleConstant(HloInstruction* constant) override;

  Status HandleConcatenate(HloInstruction* concatenate) override;

  Status HandleReshape(HloInstruction* reshape) override;

  Status HandleTranspose(HloInstruction* transpose) override;

  Status HandleIsFinite(HloInstruction* is_finite) override;

  Status HandleCompare(HloInstruction* compare) override;

  Status HandleTuple(HloInstruction* tuple) override;

  Status HandleGather(HloInstruction* gather) override;

  Status HandleGetTupleElement(HloInstruction* get_tuple_element) override;

  Status HandleCopy(HloInstruction* copy) override;

  Status HandleConditional(HloInstruction* conditional) override;

  Status HandleCall(HloInstruction* call) override;

  Status HandleFusion(HloInstruction* fusion) override;

  Status HandleWhile(HloInstruction* while_hlo) override;

  Status HandleSelect(HloInstruction* select) override;

  Status HandleTupleSelect(HloInstruction* tuple_select) override;

  Status HandleBroadcast(HloInstruction* broadcast) override;

  Status HandleAfterAll(HloInstruction* token) override;

  Status HandleSort(HloInstruction* sort) override;

  // Returns the already-evaluated literal result for the instruction.
  // A Constant instruction is considered evaluated and its literal will be
  // returned directly without looking up the cache.
  // Crash with log if the given instruction has not been evaluated previously.
  const Literal& GetEvaluatedLiteralFor(const HloInstruction* hlo) {
    if (hlo->IsConstant()) {
      return hlo->literal();
    }
    auto it = evaluated_.find(hlo);
    CHECK(it != evaluated_.end())
        << "could not find evaluated value for: " << hlo->ToString();
    return *(it->second);
  }

  // Tracks the HLO instruction and its evaluated literal result.
  // TODO(b/35950897): have better memory management here to free instructions
  // that are no longer a parent for any other subsequent instruction in
  // post-orderring.
  // Must be cleared for each evaluation.
  tensorflow::gtl::FlatMap<const HloInstruction*, std::unique_ptr<Literal>>
      evaluated_;

 private:
  template <typename ReturnT, typename NativeT>
  static StatusOr<std::unique_ptr<Literal>> ElementWiseUnaryOpImpl(
      HloInstruction* instruction,
      const std::function<ReturnT(NativeT)>& unary_op,
      const Literal& operand_literal) {
    const auto shape = instruction->shape();
    const auto* operand = instruction->operand(0);

    // TODO(b/35950897, b/27796129): add DCHECK back once implicit broadcast is
    // removed.
    if (!ShapeUtil::SameDimensions(shape, operand->shape())) {
      return Unimplemented(
          "Implicit broadcasting is currently unsupported in HLO evaluator "
          "Shape Mismatch: %s vs %s",
          ShapeUtil::HumanString(shape).c_str(),
          ShapeUtil::HumanString(operand->shape()).c_str());
    }

    auto result = MakeUnique<Literal>(shape);
    TF_RETURN_IF_ERROR(result->Populate<ReturnT>(
        [&](tensorflow::gtl::ArraySlice<int64> multi_index) {
          return unary_op(operand_literal.Get<NativeT>(multi_index));
        }));
    return std::move(result);
  }

  // Map from a primitive type to its associated (templated) DfsHloVisitor.
  // Note: the hash function here is only needed because current gcc std::hash
  // does not specialize for enum types. This should however be fixed in the
  // future: https://gcc.gnu.org/bugzilla/show_bug.cgi?id=60970#c5
  tensorflow::gtl::FlatMap<PrimitiveType, std::unique_ptr<DfsHloVisitor>,
                           std::hash<int>>
      typed_visitors_;

  // Caches pointers to input literals, assuming they are in post-order.
  // Literals are not owned by this class, and they must outlive the lifetime of
  // each invocation to the Evaluate* method.
  // Must be cleared for each evaluation.
  std::vector<const Literal*> arg_literals_;

  // Max loop iterations to execute with no maximum if negative.
  int64 max_loop_iterations_;

  TF_DISALLOW_COPY_AND_ASSIGN(HloEvaluator);
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_HLO_EVALUATOR_H_

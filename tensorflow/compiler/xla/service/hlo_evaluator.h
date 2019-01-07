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

#include "absl/container/node_hash_map.h"
#include "absl/memory/memory.h"
#include "absl/types/span.h"
#include "tensorflow/compiler/xla/array2d.h"
#include "tensorflow/compiler/xla/service/dfs_hlo_visitor_with_default.h"
#include "tensorflow/compiler/xla/service/dynamic_dimension_inference.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/shape_inference.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
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

  // Evaluates an HLO module and an array of pointers to literals.  Returns the
  // evaluated result as a literal if successful.
  //
  // Precondition: The indices of arg_literals correspond to the parameter
  // numbers of the HLO parameters in the computation. See comment below for an
  // example.
  //
  // (Dummy template arg is to reduce the overloading priority of one overload
  // so that Evaluate(module, {}) resolves unambiguously.)
  StatusOr<Literal> Evaluate(const HloModule& module,
                             absl::Span<const Literal* const> arg_literals) {
    return Evaluate(*module.entry_computation(), arg_literals);
  }
  template <typename Dummy = void>
  StatusOr<Literal> Evaluate(const HloModule& module,
                             absl::Span<const Literal> arg_literals) {
    return Evaluate(*module.entry_computation(), arg_literals);
  }

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
  //
  // (Dummy template arg is to reduce the overloading priority of one overload
  // so that Evaluate(module, {}) resolves unambiguously.)
  StatusOr<Literal> Evaluate(const HloComputation& computation,
                             absl::Span<const Literal* const> arg_literals);
  template <typename Dummy = void>
  StatusOr<Literal> Evaluate(const HloComputation& computation,
                             absl::Span<const Literal> arg_literals) {
    std::vector<const Literal*> arg_literal_ptrs;
    for (const auto& l : arg_literals) {
      arg_literal_ptrs.push_back(&l);
    }
    return Evaluate(computation, arg_literal_ptrs);
  }

  // Gets the value of running a single HLO instruction.
  //
  // All of the operands to this instruction must be constants.
  StatusOr<Literal> Evaluate(HloInstruction* instruction);

  // Same as Evaluate, except returning false on error and accepts an output
  // pointer.
  bool TryEvaluate(HloInstruction* instruction, Literal* result);

  // Evaluates a single HLO instruction, substituting the given literals for
  // some of the instruction's operands.
  //
  // For example, given instruction = op(A, B, C) and the map
  // {A = x, C = y}, this evaluates op(x, B, y).
  StatusOr<Literal> EvaluateWithSubstitutions(
      const HloInstruction* instruction,
      const std::unordered_map<const HloInstruction*, const Literal*>&
          substitutions);

  StatusOr<Literal> EvaluateElementwiseBinaryOp(HloOpcode opcode,
                                                const Literal& lhs,
                                                const Literal& rhs);

  StatusOr<Literal> EvaluateElementwiseUnaryOp(HloOpcode opcode,
                                               const Literal& operand);

  StatusOr<Literal> EvaluateDotOp(const DotDimensionNumbers& dim_numbers,
                                  const PrecisionConfig& precision_config,
                                  const Literal& lhs, const Literal& rhs);

  void set_dynamic_dimension_inference(
      DynamicDimensionInference* dynamic_dimension_inference) {
    dynamic_dimension_inference_ = dynamic_dimension_inference;
  }

  // Enable the fast path for certain operations like dot or convolution.
  void set_use_fast_path(bool value) { use_fast_path_ = value; }

  // Returns the result of a matrix multiply `lhs x rhs`.
  static std::unique_ptr<Array2D<Eigen::half>> MatmulArray2D(
      const Array2D<Eigen::half>& lhs, const Array2D<Eigen::half>& rhs);
  static std::unique_ptr<Array2D<float>> MatmulArray2D(
      const Array2D<float>& lhs, const Array2D<float>& rhs);
  static std::unique_ptr<Array2D<double>> MatmulArray2D(
      const Array2D<double>& lhs, const Array2D<double>& rhs);

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
    return hlo->Visit(typed_visitors_[hlo->shape().element_type()].get());
  }

  Status Preprocess(HloInstruction* hlo) override;

  Status Postprocess(HloInstruction* hlo) override;

  // Operations that are type-agnostic or always return a specific type, such as
  // HandleIsFinite where boolean is always returned.
  //
  Status HandleBitcast(HloInstruction* bitcast) override;

  Status HandleGetDimensionSize(HloInstruction* get_dimension_size) override;

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

  Status HandleAfterAll(HloInstruction* after_all) override;

  Status HandleAddDependency(HloInstruction* add_dependency) override;

  Status HandleSort(HloInstruction* sort) override;

  Status HandleReal(HloInstruction* real) override;

  Status HandleImag(HloInstruction* imag) override;

  Status HandleComplex(HloInstruction* complex) override;

  Status HandleReduce(HloInstruction* reduce) override;

  // Unsupported HLOs, note some of them (such as BatchNorm*) are typically
  // expanded in a semantic-preserving way into other HLOs by adding exanpsion
  // HLO pass to the HLO optimization pass during compilation, which can then be
  // handled by the evaluator.
  Status HandleBatchNormGrad(HloInstruction* batch_norm_grad) override {
    return Unimplemented("BatchNormGrad HLO is unsupported by the evaluator.");
  };
  Status HandleBatchNormInference(
      HloInstruction* batch_norm_inference) override {
    return Unimplemented(
        "BatchNormInference HLO is unsupported by the evaluator.");
  };
  Status HandleBatchNormTraining(HloInstruction* batch_norm_training) override {
    return Unimplemented(
        "BatchNormTraining HLO is unsupported by the evaluator.");
  };
  Status HandleInfeed(HloInstruction* infeed) override {
    return Unimplemented("Infeed HLO is unsupported by the evaluator.");
  };
  Status HandleOutfeed(HloInstruction* outfeed) override {
    return Unimplemented("Outfeed HLO is unsupported by the evaluator.");
  };

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
    return it->second;
  }

  // Tracks the HLO instruction and its evaluated literal result.
  // TODO(b/35950897): have better memory management here to free instructions
  // that are no longer a parent for any other subsequent instruction in
  // post-orderring.
  // Must be cleared for each evaluation.
  // Storing Literal in place require the container to have pointer stability so
  // we cannot use flat_hash_map any more.
  absl::node_hash_map<const HloInstruction*, Literal> evaluated_;

  // Use fast path that uses eigen in the evaluator.
  bool use_fast_path_ = false;

 private:
  template <typename ReturnT, typename NativeT>
  static StatusOr<Literal> ElementWiseUnaryOpImpl(
      HloInstruction* instruction,
      const std::function<ReturnT(NativeT)>& unary_op,
      const Literal& operand_literal) {
    const auto shape = instruction->shape();
    const auto* operand = instruction->operand(0);
    TF_RET_CHECK(ShapeUtil::SameDimensions(shape, operand->shape()));

    Literal result(shape);
    TF_RETURN_IF_ERROR(
        result.Populate<ReturnT>([&](absl::Span<const int64> multi_index) {
          return unary_op(operand_literal.Get<NativeT>(multi_index));
        }));
    return std::move(result);
  }

  // Map from a primitive type to its associated (templated) DfsHloVisitor.
  std::unique_ptr<DfsHloVisitor> typed_visitors_[PrimitiveType_ARRAYSIZE];

  // Caches pointers to input literals, assuming they are in post-order.
  // Literals are not owned by this class, and they must outlive the lifetime of
  // each invocation to the Evaluate* method.
  // Must be cleared for each evaluation.
  std::vector<const Literal*> arg_literals_;

  // Max loop iterations to execute with no maximum if negative.
  int64 max_loop_iterations_;

  // Module-level seed handle.
  uint64 seed_;
  // RNG engine.
  std::minstd_rand0 engine_;

  // DynamicDimensionInference is used to evaluate GetDimensionSize, which
  // returns the dynamic dimension size of its operand.
  DynamicDimensionInference* dynamic_dimension_inference_;

  TF_DISALLOW_COPY_AND_ASSIGN(HloEvaluator);
};

std::unique_ptr<Array2D<float>> MatmulArray2D(const Array2D<float>& lhs,
                                              const Array2D<float>& rhs);
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_HLO_EVALUATOR_H_

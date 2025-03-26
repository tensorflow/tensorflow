/* Copyright 2017 The OpenXLA Authors.

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

#ifndef XLA_HLO_EVALUATOR_HLO_EVALUATOR_H_
#define XLA_HLO_EVALUATOR_HLO_EVALUATOR_H_

#define _USE_MATH_DEFINES

#include <complex>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <random>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/inlined_vector.h"
#include "absl/container/node_hash_map.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "Eigen/Core"
#include "xla/array2d.h"
#include "xla/comparison_util.h"
#include "xla/hlo/analysis/tuple_points_to_analysis.h"
#include "xla/hlo/ir/dfs_hlo_visitor.h"
#include "xla/hlo/ir/dfs_hlo_visitor_with_default.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/literal.h"
#include "xla/service/call_graph.h"
#include "xla/service/dynamic_dimension_inference.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/tsl/platform/errors.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/ml_dtypes.h"

namespace xla {

// Responsible for evaluating HLO and obtain literal as the evaluation results.
//
// This class is not thread-safe.
class HloEvaluator : public ConstDfsHloVisitorWithDefault {
 public:
  // Precomputed analyses that can be passed to Evaluate functions to avoid
  // recomputation during evaluation.
  struct PrecomputedAnalyses {
    TuplePointsToAnalysis* tuple_points_to;
    CallGraph* call_graph;
  };

  // Only evaluate up to max_loop_iterations per while-loop execution if
  // specified.
  explicit HloEvaluator(int64_t max_loop_iterations = -1);

  // Called by the evaluator to create an embedded evaluator to execute a
  // sub-region of control flow. Subclasses should override this to return an
  // instance of the subclass instead.
  virtual std::unique_ptr<HloEvaluator> CreateEmbedded(
      int64_t max_loop_iterations) {
    auto result = std::make_unique<HloEvaluator>(max_loop_iterations);
    result->set_custom_call_handler(custom_call_handler_);
    return result;
  }

  // Enables subclasses to be notified when a new computation is being
  // evaluated.
  virtual void OnEvaluateComputation(const HloComputation& computation) {}

  // Evaluates an HLO module and an array of pointers to literals.  Returns the
  // evaluated result as a literal if successful.
  //
  // Precondition: The indices of args correspond to the parameter numbers of
  // the HLO parameters in the computation. See comment below for an example.
  //
  // (Dummy template arg is to reduce the overloading priority of one overload
  // so that Evaluate(module, {}) resolves unambiguously.)
  absl::StatusOr<Literal> Evaluate(const HloModule& module,
                                   absl::Span<const Literal* const> args) {
    return Evaluate(*module.entry_computation(), args);
  }
  template <typename Dummy = void>
  absl::StatusOr<Literal> Evaluate(const HloModule& module,
                                   absl::Span<const Literal> args) {
    return Evaluate(*module.entry_computation(), args);
  }

  // Evaluates an HLO computation and an array of pointers to literals.
  // Returns the evaluated result as a literal if successful.
  // Precondition: The indices of args correspond to the parameter
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
  absl::StatusOr<Literal> Evaluate(const HloComputation& computation,
                                   absl::Span<const Literal* const> args);

  template <typename Dummy = void>
  absl::StatusOr<Literal> Evaluate(const HloComputation& computation,
                                   absl::Span<const Literal> args) {
    absl::InlinedVector<const Literal*, 8> args_ptrs;
    args_ptrs.reserve(args.size());
    for (const Literal& arg : args) {
      args_ptrs.push_back(&arg);
    }
    return Evaluate(computation, args_ptrs);
  }

  // Gets the value of running a single HLO instruction.
  //
  // This function may recursively evaluate the dependency of this instruction
  // within its parent computation until it encounters something that cannot be
  // evaluated, such as an Infeed or a Parameter instruction.
  // It makes best effort to partially evaluate a dependency if possible.
  // The caller may pass in non-null `precomputed_analyses` to avoid
  // recomputation during evaluation; the caller must ensure that any
  // precomputed analyses were performed on the module containing `instruction`.
  absl::StatusOr<Literal> Evaluate(
      const HloInstruction* instruction,
      PrecomputedAnalyses precomputed_analyses = {},
      bool recursively_evaluate_nonconstant_operands = false);

  // Same as Evaluate, except returning false on error and accepts an output
  // pointer.
  bool TryEvaluate(const HloInstruction* instruction, Literal* result,
                   bool recursively_evaluate_nonconstant_operands = false);

  // Evaluates a single HLO instruction, substituting the given literals for
  // some of the instruction's operands.
  //
  // For example, given instruction = op(A, B, C) and the map
  // {A = x, C = y}, this evaluates op(x, B, y).
  absl::StatusOr<Literal> EvaluateWithSubstitutions(
      const HloInstruction* instruction,
      const absl::flat_hash_map<const HloInstruction*, const LiteralBase*>&
          substitutions,
      bool recursively_evaluate_nonconstant_operands = false);

  absl::StatusOr<Literal> EvaluateElementwiseBinaryOp(HloOpcode opcode,
                                                      const Literal& lhs,
                                                      const Literal& rhs);

  absl::StatusOr<Literal> EvaluateElementwiseUnaryOp(HloOpcode opcode,
                                                     const Literal& operand);

  absl::StatusOr<Literal> EvaluateElementwiseTernaryOp(HloOpcode opcode,
                                                       const Literal& lhs,
                                                       const Literal& rhs,
                                                       const Literal& ehs);

  absl::StatusOr<Literal> EvaluateElementwiseCompareOp(
      ComparisonDirection direction, const Literal& lhs, const Literal& rhs);

  absl::StatusOr<Literal> EvaluateDotOp(const DotDimensionNumbers& dim_numbers,
                                        const PrecisionConfig& precision_config,
                                        const Literal& lhs, const Literal& rhs);

  void set_dynamic_dimension_inference(
      DynamicDimensionInference* dynamic_dimension_inference) {
    dynamic_dimension_inference_ = dynamic_dimension_inference;
  }

  DynamicDimensionInference* dynamic_dimension_inference() {
    return dynamic_dimension_inference_;
  }

  // Enable the fast path for certain operations like dot or convolution.
  void set_use_fast_path(bool value) { use_fast_path_ = value; }

  // Use fast path that doesn't use embedded evaluators in reduce.
  void set_reduce_use_fast_path(bool value) { use_fast_path_reduce_ = value; }

  // Handles evaluation of a custom-call op.
  // Operand literals are provided in |operands| and implementations must
  // populate |output| before returning.
  using CustomCallHandler = std::function<absl::StatusOr<Literal>(
      const HloInstruction* custom_call, absl::Span<const Literal*> operands)>;

  // Sets a handler that is called during evaluation for custom-call ops.
  // If no handler is defined the default error behavior will occur. The handler
  // will be provided evaluated literals for all operands and is expected to
  // return an output literal of the appropriate shape.
  void set_custom_call_handler(CustomCallHandler handler) {
    custom_call_handler_ = std::move(handler);
  }

  // Callback for each multiply-accumulate in each dot or convolution operation.
  using TraceMACHandler = std::function<void(
      int64_t result_index, int64_t lhs_index, int64_t rhs_index)>;

  // Sets a callback for each multiply-accumulate in each dot or convolution
  // operation.
  void set_trace_mac_handler(TraceMACHandler handler) {
    trace_mac_handler_ = std::move(handler);
  }

  // Returns the result of a matrix multiply `lhs x rhs`.
  static std::unique_ptr<Array2D<Eigen::half>> MatmulArray2D(
      const Array2D<Eigen::half>& lhs, const Array2D<Eigen::half>& rhs);
  static std::unique_ptr<Array2D<float>> MatmulArray2D(
      const Array2D<float>& lhs, const Array2D<float>& rhs);
  static std::unique_ptr<Array2D<double>> MatmulArray2D(
      const Array2D<double>& lhs, const Array2D<double>& rhs);
  static std::unique_ptr<Array2D<std::complex<float>>> MatmulArray2D(
      const Array2D<std::complex<float>>& lhs,
      const Array2D<std::complex<float>>& rhs);
  static std::unique_ptr<Array2D<std::complex<double>>> MatmulArray2D(
      const Array2D<std::complex<double>>& lhs,
      const Array2D<std::complex<double>>& rhs);
  static std::unique_ptr<Array2D<int32_t>> MatmulArray2D(
      const Array2D<int32_t>& lhs, const Array2D<int32_t>& rhs);
  static std::unique_ptr<Array2D<tsl::float8_e4m3fn>> MatmulArray2D(
      const Array2D<tsl::float8_e4m3fn>& lhs,
      const Array2D<tsl::float8_e4m3fn>& rhs);
  static std::unique_ptr<Array2D<tsl::float8_e5m2>> MatmulArray2D(
      const Array2D<tsl::float8_e5m2>& lhs,
      const Array2D<tsl::float8_e5m2>& rhs);
  static std::unique_ptr<Array2D<uint8_t>> MatmulArray2D(
      const Array2D<uint8_t>& lhs, const Array2D<uint8_t>& rhs);

 protected:
  // Evaluates the given instruction, and stores the evaluation result in the
  // evaluation state.
  //
  // When a non-empty shape_index is given, the instruction may be partially
  // evaluated at the given shape_index and the rest of the result could be
  // marked as undetermined unless it has been previously evaluated using
  // EvaluateInternal. Such partial evaluation reduces the computation and
  // memory overhead in cases where we need only one tuple element by avoiding
  // the evaluation of a full tuple. Any non-null `precomputed_analyses` will be
  // used instead of recomputing.
  absl::Status EvaluateInternal(
      const HloInstruction* instruction,
      PrecomputedAnalyses precomputed_analyses,
      const ShapeIndex& shape_index = {},
      bool recursively_evaluate_nonconstant_operands = false);

  // Evaluates the result of a `parameter` instruction by traversing the call
  // graph as given in `analyses`. `shape_index` has the same effect as in
  // EvaluateInternal above.
  absl::Status EvaluateParameterFromCallerArgument(
      const HloInstruction* parameter, const ShapeIndex& shape_index,
      PrecomputedAnalyses analyses);

  // Helper method to extract a list of int64_t from evaluated instruction for
  // start_indices for DynamicSlice and DynamicUpdateSlice.
  std::vector<int64_t> GetS64Indices(
      absl::Span<HloInstruction* const> start_indices);

  // Creates a vector of multipliers which can be used to create a linear index
  // into shape.
  //
  // Given the multidimensional index {i1, ..., iN} and
  // M = MakeDimMultipliers(shape), the corresponding linear index LI is simply
  //
  //   LI = i1 * M[1] + i2 * M[2] + ... + iN * M[N].
  //
  // This lets you calculate LI given the multidimensional indices in any order.
  static DimensionVector MakeDimMultipliers(const Shape& shape);

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
  absl::Status DefaultAction(const HloInstruction* hlo) override {
    return hlo->Visit(typed_visitors_[hlo->shape().element_type()].get());
  }

  absl::Status Preprocess(const HloInstruction* hlo) override;
  absl::Status Postprocess(const HloInstruction* hlo) override;

  // Operations that are type-agnostic or always return a specific type, such as
  // HandleIsFinite where boolean is always returned.
  //
  absl::Status HandleBitcast(const HloInstruction* bitcast) override;
  absl::Status HandleBitcastConvert(const HloInstruction* convert) override;
  absl::Status HandleGetDimensionSize(
      const HloInstruction* get_dimension_size) override;
  absl::Status HandleSetDimensionSize(
      const HloInstruction* set_dimension_size) override;
  absl::Status HandleParameter(const HloInstruction* parameter) override;
  absl::Status HandleInfeed(const HloInstruction* infeed) override;
  absl::Status HandleConstant(const HloInstruction* constant) override;
  absl::Status HandleConcatenate(const HloInstruction* concatenate) override;
  absl::Status HandleReshape(const HloInstruction* reshape) override;
  absl::Status HandleTranspose(const HloInstruction* transpose) override;
  absl::Status HandleIsFinite(const HloInstruction* is_finite) override;
  absl::Status HandleCompare(const HloInstruction* compare) override;
  absl::Status HandleTuple(const HloInstruction* tuple) override;
  absl::Status HandleFft(const HloInstruction* fft) override;
  absl::Status HandleGather(const HloInstruction* gather) override;
  absl::Status HandleScatter(const HloInstruction* hlo) override;
  absl::Status HandleGetTupleElement(
      const HloInstruction* get_tuple_element) override;
  absl::Status HandleAsyncStart(const HloInstruction* async_start) override;
  absl::Status HandleAsyncUpdate(const HloInstruction* async_update) override;
  absl::Status HandleAsyncDone(const HloInstruction* async_done) override;
  absl::Status HandleCopy(const HloInstruction* copy) override;
  absl::Status HandleCopyStart(const HloInstruction* copy_start) override;
  absl::Status HandleCopyDone(const HloInstruction* copy_done) override;
  absl::Status HandleConditional(const HloInstruction* conditional) override;
  absl::Status HandleConvert(const HloInstruction* convert) override;
  absl::Status HandleCall(const HloInstruction* call) override;
  absl::Status HandleDynamicSlice(const HloInstruction* dynamic_slice) override;
  absl::Status HandleDynamicUpdateSlice(const HloInstruction* dus) override;
  absl::Status HandleFusion(const HloInstruction* fusion) override;
  absl::Status HandleWhile(const HloInstruction* while_hlo) override;
  absl::Status HandleSelect(const HloInstruction* select) override;
  absl::Status HandleBroadcast(const HloInstruction* broadcast) override;
  absl::Status HandleAfterAll(const HloInstruction* after_all) override;
  absl::Status HandleAddDependency(
      const HloInstruction* add_dependency) override;
  absl::Status HandleReverse(const HloInstruction* reverse) override;
  absl::Status HandleSelectAndScatter(
      const HloInstruction* select_and_scatter) override;
  absl::Status HandleSlice(const HloInstruction* slice) override;
  absl::Status HandleSort(const HloInstruction* sort) override;
  absl::Status HandleStochasticConvert(
      const HloInstruction* stochastic_convert) override;
  absl::Status HandleReal(const HloInstruction* real) override;
  absl::Status HandleImag(const HloInstruction* imag) override;
  absl::Status HandleComplex(const HloInstruction* complex) override;
  absl::Status HandleReduce(const HloInstruction* hlo) override;
  absl::Status HandleReduceWindow(const HloInstruction* hlo) override;
  absl::Status HandleMap(const HloInstruction* map) override;
  absl::Status HandleCustomCall(const HloInstruction* custom_call) override;

  // Unsupported HLOs, note some of them (such as BatchNorm*) are typically
  // expanded in a semantic-preserving way into other HLOs by adding expansion
  // HLO pass to the HLO optimization pass during compilation, which can then be
  // handled by the evaluator.
  absl::Status HandleBatchNormGrad(
      const HloInstruction* batch_norm_grad) override {
    return Unimplemented("BatchNormGrad HLO is unsupported by the evaluator.");
  }
  absl::Status HandleBatchNormInference(
      const HloInstruction* batch_norm_inference) override {
    return Unimplemented(
        "BatchNormInference HLO is unsupported by the evaluator.");
  }
  absl::Status HandleBatchNormTraining(
      const HloInstruction* batch_norm_training) override {
    return Unimplemented(
        "BatchNormTraining HLO is unsupported by the evaluator.");
  }
  absl::Status HandleOutfeed(const HloInstruction* outfeed) override {
    return Unimplemented("Outfeed HLO is unsupported by the evaluator.");
  }

  // Returns the already-evaluated literal result for the instruction.
  //
  // A Constant instruction is considered evaluated and its literal will be
  // returned directly without looking up the cache.
  //
  // Similarly, a Parameter instruction is considered evaluated and its literal
  // is looked up in args.
  //
  // Crash with log if the given instruction has not been evaluated previously.
  const Literal& GetEvaluatedLiteralFor(const HloInstruction* hlo) {
    if (hlo->IsConstant()) {
      return hlo->literal();
    }
    if (hlo->opcode() == HloOpcode::kParameter && state_.has_args()) {
      return *state_.arg(hlo->parameter_number());
    }

    const Literal* literal = state_.find_evaluated(hlo);
    CHECK(literal != nullptr)
        << "could not find evaluated value for: " << hlo->ToString();
    return *literal;
  }

  // Returns the already-evaluated literal result for the instruction and
  // removes it from internal evaluate state.
  Literal ExtractEvaluatedLiteralFor(const HloInstruction* hlo) {
    if (hlo->IsConstant()) {
      return hlo->literal().Clone();
    }
    if (hlo->opcode() == HloOpcode::kParameter && state_.has_args()) {
      return state_.arg(hlo->parameter_number())->Clone();
    }

    CHECK(state_.has_evaluated(hlo))
        << "could not find evaluated value for: " << hlo->ToString();
    return state_.extract_evaluated(hlo);
  }

  // Returns true if the given hlo has been evaluated and cached.
  bool IsAlreadyEvaluated(const HloInstruction* hlo,
                          const ShapeIndex& shape_index = {}) {
    if (hlo->IsConstant()) {
      return true;
    }
    if (hlo->opcode() == HloOpcode::kParameter && state_.has_args()) {
      return true;
    }

    const Literal* literal = state_.find_evaluated(hlo);
    if (literal == nullptr) {
      return false;
    }

    // We may evaluate some elements of a tuple-shaped instruction and mark
    // the other elements as undetermined. This way we avoid the computation
    // and memory overhead of evaluating a large tuple when only some elements
    // are needed. By marking the other elements undetermined, we allow the
    // evaluator to update the cached tuple literal when more elements are
    // evaluated.
    return literal->IsDetermined(shape_index);
  }

  // Sets the evaluated literal for the given instruction.
  void SetEvaluatedLiteralFor(const HloInstruction* hlo, Literal literal) {
    state_.set_evaluated(hlo, std::move(literal));
  }

  // EvaluationState encapsulates the state of an in-progress evaluation. Once
  // evaluation is complete the state is cleaned up.
  //
  // State must be reset before each evaluation. See `ScopedEvaluateState`
  // below for an RAII helper to automatically reset the state.
  class EvaluationState {
   public:
    EvaluationState() = default;

    // Resets the state of the evaluation and sets the argument literals.
    void Reset(absl::Span<const Literal* const> args) {
      args_.clear();
      args_.insert(args_.end(), args.begin(), args.end());
      evaluated_.erase(evaluated_.begin(), evaluated_.end());
    }

    // Resets the state of the evaluation.
    void Reset() {
      args_.clear();
      evaluated_.erase(evaluated_.begin(), evaluated_.end());
    }

    // Returns the argument literals set for the evaluation.
    absl::Span<const Literal* const> args() const { return args_; }
    const Literal* arg(int64_t index) const { return args_.at(index); }
    bool has_args() const { return !args_.empty(); }

    // Sets the evaluated literal for the given instruction.
    void set_evaluated(const HloInstruction* hlo, Literal literal) {
      evaluated_[hlo] = std::move(literal);
    }

    // Returns the evaluated literal for the given instruction, or nullptr if
    // the instruction has not been evaluated.
    Literal* find_evaluated(const HloInstruction* hlo) {
      if (auto it = evaluated_.find(hlo); it != evaluated_.end()) {
        return &it->second;
      }
      return nullptr;
    }

    // Returns true if the given instruction has been evaluated.
    bool has_evaluated(const HloInstruction* hlo) const {
      return evaluated_.contains(hlo);
    }

    // Extracts the evaluated literal for the given instruction and returns it.
    Literal extract_evaluated(const HloInstruction* hlo) {
      return std::move(evaluated_.extract(hlo).mapped());
    }

   private:
    // Caches pointers to input literals, assuming they are in post-order.
    // Literals are not owned by this class, and they must outlive the
    // lifetime of each invocation to the Evaluate* method.
    std::vector<const Literal*> args_;

    // Tracks the HLO instruction and its evaluated literal result.
    //
    // Parameters and constants aren't stored here, for parameters we use
    // literals from `args_` array and for constants we use the literal from the
    // instruction itself.
    //
    // TODO(b/35950897): have better memory management here to free instructions
    // that are no longer a parent for any other subsequent instruction in
    // post-ordering.
    absl::node_hash_map<const HloInstruction*, Literal> evaluated_;
  };

  EvaluationState& state() { return state_; }

 private:
  // An RAII helper for Evaluate* methods that resets the evaluator state with
  // the given argument literals for evaluation, and resets the state when it
  // evaluation is complete.
  class ScopedEvaluateState {
   public:
    explicit ScopedEvaluateState(EvaluationState* state,
                                 absl::Span<const Literal* const> args = {})
        : state_(state) {
      state_->Reset(args);
    }

    ~ScopedEvaluateState() { state_->Reset(); }

   private:
    EvaluationState* state_;
  };

  template <typename ReturnT, typename NativeT, typename UnaryOp>
  static absl::StatusOr<Literal> ElementWiseUnaryOpImpl(
      const HloInstruction* instruction, UnaryOp&& unary_op,
      const Literal& operand_literal) {
    static_assert(std::is_invocable_r_v<ReturnT, UnaryOp, NativeT>,
                  "Invalid UnaryOp signature");

    const Shape& shape = instruction->shape();
    const auto* operand = instruction->operand(0);
    TF_RET_CHECK(ShapeUtil::SameDimensions(shape, operand->shape()));

    Literal result(shape);
    TF_RETURN_IF_ERROR(
        result.PopulateLinearParallel<ReturnT>([&](int64_t linear_index, int) {
          return unary_op(operand_literal.GetLinear<NativeT>(linear_index));
        }));
    return result;
  }

  // Module-level seed handle.
  uint64_t seed_ = 0;

  // RNG engine.
  std::minstd_rand0 engine_;

  // Map from a primitive type to its associated (templated) DfsHloVisitor.
  std::unique_ptr<ConstDfsHloVisitor> typed_visitors_[PrimitiveType_ARRAYSIZE];

  // Max loop iterations to execute with no maximum if negative.
  int64_t max_loop_iterations_ = 0;

  // Use fast path that uses eigen in the evaluator.
  bool use_fast_path_ = false;

  // Use fast path that doesn't use embedded evaluators in reduce.
  bool use_fast_path_reduce_ = true;

  // DynamicDimensionInference is used to evaluate GetDimensionSize, which
  // returns the dynamic dimension size of its operand.
  DynamicDimensionInference* dynamic_dimension_inference_ = nullptr;

  // Optional handler for custom_call ops.
  CustomCallHandler custom_call_handler_;

  // Optional handler for tracing MAC operations (eg in dot and convolution).
  TraceMACHandler trace_mac_handler_;

  // TODO(ezhulenev): Move cache members to EvaluationState.
  std::unique_ptr<CallGraph> call_graph_cache_;
  std::unique_ptr<TuplePointsToAnalysis> tuple_points_to_analysis_cache_;

  // Set by EvaluateInternal and opportunitiscally used by the HandleXXX
  // functions. When non-empty, the HandleXXX function may evaluate the
  // instruction at only the given shape index.
  //
  // TODO(ezhulenev): Move partial evaluation members to EvaluationState.
  ShapeIndex visitor_shape_index_;
  bool enable_partial_evaluation_ = false;

  // Mutable evaluation state that holds the state of an in-progress evaluation.
  EvaluationState state_;

  HloEvaluator(const HloEvaluator&) = delete;
  HloEvaluator& operator=(const HloEvaluator&) = delete;
};

std::unique_ptr<Array2D<float>> MatmulArray2D(const Array2D<float>& lhs,
                                              const Array2D<float>& rhs);

// Represents a parsed static while loop. We normalize the loop representation
// so that it starts from the induction_var_init_value and increments by
// step_size until it exceeds or goes below loop_bound.
struct ParsedStaticWhileLoop {
  // The number of iterations to be executed.
  int64_t trip_count = -1;
  // The tuple index of the induction variable in the while argument tuple.
  int64_t induction_var_index = -1;
  // The induction variable's initial value.
  int64_t induction_var_init_value = -1;
  // The induction variable is incremented by this number (could be negative)
  // in each iteration.
  int64_t step_size = -1;
  int64_t loop_bound = -1;
};

// Indicates whether a parsed while loop is static or dynamic. If the loop is
// static, it contains a value for StaticLoopInfo; otherwise the loop is
// dynamic. We consider a loop dynamic if its induction variable's initial
// value or the loop bounds value depends on the while's parent computation's
// parameter.
struct ParsedWhileLoop {
  std::optional<ParsedStaticWhileLoop> static_while_loop;
  bool is_dynamic() const { return !static_while_loop.has_value(); }
};
constexpr ParsedWhileLoop kParsedDynamicWhileLoop = ParsedWhileLoop();

// Tries to parse a while loop using a set of predefined patterns.
// Returns the parsing result. Any non-null `precompute_analyses` will be used
// instead of recomputing, and it is the caller's responsibility to ensure that
// the analyses are valid for the module that contains `while_op`.
std::optional<ParsedWhileLoop> PatternMatchParseWhileLoop(
    const HloInstruction* while_op,
    HloEvaluator::PrecomputedAnalyses precomputed_analyses = {});

// Functionality exposed for testing. Do not rely on anything in this namespace
// outside this file.
namespace internal {

// Use this class to represent the precise details of the error to enable
// special treatment.
enum class EvalErrorDetail : uint32_t {
  // The evaluation result depends on dynamic values such as parameters and
  // infeed. Therefore, the HLO's value cannot be statically evaluated.
  kDynamicValueDependence = 0,
};

extern const absl::string_view kEvalErrorDetailUrl;

std::optional<EvalErrorDetail> ParseEvalErrorDetail(const absl::Status& error);

}  // namespace internal
}  // namespace xla

#endif  // XLA_HLO_EVALUATOR_HLO_EVALUATOR_H_

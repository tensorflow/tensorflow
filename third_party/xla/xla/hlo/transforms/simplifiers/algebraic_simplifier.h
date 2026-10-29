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

#ifndef XLA_HLO_TRANSFORMS_SIMPLIFIERS_ALGEBRAIC_SIMPLIFIER_H_
#define XLA_HLO_TRANSFORMS_SIMPLIFIERS_ALGEBRAIC_SIMPLIFIER_H_

#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/dfs_hlo_visitor_with_default.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/pass/hlo_pass_interface.h"
#include "xla/hlo/pass/hlo_pass_options.h"
#include "xla/hlo/transforms/simplifiers/algebraic_simplifier_options.pb.h"  // IWYU pragma: export
#include "xla/literal.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

namespace xla {

// The maximum number of times to run the algebraic simplifier to reach a fixed
// point.
static constexpr int64_t kAlgSimpRerunLimit = 50;

// Applies `options` on top of `base`. Keys are AlgebraicSimplifierOptions field
// names; only bool and int64 fields are supported. Unknown or mistyped keys are
// left unconsumed for the caller's `CheckAllConsumed()`.
AlgebraicSimplifierOptions AlgebraicSimplifierOptionsFromHloPassOptions(
    const HloPassOptions& options, AlgebraicSimplifierOptions base = {});

// A pass which performs algebraic simplifications.
class AlgebraicSimplifier : public HloModulePass {
 public:
  // If is_layout_sensitive is true, then the simplifier preserves layout during
  // transformation. Otherwise, layout is ignored.
  explicit AlgebraicSimplifier(const AlgebraicSimplifierOptions& options)
      : options_(options) {}
  ~AlgebraicSimplifier() override = default;
  absl::string_view name() const override { return "algsimp"; }

  // Create constant from literal with tiles and element size updated in the
  // constant's layout.
  std::unique_ptr<HloInstruction> CreateConstantWithLayoutUpdated(
      Literal literal) {
    auto constant = HloInstruction::CreateConstant(std::move(literal));
    UpdateLayout(constant->mutable_shape());
    return constant;
  }

 protected:
  // Run algebraic simplification on the given computation. Returns whether the
  // computation was changed.
  absl::StatusOr<bool> RunImpl(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

  AlgebraicSimplifierOptions options_;
};

// AlgebraicSimplifierVisitor traverses the HLO computation and reduces certain
// algebraic expressions to simplified forms. Note: This only supports
// simplifications that simply look at the operands of an instruction. For the
// more general case a worklist based approach would be needed.
class AlgebraicSimplifierVisitor : public DfsHloRewriteVisitor {
 public:
  explicit AlgebraicSimplifierVisitor(const AlgebraicSimplifierOptions& options,
                                      AlgebraicSimplifier* simplifier)
      : options_(options), simplifier_(simplifier) {}

  absl::Status HandleAbs(HloInstruction* abs) override;

  absl::Status HandleAdd(HloInstruction* add) override;

  absl::Status HandleAllGather(HloInstruction* all_gather) override;

  absl::Status HandleAllReduce(HloInstruction* all_reduce) override;

  absl::Status HandleReduceScatter(HloInstruction* reduce_scatter) override;

  absl::Status HandleAllToAll(HloInstruction* all_to_all) override;

  absl::Status HandleAnd(HloInstruction* logical_and) override;

  absl::Status HandleBitcast(HloInstruction* bitcast) override;

  absl::Status HandleBitcastConvert(HloInstruction* bitcast) override;

  absl::Status HandleBroadcast(HloInstruction* broadcast) override;

  absl::Status HandleCompare(HloInstruction* compare) override;

  absl::Status HandleConditional(HloInstruction* conditional) override;

  absl::Status HandleConcatenate(HloInstruction* concatenate) override;

  absl::Status HandleConstant(HloInstruction* constant) override;

  absl::Status HandleCopy(HloInstruction* copy) override;

  absl::Status HandleConvert(HloInstruction* convert) override;

  absl::Status HandleComplex(HloInstruction* complex) override;

  absl::Status HandleCustomCall(HloInstruction* custom_call) override;

  absl::Status HandleExp(HloInstruction* exp) override;

  absl::Status HandleReal(HloInstruction* real) override;

  absl::Status HandleImag(HloInstruction* imag) override;

  absl::Status HandleIota(HloInstruction* instruction) override;

  absl::Status HandleConvolution(HloInstruction* convolution) override;

  absl::Status HandleDivide(HloInstruction* divide) override;

  absl::Status HandleDot(HloInstruction* dot) override;

  absl::Status HandleGather(HloInstruction* gather) override;

  absl::Status HandleGetTupleElement(
      HloInstruction* get_tuple_element) override;

  absl::Status HandleLog(HloInstruction* log) override;

  absl::Status HandleMaximum(HloInstruction* maximum) override;

  absl::Status HandleMinimum(HloInstruction* minimum) override;

  absl::Status HandleClamp(HloInstruction* clamp) override;

  absl::Status HandleMultiply(HloInstruction* multiply) override;

  absl::Status HandleNegate(HloInstruction* negate) override;

  absl::Status HandleNot(HloInstruction* logical_not) override;

  absl::Status HandleOptimizationBarrier(HloInstruction* barrier) override;

  absl::Status HandleOr(HloInstruction* logical_or) override;

  absl::Status HandlePad(HloInstruction* pad) override;

  absl::Status HandlePower(HloInstruction* power) override;

  absl::Status HandleRemainder(HloInstruction* remainder) override;

  absl::Status HandleReshape(HloInstruction* reshape) override;

  absl::Status HandleReduce(HloInstruction* hlo) override;

  absl::Status HandleReducePrecision(HloInstruction* hlo) override;

  absl::Status HandleReduceWindow(HloInstruction* hlo) override;

  absl::Status HandleReverse(HloInstruction* reverse) override;

  absl::Status HandleRsqrt(HloInstruction* rsqrt) override;

  absl::Status HandleShuffle(HloInstruction* hlo) override;

  absl::Status HandleSlice(HloInstruction* slice) override;

  absl::Status HandleSqrt(HloInstruction* sqrt) override;

  absl::Status HandleDynamicSlice(HloInstruction* dynamic_slice) override;

  absl::Status HandleDynamicUpdateSlice(
      HloInstruction* dynamic_update_slice) override;
  absl::Status HandleScatter(HloInstruction* hlo) override;

  absl::Status HandleSelect(HloInstruction* select) override;

  absl::Status HandleSort(HloInstruction* sort) override;

  absl::Status HandleTranspose(HloInstruction* transpose) override;

  absl::Status HandleSubtract(HloInstruction* sub) override;

  absl::Status HandleMap(HloInstruction* map) override;

  absl::Status HandleXor(HloInstruction* logical_xor) override;

  // Runs the visitor on a computation.
  bool Run(HloComputation* computation,
           const AlgebraicSimplifierOptions& options,
           AlgebraicSimplifier* simplifier);

  // Compute a function that maps from bitcasted dimensions to the resulting
  // ones. Returns the function as a vector if successful; std::optional
  // otherwise.
  static std::optional<std::vector<std::vector<int64_t>>> ComputeBitcastDimMap(
      const Shape& bitcast_shape, const Shape& operand_shape);
  // Invert the directions of the given bitcast dimension map.
  static std::vector<std::vector<int64_t>> InvertBitcastDimMap(
      const Shape& original_shape, const Shape& bitcast_shape,
      const std::vector<std::vector<int64_t>>& original_map);

  // Checks if the output of a given instruction is guaranteed to be
  // non-negative. e.g. abs
  static bool IsNonNegative(const HloInstruction* hlo,
                            const AlgebraicSimplifierOptions& options);

  // Check if the opcode of a given instruction is a non-decreasing function
  // asymptotically satisfying |f(x)| <= |x|
  static bool IsNondecreasingSublinear(const HloInstruction* hlo);

  // Modify the layout dimensions of result_shape, so that it becomes the
  // re-shaped result of applying bitcast to the original_shape, by using
  // dim_map to re-shape layout dimensions of original_shape. Returns the
  // result_shape with modified layout if the conversion succeeds; Returns
  // std::nullopt if fails.
  static std::optional<Shape> ReshapeLayoutDimensions(
      const Shape& original_shape, const Shape& result_shape,
      const std::vector<std::vector<int64_t>>& original_map,
      const std::vector<std::vector<int64_t>>& result_map);

  // Allow backend constraints on tiling etc. to invalidate optimizations.
  virtual bool IsValidLayout(const Shape& shape) { return true; }
  // Allow backend targets to determine whether a layout is inefficient.
  virtual bool ShouldStrengthReduceDotToReduce(const HloInstruction* hlo);

 protected:
  // Returns whether a reshape from `from_shape` to `to_shape` is a bitcast.
  bool ReshapeIsBitcast(const Shape& from_shape, const Shape& to_shape) const {
    return options_.is_layout_sensitive() &&
           options_.enable_reshape_is_bitcast() &&
           TargetReshapeIsBitcast(from_shape, to_shape);
  }

  // Platform-dependent hook to determine if a reshape `from_shape` to
  // `to_shape` is a bitcast.
  virtual bool TargetReshapeIsBitcast(const Shape& from_shape,
                                      const Shape& to_shape) const {
    return ShapeUtil::ReshapeIsBitcast(from_shape, to_shape);
  }

  // Platform-dependent hook to determine if convolution `conv` is lowerable.
  virtual bool ConvIsLowerable(HloInstruction* conv) const { return true; }

  // A method that allows various backends to specialize the propagation of
  // various attributes to the new instruction.
  virtual void SetupDerivedInstruction(HloInstruction* old_inst,
                                       HloInstruction* new_inst,
                                       bool preserve_user_fusion_attr) {
    old_inst->SetupDerivedInstruction(new_inst);
  }

  // The backend-specific options selected for the algebraic simplifier.
  const AlgebraicSimplifierOptions& options_;

  // Rewrite dot as mul(broadcast(transpose(x)),broadcast(transpose(y)))
  absl::Status RewriteAsMultiplyDotWithZeroLhsContractingDim(
      HloInstruction* dot, HloInstruction* lhs, HloInstruction* rhs,
      const DotDimensionNumbers& dnums);

  // Rewrites a dot as
  // reduce(mul(broadcast(transpose(x)),broadcast(transpose(y)))), where the
  // reduction is over the contracting dimensions. Supports arbitrary dots: the
  // lhs and rhs may each have any number of batch, contracting and
  // non-contracting dimensions.
  absl::Status RewriteBatchPlusContractingAsReduce(
      HloDotInstruction* dot, HloInstruction* lhs, HloInstruction* rhs,
      const DotDimensionNumbers& dnums);

 private:
  // Returns an operand of a chain of reshapes and copies that is bit-wise
  // identical to first reshape or copy in the chain.
  HloInstruction* BitcastingOperandOfReshapeOrCopyChain(
      HloInstruction* instr) const;

  // Returns whether the dot precision config is supported by simplifier.
  virtual bool SupportedDotPrecisionConfig(const PrecisionConfig& config,
                                           bool has_contracting_dim);

  // Makes algorithm specific set of instructions for multiply with precision
  // algorithm in mind. In the trivial case it returns just multiply.
  // For x3 or x6 algorithms it adds the parameters split instructions and the
  // corresponding multiply instructions.
  virtual absl::StatusOr<HloInstruction*> MakeMultiplyForPrecisionAlgorithm(
      HloInstruction* dot, HloInstruction* lhs, HloInstruction* rhs);

  absl::Status HandleAllReduceOrReduceScatter(HloInstruction* collective);

  enum class RewriteResult {
    kNoRewrite,
    kRewritten,
    kStopRewrites,
  };

  // Reorder nested dots with associativity using flops as a heuristic
  // Could return kStopRewrites if the rewrite is too expensive.
  absl::StatusOr<RewriteResult> AssociativeReorderNestedDot(
      HloDotInstruction* dot, HloInstruction* lhs, HloInstruction* rhs);

  // Removes degenerate dimension from dot.
  absl::StatusOr<bool> RemoveDegenerateDimensionFromDot(HloDotInstruction* dot);

  // Moves the transpose to the broadcast if possible. Can also be called with a
  // bitcast transpose.
  absl::Status SimplifyTransposeOfBroadcast(
      HloInstruction* transpose, absl::Span<const int64_t> dimensions);

  // Folds a trailing Transpose into a Scatter operation by transposing the
  // scatter operand and updates upfront.
  absl::StatusOr<bool> TryFoldTransposeIntoScatter(HloInstruction* transpose);

  // Converts to primitive type if the input hlo is not that type, otherwise
  // returns the original hlo.
  HloInstruction* AsType(HloInstruction* hlo,
                         const PrimitiveType element_type) {
    if (hlo->shape().element_type() == element_type) {
      return hlo;
    }
    Shape changed_shape =
        ShapeUtil::ChangeElementType(hlo->shape(), element_type);
    simplifier_->UpdateLayout(&changed_shape);
    return computation_->AddInstruction(
        HloInstruction::CreateConvert(changed_shape, hlo));
  }

  // Transposes a dot operand such that the batch dimensions are the most major,
  // and the contracting dimensions are most minor.
  absl::StatusOr<HloInstruction*>
  NormalizeDotOperandToBatchMajorAndContractingMinor(
      HloInstruction* dot_operand, absl::Span<const int64_t> batch_dimensions,
      absl::Span<const int64_t> contracting_dimensions);

  // Simplify dot(transpose(a), transpose(b)) to transpose(dot(b,a)) (or
  // transpose(dot(a,b)) if only the batch dims are transposed).
  //
  // Requires the dot has been canonicalized by DotDecomposer into
  //
  //   LHS [batch dims..., non-contracting dim, contracting dim]
  //   RHS [batch dims..., contracting dim, non-contracting dim].
  absl::StatusOr<bool> RemoveTransposesFromDotOperands(HloDotInstruction* dot);

  // Swap the operands of dots, if one operand is "parameter-like" (i.e. a
  // parameter, or a pointwise transformation of a parameter), so the
  // "parameter-like" operand (e.g. a weight tensor) is placed on the RHS.
  absl::StatusOr<bool> MoveDotParamToRhs(HloDotInstruction* dot);

  // Helper method to perform and add reduction on a list of dimensions.
  HloInstruction* AddReduce(HloInstruction* hlo, absl::Span<const int64_t> dims,
                            PrimitiveType type);

  // Move scalar multiply to the smallest side of convolution to
  // reduce multiply computations.
  absl::Status ScalarMultiplyReduction(HloInstruction* dot);

  // Convenience method for replacing an instruction with a bitcast. If operand
  // is not null, then the bitcast will use the specified operand instead of the
  // operand of the instruction.
  void ReplaceWithBitcast(HloInstruction* instruction,
                          HloInstruction* operand = nullptr);

  // Change copy(bitcast...(copy)) into copy(bitcast) or bitcast(copy) so that
  // the replicated copies are combined when allowed by layout/tiling assignment
  // constraints.
  bool SwapCopyBitcastCopy(HloInstruction* root_copy);

  // Replace old instruction with new instruction if old and new instructions
  // are compatible (have the same shape and replacement preserves sharding).
  // Updates uses and root instruction. Returns whether a replacement was made.
  bool ReplaceInstructionIfCompatible(HloInstruction* old_instruction,
                                      HloInstruction* new_instruction);
  // Similar to above but tuplizes `new_instructions` if there are more than 1
  // instructions.
  bool ReplaceInstructionIfCompatible(
      HloInstruction* old_instruction,
      absl::Span<HloInstruction* const> new_instructions);

  // Returns whether the shape of the output of the given instructions are the
  // same for the purposes of simplification. If options_.is_layout_sensitive()
  // is true, then this tests shape equality including layout
  // (ShapeUtil::Equal). If options_.is_layout_sensitive() is false, then the
  // tests shape compatibility (ShapeUtil::Compatible).
  bool SameShape(const HloInstruction* lhs, const HloInstruction* rhs) const;

  // Same as above but takes shape arguments directly.
  bool SameShape(const Shape& lhs, const Shape& rhs) const;

  // Attempts to sink broadcasts to after element-wise operations if all
  // operands of that element-wise operation are compatible broadcasts. Returns
  // whether a change was made.
  absl::StatusOr<bool> TryToSinkBroadcastAfterElementwiseOps(
      HloInstruction* broadcast);

  absl::StatusOr<HloInstruction*> OptimizeDotOfConcat(HloInstruction* dot);
  absl::StatusOr<HloInstruction*> OptimizeDotOfConcatHelper(
      HloInstruction* dot, HloInstruction* lhs, int64_t lhs_contracting_dim,
      HloInstruction* rhs, int64_t rhs_contracting_dim, bool swapped);

  absl::StatusOr<HloInstruction*> OptimizeDotOfGather(HloInstruction* dot);

  absl::StatusOr<HloInstruction*> OptimizeDotOfReorderContractingDims(
      HloInstruction* dot);

  absl::StatusOr<HloInstruction*> AssociativeReorderDotOperator(
      HloDotInstruction* dot);

  HloComputation* GetOrCreateScalarAddComputation(PrimitiveType type) {
    HloComputation*& scalar_add_computation = scalar_add_computations_[type];
    if (scalar_add_computation) {
      return scalar_add_computation;
    }

    HloComputation::Builder b("scalar_add_computation");
    Shape shape = ShapeUtil::MakeShape(type, {});
    simplifier_->UpdateLayout(&shape);
    auto scalar_lhs = b.AddInstruction(
        HloInstruction::CreateParameter(0, shape, "scalar_lhs"));
    auto scalar_rhs = b.AddInstruction(
        HloInstruction::CreateParameter(1, shape, "scalar_rhs"));
    auto scalar_op = b.AddInstruction(HloInstruction::CreateBinary(
        shape, HloOpcode::kAdd, scalar_lhs, scalar_rhs));
    scalar_add_computation =
        computation_->parent()->AddEmbeddedComputation(b.Build(scalar_op));
    return scalar_add_computation;
  }

  // Tries to fold a kPad in the input or filter into the convolution
  // instruction's window.
  virtual absl::StatusOr<bool> FoldConvInputPad(HloInstruction* convolution);
  absl::StatusOr<bool> FoldConvFilterPad(HloInstruction* convolution);

  // Tries to swap convolution operands if they would result in a more efficient
  // convolution.
  absl::StatusOr<bool> SwapConvOperands(HloInstruction* convolution);

  // Checks if the given convolution is in BF16 and is oneDNN rewritable, if not
  // then it promotes the data type of the convolution to F32
  absl::StatusOr<bool> PromoteConvolutionToF32IfNotOnednnCompatible(
      HloInstruction** convolution);

  // Tries to use a kDot in place of the given convolution.
  absl::StatusOr<bool> SimplifyConvToDot(HloInstruction* convolution);
  // Tries to use a multiplication in place of the given convolution.
  absl::StatusOr<bool> SimplifyConvToMultiply(HloInstruction* convolution);

  // Tries to reorder mul(add(conv(input, filter), bias), multiplier) ->
  // add(conv(input, mul(filter, multiplier)), mul(bias, multiplier)). It only
  // does that when the multiplier is a 1D constant of the size equal to the
  // convolution output feature dimension.
  absl::Status TryToReorderConvAddMultiply(HloInstruction* multiply);

  // Tries to simplify a slice where the result of the slice is a scalar.
  absl::StatusOr<bool> TrySimplifyScalarSlice(HloInstruction* slice);

  // Tries to convert slice(reshape(X)) into reshape(slice(X))
  absl::StatusOr<bool> TryToReorderSliceAndReshape(HloInstruction* slice);

  // Tries to convert slice(reverse(X)) into reverse(slice(X))
  absl::StatusOr<bool> TryToReorderSliceAndReverse(HloInstruction* slice);

  // Tries to simplify `(and (< a N) (< a K))` in cases where `N <= K` into
  // `(< a N)`. This is crucial for being able to figure out the loop trip
  // count.
  //
  // Assumes that the input is conjunction.
  absl::StatusOr<bool> TrySimplifyTautologicalCompare(
      HloInstruction* conjunction);

  // Tries to simplify (bitcast-convert (concat (bitcast-convert A) ...)) where
  // the types of inner and outer bitcast-convert cancel out.
  absl::StatusOr<bool> TrySimplifyTautologicalBitcastConvert(
      HloInstruction* bitcast);

  // Tries to remove surrounding converts around a binary op where the op has a
  // more precise type than its inputs and output.
  //
  // convert<TS>(bin_op<TL>(convert<TL>(data1<TS>),
  //                        convert<TL>(data2<TS>)))
  //  where TS is a smaller point type than TL (ex, TS=fp16, TL=fp32)
  // ->
  // bin_op<TS>(data1<TS>, data2<TS>)
  absl::Status TryRemoveUpcastAndDowncastSurroundingBinaryOp(
      HloInstruction* convert_instruction);

  // Useful when we want to use the same visitor over multiple computations.
  void ResetState(HloComputation* computation);

  // For cases where the stride won't end up being used, we update the limit
  // and reset the stride to 1. Returns true if the stride is redundant (and the
  // slice instruction is replaced).
  // - For example in slices=([0:X:X]), where X == dimension
  absl::StatusOr<bool> RemoveRedundantStride(
      HloInstruction* absl_nonnull slice);

  // Helper function for HandleReduce. Replaces a reduce with a broadcast of the
  // init values if the reduce is operating on a zero-element array or the
  // result of the reduce is a zero-element array.
  absl::Status BroadcastReduce(const Shape& reduce_result_shape,
                               bool multi_output_reduce,
                               HloReduceInstruction* reduce);

  // Helper function for HandleReduce. Converts a (both single and multi output)
  // reduce to a reshape.
  absl::Status ReplaceReduceWithReshape(const Shape& reduce_result_shape,
                                        bool multi_output_reduce,
                                        HloReduceInstruction* reduce);

  // Detects a chain of transposes and reshapes (or bitcasts) that can be
  // replaced with a nop.
  absl::StatusOr<bool> TryRemovingBitcastOrReshapeTransposeChain(
      HloInstruction* instruction);

  // Tries to hoist transpose over reshape:
  // Reshape(Transpose(Reshape(x))) -> Transpose(Reshape(x))
  absl::StatusOr<bool> TryHoistTransposeOfReshape(HloInstruction* reshape);

  // Helper function for HandleReduce. Reorders reduce dot
  // to a dot reduce. reduce(dot(A, B)) to dot(A, reduce(B))
  std::optional<absl::Status> ReorderReduceDotToDotReduce(
      HloInstruction* arg, HloInstruction* init_value, HloComputation* function,
      HloReduceInstruction* reduce);

  // Helper function for HandleReduce. Merge two reduces with same computation
  // and initial value into a single reduce.
  absl::Status MergeReduces(const Shape& reduce_result_shape,
                            HloInstruction* init_value,
                            HloComputation* function, HloInstruction* arg,
                            HloReduceInstruction* reduce);

  // Current HloComputation instance the AlgebraicSimplifierVisitor is
  // traversing.
  HloComputation* computation_;

  // Cached computation for adding two scalars of a given type.
  absl::flat_hash_map<PrimitiveType, HloComputation*> scalar_add_computations_;

  AlgebraicSimplifier* simplifier_ = nullptr;
};

}  // namespace xla

#endif  // XLA_HLO_TRANSFORMS_SIMPLIFIERS_ALGEBRAIC_SIMPLIFIER_H_

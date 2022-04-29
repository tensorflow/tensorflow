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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_ALGEBRAIC_SIMPLIFIER_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_ALGEBRAIC_SIMPLIFIER_H_

#include <cstdint>
#include <functional>
#include <utility>

#include "absl/container/inlined_vector.h"
#include "tensorflow/compiler/xla/service/dfs_hlo_visitor_with_default.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_pass_interface.h"
#include "tensorflow/compiler/xla/util.h"

namespace xla {

class AlgebraicSimplifierOptions {
 public:
  // Platform dependent callback to determine if a reshape `from_shape` to
  // `to_shape` is a bitcast.
  using ReshapeIsBitcastCallback =
      std::function<bool(const Shape& from_shape, const Shape& to_shape)>;
  // Platform dependent callback to determine if a set of reverse dimensions is
  // lowerable
  using ConvIsLowerableCallback = std::function<bool(HloInstruction* window)>;

  explicit AlgebraicSimplifierOptions(
      ReshapeIsBitcastCallback reshape_is_bitcast_callback = {},
      ConvIsLowerableCallback conv_is_lowerable_callback = {})
      : reshape_is_bitcast_callback_(std::move(reshape_is_bitcast_callback)),
        conv_is_lowerable_callback_(std::move(conv_is_lowerable_callback)) {}

  // Use the platform specific callback if set. It is not sensible to return
  // true here if the options are not layout sensitive.
  bool ReshapeIsBitcast(const Shape& from_shape, const Shape& to_shape) const {
    if (!is_layout_sensitive_) {
      return false;
    }
    if (!reshape_is_bitcast_callback_) {
      return ShapeUtil::ReshapeIsBitcast(from_shape, to_shape);
    }
    return reshape_is_bitcast_callback_(from_shape, to_shape);
  }

  // Use the platform specific callback if set. Otherwise, return true.
  bool ConvIsLowerable(HloInstruction* reverse_dims) const {
    if (!conv_is_lowerable_callback_) {
      return true;
    }
    return conv_is_lowerable_callback_(reverse_dims);
  }

  // If is_layout_sensitive is true, then the simplifier preserves layout during
  // transformation. Otherwise, layout is ignored.
  void set_is_layout_sensitive(bool is_layout_sensitive) {
    is_layout_sensitive_ = is_layout_sensitive;
  }

  bool is_layout_sensitive() const { return is_layout_sensitive_; }

  // Enable dot simplification on platforms where it is profitable.
  void set_enable_dot_strength_reduction(bool enable_dot_strength_reduction) {
    enable_dot_strength_reduction_ = enable_dot_strength_reduction;
  }

  bool enable_dot_strength_reduction() const {
    return enable_dot_strength_reduction_;
  }

  // Enable dot->multiple rewrite for dot as an outer-product
  void set_enable_dot_to_multiply_rewrite(bool enable_dot_to_multiply_rewrite) {
    enable_dot_to_multiply_rewrite_ = enable_dot_to_multiply_rewrite;
  }

  bool enable_dot_to_multiply_rewrite() const {
    return enable_dot_to_multiply_rewrite_;
  }

  // Enable convolution simplification on platforms where it is profitable.
  void set_enable_conv_simplification(bool enable_conv_simplification) {
    enable_conv_simplification_ = enable_conv_simplification;
  }
  bool enable_conv_simplification() const {
    return enable_conv_simplification_;
  }

  // Enable convolution operand swapping on platforms where it is supported.
  void set_enable_conv_operand_swap(bool enable_conv_operand_swap) {
    enable_conv_operand_swap_ = enable_conv_operand_swap;
  }
  bool enable_conv_operand_swap() const { return enable_conv_operand_swap_; }

  // Move constant scalar multiply to one operand or output of convolutions with
  // the smallest tensor size, to reduce the number of scalar multiply.
  void set_enable_scalar_multiply_reduction(
      bool enable_scalar_multiply_reduction) {
    enable_scalar_multiply_reduction_ = enable_scalar_multiply_reduction;
  }

  bool enable_scalar_multiply_reduction() const {
    return enable_scalar_multiply_reduction_;
  }

  // Also the algebraic simplifer to treat floating point values like real
  // numbers.
  void set_enable_floats_are_real(bool enable_floats_are_real) {
    enable_floats_are_real_ = enable_floats_are_real;
  }

  bool enable_floats_are_real() const { return enable_floats_are_real_; }

  // If enable_window_reduce_replacement is true, the kReduceWindow instruction
  // can be optimized by replacement with simpler operations.
  void set_enable_window_reduce_to_reduce_replacement(
      bool enable_window_reduce_to_reduce_replacement) {
    enable_window_reduce_to_reduce_replacement_ =
        enable_window_reduce_to_reduce_replacement;
  }

  bool enable_window_reduce_to_reduce_replacement() const {
    return enable_window_reduce_to_reduce_replacement_;
  }

  // Sets the size of a gather operand that can be unrolled into many selects.
  void set_very_small_gather_size(int64_t size) {
    very_small_gather_size_ = size;
  }

  int64_t very_small_gather_size() const { return very_small_gather_size_; }

  void set_cudnn_batchnorm_forward_training_metadata(const std::string& c) {
    metadata_.cudnn_batchnorm_forward_training_metadata = c;
  }

  const std::string& get_cudnn_batchnorm_forward_training_metadata() const {
    return metadata_.cudnn_batchnorm_forward_training_metadata;
  }

  void set_enable_reduce_of_reshape(bool enable_reduce_of_reshape) {
    enable_reduce_of_reshape_ = enable_reduce_of_reshape;
  }

  bool enable_reduce_of_reshape() const { return enable_reduce_of_reshape_; }

  void set_enable_negative_padding_replacement(
      bool enable_negative_padding_replacement) {
    enable_negative_padding_replacement_ = enable_negative_padding_replacement;
  }

  bool enable_negative_padding_replacement() const {
    return enable_negative_padding_replacement_;
  }

  void set_enable_sink_broadcast(bool enable_sink_broadcast) {
    enable_sink_broadcast_ = enable_sink_broadcast;
  }

  bool enable_sink_broadcast() const { return enable_sink_broadcast_; }

  void set_replace_transpose_with_bitcast(bool replace_transpose_with_bitcast) {
    replace_transpose_with_bitcast_ = replace_transpose_with_bitcast;
  }

  bool replace_transpose_with_bitcast() const {
    return replace_transpose_with_bitcast_;
  }

  // If true, min(x, NaN) = NaN.  If false, min(x, NaN) = x.
  //
  // TODO(b/209827141): Remove this and make minmax_propagate_nan uncondtionally
  // true.
  bool minmax_propagate_nan() const { return minmax_propagate_nan_; }
  void set_minmax_propagate_nan(bool val) { minmax_propagate_nan_ = val; }

 private:
  // Metadata struct can be used to store any metadata information encapsulated
  // with the AlgebraicSimplierOptions that can be later used in an
  // AlgebraicSimplifier pass. For example,
  // cudnn_batchnorm_forward_training_metadata can be used to store the name of
  // a custom call. If the custom call is
  // __cudnn$batchNormalizationForwardTraining, the output with index 2 is
  // guaranteed to be postive. This property has been used to recursively
  // determine if the operand of an instruction is always positive.
  struct Metadata {
    std::string cudnn_batchnorm_forward_training_metadata{""};
    Metadata() {}
  };
  ReshapeIsBitcastCallback reshape_is_bitcast_callback_;
  ConvIsLowerableCallback conv_is_lowerable_callback_;
  bool is_layout_sensitive_{false};
  bool enable_dot_strength_reduction_{true};
  bool enable_dot_to_multiply_rewrite_{true};
  bool enable_conv_simplification_{true};
  bool enable_conv_operand_swap_{true};
  bool enable_scalar_multiply_reduction_{false};
  bool enable_floats_are_real_{false};
  bool enable_window_reduce_to_reduce_replacement_{true};
  bool enable_reduce_of_reshape_{true};
  bool enable_negative_padding_replacement_{true};
  bool enable_sink_broadcast_{true};
  bool replace_transpose_with_bitcast_{true};
  int64_t very_small_gather_size_{4};
  bool minmax_propagate_nan_{true};
  Metadata metadata_;
};

// A pass which performs algebraic simplifications.
class AlgebraicSimplifier : public HloModulePass {
 public:
  // If is_layout_sensitive is true, then the simplifier preserves layout during
  // transformation. Otherwise, layout is ignored.
  explicit AlgebraicSimplifier(const AlgebraicSimplifierOptions& options)
      : options_(options) {}
  ~AlgebraicSimplifier() override = default;
  absl::string_view name() const override { return "algsimp"; }

  // Run algebraic simplification on the given computation. Returns whether the
  // computation was changed.
  StatusOr<bool> Run(HloModule* module) override;

  // Create constant from literal with tiles and element size updated in the
  // constant's layout.
  std::unique_ptr<HloInstruction> CreateConstantWithLayoutUpdated(
      Literal literal) {
    auto constant = HloInstruction::CreateConstant(std::move(literal));
    UpdateLayout(constant->mutable_shape());
    return constant;
  }

 protected:
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

  Status HandleAbs(HloInstruction* abs) override;

  Status HandleAdd(HloInstruction* add) override;

  Status HandleAnd(HloInstruction* logical_and) override;

  Status HandleBitcast(HloInstruction* bitcast) override;

  Status HandleBitcastConvert(HloInstruction* bitcast) override;

  Status HandleBroadcast(HloInstruction* broadcast) override;

  Status HandleCompare(HloInstruction* compare) override;

  Status HandleConcatenate(HloInstruction* concatenate) override;

  Status HandleConstant(HloInstruction* constant) override;

  Status HandleCopy(HloInstruction* copy) override;

  Status HandleConvert(HloInstruction* convert) override;

  Status HandleComplex(HloInstruction* complex) override;

  Status HandleReal(HloInstruction* real) override;

  Status HandleImag(HloInstruction* imag) override;

  Status HandleIota(HloInstruction* instruction) override;

  Status HandleConvolution(HloInstruction* convolution) override;

  Status HandleDivide(HloInstruction* divide) override;

  Status HandleDot(HloInstruction* dot) override;

  Status HandleGather(HloInstruction* gather) override;

  Status HandleGetTupleElement(HloInstruction* get_tuple_element) override;

  Status HandleLog(HloInstruction* log) override;

  Status HandleMaximum(HloInstruction* maximum) override;

  Status HandleMinimum(HloInstruction* minimum) override;

  Status HandleClamp(HloInstruction* clamp) override;

  Status HandleMultiply(HloInstruction* multiply) override;

  Status HandleNegate(HloInstruction* negate) override;

  Status HandleNot(HloInstruction* logical_not) override;

  Status HandleOptimizationBarrier(HloInstruction* barrier) override;

  Status HandleOr(HloInstruction* logical_or) override;

  Status HandlePad(HloInstruction* pad) override;

  Status HandlePower(HloInstruction* power) override;

  Status HandleRemainder(HloInstruction* remainder) override;

  Status HandleReshape(HloInstruction* reshape) override;

  Status HandleReduce(HloInstruction* hlo) override;

  Status HandleReduceWindow(HloInstruction* hlo) override;

  Status HandleReverse(HloInstruction* reverse) override;

  Status HandleRsqrt(HloInstruction* rsqrt) override;

  Status HandleSlice(HloInstruction* slice) override;

  Status HandleSqrt(HloInstruction* sqrt) override;

  Status HandleDynamicSlice(HloInstruction* dynamic_slice) override;

  Status HandleDynamicUpdateSlice(
      HloInstruction* dynamic_update_slice) override;
  Status HandleScatter(HloInstruction* scatter) override;

  Status HandleSelect(HloInstruction* select) override;

  Status HandleSort(HloInstruction* sort) override;

  Status HandleTranspose(HloInstruction* transpose) override;

  Status HandleSubtract(HloInstruction* sub) override;

  Status HandleMap(HloInstruction* map) override;

  // Runs the visitor on a computation.
  bool Run(HloComputation* computation,
           const AlgebraicSimplifierOptions& options,
           AlgebraicSimplifier* simplifier);

  // Compute a function that maps from bitcasted dimensions to the resulting
  // ones. Returns the function as a vector if successful; absl::optional
  // otherwise.
  static absl::optional<std::vector<std::vector<int64_t>>> ComputeBitcastDimMap(
      const Shape& bitcast_shape, const Shape& operand_shape);
  // Invert the directions of the given bitcast dimension map.
  static std::vector<std::vector<int64_t>> InvertBitcastDimMap(
      const Shape& original_shape, const Shape& bitcast_shape,
      const std::vector<std::vector<int64_t>>& original_map);

  // Modify the layout dimensions of result_shape, so that it becomes the
  // re-shaped result of applying bitcast to the original_shape, by using
  // dim_map to re-shape layout dimensions of original_shape. Returns the
  // result_shape with modified layout if the conversion succeeds; Returns
  // absl::nullopt if fails.
  static absl::optional<Shape> ReshapeLayoutDimensions(
      const Shape& original_shape, const Shape& result_shape,
      const std::vector<std::vector<int64_t>>& original_map,
      const std::vector<std::vector<int64_t>>& result_map);

  // Allow backend constraints on tiling etc. to invalidate optimizations.
  virtual bool IsValidLayout(const Shape& shape) { return true; }

 protected:
  // The backend-specific options selected for the algebraic simplifier.
  const AlgebraicSimplifierOptions& options_;

 private:
  // Removes degenerate dimension from dot.
  StatusOr<bool> RemoveDegenerateDimensionFromDot(HloInstruction* dot);

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
  StatusOr<HloInstruction*> NormalizeDotOperandToBatchMajorAndContractingMinor(
      HloInstruction* dot_operand, absl::Span<const int64_t> batch_dimensions,
      absl::Span<const int64_t> contracting_dimensions);

  // Simplify dot(transpose(a), transpose(b)) to transpose(dot(b,a)) (or
  // transpose(dot(a,b)) if only the batch dims are transposed).
  //
  // Requires the dot has been canonicalized by DotDecomposer into
  //
  //   LHS [batch dims..., non-contracting dim, contracting dim]
  //   RHS [batch dims..., contracting dim, non-contracting dim].
  StatusOr<bool> RemoveTransposesFromDotOperands(HloInstruction* dot);

  // Helper method to perform and add reduction on a list of dimensions.
  HloInstruction* AddReduce(HloInstruction* hlo, absl::Span<const int64_t> dims,
                            PrimitiveType type);

  // Move scalar multiply to the smallest side of convolution to
  // reduce multiply computations.
  Status ScalarMultiplyReduction(HloInstruction* dot);

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

  // Returns whether the shape of the output of the given instructions are the
  // same for the purposes of simplification. If options_.is_layout_sensitive()
  // is true, then this tests shape equality including layout
  // (ShapeUtil::Equal). If options_.is_layout_sensitive() is false, then the
  // tests shape compatibility (ShapeUtil::Compatible).
  bool SameShape(const HloInstruction* lhs, const HloInstruction* rhs) const;

  // A Broadcast that feeds an element-wise operation with a unique non-scalar
  // operand can sink to after the operation.
  StatusOr<bool> TryToSinkBroadcastAfterOpWithUniqueNonScalarOperand(
      HloInstruction* broadcast);

  StatusOr<HloInstruction*> OptimizeDotOfConcat(HloInstruction* dot);
  StatusOr<HloInstruction*> OptimizeDotOfConcatHelper(
      HloInstruction* dot, HloInstruction* lhs, int64_t lhs_contracting_dim,
      HloInstruction* rhs, int64_t rhs_contracting_dim, bool swapped);

  StatusOr<HloInstruction*> OptimizeDotOfGather(HloInstruction* dot);

  StatusOr<HloInstruction*> OptimizeDotOfReorderContractingDims(
      HloInstruction* dot);

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
  virtual StatusOr<bool> FoldConvInputPad(HloInstruction* convolution);
  StatusOr<bool> FoldConvFilterPad(HloInstruction* convolution);

  // Tries to swap convolution operands if they would result in a more efficient
  // convolution.
  StatusOr<bool> SwapConvOperands(HloInstruction* convolution);

  // Tries to use a kDot in place of the given convolution.
  StatusOr<bool> SimplifyConvToDot(HloInstruction* convolution);

  // Tries to simplify a slice where the result of the slice is a scalar.
  StatusOr<bool> TrySimplifyScalarSlice(HloInstruction* slice);

  // Tries to convert slice(reshape(X)) into reshape(slice(X))
  StatusOr<bool> TryToReorderSliceAndReshape(HloInstruction* slice);

  // Tries to convert slice(reverse(X)) into reverse(slice(X))
  StatusOr<bool> TryToReorderSliceAndReverse(HloInstruction* slice);

  // Tries to simplify `(and (< a N) (< a K))` in cases where `N <= K` into
  // `(< a N)`. This is crucial for being able to figure out the loop trip
  // count.
  //
  // Assumes that the input is conjunction.
  StatusOr<bool> TrySimplifyTautologicalCompare(HloInstruction* conjunction);

  // Tries to simlplify (bitcast-convert (concat (bitcast-convert A) ...)) where
  // the types of inner and outer bitcast-convert cancel out.
  StatusOr<bool> TrySimplifyTautologicalBitcastConvert(HloInstruction* bitcast);

  // Useful when we want to use the same visitor over multiple computations.
  void ResetState(HloComputation* computation);

  // Current HloComputation instance the AlgebraicSimplifierVisitor is
  // traversing.
  HloComputation* computation_;

  // Cached computation for adding two scalars of a given type.
  absl::flat_hash_map<PrimitiveType, HloComputation*> scalar_add_computations_;

  AlgebraicSimplifier* simplifier_ = nullptr;
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_ALGEBRAIC_SIMPLIFIER_H_

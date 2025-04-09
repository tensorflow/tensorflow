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
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/dfs_hlo_visitor_with_default.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/pass/hlo_pass_interface.h"
#include "xla/literal.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

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

  void set_conv_is_lowerable_callback(
      ConvIsLowerableCallback conv_is_lowerable_callback) {
    conv_is_lowerable_callback_ = std::move(conv_is_lowerable_callback);
  }

  // If is_layout_sensitive is true, then the simplifier preserves layout during
  // transformation. Otherwise, layout is ignored.
  void set_is_layout_sensitive(bool is_layout_sensitive) {
    is_layout_sensitive_ = is_layout_sensitive;
  }

  bool is_layout_sensitive() const { return is_layout_sensitive_; }

  void set_use_associative_reordering(bool use_associative_reordering) {
    use_associative_reordering_ = use_associative_reordering;
  }

  bool use_associative_reordering() const {
    return use_associative_reordering_;
  }

  void set_associative_reordering_threshold(
      double associative_reordering_threshold) {
    associative_reordering_threshold_ = associative_reordering_threshold;
  }

  double associative_reordering_threshold() const {
    return associative_reordering_threshold_;
  }

  void set_use_convert_constant_folding(bool use_convert_constant_folding) {
    use_convert_constant_folding_ = use_convert_constant_folding;
  }

  bool use_convert_constant_folding() const {
    return use_convert_constant_folding_;
  }

  void set_raise_slice_and_reduce_through_dot(
      bool raise_slice_and_reduce_through_dot) {
    raise_slice_and_reduce_through_dot_ = raise_slice_and_reduce_through_dot;
  }

  bool raise_slice_and_reduce_through_dot() const {
    return raise_slice_and_reduce_through_dot_;
  }

  void set_raise_slice_and_reduce_through_dot_threshold(
      double raise_slice_and_reduce_through_dot_threshold) {
    raise_slice_and_reduce_through_dot_threshold_ =
        raise_slice_and_reduce_through_dot_threshold;
  }

  double raise_slice_and_reduce_through_dot_threshold() const {
    return raise_slice_and_reduce_through_dot_threshold_;
  }

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

  void set_enable_move_dot_param_to_rhs(bool enable_move_dot_param_to_rhs) {
    enable_move_dot_param_to_rhs_ = enable_move_dot_param_to_rhs;
  }

  bool enable_move_dot_param_to_rhs() const {
    return enable_move_dot_param_to_rhs_;
  }

  // This platform will not run the DotDecomposer to canonicalize dots.
  void set_supports_non_canonical_dots(bool supports_non_canonical_dots) {
    supports_non_canonical_dots_ = supports_non_canonical_dots;
  }
  bool supports_non_canonical_dots() const {
    return supports_non_canonical_dots_;
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

  // Enable rewrite of convolution + add + multiply -> multiply + convolution +
  // add.
  void set_enable_conv_add_multiply_reorder(
      bool enable_conv_add_multiply_reorder) {
    enable_conv_add_multiply_reorder_ = enable_conv_add_multiply_reorder;
  }

  bool enable_conv_add_multiply_reorder() const {
    return enable_conv_add_multiply_reorder_;
  }

  // Move constant scalar multiply to one operand or output of convolutions with
  // the smallest tensor size, to reduce the number of scalar multiply.
  void set_enable_scalar_multiply_reduction(
      bool enable_scalar_multiply_reduction) {
    enable_scalar_multiply_reduction_ = enable_scalar_multiply_reduction;
  }

  bool enable_scalar_multiply_reduction() const {
    return enable_scalar_multiply_reduction_;
  }

  // Set the algebraic simplifier to treat floats as real numbers.
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

  // If true, always simplify reduce(transpose(x)) and reduce(reshape(x)), even
  // if the transpose/reshape has multiple users.  This can be beneficial
  // on platforms where the extra transpose/reshape isn't as expensive as
  // the optimization benefits brought about by simplifying the graph.
  bool unconditionally_simplify_reduce_of_transpose_or_reshape() const {
    return unconditionally_simplify_reduce_of_transpose_or_reshape_;
  }
  void set_unconditionally_simplify_reduce_of_transpose_or_reshape(bool val) {
    unconditionally_simplify_reduce_of_transpose_or_reshape_ = val;
  }

  // If true, min(x, NaN) = NaN.  If false, min(x, NaN) = x.
  //
  // TODO(b/209827141): Remove this and make minmax_propagate_nan
  // unconditionally true.
  bool minmax_propagate_nan() const { return minmax_propagate_nan_; }
  void set_minmax_propagate_nan(bool val) { minmax_propagate_nan_ = val; }

  // When true, always replaces Reduce(concat({a,b,...})) with
  // map(reduce(a),map(reduce(b),...,)). If false, only does the replacement if
  // the shapes of a,b,... have the same dimensions.
  bool enable_unconditional_reduce_of_concat_replacement() const {
    return enable_unconditional_reduce_of_concat_replacement_;
  }
  void set_enable_unconditional_reduce_of_concat_replacement(
      bool enable_unconditional_reduce_of_concat_replacement) {
    enable_unconditional_reduce_of_concat_replacement_ =
        enable_unconditional_reduce_of_concat_replacement;
  }

  // Indicates whether running on CPU
  bool executing_on_cpu() const { return executing_on_cpu_; }
  void set_executing_on_cpu(bool executing_on_cpu) {
    executing_on_cpu_ = executing_on_cpu;
  }

  // Option to disable conversion of dynamic-slice to slice.
  void set_disable_dynamic_slice_to_slice_conversion(bool disable) {
    disable_dynamic_slice_to_slice_conversion_ = disable;
  }
  bool disable_dynamic_slice_to_slice_conversion() const {
    return disable_dynamic_slice_to_slice_conversion_;
  }

  // Option to set finite math.
  void set_enable_fast_math(bool enable_fast_math) {
    enable_fast_math_ = enable_fast_math;
  }
  bool enable_fast_math() const { return enable_fast_math_; }

  void set_enable_broadcast_degenerate_dimension(
      bool enable_broadcast_degenerate_dimension) {
    enable_broadcast_degenerate_dimension_ =
        enable_broadcast_degenerate_dimension;
  }
  bool enable_broadcast_degenerate_dimension() const {
    return enable_broadcast_degenerate_dimension_;
  }

  void set_enable_remove_no_op_reduce_precision(
      bool enable_remove_no_op_reduce_precision) {
    enable_remove_no_op_reduce_precision_ =
        enable_remove_no_op_reduce_precision;
  }

  bool enable_remove_no_op_reduce_precision() const {
    return enable_remove_no_op_reduce_precision_;
  }

  bool enable_onednn_support() const { return enable_onednn_support_; }

  void set_enable_onednn_support(bool enable_onednn_support) {
    enable_onednn_support_ = enable_onednn_support;
  }

  bool rewrite_reshape_transpose_as_slice_concatenate() const {
    return rewrite_reshape_transpose_as_slice_concatenate_;
  }

  void set_rewrite_reshape_transpose_as_slice_concatenate(bool value) {
    rewrite_reshape_transpose_as_slice_concatenate_ = value;
  }

 private:
  // Metadata struct can be used to store any metadata information encapsulated
  // with the AlgebraicSimplifierOptions that can be later used in an
  // AlgebraicSimplifier pass. For example,
  // cudnn_batchnorm_forward_training_metadata can be used to store the name of
  // a custom call. If the custom call is
  // __cudnn$batchNormalizationForwardTraining, the output with index 2 is
  // guaranteed to be positive. This property has been used to recursively
  // determine if the operand of an instruction is always positive.
  struct Metadata {
    std::string cudnn_batchnorm_forward_training_metadata{""};
    Metadata() {}
  };
  ReshapeIsBitcastCallback reshape_is_bitcast_callback_;
  ConvIsLowerableCallback conv_is_lowerable_callback_;
  bool is_layout_sensitive_{false};
  bool enable_dot_strength_reduction_{true};
  bool supports_non_canonical_dots_{true};
  bool enable_dot_to_multiply_rewrite_{true};
  bool enable_move_dot_param_to_rhs_{false};
  bool enable_conv_simplification_{true};
  bool enable_conv_operand_swap_{true};
  bool enable_conv_add_multiply_reorder_{false};
  bool enable_scalar_multiply_reduction_{false};
  bool enable_floats_are_real_{false};
  bool enable_window_reduce_to_reduce_replacement_{true};
  bool enable_reduce_of_reshape_{true};
  bool enable_negative_padding_replacement_{true};
  bool enable_sink_broadcast_{true};
  bool unconditionally_simplify_reduce_of_transpose_or_reshape_{false};
  int64_t very_small_gather_size_{4};
  bool minmax_propagate_nan_{true};
  bool enable_unconditional_reduce_of_concat_replacement_{true};
  bool executing_on_cpu_{false};
  bool use_associative_reordering_{false};
  double associative_reordering_threshold_{2.0};
  bool raise_slice_and_reduce_through_dot_{false};
  double raise_slice_and_reduce_through_dot_threshold_{2.0};
  bool use_convert_constant_folding_{false};
  bool disable_dynamic_slice_to_slice_conversion_{false};
  bool enable_fast_math_{false};
  bool enable_broadcast_degenerate_dimension_{true};
  bool enable_remove_no_op_reduce_precision_{false};
  bool enable_onednn_support_{
#ifdef INTEL_MKL
      // Deprecation warning: This config-dependent default value is a temporary
      // measure to preserve existing behavior until downstream users can update
      // their code. The option will default to `false` in a future version;
      // please explicitly call `set_enable_onednn_support(true)` if you depend
      // on it being `true`.
      true
#else   // INTEL_MKL
      false
#endif  // INTEL_MKL
  };
  bool rewrite_reshape_transpose_as_slice_concatenate_{true};
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
  using HloPassInterface::Run;
  absl::StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

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

  absl::Status HandleAbs(HloInstruction* abs) override;

  absl::Status HandleAdd(HloInstruction* add) override;

  absl::Status HandleAllGather(HloInstruction* all_gather) override;

  absl::Status HandleAllToAll(HloInstruction* all_to_all) override;

  absl::Status HandleAnd(HloInstruction* logical_and) override;

  absl::Status HandleBitcast(HloInstruction* bitcast) override;

  absl::Status HandleBitcastConvert(HloInstruction* bitcast) override;

  absl::Status HandleBroadcast(HloInstruction* broadcast) override;

  absl::Status HandleCompare(HloInstruction* compare) override;

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
  virtual bool ShouldStrengthReduceDotToReduce(const HloInstruction* hlo) {
    return true;
  }

 protected:
  // The backend-specific options selected for the algebraic simplifier.
  const AlgebraicSimplifierOptions& options_;

 private:
  // Returns whether the dot precision config is supported by simplifier.
  virtual bool SupportedDotPrecisionConfig(const PrecisionConfig& config,
                                           bool has_contracting_dim);

  // Makes algorithm specific set of instructions for multiply with precision
  // algorithm in mind. In the trivial case it returns just multiply.
  // For x3 or x6 algorithms it adds the parameters split instructions and the
  // corresponding multiply instructions.
  virtual absl::StatusOr<HloInstruction*> MakeMultiplyForPrecisionAlgorithm(
      HloInstruction* dot, HloInstruction* lhs, HloInstruction* rhs);

  // Rewrite dot as mul(broadcast(transpose(x)),broadcast(transpose(y)))
  absl::Status RewriteAsMultiplyDotWithZeroLhsContractingDim(
      HloInstruction* dot, HloInstruction* lhs, HloInstruction* rhs,
      const DotDimensionNumbers& dnums);

  enum class RewriteResult {
    kNoRewrite,
    kRewritten,
    kStopRewrites,
  };

  // Reorder nested dots with associativity using flops as a heuristic
  // Could return kStopRewrites if the rewrite is too expensive.
  absl::StatusOr<RewriteResult> AssociativeReorderNestedDot(
      HloDotInstruction* dot, HloInstruction* lhs, HloInstruction* rhs);

  // If the lhs or rhs have only batch and contracting dimensions, a dot can be
  // rewritten as reduce(mul(broadcast(transpose(x)),broadcast(transpose(y))))
  absl::Status RewriteBatchPlusContractingAsReduce(
      HloDotInstruction* dot, HloInstruction* lhs, HloInstruction* rhs,
      const DotDimensionNumbers& dnums);

  // Removes degenerate dimension from dot.
  absl::StatusOr<bool> RemoveDegenerateDimensionFromDot(HloDotInstruction* dot);

  // Moves the transpose to the broadcast if possible. Can also be called with a
  // bitcast transpose.
  absl::Status SimplifyTransposeOfBroadcast(
      HloInstruction* transpose, absl::Span<const int64_t> dimensions);

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
      absl::Nonnull<HloInstruction*> slice);

  // Current HloComputation instance the AlgebraicSimplifierVisitor is
  // traversing.
  HloComputation* computation_;

  // Cached computation for adding two scalars of a given type.
  absl::flat_hash_map<PrimitiveType, HloComputation*> scalar_add_computations_;

  AlgebraicSimplifier* simplifier_ = nullptr;
};

}  // namespace xla

#endif  // XLA_HLO_TRANSFORMS_SIMPLIFIERS_ALGEBRAIC_SIMPLIFIER_H_

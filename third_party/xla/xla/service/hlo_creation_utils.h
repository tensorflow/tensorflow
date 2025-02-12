/* Copyright 2018 The OpenXLA Authors.

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

#ifndef XLA_SERVICE_HLO_CREATION_UTILS_H_
#define XLA_SERVICE_HLO_CREATION_UTILS_H_

#include <cstddef>
#include <memory>
#include <optional>
#include <vector>

#include "absl/types/span.h"
#include "xla/hlo/builder/xla_computation.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/literal_util.h"
#include "xla/xla_data.pb.h"

namespace xla {

// Some lightweight utilities intended to make HLO instruction creation more
// ergonomic.  We don't have a complete set of helpers yet -- I expect we'll
// expand this interface as needed on an ad-hoc basis.

// Creates a unary HLO instruction and adds it to the computation containing
// `operand`.
absl::StatusOr<HloInstruction*> MakeUnaryHlo(
    HloOpcode opcode, HloInstruction* operand,
    const OpMetadata* metadata = nullptr);

// Creates a binary HLO instruction and adds it to the computation containing
// `lhs` and `rhs` (`lhs` and `rhs` must be in the same computation).
absl::StatusOr<HloInstruction*> MakeBinaryHlo(
    HloOpcode opcode, HloInstruction* lhs, HloInstruction* rhs,
    const OpMetadata* metadata = nullptr,
    const FrontendAttributes* frontend_attributes = nullptr);

// Creates a kCopy HLO.
HloInstruction* MakeCopyHlo(HloInstruction* from, const Shape& to);

// Creates a compare HLO instruction and adds it to the computation containing
// `lhs` and `rhs` (`lhs` and `rhs` must be in the same computation).
absl::StatusOr<HloInstruction*> MakeCompareHlo(
    Comparison::Direction direction, HloInstruction* lhs, HloInstruction* rhs,
    const OpMetadata* metadata = nullptr,
    const FrontendAttributes* frontend_attributes = nullptr);

// Creates a pad HLO instruction and adds it to the computation containing
// `operand` and `padding_value` (`operand` and `padding_value` must be in the
// same computation).
absl::StatusOr<HloInstruction*> MakePadHlo(
    HloInstruction* operand, HloInstruction* padding_value,
    const PaddingConfig& padding_config, const OpMetadata* metadata = nullptr,
    const FrontendAttributes* frontend_attributes = nullptr);

// Creates a slice HLO instruction and adds it to the computation containing
// `operand`.
absl::StatusOr<HloInstruction*> MakeSliceHlo(
    HloInstruction* operand, absl::Span<const int64_t> start_indices,
    absl::Span<const int64_t> limit_indices, absl::Span<const int64_t> strides,
    const OpMetadata* metadata = nullptr,
    const FrontendAttributes* frontend_attributes = nullptr);

// Creates a convolution HLO instruction and adds it to the computation
// containing `lhs` and `rhs` (`lhs` and `rhs` must be in the same computation).
// If the result shape has integral element type, an optional
// preferred_element_type can be specified to override the element type.
absl::StatusOr<HloInstruction*> MakeConvolveHlo(
    HloInstruction* lhs, HloInstruction* rhs, int64_t feature_group_count,
    int64_t batch_group_count, const Window& window,
    const ConvolutionDimensionNumbers& dimension_numbers,
    const PrecisionConfig& precision_config,
    std::optional<PrimitiveType> preferred_element_type,
    const OpMetadata* metadata = nullptr,
    const FrontendAttributes* frontend_attributes = nullptr);

// Creates a transpose HLO instruction and adds it to the computation containing
// `operand`.
absl::StatusOr<HloInstruction*> MakeTransposeHlo(
    HloInstruction* operand, absl::Span<const int64_t> dimensions);

// Creates a reshape HLO instruction and adds it to the computation containing
// `operand`.
absl::StatusOr<HloInstruction*> MakeReshapeHlo(const Shape& result_shape,
                                               HloInstruction* operand);

absl::StatusOr<HloInstruction*> MakeReshapeHlo(
    absl::Span<const int64_t> result_shape_dim_bounds, HloInstruction* operand);

// Creates a dynamic-slice HLO instruction and adds it to the computation
// containing `operand` and `start_indices` (`operand` and `start_indices` must
// be in the same computation).
absl::StatusOr<HloInstruction*> MakeDynamicSliceHlo(
    HloInstruction* operand, absl::Span<HloInstruction* const> start_indices,
    absl::Span<const int64_t> slice_sizes,
    const OpMetadata* metadata = nullptr);
absl::StatusOr<HloInstruction*> MakeDynamicSliceHlo(
    HloInstruction* operand, HloInstruction* start_indices,
    absl::Span<const int64_t> slice_sizes,
    const OpMetadata* metadata = nullptr);

// Creates a dynamic-update-slice HLO instruction and adds it to the computation
// containing `operand`, `update` and `start_indices` (`operand`, `update` and
// `start_indices` must be in the same computation).
absl::StatusOr<HloInstruction*> MakeDynamicUpdateSliceHlo(
    HloInstruction* operand, HloInstruction* update,
    HloInstruction* start_indices, const OpMetadata* metadata = nullptr);

// a variant of dynamic-update-slice where `start_indices` is a vector of HLO
// instructions
absl::StatusOr<HloInstruction*> MakeDynamicUpdateSliceHlo(
    HloInstruction* operand, HloInstruction* update,
    absl::Span<HloInstruction* const> start_indices,
    const OpMetadata* metadata = nullptr);

// Creates a broadcast HLO instruction and adds it to the computation containing
// `operand`.
HloInstruction* MakeBroadcastHlo(
    HloInstruction* operand, absl::Span<const int64_t> broadcast_dimensions,
    absl::Span<const int64_t> result_shape_bounds,
    const OpMetadata* metadata = nullptr,
    const FrontendAttributes* frontend_attributes = nullptr);
HloInstruction* MakeBroadcastHlo(
    HloInstruction* operand, absl::Span<const int64_t> broadcast_dimensions,
    const Shape& shape, const OpMetadata* metadata = nullptr,
    const FrontendAttributes* frontend_attributes = nullptr);

// Creates a GetTupleElement HLO instruction and adds it to the computation
// containing `operand`.
absl::StatusOr<HloInstruction*> MakeGetTupleElementHlo(
    HloInstruction* operand, int64_t index,
    const OpMetadata* metadata = nullptr);

// Creates a Concatenate HLO instruction and adds it to the computation
// containing `operands` (`operands` must be non-empty and every element must be
// contained in the same computation).
absl::StatusOr<HloInstruction*> MakeConcatHlo(
    absl::Span<HloInstruction* const> operands, int64_t dimension,
    const OpMetadata* metadata = nullptr,
    const FrontendAttributes* frontend_attributes = nullptr);

// Creates a Convert HLO instruction that converts the given instruction to have
// the given primitive type.
HloInstruction* MakeConvertToHlo(HloInstruction* hlo, PrimitiveType type,
                                 const OpMetadata* metadata = nullptr);

// Creates a Bitcast HLO instruction to the given shape+layout.
HloInstruction* MakeBitcastHlo(HloInstruction* hlo, const Shape& shape,
                               const OpMetadata* metadata = nullptr);

// Creates a BitcastConvert HLO instruction.
HloInstruction* MakeBitcastConvertToHlo(HloInstruction* hlo, PrimitiveType type,
                                        const OpMetadata* metadata = nullptr);

// Creates an Iota HLO instruction.
HloInstruction* MakeIotaHlo(HloComputation* computation, const Shape& shape,
                            int64_t iota_dimension);

// Creates a Dot HLO instruction and adds it to the computation containing `lhs`
// and `rhs` (both must be in the same computation). If the result shape has
// integral element type, an optional preferred_element_type can be specified to
// override the element type. If 'sparsity' is set, then 'sparse_meta' must also
// be present (and have the same size).
absl::StatusOr<HloInstruction*> MakeDotHlo(
    HloInstruction* lhs, HloInstruction* rhs,
    const DotDimensionNumbers& dim_numbers,
    const PrecisionConfig& precision_config,
    std::optional<PrimitiveType> preferred_element_type,
    std::vector<SparsityDescriptor> sparsity = {},
    absl::Span<HloInstruction* const> sparse_meta = {},
    const OpMetadata* metadata = nullptr);

// Creates a RaggedDot HLO instruction and adds it to the computation containing
// `lhs`, `rhs`, and `group_sizes` (all must be in the same computation). An
// optional preferred_element_type can be specified to override the element
// type.
absl::StatusOr<HloInstruction*> MakeRaggedDotHlo(
    HloInstruction* lhs, HloInstruction* rhs, HloInstruction* group_sizes,
    const RaggedDotDimensionNumbers& dim_numbers,
    const PrecisionConfig& precision_config,
    std::optional<PrimitiveType> preferred_element_type);

// Creates a Map HLO instruction and adds it to the computation containing the
// operands. All operands must be in the same computation.
absl::StatusOr<HloInstruction*> MakeMapHlo(
    absl::Span<HloInstruction* const> operands, HloComputation* map_computation,
    const OpMetadata* metadata = nullptr);

// Creates a reduce-precision op, where operand is the data to reduce in
// precision, and exponent_bits and mantissa_bits describe the precision to
// reduce it to.
HloInstruction* MakeReducePrecisionHlo(HloInstruction* operand,
                                       int exponent_bits, int mantissa_bits,
                                       const OpMetadata* metadata = nullptr);

absl::StatusOr<HloInstruction*> MakeReduceWindowHlo(
    HloInstruction* operand, HloInstruction* init_value, const Window& window,
    HloComputation* reduce_computation, const OpMetadata* metadata = nullptr);

absl::StatusOr<HloInstruction*> MakeReduceWindowHlo(
    HloInstruction* operand, HloInstruction* init_value, const Window& window,
    HloOpcode binary_opcode, const OpMetadata* metadata = nullptr);

// Creates a Reduce HLO instruction and adds it to the computation containing
// the operand. This will create the sub-computation needed for the reduction in
// the given module. binary_opcode should represent a binary operation.
absl::StatusOr<HloInstruction*> MakeReduceHlo(
    HloInstruction* operand, HloInstruction* init_value,
    absl::Span<const int64_t> dimensions, HloOpcode binary_opcode,
    const OpMetadata* metadata = nullptr,
    const FrontendAttributes* frontend_attributes = nullptr);

absl::StatusOr<HloInstruction*> MakeReduceHlo(
    HloInstruction* operand, HloInstruction* init_value,
    absl::Span<const int64_t> dimensions, HloComputation* reduce_computation,
    const OpMetadata* metadata = nullptr,
    const FrontendAttributes* frontend_attributes = nullptr);

absl::StatusOr<HloInstruction*> MakeReduceHlo(
    HloInstruction* operand, HloInstruction* init_value,
    HloOpcode binary_opcode, HloModule* module,
    const OpMetadata* metadata = nullptr,
    const FrontendAttributes* frontend_attributes = nullptr);

// Generic helper function to create a reduction.
//
// Precondition: size of operands is equal to the size of init values and equal
// to the size of the computation output shape.
//
// Creates a non-variadic reduction if the size is singular, and a variadic one
// otherwise.
absl::StatusOr<HloInstruction*> MakeReduceHlo(
    absl::Span<HloInstruction* const> operands,
    absl::Span<HloInstruction* const> init_values,
    absl::Span<const int64_t> dimensions, HloComputation* reduce_computation,
    const OpMetadata* metadata = nullptr,
    const FrontendAttributes* frontend_attributes = nullptr);

// Creates a Reverse HLO instruction and adds it to the computation containing
// `operand`.
absl::StatusOr<HloInstruction*> MakeReverseHlo(
    HloInstruction* operand, absl::Span<const int64_t> dimensions,
    const OpMetadata* metadata = nullptr);

// Creates a Select HLO instruction and adds it to the computation containing
// the predicate. The on_true and on_false instructions must also be contained
// in the same computation. If on_true and on_false are tuples, create a tuple
// select instead. `pred` is broadcasted up from a scalar if necessary.
absl::StatusOr<HloInstruction*> MakeSelectHlo(
    HloInstruction* pred, HloInstruction* on_true, HloInstruction* on_false,
    HloInstruction* derived_from = nullptr,
    const OpMetadata* metadata = nullptr,
    const FrontendAttributes* frontend_attributes = nullptr);

// Forwards the first operand if operands.size() == 1, or creates a tuple
// instruction with all the operands. Crashes if `operands` is empty.
HloInstruction* MaybeMakeTuple(absl::Span<HloInstruction* const> operands);

// Creates a HloComputation in the destination module from a builder's
// XlaComputation.
absl::StatusOr<HloComputation*> XlaComputationToHloComputation(
    XlaComputation& src_comp, HloModule* dest_module);

// Creates a Sort HLO instruction and adds it to the computation containing the
// operands. All operands must be in the same computation. Also creates a
// default compare sub-computation which sorts the first operand into ascending
// order. 'is_stable' specifies whether the sorting should be stable.
absl::StatusOr<HloInstruction*> MakeSortHlo(
    const Shape& sort_shape, absl::Span<HloInstruction* const> operands,
    int64_t dimension_to_sort, bool is_stable, HloComputation::Builder* builder,
    HloModule* module, const OpMetadata* metadata = nullptr);

// Creates an R1 Constant HLO instruction of the given PrimitiveType with the
// given values and adds it to the given computation.
template <typename NativeT>
absl::StatusOr<HloInstruction*> MakeR1ConstantHlo(
    HloComputation* computation, PrimitiveType type,
    absl::Span<const NativeT> values) {
  Literal literal = LiteralUtil::CreateR1<NativeT>(values);
  if (literal.shape().element_type() != type) {
    TF_ASSIGN_OR_RETURN(literal, literal.Convert(type));
  }
  return computation->AddInstruction(
      HloInstruction::CreateConstant(std::move(literal)));
}

// Creates an R0 Constant HLO instruction of the PrimitiveType corresponding to
// `NativeT` with the given value and adds it to the given computation.
template <class NativeT>
HloInstruction* MakeR0ConstantHlo(HloComputation* computation, NativeT value) {
  return computation->AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<NativeT>(value)));
}

// Makes a scalar that is elementwise compatible with the shape of the base
// instruction.
template <class NativeT>
HloInstruction* MakeScalarLike(HloInstruction* base, NativeT value) {
  auto scalar = base->AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<NativeT>(value)
                                         .Convert(base->shape().element_type())
                                         .value()));
  if (base->shape().rank() == 0) {
    *scalar->mutable_shape() = base->shape();
    return scalar;
  }
  return base->AddInstruction(HloInstruction::CreateBroadcast(
      ShapeUtil::MakeStaticShape(base->shape()), scalar, {}));
}

// Creates a fusion instruction and fuses `fused` into the created fusion
// instruction.
absl::StatusOr<HloInstruction*> MakeFusionInstruction(
    HloInstruction* fused, HloInstruction::FusionKind kind);

// -----------------------------------------------------------------------------
// Some other miscellaneous helpers to generate common HLO patterns.  All of
// these add all the instructions they generate into the computation containing
// their operand(s).

// Collapses (via reshape) the first N (logical) dimensions of `operand` into a
// single leading dimension.  `operand` must have rank > `n` and `n` must not be
// 0.
//
// For instance if `operand` has shape f32[7,8,9] and n is 2 then the output is
// the `operand` reshaped to [56,9].
absl::StatusOr<HloInstruction*> CollapseFirstNDims(HloInstruction* operand,
                                                   int64_t n);

// Prepends `n` degenerate dimensions (dimensions with bound = 1) to `operand`
// using a reshape.
//
// For instance if operand has shape f32[3,4,5] then this returns the operand
// reshaped to f32[1,3,4,5].  If the operand is a f32 scalar (i.e. has shape
// f32[]) then this returns the operand reshaped to f32[1].
absl::StatusOr<HloInstruction*> PrependDegenerateDims(HloInstruction* operand,
                                                      int64_t n);

// Expands (via reshape) the first (logical) dimension of `operand` into a
// sequence of `expanded_dims` dimensions.  `operand` must at least be of rank 1
// and the number of elements in its first dimension must be equal to the
// product of `expanded_dims`.
//
// For instance if `operand` has shape f32[200,9,7] and expanded_dims is
// {2,5,20} the result is `operand` reshaped to [2,5,20,9,7].
absl::StatusOr<HloInstruction*> ExpandFirstDimIntoNDims(
    HloInstruction* operand, absl::Span<const int64_t> expanded_dims);

// Elides (via reshape) a set of degenerate dimensions (dimensions containing
// exactly one element), `dims_to_elide` from `operand`.  Every dimension in
// `dims_to_elide` must be a degenerate dimension.  `dims_to_elide` must be
// sorted and not contain duplicates.
//
// For example if `operand` is of shape f32[19,1,20,1,7,1,9] and dims_to_elide
// is {1,5} then the result is `operand` reshaped to [19,20,1,7,9].
absl::StatusOr<HloInstruction*> ElideDegenerateDims(
    HloInstruction* operand, absl::Span<const int64_t> dims_to_elide);

// Inserts (via reshape) a set of degenerate dimensions (dimensions containing
// exactly one element), `dims_to_insert` into `operand`. The dimensions in
// `dims_to_insert` refer to the dimensions in the result, and hence should be
// less than the rank of the result. Also, `dims_to_insert` must be sorted.
//
// For example, if `operand` is of shape f32[12,21,8,34] and dims_to_insert is
// {0, 2}, then the result is `operand` reshaped to [1,12,1,21,8,34].
absl::StatusOr<HloInstruction*> InsertDegenerateDims(
    HloInstruction* operand, absl::Span<const int64_t> dims_to_insert);

// Pads `operand` (which must have rank 1) with `zeros_to_prepend` zeros in the
// front and `zeros_to_append` zeros in the back.
absl::StatusOr<HloInstruction*> PadVectorWithZeros(HloInstruction* operand,
                                                   int64_t zeros_to_prepend,
                                                   int64_t zeros_to_append);

// Broadcasts a zero value of type `element_type` into a tensor with element
// type `element_type` and dimension bounds `broadcast_dimensions`.  The
// broadcast instruction is emitted into `computation`.
HloInstruction* BroadcastZeros(HloComputation* computation,
                               PrimitiveType element_type,
                               absl::Span<const int64_t> broadcast_dimensions);

// Same as above, but allows to specify the broadcast shape.
HloInstruction* BroadcastZeros(HloComputation* computation,
                               const Shape& broadcast_shape);

// Same as above, but fill the tensor with ones.
HloInstruction* BroadcastOnes(HloComputation* computation,
                              PrimitiveType element_type,
                              absl::Span<const int64_t> broadcast_dimensions);

// Creates a HLO computation that takes arguments of type `domain` and produces
// a value of type `range`.
absl::StatusOr<std::unique_ptr<HloComputation>> CreateComputationWithSignature(
    absl::Span<const Shape* const> domain, const Shape& range,
    absl::string_view name);

// Expands a general degenerate reshape operation to a sequence of degenerate
// adding and removing reshapes that changes only a single dimension.
HloInstruction* ExpandDegenerateReshape(HloInstruction* inst);

// Creates a scalar constant with the given shape and native value.
template <typename NativeT>
std::unique_ptr<HloInstruction> MakeScalarConstantWithShape(const Shape& shape,
                                                            NativeT value) {
  return primitive_util::PrimitiveTypeSwitch<std::unique_ptr<HloInstruction>>(
      [&](auto literal_constant) -> std::unique_ptr<HloInstruction> {
        if constexpr (primitive_util::IsIntegralType(literal_constant) ||
                      primitive_util::IsFloatingPointType(literal_constant)) {
          auto constant = HloInstruction::CreateConstant(
              LiteralUtil::CreateR0<NativeT>(value)
                  .Convert(shape.element_type())
                  .value());
          *constant->mutable_shape() = shape;
          return std::move(constant);
        }
        LOG(FATAL) << "Provided shape is not a float or int type.";
      },
      shape.element_type());
}

}  // namespace xla

#endif  // XLA_SERVICE_HLO_CREATION_UTILS_H_

/* Copyright 2022 The OpenXLA Authors.

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

#include "xla/service/layout_normalization.h"

#include <cstdint>
#include <optional>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/dfs_hlo_visitor_with_default.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/layout.h"
#include "xla/layout_util.h"
#include "xla/literal.h"
#include "xla/permutation_util.h"
#include "xla/service/hlo_creation_utils.h"
#include "xla/service/shape_inference.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace {

// Layout normalization visitor. Aims to achieve the global postcondition that
// every layout is strictly descending (the layout permutation is effectively
// applied to the shape itself).
//
// Local precondition for every call:
//    -> Input is a bitcast from a normalized layout.
//
// Local postcondition:
//    -> Input and output of a processed operation have descending layout*
//
// *: For current fusion limitations this is currently not applicable to
// unnested reductions only.
class LayoutNormalizationVisitor : public DfsHloRewriteVisitor {
 public:
  explicit LayoutNormalizationVisitor(
      LayoutNormalization* normalization,
      const CustomCallTransformer& custom_call_transformer = nullptr)
      : normalization_(normalization),
        custom_call_transformer_(custom_call_transformer) {}

  // To handle a constant, just give the literal data a new layout.
  absl::Status HandleConstant(HloInstruction* hlo) override {
    Literal& literal = *Cast<HloConstantInstruction>(hlo)->mutable_literal();
    if (literal.shape().IsTuple()) {
      // TODO(cheshire): Tuple constants.
      return absl::OkStatus();
    }

    const Shape& shape = hlo->shape();
    Shape normalized_shape = Normalize(shape);
    *literal.mutable_shape_do_not_use() = normalized_shape;
    // Ensure element_size_in_bits of literal is 0, because literals do not
    // support packed values.
    literal.mutable_shape_do_not_use()
        ->mutable_layout()
        ->set_element_size_in_bits(0);

    HloInstruction* bc_to_orig = MakeBitcastHlo(hlo, shape);
    *hlo->mutable_shape() = normalized_shape;
    TF_RETURN_IF_ERROR(hlo->ReplaceAllUsesWithDifferentShape(bc_to_orig));
    MarkAsChanged();
    return absl::OkStatus();
  }

  // Slice is layout-preserving, so handling is analoguous to elementwise unary,
  // and transposing the elements inside the metadata.
  absl::Status HandleSlice(HloInstruction* hlo) override {
    HloInstruction* operand = hlo->mutable_operand(0);
    const Shape& s = hlo->shape();
    const Shape& operand_shape = operand->shape();
    TF_RET_CHECK(s.layout() == operand_shape.layout());
    TF_ASSIGN_OR_RETURN(HloInstruction * normalized_input,
                        GetNormalizedInput(operand));

    Shape normalized = Normalize(operand_shape);
    std::vector<int64_t> layout_as_permutation =
        ToTransposeDimensions(hlo->shape().layout());

    auto normalize_slice_attr = [&](absl::Span<int64_t const> input) {
      return Permute(input, layout_as_permutation);
    };

    TF_ASSIGN_OR_RETURN(HloInstruction * normalized_slice,
                        MakeSliceHlo(normalized_input,
                                     normalize_slice_attr(hlo->slice_starts()),
                                     normalize_slice_attr(hlo->slice_limits()),
                                     normalize_slice_attr(hlo->slice_strides()),
                                     &hlo->metadata()));
    *normalized_slice->mutable_shape()->mutable_layout() =
        normalized_input->shape().layout();
    SetVisited(*normalized_slice);
    HloInstruction* bc_to_orig = MakeBitcastHlo(normalized_slice, s);
    TF_RETURN_IF_ERROR(ReplaceInstruction(hlo, bc_to_orig));
    return absl::OkStatus();
  }

  // Default action: ensure local postcondition that any input is always a
  // bitcast from canonical layout for any rewrites of the HLO users.
  //
  // Bitcast to descending layout and then bitcast back to make sure that shapes
  // match.
  absl::Status DefaultAction(HloInstruction* hlo) override {
    if (!hlo->user_count()) {
      // The local postcondition does not have to apply to the case when there
      // are no users.
      return absl::OkStatus();
    }
    auto users = hlo->users();
    auto shape = hlo->shape();
    if (shape.IsTuple() || shape.IsToken()) {
      // GTEs will be transformed individually, tokens should be skipped.
      return absl::OkStatus();
    }

    auto normalized_shape = Normalize(shape);
    auto bc_to_normalized = MakeBitcastHlo(hlo, normalized_shape);
    SetVisited(*bc_to_normalized);
    auto bc_to_orig = MakeBitcastHlo(bc_to_normalized, shape);
    TF_RETURN_IF_ERROR(hlo->ReplaceUsesWith(users, bc_to_orig));
    MarkAsChanged();
    return absl::OkStatus();
  }

  // Converts concatenation to normalized layout.
  //
  // With respect to layouts, concatenations are simple, as they are
  // layout-preserving.
  absl::Status HandleConcatenate(HloInstruction* hlo) override {
    const Shape& s = hlo->shape();
    int64_t orig_concat_dim = hlo->dimensions(0);

    std::vector<HloInstruction*> normalized_inputs;
    for (HloInstruction* operand : hlo->mutable_operands()) {
      TF_ASSIGN_OR_RETURN(auto normalized_input, GetNormalizedInput(operand));
      normalized_inputs.push_back(normalized_input);
    }
    auto normalized_shape = Normalize(s);
    auto layout_as_permutation = ToTransposeDimensions(s.layout());
    int64_t normalized_concat_dim =
        InversePermutation(layout_as_permutation)[orig_concat_dim];
    auto normalized_concat =
        hlo->AddInstruction(HloInstruction::CreateConcatenate(
            normalized_shape, normalized_inputs, normalized_concat_dim));
    SetVisited(*normalized_concat);
    auto bc_to_orig = MakeBitcastHlo(normalized_concat, hlo->shape());
    TF_RETURN_IF_ERROR(ReplaceInstruction(hlo, bc_to_orig));
    return absl::OkStatus();
  }

  absl::Status HandleReduceWindow(HloInstruction* hlo) override {
    if (hlo->shape().IsTuple()) {
      // TODO(cheshire): Handle variadic reductions.
      return absl::OkStatus();
    }

    HloInstruction* operand = hlo->mutable_operand(0);
    TF_RET_CHECK(hlo->shape().layout() == operand->shape().layout());
    TF_ASSIGN_OR_RETURN(HloInstruction * normalized_input,
                        GetNormalizedInput(operand));

    std::vector<int64_t> layout_as_permutation =
        ToTransposeDimensions(hlo->shape().layout());

    std::vector<WindowDimension> window_dimensions;
    for (const WindowDimension& d : hlo->window().dimensions()) {
      window_dimensions.push_back(d);
    }
    window_dimensions = Permute(window_dimensions, layout_as_permutation);

    Window new_window;
    for (const WindowDimension& d : window_dimensions) {
      *new_window.add_dimensions() = d;
    }

    TF_ASSIGN_OR_RETURN(
        HloInstruction * rw,
        MakeReduceWindowHlo(normalized_input, hlo->mutable_operand(1),
                            new_window, hlo->called_computations()[0],
                            &hlo->metadata()));
    normalization_->UpdateLayout(rw->mutable_shape());
    SetVisited(*rw);

    HloInstruction* bc_to_orig = MakeBitcastHlo(rw, hlo->shape());
    TF_RETURN_IF_ERROR(ReplaceInstruction(hlo, bc_to_orig));
    return absl::OkStatus();
  }

  // Converts broadcast input and output to normalized layout.
  //
  // Converts:
  //
  //  A{I} -> bitcast{L} -> broadcast[S]{L'}
  //
  // Into:
  //
  //  A{I} -> broadcast[S']{I} -> bitcast[S]{L'}
  absl::Status HandleBroadcast(HloInstruction* hlo) override {
    VLOG(3) << "Input broadcast: " << hlo->ToString();
    auto s = hlo->shape();
    auto operand = hlo->mutable_operand(0);
    TF_ASSIGN_OR_RETURN(auto normalized_input, GetNormalizedInput(operand));
    auto normalized_shape = Normalize(s);
    std::vector<int64_t> layout_as_permutation =
        ToTransposeDimensions(operand->shape().layout());
    std::vector<int64_t> orig_output_layout_as_permutation =
        ToTransposeDimensions(s.layout());
    std::vector<int64_t> br_dimensions;
    if (!hlo->dimensions().empty()) {
      br_dimensions.reserve(hlo->dimensions().size());
      auto inverse_perm = InversePermutation(orig_output_layout_as_permutation);
      for (int64_t dim :
           ComposePermutations(hlo->dimensions(), layout_as_permutation)) {
        br_dimensions.push_back(inverse_perm[dim]);
      }
    }
    auto normalized_broadcast = MakeBroadcastHlo(
        normalized_input, br_dimensions, normalized_shape, &hlo->metadata());
    SetVisited(*normalized_broadcast);
    VLOG(3) << "Generated broadcast: " << normalized_broadcast->ToString();
    auto bc_to_orig = MakeBitcastHlo(normalized_broadcast, s);
    TF_RETURN_IF_ERROR(ReplaceInstruction(hlo, bc_to_orig));
    return absl::OkStatus();
  }

  absl::Status HandleIota(HloInstruction* hlo) override {
    VLOG(3) << "Input iota: " << hlo->ToString();
    auto s = hlo->shape();
    auto normalized_shape = Normalize(s);
    std::vector<int64_t> orig_output_layout_as_permutation =
        ToTransposeDimensions(s.layout());
    int64_t new_iota_dimension = InversePermutation(
        orig_output_layout_as_permutation)[hlo->dimensions()[0]];
    auto normalized_iota = hlo->AddInstruction(
        HloInstruction::CreateIota(normalized_shape, new_iota_dimension));
    SetVisited(*normalized_iota);
    VLOG(3) << "Generated iota: " << normalized_iota->ToString();
    auto bc_to_orig = MakeBitcastHlo(normalized_iota, s);
    TF_RETURN_IF_ERROR(ReplaceInstruction(hlo, bc_to_orig));
    return absl::OkStatus();
  }

  // BitcastConvert is only layout-preserving if it doesn't change the rank.
  absl::Status HandleBitcastConvert(HloInstruction* hlo) override {
    // If the rank isn't changing this is just an unary op.
    if (hlo->shape().rank() == hlo->operand(0)->shape().rank()) {
      return HandleElementwiseUnary(hlo);
    }

    return DefaultAction(hlo);
  }

  // Pushes down the bitcast across the unary.
  // That is, converts:
  //
  //    H_0{I} -> B{L} -> U{L}
  //
  // into
  //
  //    H_0{I} -> U{I} -> B{L}
  //
  // where {I} denotes default layout.
  absl::Status HandleElementwiseUnary(HloInstruction* hlo) override {
    auto s = hlo->shape();
    auto operand = hlo->mutable_operand(0);
    auto operand_shape = operand->shape();

    // Precondition: elementwise unary leaves layout intact.
    TF_RET_CHECK(
        Layout::Equal().IgnoreElementSize()(s.layout(), operand_shape.layout()))
        << "Unexpected non-layout preserving elementwise unary: "
        << hlo->ToString();
    TF_ASSIGN_OR_RETURN(auto normalized_input, GetNormalizedInput(operand));

    PrimitiveType to_element_type = s.element_type();
    HloInstruction* new_unary;
    if (hlo->opcode() == HloOpcode::kConvert) {
      new_unary =
          MakeConvertToHlo(normalized_input, to_element_type, &hlo->metadata());
    } else if (hlo->opcode() == HloOpcode::kReducePrecision) {
      new_unary =
          MakeReducePrecisionHlo(normalized_input, hlo->exponent_bits(),
                                 hlo->mantissa_bits(), &hlo->metadata());
    } else if (hlo->opcode() == HloOpcode::kBitcastConvert) {
      new_unary = MakeBitcastConvertToHlo(normalized_input, to_element_type,
                                          &hlo->metadata());
    } else {
      TF_ASSIGN_OR_RETURN(
          new_unary,
          MakeUnaryHlo(hlo->opcode(), normalized_input, &hlo->metadata()));
    }
    if (normalized_input != new_unary) {
      // SetVisited() should only be called for unvisited ops.
      // 'normalized_input' is already marked as visited.
      SetVisited(*new_unary);
    }
    auto bc_to_orig = MakeBitcastHlo(new_unary, s);
    TF_RETURN_IF_ERROR(ReplaceInstruction(hlo, bc_to_orig));
    return absl::OkStatus();
  }

  // Pushes down the bitcast across the binary. Converts:
  //
  //  A1{I} -> bitcast{L}
  //            \
  //            B{L}
  //            /
  //  A2{I} -> bitcast{L}
  //
  // Into:
  //
  //  A1{I}
  //        \
  //         B{I} - bitcast{L}
  //        /
  //  A2{I}
  absl::Status HandleElementwiseBinary(HloInstruction* hlo) override {
    auto s = hlo->shape();
    auto a = hlo->mutable_operand(0);
    auto b = hlo->mutable_operand(1);
    auto layout_equal = Layout::Equal();
    if (hlo->opcode() == HloOpcode::kCompare) {
      layout_equal.IgnoreElementSize();
    }
    TF_RET_CHECK(layout_equal(a->shape().layout(), s.layout()));
    TF_ASSIGN_OR_RETURN(auto a0, GetNormalizedInput(a));
    TF_ASSIGN_OR_RETURN(auto b0, GetNormalizedInput(b));

    HloInstruction* new_binary;
    if (hlo->opcode() == HloOpcode::kCompare) {
      TF_ASSIGN_OR_RETURN(new_binary,
                          MakeCompareHlo(hlo->comparison_direction(), a0, b0,
                                         &hlo->metadata()));
    } else {
      TF_ASSIGN_OR_RETURN(
          new_binary, MakeBinaryHlo(hlo->opcode(), a0, b0, &hlo->metadata()));
    }
    SetVisited(*new_binary);
    auto bc_to_orig = MakeBitcastHlo(new_binary, s);
    TF_RETURN_IF_ERROR(ReplaceInstruction(hlo, bc_to_orig));
    return absl::OkStatus();
  }

  // The ReshapeDecomposer already gives us a precondition that a reshape is
  // bitcast. Converts:
  //
  // A{I} -> bitcast [S0]{L1} -> R [S]{L2}
  //
  // Into:
  //
  // A{I} -> R [S']{I} -> bitcast[S]{L2}
  //
  absl::Status HandleReshape(HloInstruction* hlo) override {
    auto s = hlo->shape();
    auto operand = hlo->mutable_operand(0);
    TF_RET_CHECK(ShapeUtil::ReshapeIsBitcast(s, operand->shape()));
    TF_ASSIGN_OR_RETURN(auto a0, GetNormalizedInput(operand));
    auto normalized_reshape_s = Normalize(s);
    TF_ASSIGN_OR_RETURN(auto new_reshape,
                        MakeReshapeHlo(normalized_reshape_s, a0));
    SetVisited(*new_reshape);
    auto bc_to_orig = MakeBitcastHlo(new_reshape, s);
    TF_RETURN_IF_ERROR(ReplaceInstruction(hlo, bc_to_orig));
    return absl::OkStatus();
  }

  // Scatter is layout-preserving regarding the scatter operands, so we only
  // have to permute values inside the ScatterDimensionNumbers.
  absl::Status HandleScatter(HloInstruction* hlo) override {
    auto* scatter = Cast<HloScatterInstruction>(hlo);
    std::vector<HloInstruction*> normalized_operands;
    normalized_operands.reserve(scatter->scatter_operand_count());
    Shape operand_shape = scatter->scatter_operands().front()->shape();
    for (HloInstruction* operand : scatter->scatter_operands()) {
      if (operand->shape().layout() != operand_shape.layout()) {
        return FailedPrecondition(
            "All scatter operands must have the same layout");
      }
      TF_ASSIGN_OR_RETURN(auto normalized_operand, GetNormalizedInput(operand));
      normalized_operands.push_back(normalized_operand);
    }
    std::vector<HloInstruction*> normalized_updates;
    normalized_updates.reserve(scatter->scatter_operand_count());
    Shape update_shape = scatter->scatter_updates().front()->shape();
    for (HloInstruction* operand : scatter->scatter_updates()) {
      if (operand->shape().layout() != update_shape.layout()) {
        return FailedPrecondition(
            "All scatter updates must have the same layout");
      }
      TF_ASSIGN_OR_RETURN(auto normalized_update, GetNormalizedInput(operand));
      normalized_updates.push_back(normalized_update);
    }

    // Since normalization might reorder the 'scatter_updates' operands
    // differently than the 'scatter_indices' update, we have no way to specify
    // the order of 'scatter' (batch) dimensions, as that is not an attribute in
    // ScatterDimensionNumbers. Scatter implicitly assumes that the 'scatter'
    // dimensions appear in the same order in 'scatter_updates' and
    // 'scatter_indices'. So we require that there is just a single
    // 'scatter' dimension. This is ensured by the ScatterSimplifier pass.
    const auto& dims = scatter->scatter_dimension_numbers();
    if (scatter->scatter_updates().front()->shape().rank() -
            dims.update_window_dims_size() >
        1) {
      return FailedPrecondition(
          "There should be just a single scatter dimension. Make sure to run "
          "ScatterSimplifier before LayoutNormalization");
    }
    TF_ASSIGN_OR_RETURN(auto normalized_indices,
                        GetNormalizedInput(scatter->scatter_indices()));

    // The scatter operands are normalized by applying a permutation such that
    // perm(layout) = standard layout -> inverse layout permutation is applied.
    auto indices_permutation = InversePermutation(
        ToTransposeDimensions(scatter->scatter_indices()->shape().layout()));

    auto layout_permutation =
        ToTransposeDimensions(scatter->scatter_operands()[0]->shape().layout());
    auto operand_permutation = InversePermutation(layout_permutation);

    auto update_permutation = InversePermutation(
        ToTransposeDimensions(scatter->scatter_updates()[0]->shape().layout()));

    // scatter_dims_to_operand_dims -> mapping from scatter dimensions to
    // operand dimensions. scatter dimension i corresponds to
    // scatter_dims_to_operand_dims[i] operand dimension.

    ScatterDimensionNumbers normalized_dims;
    normalized_dims.set_index_vector_dim(
        indices_permutation[dims.index_vector_dim()]);
    for (int64_t dim : dims.scatter_dims_to_operand_dims()) {
      normalized_dims.add_scatter_dims_to_operand_dims(
          operand_permutation[dim]);
    }
    std::vector<int64_t> normalized_update_window_dims;
    normalized_update_window_dims.reserve(dims.update_window_dims_size());
    for (int64_t dim : dims.update_window_dims()) {
      normalized_update_window_dims.push_back(update_permutation[dim]);
    }

    // Now reorder 'normalized_update_window_dims' and 'inserted_window_dims'
    // according to the output permutation, so that the window dimensions again
    // appear in the same order as in the output. First we need to build a
    // combined array of window dimensions. Note: 'inserted_window_dims' and
    // 'update_window_dims' must be sorted according to shape inference/hlo
    // verifier. We will temporarily create an unsorted update_window_dims
    // attribute and rely on ScatterSimplifier to clean this up.
    std::vector<int64_t> window_dimensions(operand_permutation.size());
    for (int64_t i = 0, j = 0, k = 0; i < window_dimensions.size(); ++i) {
      if (j < dims.inserted_window_dims_size() &&
          dims.inserted_window_dims(j) == i) {
        window_dimensions[i] = -1;
        ++j;
      } else {
        window_dimensions[i] = normalized_update_window_dims[k];
        ++k;
      }
    }
    std::vector<int64_t> permuted_window_dimensions =
        ComposePermutations(window_dimensions, layout_permutation);
    for (int64_t i = 0; i < permuted_window_dimensions.size(); ++i) {
      if (permuted_window_dimensions[i] == -1) {
        normalized_dims.add_inserted_window_dims(i);
      } else {
        normalized_dims.add_update_window_dims(permuted_window_dimensions[i]);
      }
    }

    auto normalized_shape = normalized_operands.front()->shape();
    if (scatter->shape().IsTuple()) {
      std::vector<Shape> tuple_shapes;
      tuple_shapes.reserve(normalized_operands.size());
      for (HloInstruction* operand : normalized_operands) {
        tuple_shapes.push_back(operand->shape());
      }
      normalized_shape = ShapeUtil::MakeTupleShape(tuple_shapes);
    }
    auto normalized_scatter = hlo->AddInstruction(HloInstruction::CreateScatter(
        normalized_shape, normalized_operands, normalized_indices,
        normalized_updates, scatter->to_apply(), normalized_dims,
        scatter->indices_are_sorted(), scatter->unique_indices()));
    SetVisited(*normalized_scatter);
    auto bc_to_orig = MakeBitcastHlo(normalized_scatter, scatter->shape());
    TF_RETURN_IF_ERROR(ReplaceInstruction(scatter, bc_to_orig));
    return absl::OkStatus();
  }

  // Converts:
  //
  // A{I} -> bitcast[S0]{L} -> transpose[S]{L2}
  //
  // Into:
  //
  // A{I} -> transpose[S']{I} -> bitcast{L2}
  //
  // Where S' is the normalization of [S]{L2}, and `dimensions` attribute is
  //
  // The `dimensions` of the new transposition is given by:
  //
  //  L^-1 o `dim_0` o L2
  //
  // where dim_0 is dimensions of the original transposition, and `o` denotes
  // permutation composition.
  absl::Status HandleTranspose(HloInstruction* hlo) override {
    auto s = hlo->shape();
    auto operand = hlo->mutable_operand(0);
    auto operand_s = operand->shape();
    TF_ASSIGN_OR_RETURN(auto a0, GetNormalizedInput(operand));
    auto normalized_shape = Normalize(s);
    VLOG(3) << "Input transpose: " << hlo->ToString();

    auto l0_perm =
        InversePermutation(ToTransposeDimensions(operand_s.layout()));
    auto l_perm = ToTransposeDimensions(s.layout());

    auto t = ComposePermutations(l0_perm, hlo->dimensions());
    auto dimensions = ComposePermutations(t, l_perm);
    HloInstruction* normalized_transpose;

    if (IsIdentityPermutation(dimensions)) {
      // If we're dealing with an identity transposition, there's no need to
      // actually create the transpose.
      normalized_transpose = a0;
    } else {
      normalized_transpose = hlo->AddInstruction(
          HloInstruction::CreateTranspose(normalized_shape, a0, dimensions));
      SetVisited(*normalized_transpose);
      VLOG(3) << "Generated normalized physical transpose: "
              << normalized_transpose->ToString();
    }

    auto bc_to_orig = MakeBitcastHlo(normalized_transpose, s);
    return ReplaceInstruction(hlo, bc_to_orig);
  }

  // Converts a purely physical copy into a physical+logical transposition.
  //
  // Converts:
  //
  //  A{I} -> bitcast{L} -> copy[S]{L'}
  //
  // Into:
  //
  //  A{I} -> transpose[S']{I} -> bitcast[S]{L'}
  //
  // Where S' is normalization of [S]{L'}, and transposition dimensions are
  // given by L'.
  absl::Status HandleCopy(HloInstruction* hlo) override {
    VLOG(3) << "Processing copy: " << hlo->ToString();
    auto s = hlo->shape();
    auto operand = hlo->mutable_operand(0);
    TF_ASSIGN_OR_RETURN(auto a0, GetNormalizedInput(operand));
    auto s_normalized = Normalize(s);
    auto l0_perm =
        InversePermutation(ToTransposeDimensions(operand->shape().layout()));
    auto l_perm = ToTransposeDimensions(s.layout());
    auto dimensions = ComposePermutations(l0_perm, l_perm);
    auto t = hlo->AddInstruction(
        HloInstruction::CreateTranspose(s_normalized, a0, dimensions));
    SetVisited(*t);
    auto bc_to_orig = MakeBitcastHlo(t, s);
    TF_RETURN_IF_ERROR(ReplaceInstruction(hlo, bc_to_orig));
    return absl::OkStatus();
  }

  // The reverse HLO has a list of dimensions it reverses.
  absl::Status HandleReverse(HloInstruction* hlo) override {
    auto s = hlo->shape();
    auto operand = hlo->mutable_operand(0);
    TF_ASSIGN_OR_RETURN(auto a0, GetNormalizedInput(operand));
    std::vector<int64_t> layout_as_permutation =
        ToTransposeDimensions(hlo->shape().layout());
    std::vector<int64_t> new_dimensions;
    new_dimensions.reserve(hlo->dimensions().size());
    auto inverse_perm = InversePermutation(layout_as_permutation);
    for (int64_t dim : hlo->dimensions()) {
      new_dimensions.push_back(inverse_perm[dim]);
    }
    absl::c_sort(new_dimensions);
    auto normalized_reverse = hlo->AddInstruction(
        HloInstruction::CreateReverse(a0->shape(), a0, new_dimensions));
    SetVisited(*normalized_reverse);
    auto bc_to_orig = MakeBitcastHlo(normalized_reverse, s);
    TF_RETURN_IF_ERROR(ReplaceInstruction(hlo, bc_to_orig));
    return absl::OkStatus();
  }

  // Padding is layout-preserving, so we only have to permute values inside the
  // padding config.
  absl::Status HandlePad(HloInstruction* hlo) override {
    auto s = hlo->shape();
    auto operand = hlo->mutable_operand(0);
    auto padded_by = hlo->mutable_operand(1);
    auto padded_config = hlo->padding_config();
    TF_ASSIGN_OR_RETURN(HloInstruction * normalized_input,
                        GetNormalizedInput(operand));

    auto s_normalized = Normalize(s);
    auto layout_as_permutation = ToTransposeDimensions(s.layout());

    PaddingConfig new_padding;
    new_padding.mutable_dimensions()->Reserve(s_normalized.rank());
    for (int dim = 0; dim < s_normalized.rank(); dim++) {
      new_padding.add_dimensions();
    }

    auto inverse_perm = InversePermutation(layout_as_permutation);
    for (int dim = 0; dim < s.rank(); dim++) {
      int tr_dim = static_cast<int>(inverse_perm[dim]);
      *new_padding.mutable_dimensions(tr_dim) = padded_config.dimensions(dim);
    }

    auto padded_normalized = hlo->AddInstruction(HloInstruction::CreatePad(
        s_normalized, normalized_input, padded_by, new_padding));
    SetVisited(*padded_normalized);
    auto bc_to_orig = MakeBitcastHlo(padded_normalized, s);
    TF_RETURN_IF_ERROR(ReplaceInstruction(hlo, bc_to_orig));
    return absl::OkStatus();
  }

  absl::Status HandleCustomCall(HloInstruction* hlo) override {
    if (custom_call_transformer_) {
      TF_ASSIGN_OR_RETURN(
          std::optional<HloInstruction*> transformed_custom_call,
          custom_call_transformer_(Cast<HloCustomCallInstruction>(hlo)));
      if (transformed_custom_call) {
        SetVisited(*(*transformed_custom_call)->operand(0));
        TF_RETURN_IF_ERROR(ReplaceInstruction(hlo, *transformed_custom_call));
        return absl::OkStatus();
      }
    }
    return DefaultAction(hlo);
  }

  // Pushes down bitcast across the ternary select operation: same logic as
  // HandleElementwiseBinary.
  absl::Status HandleSelect(HloInstruction* hlo) override {
    return HandleTernary(hlo);
  }

  // DyanmicSlice is layout-preserving, so handling is analoguous to elementwise
  // unary, and transposing the elements inside the metadata, as well as the
  // operands specifying dimension sizes.
  absl::Status HandleDynamicSlice(HloInstruction* hlo) override {
    const Shape& s = hlo->shape();
    HloInstruction* operand = hlo->mutable_operand(0);
    const Shape& operand_shape = operand->shape();
    TF_RET_CHECK(s.layout() == operand_shape.layout());

    TF_ASSIGN_OR_RETURN(HloInstruction * normalized_input,
                        GetNormalizedInput(operand));

    Shape normalized = Normalize(operand_shape);

    std::vector<int64_t> layout_as_permutation =
        ToTransposeDimensions(hlo->shape().layout());
    std::vector<HloInstruction*> new_start_indices =
        GetNewStartIdxs(hlo, /*param_offset=*/1, layout_as_permutation);

    auto normalize_slice_attr = [&](absl::Span<int64_t const> input) {
      return Permute(input, layout_as_permutation);
    };
    TF_ASSIGN_OR_RETURN(
        HloInstruction * normalized_dynamic_slice,
        MakeDynamicSliceHlo(normalized_input, new_start_indices,
                            normalize_slice_attr(hlo->dynamic_slice_sizes()),
                            &hlo->metadata()));
    *normalized_dynamic_slice->mutable_shape()->mutable_layout() =
        normalized_input->shape().layout();
    SetVisited(*normalized_dynamic_slice);
    HloInstruction* bc_to_orig = MakeBitcastHlo(normalized_dynamic_slice, s);
    TF_RETURN_IF_ERROR(ReplaceInstruction(hlo, bc_to_orig));
    return absl::OkStatus();
  }

  absl::Status HandleDynamicUpdateSlice(HloInstruction* hlo) override {
    const Shape& s = hlo->shape();
    HloInstruction* operand = hlo->mutable_operand(0);
    HloInstruction* update = hlo->mutable_operand(1);
    const Shape& operand_shape = operand->shape();
    TF_RET_CHECK(s.layout() == operand_shape.layout());
    std::vector<int64_t> layout_as_permutation =
        ToTransposeDimensions(hlo->shape().layout());

    TF_ASSIGN_OR_RETURN(HloInstruction * new_operand,
                        GetNormalizedInput(operand));
    TF_ASSIGN_OR_RETURN(HloInstruction * new_update,
                        GetNormalizedInput(update));
    std::vector<HloInstruction*> new_start_indices =
        GetNewStartIdxs(hlo, /*param_offset=*/2, layout_as_permutation);

    TF_ASSIGN_OR_RETURN(
        HloInstruction * new_dus,
        MakeDynamicUpdateSliceHlo(new_operand, new_update, new_start_indices,
                                  &hlo->metadata()));
    *new_dus->mutable_shape()->mutable_layout() = new_operand->shape().layout();
    SetVisited(*new_dus);

    HloInstruction* bc_to_orig = MakeBitcastHlo(new_dus, s);
    TF_RETURN_IF_ERROR(ReplaceInstruction(hlo, bc_to_orig));

    return absl::OkStatus();
  }

  absl::Status HandleClamp(HloInstruction* hlo) override {
    return HandleTernary(hlo);
  }

 private:
  // Replace clamp/select ternary operation with a normalized one.
  absl::Status HandleTernary(HloInstruction* hlo) {
    Shape s = hlo->shape();
    HloOpcode opcode = hlo->opcode();
    TF_RET_CHECK(opcode == HloOpcode::kClamp || opcode == HloOpcode::kSelect);
    HloInstruction* arg0 = hlo->mutable_operand(0);
    HloInstruction* arg1 = hlo->mutable_operand(1);
    HloInstruction* arg2 = hlo->mutable_operand(2);
    if (opcode == HloOpcode::kClamp) {
      TF_RET_CHECK(arg1->shape().layout() == s.layout());
    } else if (opcode == HloOpcode::kSelect) {
      TF_RET_CHECK(arg1->shape().layout() == s.layout());
      TF_RET_CHECK(arg2->shape().layout() == s.layout());
    } else {
      TF_RET_CHECK(false);
    }

    TF_ASSIGN_OR_RETURN(HloInstruction * normalized_arg0,
                        GetNormalizedInput(arg0));
    TF_ASSIGN_OR_RETURN(HloInstruction * normalized_arg1,
                        GetNormalizedInput(arg1));
    TF_ASSIGN_OR_RETURN(HloInstruction * normalized_arg2,
                        GetNormalizedInput(arg2));

    TF_ASSIGN_OR_RETURN(Shape new_shape, ShapeInference::InferTernaryOpShape(
                                             opcode, normalized_arg0,
                                             normalized_arg1, normalized_arg2));
    HloInstruction* normalized = hlo->parent()->AddInstruction(
        HloInstruction::CreateTernary(new_shape, opcode, normalized_arg0,
                                      normalized_arg1, normalized_arg2));
    hlo->SetupDerivedInstruction(normalized);
    SetVisited(*normalized);

    HloInstruction* bc_to_orig = MakeBitcastHlo(normalized, s);
    TF_RETURN_IF_ERROR(ReplaceInstruction(hlo, bc_to_orig));
    return absl::OkStatus();
  }

  std::vector<HloInstruction*> GetNewStartIdxs(
      HloInstruction* hlo, int param_offset,
      const std::vector<int64_t> layout_as_permutation) {
    std::vector<HloInstruction*> start_indices;
    for (int i = param_offset; i < hlo->operand_count(); i++) {
      start_indices.push_back(hlo->mutable_operand(i));
    }
    std::vector<HloInstruction*> permuted_start_indices =
        Permute(start_indices, layout_as_permutation);
    return permuted_start_indices;
  }

  // Converts a layout to a dimensions transposition necessary to get to that
  // layout from identity.
  std::vector<int64_t> ToTransposeDimensions(const Layout& l) {
    std::vector<int64_t> out(l.minor_to_major().begin(),
                             l.minor_to_major().end());
    absl::c_reverse(out);
    return out;
  }

  // Due to Local Precondition we have, the input to all processed ops should
  // be HLO in descending layout piped through bitcast.
  absl::StatusOr<HloInstruction*> GetNormalizedInput(HloInstruction* hlo) {
    TF_RET_CHECK(hlo->opcode() == HloOpcode::kBitcast)
        << "Unexpected HLO input: " << hlo->ToString();
    auto input = hlo->mutable_operand(0);
    auto input_shape = input->shape();
    TF_RET_CHECK(Layout::Equal().IgnoreElementSize()(
        input_shape.layout(),
        LayoutUtil::GetDefaultLayoutForShape(input_shape)));
    return input;
  }

  // Forces the layout to be descending.
  Shape Normalize(const Shape& s) {
    return ShapeUtil::MakeShapeWithDescendingLayoutAndSamePhysicalLayout(s);
  }

  LayoutNormalization* normalization_;
  CustomCallTransformer custom_call_transformer_;
};

}  // end namespace

absl::StatusOr<bool> LayoutNormalization::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  return LayoutNormalizationVisitor{this, custom_call_transformer_}.RunOnModule(
      module, execution_threads);
}

}  // end namespace xla

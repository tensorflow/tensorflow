/* Copyright 2019 The OpenXLA Authors.

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

#include "xla/service/dynamic_dimension_inference.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "absl/functional/function_ref.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/comparison_util.h"
#include "xla/hlo/ir/dfs_hlo_visitor_with_default.h"
#include "xla/hlo/ir/dynamic_parameter_binding.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/service/call_inliner.h"
#include "xla/service/dynamic_window_utils.h"
#include "xla/service/hlo_creation_utils.h"
#include "xla/service/hlo_dataflow_analysis.h"
#include "xla/service/hlo_value.h"
#include "xla/service/tuple_util.h"
#include "xla/service/while_util.h"
#include "xla/shape.h"
#include "xla/shape_tree.h"
#include "xla/shape_util.h"
#include "xla/status.h"
#include "xla/status_macros.h"
#include "xla/statusor.h"
#include "xla/util.h"
#include "xla/window_util.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/status.h"
#include "tsl/platform/statusor.h"

namespace xla {

namespace {
// Replace `narrow_comp` with a new computation with `wide_shape` as input.
absl::StatusOr<std::pair<HloComputation*, CallInliner::InlinedInstructionMap>>
WidenComputation(HloComputation* narrow_comp, const Shape& wide_shape) {
  TF_RET_CHECK(wide_shape.IsTuple());
  const Shape& narrow_shape = narrow_comp->parameter_instruction(0)->shape();
  if (Shape::Equal()(wide_shape, narrow_shape)) {
    // No need to widen the computation.
    return std::make_pair(narrow_comp, CallInliner::InlinedInstructionMap());
  }
  HloComputation* wide_comp = [&]() {
    HloComputation::Builder builder(absl::StrCat("wide.", narrow_comp->name()));
    builder.AddInstruction(HloInstruction::CreateParameter(
        0, wide_shape,
        absl::StrCat("wide.", narrow_comp->parameter_instruction(0)->name())));
    return narrow_comp->parent()->AddEmbeddedComputation(builder.Build());
  }();

  HloInstruction* wide_parameter = wide_comp->parameter_instruction(0);
  HloInstruction* truncated_parameter = TupleUtil::ExtractPrefix(
      wide_parameter, narrow_shape.tuple_shapes_size(),
      absl::StrCat("renarrowed.",
                   narrow_comp->parameter_instruction(0)->name()));
  HloInstruction* call_narrow_comp = wide_comp->AddInstruction(
      HloInstruction::CreateCall(narrow_comp->root_instruction()->shape(),
                                 {truncated_parameter}, narrow_comp));
  wide_comp->set_root_instruction(call_narrow_comp,
                                  /*accept_different_shape=*/true);
  TF_ASSIGN_OR_RETURN(auto inline_map, CallInliner::Inline(call_narrow_comp));
  return std::make_pair(wide_comp, std::move(inline_map));
}
}  // namespace

class DynamicDimensionInferenceVisitor : public DfsHloRewriteVisitor {
 public:
  explicit DynamicDimensionInferenceVisitor(
      const DynamicParameterBinding& param_bindings,
      HloDataflowAnalysis& dataflow_analysis, DynamicDimensionInference* parent,
      DynamicDimensionInference::CustomCallInferenceHandler custom_call_handler,
      DynamicDimensionInference::ShapeCheckMode shape_check_mode,
      DynamicDimensionInference::AssertionGenerator assertion_generator)
      : param_bindings_(param_bindings),
        dataflow_analysis_(dataflow_analysis),
        parent_(parent),
        custom_call_handler_(std::move(custom_call_handler)),
        shape_check_mode_(shape_check_mode),
        assertion_generator_(assertion_generator) {}

  Status DefaultAction(HloInstruction* hlo) override;

  static absl::StatusOr<bool> Run(
      HloComputation* computation, HloDataflowAnalysis& dataflow_analysis,
      const DynamicParameterBinding& param_bindings,
      DynamicDimensionInference* parent,
      DynamicDimensionInference::CustomCallInferenceHandler
          custom_call_handler = nullptr,
      DynamicDimensionInference::ShapeCheckMode shape_check_mode =
          DynamicDimensionInference::ShapeCheckMode::kIgnore,
      const DynamicDimensionInference::AssertionGenerator& assertion_generator =
          nullptr) {
    if (!HloInstruction::IsThreadIncluded(computation->execution_thread(),
                                          parent->execution_threads_)) {
      return false;
    }
    DynamicDimensionInferenceVisitor visitor(
        param_bindings, dataflow_analysis, parent,
        std::move(custom_call_handler), shape_check_mode, assertion_generator);

    TF_RETURN_IF_ERROR(computation->Accept(&visitor));
    if (visitor.shape_assertion_ != nullptr) {
      CHECK(assertion_generator);
      assertion_generator(visitor.shape_assertion_);
    }
    return visitor.changed();
  }

  Status HandleParameter(HloInstruction* hlo) override;

  Status HandleInfeed(HloInstruction* hlo) override;

  Status HandleConstant(HloInstruction* hlo) override;

  Status HandleReduce(HloInstruction* hlo) override;

  Status HandleDot(HloInstruction* hlo) override;

  Status HandleTuple(HloInstruction* hlo) override;

  Status HandleTranspose(HloInstruction* hlo) override;

  Status HandleDynamicReshape(HloInstruction* hlo) override;

  Status HandleReshape(HloInstruction* hlo) override;

  Status HandleSort(HloInstruction* hlo) override;

  Status HandlePad(HloInstruction* hlo) override;

  Status HandleCustomCall(HloInstruction* hlo) override;

  Status HandleBroadcast(HloInstruction* hlo) override;

  Status HandleGetDimensionSize(HloInstruction* hlo) override;

  Status HandleSetDimensionSize(HloInstruction* hlo) override;

  Status HandleSelect(HloInstruction* hlo) override;

  Status HandleConvolution(HloInstruction* hlo) override;

  Status HandleConcatenate(HloInstruction* hlo) override;

  Status HandleReduceWindow(HloInstruction* hlo) override;

  Status HandleReverse(HloInstruction* hlo) override;

  Status HandleSelectAndScatter(HloInstruction* hlo) override;

  Status HandleGetTupleElement(HloInstruction* hlo) override;

  Status HandleElementwiseUnary(HloInstruction* hlo) override;

  Status HandleElementwiseNary(HloInstruction* hlo);

  Status HandleElementwiseBinary(HloInstruction* hlo) override;

  Status HandleClamp(HloInstruction* hlo) override;

  Status HandleConditional(HloInstruction* hlo) override;

  Status HandleWhile(HloInstruction* hlo) override;

  Status HandleSlice(HloInstruction* hlo) override;

  Status HandleDynamicSlice(HloInstruction* hlo) override;

  Status HandleDynamicUpdateSlice(HloInstruction* hlo) override;

  Status HandleGather(HloInstruction* hlo) override;

  Status HandleScatter(HloInstruction* hlo) override;

  Status HandleMap(HloInstruction* hlo) override;

  Status HandleDomain(HloInstruction* hlo) override;

  Status HandleAsyncStart(HloInstruction* hlo) override;

  Status HandleAsyncDone(HloInstruction* hlo) override;

 private:
  using OperandDynamicDimensionFn = absl::FunctionRef<Status(
      HloInstruction* operand, ShapeIndex index, int64_t dimension,
      int64_t operand_index, HloInstruction* dynamic_size)>;

  using DynamicDimensionFn = std::function<Status(
      ShapeIndex index, int64_t dimension, HloInstruction* dynamic_size)>;

  void SetDynamicSize(HloInstruction* inst, const ShapeIndex& index,
                      int64_t dim, HloInstruction* size,
                      bool clear_dynamic_dimension = true);

  void SetDynamicSizes(HloInstruction* inst, const ShapeIndex& index,
                       absl::Span<HloInstruction* const> sizes);

  Status HandleDynamicConvolutionForward(HloInstruction* hlo,
                                         int64_t operand_index,
                                         int64_t dimension,
                                         HloInstruction* dynamic_size);

  Status HandleDynamicConvolutionKernelGrad(HloInstruction* hlo,
                                            int64_t operand_index,
                                            int64_t dimension);

  Status HandleDynamicConvolutionInputGrad(HloInstruction* hlo,
                                           int64_t operand_index,
                                           int64_t dimension);

  Status HandleDynamicWindowSamePadding(HloInstruction* hlo,
                                        HloInstruction* dynamic_size,
                                        int64_t operand_index,
                                        int64_t dimension);

  Status ForEachOperandDynamicDimension(HloInstruction* inst,
                                        OperandDynamicDimensionFn);
  Status ForEachDynamicDimensionInOperand(HloInstruction* inst,
                                          int64_t operand_index,
                                          OperandDynamicDimensionFn);
  Status ForEachDynamicDimension(HloInstruction* inst,
                                 const DynamicDimensionFn& fn);

  bool CanInfer(HloInstruction* hlo) { return parent_->CanInfer(hlo); }

  // Return true unless all users of the instruction can consume a dynamic shape
  // (including uses across control flow, but only within the same thread). The
  // given `ShapeIndex` is the leaf array returned by the given instruction that
  // will be considered.
  absl::StatusOr<bool> RequiresPadToStatic(HloInstruction* instr,
                                           ShapeIndex shape_index);

  // Insert pad-to-static after `inst` if `inst` has dynamic dimensions in it
  // and `RequiresPadToStatic` is true for all leaves. If the instruction
  // produces a tuple, each tuple component will be considered independently.
  // Returns the original instruction, with all arrays converted to static
  // shapes.
  Status InsertPadToStaticOnInstruction(HloInstruction* inst);

  // Insert shape check to make sure `dim1` is equal to `dim2`. If
  // support_implicit_broadcast is true, the check will pass if either of them
  // is 1, even if they are different.
  Status InsertShapeCheck(HloInstruction* dim1, HloInstruction* dim2,
                          bool support_implicit_broadcast);

  // Pass through a dynamic dimension from the input to the output with the
  // same value and index in the shape. This is a helper function to handle
  // trivial instructions like elementwise operations.
  Status PassThroughDynamicDimension(HloInstruction*);

  // The dynamic parameter bindings of this computation.
  const DynamicParameterBinding& param_bindings_;

  HloDataflowAnalysis& dataflow_analysis_;

  // A pointer to DynamicDimensionInference, used to update the dynamic mapping.
  DynamicDimensionInference* parent_;

  // A handler for custom calls.
  DynamicDimensionInference::CustomCallInferenceHandler custom_call_handler_;

  // Indicates what to do at places where shape check is needed.
  DynamicDimensionInference::ShapeCheckMode shape_check_mode_;

  // Value which has to be `true` for the shapes to match.
  HloInstruction* shape_assertion_ = nullptr;

  DynamicDimensionInference::AssertionGenerator assertion_generator_;
};

void DynamicDimensionInferenceVisitor::SetDynamicSize(
    HloInstruction* inst, const ShapeIndex& index, int64_t dim,
    HloInstruction* size, bool clear_dynamic_dimension) {
  parent_->SetDynamicSize(inst, index, dim, size);
  // Clear the dynamic dimension since we have recorded a dynamic size.
  // If there are any dynamic dimensions left after DynamicPadder has completely
  // run, we will raise an error.
  if (clear_dynamic_dimension) {
    ShapeUtil::GetMutableSubshape(inst->mutable_shape(), index)
        ->set_dynamic_dimension(dim, false);
  }
  MarkAsChanged();
}

void DynamicDimensionInferenceVisitor::SetDynamicSizes(
    HloInstruction* inst, const ShapeIndex& index,
    absl::Span<HloInstruction* const> sizes) {
  const Shape& subshape = ShapeUtil::GetSubshape(inst->shape(), index);
  CHECK(subshape.IsArray() && subshape.rank() == sizes.size());
  for (int64_t dimension = 0; dimension < subshape.rank(); ++dimension) {
    if (sizes[dimension] != nullptr) {
      SetDynamicSize(inst, index, dimension, sizes[dimension]);
    }
  }
}

Status DynamicDimensionInferenceVisitor::DefaultAction(HloInstruction* hlo) {
  return ForEachOperandDynamicDimension(
      hlo, [&](HloInstruction* operand, ShapeIndex index, int64_t dimension,
               int64_t operand_index, HloInstruction* dynamic_size) {
        return UnimplementedStrCat(
            "Asked to propagate a dynamic dimension from hlo ", operand->name(),
            "@", index.ToString(), "@", dimension, " to hlo ", hlo->ToString(),
            ", which is not implemented.");
      });
}

Status DynamicDimensionInferenceVisitor::HandleGetTupleElement(
    HloInstruction* hlo) {
  if (!CanInfer(hlo)) {
    return OkStatus();
  }
  return ForEachOperandDynamicDimension(
      hlo,
      [&](HloInstruction* operand, ShapeIndex index, int64_t dimension,
          int64_t operand_index, HloInstruction* dynamic_size) -> Status {
        if (hlo->tuple_index() != index[0]) {
          return OkStatus();
        }
        ShapeIndex new_index(ShapeIndexView(index).subspan(1));
        SetDynamicSize(hlo, new_index, dimension, dynamic_size);
        return OkStatus();
      });
}

Status DynamicDimensionInferenceVisitor::HandleTuple(HloInstruction* hlo) {
  if (!CanInfer(hlo)) {
    return OkStatus();
  }
  TF_RETURN_IF_ERROR(ForEachOperandDynamicDimension(
      hlo, [&](HloInstruction*, ShapeIndex index, int64_t dimension,
               int64_t operand_index, HloInstruction* dynamic_size) {
        index.push_front(operand_index);
        SetDynamicSize(hlo, index, dimension, dynamic_size);
        return OkStatus();
      }));
  return OkStatus();
}

Status DynamicDimensionInferenceVisitor::HandleBroadcast(HloInstruction* hlo) {
  if (!CanInfer(hlo)) {
    return OkStatus();
  }
  return ForEachOperandDynamicDimension(
      hlo, [&](HloInstruction* operand, ShapeIndex index, int64_t dimension,
               int64_t operand_index, HloInstruction* dynamic_size) {
        int64_t broadcast_dim = hlo->dimensions(dimension);
        SetDynamicSize(hlo, {}, broadcast_dim, dynamic_size);
        return OkStatus();
      });
}

Status DynamicDimensionInferenceVisitor::HandleConstant(HloInstruction* hlo) {
  if (!hlo->shape().is_dynamic()) {
    return OkStatus();
  }
  auto* constant = Cast<HloConstantInstruction>(hlo);
  ShapeTree<bool> do_pad(constant->shape(), false);
  Shape padded_shape = constant->shape();
  bool pad_any = false;
  TF_RETURN_IF_ERROR(ShapeUtil::ForEachMutableSubshapeWithStatus(
      &padded_shape, [&](Shape* subshape, const ShapeIndex& index) -> Status {
        if (!subshape->IsArray()) {
          return OkStatus();
        }
        TF_ASSIGN_OR_RETURN(bool requires_pad, RequiresPadToStatic(hlo, index));
        if (requires_pad) {
          pad_any = *do_pad.mutable_element(index) = true;
          *subshape = ShapeUtil::MakeStaticShape(*subshape);
        }
        return OkStatus();
      }));
  if (!pad_any) {
    return OkStatus();
  }
  Literal padded_literal(padded_shape);
  do_pad.ForEachElement([&](const ShapeIndex& index, bool requires_pad) {
    const Shape& subshape = ShapeUtil::GetSubshape(padded_shape, index);
    if (!subshape.IsArray()) {
      return OkStatus();
    }
    TF_RETURN_IF_ERROR(padded_literal.CopyFrom(constant->literal(), index,
                                               index,
                                               /*only_dynamic_bound=*/true));
    if (!requires_pad) {
      for (int64_t dimension = 0; dimension < subshape.rank(); ++dimension) {
        if (subshape.is_dynamic_dimension(dimension)) {
          padded_literal.SetDynamicSize(
              dimension, index,
              constant->literal().GetDynamicSize(dimension, index));
        }
      }
    }
    return OkStatus();
  });
  auto* padded_constant = hlo->AddInstruction(
      HloInstruction::CreateConstant(std::move(padded_literal)));
  TF_RETURN_IF_ERROR(constant->ReplaceAllUsesWith(padded_constant));
  SetVisited(*padded_constant);
  TF_RETURN_IF_ERROR(do_pad.ForEachElementWithStatus(
      [&](const ShapeIndex& index, bool requires_pad) -> Status {
        if (!requires_pad) {
          return OkStatus();
        }
        const Shape& subshape =
            ShapeUtil::GetSubshape(constant->shape(), index);
        TF_RET_CHECK(subshape.IsArray());
        for (int64_t dimension = 0; dimension < subshape.rank(); ++dimension) {
          if (!subshape.is_dynamic_dimension(dimension)) {
            continue;
          }
          HloInstruction* dynamic_size = hlo->AddInstruction(
              HloInstruction::CreateConstant(LiteralUtil::CreateR0<int32_t>(
                  constant->literal().GetDynamicSize(dimension, index))));
          SetVisited(*dynamic_size);
          SetDynamicSize(padded_constant, index, dimension, dynamic_size);
        }
        return OkStatus();
      }));
  MarkAsChanged();
  return OkStatus();
}

Status DynamicDimensionInferenceVisitor::HandleCustomCall(HloInstruction* hlo) {
  if (hlo->custom_call_target() == "PadToStatic") {
    for (int64_t i = 0; i < hlo->operand(0)->shape().rank(); ++i) {
      if (hlo->operand(0)->shape().is_dynamic_dimension(i)) {
        HloInstruction* dynamic_size =
            hlo->parent()->AddInstruction(HloInstruction::CreateGetTupleElement(
                ShapeUtil::MakeScalarShape(S32), hlo, i + 1));
        // PadToStatic converts a dynamic dimension to static dimension. It then
        // returns the padded data output and the dynamic sizes of input
        // dimensions.
        ShapeIndex data_output = {0};
        SetDynamicSize(hlo, data_output, i, dynamic_size);
      }
    }
    return OkStatus();
  }

  if (!CanInfer(hlo)) {
    return OkStatus();
  }

  if (custom_call_handler_) {
    TF_RETURN_IF_ERROR(custom_call_handler_(hlo, parent_));
  } else {
    TF_RETURN_IF_ERROR(ForEachOperandDynamicDimension(
        hlo,
        [&](HloInstruction* operand, ShapeIndex index, int64_t dimension,
            int64_t operand_index,
            HloInstruction* dynamic_size) -> absl::Status {
          // Resize custom call should propagate dynamic batch (0) and channel
          // (3) dimensions.
          if (hlo->custom_call_target() == "SliceToDynamic" ||
              hlo->custom_call_target() == "Sharding" ||
              (absl::StartsWith(hlo->custom_call_target(), "Resize") &&
               (dimension == 0 || dimension == 3))) {
            SetDynamicSize(hlo, {}, dimension, dynamic_size);
            return OkStatus();
          }
          if (hlo->custom_call_target() == "DynamicReduceWindowSamePadding") {
            if (hlo->operand_count() > 2) {
              return Unimplemented(
                  "DynamicReduceWindowSamePadding doesn't support variadic "
                  "reduce window %s",
                  hlo->ToString());
            }
            return HandleDynamicWindowSamePadding(hlo, dynamic_size,
                                                  operand_index, dimension);
          }

          if (hlo->custom_call_target() ==
              "DynamicSelectAndScatterSamePadding") {
            if (operand_index == 1) {
              // Operand 0 (input) determines dynamic output size. We ignore the
              // dynamic size in the operand 1 (output gradient).
              return OkStatus();
            }
            SetDynamicSize(hlo, {}, dimension, dynamic_size);
            return OkStatus();
          }

          if (hlo->custom_call_target() == "DynamicConvolutionInputGrad") {
            return HandleDynamicConvolutionInputGrad(hlo, operand_index,
                                                     dimension);
          }

          if (hlo->custom_call_target() == "DynamicConvolutionKernelGrad") {
            return HandleDynamicConvolutionKernelGrad(hlo, operand_index,
                                                      dimension);
          }

          if (hlo->custom_call_target() == "DynamicConvolutionForward") {
            return HandleDynamicConvolutionForward(hlo, operand_index,
                                                   dimension, dynamic_size);
          }
          return Unimplemented(
              "CustomCall \"%s\" is not supported to have a dynamic dimension",
              hlo->custom_call_target());
        }));
  }

  return InsertPadToStaticOnInstruction(hlo);
}

Status DynamicDimensionInferenceVisitor::HandleSort(HloInstruction* hlo) {
  if (!CanInfer(hlo)) {
    return OkStatus();
  }
  return ForEachOperandDynamicDimension(
      hlo,
      [&](HloInstruction* operand, ShapeIndex index, int64_t dynamic_dimension,
          int64_t operand_index, HloInstruction* dynamic_size) {
        HloSortInstruction* sort = Cast<HloSortInstruction>(hlo);
        if (sort->values_count() == 0) {
          SetDynamicSize(hlo, {}, dynamic_dimension, dynamic_size);
        } else {
          SetDynamicSize(hlo, {operand_index}, dynamic_dimension, dynamic_size);
        }

        return OkStatus();
      });
}

Status DynamicDimensionInferenceVisitor::HandlePad(HloInstruction* hlo) {
  if (!CanInfer(hlo)) {
    return OkStatus();
  }
  return ForEachOperandDynamicDimension(
      hlo,
      [&](HloInstruction* operand, ShapeIndex index, int64_t dimension,
          int64_t operand_index, HloInstruction* dynamic_size) -> absl::Status {
        if (operand_index != 0) {
          return Unimplemented(
              "Dynamic dimension on padding value is not supported");
        }
        const PaddingConfig_PaddingConfigDimension& padding_config =
            hlo->padding_config().dimensions(dimension);

        HloInstruction* dynamic_size_adjusted = dynamic_size;
        if (padding_config.interior_padding() != 0) {
          // Adjust for interior padding :
          // Size' = max((Size - 1), 0) * interior_padding + Size
          HloInstruction* one =
              hlo->parent()->AddInstruction(HloInstruction::CreateConstant(
                  LiteralUtil::CreateR0<int32_t>(1)));
          HloInstruction* zero =
              hlo->parent()->AddInstruction(HloInstruction::CreateConstant(
                  LiteralUtil::CreateR0<int32_t>(0)));
          HloInstruction* interior_padding = hlo->parent()->AddInstruction(
              HloInstruction::CreateConstant(LiteralUtil::CreateR0<int32_t>(
                  padding_config.interior_padding())));
          dynamic_size_adjusted =
              hlo->parent()->AddInstruction(HloInstruction::CreateBinary(
                  dynamic_size_adjusted->shape(), HloOpcode::kSubtract,
                  dynamic_size_adjusted, one));
          dynamic_size_adjusted =
              hlo->parent()->AddInstruction(HloInstruction::CreateBinary(
                  dynamic_size_adjusted->shape(), HloOpcode::kMaximum,
                  dynamic_size_adjusted, zero));
          dynamic_size_adjusted =
              hlo->parent()->AddInstruction(HloInstruction::CreateBinary(
                  dynamic_size_adjusted->shape(), HloOpcode::kMultiply,
                  dynamic_size_adjusted, interior_padding));
          dynamic_size_adjusted =
              hlo->parent()->AddInstruction(HloInstruction::CreateBinary(
                  dynamic_size_adjusted->shape(), HloOpcode::kAdd,
                  dynamic_size_adjusted, dynamic_size));
        }
        HloInstruction* adjustment = hlo->parent()->AddInstruction(
            HloInstruction::CreateConstant(LiteralUtil::CreateR0<int32_t>(
                padding_config.edge_padding_low() +
                padding_config.edge_padding_high())));
        dynamic_size_adjusted =
            hlo->parent()->AddInstruction(HloInstruction::CreateBinary(
                dynamic_size_adjusted->shape(), HloOpcode::kAdd,
                dynamic_size_adjusted, adjustment));
        SetDynamicSize(hlo, {}, dimension, dynamic_size_adjusted);
        return OkStatus();
      });
}

Status DynamicDimensionInferenceVisitor::HandleReduce(HloInstruction* hlo) {
  if (!CanInfer(hlo)) {
    return OkStatus();
  }
  auto* reduce = Cast<HloReduceInstruction>(hlo);
  int64_t rank = -1;
  TF_RETURN_IF_ERROR(ShapeUtil::ForEachSubshapeWithStatus(
      reduce->shape(),
      [&](const Shape& subshape, const ShapeIndex& index) -> Status {
        if (!subshape.IsArray()) {
          return OkStatus();
        }
        if (rank < 0) {
          rank = subshape.rank();
        } else {
          TF_RET_CHECK(rank == subshape.rank());
        }
        return OkStatus();
      }));
  TF_RET_CHECK(rank >= 0);
  absl::InlinedVector<HloInstruction*, 4> dynamic_sizes(rank, nullptr);

  TF_RETURN_IF_ERROR(ForEachOperandDynamicDimension(
      hlo, [&](HloInstruction* operand, ShapeIndex index, int64_t dimension,
               int64_t operand_index, HloInstruction* dynamic_size) {
        int64_t operand_count = reduce->operand_count();
        CHECK_EQ(operand_count % 2, 0);
        if (operand_index >= reduce->input_count()) {
          // Init values doesn't have dynamic size.
          return OkStatus();
        }
        if (absl::c_count(reduce->dimensions(), dimension) != 0) {
          // Dimension is to be reduced, stop tracing.
          return OkStatus();
        }

        // Find out the new dynamic dimension after reduce.
        int64_t dimensions_not_reduced_count = 0;
        for (int64_t i = 0; i < operand->shape().rank(); ++i) {
          if (dimension == i) {
            // The dimensions of all data operands of a variadic reduce have
            // to be the same.  This means that if one operand of variadic
            // reduce has a dynamic dimension, we set all outputs to use the
            // same dynamic size in corresponding dimensions.
            dynamic_sizes[dimensions_not_reduced_count] = dynamic_size;
            return OkStatus();
          }
          if (!absl::c_linear_search(reduce->dimensions(), i)) {
            dimensions_not_reduced_count++;
          }
        }

        return OkStatus();
      }));

  ShapeUtil::ForEachSubshape(
      reduce->shape(), [&](const Shape& subshape, ShapeIndex shape_index) {
        if (!subshape.IsArray()) {
          return;
        }
        SetDynamicSizes(reduce, shape_index, dynamic_sizes);
      });

  return OkStatus();
}

Status DynamicDimensionInferenceVisitor::HandleDot(HloInstruction* hlo) {
  if (!CanInfer(hlo)) {
    return OkStatus();
  }
  absl::InlinedVector<HloInstruction*, 4> dynamic_sizes(hlo->shape().rank(),
                                                        nullptr);
  TF_RETURN_IF_ERROR(ForEachOperandDynamicDimension(
      hlo,
      [&](HloInstruction* operand, ShapeIndex operand_shape_index,
          int64_t operand_dimension, int64_t operand_index,
          HloInstruction* dynamic_size) -> Status {
        // There are three types of dimensions in a dot:
        // A. batch dims
        // B. contracting dims
        // C. non-batch non-contracting dims.
        // The output dimensions of a dot has three parts with the following
        // order:
        // [(type A), (lhs type C), (rhs type C)]
        //
        // Note that both lhs and rhs have the same dimension sizes for batch,
        // but the dimension index could be different.
        //
        // Given one dynamic input dimension, either lhs or rhs, we use a
        // mapping to find the corresponding output dimension.
        HloInstruction* dot = hlo;
        const DotDimensionNumbers& dimension_numbers =
            dot->dot_dimension_numbers();
        // A map from the operand dimensions to result dimension.
        absl::flat_hash_map<int64_t, int64_t> result_dim_mapping;
        int64_t current_result_dims = 0;

        bool lhs = operand_index == 0;

        // The first loop keep tracks of batch dimension. RHS and LHS could have
        // different batch dimension numbers.
        if (lhs) {
          for (int64_t i : dimension_numbers.lhs_batch_dimensions()) {
            result_dim_mapping[i] = current_result_dims++;
          }
        } else {
          for (int64_t i : dimension_numbers.rhs_batch_dimensions()) {
            result_dim_mapping[i] = current_result_dims++;
          }
        }

        // Handle dimensions in the lhs.
        for (int64_t i = 0; i < dot->operand(0)->shape().rank(); i++) {
          // Look for non-contracting and non-batching dimension.
          if (absl::c_linear_search(
                  dimension_numbers.lhs_contracting_dimensions(), i)) {
            continue;
          }
          if (absl::c_linear_search(dimension_numbers.lhs_batch_dimensions(),
                                    i)) {
            continue;
          }
          if (lhs) {
            result_dim_mapping[i] = current_result_dims;
          }
          current_result_dims++;
        }

        // Handle dimensions in the rhs.
        for (int64_t i = 0; i < dot->operand(1)->shape().rank(); i++) {
          // Look for non-contracting and non-batching dimension.
          if (absl::c_linear_search(
                  dimension_numbers.rhs_contracting_dimensions(), i)) {
            continue;
          }
          if (absl::c_linear_search(dimension_numbers.rhs_batch_dimensions(),
                                    i)) {
            continue;
          }
          if (!lhs) {
            result_dim_mapping[i] = current_result_dims;
          }
          current_result_dims++;
        }

        // Check if the operand dim is in the result shape. If so, add another
        // work item to trace that dimension.
        auto iter = result_dim_mapping.find(operand_dimension);
        if (iter != result_dim_mapping.end()) {
          dynamic_sizes[iter->second] = dynamic_size;
        }

        return OkStatus();
      }));

  SetDynamicSizes(hlo, {}, dynamic_sizes);

  return OkStatus();
}

Status DynamicDimensionInferenceVisitor::HandleTranspose(HloInstruction* hlo) {
  if (!CanInfer(hlo)) {
    return OkStatus();
  }
  return ForEachOperandDynamicDimension(
      hlo,
      [&](HloInstruction* operand, ShapeIndex index, int64_t dimension,
          int64_t operand_index, HloInstruction* dynamic_size) -> Status {
        int64_t permuted_dim = -1;
        for (int64_t i = 0; i < hlo->dimensions().size(); ++i) {
          if (hlo->dimensions()[i] == dimension) {
            TF_RET_CHECK(permuted_dim == -1);
            permuted_dim = i;
          }
        }
        SetDynamicSize(hlo, {}, permuted_dim, dynamic_size);
        return OkStatus();
      });
}

Status DynamicDimensionInferenceVisitor::HandleConvolution(
    HloInstruction* hlo) {
  if (!CanInfer(hlo)) {
    return OkStatus();
  }
  return ForEachOperandDynamicDimension(
      hlo,
      [&](HloInstruction* operand, ShapeIndex index, int64_t dimension,
          int64_t operand_index, HloInstruction* dynamic_size) -> absl::Status {
        HloInstruction* conv = hlo;
        const ConvolutionDimensionNumbers& dimension_numbers =
            conv->convolution_dimension_numbers();
        if (operand_index == 0) {
          if (dimension == dimension_numbers.input_batch_dimension()) {
            SetDynamicSize(conv, {}, dimension_numbers.output_batch_dimension(),
                           dynamic_size);
            return OkStatus();
          }

          if (dimension == dimension_numbers.input_feature_dimension()) {
            return OkStatus();
          }
        } else {
          if (dimension == dimension_numbers.kernel_input_feature_dimension()) {
            return OkStatus();
          }
        }

        return Unimplemented("Dynamic Spatial Convolution is not supported: %s",
                             conv->ToString());
      });
}

Status DynamicDimensionInferenceVisitor::HandleConcatenate(
    HloInstruction* hlo) {
  if (!CanInfer(hlo)) {
    return OkStatus();
  }
  // First handle concatenate dimensions. We do this by iterating through all
  // operands while tracking both dynamic and static dimensions.

  // static_size is used to keep track of the concatenated size of static
  // dimensions.
  int64_t static_size = 0;
  std::vector<HloInstruction*> dynamic_concat_dims;
  for (int64_t i = 0; i < hlo->operand_count(); ++i) {
    HloInstruction* concat_dim_size = nullptr;
    for (int64_t dimension = 0; dimension < hlo->operand(i)->shape().rank();
         ++dimension) {
      if (dimension == hlo->concatenate_dimension()) {
        HloInstruction* dynamic_size =
            parent_->GetDynamicSize(hlo->mutable_operand(i), {}, dimension);
        concat_dim_size = dynamic_size;
      }
    }
    if (concat_dim_size == nullptr) {
      // This is a static dimension.
      static_size +=
          hlo->operand(i)->shape().dimensions(hlo->concatenate_dimension());
    } else {
      dynamic_concat_dims.push_back(concat_dim_size);
    }
  }
  // If concat dimension is dynamic, calculate its size by summing up static
  // dims and dynamic dims together.
  std::vector<HloInstruction*> dynamic_sizes(hlo->shape().rank(), nullptr);
  if (!dynamic_concat_dims.empty()) {
    HloInstruction* dim_size_total =
        hlo->parent()->AddInstruction(HloInstruction::CreateConstant(
            LiteralUtil::CreateR0<int32_t>(static_size)));
    for (HloInstruction* dynamic_dim : dynamic_concat_dims) {
      dim_size_total = hlo->parent()->AddInstruction(
          HloInstruction::CreateBinary(dim_size_total->shape(), HloOpcode::kAdd,
                                       dim_size_total, dynamic_dim));
    }
    dynamic_sizes[hlo->concatenate_dimension()] = dim_size_total;
  }

  // Simply pass through non-concat dynamic dimensions.
  TF_RETURN_IF_ERROR(ForEachOperandDynamicDimension(
      hlo,
      [&](HloInstruction* operand, ShapeIndex index, int64_t dimension,
          int64_t operand_index, HloInstruction* dynamic_size) -> Status {
        TF_RET_CHECK(index.empty());
        int64_t concatenate_dimension = hlo->concatenate_dimension();
        if (concatenate_dimension == dimension) {
          return OkStatus();
        }
        dynamic_sizes[dimension] = dynamic_size;
        return OkStatus();
      }));

  SetDynamicSizes(hlo, {}, dynamic_sizes);

  return OkStatus();
}

Status DynamicDimensionInferenceVisitor::HandleGetDimensionSize(
    HloInstruction* gds) {
  // Dynamic dimension doesn't propagate through GetDimensionSize:
  //
  //   Input: F32[x, y, z]
  //     |
  //   GetDimensionSize(1): S32[]
  //
  // The returned value is a scalar, which doesn't have any dynamic dimension in
  // the shape (although the value contains the real size of the dynamic
  // dimension of the input).
  int64_t dim = gds->dimension();
  TF_RET_CHECK(dim < gds->operand(0)->shape().rank()) << gds->ToString();
  HloInstruction* operand = gds->mutable_operand(0);
  TF_RET_CHECK(dim < operand->shape().rank());
  HloInstruction* replacement = parent_->GetDynamicSize(operand, {}, dim);
  HloComputation* computation = gds->parent();
  if (replacement == nullptr &&
      !gds->operand(0)->shape().is_dynamic_dimension(dim)) {
    TF_RET_CHECK(dim < gds->operand(0)->shape().rank());
    int32_t size = gds->operand(0)->shape().dimensions(dim);
    replacement = computation->AddInstruction(
        HloInstruction::CreateConstant(LiteralUtil::CreateR0<int32_t>(size)),
        gds->name());
  }

  if (replacement != nullptr) {
    TF_RETURN_IF_ERROR(gds->ReplaceAllUsesWith(replacement));
    // The dependency between an instruction and its dynamic dimensions is not
    // modeled in the IR. As instr is being replaced by dynamic_size, also tell
    // dynamic dimension inference that the instruction is being replaced.
    parent_->ReplaceAllDynamicDimensionUsesWith(gds, replacement);
    MarkAsChanged();
  }
  return OkStatus();
}

Status DynamicDimensionInferenceVisitor::HandleSetDimensionSize(
    HloInstruction* hlo) {
  bool dimension_is_static = false;
  const HloInstruction* size = hlo->operand(1);
  if (size->opcode() == HloOpcode::kConstant) {
    // Check if we are setting a dimension size to its static size. If so,
    // removes the dynamic dimension.
    //
    // size = s32[] constant(5)
    // s32[2, 5] = set-dimension-size(s32[2,<=5]{1,0} %param, s32[] %size),
    //                                                        dimensions={1}
    // The result shape has no dynamic dimension.
    TF_RET_CHECK(size->shape().rank() == 0);
    if (size->literal().Get<int32_t>({}) ==
            hlo->shape().dimensions(hlo->dimension()) &&
        !hlo->shape().is_dynamic_dimension(hlo->dimension())) {
      dimension_is_static = true;
    }
  }

  if (!dimension_is_static) {
    // Propagate dynamic dimension indicated by this set dimension size
    // instruction.
    SetDynamicSize(hlo, {}, hlo->dimension(), hlo->mutable_operand(1),
                   /*clear_dynamic_dimension=*/false);
  }

  // Also Propagate dynamic dimension already set by operands.
  TF_RETURN_IF_ERROR(ForEachOperandDynamicDimension(
      hlo,
      [&](HloInstruction* operand, ShapeIndex index, int64_t dimension,
          int64_t operand_index, HloInstruction* dynamic_size) -> Status {
        TF_RET_CHECK(operand_index == 0);
        if (dimension != hlo->dimension()) {
          SetDynamicSize(hlo, index, dimension, dynamic_size,
                         /*clear_dynamic_dimension=*/false);
        }
        return OkStatus();
      }));

  return OkStatus();
}

Status DynamicDimensionInferenceVisitor::HandleDynamicConvolutionForward(
    HloInstruction* hlo, int64_t operand_index, int64_t dimension,
    HloInstruction* dynamic_size) {
  if (!CanInfer(hlo)) {
    return OkStatus();
  }
  TF_RET_CHECK(operand_index == 0);
  const ConvolutionDimensionNumbers& dimension_numbers =
      hlo->convolution_dimension_numbers();

  if (dimension == dimension_numbers.input_batch_dimension()) {
    // Batch dimension is propagated without any changes.
    SetDynamicSize(hlo, {}, dimension_numbers.output_batch_dimension(),
                   dynamic_size);
    return OkStatus();
  }

  for (int64_t spatial_dim_index = 0;
       spatial_dim_index < dimension_numbers.input_spatial_dimensions_size();
       ++spatial_dim_index) {
    int64_t input_spatial_dim =
        dimension_numbers.input_spatial_dimensions(spatial_dim_index);
    int64_t output_spatial_dim =
        dimension_numbers.output_spatial_dimensions(spatial_dim_index);
    if (dimension == input_spatial_dim) {
      // This is a dynamic spatial dimension. Calculate the output size.
      WindowDimension window_dim = hlo->window().dimensions(spatial_dim_index);
      DynamicWindowDims dynamic_window_dims = GetWindowedOutputSize(
          dynamic_size, window_dim.size(), window_dim.window_dilation(),
          window_dim.stride(), hlo->padding_type());
      TF_RET_CHECK(window_dim.base_dilation() == 1);
      SetDynamicSize(hlo, {}, output_spatial_dim,
                     dynamic_window_dims.output_size);
      return OkStatus();
    }
  }
  // Input Feature dim disappears after convolution.
  return OkStatus();
}

Status DynamicDimensionInferenceVisitor::HandleDynamicWindowSamePadding(
    HloInstruction* hlo, HloInstruction* dynamic_size, int64_t operand_index,
    int64_t dimension) {
  if (!CanInfer(hlo)) {
    return OkStatus();
  }
  const Window& window = hlo->window();
  const WindowDimension& window_dim = window.dimensions(dimension);
  if (!window_util::IsTrivialWindowDimension(window_dim)) {
    DynamicWindowDims dynamic_window_dims = GetWindowedOutputSize(
        dynamic_size, window_dim.size(), window_dim.window_dilation(),
        window_dim.stride(), PaddingType::PADDING_SAME);
    SetDynamicSize(hlo, {}, dimension, dynamic_window_dims.output_size);
  } else {
    SetDynamicSize(hlo, {}, dimension, dynamic_size);
  }

  return OkStatus();
}

Status DynamicDimensionInferenceVisitor::HandleDynamicConvolutionInputGrad(
    HloInstruction* hlo, int64_t operand_index, int64_t dimension) {
  if (!CanInfer(hlo)) {
    return OkStatus();
  }
  // The output size of convolution input grad is corresponding input size.
  HloInstruction* input_sizes = hlo->mutable_operand(0);
  HloComputation* comp = hlo->parent();
  TF_RET_CHECK(input_sizes->shape().rank() == 1) << hlo->ToString();
  TF_RET_CHECK(input_sizes->shape().element_type() == S32) << hlo->ToString();
  TF_RET_CHECK(input_sizes->shape().dimensions(0) ==
               hlo->shape().dimensions_size())
      << hlo->ToString();
  // Slice to get corresponding input size.
  HloInstruction* slice = comp->AddInstruction(
      HloInstruction::CreateSlice(ShapeUtil::MakeShape(S32, {1}), input_sizes,
                                  {dimension}, {dimension + 1}, {1}));
  HloInstruction* reshape = comp->AddInstruction(
      HloInstruction::CreateReshape(ShapeUtil::MakeScalarShape(S32), slice));
  SetDynamicSize(hlo, {}, dimension, reshape);
  return OkStatus();
}

Status DynamicDimensionInferenceVisitor::HandleDynamicConvolutionKernelGrad(
    HloInstruction* hlo, int64_t operand_index, int64_t dimension) {
  // Dynamic convolution kernel grad produces static shape outputs.
  return OkStatus();
}

Status DynamicDimensionInferenceVisitor::PassThroughDynamicDimension(
    HloInstruction* hlo) {
  if (!CanInfer(hlo)) {
    return OkStatus();
  }
  // TODO(b/298671312): This is ambiguous with respect to which operand provides
  // the dynamic size.
  ShapeTree<absl::InlinedVector<HloInstruction*, 2>> dynamic_sizes(
      hlo->shape());
  TF_RETURN_IF_ERROR(ForEachOperandDynamicDimension(
      hlo, [&](HloInstruction* operand, ShapeIndex index, int64_t dimension,
               int64_t operand_index, HloInstruction* dynamic_size) {
        const Shape& subshape = ShapeUtil::GetSubshape(hlo->shape(), index);
        auto* element = dynamic_sizes.mutable_element(index);
        element->resize(subshape.rank(), nullptr);
        (*element)[dimension] = dynamic_size;
        return OkStatus();
      }));
  dynamic_sizes.ForEachElement([&](const ShapeIndex& index, const auto& sizes) {
    if (sizes.empty()) {
      return;
    }
    SetDynamicSizes(hlo, index, sizes);
  });
  return OkStatus();
}

Status DynamicDimensionInferenceVisitor::HandleDomain(HloInstruction* hlo) {
  return PassThroughDynamicDimension(hlo);
}

Status DynamicDimensionInferenceVisitor::HandleAsyncStart(HloInstruction* hlo) {
  if (!HloInstruction::IsThreadIncluded(hlo->async_execution_thread(),
                                        parent_->execution_threads_)) {
    // Async-start not included in specified execution thread set will use
    // metadata-prefix version of dynamic shapes (result of slice-to-dynamic) so
    // there is no need to propagate dynamic dimension info.
    return OkStatus();
  }
  return DefaultAction(hlo);
}

Status DynamicDimensionInferenceVisitor::HandleAsyncDone(HloInstruction* hlo) {
  if (!HloInstruction::IsThreadIncluded(hlo->async_execution_thread(),
                                        parent_->execution_threads_)) {
    // Other threads can return a dynamic shape directly, so we may need to
    // insert PadToStatic.
    return InsertPadToStaticOnInstruction(hlo);
  }
  return DefaultAction(hlo);
}

Status DynamicDimensionInferenceVisitor::HandleElementwiseUnary(
    HloInstruction* hlo) {
  return PassThroughDynamicDimension(hlo);
}

Status DynamicDimensionInferenceVisitor::HandleSelect(HloInstruction* hlo) {
  return HandleElementwiseNary(hlo);
}

Status DynamicDimensionInferenceVisitor::HandleElementwiseNary(
    HloInstruction* hlo) {
  if (!CanInfer(hlo)) {
    return OkStatus();
  }
  HloComputation* comp = hlo->parent();
  // First find all the dynamic sizes of the operands, and arrange them by
  // dimension.
  absl::InlinedVector<absl::InlinedVector<HloInstruction*, 2>, 2> operand_sizes(
      hlo->shape().rank(),
      absl::InlinedVector<HloInstruction*, 2>(hlo->operand_count(), nullptr));
  TF_RETURN_IF_ERROR(ForEachOperandDynamicDimension(
      hlo,
      [&](HloInstruction* operand, ShapeIndex index, int64_t dimension,
          int64_t operand_index, HloInstruction* dynamic_size) -> Status {
        TF_RET_CHECK(index.empty());
        operand_sizes[dimension][operand_index] = dynamic_size;
        return OkStatus();
      }));

  absl::InlinedVector<HloInstruction*, 2> existing_sizes(hlo->shape().rank(),
                                                         nullptr);
  for (int operand_index = 0; operand_index < hlo->operand_count();
       ++operand_index) {
    for (int64_t dimension = 0; dimension < hlo->shape().rank(); ++dimension) {
      HloInstruction* dynamic_size = operand_sizes[dimension][operand_index];
      if (dynamic_size == nullptr) {
        continue;
      }
      HloInstruction* existing_size = existing_sizes[dimension];
      if (existing_size == nullptr) {
        existing_sizes[dimension] = dynamic_size;
      } else if (existing_sizes[dimension] != dynamic_size) {
        TF_RETURN_IF_ERROR(
            InsertShapeCheck(existing_size, dynamic_size,
                             /*support_implicit_broadcast=*/true));

        auto one = comp->AddInstruction(
            HloInstruction::CreateConstant(LiteralUtil::One(S32)));

        auto operand_needs_broadcast =
            comp->AddInstruction(HloInstruction::CreateCompare(
                ShapeUtil::MakeShape(PRED, {}), dynamic_size, existing_size,
                ComparisonDirection::kLt));
        auto is_one = comp->AddInstruction(HloInstruction::CreateCompare(
            ShapeUtil::MakeShape(PRED, {}), dynamic_size, one,
            ComparisonDirection::kEq));
        operand_needs_broadcast =
            comp->AddInstruction(HloInstruction::CreateBinary(
                ShapeUtil::MakeShape(PRED, {}), HloOpcode::kAnd, is_one,
                operand_needs_broadcast));

        auto existing_needs_broadcast =
            comp->AddInstruction(HloInstruction::CreateCompare(
                ShapeUtil::MakeShape(PRED, {}), existing_size, dynamic_size,
                ComparisonDirection::kLt));
        is_one = comp->AddInstruction(HloInstruction::CreateCompare(
            ShapeUtil::MakeShape(PRED, {}), existing_size, one,
            ComparisonDirection::kEq));
        existing_needs_broadcast =
            comp->AddInstruction(HloInstruction::CreateBinary(
                ShapeUtil::MakeShape(PRED, {}), HloOpcode::kAnd, is_one,
                existing_needs_broadcast));

        auto needs_broadcast =
            comp->AddInstruction(HloInstruction::CreateBinary(
                ShapeUtil::MakeShape(PRED, {}), HloOpcode::kOr,
                operand_needs_broadcast, existing_needs_broadcast));
        auto max_size = comp->AddInstruction(HloInstruction::CreateBinary(
            ShapeUtil::MakeScalarShape(S32), HloOpcode::kMaximum, dynamic_size,
            existing_size));
        auto min_size = comp->AddInstruction(HloInstruction::CreateBinary(
            ShapeUtil::MakeScalarShape(S32), HloOpcode::kMinimum, dynamic_size,
            existing_size));
        auto select_size = comp->AddInstruction(HloInstruction::CreateTernary(
            ShapeUtil::MakeScalarShape(S32), HloOpcode::kSelect,
            needs_broadcast, max_size, min_size));
        existing_sizes[dimension] = select_size;
      }
    }
  }

  SetDynamicSizes(hlo, {}, existing_sizes);

  return OkStatus();
}

Status DynamicDimensionInferenceVisitor::HandleElementwiseBinary(
    HloInstruction* hlo) {
  return HandleElementwiseNary(hlo);
}

Status DynamicDimensionInferenceVisitor::HandleClamp(HloInstruction* hlo) {
  return PassThroughDynamicDimension(hlo);
}

Status DynamicDimensionInferenceVisitor::HandleDynamicReshape(
    HloInstruction* hlo) {
  if (!CanInfer(hlo)) {
    return OkStatus();
  }
  HloDynamicReshapeInstruction* dynamic_reshape =
      Cast<HloDynamicReshapeInstruction>(hlo);
  for (int64_t i = 0; i < hlo->shape().rank(); ++i) {
    if (hlo->shape().is_dynamic_dimension(i)) {
      SetDynamicSize(hlo, {}, i, dynamic_reshape->dim_sizes(i));
    }
  }
  return OkStatus();
}

Status DynamicDimensionInferenceVisitor::HandleReshape(
    HloInstruction* const hlo) {
  if (!CanInfer(hlo)) {
    return OkStatus();
  }
  VLOG(2) << "Handle reshape: " << hlo->ToString() << "\n";

  absl::InlinedVector<HloInstruction*, 2> dynamic_sizes(hlo->shape().rank(),
                                                        nullptr);
  using ReshapeGroup = std::pair<int64_t, int64_t>;
  using ReshapeGroupPair = std::pair<ReshapeGroup, ReshapeGroup>;
  auto is_reverse_reshape_group_pair =
      [&](const HloInstruction* op1, const ReshapeGroupPair& p1,
          const HloInstruction* op2, const ReshapeGroupPair& p2) -> bool {
    return ShapeUtil::EqualStructure(
               ShapeUtil::GetSubshape(
                   op1->operand(0)->shape(),
                   ShapeIndex(p1.first.first, p1.first.second)),
               ShapeUtil::GetSubshape(
                   op2->operand(0)->shape(),
                   ShapeIndex(p2.second.first, p2.second.second))) &&
           ShapeUtil::EqualStructure(
               ShapeUtil::GetSubshape(
                   op1->shape(), ShapeIndex(p1.second.first, p1.second.second)),
               ShapeUtil::GetSubshape(
                   op2->operand(0)->shape(),
                   ShapeIndex(p2.first.first, p2.first.second)));
  };
  auto find_reshape_group_pair = [](HloInstruction* reshape,
                                    int64_t input_dynamic_dimension) {
    VLOG(2) << "Find reshape pair: " << reshape->ToString() << "\n";
    auto common_factors =
        CommonFactors(reshape->operand(0)->shape().dimensions(),
                      reshape->shape().dimensions());
    ReshapeGroup input_dim = {-1, -1}, output_dim = {-1, -1};
    bool found = false;
    // Find common_factors that the input belongs to.
    for (int64_t i = 0; i < common_factors.size() - 1; ++i) {
      auto start = common_factors[i];
      auto end = common_factors[i + 1];
      if (input_dynamic_dimension >= start.first &&
          input_dynamic_dimension < end.first) {
        // Found the common_factor group that the input_dim belongs to.
        input_dim.first = start.first;
        input_dim.second = end.first;
        output_dim.first = start.second;
        output_dim.second = end.second;
        VLOG(3) << "Found common_factor group pair: " << input_dim.first << ","
                << input_dim.second << "->" << output_dim.first << ","
                << output_dim.second << "\n";
        found = true;
        break;
      }
    }
    CHECK(found);
    return ReshapeGroupPair(input_dim, output_dim);
  };
  auto reshape_group_pair_needs_flatten =
      [](const ReshapeGroupPair& reshape_pair) {
        return reshape_pair.first.second - reshape_pair.first.first > 1 &&
               reshape_pair.second.second - reshape_pair.second.first > 1;
      };
  std::function<bool(HloInstruction*, const ReshapeGroupPair&, int64_t)>
      find_reverse_past_reshape = [&](HloInstruction* op,
                                      const ReshapeGroupPair reshape_pair,
                                      int64_t dynamic_dimension_size) {
        VLOG(2) << "Find reverse past reshape from " << op->ToString()
                << " for " << dynamic_dimension_size << "\n";
        absl::InlinedVector<int64_t, 4> found_dims;
        for (int op_dim_index = 0; op_dim_index < op->shape().rank();
             ++op_dim_index) {
          if (op->shape().dimensions(op_dim_index) == dynamic_dimension_size) {
            found_dims.push_back(op_dim_index);
          }
        }
        if (found_dims.empty()) {
          return false;
        }
        VLOG(3) << "Found " << found_dims.size() << "\n";
        if (op->opcode() == HloOpcode::kReshape) {
          for (auto op_dim_index : found_dims) {
            auto orig_reshape_pair = find_reshape_group_pair(op, op_dim_index);
            if (is_reverse_reshape_group_pair(op, orig_reshape_pair, hlo,
                                              reshape_pair)) {
              TF_CHECK_OK(ForEachOperandDynamicDimension(
                  op,
                  [&](HloInstruction* operand, ShapeIndex index,
                      int64_t op_dynamic_dimension, int64_t operand_index,
                      HloInstruction* operand_dynamic_size) -> Status {
                    if (op_dynamic_dimension >= orig_reshape_pair.first.first &&
                        op_dynamic_dimension < orig_reshape_pair.first.second) {
                      auto dynamic_size =
                          parent_->GetDynamicSize(op, {}, op_dynamic_dimension);
                      CHECK_NE(dynamic_size, nullptr);
                      auto hlo_dimension_index = op_dynamic_dimension -
                                                 orig_reshape_pair.first.first +
                                                 reshape_pair.second.first;
                      dynamic_sizes[hlo_dimension_index] = dynamic_size;
                    }
                    return OkStatus();
                  }));
              return true;
            }
          }
        }
        for (auto operand : op->mutable_operands()) {
          if (find_reverse_past_reshape(operand, reshape_pair,
                                        dynamic_dimension_size)) {
            return true;
          }
          VLOG(3) << "Checking " << operand->ToString() << "\n";
        }
        return false;
      };
  // First scan to see if we need to decompose the dynamic reshape
  // into a flatten-unflatten pair. If so, find the dynamic
  // dimension using hlo->inferred_dimension() and calculate the
  // dynamic size for that dimension.
  absl::flat_hash_map<int64_t, ReshapeGroupPair> reshape_group_pairs;
  // For a reshape we need the inferred_dimension to be present to
  // disambiguate dynamic dimensions of hlo.
  // HloOpcode::kDynamicReshape on the other hand allows more
  // precise specification of dynamic dimensions of hlo's shape.
  bool need_flatten_unflatten =
      hlo->inferred_dimension() != -1 &&
      hlo->shape().dimensions(hlo->inferred_dimension()) == 1;
  TF_RETURN_IF_ERROR(ForEachOperandDynamicDimension(
      hlo,
      [&](HloInstruction* operand, ShapeIndex index,
          int64_t input_dynamic_dimension, int64_t operand_index,
          HloInstruction* operand_dynamic_size) -> Status {
        auto reshape_pair =
            find_reshape_group_pair(hlo, input_dynamic_dimension);
        reshape_group_pairs[input_dynamic_dimension] = reshape_pair;
        if (reshape_group_pair_needs_flatten(reshape_pair)) {
          need_flatten_unflatten = true;
        }
        return OkStatus();
      }));
  if (need_flatten_unflatten) {
    if (hlo->inferred_dimension() != -1) {
      HloInstruction* operand = hlo->mutable_operand(0);
      HloComputation* comp = hlo->parent();
      HloInstruction* dynamic_size = comp->AddInstruction(
          HloInstruction::CreateConstant(LiteralUtil::CreateR0<int32_t>(1)));
      int64_t static_size = 1;
      for (int64_t i = 0; i < operand->shape().rank(); i++) {
        HloInstruction* dynamic_dim_size =
            parent_->GetDynamicSize(operand, {}, i);
        if (dynamic_dim_size == nullptr) {
          static_size *= operand->shape().dimensions(i);
        } else {
          dynamic_size = comp->AddInstruction(HloInstruction::CreateBinary(
              dynamic_size->shape(), HloOpcode::kMultiply, dynamic_size,
              dynamic_dim_size));
        }
      }
      HloInstruction* static_size_hlo =
          comp->AddInstruction(HloInstruction::CreateConstant(
              LiteralUtil::CreateR0<int32_t>(static_size)));
      // Total dynamic shape size.
      dynamic_size = comp->AddInstruction(HloInstruction::CreateBinary(
          dynamic_size->shape(), HloOpcode::kMultiply, dynamic_size,
          static_size_hlo));

      int64_t size_without_inferred_dim =
          ShapeUtil::ElementsIn(hlo->shape()) /
          hlo->shape().dimensions(hlo->inferred_dimension());
      HloInstruction* size_without_inferred_dim_hlo =
          comp->AddInstruction(HloInstruction::CreateConstant(
              LiteralUtil::CreateR0<int32_t>(size_without_inferred_dim)));
      dynamic_size = comp->AddInstruction(HloInstruction::CreateBinary(
          dynamic_size->shape(), HloOpcode::kDivide, dynamic_size,
          size_without_inferred_dim_hlo));
      dynamic_sizes[hlo->inferred_dimension()] = dynamic_size;
      VLOG(3)
          << "Need to decompose a dynamic reshape to flatten-unflatten pair. "
          << comp->parent()->ToString();
      SetDynamicSizes(hlo, {}, dynamic_sizes);
      return OkStatus();
    }
    return Internal(
        "Need inferred dimension to be set to "
        "flatten-unflatten pair. %s",
        hlo->ToString());
  }

  TF_RETURN_IF_ERROR(ForEachOperandDynamicDimension(
      hlo,
      [&](HloInstruction* operand, ShapeIndex index,
          int64_t input_dynamic_dimension, int64_t operand_index,
          HloInstruction* operand_dynamic_size) -> Status {
        HloInstruction* const reshape = hlo;
        if (reshape->shape().rank() == 0) {
          VLOG(0) << "Reshaping a dynamic dimension into a scalar, which has "
                     "undefined behavior when input size is 0. The offending "
                     "instruction is: "
                  << reshape->ToString();
          return OkStatus();
        }
        auto iter = reshape_group_pairs.find(input_dynamic_dimension);
        CHECK(iter != reshape_group_pairs.end());
        ReshapeGroupPair reshape_group_pair = iter->second;
        auto output_dim_start = reshape_group_pair.second.first,
             output_dim_end = reshape_group_pair.second.second;
        int64_t output_dynamic_dimension = -1;

        if (operand->shape().dimensions(input_dynamic_dimension) == 1) {
          // If dynamic dimension is 1, it can only be most-major or
          // most-minor.
          if (input_dynamic_dimension == 0) {
            output_dynamic_dimension = 0;
          } else if (input_dynamic_dimension == operand->shape().rank() - 1) {
            output_dynamic_dimension = reshape->shape().rank() - 1;
          }

          if (output_dynamic_dimension == -1) {
            return Unimplemented(
                "Dynamic degenerated dimension that's not most-minor nor "
                "most-major is not supported %s",
                reshape->ToString());
          }
        }

        if (output_dynamic_dimension == -1 &&
            output_dim_end - output_dim_start == 1) {
          // Only one possible output dimension.
          output_dynamic_dimension = output_dim_start;
        }

        if (output_dynamic_dimension == -1 &&
            output_dim_end - output_dim_start > 1) {
          // One input dimension is splitted into multiple output dimensions.
          // Output dimension is decomposed from input most major dimension.
          // In this case, we don't know which one is dynamic, e.g., when we
          // have:
          //
          //           [<=a/c, c, b]
          //              | Reshape
          //           [<=a, b] // a is dynamic, has to be multiple of c.
          //             |  Reshape
          // [1, 1, ... , a/c, c, b]
          //
          // Any dimension from the first '1' to 'a/c' can be dynamic.
          //
          // We use the following logics to disambiguate:
          // 1. If the user sets "inferred_dimension", then use that as
          // dynamic dimension.
          // 2. If the one dimension in the reshape is dynamic, use that as
          // dynamic dimension.
          // E.g.:
          //     [<=4]
          //      |
          //   reshape
          //      |
          //   [1, <=2, 2]
          // We use second dim as dynamic dimension.
          //
          // 3. If all logics above cannot disambiguate, e.g.,:
          //
          //     [<=1]
          //      |
          //   reshape
          //      |
          //   [1, 1, 1]
          //
          //   We bail out and return an error.
          // TODO(yunxing): Further simplify this, remove 1. and fully rely
          // on 2.
          output_dynamic_dimension = reshape->inferred_dimension();
          if (output_dynamic_dimension == -1) {
            // Try find dynamic dimension from the result shape.
            for (int64_t i = output_dim_start; i < output_dim_end; ++i) {
              if (reshape->shape().is_dynamic_dimension(i)) {
                output_dynamic_dimension = i;
              }
            }
          }

          if (output_dynamic_dimension == -1) {
            std::vector<int64_t> output_non_degenerated;
            for (int64_t i = output_dim_start; i < output_dim_end; ++i) {
              if (reshape->shape().dimensions(i) != 1) {
                output_non_degenerated.push_back(i);
              }
            }
            if (output_non_degenerated.size() == 1) {
              output_dynamic_dimension = output_non_degenerated[0];
            }
          }

          if (output_dynamic_dimension == -1 &&
              find_reverse_past_reshape(
                  hlo->mutable_operand(0), reshape_group_pair,
                  hlo->mutable_operand(0)->shape().dimensions(
                      input_dynamic_dimension))) {
            return OkStatus();
          }
          if (output_dynamic_dimension == -1) {
            return InvalidArgument(
                "Reshape's input dynamic dimension is decomposed into "
                "multiple output dynamic dimensions, but the constraint is "
                "ambiguous and XLA can't infer the output dimension %s. ",
                hlo->ToString());
          }
        }

        CHECK_NE(output_dynamic_dimension, -1);
        const int64_t input_dim_size =
            operand->shape().dimensions(input_dynamic_dimension);
        const int64_t output_dim_size =
            reshape->shape().dimensions(output_dynamic_dimension);
        VLOG(2) << "input_dim_size: " << input_dim_size
                << " output_dim_size: " << output_dim_size;

        if (input_dim_size == output_dim_size) {
          // Simply forward dynamic dimension.
          dynamic_sizes[output_dynamic_dimension] = operand_dynamic_size;
        }

        if (input_dim_size > output_dim_size) {
          TF_RET_CHECK(input_dim_size % output_dim_size == 0)
              << reshape->ToString();
          const int64_t divisor = input_dim_size / output_dim_size;
          HloInstruction* divisor_hlo =
              hlo->parent()->AddInstruction(HloInstruction::CreateConstant(
                  LiteralUtil::CreateR0<int32_t>(divisor)));

          HloInstruction* new_dynamic_size =
              hlo->parent()->AddInstruction(HloInstruction::CreateBinary(
                  operand_dynamic_size->shape(), HloOpcode::kDivide,
                  operand_dynamic_size, divisor_hlo));

          dynamic_sizes[output_dynamic_dimension] = new_dynamic_size;
        }

        if (input_dim_size < output_dim_size) {
          // Input dimension is combined with other input dimensions.
          //
          // Adjust the output size by the ratio of dynamic_input_dim /
          // static_input_dim.
          //
          // For example if we have  [<=3, 3] -> [9], if the dynamic size is 2,
          // the new output dynamic isze is 9 / 3 * 2 = 6.
          //
          // If it turns out the second dimension is also dynamic:
          // [<=3, <=3] -> [9], and the dynamic size is also 2, the new output
          // dynamic size is 6 / 3 * 2 = 4.
          //
          //
          HloInstruction* output_dynamic_size =
              dynamic_sizes[output_dynamic_dimension];
          if (output_dynamic_size == nullptr) {
            output_dynamic_size =
                hlo->parent()->AddInstruction(HloInstruction::CreateConstant(
                    LiteralUtil::CreateR0<int32_t>(output_dim_size)));
          }
          HloInstruction* divisor_hlo = hlo->parent()->AddInstruction(
              HloInstruction::CreateConstant(LiteralUtil::CreateR0<int32_t>(
                  operand->shape().dimensions(input_dynamic_dimension))));

          HloInstruction* new_dynamic_size =
              hlo->parent()->AddInstruction(HloInstruction::CreateBinary(
                  output_dynamic_size->shape(), HloOpcode::kDivide,
                  output_dynamic_size, divisor_hlo));

          new_dynamic_size =
              hlo->parent()->AddInstruction(HloInstruction::CreateBinary(
                  output_dynamic_size->shape(), HloOpcode::kMultiply,
                  new_dynamic_size, operand_dynamic_size));
          dynamic_sizes[output_dynamic_dimension] = new_dynamic_size;
        }

        return OkStatus();
      }));

  SetDynamicSizes(hlo, {}, dynamic_sizes);

  return OkStatus();
}

Status DynamicDimensionInferenceVisitor::HandleReduceWindow(
    HloInstruction* hlo) {
  if (!CanInfer(hlo)) {
    return OkStatus();
  }
  ShapeTree<absl::InlinedVector<HloInstruction*, 2>> dynamic_sizes(
      hlo->shape());
  TF_RETURN_IF_ERROR(ForEachOperandDynamicDimension(
      hlo, [&](HloInstruction* operand, ShapeIndex index, int64_t dimension,
               int64_t operand_index, HloInstruction* dynamic_size) {
        auto* reduce_window = Cast<HloReduceWindowInstruction>(hlo);
        const WindowDimension& window_dim =
            reduce_window->window().dimensions(dimension);

        if (operand_index >= reduce_window->input_count()) {
          // Init values doesn't have dynamic size.
          return OkStatus();
        }

        if (!window_util::IsTrivialWindowDimension(window_dim)) {
          DynamicWindowDims dynamic_window_dims = GetWindowedOutputSize(
              dynamic_size, window_dim.size(), window_dim.window_dilation(),
              window_dim.stride(), PaddingType::PADDING_VALID);
          dynamic_size = dynamic_window_dims.output_size;
        }

        // The dimensions of all data operands of a variadic reduce window have
        // to be the same.  This means that if one operand of variadic
        // reduce has a dynamic dimension, we set all outputs to use the
        // same dynamic size in corresponding dimensions.
        ShapeUtil::ForEachSubshape(
            reduce_window->shape(),
            [&](const Shape& subshape, ShapeIndex reduce_window_result_index) {
              if (!ShapeUtil::IsLeafIndex(reduce_window->shape(),
                                          reduce_window_result_index)) {
                return;
              }
              auto* leaf_dynamic_sizes =
                  dynamic_sizes.mutable_element(reduce_window_result_index);
              leaf_dynamic_sizes->resize(subshape.rank(), nullptr);
              (*leaf_dynamic_sizes)[dimension] = dynamic_size;
            });

        return OkStatus();
      }));
  dynamic_sizes.ForEachElement(
      [&](const ShapeIndex& shape_index,
          const absl::InlinedVector<HloInstruction*, 2> sizes) {
        if (sizes.empty()) {
          return;
        }
        SetDynamicSizes(hlo, shape_index, sizes);
      });
  return OkStatus();
}

Status DynamicDimensionInferenceVisitor::HandleSelectAndScatter(
    HloInstruction* hlo) {
  if (!CanInfer(hlo)) {
    return OkStatus();
  }
  return ForEachOperandDynamicDimension(
      hlo, [&](HloInstruction* operand, ShapeIndex index, int64_t dimension,
               int64_t operand_index, HloInstruction* dynamic_size) {
        if (operand_index == 1) {
          // Operand 0 (input) determines dynamic output size. We ignore the
          // dynamic size in the operand 1 (output gradient).
          return OkStatus();
        }
        SetDynamicSize(hlo, {}, dimension, dynamic_size);

        return OkStatus();
      });
}

Status DynamicDimensionInferenceVisitor::HandleSlice(HloInstruction* hlo) {
  if (!CanInfer(hlo)) {
    return OkStatus();
  }
  return ForEachOperandDynamicDimension(
      hlo,
      [&](HloInstruction* operand, ShapeIndex /*index*/, int64_t dimension,
          int64_t /*operand_index*/, HloInstruction* dynamic_size) -> Status {
        int64_t start = hlo->slice_starts(dimension);
        int64_t limit = hlo->slice_limits(dimension);
        int64_t stride = hlo->slice_strides(dimension);
        int64_t size = CeilOfRatio<int64_t>(limit - start, stride);
        if (size == 1) {
          TF_RET_CHECK(!hlo->shape().is_dynamic_dimension(dimension));
          // Slicing a single element out eliminates the dynamic dimension.
          return OkStatus();
        }

        TF_RET_CHECK(hlo->shape().is_dynamic_dimension(dimension));
        if (start != 0) {
          dynamic_size = hlo->AddInstruction(HloInstruction::CreateBinary(
              dynamic_size->shape(), HloOpcode::kSubtract, dynamic_size,
              hlo->AddInstruction(HloInstruction::CreateConstant(
                  LiteralUtil::CreateR0<int32_t>(start)))));
        }
        if (stride != 1) {
          dynamic_size = hlo->AddInstruction(HloInstruction::CreateBinary(
              dynamic_size->shape(), HloOpcode::kAdd, dynamic_size,
              hlo->AddInstruction(HloInstruction::CreateConstant(
                  LiteralUtil::CreateR0<int32_t>(stride - 1)))));
          dynamic_size = hlo->AddInstruction(HloInstruction::CreateBinary(
              dynamic_size->shape(), HloOpcode::kDivide, dynamic_size,
              hlo->AddInstruction(HloInstruction::CreateConstant(
                  LiteralUtil::CreateR0<int32_t>(stride)))));
        }
        SetDynamicSize(hlo, {}, dimension, dynamic_size);

        return OkStatus();
      });
}

Status DynamicDimensionInferenceVisitor::HandleDynamicSlice(
    HloInstruction* hlo) {
  if (!CanInfer(hlo)) {
    return OkStatus();
  }
  return ForEachOperandDynamicDimension(
      hlo,
      [&](HloInstruction* operand, ShapeIndex index, int64_t dimension,
          int64_t operand_index, HloInstruction* dynamic_size) -> Status {
        // Slicing a single element out kills the dynamic dimension.
        if (hlo->shape().dimensions(dimension) == 1) {
          return OkStatus();
        }
        if (hlo->shape().dimensions(dimension) !=
            hlo->operand(0)->shape().dimensions(dimension)) {
          return Unimplemented(
              "Dynamic dimension propagation on DynamicSlice where a partial "
              "dimension is selected %s",
              hlo->ToString());
        }

        // Only the base operand should be dynamic (since the rest are scalars).
        TF_RET_CHECK(operand_index == 0);

        TF_RET_CHECK(index.empty());
        SetDynamicSize(hlo, {}, dimension, dynamic_size);

        return OkStatus();
      });
}

Status DynamicDimensionInferenceVisitor::HandleDynamicUpdateSlice(
    HloInstruction* hlo) {
  if (!CanInfer(hlo)) {
    return OkStatus();
  }
  absl::InlinedVector<HloInstruction*, 2> output_dynamic_sizes(
      hlo->shape().rank(), nullptr);
  TF_RETURN_IF_ERROR(ForEachOperandDynamicDimension(
      hlo,
      [&](HloInstruction* operand, ShapeIndex index, int64_t dimension,
          int64_t operand_index, HloInstruction* dynamic_size) -> Status {
        TF_RET_CHECK(index.empty());

        if (hlo->shape().dimensions(dimension) !=
            hlo->operand(0)->shape().dimensions(dimension)) {
          return Unimplemented(
              "Dynamic dimension propagation on DynamicUpdateSlice where a "
              "partial dimension is selected %s",
              hlo->ToString());
        }

        if (operand_index == 1 &&
            hlo->operand(1)->shape().dimensions(dimension) <
                hlo->operand(0)->shape().dimensions(dimension)) {
          // DUS(input=[A], update=[<=B])
          //
          // If update dim is smaller than input dim (B < A) , then we are doing
          // a partial update, no need to set the output dynamic dimension.
          //
          // The dynamic shape in `update` doesn't change output dynamic shape.
          hlo->mutable_shape()->set_dynamic_dimension(dimension, false);
          return OkStatus();
        }

        output_dynamic_sizes[dimension] = dynamic_size;

        return OkStatus();
      }));
  SetDynamicSizes(hlo, {}, output_dynamic_sizes);
  return OkStatus();
}

Status DynamicDimensionInferenceVisitor::HandleReverse(HloInstruction* hlo) {
  return PassThroughDynamicDimension(hlo);
}

Status DynamicDimensionInferenceVisitor::HandleGather(HloInstruction* hlo) {
  if (!CanInfer(hlo)) {
    return OkStatus();
  }
  absl::InlinedVector<HloInstruction*, 2> output_dynamic_sizes(
      hlo->shape().rank(), nullptr);
  TF_RETURN_IF_ERROR(ForEachOperandDynamicDimension(
      hlo,
      [&](HloInstruction* operand, ShapeIndex /*index*/,
          int64_t input_dynamic_dimension, int64_t operand_index,
          HloInstruction* dynamic_size) -> Status {
        const GatherDimensionNumbers& gather_dims =
            hlo->gather_dimension_numbers();
        if (operand_index == 0) {
          if (hlo->gather_slice_sizes()[input_dynamic_dimension] == 1) {
            // Gathering a size 1 dimension out of a dynamic dimension removes
            // the dynamicity.
            return OkStatus();
          }
          if (hlo->gather_slice_sizes()[input_dynamic_dimension] ==
              operand->shape().dimensions(input_dynamic_dimension)) {
            int64_t operand_dimension = 0;
            for (int64_t output_dimension : gather_dims.offset_dims()) {
              TF_RET_CHECK(output_dimension < hlo->shape().rank());
              while (operand_dimension < operand->shape().rank() &&
                     absl::c_linear_search(gather_dims.collapsed_slice_dims(),
                                           operand_dimension)) {
                ++operand_dimension;
              }
              TF_RET_CHECK(operand_dimension < operand->shape().rank());
              if (operand_dimension == input_dynamic_dimension) {
                output_dynamic_sizes[output_dimension] = dynamic_size;
                return OkStatus();
              }
              ++operand_dimension;
            }
            return Internal("Invalid instruction: %s", hlo->ToString());
          }
          return Unimplemented(
              "Detects a dynamic dimension on the data input of gather, which "
              "is not supported: %s, %lld",
              hlo->ToString(), input_dynamic_dimension);
        }
        int64_t indices_rank = hlo->operand(1)->shape().rank();
        if (gather_dims.index_vector_dim() == indices_rank) {
          ++indices_rank;
        }
        int64_t output_rank = hlo->shape().rank();

        // indices_dim is an iterator over indices dimensions.
        int64_t indices_dim = 0;
        // Find the corresponding batch dimension in the output.
        for (int64_t output_dim = 0; output_dim < output_rank; ++output_dim) {
          if (!absl::c_linear_search(gather_dims.offset_dims(), output_dim)) {
            // Skips index vector dimension.
            if (indices_dim == gather_dims.index_vector_dim()) {
              indices_dim++;
            }
            if (indices_dim++ == input_dynamic_dimension) {
              output_dynamic_sizes[output_dim] = dynamic_size;
              return OkStatus();
            }
          }
        }
        CHECK(indices_dim == indices_rank);

        return Unimplemented(
            "Detects a non-batch dynamic dimension of gather, "
            "which is not supported: %s",
            hlo->ToString());
      }));
  SetDynamicSizes(hlo, {}, output_dynamic_sizes);
  return OkStatus();
}

Status DynamicDimensionInferenceVisitor::HandleConditional(
    HloInstruction* hlo) {
  if (!CanInfer(hlo)) {
    return OkStatus();
  }
  // Conditionals are handled by producing additional inputs and outputs of
  // the conditional instruction.
  std::vector<HloComputation*> new_branch_computations;
  std::vector<HloInstruction*> new_operands;
  // If the output of the conditional contains dynamic dimension. We send
  // dynamic dimension size out by adding additional root element. A mapping
  // from the root instruction's dynamic dimension index (represented by a shape
  // index as output index and a int64_t dimension number) to output index
  // (represented by an int64_t) is tracked for the conditional instruction (all
  // branches should have the same mapping).
  ShapeTree<absl::flat_hash_map<int64_t, int64_t>> dynamic_output_mapping(
      hlo->shape());

  bool need_rewrite = false;
  for (int64_t branch_index = 0; branch_index < hlo->branch_count();
       ++branch_index) {
    std::vector<HloInstruction*> operands_to_add;

    absl::flat_hash_map<HloInstruction*, int64_t>
        dynamic_size_to_operand_id_index_map;
    // Only look at branch_index + 1, the correct operand index for a
    // given branch.
    const int64_t operand_index = branch_index + 1;

    int operand_count =
        hlo->operand(operand_index)->shape().tuple_shapes_size();
    // Prepare to pass dynamic dimension into the new computation and add
    // dynamic dimension sizes as parameters to the new tuple.
    TF_RETURN_IF_ERROR(ForEachDynamicDimensionInOperand(
        hlo, operand_index,
        [&](HloInstruction*, ShapeIndex, int64_t, int64_t,
            HloInstruction* dynamic_size) -> Status {
          TF_RET_CHECK(hlo->operand(operand_index)->shape().IsTuple())
              << "Only tuple typed inputs can have dynamic dimension. Please "
                 "file a bug against XLA team.";
          const HloInstruction* tuple_operand = hlo->operand(operand_index);
          for (int64_t i = 0; i < tuple_operand->operand_count(); ++i) {
            // If the dynamic size is already an operand to the computation,
            // skip adding it to the computation input again.
            if (dynamic_size == tuple_operand->operand(i)) {
              dynamic_size_to_operand_id_index_map[dynamic_size] = i;
              return OkStatus();
            }
          }
          auto iter = dynamic_size_to_operand_id_index_map.find(dynamic_size);
          if (iter == dynamic_size_to_operand_id_index_map.end()) {
            operands_to_add.push_back(dynamic_size);
            dynamic_size_to_operand_id_index_map[dynamic_size] =
                operand_count++;
          }
          return OkStatus();
        }));

    HloInstruction* original_input = hlo->mutable_operand(operand_index);
    HloComputation* branch_computation = hlo->branch_computation(branch_index);

    HloComputation* new_computation = branch_computation;
    CallInliner::InlinedInstructionMap inline_map;
    HloInstruction* new_operand = hlo->mutable_operand(operand_index);
    Shape new_param_shape =
        branch_computation->parameter_instruction(0)->shape();
    if (!operands_to_add.empty()) {
      TF_RET_CHECK(original_input->shape().IsTuple());
      need_rewrite = true;
      new_operand = TupleUtil::AppendSuffix(original_input, operands_to_add);
      for (HloInstruction* operand : operands_to_add) {
        ShapeUtil::AppendShapeToTuple(operand->shape(), &new_param_shape);
      }
      TF_ASSIGN_OR_RETURN(
          std::tie(new_computation, inline_map),
          WidenComputation(branch_computation, new_param_shape));
    }
    // Set the dynamic dimensions for the newly created branch computation's
    // parameters so that the hlos inside the computation can see dynamic
    // dimensions.
    DynamicParameterBinding dynamic_parameter_binding;
    TF_RETURN_IF_ERROR(ForEachDynamicDimensionInOperand(
        hlo, operand_index,
        [&](HloInstruction*, ShapeIndex index, int64_t dimension,
            int64_t operand_index, HloInstruction* dynamic_size) {
          DynamicParameterBinding::DynamicSizeParameter dynamic_parameter{
              0, {dynamic_size_to_operand_id_index_map[dynamic_size]}};
          DynamicParameterBinding::DynamicDimension dynamic_dimension{
              0, {index}, dimension};
          TF_RETURN_IF_ERROR(dynamic_parameter_binding.Bind(dynamic_parameter,
                                                            dynamic_dimension));

          return OkStatus();
        }));
    VLOG(2) << "dynamic_parameter_binding for conditional branch"
            << dynamic_parameter_binding;

    for (auto [old_inst, new_inst] : inline_map) {
      parent_->CopyMapping(
          /*from=*/old_inst,
          /*to=*/new_inst,
          /*dynamic_size_map=*/&inline_map);
    }

    TF_ASSIGN_OR_RETURN(
        bool changed,
        DynamicDimensionInferenceVisitor::Run(
            new_computation, dataflow_analysis_, dynamic_parameter_binding,
            parent_, custom_call_handler_, shape_check_mode_,
            assertion_generator_));
    if (changed) {
      MarkAsChanged();
    }

    new_branch_computations.push_back(new_computation);
    new_operands.push_back(new_operand);
  }
  int tuple_count = hlo->shape().tuple_shapes_size();
  // The dynamism of the output of branches can be different.
  // E.g.,
  //   true_branch  (s32[<=4])
  //   false_branch (s32[4])
  //
  // The following loop populates dynamic_output_mapping and account for
  // dynamism across all branches.
  ShapeUtil::ForEachSubshape(
      hlo->shape(), [&](const Shape& subshape, const ShapeIndex& index) {
        if (!subshape.IsArray()) {
          return;
        }
        for (int64_t i = 0; i < subshape.rank(); ++i) {
          for (int64_t j = 0; j < new_branch_computations.size(); ++j) {
            HloInstruction* dynamic_size = parent_->GetDynamicSize(
                new_branch_computations[j]->root_instruction(), index, i);
            if (dynamic_size) {
              if (dynamic_output_mapping.element(index).contains(i)) {
                continue;
              }
              dynamic_output_mapping.mutable_element(index)->emplace(
                  i, tuple_count++);
            }
          }
        }
      });
  for (int64_t branch_index = 0; branch_index < hlo->branch_count();
       ++branch_index) {
    std::vector<HloInstruction*> hlos_to_add_in_root;
    // There may be some dynamic dimensions coming out of the computation, wire
    // that into the root instruction as additional tuple elements.
    ShapeUtil::ForEachSubshape(
        hlo->shape(), [&](const Shape& subshape, const ShapeIndex& index) {
          if (!subshape.IsArray()) {
            return;
          }
          for (int64_t i = 0; i < subshape.rank(); ++i) {
            if (dynamic_output_mapping.element(index).contains(i)) {
              HloInstruction* dynamic_size = parent_->GetDynamicSize(
                  new_branch_computations[branch_index]->root_instruction(),
                  index, i);
              if (dynamic_size) {
                hlos_to_add_in_root.push_back(dynamic_size);
              } else {
                HloInstruction* constant_size =
                    new_branch_computations[branch_index]->AddInstruction(
                        HloInstruction::CreateConstant(
                            LiteralUtil::CreateR0<int32_t>(
                                subshape.dimensions(i))));
                hlos_to_add_in_root.push_back(constant_size);
              }
            }
          }
        });

    VLOG(2) << "hlos_to_add_in_root:" << hlos_to_add_in_root.size();
    if (!hlos_to_add_in_root.empty()) {
      need_rewrite = true;
      HloInstruction* new_branch_root = TupleUtil::AppendSuffix(
          new_branch_computations[branch_index]->root_instruction(),
          hlos_to_add_in_root);
      new_branch_computations[branch_index]->set_root_instruction(
          new_branch_root,
          /*accept_different_shape=*/true);
    }
  }

  if (!need_rewrite) {
    return OkStatus();
  }
  // Create a new conditional with the new operations and computations.
  HloInstruction* new_conditional =
      hlo->parent()->AddInstruction(HloInstruction::CreateConditional(
          new_branch_computations[0]->root_instruction()->shape(),
          hlo->mutable_operand(0), new_branch_computations, new_operands));

  HloInstruction* new_conditional_extracted = TupleUtil::ExtractPrefix(
      new_conditional, hlo->shape().tuple_shapes_size());
  // Now set the dynamic dimensions of the newly created conditional.
  dynamic_output_mapping.ForEachElement(
      [&](const ShapeIndex& index,
          const absl::flat_hash_map<int64_t, int64_t>& dim_to_output) {
        for (auto iter : dim_to_output) {
          int64_t dim = iter.first;
          int64_t output_index = iter.second;
          HloInstruction* dynamic_size = hlo->parent()->AddInstruction(
              HloInstruction::CreateGetTupleElement(
                  ShapeUtil::MakeScalarShape(S32), new_conditional,
                  output_index));
          SetDynamicSize(new_conditional, index, dim, dynamic_size,
                         /*clear_dynamic_dimension=*/false);
          SetDynamicSize(new_conditional_extracted, index, dim, dynamic_size,
                         /*clear_dynamic_dimension=*/false);
        }
      });

  TF_RETURN_IF_ERROR(hlo->ReplaceAllUsesWith(new_conditional_extracted));
  // Remove the original instruction even if has side-effects.
  TF_RETURN_IF_ERROR(hlo->parent()->RemoveInstruction(hlo));
  SetVisited(*new_conditional);
  SetVisited(*new_conditional_extracted);
  MarkAsChanged();
  return OkStatus();
}

Status DynamicDimensionInferenceVisitor::HandleMap(HloInstruction* hlo) {
  if (!CanInfer(hlo)) {
    return OkStatus();
  }
  return HandleElementwiseNary(hlo);
}

Status DynamicDimensionInferenceVisitor::HandleScatter(HloInstruction* hlo) {
  if (!CanInfer(hlo)) {
    return OkStatus();
  }
  return ForEachOperandDynamicDimension(
      hlo,
      [&](HloInstruction* operand, ShapeIndex dynamic_index, int64_t dimension,
          int64_t operand_index,
          HloInstruction* operand_dynamic_size) -> absl::Status {
        if (operand_index == 0) {
          SetDynamicSize(hlo, {}, dimension, operand_dynamic_size);
          return OkStatus();
        }

        const ScatterDimensionNumbers& scatter_dims =
            hlo->scatter_dimension_numbers();
        if (operand_index == 2 &&
            absl::c_linear_search(scatter_dims.update_window_dims(),
                                  dimension)) {
          // Dynamic update window dimension is only allowed if it is exactly
          // the same as the corresponding operand dimension.
          std::vector<int64_t> update_window_dims_in_operand;
          for (int64_t i = 0; i < hlo->operand(0)->shape().rank(); ++i) {
            if (absl::c_linear_search(scatter_dims.inserted_window_dims(), i)) {
              continue;
            }
            update_window_dims_in_operand.push_back(i);
          }

          for (int64_t i = 0; i < scatter_dims.update_window_dims_size(); ++i) {
            if (scatter_dims.update_window_dims(i) == dimension) {
              const Shape& operand_shape = hlo->operand(0)->shape();
              const Shape& update_shape = hlo->operand(2)->shape();
              int64_t dim_in_operand = update_window_dims_in_operand[i];
              if (operand_shape.dimensions(dim_in_operand) !=
                  update_shape.dimensions(dimension)) {
                return Unimplemented(
                    "Dynamic dimension of update window dims that are not the "
                    "same as corresponding operand dim is not supported: "
                    "%s : %d : %d : %d",
                    hlo->ToString(), i, update_shape.dimensions(dimension),
                    operand_shape.dimensions(dim_in_operand));
              }
              HloInstruction* base_dynamic_size = parent_->GetDynamicSize(
                  hlo->mutable_operand(0), {}, dim_in_operand);
              // Sometimes the incoming operand dimension is no longer dynamic,
              // Simply return OK in this case.
              if (base_dynamic_size == nullptr ||
                  !operand_shape.is_dynamic_dimension(dim_in_operand)) {
                return OkStatus();
              }
              if (base_dynamic_size != operand_dynamic_size) {
                return Unimplemented(
                    "Dynamic dimension size of update window dims that are not "
                    "the same as corresponding operand dim is not supported: "
                    "%s.\n Dynamic dim size of base: %s, dynamic dim size of "
                    "update: %s",
                    hlo->ToString(), base_dynamic_size->ToString(),
                    operand_dynamic_size->ToString());
              }
            }
          }
        }
        // The dynamic dimension is collapsed and won't show up in the output.
        // Do nothing here.
        return OkStatus();
      });
}

Status DynamicDimensionInferenceVisitor::HandleWhile(HloInstruction* hlo) {
  if (!CanInfer(hlo)) {
    return OkStatus();
  }
  // If the output of the kWhile contains dynamic dimension, we send
  // dynamic dimension size into the while body by adding additional root/body
  // element. A mapping from the root instruction's dynamic dimension index
  // (represented by a shape index as output index and an int64_t dimension
  // number) to output index (represented by an int64_t) is tracked for the
  // while instruction.
  Shape original_shape = hlo->shape();
  ShapeTree<absl::flat_hash_map<int64_t, int64_t>> dynamic_output_mapping(
      original_shape);
  std::vector<HloInstruction*> operands_to_add;
  const int original_tuple_count = original_shape.tuple_shapes_size();
  int operand_count = original_tuple_count;
  // Clean up the result shape
  DynamicParameterBinding binding_for_while;
  TF_RETURN_IF_ERROR(ForEachOperandDynamicDimension(
      hlo,
      [&](HloInstruction* operand, ShapeIndex index, int64_t dim,
          int64_t operand_num, HloInstruction* dynamic_size) -> Status {
        TF_RET_CHECK(operand_num == 0);
        operands_to_add.push_back(dynamic_size);
        dynamic_output_mapping.mutable_element(index)->emplace(dim,
                                                               operand_count);
        DynamicParameterBinding::DynamicDimension dynamic_dimension{
            /*parameter_num=*/0,
            /*parameter_index=*/index,
            /*dimension=*/dim,
        };
        DynamicParameterBinding::DynamicSizeParameter dynamic_size_param{
            /*parameter_num=*/0,
            /*parameter_index=*/{operand_count},
        };
        TF_RETURN_IF_ERROR(
            binding_for_while.Bind(dynamic_size_param, dynamic_dimension));
        ++operand_count;
        return OkStatus();
      }));
  if (operands_to_add.empty()) {
    return OkStatus();
  }

  HloInstruction* old_tuple_operand = hlo->mutable_operand(0);
  HloInstruction* old_body_root = hlo->while_body()->root_instruction();
  // HloInstruction* old_body_parameter =
  //     hlo->while_body()->parameter_instruction(0);
  // HloInstruction* old_condition_parameter =
  //     hlo->while_condition()->parameter_instruction(0);
  TF_ASSIGN_OR_RETURN(WhileUtil::MakeInstructionsLiveInResult result,
                      WhileUtil::MakeInstructionsLiveIn(hlo, operands_to_add));
  TF_RET_CHECK(result.replacement_instr->opcode() == HloOpcode::kTuple);
  // WhileUtil creates a new while hlo and tuple. Update the dynamic size
  // mapping for the newly created tuple.
  HloInstruction* new_tuple_operand =
      result.new_while_instr->mutable_operand(0);
  parent_->CopyMapping(/*from=*/old_tuple_operand,
                       /*to=*/new_tuple_operand);

  hlo = result.new_while_instr;

  // Set the replacement instruction as visited to avoid visiting it again.
  SetVisited(*hlo);

  for (auto [old_inst, new_inst] : result.while_body_instruction_map) {
    parent_->CopyMapping(
        /*from=*/old_inst,
        /*to=*/new_inst,
        /*dynamic_size_map=*/&result.while_body_instruction_map);
  }
  // MakeInstructionsLiveIn does not include the new root tuple in the
  // instruction map, so we have to copy the mapping here.
  parent_->CopyMapping(/*from=*/old_body_root,
                       /*to=*/hlo->while_body()->root_instruction(),
                       &result.while_body_instruction_map);
  for (auto [old_inst, new_inst] : result.while_condition_instruction_map) {
    parent_->CopyMapping(
        /*from=*/old_inst,
        /*to=*/new_inst,
        /*dynamic_size_map=*/&result.while_condition_instruction_map);
  }

  // Rerun inference on the body and condition now that we have added dynamic
  // size parameters.
  TF_RETURN_IF_ERROR(DynamicDimensionInferenceVisitor::Run(
                         hlo->while_body(), dataflow_analysis_,
                         binding_for_while, parent_, custom_call_handler_,
                         shape_check_mode_, assertion_generator_)
                         .status());
  TF_RETURN_IF_ERROR(DynamicDimensionInferenceVisitor::Run(
                         hlo->while_condition(), dataflow_analysis_,
                         binding_for_while, parent_, custom_call_handler_,
                         shape_check_mode_, assertion_generator_)
                         .status());

  // The dynamic dimension size could have been changed in the loop body (e.g, A
  // loop that inserts items in a stack, the stack size increases with each
  // iteration). Rewrite the dynamic dimension size at the root.
  HloInstruction* body_root = hlo->while_body()->root_instruction();
  std::vector<HloInstruction*> new_root_operands(body_root->operand_count(),
                                                 nullptr);

  // Original non-dynamic-dim operands of root are pass-through.
  for (int i = 0; i < original_tuple_count; ++i) {
    new_root_operands[i] =
        body_root->AddInstruction(HloInstruction::CreateGetTupleElement(
            body_root->shape().tuple_shapes(i), body_root, i));
  }
  // Add dynamic dimension size as new outputs of the while loop body.
  TF_RETURN_IF_ERROR(dynamic_output_mapping.ForEachElementWithStatus(
      [&](const ShapeIndex& index,
          const absl::flat_hash_map<int64_t, int64_t>& dim_to_size) -> Status {
        for (auto [dimension, output_index] : dim_to_size) {
          TF_RET_CHECK(new_root_operands[output_index] == nullptr);
          HloInstruction* dynamic_size =
              parent_->GetDynamicSize(body_root, index, dimension);
          TF_RET_CHECK(dynamic_size != nullptr);
          new_root_operands[output_index] = dynamic_size;
        }
        return OkStatus();
      }));
  for (auto operand : new_root_operands) {
    TF_RET_CHECK(operand != nullptr);
  }
  HloInstruction* new_body_root = hlo->while_body()->AddInstruction(
      HloInstruction::CreateTuple(new_root_operands));
  for (int i = 0; i < original_tuple_count; ++i) {
    TF_RETURN_IF_ERROR(ForEachDynamicDimension(
        body_root,
        [&](ShapeIndex index, int64_t dimension,
            HloInstruction* dynamic_size) -> Status {
          SetDynamicSize(new_body_root, index, dimension, dynamic_size);
          if (index.empty() || index.front() != i) {
            return OkStatus();
          }
          index.pop_front();
          SetDynamicSize(new_root_operands[i], index, dimension, dynamic_size);
          return OkStatus();
        }));
  }
  hlo->while_body()->set_root_instruction(new_body_root);
  MarkAsChanged();

  // Record the dynamic sizes of while loop output.
  return dynamic_output_mapping.ForEachElementWithStatus(
      [&](const ShapeIndex& index,
          const absl::flat_hash_map<int64_t, int64_t>& dim_to_size) -> Status {
        for (auto [dimension, output_index] : dim_to_size) {
          HloInstruction* dynamic_size = hlo->AddInstruction(
              HloInstruction::CreateGetTupleElement(hlo, output_index));
          SetDynamicSize(result.replacement_instr, index, dimension,
                         dynamic_size);
          ShapeUtil::GetMutableSubshape(hlo->mutable_shape(), index)
              ->set_dynamic_dimension(dimension, false);
          TF_RET_CHECK(!index.empty());
          HloInstruction* gte =
              result.replacement_instr->mutable_operand(index.front());
          TF_RET_CHECK(gte->opcode() == HloOpcode::kGetTupleElement);
          TF_RET_CHECK(gte->operand(0) == hlo);
          ShapeUtil::GetMutableSubshape(gte->mutable_shape(),
                                        ShapeIndexView(index).subspan(1))
              ->set_dynamic_dimension(dimension, false);
        }
        return OkStatus();
      });
}

Status DynamicDimensionInferenceVisitor::HandleParameter(HloInstruction* hlo) {
  if (hlo->parent()->IsEntryComputation()) {
    TF_RET_CHECK(param_bindings_.empty());
    return InsertPadToStaticOnInstruction(hlo);
  }

  return param_bindings_.ForEachBinding(
      [&](const DynamicParameterBinding::DynamicSizeParameter& dynamic_size,
          const DynamicParameterBinding::DynamicDimension& dynamic_dimension)
          -> Status {
        if (dynamic_dimension.parameter_num == hlo->parameter_number()) {
          SetDynamicSize(
              hlo, dynamic_dimension.parameter_index,
              dynamic_dimension.dimension,
              TupleUtil::AddGetTupleElements(HloPosition{
                  /*instruction=*/hlo->parent()->parameter_instruction(
                      dynamic_size.parameter_num),
                  /*index=*/dynamic_size.parameter_index,
              }));
        }
        return OkStatus();
      });
}

Status DynamicDimensionInferenceVisitor::HandleInfeed(HloInstruction* hlo) {
  return InsertPadToStaticOnInstruction(hlo);
}

Status DynamicDimensionInferenceVisitor::ForEachDynamicDimension(
    HloInstruction* inst, const DynamicDimensionFn& fn) {
  auto iter = parent_->per_hlo_dynamic_dimensions_.find(inst);
  if (iter != parent_->per_hlo_dynamic_dimensions_.end()) {
    for (auto& dynamic_dimension : iter->second) {
      HloInstruction* dynamic_size = parent_->GetDynamicSize(
          dynamic_dimension.inst, dynamic_dimension.index,
          dynamic_dimension.dim);
      TF_RETURN_IF_ERROR(
          fn(dynamic_dimension.index, dynamic_dimension.dim, dynamic_size));
    }
  }
  return OkStatus();
}

absl::StatusOr<bool> DynamicDimensionInferenceVisitor::RequiresPadToStatic(
    HloInstruction* instr, ShapeIndex shape_index) {
  TF_RET_CHECK(ShapeUtil::IsLeafIndex(instr->shape(), shape_index))
      << instr->shape() << " @ " << shape_index;
  if (ShapeUtil::GetSubshape(instr->shape(), shape_index).is_static()) {
    return false;
  }
  auto uses =
      dataflow_analysis_.GetValueDefinedAt(instr, shape_index).GetUses();
  for (const auto& use : uses) {
    if (use.instruction->opcode() == HloOpcode::kAsyncStart ||
        use.instruction->opcode() == HloOpcode::kAsyncUpdate ||
        use.instruction->opcode() == HloOpcode::kAsyncDone ||
        use.instruction->opcode() == HloOpcode::kCall ||
        use.instruction->opcode() == HloOpcode::kTuple ||
        use.instruction->opcode() == HloOpcode::kGetTupleElement ||
        use.instruction->opcode() == HloOpcode::kConditional) {
      // These uses do not require padding as they do not operate the data.
      continue;
    }
    if (use.instruction->opcode() == HloOpcode::kWhile) {
      TF_RET_CHECK(use.operand_number == 0);
      HloInstruction* root = use.instruction->while_body()->root_instruction();
      if (parent_->HasDynamicDimension(root, use.operand_index)) {
        return true;
      }
      continue;
    }
    if (use.instruction->opcode() == HloOpcode::kSetDimensionSize) {
      // The dynamic size cannot itself be dynamic.
      TF_RET_CHECK(use.operand_number == 0);
      // SetDimensionSize will be removed, so the array must be padded if it
      // is a user of the array.
      return true;
    }
    if (use.instruction->opcode() == HloOpcode::kGetDimensionSize) {
      return true;
    }
    if (use.instruction->opcode() != HloOpcode::kCustomCall ||
        use.instruction->custom_call_target() != "PadToStatic") {
      if (parent_->op_supports_dynamism_handler_ == nullptr) {
        return true;
      }
      if (parent_->op_supports_dynamism_handler_(use.instruction) ==
          OpDynamismSupport::kNoSupport) {
        return true;
      }
    }
  }

  // Don't do pad-to-static.
  return false;
}

// Insert pad-to-static after `inst` if `inst` has dynamic dimensions in it.
// If the instruction produces a tuple, each tuple component will be considered
// independently.
Status DynamicDimensionInferenceVisitor::InsertPadToStaticOnInstruction(
    HloInstruction* inst) {
  if (inst->shape().is_static()) {
    return OkStatus();
  }

  // Decide while leaf arrays need to be padded.
  ShapeTree<bool> needs_pad(inst->shape(), false);
  bool any_needs_pad = false;
  TF_RETURN_IF_ERROR(ShapeUtil::ForEachSubshapeWithStatus(
      inst->shape(), [&](const Shape& subshape, const ShapeIndex& shape_index) {
        if (subshape.IsTuple()) {
          return OkStatus();
        }
        TF_ASSIGN_OR_RETURN(bool do_pad,
                            RequiresPadToStatic(inst, shape_index));
        if (do_pad) {
          *needs_pad.mutable_element(shape_index) = true;
          any_needs_pad = true;
        }
        return OkStatus();
      }));

  if (!any_needs_pad) {
    return OkStatus();
  }

  auto users = inst->users();

  ShapeTree<HloInstruction*> gtes =
      TupleUtil::DisassembleTupleInstruction(inst);

  // Add PadToStatic to the leaf arrays and record the dynamic dimensions.
  ShapeTree<HloInstruction*> padded(inst->shape(), nullptr);
  TF_RETURN_IF_ERROR(ShapeUtil::ForEachSubshapePostOrderWithStatus(
      inst->shape(),
      [&](const Shape& subshape, const ShapeIndex& shape_index) -> Status {
        HloInstruction* element = gtes.element(shape_index);
        SetVisited(*gtes.element(shape_index));
        if (subshape.IsTuple()) {
          absl::InlinedVector<HloInstruction*, 2> children;
          ShapeIndex child_index = shape_index;
          for (int i = 0; i < subshape.tuple_shapes_size(); ++i) {
            child_index.push_back(i);
            children.push_back(padded.element(child_index));
            child_index.pop_back();
          }
          HloInstruction* tuple =
              element->AddInstruction(HloInstruction::CreateVariadic(
                  subshape, HloOpcode::kTuple, children));
          TF_CHECK_OK(ForEachOperandDynamicDimension(
              tuple,
              [&](HloInstruction* operand, ShapeIndex index, int64_t dimension,
                  int64_t operand_index, HloInstruction* dynamic_size) {
                index.push_front(operand_index);
                SetDynamicSize(tuple, index, dimension, dynamic_size);
                return OkStatus();
              }));
          *padded.mutable_element(shape_index) = tuple;
          return OkStatus();
        }
        if (needs_pad.element(shape_index)) {
          // The output shape of pad static is a tuple. The 0th element is the
          // data output, which is the same as input shape, but without
          // dynamic dimensions; i-th element is the dynamic dimension size
          // for i-1th input dimension.
          Shape data_output_shape =
              ShapeUtil::MakeStaticShape(element->shape());  // 0th element.
          Shape output_shape = ShapeUtil::MakeTupleShape({data_output_shape});
          for (int64_t i = 0; i < element->shape().rank(); ++i) {
            ShapeUtil::AppendShapeToTuple(ShapeUtil::MakeScalarShape(S32),
                                          &output_shape);
          }
          HloInstruction* pad_to_static = inst->parent()->AddInstruction(
              HloInstruction::CreateCustomCall(output_shape, {element},
                                               "PadToStatic"),
              absl::StrCat(element->name(), ".padded"));
          SetVisited(*pad_to_static);
          HloInstruction* data_output = inst->parent()->AddInstruction(
              HloInstruction::CreateGetTupleElement(data_output_shape,
                                                    pad_to_static, 0),
              absl::StrCat(element->name(), ".data"));
          SetVisited(*data_output);
          for (int64_t i = 0; i < element->shape().rank(); ++i) {
            if (!element->shape().is_dynamic_dimension(i)) {
              continue;
            }
            HloInstruction* dynamic_size_output =
                inst->parent()->AddInstruction(
                    HloInstruction::CreateGetTupleElement(
                        output_shape.tuple_shapes(i + 1), pad_to_static, i + 1),
                    absl::StrCat(element->name(), ".size"));
            SetVisited(*dynamic_size_output);
            SetDynamicSize(data_output, {}, i, dynamic_size_output,
                           /*clear_dynamic_dimension=*/false);
          }
          *padded.mutable_element(shape_index) = data_output;
        } else {
          *padded.mutable_element(shape_index) = element;
        }
        return OkStatus();
      }));

  HloInstruction* result = padded.element({});

  // Replace all uses of the original instruction with the padded outputs.
  for (auto user : users) {
    for (int64_t i : user->OperandIndices(inst)) {
      TF_RETURN_IF_ERROR(user->ReplaceOperandWith(i, result));
    }
  }
  if (inst->IsRoot()) {
    inst->parent()->set_root_instruction(result);
  }

  MarkAsChanged();

  return OkStatus();
}

Status DynamicDimensionInferenceVisitor::InsertShapeCheck(
    HloInstruction* dim1, HloInstruction* dim2,
    bool support_implicit_broadcast) {
  switch (shape_check_mode_) {
    case DynamicDimensionInference::kIgnore:
      return OkStatus();
    case DynamicDimensionInference::kCompileTime:
      return InvalidArgument(
          "Fail to proof the equality of two dimensions at compile time: "
          "%s vs %s",
          dim1->ToString(), dim2->ToString());
    case DynamicDimensionInference::kRuntime: {
      TF_ASSIGN_OR_RETURN(
          HloInstruction * assertion,
          MakeCompareHlo(Comparison::Direction::kEq, dim1, dim2));
      if (shape_assertion_ == nullptr) {
        shape_assertion_ = assertion;
      } else {
        TF_ASSIGN_OR_RETURN(
            shape_assertion_,
            MakeBinaryHlo(HloOpcode::kAnd, shape_assertion_, assertion));
      }
      return OkStatus();
    }
    default:
      LOG(FATAL) << "Unreachable";
  }
}

Status DynamicDimensionInferenceVisitor::ForEachDynamicDimensionInOperand(
    HloInstruction* inst, int64_t operand_index, OperandDynamicDimensionFn fn) {
  auto iter =
      parent_->per_hlo_dynamic_dimensions_.find(inst->operand(operand_index));
  if (iter != parent_->per_hlo_dynamic_dimensions_.end()) {
    for (auto& dynamic_dimension : iter->second) {
      HloInstruction* dynamic_size = parent_->GetDynamicSize(
          dynamic_dimension.inst, dynamic_dimension.index,
          dynamic_dimension.dim);
      TF_RETURN_IF_ERROR(fn(dynamic_dimension.inst, dynamic_dimension.index,
                            dynamic_dimension.dim, operand_index,
                            dynamic_size));
    }
  }
  return OkStatus();
}

Status DynamicDimensionInferenceVisitor::ForEachOperandDynamicDimension(
    HloInstruction* inst, OperandDynamicDimensionFn fn) {
  for (int64_t operand_index = 0; operand_index < inst->operand_count();
       ++operand_index) {
    TF_RETURN_IF_ERROR(
        ForEachDynamicDimensionInOperand(inst, operand_index, fn));
  }
  return OkStatus();
}

void DynamicDimensionInference::SetDynamicSize(HloInstruction* inst,
                                               const ShapeIndex& index,
                                               int64_t dim,
                                               HloInstruction* size) {
  CHECK_NE(inst, nullptr);
  CHECK_NE(size, nullptr);
  VLOG(1) << "Set dimension inst " << inst->ToString() << " index "
          << index.ToString() << "@" << dim << " to " << size->ToShortString();
  const Shape& subshape = ShapeUtil::GetSubshape(inst->shape(), index);
  CHECK(!subshape.IsTuple()) << "Can't set a tuple shape to dynamic dimension";
  CHECK(dim < subshape.rank() && dim >= 0)
      << "Asked to set invalid dynamic dimension. Shape: "
      << subshape.ToString() << ", Dimension: " << dim;
  DynamicDimension dynamic_dimension{inst, index, dim};
  // If we have already set the dynamic size, it should be the same.
  auto [it, inserted] = dynamic_mapping_.try_emplace(dynamic_dimension, size);
  if (!inserted) {
    CHECK_EQ(size, it->second) << "old: " << it->second->ToShortString()
                               << ", new: " << size->ToShortString();
  }
  auto iter = per_hlo_dynamic_dimensions_.try_emplace(inst);
  iter.first->second.emplace(dynamic_dimension);
}

void DynamicDimensionInference::CopyMapping(
    HloInstruction* from, HloInstruction* to,
    const absl::flat_hash_map<HloInstruction*, HloInstruction*>*
        dynamic_size_map) {
  auto iter = per_hlo_dynamic_dimensions_.find(from);
  if (iter != per_hlo_dynamic_dimensions_.end()) {
    for (auto& dynamic_dimension : iter->second) {
      HloInstruction* dynamic_size =
          GetDynamicSize(dynamic_dimension.inst, dynamic_dimension.index,
                         dynamic_dimension.dim);
      if (dynamic_size_map != nullptr) {
        dynamic_size = dynamic_size_map->at(dynamic_size);
      }
      SetDynamicSize(to, dynamic_dimension.index, dynamic_dimension.dim,
                     dynamic_size);
    }
  }
}

/* static */
absl::StatusOr<DynamicDimensionInference> DynamicDimensionInference::Run(
    HloModule* module, OpSupportsDynamismHandler op_supports_dynamism_handler,
    CustomCallInferenceHandler custom_call_handler,
    ShapeCheckMode shape_check_mode,
    const AssertionGenerator& assertion_generator,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  DynamicDimensionInference inference(
      module, std::move(op_supports_dynamism_handler),
      std::move(custom_call_handler), shape_check_mode, assertion_generator,
      execution_threads);
  TF_RETURN_IF_ERROR(inference.AnalyzeDynamicDimensions());
  return std::move(inference);
}

std::string DynamicDimensionInference::ToString() const {
  std::vector<std::string> pieces;
  pieces.push_back("DynamicDimensionInference: ");
  for (const auto& mapping : dynamic_mapping_) {
    const DynamicDimension& dynamic_dimension = mapping.first;
    pieces.push_back(absl::StrFormat(
        " -- instruction %s at %s has dim %lld as dynamic"
        " dimension, which is represented by instruction %s",
        dynamic_dimension.inst->ToString(), dynamic_dimension.index.ToString(),
        dynamic_dimension.dim, mapping.second->ToString()));
  }
  return absl::StrJoin(pieces, "\n");
}

DynamicDimensionInference::DynamicDimensionInference(
    HloModule* module, OpSupportsDynamismHandler op_supports_dynamism_handler,
    CustomCallInferenceHandler custom_call_handler,
    ShapeCheckMode shape_check_mode, AssertionGenerator assertion_generator,
    const absl::flat_hash_set<absl::string_view>& execution_threads)
    : module_(module),
      op_supports_dynamism_handler_(std::move(op_supports_dynamism_handler)),
      custom_call_handler_(std::move(custom_call_handler)),
      shape_check_mode_(shape_check_mode),
      assertion_generator_(assertion_generator),
      execution_threads_(execution_threads) {}

Status DynamicDimensionInference::AnalyzeDynamicDimensions() {
  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<HloDataflowAnalysis> dataflow_analysis,
      HloDataflowAnalysis::Run(*module_, /*ssa_form=*/false,
                               /*bitcast_defines_value=*/true,
                               /*can_share_buffer=*/nullptr,
                               /*forwards_value=*/nullptr, execution_threads_));
  for (HloComputation* computation : module_->MakeComputationPostOrder()) {
    if (!HloInstruction::IsThreadIncluded(computation->execution_thread(),
                                          execution_threads_)) {
      continue;
    }
    TF_ASSIGN_OR_RETURN(
        bool changed,
        DynamicDimensionInferenceVisitor::Run(
            computation, *dataflow_analysis, {}, this, custom_call_handler_,
            shape_check_mode_, assertion_generator_));
    changed_ |= changed;
  }
  return OkStatus();
}

void DynamicDimensionInference::ReplaceAllDynamicDimensionUsesWith(
    HloInstruction* replace, HloInstruction* with) {
  CHECK(Shape::Equal().IgnoreLayout()(replace->shape(),
                                      ShapeUtil::MakeScalarShape(S32)));
  CHECK(Shape::Equal().IgnoreLayout()(with->shape(),
                                      ShapeUtil::MakeScalarShape(S32)));
  for (auto& kv : dynamic_mapping_) {
    if (kv.second == replace) {
      kv.second = with;
    }
  }
}

Status DynamicDimensionInference::ForwardDynamicSize(HloInstruction* inst,
                                                     HloInstruction* new_inst,
                                                     const ShapeIndex& index) {
  TF_RET_CHECK(ShapeUtil::Compatible(inst->shape(), new_inst->shape()));

  for (int64_t dim = 0; dim < inst->shape().rank(); ++dim) {
    DynamicDimension dynamic_dimension_new{new_inst, index, dim};
    DynamicDimension dynamic_dimension{inst, index, dim};
    auto iter = dynamic_mapping_.find(dynamic_dimension);
    if (iter != dynamic_mapping_.end()) {
      dynamic_mapping_.insert({dynamic_dimension_new, iter->second});
      auto iter = per_hlo_dynamic_dimensions_.try_emplace(new_inst);
      iter.first->second.emplace(dynamic_dimension_new);
    }
  }

  return OkStatus();
}

bool DynamicDimensionInference::HasDynamicDimension(
    HloInstruction* inst, ShapeIndexView index) const {
  bool has_dynamic_dim = false;
  ShapeUtil::ForEachSubshape(inst->shape(), [&](const Shape& subshape,
                                                const ShapeIndex& subindex) {
    if (subshape.IsTuple()) {
      return;
    }
    if (ShapeIndexView(subindex).subspan(0, index.size()) != index) {
      return;
    }
    for (int64_t i = 0; i < subshape.dimensions_size(); ++i) {
      HloInstruction* operand_dynamic_size = GetDynamicSize(inst, subindex, i);
      if (operand_dynamic_size != nullptr) {
        has_dynamic_dim = true;
      }
    }
  });
  return has_dynamic_dim;
}

Shape DynamicDimensionInference::GetDynamicShape(HloInstruction* inst) {
  Shape shape = inst->shape();
  ShapeUtil::ForEachMutableSubshape(
      &shape, [&](Shape* subshape, const ShapeIndex& index) {
        if (!subshape->IsArray()) {
          return;
        }
        for (int64_t dimension = 0; dimension < subshape->rank(); ++dimension) {
          if (GetDynamicSize(inst, index, dimension) != nullptr) {
            subshape->set_dynamic_dimension(dimension, true);
          }
        }
      });

  return shape;
}

HloInstruction* DynamicDimensionInference::GetDynamicSize(
    HloInstruction* inst, const ShapeIndex& index, int64_t dim) const {
  auto iter = dynamic_mapping_.find(DynamicDimension{inst, index, dim});
  if (iter != dynamic_mapping_.end()) {
    return iter->second;
  }
  return nullptr;
}

const HloInstruction* DynamicDimensionInference::GetDynamicSize(
    const HloInstruction* inst, const ShapeIndex& index, int64_t dim) const {
  return GetDynamicSize(const_cast<HloInstruction*>(inst), index, dim);
}

std::vector<HloInstruction*> DynamicDimensionInference::GetDynamicSizes(
    HloInstruction* inst, const ShapeIndex& index) const {
  CHECK(ShapeUtil::IndexIsValid(inst->shape(), index));
  const int64_t rank = ShapeUtil::GetSubshape(inst->shape(), index).rank();
  std::vector<HloInstruction*> result(rank, nullptr);
  for (int64_t i = 0; i < rank; ++i) {
    result[i] = GetDynamicSize(inst, index, i);
  }
  return result;
}

bool DynamicDimensionInference::CanInfer(HloInstruction* hlo) {
  // If the result shape is static, there are no dynamic dimensions to infer.
  // However, if there are called computations, we may need to run inference on
  // them.  Similarly, custom calls can do anything based on the user callbacks.
  if (hlo->shape().is_static() && hlo->called_computations().empty() &&
      hlo->opcode() != HloOpcode::kCustomCall) {
    return false;
  }
  // The dimensions of all operands must either be 1) not dynamic, or 2) have a
  // recorded dynamic size.  The only case where a dimension can be dynamic, but
  // where we have recorded a dynamic size is for SetDynamicSize instructions.
  bool ok = true;
  for (int64_t operand_index = 0; operand_index < hlo->operand_count();
       ++operand_index) {
    ShapeUtil::ForEachSubshape(
        hlo->operand(operand_index)->shape(),
        [&](const Shape& subshape, const ShapeIndex& shape_index) {
          if (!subshape.IsArray()) {
            return;
          }
          for (int64_t dimension = 0; dimension < subshape.rank();
               ++dimension) {
            bool shape_is_dynamic = subshape.is_dynamic_dimension(dimension);
            bool dynamic_size_recorded =
                GetDynamicSize(hlo->operand(operand_index), shape_index,
                               dimension) != nullptr;
            if (shape_is_dynamic && !dynamic_size_recorded) {
              VLOG(2) << "cannot infer " << hlo->ToShortString()
                      << " because operand " << operand_index << " ("
                      << hlo->operand(operand_index)->ToShortString() << ")"
                      << " subshape " << shape_index.ToString()
                      << " is missing dynamic size for dimension " << dimension;
              ok = false;
            }
            // Sanity check that we have cleared the dynamic dimension on the
            // shape if we have recorded the dynamic size.
            CHECK(hlo->operand(operand_index)->opcode() ==
                      HloOpcode::kSetDimensionSize ||
                  hlo->operand(operand_index)->opcode() ==
                      HloOpcode::kCustomCall ||
                  !shape_is_dynamic || !dynamic_size_recorded);
          }
        });
  }
  return ok;
}

}  // namespace xla

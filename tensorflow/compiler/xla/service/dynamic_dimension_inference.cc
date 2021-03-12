/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/dynamic_dimension_inference.h"

#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/match.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/service/dfs_hlo_visitor_with_default.h"
#include "tensorflow/compiler/xla/service/dynamic_window_utils.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/tuple_util.h"
#include "tensorflow/compiler/xla/service/while_util.h"
#include "tensorflow/compiler/xla/shape_tree.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/compiler/xla/window_util.h"
namespace xla {

namespace {
// Replace `narrow_comp` with a new computation with `wide_shape` as input.
StatusOr<HloComputation*> WidenComputation(HloComputation* narrow_comp,
                                           const Shape& wide_shape) {
  TF_RET_CHECK(wide_shape.IsTuple());
  const Shape& narrow_shape = narrow_comp->parameter_instruction(0)->shape();
  if (Shape::Equal()(wide_shape, narrow_shape)) {
    // No need to widen the computation.
    return narrow_comp;
  }
  HloComputation* wide_comp = [&]() {
    HloComputation::Builder builder(absl::StrCat("wide.", narrow_comp->name()));
    builder.AddInstruction(
        HloInstruction::CreateParameter(0, wide_shape, "wide_param"));
    return narrow_comp->parent()->AddEmbeddedComputation(builder.Build());
  }();

  HloInstruction* wide_parameter = wide_comp->parameter_instruction(0);
  HloInstruction* truncated_parameter = TupleUtil::ExtractPrefix(
      wide_parameter, narrow_shape.tuple_shapes_size());
  HloInstruction* call_narrow_comp = wide_comp->AddInstruction(
      HloInstruction::CreateCall(narrow_comp->root_instruction()->shape(),
                                 {truncated_parameter}, narrow_comp));
  wide_comp->set_root_instruction(call_narrow_comp,
                                  /*accept_different_shape=*/true);
  TF_RETURN_IF_ERROR(CallInliner::Inline(call_narrow_comp).status());
  return wide_comp;
}
}  // namespace

class DynamicDimensionInferenceVisitor : public DfsHloVisitorWithDefault {
 public:
  explicit DynamicDimensionInferenceVisitor(
      const DynamicParameterBinding& param_bindings,
      DynamicDimensionInference* parent,
      DynamicDimensionInference::CustomCallInferenceHandler custom_call_handler)
      : param_bindings_(param_bindings),
        parent_(parent),
        custom_call_handler_(std::move(custom_call_handler)) {}

  Status DefaultAction(HloInstruction* hlo) override;

  static Status Run(HloComputation* computation,
                    const DynamicParameterBinding& param_bindings,
                    DynamicDimensionInference* parent,
                    DynamicDimensionInference::CustomCallInferenceHandler
                        custom_call_handler = nullptr) {
    DynamicDimensionInferenceVisitor visitor(param_bindings, parent,
                                             std::move(custom_call_handler));
    return computation->Accept(&visitor);
  }

  Status HandleParameter(HloInstruction* hlo) override;

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

  Status HandleElementwiseBinary(HloInstruction* hlo) override;

  Status HandleClamp(HloInstruction* hlo) override;

  Status HandleConditional(HloInstruction* hlo) override;

  Status HandleWhile(HloInstruction* hlo) override;

  Status HandleSlice(HloInstruction* hlo) override;

  Status HandleDynamicSlice(HloInstruction* hlo) override;

  Status HandleDynamicUpdateSlice(HloInstruction* hlo) override;

  Status HandleGather(HloInstruction* hlo) override;

  Status HandleScatter(HloInstruction* hlo) override;

  Status HandleDomain(HloInstruction* hlo) override;

 private:
  using OperandDynamicDimensionFn = std::function<Status(
      HloInstruction* operand, ShapeIndex index, int64 dimension,
      int64 operand_index, HloInstruction* dynamic_size)>;

  using DynamicDimensionFn = std::function<Status(
      ShapeIndex index, int64 dimension, HloInstruction* dynamic_size)>;

  Status HandleDynamicConvolutionForward(HloInstruction* hlo,
                                         int64 operand_index, int64 dimension,
                                         HloInstruction* dynamic_size);

  Status HandleDynamicConvolutionKernelGrad(HloInstruction* hlo,
                                            int64 operand_index,
                                            int64 dimension);

  Status HandleDynamicConvolutionInputGrad(HloInstruction* hlo,
                                           int64 operand_index,
                                           int64 dimension);

  Status HandleDynamicWindowSamePadding(HloInstruction* hlo,
                                        HloInstruction* dynamic_size,
                                        int64 operand_index, int64 dimension);

  Status ForEachOperandDynamicDimension(HloInstruction* inst,
                                        const OperandDynamicDimensionFn&);
  Status ForEachDynamicDimensionInOperand(HloInstruction* inst,
                                          int64 operand_index,
                                          const OperandDynamicDimensionFn&);
  Status ForEachDynamicDimension(HloInstruction* inst,
                                 const DynamicDimensionFn& fn);

  // Pass through a dynamic dimension from the input to the output with the
  // same value and index in the shape. This is a helper function to handle
  // trivial instructions like elementwise operations.
  Status PassThroughDynamicDimension(HloInstruction*);

  // The dynamic parameter bindings of this computation.
  const DynamicParameterBinding& param_bindings_;

  // A pointer to DynamicDimensionInference, used to update the dynamic mapping.
  DynamicDimensionInference* parent_;

  // A handler for custom calls.
  DynamicDimensionInference::CustomCallInferenceHandler custom_call_handler_;
};

Status DynamicDimensionInferenceVisitor::DefaultAction(HloInstruction* hlo) {
  return ForEachOperandDynamicDimension(
      hlo, [&](HloInstruction* operand, ShapeIndex index, int64 dimension,
               int64 operand_index, HloInstruction* dynamic_size) {
        return UnimplementedStrCat(
            "Asked to propagate a dynamic dimension from hlo ", operand->name(),
            "@", index.ToString(), "@", dimension, " to hlo ", hlo->ToString(),
            ", which is not implemented.");
      });
}

Status DynamicDimensionInferenceVisitor::HandleGetTupleElement(
    HloInstruction* hlo) {
  return ForEachOperandDynamicDimension(
      hlo, [&](HloInstruction* operand, ShapeIndex index, int64 dimension,
               int64 operand_index, HloInstruction* dynamic_size) {
        if (hlo->tuple_index() == index[0]) {
          ShapeIndex new_index =
              ShapeIndexView(index).ConsumeFront().ToShapeIndex();
          parent_->SetDynamicSize(hlo, new_index, dimension, dynamic_size);
        }
        return Status::OK();
      });
}

Status DynamicDimensionInferenceVisitor::HandleTuple(HloInstruction* hlo) {
  return ForEachOperandDynamicDimension(
      hlo, [&](HloInstruction*, ShapeIndex index, int64 dimension,
               int64 operand_index, HloInstruction* dynamic_size) {
        index.push_front(operand_index);
        parent_->SetDynamicSize(hlo, index, dimension, dynamic_size);
        return Status::OK();
      });
}

Status DynamicDimensionInferenceVisitor::HandleBroadcast(HloInstruction* hlo) {
  return ForEachOperandDynamicDimension(
      hlo, [&](HloInstruction* operand, ShapeIndex index, int64 dimension,
               int64 operand_index, HloInstruction* dynamic_size) {
        int64 broadcast_dim = hlo->dimensions(dimension);
        parent_->SetDynamicSize(hlo, {}, broadcast_dim, dynamic_size);
        return Status::OK();
      });
}

Status DynamicDimensionInferenceVisitor::HandleCustomCall(HloInstruction* hlo) {
  if (hlo->custom_call_target() == "PadToStatic") {
    for (int64 i = 0; i < hlo->operand(0)->shape().rank(); ++i) {
      if (hlo->operand(0)->shape().is_dynamic_dimension(i)) {
        HloInstruction* dynamic_size =
            hlo->parent()->AddInstruction(HloInstruction::CreateGetTupleElement(
                ShapeUtil::MakeScalarShape(S32), hlo, i + 1));
        // PadToStatic converts a dynamic dimension to static dimension. It then
        // returns the padded data output and the dynamic sizes of input
        // dimensions.
        ShapeIndex data_output = {0};
        parent_->SetDynamicSize(hlo, data_output, i, dynamic_size);
      }
    }
    return Status::OK();
  }
  if (custom_call_handler_) {
    return custom_call_handler_(hlo, parent_);
  }

  if (hlo->custom_call_target() == "DynamicConvolutionForward") {
    // If input feature is dynamic and kernel feature is static, we can infer
    // that input feature is also static.
    // E.g.,:
    // lhs = [B, X, Y, ?]
    // rhs = [X, Y, I, O]
    // dim_labels = b01f_01io
    // We can infer that the dynamic dimension in rhs is static I.
    const ConvolutionDimensionNumbers& dnums =
        hlo->convolution_dimension_numbers();
    HloInstruction* input_feature = parent_->GetDynamicSize(
        hlo->mutable_operand(0), {}, dnums.input_feature_dimension());
    HloInstruction* kernel_feature = parent_->GetDynamicSize(
        hlo->mutable_operand(1), {}, dnums.kernel_input_feature_dimension());

    if (input_feature != nullptr && kernel_feature == nullptr) {
      if (hlo->mutable_operand(0)->shape().dimensions(
              dnums.input_feature_dimension()) ==
          hlo->mutable_operand(1)->shape().dimensions(
              dnums.kernel_input_feature_dimension()))
        parent_->SetDynamicSize(hlo->mutable_operand(0), {},
                                dnums.input_feature_dimension(), nullptr);
    }
  }
  return ForEachOperandDynamicDimension(
      hlo, [&](HloInstruction* operand, ShapeIndex index, int64 dimension,
               int64 operand_index, HloInstruction* dynamic_size) {
        // Resize custom call should propagate dynamic batch (0) and channel (3)
        // dimensions.
        if (hlo->custom_call_target() == "SliceToDynamic" ||
            hlo->custom_call_target() == "Sharding" ||
            (absl::StartsWith(hlo->custom_call_target(), "Resize") &&
             (dimension == 0 || dimension == 3))) {
          parent_->SetDynamicSize(hlo, {}, dimension, dynamic_size);
          return Status::OK();
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

        if (hlo->custom_call_target() == "DynamicSelectAndScatterSamePadding") {
          if (operand_index == 1) {
            // Operand 0 (input) determines dynamic output size. We ignore the
            // dynamic size in the operand 1 (output gradient).
            return Status::OK();
          }
          parent_->SetDynamicSize(hlo, {}, dimension, dynamic_size);
          return Status::OK();
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
          return HandleDynamicConvolutionForward(hlo, operand_index, dimension,
                                                 dynamic_size);
        }
        return Unimplemented(
            "CustomCall \"%s\" is not supported to have a dynamic dimension",
            hlo->custom_call_target());
      });
}

Status DynamicDimensionInferenceVisitor::HandleSort(HloInstruction* hlo) {
  return ForEachOperandDynamicDimension(
      hlo,
      [&](HloInstruction* operand, ShapeIndex index, int64 dynamic_dimension,
          int64 operand_index, HloInstruction* dynamic_size) {
        HloSortInstruction* sort = Cast<HloSortInstruction>(hlo);
        if (sort->values_count() == 0) {
          parent_->SetDynamicSize(hlo, {}, dynamic_dimension, dynamic_size);
        } else {
          parent_->SetDynamicSize(hlo, {operand_index}, dynamic_dimension,
                                  dynamic_size);
        }

        return Status::OK();
      });
}

Status DynamicDimensionInferenceVisitor::HandlePad(HloInstruction* hlo) {
  return ForEachOperandDynamicDimension(
      hlo, [&](HloInstruction* operand, ShapeIndex index, int64 dimension,
               int64 operand_index, HloInstruction* dynamic_size) {
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
          HloInstruction* one = hlo->parent()->AddInstruction(
              HloInstruction::CreateConstant(LiteralUtil::CreateR0<int32>(1)));
          HloInstruction* zero = hlo->parent()->AddInstruction(
              HloInstruction::CreateConstant(LiteralUtil::CreateR0<int32>(0)));
          HloInstruction* interior_padding = hlo->parent()->AddInstruction(
              HloInstruction::CreateConstant(LiteralUtil::CreateR0<int32>(
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
            HloInstruction::CreateConstant(LiteralUtil::CreateR0<int32>(
                padding_config.edge_padding_low() +
                padding_config.edge_padding_high())));
        dynamic_size_adjusted =
            hlo->parent()->AddInstruction(HloInstruction::CreateBinary(
                dynamic_size_adjusted->shape(), HloOpcode::kAdd,
                dynamic_size_adjusted, adjustment));
        parent_->SetDynamicSize(hlo, {}, dimension, dynamic_size_adjusted);
        return Status::OK();
      });
}

Status DynamicDimensionInferenceVisitor::HandleReduce(HloInstruction* hlo) {
  return ForEachOperandDynamicDimension(
      hlo, [&](HloInstruction* operand, ShapeIndex index, int64 dimension,
               int64 operand_index, HloInstruction* dynamic_size) {
        HloInstruction* reduce = hlo;
        int64 operand_count = reduce->operand_count();
        bool is_variadic_reduce = operand_count > 2;
        CHECK_EQ(operand_count % 2, 0);
        if (operand_index >= operand_count / 2) {
          // Init values doesn't have dynamic size.
          return Status::OK();
        }
        if ((absl::c_count(reduce->dimensions(), dimension) != 0)) {
          // Dimension is to be reduced, stop tracing.
          return Status::OK();
        }

        // Find out the new dynamic dimension after reduce.
        int64 dimensions_not_reduced_count = 0;
        for (int i = 0; i < operand->shape().rank(); ++i) {
          if (dimension == i) {
            ShapeIndex result_index = {};

            if (is_variadic_reduce) {
              // The dimensions of all data operands of a variadic reduce have
              // to be the same.  This means that if one operand of variadic
              // reduce has a dynamic dimension, we set all outputs to use the
              // same dynamic size in corresponding dimensions.
              for (int64 i = 0; i < operand_count / 2; ++i) {
                parent_->SetDynamicSize(
                    reduce, {i}, dimensions_not_reduced_count, dynamic_size);
              }
            } else {
              parent_->SetDynamicSize(reduce, {}, dimensions_not_reduced_count,
                                      dynamic_size);
            }

            return Status::OK();
          }
          if (absl::c_count(reduce->dimensions(), i) == 0) {
            dimensions_not_reduced_count++;
          }
        }

        return Status::OK();
      });
}

Status DynamicDimensionInferenceVisitor::HandleDot(HloInstruction* hlo) {
  return ForEachOperandDynamicDimension(
      hlo, [&](HloInstruction* operand, ShapeIndex operand_shape_index,
               int64 operand_dimension, int64 operand_index,
               HloInstruction* dynamic_size) {
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
        absl::flat_hash_map<int64, int64> result_dim_mapping;
        int64 current_result_dims = 0;

        bool lhs = operand_index == 0;

        // The first loop keep tracks of batch dimension. RHS and LHS could have
        // different batch dimension numbers.
        if (lhs) {
          for (int64 i : dimension_numbers.lhs_batch_dimensions()) {
            result_dim_mapping[i] = current_result_dims++;
          }
        } else {
          for (int64 i : dimension_numbers.rhs_batch_dimensions()) {
            result_dim_mapping[i] = current_result_dims++;
          }
        }

        // Handle dimensions in the lhs.
        for (int64 i = 0; i < dot->operand(0)->shape().rank(); i++) {
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
        for (int64 i = 0; i < dot->operand(1)->shape().rank(); i++) {
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
          parent_->SetDynamicSize(dot, {}, iter->second, dynamic_size);
        }

        return Status::OK();
      });
}

Status DynamicDimensionInferenceVisitor::HandleTranspose(HloInstruction* hlo) {
  return ForEachOperandDynamicDimension(
      hlo,
      [&](HloInstruction* operand, ShapeIndex index, int64 dimension,
          int64 operand_index, HloInstruction* dynamic_size) -> Status {
        int64 permuted_dim = -1;
        for (int64 i = 0; i < hlo->dimensions().size(); ++i) {
          if (hlo->dimensions()[i] == dimension) {
            TF_RET_CHECK(permuted_dim == -1);
            permuted_dim = i;
          }
        }
        parent_->SetDynamicSize(hlo, {}, permuted_dim, dynamic_size);
        return Status::OK();
      });
}

Status DynamicDimensionInferenceVisitor::HandleConvolution(
    HloInstruction* hlo) {
  return ForEachOperandDynamicDimension(
      hlo, [&](HloInstruction* operand, ShapeIndex index, int64 dimension,
               int64 operand_index, HloInstruction* dynamic_size) {
        HloInstruction* conv = hlo;
        const ConvolutionDimensionNumbers& dimension_numbers =
            conv->convolution_dimension_numbers();
        if (operand_index == 0) {
          if (dimension == dimension_numbers.input_batch_dimension()) {
            parent_->SetDynamicSize(conv, {},
                                    dimension_numbers.output_batch_dimension(),
                                    dynamic_size);
            return Status::OK();
          }

          if (dimension == dimension_numbers.input_feature_dimension()) {
            return Status::OK();
          }
        } else {
          if (dimension == dimension_numbers.kernel_input_feature_dimension()) {
            return Status::OK();
          }
        }

        return Unimplemented("Dynamic Spatial Convolution is not supported: %s",
                             conv->ToString());
      });
}

Status DynamicDimensionInferenceVisitor::HandleConcatenate(
    HloInstruction* hlo) {
  // First handle concatenate dimensions. We do this by iterating through all
  // operands while tracking both dynamic and static dimensions.

  // static_size is used to keep track of the concated size of static
  // dimensions.
  int64 static_size = 0;
  std::vector<HloInstruction*> dynamic_concat_dims;
  for (int64 i = 0; i < hlo->operand_count(); ++i) {
    HloInstruction* dynamic_size = parent_->GetDynamicSize(
        hlo->mutable_operand(i), {}, hlo->concatenate_dimension());
    if (dynamic_size == nullptr) {
      // This is a static dimension.
      static_size +=
          hlo->operand(i)->shape().dimensions(hlo->concatenate_dimension());
    } else {
      dynamic_concat_dims.push_back(dynamic_size);
    }
  }
  // If concat dimension is dynamic, calculate its size by summing up static
  // dims and dynamic dims together.
  if (!dynamic_concat_dims.empty()) {
    HloInstruction* dim_size_total =
        hlo->parent()->AddInstruction(HloInstruction::CreateConstant(
            LiteralUtil::CreateR0<int32>(static_size)));
    for (HloInstruction* dynamic_dim : dynamic_concat_dims) {
      dim_size_total = hlo->parent()->AddInstruction(
          HloInstruction::CreateBinary(dim_size_total->shape(), HloOpcode::kAdd,
                                       dim_size_total, dynamic_dim));
    }
    parent_->SetDynamicSize(hlo, {}, hlo->concatenate_dimension(),
                            dim_size_total);
  }

  // Simply pass through non-concat dynamic dimensions.
  return ForEachOperandDynamicDimension(
      hlo, [&](HloInstruction* operand, ShapeIndex index, int64 dimension,
               int64 operand_index, HloInstruction* dynamic_size) {
        int64 concatenate_dimension = hlo->concatenate_dimension();
        if (concatenate_dimension == dimension) {
          return Status::OK();
        }
        parent_->SetDynamicSize(hlo, index, dimension, dynamic_size);
        return Status::OK();
      });
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
  int64 dim = gds->dimension();
  HloInstruction* operand = gds->mutable_operand(0);
  HloInstruction* dynamic_size = parent_->GetDynamicSize(operand, {}, dim);
  HloComputation* computation = gds->parent();
  if (dynamic_size != nullptr) {
    TF_RETURN_IF_ERROR(gds->ReplaceAllUsesWith(dynamic_size));
    // The dependency between an instruction and its dynamic dimensions is not
    // modeled in the IR. As instr is being replaced by dynamic_size, also tell
    // dynamic dimension inference that the instruction is being replaced.
    parent_->ReplaceAllDynamicDimensionUsesWith(gds, dynamic_size);
  } else {
    TF_RET_CHECK(dim < gds->operand(0)->shape().rank());
    int32 size = gds->operand(0)->shape().dimensions(dim);
    HloInstruction* new_instr = computation->AddInstruction(
        HloInstruction::CreateConstant(LiteralUtil::CreateR0<int32>(size)));
    TF_RETURN_IF_ERROR(gds->ReplaceAllUsesWith(new_instr));
    parent_->ReplaceAllDynamicDimensionUsesWith(gds, new_instr);
  }
  return Status::OK();
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
    if (size->literal().Get<int32>({}) ==
        hlo->shape().dimensions(hlo->dimension())) {
      dimension_is_static = true;
    }
  }

  if (!dimension_is_static) {
    // Propagate dynamic dimension indicated by this set dimension size
    // instruction.
    parent_->SetDynamicSize(hlo, {}, hlo->dimension(), hlo->mutable_operand(1));
  }

  // Also Propagate dynamic dimension already set by operands.
  TF_RETURN_IF_ERROR(ForEachOperandDynamicDimension(
      hlo, [&](HloInstruction* operand, ShapeIndex index, int64 dimension,
               int64 operand_index, HloInstruction* dynamic_size) {
        if (dimension != hlo->dimension()) {
          parent_->SetDynamicSize(hlo, index, dimension, dynamic_size);
        }
        return Status::OK();
      }));

  return Status::OK();
}

Status DynamicDimensionInferenceVisitor::HandleDynamicConvolutionForward(
    HloInstruction* hlo, int64 operand_index, int64 dimension,
    HloInstruction* dynamic_size) {
  TF_RET_CHECK(operand_index == 0);
  const ConvolutionDimensionNumbers& dimension_numbers =
      hlo->convolution_dimension_numbers();

  if (dimension == dimension_numbers.input_batch_dimension()) {
    // Batch dimension is propagated without any changes.
    parent_->SetDynamicSize(hlo, {}, dimension_numbers.output_batch_dimension(),
                            dynamic_size);
    return Status::OK();
  }

  for (int64 spatial_dim_index = 0;
       spatial_dim_index < dimension_numbers.input_spatial_dimensions_size();
       ++spatial_dim_index) {
    int64 input_spatial_dim =
        dimension_numbers.input_spatial_dimensions(spatial_dim_index);
    int64 output_spatial_dim =
        dimension_numbers.output_spatial_dimensions(spatial_dim_index);
    if (dimension == input_spatial_dim) {
      // This is a dynamic spatial dimension. Calculate the output size.
      WindowDimension window_dim = hlo->window().dimensions(spatial_dim_index);
      DynamicWindowDims dynamic_window_dims = GetWindowedOutputSize(
          dynamic_size, window_dim.size(), window_dim.window_dilation(),
          window_dim.stride(), hlo->padding_type());
      TF_RET_CHECK(window_dim.base_dilation() == 1);
      parent_->SetDynamicSize(hlo, {}, output_spatial_dim,
                              dynamic_window_dims.output_size);
      return Status::OK();
    }
  }
  // Input Feature dim disappears after convolution.
  return Status::OK();
}

Status DynamicDimensionInferenceVisitor::HandleDynamicWindowSamePadding(
    HloInstruction* hlo, HloInstruction* dynamic_size, int64 operand_index,
    int64 dimension) {
  const Window& window = hlo->window();
  const WindowDimension& window_dim = window.dimensions(dimension);
  if (!window_util::IsTrivialWindowDimension(window_dim)) {
    DynamicWindowDims dynamic_window_dims = GetWindowedOutputSize(
        dynamic_size, window_dim.size(), window_dim.window_dilation(),
        window_dim.stride(), PaddingType::PADDING_SAME);
    parent_->SetDynamicSize(hlo, {}, dimension,
                            dynamic_window_dims.output_size);
    return Status::OK();
  }

  parent_->SetDynamicSize(hlo, {}, dimension, dynamic_size);

  return Status::OK();
}

Status DynamicDimensionInferenceVisitor::HandleDynamicConvolutionInputGrad(
    HloInstruction* hlo, int64 operand_index, int64 dimension) {
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
  parent_->SetDynamicSize(hlo, {}, dimension, reshape);
  return Status::OK();
}

Status DynamicDimensionInferenceVisitor::HandleDynamicConvolutionKernelGrad(
    HloInstruction* hlo, int64 operand_index, int64 dimension) {
  // Dynamic convolution kernel grad produces static shape outputs.
  return Status::OK();
}

Status DynamicDimensionInferenceVisitor::PassThroughDynamicDimension(
    HloInstruction* hlo) {
  return ForEachOperandDynamicDimension(
      hlo, [&](HloInstruction* operand, ShapeIndex index, int64 dimension,
               int64 operand_index, HloInstruction* dynamic_size) {
        parent_->SetDynamicSize(hlo, index, dimension, dynamic_size);
        return Status::OK();
      });
}

Status DynamicDimensionInferenceVisitor::HandleDomain(HloInstruction* hlo) {
  return PassThroughDynamicDimension(hlo);
}

Status DynamicDimensionInferenceVisitor::HandleElementwiseUnary(
    HloInstruction* hlo) {
  return PassThroughDynamicDimension(hlo);
}

Status DynamicDimensionInferenceVisitor::HandleSelect(HloInstruction* hlo) {
  return PassThroughDynamicDimension(hlo);
}

Status DynamicDimensionInferenceVisitor::HandleElementwiseBinary(
    HloInstruction* hlo) {
  HloComputation* comp = hlo->parent();
  return ForEachOperandDynamicDimension(
      hlo, [&](HloInstruction* operand, ShapeIndex index, int64 dimension,
               int64 operand_index, HloInstruction* dynamic_size) {
        HloInstruction* existing_size =
            parent_->GetDynamicSize(hlo, index, dimension);
        if (existing_size == nullptr || existing_size == dynamic_size) {
          parent_->SetDynamicSize(hlo, index, dimension, dynamic_size);
        } else {
          HloInstruction* max =
              comp->AddInstruction(HloInstruction::CreateBinary(
                  ShapeUtil::MakeScalarShape(S32), HloOpcode::kMaximum,
                  dynamic_size, existing_size));
          parent_->SetDynamicSize(hlo, index, dimension, max);
        }
        return Status::OK();
      });
}

Status DynamicDimensionInferenceVisitor::HandleClamp(HloInstruction* hlo) {
  return PassThroughDynamicDimension(hlo);
}

Status DynamicDimensionInferenceVisitor::HandleDynamicReshape(
    HloInstruction* hlo) {
  HloDynamicReshapeInstruction* dynamic_reshape =
      Cast<HloDynamicReshapeInstruction>(hlo);
  for (int64 i = 0; i < hlo->shape().rank(); ++i) {
    if (hlo->shape().is_dynamic_dimension(i)) {
      parent_->SetDynamicSize(hlo, {}, i, dynamic_reshape->dim_sizes(i));
    }
  }
  return Status::OK();
}

Status DynamicDimensionInferenceVisitor::HandleReshape(HloInstruction* hlo) {
  return ForEachOperandDynamicDimension(
      hlo,
      [&](HloInstruction* operand, ShapeIndex index,
          int64 input_dynamic_dimension, int64 operand_index,
          HloInstruction* operand_dynamic_size) -> Status {
        HloInstruction* reshape = hlo;
        if (reshape->shape().rank() == 0) {
          VLOG(0) << "Reshaping a dynamic dimension into a scalar, which has "
                     "undefined behavior when input size is 0. The offending "
                     "instruction is: "
                  << reshape->ToString();
          return Status::OK();
        }
        auto common_factors = CommonFactors(operand->shape().dimensions(),
                                            reshape->shape().dimensions());
        int64 input_dim_start = -1;
        int64 input_dim_end = -1;
        int64 output_dim_start = -1;
        int64 output_dim_end = -1;
        // Find common_factors that the input belongs to.
        for (int64 i = 0; i < common_factors.size() - 1; ++i) {
          auto start = common_factors[i];
          auto end = common_factors[i + 1];
          if (input_dynamic_dimension >= start.first &&
              input_dynamic_dimension < end.first) {
            // Found the common_factor group that the input_dim belongs to.
            input_dim_start = start.first;
            input_dim_end = end.first;
            output_dim_start = start.second;
            output_dim_end = end.second;
          }
        }

        VLOG(2) << "Input dim start: " << input_dim_start
                << " Input dim end: " << input_dim_end
                << " output dim start: " << output_dim_start
                << " output dim end: " << output_dim_end;

        if ((input_dim_end - input_dim_start) > 1 &&
            (output_dim_end - output_dim_start) > 1) {
          // We don't support the case when a dynamic dimension is both combined
          // with and splitted into other dimensions:
          //
          //  [x, yz]
          //     | Reshape
          //  [xy, z]
          //
          // TODO(yunxing): This can be supported by canonicalizing
          // the offending reshape into two reshapes:
          //
          //  [x,yz]
          //     | Reshape
          //  [x, y, z]
          //     | Reshape
          //  [xy, z]
          //
          return Unimplemented(
              "Dynamic input dimension to reshape that is both splitted and "
              "combined is not supported %s",
              hlo->ToString());
        }

        for (auto common_factor : common_factors) {
          // Expand common factor to include degenerated output dimensions.
          if (common_factor.first == input_dim_start) {
            output_dim_start = std::min(output_dim_start, common_factor.second);
          }
          if (common_factor.first == input_dim_end) {
            output_dim_end = std::max(output_dim_end, common_factor.second);
          }
        }

        int64 output_dynamic_dimension = -1;

        if (operand->shape().dimensions(input_dynamic_dimension) == 1) {
          // If dynamic dimension is 1, it can only be most-major or
          // most-minor.
          if (input_dynamic_dimension == 0) {
            output_dynamic_dimension = 0;
          }
          if (input_dynamic_dimension == operand->shape().rank() - 1) {
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
            for (int64 i = output_dim_start; i < output_dim_end; ++i) {
              if (reshape->shape().is_dynamic_dimension(i)) {
                output_dynamic_dimension = i;
              }
            }
          }

          if (output_dynamic_dimension == -1) {
            std::vector<int64> output_non_degenerated;
            for (int64 i = output_dim_start; i < output_dim_end; ++i) {
              if (reshape->shape().dimensions(i) != 1) {
                output_non_degenerated.push_back(i);
              }
            }
            if (output_non_degenerated.size() == 1) {
              output_dynamic_dimension = output_non_degenerated[0];
            }
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
        const int64 input_dim_size =
            operand->shape().dimensions(input_dynamic_dimension);
        const int64 output_dim_size =
            reshape->shape().dimensions(output_dynamic_dimension);
        VLOG(2) << "input_dim_size: " << input_dim_size
                << " output_dim_size: " << output_dim_size;

        if (input_dim_size == output_dim_size) {
          // Simply forward dynamic dimension.
          parent_->SetDynamicSize(reshape, {}, output_dynamic_dimension,
                                  operand_dynamic_size);
        }

        if (input_dim_size > output_dim_size) {
          TF_RET_CHECK(input_dim_size % output_dim_size == 0)
              << reshape->ToString();
          const int64 divisor = input_dim_size / output_dim_size;
          HloInstruction* divisor_hlo =
              hlo->parent()->AddInstruction(HloInstruction::CreateConstant(
                  LiteralUtil::CreateR0<int32>(divisor)));

          HloInstruction* new_dynamic_size =
              hlo->parent()->AddInstruction(HloInstruction::CreateBinary(
                  operand_dynamic_size->shape(), HloOpcode::kDivide,
                  operand_dynamic_size, divisor_hlo));

          parent_->SetDynamicSize(reshape, {}, output_dynamic_dimension,
                                  new_dynamic_size);
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
              parent_->GetDynamicSize(reshape, {}, output_dynamic_dimension);
          if (output_dynamic_size == nullptr) {
            output_dynamic_size =
                hlo->parent()->AddInstruction(HloInstruction::CreateConstant(
                    LiteralUtil::CreateR0<int32>(output_dim_size)));
          }
          HloInstruction* divisor_hlo = hlo->parent()->AddInstruction(
              HloInstruction::CreateConstant(LiteralUtil::CreateR0<int32>(
                  operand->shape().dimensions(input_dynamic_dimension))));

          HloInstruction* new_dynamic_size =
              hlo->parent()->AddInstruction(HloInstruction::CreateBinary(
                  output_dynamic_size->shape(), HloOpcode::kDivide,
                  output_dynamic_size, divisor_hlo));

          new_dynamic_size =
              hlo->parent()->AddInstruction(HloInstruction::CreateBinary(
                  output_dynamic_size->shape(), HloOpcode::kMultiply,
                  new_dynamic_size, operand_dynamic_size));
          parent_->SetDynamicSize(reshape, {}, output_dynamic_dimension,
                                  new_dynamic_size);
        }

        return Status::OK();
      });
}

Status DynamicDimensionInferenceVisitor::HandleReduceWindow(
    HloInstruction* hlo) {
  return ForEachOperandDynamicDimension(
      hlo, [&](HloInstruction* operand, ShapeIndex index, int64 dimension,
               int64 operand_index, HloInstruction* dynamic_size) {
        HloInstruction* reduce_window = hlo;
        const WindowDimension& window_dim =
            reduce_window->window().dimensions(dimension);

        if (!window_util::IsTrivialWindowDimension(window_dim)) {
          DynamicWindowDims dynamic_window_dims = GetWindowedOutputSize(
              dynamic_size, window_dim.size(), window_dim.window_dilation(),
              window_dim.stride(), PaddingType::PADDING_VALID);
          parent_->SetDynamicSize(hlo, {}, dimension,
                                  dynamic_window_dims.output_size);
          return Status::OK();
        }

        parent_->SetDynamicSize(reduce_window, {}, dimension, dynamic_size);

        return Status::OK();
      });
}

Status DynamicDimensionInferenceVisitor::HandleSelectAndScatter(
    HloInstruction* hlo) {
  return ForEachOperandDynamicDimension(
      hlo, [&](HloInstruction* operand, ShapeIndex index, int64 dimension,
               int64 operand_index, HloInstruction* dynamic_size) {
        if (operand_index == 1) {
          // Operand 0 (input) determines dynamic output size. We ignore the
          // dynamic size in the operand 1 (output gradient).
          return Status::OK();
        }
        parent_->SetDynamicSize(hlo, {}, dimension, dynamic_size);

        return Status::OK();
      });
}

Status DynamicDimensionInferenceVisitor::HandleSlice(HloInstruction* hlo) {
  return ForEachOperandDynamicDimension(
      hlo, [&](HloInstruction* operand, ShapeIndex /*index*/, int64 dimension,
               int64 /*operand_index*/, HloInstruction* dynamic_size) {
        if (hlo->slice_starts(dimension) != 0 ||
            hlo->slice_strides(dimension) != 1 ||
            hlo->slice_limits(dimension) !=
                operand->shape().dimensions(dimension)) {
          // Slicing a partial element out eliminates the dynamic dimension.
          return Status::OK();
        }

        parent_->SetDynamicSize(hlo, {}, dimension, dynamic_size);

        return Status::OK();
      });
}

Status DynamicDimensionInferenceVisitor::HandleDynamicSlice(
    HloInstruction* hlo) {
  return ForEachOperandDynamicDimension(
      hlo, [&](HloInstruction*, ShapeIndex /*index*/, int64 dimension,
               int64 /*operand_index*/, HloInstruction* dynamic_size) {
        if (hlo->shape().dimensions(dimension) !=
            hlo->operand(0)->shape().dimensions(dimension)) {
          // Slicing a single element out kills the dynamic dimension.
          if (hlo->shape().dimensions(dimension) == 1) {
            return Status::OK();
          }
          return Unimplemented(
              "Dynamic dimension propagation on DynamicSlice where a partial "
              "dimension is selected %s",
              hlo->ToString());
        }

        parent_->SetDynamicSize(hlo, {}, dimension, dynamic_size);

        return Status::OK();
      });
}

Status DynamicDimensionInferenceVisitor::HandleDynamicUpdateSlice(
    HloInstruction* hlo) {
  return ForEachOperandDynamicDimension(
      hlo,
      [&](HloInstruction* /*operand*/, ShapeIndex /*index*/, int64 dimension,
          int64 operand_index, HloInstruction* dynamic_size) {
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
          return Status::OK();
        }

        parent_->SetDynamicSize(hlo, {}, dimension, dynamic_size);

        return Status::OK();
      });
}

Status DynamicDimensionInferenceVisitor::HandleReverse(HloInstruction* hlo) {
  return ForEachOperandDynamicDimension(
      hlo,
      [&](HloInstruction* /*operand*/, ShapeIndex /*index*/, int64 dimension,
          int64 /*operand_index*/, HloInstruction* dynamic_size) {
        if (absl::c_linear_search(hlo->dimensions(), dimension)) {
          return Unimplemented(
              "Dynamic dimension propagation on reversed dimension is not "
              "supported %s",
              hlo->ToString());
        }
        parent_->SetDynamicSize(hlo, {}, dimension, dynamic_size);

        return Status::OK();
      });
}

Status DynamicDimensionInferenceVisitor::HandleGather(HloInstruction* hlo) {
  return ForEachOperandDynamicDimension(
      hlo, [&](HloInstruction* operand, ShapeIndex /*index*/,
               int64 input_dynamic_dimension, int64 operand_index,
               HloInstruction* dynamic_size) {
        const GatherDimensionNumbers& gather_dims =
            hlo->gather_dimension_numbers();
        if (operand_index != 1) {
          if (hlo->gather_slice_sizes()[input_dynamic_dimension] == 1) {
            // Gathering a size 1 dimension out of a dynamic dimension removes
            // the dynamicity.
            return Status::OK();
          }
          if (hlo->gather_slice_sizes()[input_dynamic_dimension] ==
              operand->shape().dimensions(input_dynamic_dimension)) {
            // Gathering a full-sized dimension out of a dynamic dimension
            // propagates the dynamicity to output.
            int64 output_dimension = input_dynamic_dimension;
            for (int64 collapsed_dim : gather_dims.collapsed_slice_dims()) {
              if (collapsed_dim < input_dynamic_dimension) {
                // This output dimension is collapsed.
                output_dimension--;
              }
            }
            parent_->SetDynamicSize(hlo, {}, output_dimension, dynamic_size);
            return Status::OK();
          }
          return Unimplemented(
              "Detects a dynamic dimension on the data input of gather, which "
              "is not supported: %s, %lld",
              hlo->ToString(), input_dynamic_dimension);
        }
        // A mapping from output to input batch dim number. -1 means not a batch
        // dimension.
        int64 indices_rank = hlo->operand(1)->shape().rank();
        int64 output_rank = hlo->shape().rank();

        // indices_dim is an iterator over indices dimensions.
        int64 indices_dim = 0;
        // Find the corresponding batch dimension in the output.
        for (int64 output_dim = 0; output_dim < output_rank; ++output_dim) {
          if (!absl::c_linear_search(gather_dims.offset_dims(), output_dim)) {
            // Skips index vector dimension.
            if (indices_dim == gather_dims.index_vector_dim()) {
              indices_dim++;
            }
            if (indices_dim++ == input_dynamic_dimension) {
              parent_->SetDynamicSize(hlo, {}, output_dim, dynamic_size);
              return Status::OK();
            }
          }
        }
        CHECK(indices_dim == indices_rank);

        return Unimplemented(
            "Detects a non-batch dynamic dimension of gather, "
            "which is not supported: %s",
            hlo->ToString());
      });
}

Status DynamicDimensionInferenceVisitor::HandleConditional(
    HloInstruction* hlo) {
  // Conditionals are handled by producing additional inputs and outputs of
  // the conditional instruction.
  std::vector<HloComputation*> new_branch_computations;
  std::vector<HloInstruction*> new_operands;
  // If the output of the conditional contains dynamic dimension. We send
  // dynamic dimension size out by adding additional root element. A mapping
  // from the root instruction's dynamic dimension index (represented by a shape
  // index as output index and a int64 dimension number) to output index
  // (represented by an int64) is tracked for the conditional intsruction (all
  // branches should have the same mapping).
  ShapeTree<absl::flat_hash_map<int64, int64>> dynamic_output_mapping(
      hlo->shape());

  bool need_rewrite = false;

  for (int64 branch_index = 0; branch_index < hlo->branch_count();
       ++branch_index) {
    std::vector<HloInstruction*> operands_to_add;

    absl::flat_hash_map<HloInstruction*, int64>
        dynamic_size_to_operand_id_index_map;
    // Only look at branch_index + 1, the correct operand index for a
    // given branch.
    const int64 operand_index = branch_index + 1;

    int64 operand_count =
        hlo->operand(operand_index)->shape().tuple_shapes_size();
    // Prepare to pass dynamic dimension into the new computation and add
    // dynamic dimension sizes as parameters to the new tuple.
    TF_RETURN_IF_ERROR(ForEachDynamicDimensionInOperand(
        hlo, operand_index,
        [&](HloInstruction*, ShapeIndex, int64, int64,
            HloInstruction* dynamic_size) -> Status {
          TF_RET_CHECK(hlo->operand(operand_index)->shape().IsTuple())
              << "Only tuple typed inputs can have dynamic dimension. Please "
                 "file a bug against XLA team.";
          const HloInstruction* tuple_operand = hlo->operand(operand_index);
          for (int64 i = 0; i < tuple_operand->operand_count(); ++i) {
            // If the dynamic size is already an operand to the computation,
            // skip adding it to the computation input again.
            if (dynamic_size == tuple_operand->operand(i)) {
              dynamic_size_to_operand_id_index_map[dynamic_size] = i;
              return Status::OK();
            }
          }
          auto iter = dynamic_size_to_operand_id_index_map.find(dynamic_size);
          if (iter == dynamic_size_to_operand_id_index_map.end()) {
            operands_to_add.push_back(dynamic_size);
            dynamic_size_to_operand_id_index_map[dynamic_size] =
                operand_count++;
          }
          return Status::OK();
        }));

    HloInstruction* original_input = hlo->mutable_operand(operand_index);
    HloComputation* branch_computation = hlo->branch_computation(branch_index);

    HloComputation* new_computation = branch_computation;
    HloInstruction* new_operand = hlo->mutable_operand(operand_index);
    if (!operands_to_add.empty()) {
      TF_RET_CHECK(original_input->shape().IsTuple());
      need_rewrite = true;
      new_operand = TupleUtil::AppendSuffix(original_input, operands_to_add);
      TF_ASSIGN_OR_RETURN(
          new_computation,
          WidenComputation(branch_computation, new_operand->shape()));
    }
    // Set the dynamic dimensions for the newly created branch computation's
    // parameters so that the hlos inside the computation can see dynamic
    // dimensions.
    DynamicParameterBinding dynamic_parameter_binding;
    TF_RETURN_IF_ERROR(ForEachDynamicDimensionInOperand(
        hlo, operand_index,
        [&](HloInstruction*, ShapeIndex index, int64 dimension,
            int64 operand_index, HloInstruction* dynamic_size) {
          DynamicParameterBinding::DynamicParameter dynamic_parameter{
              0, {dynamic_size_to_operand_id_index_map[dynamic_size]}};
          DynamicParameterBinding::DynamicDimension dynamic_dimension{
              0, {index}, dimension};
          TF_RETURN_IF_ERROR(dynamic_parameter_binding.Bind(dynamic_parameter,
                                                            dynamic_dimension));

          return Status::OK();
        }));
    VLOG(2) << "dynamic_parameter_binding for conditional branch"
            << dynamic_parameter_binding;
    TF_RETURN_IF_ERROR(DynamicDimensionInferenceVisitor::Run(
        new_computation, dynamic_parameter_binding, parent_));
    std::vector<HloInstruction*> hlos_to_add_in_root;
    int64 original_tuple_count = hlo->shape().tuple_shapes_size();
    // There may be some dynamic dimensions coming out of the computation, wire
    // that into the root instruction as additional tuple elements.
    TF_RETURN_IF_ERROR(ForEachDynamicDimension(
        new_computation->root_instruction(),
        [&](ShapeIndex index, int64 dim,
            HloInstruction* dynamic_size) -> Status {
          TF_RET_CHECK(hlo->shape().IsTuple())
              << "Only tuple typed conditionals can have dynamic dimension. "
                 "Please file a bug against XLA team.";
          dynamic_output_mapping.mutable_element(index)->emplace(
              dim, original_tuple_count++);
          hlos_to_add_in_root.push_back(dynamic_size);
          return Status::OK();
        }));

    VLOG(2) << "hlos_to_add_in_root:" << hlos_to_add_in_root.size();
    if (!hlos_to_add_in_root.empty()) {
      need_rewrite = true;
      HloInstruction* new_branch_root = TupleUtil::AppendSuffix(
          new_computation->root_instruction(), hlos_to_add_in_root);
      new_computation->set_root_instruction(new_branch_root,
                                            /*accept_different_shape=*/true);
    }

    new_branch_computations.push_back(new_computation);
    new_operands.push_back(new_operand);
  }
  if (!need_rewrite) {
    return Status::OK();
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
          const absl::flat_hash_map<int64, int64>& dim_to_output) {
        for (auto iter : dim_to_output) {
          int64 dim = iter.first;
          int64 output_index = iter.second;
          HloInstruction* dynamic_size = hlo->parent()->AddInstruction(
              HloInstruction::CreateGetTupleElement(
                  ShapeUtil::MakeScalarShape(S32), new_conditional,
                  output_index));
          parent_->SetDynamicSize(new_conditional, index, dim, dynamic_size);
          parent_->SetDynamicSize(new_conditional_extracted, index, dim,
                                  dynamic_size);
        }
      });

  TF_RETURN_IF_ERROR(hlo->ReplaceAllUsesWith(new_conditional_extracted));
  // Remove the original instruction even if has side-effects.
  TF_RETURN_IF_ERROR(hlo->parent()->RemoveInstruction(hlo));
  SetVisited(*new_conditional);
  SetVisited(*new_conditional_extracted);
  return Status::OK();
}

Status DynamicDimensionInferenceVisitor::HandleScatter(HloInstruction* hlo) {
  return ForEachOperandDynamicDimension(
      hlo,
      [&](HloInstruction* /*operand*/, ShapeIndex /*index*/, int64 dimension,
          int64 operand_index, HloInstruction* operand_dynamic_size) {
        if (operand_index == 0) {
          parent_->SetDynamicSize(hlo, {}, dimension, operand_dynamic_size);
          return Status::OK();
        }

        const ScatterDimensionNumbers& scatter_dims =
            hlo->scatter_dimension_numbers();
        if (operand_index == 2 &&
            absl::c_linear_search(scatter_dims.update_window_dims(),
                                  dimension)) {
          // Dynamic update window dimension is only allowed if it is exactly
          // the same as the corresponding operand dimension.
          std::vector<int64> update_window_dims_in_operand;
          for (int64 i = 0; i < hlo->operand(0)->shape().rank(); ++i) {
            if (absl::c_linear_search(scatter_dims.inserted_window_dims(), i)) {
              continue;
            }
            update_window_dims_in_operand.push_back(i);
          }

          for (int64 i = 0; i < scatter_dims.update_window_dims_size(); ++i) {
            if (scatter_dims.update_window_dims(i) == dimension) {
              const Shape& operand_shape = hlo->operand(0)->shape();
              const Shape& update_shape = hlo->operand(2)->shape();
              int64 dim_in_operand = update_window_dims_in_operand[i];
              if (operand_shape.dimensions(dim_in_operand) !=
                      update_shape.dimensions(dimension) ||
                  !operand_shape.is_dynamic_dimension(dim_in_operand)) {
                return Unimplemented(
                    "Dynamic dimension of update window dims that are not the "
                    "same as corresponding operand dim is not supported: "
                    "%s",
                    hlo->ToString());
              }
              HloInstruction* base_dynamic_size = parent_->GetDynamicSize(
                  hlo->mutable_operand(0), {}, dim_in_operand);
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
        return Status::OK();
      });
}

Status DynamicDimensionInferenceVisitor::HandleWhile(HloInstruction* hlo) {
  // If the output of the kWhile contains dynamic dimension, we send
  // dynamic dimension size into the while body by adding additional root/body
  // element. A mapping from the root instruction's dynamic dimension index
  // (represented by a shape index as output index and an int64 dimension
  // number) to output index (represented by an int64) is tracked for the
  // conditional instruction.
  ShapeTree<absl::flat_hash_map<int64, int64>> dynamic_output_mapping(
      hlo->shape());
  std::vector<HloInstruction*> operands_to_add;
  const int64 original_tuple_count = hlo->shape().tuple_shapes_size();
  int64 operand_count = original_tuple_count;
  TF_RETURN_IF_ERROR(ForEachOperandDynamicDimension(
      hlo, [&](HloInstruction*, ShapeIndex index, int64 dim, int64,
               HloInstruction* dynamic_size) {
        operands_to_add.push_back(dynamic_size);
        dynamic_output_mapping.mutable_element(index)->emplace(dim,
                                                               operand_count++);
        return Status::OK();
      }));

  DynamicParameterBinding binding_for_while;
  if (!operands_to_add.empty()) {
    // Only replace the while loop if there are new parameters to add.
    HloInstruction* old_tuple_operand = hlo->mutable_operand(0);
    TF_ASSIGN_OR_RETURN(
        WhileUtil::MakeInstructionsLiveInResult result,
        WhileUtil::MakeInstructionsLiveIn(hlo, operands_to_add));
    // WhileUtil creates a new while hlo and tuple. Update the dynamic size
    // mapping for the newly created tuple.
    HloInstruction* new_tuple_operand =
        result.new_while_instr->mutable_operand(0);
    parent_->CopyMapping(/*from=*/old_tuple_operand,
                         /*to=*/new_tuple_operand);
    hlo = result.new_while_instr;
    // We have replaced the while loop, now set the dynamic dimensions for the
    // newly created while loop so that the hlos that consumes the while loop
    // can see the dynamic dimensions. Also sets the dynamic parameter binding
    // for running inference in the while loop.
    TF_RETURN_IF_ERROR(ForEachOperandDynamicDimension(
        hlo,
        [&](HloInstruction*, ShapeIndex index, int64 dimension,
            int64 operand_index, HloInstruction* dynamic_size) -> Status {
          TF_RET_CHECK(!operands_to_add.empty());
          const int64 output_dynamic_size_index =
              dynamic_output_mapping.element(index).at(dimension);
          DynamicParameterBinding::DynamicParameter dynamic_parameter{
              operand_index, {output_dynamic_size_index}};
          DynamicParameterBinding::DynamicDimension dynamic_dimension{
              operand_index, index, dimension};
          TF_RETURN_IF_ERROR(
              binding_for_while.Bind(dynamic_parameter, dynamic_dimension));
          // This is the updated output dynamic size coming out of hlo while
          // loop.
          HloInstruction* output_dynamic_size = hlo->parent()->AddInstruction(
              HloInstruction::CreateGetTupleElement(
                  ShapeUtil::MakeScalarShape(S32), hlo,
                  output_dynamic_size_index));
          parent_->SetDynamicSize(result.replacement_instr, index, dimension,
                                  output_dynamic_size);
          return Status::OK();
        }));
    // Set the replacement instruction as visited to avoid visiting it again.
    SetVisited(*result.replacement_instr);
  }

  // Run inference in while body and condition.
  TF_RETURN_IF_ERROR(DynamicDimensionInferenceVisitor::Run(
      hlo->while_body(), binding_for_while, parent_));
  TF_RETURN_IF_ERROR(DynamicDimensionInferenceVisitor::Run(
      hlo->while_condition(), binding_for_while, parent_));

  if (operands_to_add.empty()) {
    // No dynamic dimension in the inputs and outputs.
    return Status::OK();
  }

  // The dynamic dimension size could have been changed in the loop body (e.g, A
  // loop that inserts items in a stack, the stack size increases with each
  // iteration). Rewrite the dynamic dimension size at the root.
  HloInstruction* body_root = hlo->while_body()->root_instruction();
  std::vector<HloInstruction*> new_root_operands(body_root->operand_count(),
                                                 nullptr);

  // Original non-dynamic-dim operands of root are pass-through.
  for (int64 i = 0; i < original_tuple_count; ++i) {
    new_root_operands[i] =
        hlo->while_body()->AddInstruction(HloInstruction::CreateGetTupleElement(
            body_root->shape().tuple_shapes(i), body_root, i));
  }
  // Add dynamic dimension size as new parameters.
  TF_RETURN_IF_ERROR(ForEachDynamicDimension(
      hlo->while_body()->root_instruction(),
      [&](ShapeIndex index, int64 dim, HloInstruction* dynamic_size) -> Status {
        const int64 output_index =
            dynamic_output_mapping.element(index).at(dim);
        new_root_operands[output_index] = dynamic_size;
        return Status::OK();
      }));
  for (auto operand : new_root_operands) {
    TF_RET_CHECK(operand != nullptr);
  }
  HloInstruction* new_body_root = hlo->while_body()->AddInstruction(
      HloInstruction::CreateTuple(new_root_operands));
  hlo->while_body()->set_root_instruction(new_body_root);
  return Status::OK();
}

Status DynamicDimensionInferenceVisitor::HandleParameter(HloInstruction* hlo) {
  return param_bindings_.ForEachBinding(
      [&](const DynamicParameterBinding::DynamicParameter& dynamic_parameter,
          const DynamicParameterBinding::DynamicDimension& dynamic_dimension) {
        if (dynamic_dimension.parameter_num != hlo->parameter_number()) {
          return Status::OK();
        }
        HloComputation* computation = hlo->parent();
        HloInstruction* target_parameter =
            computation->parameter_instruction(dynamic_dimension.parameter_num);

        HloInstruction* dynamic_size =
            computation->parameter_instruction(dynamic_parameter.parameter_num);
        for (int64 i : dynamic_parameter.parameter_index) {
          dynamic_size =
              computation->AddInstruction(HloInstruction::CreateGetTupleElement(
                  ShapeUtil::GetSubshape(dynamic_size->shape(), {i}),
                  dynamic_size, i));
        }

        parent_->SetDynamicSize(target_parameter,
                                dynamic_dimension.parameter_index,
                                dynamic_dimension.dimension, dynamic_size);
        return Status::OK();
      });
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
  return Status::OK();
}

Status DynamicDimensionInferenceVisitor::ForEachDynamicDimensionInOperand(
    HloInstruction* inst, int64 operand_index,
    const OperandDynamicDimensionFn& fn) {
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
  return Status::OK();
}

Status DynamicDimensionInferenceVisitor::ForEachOperandDynamicDimension(
    HloInstruction* inst, const OperandDynamicDimensionFn& fn) {
  for (int64 operand_index = 0; operand_index < inst->operand_count();
       ++operand_index) {
    TF_RETURN_IF_ERROR(
        ForEachDynamicDimensionInOperand(inst, operand_index, fn));
  }
  return Status::OK();
}

void DynamicDimensionInference::SetDynamicSize(HloInstruction* inst,
                                               const ShapeIndex& index,
                                               int64 dim,
                                               HloInstruction* size) {
  VLOG(1) << "Set dimension inst " << inst->ToString() << " index "
          << index.ToString() << "@" << dim << " to " << size->ToShortString();
  Shape subshape = ShapeUtil::GetSubshape(inst->shape(), index);
  CHECK(!subshape.IsTuple()) << "Can't set a tuple shape to dynamic dimension";
  CHECK(dim < subshape.rank() && dim >= 0)
      << "Asked to set invalid dynamic dimension. Shape: "
      << subshape.ToString() << ", Dimension: " << dim;
  DynamicDimension dynamic_dimension{inst, index, dim};
  // Updating a dynamic dimension twice overwrites the previous one.
  dynamic_mapping_[dynamic_dimension] = size;
  auto iter = per_hlo_dynamic_dimensions_.try_emplace(inst);
  iter.first->second.emplace(dynamic_dimension);
}

void DynamicDimensionInference::CopyMapping(HloInstruction* from,
                                            HloInstruction* to) {
  auto iter = per_hlo_dynamic_dimensions_.find(from);
  if (iter != per_hlo_dynamic_dimensions_.end()) {
    for (auto& dynamic_dimension : iter->second) {
      HloInstruction* dynamic_size =
          GetDynamicSize(dynamic_dimension.inst, dynamic_dimension.index,
                         dynamic_dimension.dim);
      SetDynamicSize(to, dynamic_dimension.index, dynamic_dimension.dim,
                     dynamic_size);
    }
  }
}

/* static */
StatusOr<DynamicDimensionInference> DynamicDimensionInference::Run(
    HloModule* module, CustomCallInferenceHandler custom_call_handler) {
  VLOG(2) << "Param Config " << module->dynamic_parameter_binding().ToString();
  DynamicDimensionInference inference(module, std::move(custom_call_handler));
  TF_RETURN_IF_ERROR(inference.AnalyzeDynamicDimensions());
  return inference;
}

string DynamicDimensionInference::ToString() const {
  std::vector<string> pieces;
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
    HloModule* module, CustomCallInferenceHandler custom_call_handler)
    : module_(module), custom_call_handler_(std::move(custom_call_handler)) {}

Status DynamicDimensionInference::AnalyzeDynamicDimensions() {
  return DynamicDimensionInferenceVisitor::Run(
      module_->entry_computation(), module_->dynamic_parameter_binding(), this,
      custom_call_handler_);
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
  CHECK(Shape::Equal()(inst->shape(), new_inst->shape()));

  for (int64 dim = 0; dim < inst->shape().rank(); ++dim) {
    DynamicDimension dynamic_dimension_new{new_inst, index, dim};
    DynamicDimension dynamic_dimension{inst, index, dim};
    auto iter = dynamic_mapping_.find(dynamic_dimension);
    if (iter != dynamic_mapping_.end()) {
      dynamic_mapping_.insert({dynamic_dimension_new, iter->second});
      auto iter = per_hlo_dynamic_dimensions_.try_emplace(new_inst);
      iter.first->second.emplace(dynamic_dimension_new);
    }
  }

  return Status::OK();
}

bool DynamicDimensionInference::HasDynamicDimension(
    HloInstruction* inst) const {
  bool has_dynamic_dim = false;
  ShapeUtil::ForEachSubshape(
      inst->shape(), [&](const Shape& subshape, const ShapeIndex& index) {
        if (subshape.IsTuple()) {
          return;
        }
        for (int64 i = 0; i < subshape.dimensions_size(); ++i) {
          HloInstruction* operand_dynamic_size = GetDynamicSize(inst, index, i);
          if (operand_dynamic_size != nullptr) {
            has_dynamic_dim = true;
          }
        }
      });
  return has_dynamic_dim;
}

HloInstruction* DynamicDimensionInference::GetDynamicSize(
    HloInstruction* inst, const ShapeIndex& index, int64 dim) const {
  auto iter = dynamic_mapping_.find(DynamicDimension{inst, index, dim});
  if (iter != dynamic_mapping_.end()) {
    return iter->second;
  }
  return nullptr;
}

std::vector<HloInstruction*> DynamicDimensionInference::GetDynamicSizes(
    HloInstruction* inst, const ShapeIndex& index) const {
  CHECK(ShapeUtil::IndexIsValid(inst->shape(), index));
  const int64 rank = ShapeUtil::GetSubshape(inst->shape(), index).rank();
  std::vector<HloInstruction*> result(rank, nullptr);
  for (int64 i = 0; i < rank; ++i) {
    result[i] = GetDynamicSize(inst, {}, i);
  }
  return result;
}

}  // namespace xla

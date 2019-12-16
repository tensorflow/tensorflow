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

#include "absl/strings/match.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/service/dfs_hlo_visitor_with_default.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/while_util.h"
#include "tensorflow/compiler/xla/window_util.h"

namespace xla {

class DynamicDimensionInferenceVisitor : public DfsHloVisitorWithDefault {
 public:
  explicit DynamicDimensionInferenceVisitor(
      const DynamicParameterBinding& param_bindings,
      DynamicDimensionInference* parent)
      : param_bindings_(param_bindings), parent_(parent) {}

  Status DefaultAction(HloInstruction* hlo) override;

  static Status Run(HloComputation* computation,
                    const DynamicParameterBinding& param_bindings,
                    DynamicDimensionInference* parent) {
    DynamicDimensionInferenceVisitor visitor(param_bindings, parent);
    return computation->Accept(&visitor);
  }

  Status HandleParameter(HloInstruction* hlo) override;

  Status HandleReduce(HloInstruction* hlo) override;

  Status HandleDot(HloInstruction* hlo) override;

  Status HandleTuple(HloInstruction* hlo) override;

  Status HandleTranspose(HloInstruction* hlo) override;

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

  Status HandleWhile(HloInstruction* hlo) override;

  Status HandleSlice(HloInstruction* hlo) override;

  Status HandleDynamicSlice(HloInstruction* hlo) override;

  Status HandleDynamicUpdateSlice(HloInstruction* hlo) override;

  Status HandleGather(HloInstruction* hlo) override;

  Status HandleScatter(HloInstruction* hlo) override;

 private:
  using DimensionConstraint = DynamicDimensionInference::DimensionConstraint;
  using OperandDynamicDimensionFn = std::function<Status(
      HloInstruction* operand, ShapeIndex index, int64 dimension,
      int64 operand_index, HloInstruction* dynamic_size,
      DimensionConstraint constraint)>;

  Status ForEachOperandDynamicDimension(HloInstruction* inst,
                                        const OperandDynamicDimensionFn&);
  Status ForEachDynamicDimensionInOperand(HloInstruction* inst,
                                          int64 operand_index,
                                          const OperandDynamicDimensionFn&);

  // Pass through a dynamic dimension from the input to the output with the same
  // value and index in the shape. This is a helper function to handle trivial
  // instructions like elementwise operations.
  Status PassThroughDynamicDimension(HloInstruction*);

  // The dynamic parameter bindings of this computation.
  const DynamicParameterBinding& param_bindings_;

  // A pointer to DynamicDimensionInference, used to update the dynamic mapping.
  DynamicDimensionInference* parent_;
};

Status DynamicDimensionInferenceVisitor::DefaultAction(HloInstruction* hlo) {
  return ForEachOperandDynamicDimension(
      hlo, [&](HloInstruction* operand, ShapeIndex index, int64 dimension,
               int64 operand_index, HloInstruction* dynamic_size,
               DimensionConstraint constraint) {
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
               int64 operand_index, HloInstruction* dynamic_size,
               DimensionConstraint constraint) {
        if (hlo->tuple_index() == index[0]) {
          ShapeIndex new_index =
              ShapeIndexView(index).ConsumeFront().ToShapeIndex();
          parent_->SetDynamicSize(hlo, new_index, dimension, dynamic_size,
                                  constraint);
        }
        return Status::OK();
      });
}

Status DynamicDimensionInferenceVisitor::HandleTuple(HloInstruction* hlo) {
  return ForEachOperandDynamicDimension(
      hlo, [&](HloInstruction*, ShapeIndex index, int64 dimension,
               int64 operand_index, HloInstruction* dynamic_size,
               DimensionConstraint constraint) {
        index.push_front(operand_index);
        parent_->SetDynamicSize(hlo, index, dimension, dynamic_size,
                                constraint);
        return Status::OK();
      });
}

Status DynamicDimensionInferenceVisitor::HandleBroadcast(HloInstruction* hlo) {
  return ForEachOperandDynamicDimension(
      hlo, [&](HloInstruction* operand, ShapeIndex index, int64 dimension,
               int64 operand_index, HloInstruction* dynamic_size,
               DimensionConstraint constraint) {
        int64 broadcast_dim = hlo->dimensions(dimension);
        parent_->SetDynamicSize(hlo, {}, broadcast_dim, dynamic_size,
                                constraint);
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
        parent_->SetDynamicSize(hlo, data_output, i, dynamic_size,
                                {.stride = 1, .multiple_of = 1});
      }
    }
    return Status::OK();
  }
  return ForEachOperandDynamicDimension(
      hlo, [&](HloInstruction* operand, ShapeIndex index, int64 dimension,
               int64 operand_index, HloInstruction* dynamic_size,
               DimensionConstraint constraint) {
        if (hlo->custom_call_target() != "SliceToDynamic" ||
            absl::StartsWith(hlo->custom_call_target(), "Resize")) {
          return Unimplemented(
              "CustomCall is not supported to have a dynamic dimension");
        }

        parent_->SetDynamicSize(hlo, {}, dimension, dynamic_size, constraint);
        return Status::OK();
      });
}

Status DynamicDimensionInferenceVisitor::HandleSort(HloInstruction* hlo) {
  return ForEachOperandDynamicDimension(
      hlo, [&](HloInstruction* operand, ShapeIndex index,
               int64 dynamic_dimension, int64 operand_index,
               HloInstruction* dynamic_size, DimensionConstraint constraint) {
        HloSortInstruction* sort = Cast<HloSortInstruction>(hlo);
        int64 sort_dimension = sort->sort_dimension();
        if (sort_dimension == dynamic_dimension) {
          return Unimplemented(
              "Dynamic dimension on sorting dimension is not supported");
        }
        if (sort->values_count() == 0) {
          parent_->SetDynamicSize(hlo, {}, dynamic_dimension, dynamic_size,
                                  constraint);
        } else {
          parent_->SetDynamicSize(hlo, {operand_index}, dynamic_dimension,
                                  dynamic_size, constraint);
        }

        return Status::OK();
      });
}

Status DynamicDimensionInferenceVisitor::HandlePad(HloInstruction* hlo) {
  return ForEachOperandDynamicDimension(
      hlo, [&](HloInstruction* operand, ShapeIndex index, int64 dimension,
               int64 operand_index, HloInstruction* dynamic_size,
               DimensionConstraint constraint) {
        if (operand_index != 0) {
          return Unimplemented(
              "Dynamic dimension on padding value is not supported");
        }
        const PaddingConfig_PaddingConfigDimension& padding_config =
            hlo->padding_config().dimensions(dimension);
        if (padding_config.interior_padding() == 0 &&
            padding_config.edge_padding_low() == 0 &&
            padding_config.edge_padding_high() == 0) {
          parent_->SetDynamicSize(hlo, {}, dimension, dynamic_size, constraint);
          return Status::OK();
        } else {
          return Unimplemented(
              "Dynamic dimension propagation on padding dimension is not "
              "supported.");
        }
      });
}

Status DynamicDimensionInferenceVisitor::HandleReduce(HloInstruction* hlo) {
  return ForEachOperandDynamicDimension(
      hlo, [&](HloInstruction* operand, ShapeIndex index, int64 dimension,
               int64 operand_index, HloInstruction* dynamic_size,
               DimensionConstraint constraint) {
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
                parent_->SetDynamicSize(reduce, {i},
                                        dimensions_not_reduced_count,
                                        dynamic_size, constraint);
              }
            } else {
              parent_->SetDynamicSize(reduce, {}, dimensions_not_reduced_count,
                                      dynamic_size, constraint);
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
               HloInstruction* dynamic_size, DimensionConstraint constraint) {
        // There are three types of dimensions in a dot:
        // A. batch dims
        // B. contracting dims
        // C. non-batch non-contracting dims.
        // The output dimemsions of a dot has three parts with the following
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
        // diffrent batch dimension numbers.
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
          parent_->SetDynamicSize(dot, {}, iter->second, dynamic_size,
                                  constraint);
        }

        return Status::OK();
      });
}

Status DynamicDimensionInferenceVisitor::HandleTranspose(HloInstruction* hlo) {
  return ForEachOperandDynamicDimension(
      hlo, [&](HloInstruction* operand, ShapeIndex index, int64 dimension,
               int64 operand_index, HloInstruction* dynamic_size,
               DimensionConstraint constraint) {
        parent_->SetDynamicSize(hlo, {}, hlo->dimensions()[dimension],
                                dynamic_size, constraint);
        return Status::OK();
      });
}

Status DynamicDimensionInferenceVisitor::HandleConvolution(
    HloInstruction* hlo) {
  return ForEachOperandDynamicDimension(
      hlo, [&](HloInstruction* operand, ShapeIndex index, int64 dimension,
               int64 operand_index, HloInstruction* dynamic_size,
               DimensionConstraint constraint) {
        HloInstruction* conv = hlo;
        const ConvolutionDimensionNumbers& dimension_numbers =
            conv->convolution_dimension_numbers();

        if (operand_index == 0) {
          if (dimension == dimension_numbers.input_batch_dimension()) {
            parent_->SetDynamicSize(conv, {},
                                    dimension_numbers.output_batch_dimension(),
                                    dynamic_size, constraint);
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
                            dim_size_total, {.stride = 1, .multiple_of = 1});
  }

  // Simply pass through non-concat dynamic dimensions.
  return ForEachOperandDynamicDimension(
      hlo, [&](HloInstruction* operand, ShapeIndex index, int64 dimension,
               int64 operand_index, HloInstruction* dynamic_size,
               DimensionConstraint constraint) {
        int64 concatenate_dimension = hlo->concatenate_dimension();
        if (concatenate_dimension == dimension) {
          return Status::OK();
        }
        parent_->SetDynamicSize(hlo, index, dimension, dynamic_size,
                                constraint);
        return Status::OK();
      });
}

Status DynamicDimensionInferenceVisitor::HandleGetDimensionSize(
    HloInstruction*) {
  // Dynamic dimension doesn't propagate through GetDimensionSize:
  //
  //   Input: F32[x, y, z]
  //     |
  //   GetDimensionSize(1): S32[]
  //
  // The returned value is a scalar, which doesn't have any dynamic dimension in
  // the shape (although the value contains the real size of the dynamic
  // dimension of the input).
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
    parent_->SetDynamicSize(hlo, {}, hlo->dimension(), hlo->mutable_operand(1),
                            {.stride = 1, .multiple_of = 1});
  }

  // Also Propagate dynamic dimension already set by operands.
  TF_RETURN_IF_ERROR(ForEachOperandDynamicDimension(
      hlo, [&](HloInstruction* operand, ShapeIndex index, int64 dimension,
               int64 operand_index, HloInstruction* dynamic_size,
               DimensionConstraint constraint) {
        if (dimension != hlo->dimension()) {
          parent_->SetDynamicSize(hlo, index, dimension, dynamic_size,
                                  constraint);
        }
        return Status::OK();
      }));

  return Status::OK();
}

Status DynamicDimensionInferenceVisitor::PassThroughDynamicDimension(
    HloInstruction* hlo) {
  return ForEachOperandDynamicDimension(
      hlo, [&](HloInstruction* operand, ShapeIndex index, int64 dimension,
               int64 operand_index, HloInstruction* dynamic_size,
               DimensionConstraint constraint) {
        parent_->SetDynamicSize(hlo, index, dimension, dynamic_size,
                                constraint);
        return Status::OK();
      });
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
  return PassThroughDynamicDimension(hlo);
}

Status DynamicDimensionInferenceVisitor::HandleClamp(HloInstruction* hlo) {
  return PassThroughDynamicDimension(hlo);
}

Status DynamicDimensionInferenceVisitor::HandleReshape(HloInstruction* hlo) {
  return ForEachOperandDynamicDimension(
      hlo,
      [&](HloInstruction* operand, ShapeIndex index,
          int64 input_dynamic_dimension, int64 operand_index,
          HloInstruction* operand_dynamic_size,
          DimensionConstraint constraint) -> Status {
        HloInstruction* reshape = hlo;
        TF_RET_CHECK(reshape->shape().rank() > 0)
            << "Reshaping a dynamic dimension into a scalar, which has "
               "undefined behavior. The offending instruction is: "
            << reshape->ToString();
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
          // TODO(yunxing): We now have a better way to decide output dimension
          // in the bridge. No need for this constraint propagation logic.
          //
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
          //
          // 2. Use the "multiple_of" constraint, e.g, :
          //    [<=2, 4]
          //     | Reshape
          //    [<=8]
          //     | Reshape
          //    [2, 4] // Which is dynamic?
          //
          //    If the dynamic value has to be multiple of 4 (constraint
          //    created by the first reshape), then 2 must be the dynamic
          //    dimension.
          //
          //    But this logic doesn't help with the case where two
          //    dimensions are the same:
          //
          //    [<=3, 3]
          //     | Reshape
          //    [<=9]
          //     | Reshape
          //    [3, 3]  // Which is dynamic?
          //
          //    Both dynamic dimension can be multiple of 3.
          //
          //    We then need the next constraint to disambiguate this case:
          //
          // 3. Use the "stride" constraint (also see the comment at the
          // definition):
          //
          //        [<=3, 3]
          //           | Reshape
          //         [<=9] // constraint.stride = 1
          //          | Reshape
          //        [3, 3]
          //         ^  ^
          //         |  |
          // stride= 1  3
          //
          //    Each dimension will have different strides, only one will
          //    satisfy the stride constraint.
          //
          //    Note that the stride constrint itself is not enough:
          //
          //
          //         [<=128]
          //           | Reshape
          //         [1, 128]
          //          ^  ^
          //          |  |
          //  stride= 1  1
          //
          //    In this case, both dimensions have the same stride, which is
          //    ambiguous. That's why we need the "multiple_of" constraint
          //    as used above.
          //
          // 4. If all logics above cannot disambiguate, e.g.,:
          //
          //     [<=1]
          //      |
          //   reshape
          //      |
          //   [1, 1, 1]
          //
          //   We bail out and return an error.
          output_dynamic_dimension = reshape->inferred_dimension();
          if (output_dynamic_dimension == -1) {
            // The user of XLA didn't specify a dynamic dimension, try infer
            // it from the current constraint.
            //
            // Find all output dimensions that are decomposed from the first
            // dimension. Among those dimensions, find all dimensions that
            // satisfy the constraint of the dynamic dimension. In the
            // previous example, if `a` is 9 and constraint is a multiple of
            // `3', then in the output shape both a/c and c can be dynamic.
            int64 current_product = 1;
            int64 dimension_iter = output_dim_start;

            // compatible_dimensions are dimensions that satisfies
            // "multiple_of" constraints.
            std::vector<int64> compatible_dimensions;
            while (current_product <
                   operand->shape().dimensions(input_dynamic_dimension)) {
              current_product *= reshape->shape().dimensions(dimension_iter);
              if (operand->shape().dimensions(input_dynamic_dimension) /
                      reshape->shape().dimensions(dimension_iter) ==
                  constraint.multiple_of) {
                compatible_dimensions.push_back(dimension_iter);
              }
              dimension_iter++;
            }
            CHECK_EQ(current_product,
                     operand->shape().dimensions(input_dynamic_dimension))
                << "Not a valid reshape: " << hlo->ToString();
            // If there is only one compatible dimension, it must be the
            // dynamic one in the output.
            if (compatible_dimensions.size() == 1) {
              output_dynamic_dimension = compatible_dimensions[0];
            }

            // When there are multiple compatible dimensions, e.g:
            //     [<=9]
            //      | Reshape
            //    [3, 3]
            // Use stride constraint to figure out which one is the true
            // dynamic one.
            //
            //         [<=9]
            //          | Reshape
            //        [3, 3]
            //         ^  ^
            //         |  |
            // stride= 1  3
            //
            std::vector<int64> compatible_dimensions_with_stride;
            absl::c_copy_if(
                compatible_dimensions,
                std::back_inserter(compatible_dimensions_with_stride),
                [&](int64 dimension) {
                  int64 stride_total = 1;
                  for (int64 i = 0; i < dimension + 1; ++i) {
                    stride_total *= reshape->shape().dimensions(dimension);
                  }
                  return stride_total == constraint.stride;
                });
            if (compatible_dimensions_with_stride.size() == 1) {
              output_dynamic_dimension = compatible_dimensions_with_stride[0];
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
                "ambiguous and XLA can't infer the output dimension %s. "
                "Constraint: multiple_of: %lld, stride: %lld",
                hlo->ToString(), constraint.multiple_of, constraint.stride);
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
                                  operand_dynamic_size, constraint);
        }

        if (input_dim_size > output_dim_size) {
          TF_RET_CHECK(input_dim_size % output_dim_size == 0);
          const int64 divisor = input_dim_size / output_dim_size;
          HloInstruction* divisor_hlo =
              hlo->parent()->AddInstruction(HloInstruction::CreateConstant(
                  LiteralUtil::CreateR0<int32>(divisor)));

          HloInstruction* new_dynamic_size =
              hlo->parent()->AddInstruction(HloInstruction::CreateBinary(
                  operand_dynamic_size->shape(), HloOpcode::kDivide,
                  operand_dynamic_size, divisor_hlo));

          parent_->SetDynamicSize(
              reshape, {}, output_dynamic_dimension, new_dynamic_size,
              {.stride = 1, .multiple_of = constraint.multiple_of / divisor});
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
          parent_->SetDynamicSize(
              reshape, {}, output_dynamic_dimension, new_dynamic_size,
              {.stride = 1,
               .multiple_of =
                   constraint.multiple_of * output_dim_size /
                   operand->shape().dimensions(input_dynamic_dimension)});
        }

        return Status::OK();
      });
}

Status DynamicDimensionInferenceVisitor::HandleReduceWindow(
    HloInstruction* hlo) {
  return ForEachOperandDynamicDimension(
      hlo, [&](HloInstruction* operand, ShapeIndex index, int64 dimension,
               int64 operand_index, HloInstruction* dynamic_size,
               DimensionConstraint constraint) {
        HloInstruction* reduce_window = hlo;
        const WindowDimension& window_dimension =
            reduce_window->window().dimensions(dimension);

        if (!window_util::IsTrivialWindowDimension(window_dimension)) {
          return Unimplemented(
              "Dynamic Spatial reduce window is not supported: %s",
              reduce_window->ToString());
        }

        parent_->SetDynamicSize(reduce_window, {}, dimension, dynamic_size,
                                constraint);

        return Status::OK();
      });
}

Status DynamicDimensionInferenceVisitor::HandleSelectAndScatter(
    HloInstruction* hlo) {
  return ForEachOperandDynamicDimension(
      hlo, [&](HloInstruction* operand, ShapeIndex index, int64 dimension,
               int64 operand_index, HloInstruction* dynamic_size,
               DimensionConstraint constraint) {
        HloInstruction* select_and_scatter = hlo;
        const WindowDimension& window_dimension =
            select_and_scatter->window().dimensions(dimension);

        if (!window_util::IsTrivialWindowDimension(window_dimension)) {
          return Unimplemented(
              "Dynamic Spatial select and scatter is not supported: %s",
              select_and_scatter->ToString());
        }

        parent_->SetDynamicSize(select_and_scatter, {}, dimension, dynamic_size,
                                constraint);

        return Status::OK();
      });
}

Status DynamicDimensionInferenceVisitor::HandleSlice(HloInstruction* hlo) {
  return ForEachOperandDynamicDimension(
      hlo, [&](HloInstruction* operand, ShapeIndex /*index*/, int64 dimension,
               int64 /*operand_index*/, HloInstruction* dynamic_size,
               DimensionConstraint constraint) {
        if (hlo->slice_starts(dimension) != 0 ||
            hlo->slice_strides(dimension) != 1 ||
            hlo->slice_limits(dimension) !=
                operand->shape().dimensions(dimension)) {
          // Slicing a single element out eliminates the dynamic dimension.
          if (hlo->shape().dimensions(dimension) == 1) {
            return Status::OK();
          }
          return Unimplemented(
              "Dynamic dimension propagation on Slice where it doesn't slice "
              "out an entire dimension is not supported %s",
              hlo->ToString());
        }

        parent_->SetDynamicSize(hlo, {}, dimension, dynamic_size, constraint);

        return Status::OK();
      });
}

Status DynamicDimensionInferenceVisitor::HandleDynamicSlice(
    HloInstruction* hlo) {
  return ForEachOperandDynamicDimension(
      hlo, [&](HloInstruction*, ShapeIndex /*index*/, int64 dimension,
               int64 /*operand_index*/, HloInstruction* dynamic_size,
               DimensionConstraint constraint) {
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

        parent_->SetDynamicSize(hlo, {}, dimension, dynamic_size, constraint);

        return Status::OK();
      });
}

Status DynamicDimensionInferenceVisitor::HandleDynamicUpdateSlice(
    HloInstruction* hlo) {
  return ForEachOperandDynamicDimension(
      hlo, [&](HloInstruction* /*operand*/, ShapeIndex /*index*/,
               int64 dimension, int64 /*operand_index*/,
               HloInstruction* dynamic_size, DimensionConstraint constraint) {
        if (hlo->shape().dimensions(dimension) !=
            hlo->operand(0)->shape().dimensions(dimension)) {
          return Unimplemented(
              "Dynamic dimension propagation on DynamicSlice where a partial "
              "dimension is selected %s",
              hlo->ToString());
        }

        parent_->SetDynamicSize(hlo, {}, dimension, dynamic_size, constraint);

        return Status::OK();
      });
}

Status DynamicDimensionInferenceVisitor::HandleReverse(HloInstruction* hlo) {
  return ForEachOperandDynamicDimension(
      hlo, [&](HloInstruction* /*operand*/, ShapeIndex /*index*/,
               int64 dimension, int64 /*operand_index*/,
               HloInstruction* dynamic_size, DimensionConstraint constraint) {
        if (absl::c_linear_search(hlo->dimensions(), dimension)) {
          return Unimplemented(
              "Dynamic dimension propagation on reversed dimension is not "
              "supported %s",
              hlo->ToString());
        }
        parent_->SetDynamicSize(hlo, {}, dimension, dynamic_size, constraint);

        return Status::OK();
      });
}

Status DynamicDimensionInferenceVisitor::HandleGather(HloInstruction* hlo) {
  return ForEachOperandDynamicDimension(
      hlo, [&](HloInstruction* operand, ShapeIndex /*index*/,
               int64 input_dynamic_dimension, int64 operand_index,
               HloInstruction* dynamic_size, DimensionConstraint constraint) {
        const GatherDimensionNumbers& gather_dims =
            hlo->gather_dimension_numbers();
        if (operand_index != 1) {
          if (hlo->gather_slice_sizes()[input_dynamic_dimension] == 1) {
            // Gathering a size 1 dimension out of a dynamic dimension removes
            // the dynamisity.
            return Status::OK();
          }
          if (hlo->gather_slice_sizes()[input_dynamic_dimension] ==
              operand->shape().dimensions(input_dynamic_dimension)) {
            // Gathering a full-sized dimension out of a dynamic dimension
            // propagates the dynamisity to output.
            int64 output_dimension = input_dynamic_dimension;
            for (int64 collapsed_dim : gather_dims.collapsed_slice_dims()) {
              if (collapsed_dim < input_dynamic_dimension) {
                // This output dimension is collapsed.
                output_dimension--;
              }
            }
            parent_->SetDynamicSize(hlo, {}, output_dimension, dynamic_size,
                                    constraint);
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
              parent_->SetDynamicSize(hlo, {}, output_dim, dynamic_size,
                                      constraint);
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

Status DynamicDimensionInferenceVisitor::HandleScatter(HloInstruction* hlo) {
  return ForEachOperandDynamicDimension(
      hlo,
      [&](HloInstruction* /*operand*/, ShapeIndex /*index*/, int64 dimension,
          int64 operand_index, HloInstruction* operand_dynamic_size,
          DimensionConstraint constraint) {
        if (operand_index == 0) {
          parent_->SetDynamicSize(hlo, {}, dimension, operand_dynamic_size,
                                  constraint);
          return Status::OK();
        }

        const ScatterDimensionNumbers& scatter_dims =
            hlo->scatter_dimension_numbers();
        if (operand_index == 2 &&
            absl::c_linear_search(scatter_dims.update_window_dims(),
                                  dimension)) {
          return Unimplemented(
              "Dynamic dimension of update window dims is not supported "
              "is not supported: %s",
              hlo->ToString());
        }
        // The dynamic dimension is collapsed and won't show up in the output.
        // Do nothing here.
        return Status::OK();
      });
}

Status DynamicDimensionInferenceVisitor::HandleWhile(HloInstruction* hlo) {
  // While loop is handled by passing dynamic size hlos as parameters into the
  // hlo while loop. This is done by replacing the original while with a new
  // one.
  //
  // Before:
  //
  // op1 = ...
  // op2 = ...
  // op1_x = ... // dynamic dimension size of op1
  // while = while(op1, op2)
  //
  //
  // After:
  //
  // op1 = ...
  // op2 = ...
  // op1_x = ... // dynamic dimension size of op1
  // while = while(op1, op2, op1_x)
  //
  // In the above graph, op_x is the bound of the dynamic dimension size of op1
  // and is wired into the while loop as new parameter.
  //
  // TODO(b/119843103): Once we implement dynamic bounds in XLA backend, dynamic
  // bound can be propagated through native xla values instead of relying on
  // additional parameter.

  // dynamic_size_to_operand_id_index_map keeps track of dynamic size operations
  // to their operand ids in the new while loop.
  absl::flat_hash_map<HloInstruction*, int64>
      dynamic_size_to_operand_id_index_map;

  // operands_to_add collects dynamic sizes that need to be added to the while
  // loop as parameters. Note that a dynamic size is ignored if it is already
  // part of the parameter. i.e.:
  //
  // We don't do:
  //
  // op1 = ...
  // op2 = ...
  // op_x = ... // dynamic dimension size of both op1 and op2
  // while = while(op1, op2, op_x, op_x) // 4 parameters
  //
  // But we do:
  //
  // op1 = ...
  // op2 = ...
  // op_x = ... // dynamic dimension size of both op1 and op2
  // while = while(op1, op2, op_x)
  //
  // An alternative is to do this in a while loop CSE pass.
  //
  std::vector<HloInstruction*> operands_to_add;
  int64 operand_count = hlo->shape().tuple_shapes_size();
  TF_RETURN_IF_ERROR(ForEachOperandDynamicDimension(
      hlo, [&](HloInstruction*, ShapeIndex, int64, int64,
               HloInstruction* dynamic_size, DimensionConstraint constraint) {
        const HloInstruction* tuple_operand = hlo->operand(0);
        for (int64 i = 0; i < tuple_operand->operand_count(); ++i) {
          if (dynamic_size == tuple_operand->operand(i)) {
            dynamic_size_to_operand_id_index_map[dynamic_size] = i;
            return Status::OK();
          }
        }
        auto iter = dynamic_size_to_operand_id_index_map.find(dynamic_size);
        if (iter == dynamic_size_to_operand_id_index_map.end()) {
          operands_to_add.push_back(dynamic_size);
          dynamic_size_to_operand_id_index_map[dynamic_size] = operand_count++;
        }
        return Status::OK();
      }));

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
  }

  // We have replaced the while loop, now set the dynamic dimensions for the
  // newly created while loop so that the hlos that consumes the while loop can
  // see the dynamic dimensions. Also sets the dynamic parameter binding for
  // running inference in the while loop.
  DynamicParameterBinding binding_for_while;
  TF_RETURN_IF_ERROR(ForEachOperandDynamicDimension(
      hlo, [&](HloInstruction*, ShapeIndex index, int64 dimension,
               int64 operand_index, HloInstruction* dynamic_size,
               DimensionConstraint constraint) {
        DynamicParameterBinding::DynamicParameter dynamic_parameter{
            operand_index,
            {dynamic_size_to_operand_id_index_map[dynamic_size]}};
        DynamicParameterBinding::DynamicDimension dynamic_dimension{
            operand_index, index, dimension};
        TF_RETURN_IF_ERROR(
            binding_for_while.Bind(dynamic_parameter, dynamic_dimension));
        parent_->SetDynamicSize(hlo, index, dimension, dynamic_size,
                                constraint);
        return Status::OK();
      }));

  // Run inference in while body and condition.
  TF_RETURN_IF_ERROR(DynamicDimensionInferenceVisitor::Run(
      hlo->while_body(), binding_for_while, parent_));
  TF_RETURN_IF_ERROR(DynamicDimensionInferenceVisitor::Run(
      hlo->while_condition(), binding_for_while, parent_));

  // Set the replacement while loop as visited to avoid visiting it again.
  SetVisited(*hlo);
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
                                dynamic_dimension.dimension, dynamic_size,
                                {.stride = 1, .multiple_of = 1});
        return Status::OK();
      });
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
      CHECK_NE(parent_->constraint_mapping_.count(dynamic_dimension), 0);
      TF_RETURN_IF_ERROR(fn(dynamic_dimension.inst, dynamic_dimension.index,
                            dynamic_dimension.dim, operand_index, dynamic_size,
                            parent_->constraint_mapping_[dynamic_dimension]));
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

void DynamicDimensionInference::CopyMapping(HloInstruction* from,
                                            HloInstruction* to) {
  auto iter = per_hlo_dynamic_dimensions_.find(from);
  if (iter != per_hlo_dynamic_dimensions_.end()) {
    for (auto& dynamic_dimension : iter->second) {
      HloInstruction* dynamic_size =
          GetDynamicSize(dynamic_dimension.inst, dynamic_dimension.index,
                         dynamic_dimension.dim);
      SetDynamicSize(to, dynamic_dimension.index, dynamic_dimension.dim,
                     dynamic_size, constraint_mapping_[dynamic_dimension]);
    }
  }
}

/* static */
StatusOr<DynamicDimensionInference> DynamicDimensionInference::Run(
    HloModule* module) {
  VLOG(2) << "Param Config " << module->dynamic_parameter_binding().ToString();
  DynamicDimensionInference inference(module);
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

DynamicDimensionInference::DynamicDimensionInference(HloModule* module)
    : module_(module) {}

Status DynamicDimensionInference::AnalyzeDynamicDimensions() {
  return DynamicDimensionInferenceVisitor::Run(
      module_->entry_computation(), module_->dynamic_parameter_binding(), this);
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
      constraint_mapping_.insert(
          {dynamic_dimension_new, constraint_mapping_[dynamic_dimension]});
      auto iter = per_hlo_dynamic_dimensions_.try_emplace(new_inst);
      iter.first->second.emplace(dynamic_dimension_new);
    }
  }

  return Status::OK();
}

HloInstruction* DynamicDimensionInference::GetDynamicSize(
    HloInstruction* inst, const ShapeIndex& index, int64 dim) const {
  auto iter = dynamic_mapping_.find(DynamicDimension{inst, index, dim});
  if (iter != dynamic_mapping_.end()) {
    return iter->second;
  }
  return nullptr;
}

}  // namespace xla

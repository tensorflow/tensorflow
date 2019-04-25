/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/service/dfs_hlo_visitor_with_default.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
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

  Status HandlePad(HloInstruction* hlo) override;

  Status HandleBroadcast(HloInstruction* hlo) override;

  Status HandleGetDimensionSize(HloInstruction* hlo) override;

  Status HandleSelect(HloInstruction* hlo) override;

  Status HandleConvolution(HloInstruction* hlo) override;

  Status HandleReduceWindow(HloInstruction* hlo) override;

  Status HandleSelectAndScatter(HloInstruction* hlo) override;

  Status HandleGetTupleElement(HloInstruction* hlo) override;

  Status HandleElementwiseUnary(HloInstruction* hlo) override;

  Status HandleElementwiseBinary(HloInstruction* hlo) override;

  Status HandleWhile(HloInstruction* hlo) override;

  Status HandleSlice(HloInstruction* hlo) override;

  Status HandleDynamicSlice(HloInstruction* hlo) override;

  Status HandleDynamicUpdateSlice(HloInstruction* hlo) override;

  Status HandleGather(HloInstruction* hlo) override;

  Status HandleScatter(HloInstruction* hlo) override;

 private:
  using OperandDynamicDimensionFn = std::function<Status(
      HloInstruction* operand, ShapeIndex index, int64 dimension,
      int64 operand_index, HloInstruction* dynamic_size)>;

  Status ForEachOperandDynamicDimension(HloInstruction* inst,
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
               int64 operand_index, HloInstruction* dynamic_size) {
        return UnimplementedStrCat(
            "Asked to propagate a dynamic dimension from hlo ",
            operand->ToString(), "@", index.ToString(), "@", dimension,
            " to hlo ", hlo->ToString(), ", which is not implemented.");
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
        if (padding_config.interior_padding() == 0 &&
            padding_config.edge_padding_low() == 0 &&
            padding_config.edge_padding_high() == 0) {
          parent_->SetDynamicSize(hlo, {}, dimension, dynamic_size);
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
               int64 operand_index, HloInstruction* dynamic_size) {
        HloInstruction* reduce = hlo;
        int64 operand_count = reduce->operand_count();
        CHECK_EQ(operand_count % 2, 0);
        if (operand_index >= operand_count / 2) {
          // Init values doesn't have dynamic size.
          return Status::OK();
        }
        if ((absl::c_count(reduce->dimensions(), dimension) != 0)) {
          // Dimension is to be reduce, stop tracing.
          return Status::OK();
        }

        // Find out the new dynamic dimension after reduce.
        int64 dimensions_not_reduced_count = 0;
        for (int i = 0; i < operand->shape().rank(); ++i) {
          if (dimension == i) {
            parent_->SetDynamicSize(reduce, {}, dimensions_not_reduced_count,
                                    dynamic_size);

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
      hlo, [&](HloInstruction* operand, ShapeIndex index, int64 dimension,
               int64 operand_index, HloInstruction* dynamic_size) {
        HloInstruction* dot = hlo;
        const DotDimensionNumbers& dimension_numbers =
            dot->dot_dimension_numbers();
        // A map from the operand dimensions to result dimension.
        absl::flat_hash_map<int64, int64> result_dim_mapping;
        int64 current_result_dims = 0;
        std::unordered_set<int64> batch_dims(
            dimension_numbers.rhs_batch_dimensions().begin(),
            dimension_numbers.rhs_batch_dimensions().end());

        for (int64 i : dimension_numbers.rhs_batch_dimensions()) {
          result_dim_mapping[i] = current_result_dims++;
        }

        for (int64 i = 0; i < dot->operand(0)->shape().rank(); i++) {
          if (!absl::c_linear_search(
                  dimension_numbers.lhs_contracting_dimensions(), i)) {
            if (operand_index == 0) {
              result_dim_mapping[i] = current_result_dims;
            }
            current_result_dims++;
          }
        }

        for (int64 i = 0; i < dot->operand(1)->shape().rank(); i++) {
          if (!absl::c_linear_search(
                  dimension_numbers.rhs_contracting_dimensions(), i) &&
              !absl::c_linear_search(dimension_numbers.rhs_batch_dimensions(),
                                     i)) {
            if (operand_index == 1) {
              result_dim_mapping[i] = current_result_dims;
            }
            current_result_dims++;
          }
        }

        // Check if the operand dim is in the result shape. If so, add another
        // work item to trace that dimension.
        auto iter = result_dim_mapping.find(dimension);
        if (iter != result_dim_mapping.end()) {
          parent_->SetDynamicSize(dot, {}, iter->second, dynamic_size);
        }

        return Status::OK();
      });
}

Status DynamicDimensionInferenceVisitor::HandleTranspose(HloInstruction* hlo) {
  return ForEachOperandDynamicDimension(
      hlo, [&](HloInstruction* operand, ShapeIndex index, int64 dimension,
               int64 operand_index, HloInstruction* dynamic_size) {
        parent_->SetDynamicSize(hlo, {}, hlo->dimensions()[dimension],
                                dynamic_size);
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

Status DynamicDimensionInferenceVisitor::HandleGetDimensionSize(
    HloInstruction*) {
  // Dynamic dimension doesn't propagate through GetDimensionSize:
  //
  //   Input: F32[x, y, z]
  //     |
  //   GetDimensionSize(1): U32[]
  //
  // The returned value is a scalar, which doesn't have any dynamic dimension in
  // the shape (although the value contains the real size of the dynamic
  // dimension of the input).
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

Status DynamicDimensionInferenceVisitor::HandleReshape(HloInstruction* hlo) {
  return ForEachOperandDynamicDimension(
      hlo, [&](HloInstruction* operand, ShapeIndex index, int64 dimension,
               int64 operand_index, HloInstruction* dynamic_size) {
        HloInstruction* reshape = hlo;
        std::vector<std::pair<int64, int64>> unmodified_dims =
            ShapeUtil::DimensionsUnmodifiedByReshape(operand->shape(),
                                                     reshape->shape());
        for (auto& unmodified : unmodified_dims) {
          if (unmodified.first == dimension) {
            parent_->SetDynamicSize(reshape, {}, unmodified.second,
                                    dynamic_size);
            return Status::OK();
          }
        }
        // A dimension modifying reshape is supported as long as it is the most
        // major one and it is combining with other non-dynamic dimensions.
        const int64 output_most_major = reshape->shape().dimensions(0);
        const int64 input_most_major = operand->shape().dimensions(0);
        if (dimension == 0) {
          if (output_most_major > input_most_major) {
            const int64 multiplier =
                reshape->shape().dimensions(0) / operand->shape().dimensions(0);
            HloInstruction* multiplier_hlo =
                hlo->parent()->AddInstruction(HloInstruction::CreateConstant(
                    LiteralUtil::CreateR0<uint32>(multiplier)));

            HloInstruction* new_dynamic_size =
                hlo->parent()->AddInstruction(HloInstruction::CreateBinary(
                    dynamic_size->shape(), HloOpcode::kMultiply, dynamic_size,
                    multiplier_hlo));
            parent_->SetDynamicSize(reshape, {}, 0, new_dynamic_size);
            return Status::OK();
          } else if (output_most_major < input_most_major) {
            const int64 divisor =
                operand->shape().dimensions(0) / reshape->shape().dimensions(0);
            HloInstruction* divisor_hlo =
                hlo->parent()->AddInstruction(HloInstruction::CreateConstant(
                    LiteralUtil::CreateR0<uint32>(divisor)));

            HloInstruction* new_dynamic_size =
                hlo->parent()->AddInstruction(HloInstruction::CreateBinary(
                    dynamic_size->shape(), HloOpcode::kDivide, dynamic_size,
                    divisor_hlo));
            parent_->SetDynamicSize(reshape, {}, 0, new_dynamic_size);
            return Status::OK();
          }
        }
        return Unimplemented(
            "Dynamic Reshape on modified dimensions is yet not supported: %s",
            reshape->ToString());
      });
}

Status DynamicDimensionInferenceVisitor::HandleReduceWindow(
    HloInstruction* hlo) {
  return ForEachOperandDynamicDimension(
      hlo, [&](HloInstruction* operand, ShapeIndex index, int64 dimension,
               int64 operand_index, HloInstruction* dynamic_size) {
        HloInstruction* reduce_window = hlo;
        const WindowDimension& window_dimension =
            reduce_window->window().dimensions(dimension);

        if (!window_util::IsTrivialWindowDimension(window_dimension)) {
          return Unimplemented(
              "Dynamic Spatial reduce window is not supported: %s",
              reduce_window->ToString());
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
        HloInstruction* select_and_scatter = hlo;
        const WindowDimension& window_dimension =
            select_and_scatter->window().dimensions(dimension);

        if (!window_util::IsTrivialWindowDimension(window_dimension)) {
          return Unimplemented(
              "Dynamic Spatial select and scatter is not supported: %s",
              select_and_scatter->ToString());
        }

        parent_->SetDynamicSize(select_and_scatter, {}, dimension,
                                dynamic_size);

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
          return Unimplemented(
              "Dynamic dimension propagation on Slice where it doesn't slice "
              "out an entire dimension is not supported %s",
              hlo->ToString());
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
          int64 /*operand_index*/, HloInstruction* dynamic_size) {
        if (hlo->shape().dimensions(dimension) !=
            hlo->operand(0)->shape().dimensions(dimension)) {
          return Unimplemented(
              "Dynamic dimension propagation on DynamicSlice where a partial "
              "dimension is selected %s",
              hlo->ToString());
        }

        parent_->SetDynamicSize(hlo, {}, dimension, dynamic_size);

        return Status::OK();
      });
}

Status DynamicDimensionInferenceVisitor::HandleGather(HloInstruction* hlo) {
  return ForEachOperandDynamicDimension(
      hlo, [&](HloInstruction* /*operand*/, ShapeIndex /*index*/,
               int64 input_dynamic_dimension, int64 operand_index,
               HloInstruction* dynamic_size) {
        if (operand_index != 1) {
          return Unimplemented(
              "Detects a dynamic dimension on the data input of gather, which "
              "is not suported: %s",
              hlo->ToString());
        }
        // A mapping from output to input batch dim number. -1 means not a batch
        // dimension.
        int64 indices_rank = hlo->operand(1)->shape().rank();
        int64 output_rank = hlo->shape().rank();
        const GatherDimensionNumbers& gather_dims =
            hlo->gather_dimension_numbers();
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

Status DynamicDimensionInferenceVisitor::HandleScatter(HloInstruction* hlo) {
  return ForEachOperandDynamicDimension(
      hlo,
      [&](HloInstruction* /*operand*/, ShapeIndex /*index*/, int64 dimension,
          int64 operand_index, HloInstruction* operand_dynamic_size) {
        if (operand_index == 0) {
          return Unimplemented(
              "Detects a dynamic dimension on the data input of scatter, which "
              "is not suported: %s",
              hlo->ToString());
        }

        const ScatterDimensionNumbers& scatter_dims =
            hlo->scatter_dimension_numbers();
        if (operand_index == 1) {
          parent_->SetDynamicSize(hlo, {}, dimension, operand_dynamic_size);
          return Status::OK();
        }

        if (operand_index == 2 &&
            absl::c_linear_search(scatter_dims.update_window_dims(),
                                  dimension)) {
          return Unimplemented(
              "Dynamic dimension of update window dims is not supported "
              "is not suported: %s",
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
               HloInstruction* dynamic_size) {
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
    parent_->CopyMapping(/*from=*/old_tuple_operand, /*to=*/new_tuple_operand);
    hlo = result.new_while_instr;
  }

  // We have replaced the while loop, now set the dynamic dimensions for the
  // newly created while loop so that the hlos that consumes the while loop can
  // see the dynamic dimensions. Also sets the dynamic parameter binding for
  // running inference in the while loop.
  DynamicParameterBinding binding_for_while;
  TF_RETURN_IF_ERROR(ForEachOperandDynamicDimension(
      hlo, [&](HloInstruction*, ShapeIndex index, int64 dimension,
               int64 operand_index, HloInstruction* dynamic_size) {
        DynamicParameterBinding::DynamicParameter dynamic_parameter{
            operand_index,
            {dynamic_size_to_operand_id_index_map[dynamic_size]}};
        DynamicParameterBinding::DynamicDimension dynamic_dimension{
            operand_index, index, dimension};
        TF_RETURN_IF_ERROR(
            binding_for_while.Bind(dynamic_parameter, dynamic_dimension));
        parent_->SetDynamicSize(hlo, index, dimension, dynamic_size);
        return Status::OK();
      }));

  // Run inference in while body and condition.
  TF_RETURN_IF_ERROR(DynamicDimensionInferenceVisitor::Run(
      hlo->while_body(), binding_for_while, parent_));
  TF_RETURN_IF_ERROR(DynamicDimensionInferenceVisitor::Run(
      hlo->while_condition(), binding_for_while, parent_));

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

Status DynamicDimensionInferenceVisitor::ForEachOperandDynamicDimension(
    HloInstruction* inst, const OperandDynamicDimensionFn& fn) {
  for (int64 operand_index = 0; operand_index < inst->operand_count();
       ++operand_index) {
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
                     dynamic_size);
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

HloInstruction* DynamicDimensionInference::GetDynamicSize(
    HloInstruction* inst, const ShapeIndex& index, int64 dim) const {
  auto iter = dynamic_mapping_.find(DynamicDimension{inst, index, dim});
  if (iter != dynamic_mapping_.end()) {
    return iter->second;
  }
  return nullptr;
}

}  // namespace xla

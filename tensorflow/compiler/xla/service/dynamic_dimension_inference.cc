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
#include "tensorflow/compiler/xla/service/dfs_hlo_visitor_with_default.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"

namespace xla {

namespace {
bool IsTrivialWindowDimension(const WindowDimension& window_dimension) {
  return window_dimension.size() == 1 && window_dimension.stride() == 1 &&
         window_dimension.padding_low() == 0 &&
         window_dimension.padding_high() == 0 &&
         window_dimension.window_dilation() == 1 &&
         window_dimension.base_dilation() == 1;
}
}  // namespace

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

Status DynamicDimensionInferenceVisitor::HandleBroadcast(HloInstruction* hlo) {
  return ForEachOperandDynamicDimension(
      hlo, [&](HloInstruction* operand, ShapeIndex index, int64 dimension,
               int64 operand_index, HloInstruction* dynamic_size) {
        int64 broadcast_dim = hlo->dimensions(dimension);
        parent_->SetDynamicSize(hlo, index, broadcast_dim, dynamic_size);
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
        for (int i = 0; i < ShapeUtil::Rank(operand->shape()); ++i) {
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

        for (int64 i = 0; i < ShapeUtil::Rank(dot->operand(0)->shape()); i++) {
          if (!absl::c_linear_search(
                  dimension_numbers.lhs_contracting_dimensions(), i)) {
            if (operand_index == 0) {
              result_dim_mapping[i] = current_result_dims;
            }
            current_result_dims++;
          }
        }

        for (int64 i = 0; i < ShapeUtil::Rank(dot->operand(1)->shape()); i++) {
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

        if (!IsTrivialWindowDimension(window_dimension)) {
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

        if (!IsTrivialWindowDimension(window_dimension)) {
          return Unimplemented(
              "Dynamic Spatial select and scatter is not supported: %s",
              select_and_scatter->ToString());
        }

        parent_->SetDynamicSize(select_and_scatter, {}, dimension,
                                dynamic_size);

        return Status::OK();
      });
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

/* static */
StatusOr<DynamicDimensionInference> DynamicDimensionInference::Run(
    HloModule* module) {
  VLOG(0) << "Param Config " << module->dynamic_parameter_binding().ToString();
  DynamicDimensionInference inference(module);
  TF_RETURN_IF_ERROR(inference.AnalyzeDynamicDimensions());
  return inference;
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

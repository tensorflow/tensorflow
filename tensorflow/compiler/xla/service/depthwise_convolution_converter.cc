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

#include "tensorflow/compiler/xla/service/depthwise_convolution_converter.h"

#include <memory>
#include <vector>

#include "absl/memory/memory.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/service/dfs_hlo_visitor_with_default.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/logging.h"

namespace xla {

namespace {

class ConvolutionVisitor : public DfsHloVisitorWithDefault {
 public:
  // Default visitor action is to do nothing and return OK.
  Status DefaultAction(HloInstruction* /*hlo_instruction*/) override {
    return Status::OK();
  }

  Status HandleConvolution(HloInstruction* convolution) override;

  Status HandleBackwardFilterBatchGroupConvolution(HloInstruction* convolution);

  // Runs the visitor on a computation.
  static bool Run(HloComputation* computation,
                  std::function<bool(HloInstruction*)> is_cost_viable);

  // Returns whether any convolution ops were rewritten.
  const bool changed() const { return changed_; }

  ~ConvolutionVisitor() override = default;

 private:
  explicit ConvolutionVisitor(
      HloComputation* computation,
      std::function<bool(HloInstruction*)> is_cost_viable)
      : computation_(computation), is_cost_viable_(is_cost_viable) {}

  // Current HloComputation instance the ConvolutionVisitor is traversing.
  HloComputation* computation_;

  // Whether rewrite has occurred.
  bool changed_ = false;

  std::function<bool(HloInstruction*)> is_cost_viable_;
};

bool ConvolutionVisitor::Run(
    HloComputation* computation,
    std::function<bool(HloInstruction*)> is_cost_viable) {
  ConvolutionVisitor visitor(computation, is_cost_viable);
  TF_CHECK_OK(computation->Accept(&visitor));
  return visitor.changed_;
}

namespace {
Shape SwapInputOutputFeatureDims(const Shape& shape, int64 input_feature_dim,
                                 int64 output_feature_dim) {
  int64 num_dims = shape.dimensions_size();
  CHECK_GE(num_dims, 2);
  Shape transformed_shape = shape;
  transformed_shape.set_dimensions(input_feature_dim,
                                   shape.dimensions(output_feature_dim));
  transformed_shape.set_dimensions(output_feature_dim,
                                   shape.dimensions(input_feature_dim));
  return transformed_shape;
}
}  // namespace

// This function handles batch_group_counts which are relevant only for
// depthwise backprop filter convolutions.
Status ConvolutionVisitor::HandleBackwardFilterBatchGroupConvolution(
    HloInstruction* convolution) {
  auto dim_numbers = convolution->convolution_dimension_numbers();
  auto lhs = convolution->mutable_operand(0);
  auto rhs = convolution->mutable_operand(1);
  int64 num_groups = convolution->batch_group_count();
  int64 input_batch_dimension = dim_numbers.input_batch_dimension();
  int64 input_batch = lhs->shape().dimensions(input_batch_dimension);

  // TODO(b/139748189): Support 'num_grous' > 1 when input_batch !=
  // num_groups.
  if (num_groups == 1 || input_batch != num_groups) {
    return Status::OK();
  }

  VLOG(2) << "Dealing with batch_group_count " << num_groups
          << " for convolution " << convolution->ToString() << "\n";

  int64 output_batch_dimension = dim_numbers.output_batch_dimension();
  int64 output_feature_dimension = dim_numbers.output_feature_dimension();

  // When mapping depthwise conv backward filter to batch grouped convolution,
  // tf2xla bridge needs to swap the output batch and feature dimension. Since
  // we want to use grouped convolution APIs, this swap needs to be reverted.
  dim_numbers.set_output_batch_dimension(output_feature_dimension);
  dim_numbers.set_output_feature_dimension(output_batch_dimension);

  if (!is_cost_viable_(convolution)) {
    Shape transformed_filter_grad_shape = SwapInputOutputFeatureDims(
        convolution->shape(), dim_numbers.output_batch_dimension(),
        dim_numbers.output_feature_dimension());

    int64 input_feature_dimension = dim_numbers.input_feature_dimension();
    int64 input_feature = lhs->shape().dimensions(input_feature_dimension);

    auto add = [&](std::unique_ptr<HloInstruction> inst) {
      return computation_->AddInstruction(std::move(inst));
    };
    // Reshape batch_dim C -> [G, C/G] - Batch and feature dims have been
    // swapped in tf2xla bridge
    std::vector<int64> reshape_dims = SpanToVector(lhs->shape().dimensions());
    reshape_dims[input_batch_dimension] =
        reshape_dims[input_batch_dimension] / num_groups;
    reshape_dims.insert(reshape_dims.begin() + input_batch_dimension,
                        num_groups);
    lhs = add(HloInstruction::CreateReshape(
        ShapeUtil::MakeShape(lhs->shape().element_type(), reshape_dims), lhs));

    // Transpose G to the axis before N, For eg: [G, C/G, H, W, N ] -> [C/G, H,
    // W, G, N]
    std::vector<int64> transpose_dims(lhs->shape().dimensions_size());
    std::iota(transpose_dims.begin(), transpose_dims.end(), 0);
    transpose_dims.erase(transpose_dims.begin() + input_batch_dimension);
    transpose_dims.insert(transpose_dims.begin() + input_feature_dimension,
                          input_batch_dimension);
    std::vector<int64> transpose_reshape_dims =
        ComposePermutations(lhs->shape().dimensions(), transpose_dims);
    lhs = add(HloInstruction::CreateTranspose(
        ShapeUtil::MakeShape(lhs->shape().element_type(),
                             transpose_reshape_dims),
        lhs, transpose_dims));

    // Merge [G,N] -> [N*G]
    Shape new_shape = lhs->shape();
    new_shape.DeleteDimension(input_feature_dimension);
    new_shape.set_dimensions(input_feature_dimension,
                             input_feature * num_groups);
    lhs = add(HloInstruction::CreateReshape(new_shape, lhs));

    std::vector<HloInstruction*> new_operands = {lhs, rhs};
    auto new_conv = convolution->CloneWithNewOperands(
        transformed_filter_grad_shape, new_operands);
    new_conv->set_feature_group_count(num_groups);
    new_conv->set_batch_group_count(1);
    new_conv->set_convolution_dimension_numbers(dim_numbers);
    auto new_convolution = computation_->AddInstruction(std::move(new_conv));

    // Another reshape is required since the filter grad shape as a result of
    // the 'new convolution` will be [kh, kw, C_i/G = 1, C_o = C_i = G ] but the
    // expected shape is [kh, kw, C_i = G, DM=1] assuming the Depth-Multiplier
    // (DM) is 1 and number of input features = G as required by the depthwise
    // conv semantics
    auto reshape =
        HloInstruction::CreateReshape(convolution->shape(), new_convolution);
    TF_RETURN_IF_ERROR(computation_->ReplaceWithNewInstruction(
        convolution, std::move(reshape)));
    changed_ = true;
  }

  return Status::OK();
}

Status ConvolutionVisitor::HandleConvolution(HloInstruction* convolution) {
  return HandleBackwardFilterBatchGroupConvolution(convolution);
}

}  // namespace

StatusOr<bool> DepthwiseConvolutionConverter::Run(HloModule* module) {
  XLA_VLOG_LINES(2, "DepthwiseConvolutionConverter::Run(), before:\n" +
                        module->ToString());
  bool changed = false;
  for (auto* comp : module->MakeNonfusionComputations()) {
    if (ConvolutionVisitor::Run(comp, is_cost_viable_)) {
      changed = true;
    }
  }
  XLA_VLOG_LINES(
      2, "DepthwiseConvolutionConverter::Run(), after:\n" + module->ToString());
  return changed;
}

}  // namespace xla

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

#include "tensorflow/compiler/xla/service/convolution_feature_group_converter.h"

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

// ConvolutionVisitor traverses the HLO computation and rewrites Convolution
// operations with feature_group_count > 1 into convolutions with
// feature_group_count = 1.
class ConvolutionVisitor : public DfsHloVisitorWithDefault {
 public:
  // Default visitor action is to do nothing and return OK.
  Status DefaultAction(HloInstruction* /*hlo_instruction*/) override {
    return Status::OK();
  }

  Status HandleConvolution(HloInstruction* convolution) override;

  // Runs the visitor on a computation.
  static bool Run(HloComputation* computation,
                  bool canonicalize_depthwise_filter);

  // Returns whether any convolution ops were rewritten.
  const bool changed() const { return changed_; }

  ~ConvolutionVisitor() override = default;

 private:
  explicit ConvolutionVisitor(HloComputation* computation,
                              bool canonicalize_depthwise_filter = false)
      : computation_(computation),
        filter_expansion_(!canonicalize_depthwise_filter) {}

  // Current HloComputation instance the ConvolutionVisitor is traversing.
  HloComputation* computation_;

  // Whether rewrite has occurred.
  bool changed_ = false;

  // Whether filter expansion is required.
  bool filter_expansion_;
};

bool ConvolutionVisitor::Run(HloComputation* computation,
                             bool canonicalize_depthwise_filter) {
  ConvolutionVisitor visitor(computation, canonicalize_depthwise_filter);
  TF_CHECK_OK(computation->Accept(&visitor));
  return visitor.changed_;
}

Shape ExpandedFilterShape(const Shape& shape, int64 group_count,
                          int64 input_feature_dim) {
  int64 num_dims = shape.dimensions_size();
  CHECK_GE(num_dims, 2);
  Shape expanded_shape = shape;
  expanded_shape.set_dimensions(
      input_feature_dim, shape.dimensions(input_feature_dim) * group_count);
  return expanded_shape;
}

// Returns a vector with 'group_count' many groups, where the i-th group
// consists of 'group_size' times the value i.
std::vector<int32> GetMaskIds(int64 group_size, int64 group_count) {
  std::vector<int32> values;
  for (int i = 0; i < group_count; ++i) {
    for (int j = 0; j < group_size; ++j) {
      values.push_back(i);
    }
  }
  return values;
}

// Create a mask for grouped convolution that will make a normal convolution
// produce the same results as a grouped convolution. For a [2, 1, 6]
// filter this returns a [2, 3, 6] mask
//   1 1 0 0 0 0
//   0 0 1 1 0 0
//   0 0 0 0 1 1
//
//   1 1 0 0 0 0
//   0 0 1 1 0 0
//   0 0 0 0 1 1
//
// The first step is to create a rank 1 constant:
//   0 1 2
//
// This is broadcasted to
//   0 0 0 0 0 0
//   1 1 1 1 1 1
//   2 2 2 2 2 2
//
//   0 0 0 0 0 0
//   1 1 1 1 1 1
//   2 2 2 2 2 2
//
// Then we create another rank 1 constant
//   0 0 1 1 2 2
//
// This is broadcasted to
//   0 0 1 1 2 2
//   0 0 1 1 2 2
//   0 0 1 1 2 2
//
//   0 0 1 1 2 2
//   0 0 1 1 2 2
//   0 0 1 1 2 2
//
// Finally we use the Eq op of these two broadcasted constants and get the
// desired mask.
HloInstruction* GetExpandedFilterMask(
    const Shape& filter_shape, int64 input_feature_dim,
    int64 output_feature_dim, int64 group_count,
    const std::function<HloInstruction*(std::unique_ptr<HloInstruction>)>&
        add_instruction) {
  Shape expanded_filter_shape =
      ExpandedFilterShape(filter_shape, group_count, input_feature_dim);
  Shape mask_shape = ShapeUtil::MakeShape(
      S32, AsInt64Slice(expanded_filter_shape.dimensions()));
  int64 output_feature = filter_shape.dimensions(output_feature_dim);
  int64 group_size = filter_shape.dimensions(input_feature_dim);

  // Create a 'input_feature' sized linspace and 'output_feature' sized linspace
  // that will be broadcasted into perpendicular dimensions and compared.
  const std::vector<int32> input_feature_filter_mask =
      GetMaskIds(group_size, group_count);
  const std::vector<int32> output_feature_filter_mask =
      GetMaskIds(output_feature / group_count, group_count);

  auto mask1 = add_instruction(HloInstruction::CreateConstant(
      LiteralUtil::CreateR1<int32>(input_feature_filter_mask)));
  auto broadcasted_mask1 = add_instruction(
      HloInstruction::CreateBroadcast(mask_shape, mask1, {input_feature_dim}));
  auto mask2 = add_instruction(HloInstruction::CreateConstant(
      LiteralUtil::CreateR1<int32>(output_feature_filter_mask)));
  auto broadcasted_mask2 = add_instruction(
      HloInstruction::CreateBroadcast(mask_shape, mask2, {output_feature_dim}));

  // Compare the broadcasted output feature linspace to the input feature
  // linspace to create a diagonal predicate.
  Shape predicate_shape = ShapeUtil::MakeShape(
      PRED, AsInt64Slice(expanded_filter_shape.dimensions()));
  return add_instruction(HloInstruction::CreateBinary(
      predicate_shape, HloOpcode::kEq, broadcasted_mask1, broadcasted_mask2));
}

Status ConvolutionVisitor::HandleConvolution(HloInstruction* convolution) {
  int64 group_count = convolution->feature_group_count();
  if (group_count == 1) {
    return Status::OK();
  }
  auto filter = convolution->mutable_operand(1);
  changed_ = true;
  auto add = [&](std::unique_ptr<HloInstruction> inst) {
    return computation_->AddInstruction(std::move(inst));
  };

  auto dim_numbers = convolution->convolution_dimension_numbers();
  int64 input_feature_dim = dim_numbers.kernel_input_feature_dimension();
  int64 group_size = filter->shape().dimensions(input_feature_dim);
  int64 output_feature_dim = dim_numbers.kernel_output_feature_dimension();
  auto expanded_filter_shape =
      ExpandedFilterShape(filter->shape(), group_count, input_feature_dim);
  HloInstruction* filter_mask = GetExpandedFilterMask(
      filter->shape(), input_feature_dim, output_feature_dim, group_count, add);
  HloInstruction* expanded_filter;

  if (group_size == 1) {
    bool depthwise_separable =
        (group_count == filter->shape().dimensions(output_feature_dim));
    // If the code generator handles depthwise separable convolutions
    // inherently, then no filter expansion is needed.
    if (!filter_expansion_ && depthwise_separable) {
      const int64 old_kernel_input_feature_dimension =
          dim_numbers.kernel_input_feature_dimension();
      const int64 old_kernel_output_feature_dimension =
          dim_numbers.kernel_output_feature_dimension();

      // For depthwise convolutions, we want the kernel input feature dimension
      // to be smaller than the output feature dimension. If that's not the
      // case, we swap the dimensions.
      if (old_kernel_input_feature_dimension >
          old_kernel_output_feature_dimension) {
        Shape reshaped_filter_shape = filter->shape();
        auto& dimensions = *reshaped_filter_shape.mutable_dimensions();
        std::swap(dimensions[old_kernel_input_feature_dimension],
                  dimensions[old_kernel_output_feature_dimension]);

        auto reshaped_filter =
            add(HloInstruction::CreateReshape(reshaped_filter_shape, filter));

        dim_numbers.set_kernel_input_feature_dimension(
            old_kernel_output_feature_dimension);

        dim_numbers.set_kernel_output_feature_dimension(
            old_kernel_input_feature_dimension);

        auto new_convolution = HloInstruction::CreateConvolve(
            convolution->shape(), convolution->mutable_operand(0),
            reshaped_filter, group_count, convolution->window(), dim_numbers,
            convolution->precision_config());

        TF_RETURN_IF_ERROR(computation_->ReplaceWithNewInstruction(
            convolution, std::move(new_convolution)));
      }
      return Status::OK();
    }
    // We want to repeat 'filter' in the 'input_feature_dim' dimension
    // 'group_count' times.
    Shape reshaped_filter_shape =
        ShapeUtil::DeleteDimension(input_feature_dim, filter->shape());
    auto reshaped_filter =
        add(HloInstruction::CreateReshape(reshaped_filter_shape, filter));
    std::vector<int64> broadcast_dims;
    for (int64 i = 0; i < filter->shape().dimensions_size(); ++i) {
      if (i == input_feature_dim) {
        continue;
      }
      broadcast_dims.push_back(i);
    }
    expanded_filter = add(HloInstruction::CreateBroadcast(
        expanded_filter_shape, reshaped_filter, broadcast_dims));
  } else {
    // We could possibly also use reshape, broadcast, reshape instead of concat
    // here, but it would require more complex code, and for depthwise
    // convolution we would never end up in this branch.
    std::vector<HloInstruction*> concat_operands(group_count, filter);
    expanded_filter = add(HloInstruction::CreateConcatenate(
        expanded_filter_shape, concat_operands, input_feature_dim));
  }
  auto zero = add(HloInstruction::CreateConstant(
      LiteralUtil::Zero(expanded_filter_shape.element_type())));
  auto zero_filter =
      add(HloInstruction::CreateBroadcast(expanded_filter_shape, zero, {}));
  auto new_filter = add(
      HloInstruction::CreateTernary(expanded_filter_shape, HloOpcode::kSelect,
                                    filter_mask, expanded_filter, zero_filter));
  auto new_convolution = HloInstruction::CreateConvolve(
      convolution->shape(), convolution->mutable_operand(0), new_filter,
      /*feature_group_count=*/1, convolution->window(), dim_numbers,
      convolution->precision_config());
  TF_RETURN_IF_ERROR(computation_->ReplaceWithNewInstruction(
      convolution, std::move(new_convolution)));
  return Status::OK();
}

}  // namespace

StatusOr<bool> ConvolutionFeatureGroupConverter::Run(HloModule* module) {
  XLA_VLOG_LINES(2, "ConvolutionFeatureGroupConverter::Run(), before:\n" +
                        module->ToString());
  bool changed = false;
  for (auto* comp : module->MakeNonfusionComputations()) {
    if (ConvolutionVisitor::Run(comp, filter_expansion_)) {
      changed = true;
    }
  }
  XLA_VLOG_LINES(2, "ConvolutionFeatureGroupConverter::Run(), after:\n" +
                        module->ToString());
  return changed;
}

}  // namespace xla

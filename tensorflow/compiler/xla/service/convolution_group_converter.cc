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

#include "tensorflow/compiler/xla/service/convolution_group_converter.h"

#include <algorithm>
#include <memory>
#include <vector>

#include "absl/memory/memory.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/service/dfs_hlo_visitor_with_default.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_creation_utils.h"
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

  Status HandleBatchGroupCount(HloInstruction* convolution);

  // Runs the visitor on a computation.
  static bool Run(HloComputation* computation,
                  std::function<bool(HloInstruction*)> is_cost_viable,
                  bool convert_batch_groups_only, bool filter_expansion);

  // Returns whether any convolution ops were rewritten.
  const bool changed() const { return changed_; }

  ~ConvolutionVisitor() override = default;

 private:
  explicit ConvolutionVisitor(
      HloComputation* computation,
      std::function<bool(HloInstruction*)> is_cost_viable,
      bool convert_batch_groups_only, bool filter_expansion)
      : computation_(computation),
        filter_expansion_(filter_expansion),
        convert_batch_groups_only_(convert_batch_groups_only),
        is_cost_viable_(is_cost_viable) {}

  // Current HloComputation instance the ConvolutionVisitor is traversing.
  HloComputation* computation_;

  // Whether rewrite has occurred.
  bool changed_ = false;

  // Whether filter expansion is required.
  bool filter_expansion_;

  // Decides whether to convert batch groups or feature groups.
  bool convert_batch_groups_only_;

  // std::function<std::vector<LloValue*>(int64, int64)> chunk_fetcher
  std::function<bool(HloInstruction*)> is_cost_viable_;
};

bool ConvolutionVisitor::Run(
    HloComputation* computation,
    std::function<bool(HloInstruction*)> is_cost_viable,
    bool convert_batch_groups_only, bool filter_expansion) {
  ConvolutionVisitor visitor(computation, is_cost_viable,
                             convert_batch_groups_only, filter_expansion);
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
    const Shape& filter_shape, int64 kernel_input_feature_dim,
    int64 kernel_output_feature_dim, int64 group_count,
    const std::function<HloInstruction*(std::unique_ptr<HloInstruction>)>&
        add_instruction) {
  Shape expanded_filter_shape =
      ExpandedFilterShape(filter_shape, group_count, kernel_input_feature_dim);
  Shape mask_shape = ShapeUtil::MakeShape(
      S32, AsInt64Slice(expanded_filter_shape.dimensions()));
  int64 output_feature = filter_shape.dimensions(kernel_output_feature_dim);
  int64 group_size = filter_shape.dimensions(kernel_input_feature_dim);

  // Create a 'input_feature' sized linspace and 'output_feature' sized linspace
  // that will be broadcasted into perpendicular dimensions and compared.
  const std::vector<int32> input_feature_filter_mask =
      GetMaskIds(group_size, group_count);
  const std::vector<int32> output_feature_filter_mask =
      GetMaskIds(output_feature / group_count, group_count);
  auto mask1 = add_instruction(HloInstruction::CreateConstant(
      LiteralUtil::CreateR1<int32>(input_feature_filter_mask)));
  auto broadcasted_mask1 = add_instruction(HloInstruction::CreateBroadcast(
      mask_shape, mask1, {kernel_input_feature_dim}));
  auto mask2 = add_instruction(HloInstruction::CreateConstant(
      LiteralUtil::CreateR1<int32>(output_feature_filter_mask)));
  auto broadcasted_mask2 = add_instruction(HloInstruction::CreateBroadcast(
      mask_shape, mask2, {kernel_output_feature_dim}));

  // Compare the broadcasted output feature linspace to the input feature
  // linspace to create a diagonal predicate.
  Shape predicate_shape = ShapeUtil::MakeShape(
      PRED, AsInt64Slice(expanded_filter_shape.dimensions()));
  return add_instruction(HloInstruction::CreateCompare(
      predicate_shape, broadcasted_mask1, broadcasted_mask2,
      ComparisonDirection::kEq));
}

// This function handles batch_group_counts which are relevant only for
// depthwise backprop filter convolutions.
Status ConvolutionVisitor::HandleBatchGroupCount(HloInstruction* convolution) {
  auto dim_numbers = convolution->convolution_dimension_numbers();
  auto activation = convolution->mutable_operand(0);
  auto filter = convolution->mutable_operand(1);
  int64 batch_group_count = convolution->batch_group_count();

  if (batch_group_count == 1) {
    return Status::OK();
  }

  VLOG(2) << "Dealing with batch_group_count " << batch_group_count
          << " for convolution " << convolution->ToString() << "\n";

  auto add = [&](std::unique_ptr<HloInstruction> inst) {
    return computation_->AddInstruction(std::move(inst));
  };

  int64 input_batch_dimension = dim_numbers.input_batch_dimension();
  const int64 input_feature_dimension = dim_numbers.input_feature_dimension();

  int64 output_batch_dimension = dim_numbers.output_batch_dimension();
  int64 output_feature_dimension = dim_numbers.output_feature_dimension();

  const int64 kernel_input_feature_dimension =
      dim_numbers.kernel_input_feature_dimension();
  const int64 kernel_output_feature_dimension =
      dim_numbers.kernel_output_feature_dimension();

  const int64 input_batch =
      activation->shape().dimensions(input_batch_dimension);
  const int64 output_feature =
      filter->shape().dimensions(kernel_output_feature_dimension);

  if (output_feature != batch_group_count || input_batch != batch_group_count) {
    // Insert a spatial dimension to the activation before the input batch
    // dimension to represent the batch group.
    std::vector<int64> input_sizes(activation->shape().dimensions().begin(),
                                   activation->shape().dimensions().end());
    input_sizes[input_batch_dimension] /= batch_group_count;
    input_sizes.insert(input_sizes.begin() + input_batch_dimension,
                       batch_group_count);
    activation = MakeReshapeHlo(input_sizes, activation).ValueOrDie();
    for (auto& d : *dim_numbers.mutable_input_spatial_dimensions()) {
      if (d > input_batch_dimension) {
        ++d;
      }
    }
    dim_numbers.add_input_spatial_dimensions(input_batch_dimension);
    dim_numbers.set_input_batch_dimension(input_batch_dimension + 1);
    if (input_feature_dimension > input_batch_dimension) {
      dim_numbers.set_input_feature_dimension(input_feature_dimension + 1);
    }

    // Insert a spatial dimension to the kernel before the output feature
    // dimension to represent the batch group.
    std::vector<int64> kernel_sizes(filter->shape().dimensions().begin(),
                                    filter->shape().dimensions().end());
    kernel_sizes[kernel_output_feature_dimension] /= batch_group_count;
    kernel_sizes.insert(kernel_sizes.begin() + kernel_output_feature_dimension,
                        batch_group_count);
    filter = MakeReshapeHlo(kernel_sizes, filter).ValueOrDie();
    for (auto& d : *dim_numbers.mutable_kernel_spatial_dimensions()) {
      if (d > kernel_output_feature_dimension) {
        ++d;
      }
    }
    dim_numbers.add_kernel_spatial_dimensions(kernel_output_feature_dimension);
    dim_numbers.set_kernel_output_feature_dimension(
        kernel_output_feature_dimension + 1);
    if (kernel_input_feature_dimension > kernel_output_feature_dimension) {
      dim_numbers.set_kernel_input_feature_dimension(
          kernel_input_feature_dimension + 1);
    }

    // Insert a spatial dimension to the output before the output feature
    // dimension to represent the batch group.
    for (auto& d : *dim_numbers.mutable_output_spatial_dimensions()) {
      if (d > output_feature_dimension) {
        ++d;
      }
    }
    dim_numbers.add_output_spatial_dimensions(output_feature_dimension);
    dim_numbers.set_output_feature_dimension(output_feature_dimension + 1);
    if (output_batch_dimension > output_feature_dimension) {
      dim_numbers.set_output_batch_dimension(output_batch_dimension + 1);
    }

    // To represent a batch group count of 3 you can slide a 3 wide window
    // [X Y Z]
    // across [A 0 0 B 0 0 C] with stride 2 to produce
    // [AX+0Y+0Z 0X+BY+0Z 0X+0Y+CZ] -> [AX BY CZ] which will behave the same as
    // a batch group count.
    Window window = convolution->window();
    auto window_dim = window.add_dimensions();
    window_dim->set_base_dilation(batch_group_count);
    window_dim->set_size(batch_group_count);
    window_dim->set_stride(batch_group_count - 1);
    window_dim->set_padding_low(0);
    window_dim->set_padding_high(0);
    window_dim->set_window_reversal(false);
    window_dim->set_window_dilation(1);
    HloInstruction* new_convolution =
        MakeConvolveHlo(activation, filter, convolution->feature_group_count(),
                        window, dim_numbers, convolution->precision_config())
            .ValueOrDie();
    convolution->SetupDerivedInstruction(new_convolution);
    TF_CHECK_OK(computation_->ReplaceInstruction(
        convolution,
        MakeReshapeHlo(convolution->shape(), new_convolution).ValueOrDie()));
    changed_ = true;
    return Status::OK();
  }

  VLOG(2) << "is_cost_viable_ " << is_cost_viable_(convolution);
  const bool cost_too_high = !is_cost_viable_(convolution);
  if (cost_too_high || filter_expansion_) {
    // We first obtain the expanded the filter (which is the convolution
    // output). The batch dimension is the expanded one (which originally
    // represents kernel input feature dimension). We mask the filter to zero
    // out the expanded regions. Next we reduce the filter in the batch
    // dimension to obtain the original filter size.

    HloInstruction* filter_mask =
        GetExpandedFilterMask(convolution->shape(), output_batch_dimension,
                              output_feature_dimension, batch_group_count, add);
    auto expanded_filter_shape = ExpandedFilterShape(
        convolution->shape(), batch_group_count, output_batch_dimension);

    VLOG(2) << "output_batch_dimension " << output_batch_dimension;
    VLOG(2) << "New output shape of convolution "
            << expanded_filter_shape.ToString();

    auto new_convolution = add(HloInstruction::CreateConvolve(
        expanded_filter_shape, activation, filter,
        /*feature_group_count=*/1, /*batch_group_count=*/1,
        convolution->window(), dim_numbers, convolution->precision_config()));

    VLOG(2) << "Expanded convolution " << new_convolution->ToString();

    auto zero = add(HloInstruction::CreateConstant(
        LiteralUtil::Zero(expanded_filter_shape.element_type())));
    auto zero_filter =
        add(HloInstruction::CreateBroadcast(expanded_filter_shape, zero, {}));

    auto new_filter = add(HloInstruction::CreateTernary(
        expanded_filter_shape, HloOpcode::kSelect, filter_mask, new_convolution,
        zero_filter));

    PrimitiveType reduce_type = new_filter->shape().element_type();
    auto reduce_window_shape = new_convolution->shape();
    reduce_window_shape.set_dimensions(output_batch_dimension, 1);

    // Ensure that data input to reduce window uses at least 32 bits.
    if (primitive_util::BitWidth(reduce_type) < primitive_util::BitWidth(F32)) {
      reduce_type = F32;
      reduce_window_shape.set_element_type(F32);
      Shape convert_shape = new_filter->shape();
      convert_shape.set_element_type(F32);
      new_filter =
          add(HloInstruction::CreateConvert(convert_shape, new_filter));
    }

    auto zero_literal = LiteralUtil::Zero(reduce_type);
    auto zero_scalar =
        add(HloInstruction::CreateConstant(std::move(zero_literal)));

    auto reduce_function = [&]() -> HloComputation* {
      HloComputation::Builder b("add_computation");
      Shape shape = ShapeUtil::MakeShape(reduce_type, {});
      auto lhs =
          b.AddInstruction(HloInstruction::CreateParameter(0, shape, "lhs"));
      auto rhs =
          b.AddInstruction(HloInstruction::CreateParameter(1, shape, "rhs"));
      auto scalar_op = b.AddInstruction(
          HloInstruction::CreateBinary(shape, HloOpcode::kAdd, lhs, rhs));
      return computation_->parent()->AddEmbeddedComputation(b.Build(scalar_op));
    };

    // Create the reduce window.
    Window window;
    for (int64 i = 0; i < new_convolution->shape().dimensions_size(); ++i) {
      auto* dim = window.add_dimensions();
      dim->set_padding_low(0);
      dim->set_padding_high(0);
      dim->set_window_dilation(1);
      dim->set_base_dilation(1);
      if (i == output_batch_dimension) {
        dim->set_stride(batch_group_count);
        dim->set_size(batch_group_count);
      } else {
        dim->set_stride(1);
        dim->set_size(1);
      }
    }
    auto reduce_window = add(HloInstruction::CreateReduceWindow(
        reduce_window_shape, new_filter, zero_scalar, window,
        reduce_function()));

    Shape convert_back_shape = reduce_window->shape();
    convert_back_shape.set_element_type(activation->shape().element_type());

    // Convert reduced data back to the original data type.
    auto reduce_window_converted =
        HloInstruction::CreateConvert(convert_back_shape, reduce_window);

    TF_CHECK_OK(computation_->ReplaceWithNewInstruction(
        convolution, std::move(reduce_window_converted)));
    changed_ = true;
  }

  return Status::OK();
}

Status ConvolutionVisitor::HandleConvolution(HloInstruction* convolution) {
  if (convert_batch_groups_only_) {
    return HandleBatchGroupCount(convolution);
  }

  auto add = [&](std::unique_ptr<HloInstruction> inst) {
    return computation_->AddInstruction(std::move(inst));
  };

  int64 group_count = convolution->feature_group_count();
  if (group_count == 1) {
    return Status::OK();
  }

  changed_ = true;
  ConvolutionDimensionNumbers dim_numbers =
      convolution->convolution_dimension_numbers();
  auto filter = convolution->mutable_operand(1);
  int64 kernel_input_feature_dim = dim_numbers.kernel_input_feature_dimension();
  int64 group_size = filter->shape().dimensions(kernel_input_feature_dim);
  int64 kernel_output_feature_dim =
      dim_numbers.kernel_output_feature_dimension();
  auto expanded_filter_shape = ExpandedFilterShape(filter->shape(), group_count,
                                                   kernel_input_feature_dim);
  HloInstruction* filter_mask =
      GetExpandedFilterMask(filter->shape(), kernel_input_feature_dim,
                            kernel_output_feature_dim, group_count, add);
  HloInstruction* expanded_filter;

  if (group_size == 1) {
    bool depthwise_separable =
        (group_count == filter->shape().dimensions(kernel_output_feature_dim));
    // If the code generator handles depthwise separable convolutions
    // inherently, then no filter expansion is needed.
    if (!filter_expansion_ && depthwise_separable) {
      changed_ = false;
      return Status::OK();
    }
    VLOG(2) << "is_cost_viable_ " << is_cost_viable_(convolution);
    // We want to repeat 'filter' in the 'input_feature_dim' dimension
    // 'group_count' times.
    if (!is_cost_viable_(convolution) || filter_expansion_) {
      Shape reshaped_filter_shape =
          ShapeUtil::DeleteDimension(kernel_input_feature_dim, filter->shape());
      auto reshaped_filter =
          add(HloInstruction::CreateReshape(reshaped_filter_shape, filter));
      std::vector<int64> broadcast_dims;
      for (int64 i = 0; i < filter->shape().dimensions_size(); ++i) {
        if (i == kernel_input_feature_dim) {
          continue;
        }
        broadcast_dims.push_back(i);
      }
      expanded_filter = add(HloInstruction::CreateBroadcast(
          expanded_filter_shape, reshaped_filter, broadcast_dims));

      auto zero = add(HloInstruction::CreateConstant(
          LiteralUtil::Zero(expanded_filter_shape.element_type())));
      auto zero_filter =
          add(HloInstruction::CreateBroadcast(expanded_filter_shape, zero, {}));
      auto new_filter = add(HloInstruction::CreateTernary(
          expanded_filter_shape, HloOpcode::kSelect, filter_mask,
          expanded_filter, zero_filter));

      auto new_convolution = HloInstruction::CreateConvolve(
          convolution->shape(), convolution->mutable_operand(0), new_filter,
          /*feature_group_count=*/1, /*batch_group_count=*/1,
          convolution->window(), dim_numbers, convolution->precision_config());
      return computation_->ReplaceWithNewInstruction(
          convolution, std::move(new_convolution));
    }
    // Add a spatial dimension to emulate a larger output feature dimension
    // to avoid creating a convolution with group_count = 1.
    std::vector<int64> new_filter_dimension;
    new_filter_dimension.reserve(filter->shape().rank() + 1);
    const int64 depthwise_multiplier =
        filter->shape().dimensions(kernel_output_feature_dim) / group_count;
    // Split the kernel output feature dimension into group count and
    // depthwise mutilipler.
    for (int64 i = 0; i < filter->shape().rank(); ++i) {
      if (i == kernel_output_feature_dim) {
        new_filter_dimension.push_back(group_count);
        new_filter_dimension.push_back(depthwise_multiplier);
      } else {
        new_filter_dimension.push_back(filter->shape().dimensions(i));
      }
    }
    if (kernel_input_feature_dim > kernel_output_feature_dim) {
      dim_numbers.set_kernel_input_feature_dimension(kernel_input_feature_dim +
                                                     1);
    }
    for (auto& dim : *dim_numbers.mutable_kernel_spatial_dimensions()) {
      if (dim > kernel_output_feature_dim) {
        ++dim;
      }
    }
    dim_numbers.add_kernel_spatial_dimensions(kernel_output_feature_dim + 1);
    HloInstruction* new_filter =
        computation_->AddInstruction(HloInstruction::CreateReshape(
            ShapeUtil::MakeShape(filter->shape().element_type(),
                                 new_filter_dimension),
            filter));

    auto new_activation_shape = convolution->operand(0)->shape();
    dim_numbers.add_input_spatial_dimensions(new_activation_shape.rank());

    // Create and activations spatial dimension of size 1 with a reversed
    // window and high and low padding equal to the depthwise_multiplier -1.
    // This emulates a larger output feature dimension with an extra spatial
    // dimension.
    ShapeUtil::AppendMajorDimension(1, &new_activation_shape);
    HloInstruction* new_activation =
        computation_->AddInstruction(HloInstruction::CreateReshape(
            new_activation_shape, convolution->mutable_operand(0)));
    auto new_window = convolution->window();
    auto new_dim = new_window.add_dimensions();
    new_dim->set_size(depthwise_multiplier);
    new_dim->set_window_reversal(true);
    new_dim->set_padding_low(depthwise_multiplier - 1);
    new_dim->set_padding_high(depthwise_multiplier - 1);
    new_dim->set_stride(1);
    new_dim->set_window_dilation(1);
    new_dim->set_base_dilation(1);

    // Split the output feature dimension into and output feature of group
    // count and depthwise multipler as an output spatial dimension.
    std::vector<int64> new_output_dimension;
    new_output_dimension.reserve(convolution->shape().rank() + 1);
    for (int64 i = 0; i < convolution->shape().rank(); ++i) {
      if (i == dim_numbers.output_feature_dimension()) {
        new_output_dimension.push_back(group_count);
        new_output_dimension.push_back(depthwise_multiplier);
      } else {
        new_output_dimension.push_back(convolution->shape().dimensions(i));
      }
    }
    if (dim_numbers.output_batch_dimension() >
        dim_numbers.output_feature_dimension()) {
      dim_numbers.set_output_batch_dimension(
          dim_numbers.output_batch_dimension() + 1);
    }
    for (auto& dim : *dim_numbers.mutable_output_spatial_dimensions()) {
      if (dim > dim_numbers.output_feature_dimension()) {
        ++dim;
      }
    }
    dim_numbers.add_output_spatial_dimensions(
        dim_numbers.output_feature_dimension() + 1);
    auto new_convolution_output_shape = ShapeUtil::MakeShape(
        convolution->shape().element_type(), new_output_dimension);
    HloInstruction* new_convolution =
        computation_->AddInstruction(HloInstruction::CreateConvolve(
            new_convolution_output_shape, new_activation, new_filter,
            /*feature_group_count=*/group_count, /*batch_group_count=*/1,
            new_window, dim_numbers, convolution->precision_config()));
    return computation_->ReplaceWithNewInstruction(
        convolution,
        HloInstruction::CreateReshape(convolution->shape(), new_convolution));
  }

  // Implement general grouped convolution using an extra spatial dimension to
  // represent the feature group count.
  //
  // Insert a spatial dimension to the input before the input feature
  // dimension to represent the feature group.
  HloInstruction* activation = convolution->mutable_operand(0);
  std::vector<int64> input_sizes(activation->shape().dimensions().begin(),
                                 activation->shape().dimensions().end());
  const int64 input_feature_dimension = dim_numbers.input_feature_dimension();
  input_sizes[input_feature_dimension] /= group_count;
  input_sizes.insert(input_sizes.begin() + input_feature_dimension,
                     group_count);
  activation = MakeReshapeHlo(input_sizes, activation).ValueOrDie();
  for (auto& d : *dim_numbers.mutable_input_spatial_dimensions()) {
    if (d > input_feature_dimension) {
      ++d;
    }
  }
  dim_numbers.add_input_spatial_dimensions(input_feature_dimension);
  dim_numbers.set_input_feature_dimension(input_feature_dimension + 1);
  if (dim_numbers.input_batch_dimension() > input_feature_dimension) {
    dim_numbers.set_input_batch_dimension(dim_numbers.input_batch_dimension() +
                                          1);
  }

  // Insert a spatial dimension to the kernel before the output feature
  // dimension to represent the feature group.
  std::vector<int64> kernel_sizes(filter->shape().dimensions().begin(),
                                  filter->shape().dimensions().end());
  const int64 kernel_output_feature_dimension =
      dim_numbers.kernel_output_feature_dimension();
  kernel_sizes[kernel_output_feature_dimension] /= group_count;
  kernel_sizes.insert(kernel_sizes.begin() + kernel_output_feature_dimension,
                      group_count);
  filter = MakeReshapeHlo(kernel_sizes, filter).ValueOrDie();
  for (auto& d : *dim_numbers.mutable_kernel_spatial_dimensions()) {
    if (d > kernel_output_feature_dimension) {
      ++d;
    }
  }
  dim_numbers.add_kernel_spatial_dimensions(kernel_output_feature_dimension);
  dim_numbers.set_kernel_output_feature_dimension(
      kernel_output_feature_dimension + 1);
  if (dim_numbers.kernel_input_feature_dimension() >
      kernel_output_feature_dimension) {
    dim_numbers.set_kernel_input_feature_dimension(
        dim_numbers.kernel_input_feature_dimension() + 1);
  }

  // Insert a spatial dimension to the output before the output feature
  // dimension to represent the feature group.
  const int64 output_feature_dimension = dim_numbers.output_feature_dimension();
  for (auto& d : *dim_numbers.mutable_output_spatial_dimensions()) {
    if (d > output_feature_dimension) {
      ++d;
    }
  }
  dim_numbers.add_output_spatial_dimensions(output_feature_dimension);
  dim_numbers.set_output_feature_dimension(output_feature_dimension + 1);
  if (dim_numbers.output_batch_dimension() > output_feature_dimension) {
    dim_numbers.set_output_batch_dimension(
        dim_numbers.output_batch_dimension() + 1);
  }

  // To represent a feature group count of 3 you can  slide a 3 wide window
  // [X Y Z]
  // across [A 0 0 B 0 0 C] with stride 2 to produce
  // [AX+0Y+0Z 0X+BY+0Z 0X+0Y+CZ] -> [AX BY CZ] which will behave the same as
  // a batch group count.
  Window window = convolution->window();
  auto window_dim = window.add_dimensions();
  window_dim->set_base_dilation(group_count);
  window_dim->set_size(group_count);
  window_dim->set_stride(group_count - 1);
  window_dim->set_padding_low(0);
  window_dim->set_padding_high(0);
  window_dim->set_window_reversal(false);
  window_dim->set_window_dilation(1);
  HloInstruction* new_convolution =
      MakeConvolveHlo(activation, filter, 1, window, dim_numbers,
                      convolution->precision_config())
          .ValueOrDie();
  convolution->SetupDerivedInstruction(new_convolution);
  changed_ = true;
  return computation_->ReplaceInstruction(
      convolution,
      MakeReshapeHlo(convolution->shape(), new_convolution).ValueOrDie());
}

}  // namespace

StatusOr<bool> ConvolutionGroupConverter::Run(HloModule* module) {
  XLA_VLOG_LINES(
      2, "ConvolutionGroupConverter::Run(), before:\n" + module->ToString());
  bool changed = false;
  for (auto* comp : module->MakeNonfusionComputations()) {
    if (ConvolutionVisitor::Run(comp, is_cost_viable_,
                                convert_batch_groups_only_,
                                filter_expansion_)) {
      changed = true;
    }
  }
  XLA_VLOG_LINES(
      2, "ConvolutionGroupConverter::Run(), after:\n" + module->ToString());
  return changed;
}

}  // namespace xla

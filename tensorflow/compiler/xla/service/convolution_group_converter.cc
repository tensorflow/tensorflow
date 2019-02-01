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

  Status HandleBatchGroupCount(HloInstruction* convolution);

  // Runs the visitor on a computation.
  static bool Run(HloComputation* computation,
                  std::function<bool(HloInstruction*)> is_cost_viable,
                  bool convert_batch_groups_only,
                  bool canonicalize_depthwise_filter);

  // Returns whether any convolution ops were rewritten.
  const bool changed() const { return changed_; }

  ~ConvolutionVisitor() override = default;

 private:
  explicit ConvolutionVisitor(
      HloComputation* computation,
      std::function<bool(HloInstruction*)> is_cost_viable,
      bool convert_batch_groups_only,
      bool canonicalize_depthwise_filter = false)
      : computation_(computation),
        filter_expansion_(!canonicalize_depthwise_filter),
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
    bool convert_batch_groups_only, bool canonicalize_depthwise_filter) {
  ConvolutionVisitor visitor(computation, is_cost_viable,
                             convert_batch_groups_only,
                             canonicalize_depthwise_filter);
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
  return add_instruction(HloInstruction::CreateBinary(
      predicate_shape, HloOpcode::kEq, broadcasted_mask1, broadcasted_mask2));
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
  int64 input_feature_dimension = dim_numbers.input_feature_dimension();
  int64 output_batch_dimension = dim_numbers.output_batch_dimension();
  int64 output_feature_dimension = dim_numbers.output_feature_dimension();
  int64 kernel_input_feature_dimension =
      dim_numbers.kernel_input_feature_dimension();

  int64 input_batch = activation->shape().dimensions(input_batch_dimension);

  // We are not yet supporting batch_group of sizes greater than 1.
  TF_RET_CHECK(input_batch == batch_group_count);

  if (is_cost_viable_(convolution)) {
    // Add a dimension to the activation, and reshape.
    Shape reshaped_activation_shape = activation->shape();
    ShapeUtil::AppendMajorDimension(1, &reshaped_activation_shape);

    activation = add(
        HloInstruction::CreateReshape(reshaped_activation_shape, activation));

    // Add a dimension to the filter, and reshape.
    Shape reshaped_filter_shape = filter->shape();
    ShapeUtil::AppendMajorDimension(1, &reshaped_filter_shape);

    filter = add(HloInstruction::CreateReshape(reshaped_filter_shape, filter));

    int64 new_spatial_dim = reshaped_activation_shape.dimensions().size() - 1;

    Shape new_output_shape = convolution->shape();
    ShapeUtil::AppendMajorDimension(1, &new_output_shape);

    int64 input_feature =
        activation->shape().dimensions(input_feature_dimension);

    // The code below edits convolution dimension numbers. Please refer to
    // conv_op_helpers.cc to find how the dimensions were set up originally.

    // Effectively, the new input batch becomes 1, and so does the kernel
    // input feature. The original input batch now becomes a spatial dimension.
    // The output batch (remember that the output is the new kernel for in
    // backprop) becomes a spatial dimension too.

    dim_numbers.set_input_batch_dimension(new_spatial_dim);
    dim_numbers.set_input_feature_dimension(input_batch_dimension);
    dim_numbers.set_kernel_input_feature_dimension(new_spatial_dim);

    dim_numbers.add_input_spatial_dimensions(input_feature_dimension);
    dim_numbers.add_kernel_spatial_dimensions(kernel_input_feature_dimension);

    dim_numbers.add_output_spatial_dimensions(output_batch_dimension);
    dim_numbers.set_output_batch_dimension(new_spatial_dim);

    // Add window for the new spatial dimension.
    Window new_window = convolution->window();
    auto* dim = new_window.add_dimensions();
    dim->set_window_dilation(1);
    dim->set_base_dilation(1);
    dim->set_stride(1);
    dim->set_size(input_feature);

    auto new_convolution = add(HloInstruction::CreateConvolve(
        new_output_shape, activation, filter,
        /*feature_group_count=*/batch_group_count, /*batch_group_count=*/1,
        new_window, dim_numbers, convolution->precision_config()));

    // Delete the extra spatial dimension, and reshape.
    Shape reshaped_convolution_shape = ShapeUtil::DeleteDimension(
        new_spatial_dim - 1, new_convolution->shape());
    auto reshaped_convolution = HloInstruction::CreateReshape(
        reshaped_convolution_shape, new_convolution);

    TF_RETURN_IF_ERROR(computation_->ReplaceWithNewInstruction(
        convolution, std::move(reshaped_convolution)));

    changed_ = true;
  } else {
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

    auto new_convolution = add(HloInstruction::CreateConvolve(
        expanded_filter_shape, activation, filter,
        /*feature_group_count=*/1, /*batch_group_count=*/1,
        convolution->window(), dim_numbers, convolution->precision_config()));

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

    TF_RETURN_IF_ERROR(computation_->ReplaceWithNewInstruction(
        convolution, std::move(reduce_window_converted)));
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
  auto dim_numbers = convolution->convolution_dimension_numbers();
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
    // We want to repeat 'filter' in the 'input_feature_dim' dimension
    // 'group_count' times.
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
        expanded_filter_shape, HloOpcode::kSelect, filter_mask, expanded_filter,
        zero_filter));

    auto new_convolution = HloInstruction::CreateConvolve(
        convolution->shape(), convolution->mutable_operand(0), new_filter,
        /*feature_group_count=*/1, /*batch_group_count=*/1,
        convolution->window(), dim_numbers, convolution->precision_config());
    TF_RETURN_IF_ERROR(computation_->ReplaceWithNewInstruction(
        convolution, std::move(new_convolution)));
  } else {
    int64 activation_input_feature_dim = dim_numbers.input_feature_dimension();

    int64 output_feature =
        filter->shape().dimensions(kernel_output_feature_dim);

    // If group_count == output_feature, then we map those grouped convolutions
    // onto depthwise convolution. This is done by adding an additional spatial
    // dimension to the activations, kernel, and the output.
    // E.g., we would turn
    // [2, 12]{B, IF} conv [3, 4]{IF, OF} into
    // [3, 2, 4]{S, B, IF} depth conv [3, 1, 4]{S, IF, OF}, where S is the
    // additional spatial dimension. The generated convolution output will be
    // [1, 2, 4]{S, B, OF} and then reshape the output back to [2, 4] {B, OF}.

    if (group_count == output_feature && !filter_expansion_) {
      auto filter = convolution->mutable_operand(1);
      auto activation = convolution->mutable_operand(0);

      // Add spatial dimension to the activation, and reshape.
      Shape reshaped_activation_shape = activation->shape();
      ShapeUtil::AppendMajorDimension(group_size, &reshaped_activation_shape);

      int64 new_spatial_dim = reshaped_activation_shape.dimensions().size() - 1;

      reshaped_activation_shape.set_dimensions(activation_input_feature_dim,
                                               group_count);
      activation = add(
          HloInstruction::CreateReshape(reshaped_activation_shape, activation));

      // Add spatial dimension to the filter, and reshape.
      Shape reshaped_filter_shape = filter->shape();
      ShapeUtil::AppendMajorDimension(1, &reshaped_filter_shape);

      filter =
          add(HloInstruction::CreateReshape(reshaped_filter_shape, filter));

      Shape new_output_shape = convolution->shape();
      ShapeUtil::AppendMajorDimension(1, &new_output_shape);

      // Edit convolution dimension numbers. Note that kernel_input_feature_dim
      // now becomes a spatial dimension, and the newly added dimension of size
      // 1 is the new kernel_input_feature_dim.
      dim_numbers.add_input_spatial_dimensions(new_spatial_dim);
      dim_numbers.add_kernel_spatial_dimensions(kernel_input_feature_dim);
      dim_numbers.set_kernel_input_feature_dimension(new_spatial_dim);
      dim_numbers.add_output_spatial_dimensions(new_spatial_dim);

      // Add window for the new spatial dimension.
      Window new_window = convolution->window();
      auto* dim = new_window.add_dimensions();
      dim->set_window_dilation(1);
      dim->set_base_dilation(1);
      dim->set_stride(1);
      dim->set_size(group_size);

      auto new_convolution = add(HloInstruction::CreateConvolve(
          new_output_shape, activation, filter, group_count,
          /*batch_group_count=*/1, new_window, dim_numbers,
          convolution->precision_config()));

      // Delete the extra spatial dimension, and reshape.
      Shape reshaped_convolution_shape =
          ShapeUtil::DeleteDimension(new_spatial_dim, new_convolution->shape());
      auto reshaped_convolution = HloInstruction::CreateReshape(
          reshaped_convolution_shape, new_convolution);

      TF_RETURN_IF_ERROR(computation_->ReplaceWithNewInstruction(
          convolution, std::move(reshaped_convolution)));

    } else {
      // The filter expansion mechanism adds zeroes in the kernel.
      // For an OF = 12, IF = 6, and kernel IF = 2, the expanded filter mask
      // would look like (IF on the Y-axis, OF on the X-axis)
      // 1 1 1 1 0 0 0 0 0 0 0 0
      // 1 1 1 1 0 0 0 0 0 0 0 0
      // 0 0 0 0 1 1 1 1 0 0 0 0
      // 0 0 0 0 1 1 1 1 0 0 0 0
      // 0 0 0 0 0 0 0 0 1 1 1 1
      // 0 0 0 0 0 0 0 0 1 1 1 1
      //
      // Instead of convolving the above with the input, we instead slice the
      // kernel into three kernels, each containing islands of 1s from the
      // filter above. We also slice the activations in the IF dimension with
      // each slice of size = group_size. For each slice, we perform
      // convolutions, and concatenate the generated outputs in the output OF
      // dimension.

      std::vector<HloInstruction*> sliced_convolutions;
      auto activation = convolution->mutable_operand(0);
      std::vector<int64> slice_strides(filter->shape().dimensions_size(), 1);
      std::vector<int64> filter_slice_starts(filter->shape().dimensions_size(),
                                             0);
      std::vector<int64> filter_slice_limits(
          filter->shape().dimensions().begin(),
          filter->shape().dimensions().end());
      std::vector<int64> activation_slice_starts(
          activation->shape().dimensions_size(), 0);
      std::vector<int64> activation_slice_limits(
          activation->shape().dimensions().begin(),
          activation->shape().dimensions().end());

      int64 output_feature =
          filter->shape().dimensions(kernel_output_feature_dim);
      auto output_feature_dim = dim_numbers.output_feature_dimension();
      int64 filter_slice_width = output_feature / group_count;

      int64 activation_input_feature_dim =
          dim_numbers.input_feature_dimension();

      for (int64 i = 0; i < group_count; i++) {
        filter_slice_starts[kernel_output_feature_dim] = i * filter_slice_width;
        filter_slice_limits[kernel_output_feature_dim] =
            (i + 1) * filter_slice_width;
        auto filter_sliced_shape = filter->shape();
        filter_sliced_shape.set_dimensions(kernel_output_feature_dim,
                                           filter_slice_width);
        auto filter_slice = add(HloInstruction::CreateSlice(
            filter_sliced_shape, filter, filter_slice_starts,
            filter_slice_limits, slice_strides));

        activation_slice_starts[activation_input_feature_dim] = i * group_size;
        activation_slice_limits[activation_input_feature_dim] =
            (i + 1) * group_size;
        auto activation_sliced_shape = activation->shape();
        activation_sliced_shape.set_dimensions(activation_input_feature_dim,
                                               group_size);
        auto activation_slice = add(HloInstruction::CreateSlice(
            activation_sliced_shape, activation, activation_slice_starts,
            activation_slice_limits, slice_strides));

        auto conv_slice_shape = convolution->shape();
        conv_slice_shape.set_dimensions(output_feature_dim, filter_slice_width);

        auto new_convolution = add(HloInstruction::CreateConvolve(
            conv_slice_shape, activation_slice, filter_slice,
            /*feature_group_count=*/1, /*batch_group_count=*/1,
            convolution->window(), dim_numbers,
            convolution->precision_config()));

        sliced_convolutions.push_back(new_convolution);
      }

      auto new_conv = HloInstruction::CreateConcatenate(
          convolution->shape(), sliced_convolutions, output_feature_dim);
      TF_RETURN_IF_ERROR(computation_->ReplaceWithNewInstruction(
          convolution, std::move(new_conv)));
    }
  }

  return Status::OK();
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

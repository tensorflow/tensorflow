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
        /*feature_group_count=*/1, convolution->window(), dim_numbers,
        convolution->precision_config());
    TF_RETURN_IF_ERROR(computation_->ReplaceWithNewInstruction(
        convolution, std::move(new_convolution)));
  } else {
    int64 activation_input_feature_dim = dim_numbers.input_feature_dimension();
    auto activation = convolution->mutable_operand(0);

    int64 output_feature =
        filter->shape().dimensions(kernel_output_feature_dim);

    int64 input_feature =
        activation->shape().dimensions(activation_input_feature_dim);

    // If group_count == output_feature, then we map those grouped convolutions
    // onto depthwise convolution + reduce. E.g., we would turn
    // [2, 12]{B, IF} conv [3, 4]{IF, OF} into
    // [2, 12]{B, IF} depth conv [1, 12]{IF, OF}, and then use a reduce window
    // of {1, 3} on the generated [2, 12] output to produce the final result of
    // [2, 4].
    if (group_count == output_feature && !filter_expansion_) {
      Shape reshaped_filter_shape = filter->shape();

      if (kernel_input_feature_dim < kernel_output_feature_dim) {
        // Transpose IF and OF on the kernel.
        std::vector<int64> filter_dims;
        for (int64 i = 0; i < dim_numbers.kernel_spatial_dimensions().size();
             ++i) {
          filter_dims.push_back(dim_numbers.kernel_spatial_dimensions(i));
        }
        filter_dims.push_back(kernel_output_feature_dim);
        filter_dims.push_back(kernel_input_feature_dim);

        Shape transposed_filter = filter->shape();
        auto& dimensions = *transposed_filter.mutable_dimensions();
        std::swap(dimensions[kernel_input_feature_dim],
                  dimensions[kernel_output_feature_dim]);

        filter = add(HloInstruction::CreateTranspose(transposed_filter, filter,
                                                     filter_dims));
      } else {
        // For depthwise convolutions, we want the kernel input feature
        // dimension to be smaller than the output feature dimension. If that's
        // not the case, we swap the dimensions.

        auto& dimensions = *reshaped_filter_shape.mutable_dimensions();
        std::swap(dimensions[kernel_input_feature_dim],
                  dimensions[kernel_output_feature_dim]);

        dim_numbers.set_kernel_input_feature_dimension(
            kernel_output_feature_dim);

        dim_numbers.set_kernel_output_feature_dimension(
            kernel_input_feature_dim);
        std::swap(kernel_output_feature_dim, kernel_input_feature_dim);
      }

      reshaped_filter_shape.set_dimensions(kernel_input_feature_dim, 1);
      reshaped_filter_shape.set_dimensions(kernel_output_feature_dim,
                                           group_count * group_size);
      auto reshaped_filter =
          add(HloInstruction::CreateReshape(reshaped_filter_shape, filter));

      Shape reshaped_convolution_shape = convolution->shape();
      reshaped_convolution_shape.set_dimensions(
          dim_numbers.output_feature_dimension(), group_count * group_size);
      auto new_convolution = add(HloInstruction::CreateConvolve(
          reshaped_convolution_shape, convolution->mutable_operand(0),
          reshaped_filter, /*feature_group_count=*/input_feature,
          convolution->window(), dim_numbers, convolution->precision_config()));

      // Create the reduce window.
      Window window;
      for (int64 i = 0; i < new_convolution->shape().dimensions_size(); ++i) {
        auto* dim = window.add_dimensions();
        dim->set_padding_low(0);
        dim->set_padding_high(0);
        dim->set_window_dilation(1);
        dim->set_base_dilation(1);
        if (i == dim_numbers.output_feature_dimension()) {
          dim->set_stride(group_size);
          dim->set_size(group_size);
        } else {
          dim->set_stride(1);
          dim->set_size(1);
        }
      }

      auto reduce_window_shape = new_convolution->shape();
      reduce_window_shape.set_dimensions(dim_numbers.output_feature_dimension(),
                                         group_count);

      auto zero_literal = LiteralUtil::CreateR0(0.0f);
      TF_ASSIGN_OR_RETURN(zero_literal, zero_literal.Convert(F32));
      auto zero = add(HloInstruction::CreateConstant(std::move(zero_literal)));

      auto reduce_function = [&]() -> HloComputation* {
        HloComputation::Builder b("add_computation");
        Shape shape = ShapeUtil::MakeShape(F32, {});
        auto lhs =
            b.AddInstruction(HloInstruction::CreateParameter(0, shape, "lhs"));
        auto rhs =
            b.AddInstruction(HloInstruction::CreateParameter(1, shape, "rhs"));
        auto scalar_op = b.AddInstruction(
            HloInstruction::CreateBinary(shape, HloOpcode::kAdd, lhs, rhs));
        return computation_->parent()->AddEmbeddedComputation(
            b.Build(scalar_op));
      };

      // Ensure that data input to reduce window is of type F32.
      if (primitive_util::BitWidth(new_convolution->shape().element_type()) <
          primitive_util::BitWidth(F32)) {
        Shape convert_shape = new_convolution->shape();
        convert_shape.set_element_type(F32);
        new_convolution = add(HloInstruction::CreateBitcastConvert(
            convert_shape, new_convolution));
      }

      auto reduce_window = add(HloInstruction::CreateReduceWindow(
          reduce_window_shape, new_convolution, zero, window,
          reduce_function()));

      Shape convert_back_shape = reduce_window->shape();
      convert_back_shape.set_element_type(activation->shape().element_type());

      // Convert reduced data back to the original data type.
      auto reduce_window_converted = HloInstruction::CreateBitcastConvert(
          convert_back_shape, reduce_window);
      TF_RETURN_IF_ERROR(computation_->ReplaceWithNewInstruction(
          convolution, std::move(reduce_window_converted)));

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
            /*feature_group_count=*/1, convolution->window(), dim_numbers,
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

/* Copyright 2026 The OpenXLA Authors.

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

#include "xla/service/gpu/conv_utils.h"

#include <optional>

#include <gtest/gtest.h>
#include "absl/log/check.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/service/shape_inference.h"
#include "xla/shape_util.h"

namespace xla {
namespace gpu {
namespace {

class ConvUtilsTest : public HloHardwareIndependentTestBase {
 public:
  ConvUtilsTest()
      : HloHardwareIndependentTestBase(
            /*verifier_layout_sensitive=*/true,
            /*allow_mixed_precision_in_hlo_verifier=*/false) {
    for (int i = 0; i < 2; ++i) {
      WindowDimension* window_dim = default_conv_window_.add_dimensions();
      window_dim->set_size(1);
      window_dim->set_stride(1);
      window_dim->set_padding_low(0);
      window_dim->set_padding_high(0);
      window_dim->set_window_dilation(1);
      window_dim->set_base_dilation(1);
    }

    dnums_for_backward_filter_.set_input_batch_dimension(3);
    dnums_for_backward_filter_.set_input_feature_dimension(0);
    dnums_for_backward_filter_.add_input_spatial_dimensions(1);
    dnums_for_backward_filter_.add_input_spatial_dimensions(2);
    dnums_for_backward_filter_.set_kernel_input_feature_dimension(0);
    dnums_for_backward_filter_.set_kernel_output_feature_dimension(3);
    dnums_for_backward_filter_.add_kernel_spatial_dimensions(1);
    dnums_for_backward_filter_.add_kernel_spatial_dimensions(2);
    dnums_for_backward_filter_.add_output_spatial_dimensions(0);
    dnums_for_backward_filter_.add_output_spatial_dimensions(1);
    dnums_for_backward_filter_.set_output_batch_dimension(2);
    dnums_for_backward_filter_.set_output_feature_dimension(3);

    dnums_for_backward_input_.set_input_batch_dimension(0);
    dnums_for_backward_input_.set_output_batch_dimension(0);
    dnums_for_backward_input_.set_input_feature_dimension(3);
    dnums_for_backward_input_.set_output_feature_dimension(3);
    dnums_for_backward_input_.add_input_spatial_dimensions(1);
    dnums_for_backward_input_.add_output_spatial_dimensions(1);
    dnums_for_backward_input_.add_input_spatial_dimensions(2);
    dnums_for_backward_input_.add_output_spatial_dimensions(2);
    dnums_for_backward_input_.set_kernel_input_feature_dimension(3);
    dnums_for_backward_input_.set_kernel_output_feature_dimension(2);
    dnums_for_backward_input_.add_kernel_spatial_dimensions(0);
    dnums_for_backward_input_.add_kernel_spatial_dimensions(1);
  }

  using ConvKind = HloConvolutionInstruction::ConvKind;
  // A convolution window with stride 1 and zero padding. The size fields are
  // not set.
  Window default_conv_window_;
  ConvolutionDimensionNumbers dnums_for_backward_filter_;
  ConvolutionDimensionNumbers dnums_for_backward_input_;
};

TEST_F(ConvUtilsTest, BackwardFilterConvolveWithPaddedActivations) {
  auto builder = HloComputation::Builder(TestName());
  HloInstruction* activations =
      builder.AddInstruction(HloInstruction::CreateParameter(
          0, ShapeUtil::MakeShape(F32, {20, 35, 35, 32}), "activations"));
  HloInstruction* gradients =
      builder.AddInstruction(HloInstruction::CreateParameter(
          1, ShapeUtil::MakeShape(F32, {20, 35, 35, 32}), "gradients"));

  Window conv_window = default_conv_window_;
  for (int i = 0; i < 2; ++i) {
    conv_window.mutable_dimensions(i)->set_size(35);
    conv_window.mutable_dimensions(i)->set_padding_low(1);
    conv_window.mutable_dimensions(i)->set_padding_high(1);
  }
  builder.AddInstruction(HloInstruction::CreateConvolve(
      ShapeUtil::MakeShape(F32, {3, 3, 32, 32}), activations, gradients,
      /*feature_group_count=*/1, /*batch_group_count=*/1, conv_window,
      dnums_for_backward_filter_, DefaultPrecisionConfig(2)));

  auto module = CreateNewVerifiedModule();
  HloComputation* entry_computation =
      module->AddEntryComputation(builder.Build());
  HloConvolutionInstruction* root_conv =
      DynCast<HloConvolutionInstruction>(entry_computation->root_instruction());
  root_conv->set_conv_kind(ConvKind::WGRAD);

  const Window restored_conv_window =
      *RestoreWindowFromBackwardFilter(root_conv);
  const ConvolutionDimensionNumbers restored_dims =
      RestoreDimNumberFromBackwardFilter(root_conv);
  for (int i = 0; i < 2; ++i) {
    const WindowDimension& window_dim = restored_conv_window.dimensions(i);
    EXPECT_EQ(1, window_dim.padding_low());
    EXPECT_EQ(1, window_dim.padding_high());
    EXPECT_EQ(1, window_dim.stride());
    EXPECT_EQ(1, window_dim.base_dilation());
    EXPECT_EQ(3, window_dim.size());
  }
  EXPECT_EQ(restored_dims.input_batch_dimension(), 0);
  EXPECT_EQ(restored_dims.input_feature_dimension(), 3);
  EXPECT_EQ(restored_dims.output_batch_dimension(), 3);
  EXPECT_EQ(restored_dims.output_feature_dimension(), 2);
  EXPECT_EQ(restored_dims.kernel_input_feature_dimension(), 3);
  EXPECT_EQ(restored_dims.kernel_output_feature_dimension(), 0);
}

TEST_F(ConvUtilsTest, BackwardInputConvolveEvenPadding) {
  auto builder = HloComputation::Builder(TestName());
  HloInstruction* output =
      builder.AddInstruction(HloInstruction::CreateParameter(
          0, ShapeUtil::MakeShape(F32, {4, 5, 16, 16}), "output"));
  HloInstruction* kernel =
      builder.AddInstruction(HloInstruction::CreateParameter(
          1, ShapeUtil::MakeShape(F32, {5, 3, 7, 7}), "kernel"));
  HloInstruction* reverse_kernel = builder.AddInstruction(
      HloInstruction::CreateReverse(kernel->shape(), kernel, {2, 3}));

  Window conv_window = default_conv_window_;
  for (int i = 0; i < 2; ++i) {
    conv_window.mutable_dimensions(i)->set_size(7);
    conv_window.mutable_dimensions(i)->set_padding_low(3);
    conv_window.mutable_dimensions(i)->set_padding_high(3);
  }
  ConvolutionDimensionNumbers conv_dnums;
  conv_dnums.set_input_batch_dimension(0);
  conv_dnums.set_output_batch_dimension(0);
  conv_dnums.set_input_feature_dimension(1);
  conv_dnums.set_output_feature_dimension(1);
  conv_dnums.add_input_spatial_dimensions(2);
  conv_dnums.add_output_spatial_dimensions(2);
  conv_dnums.add_input_spatial_dimensions(3);
  conv_dnums.add_output_spatial_dimensions(3);
  conv_dnums.set_kernel_input_feature_dimension(0);
  conv_dnums.set_kernel_output_feature_dimension(1);
  conv_dnums.add_kernel_spatial_dimensions(2);
  conv_dnums.add_kernel_spatial_dimensions(3);

  HloInstruction* conv = builder.AddInstruction(HloInstruction::CreateConvolve(
      ShapeUtil::MakeShape(F32, {4, 3, 16, 16}), /*lhs=*/output,
      /*rhs=*/reverse_kernel, /*feature_group_count=*/1,
      /*batch_group_count=*/1, conv_window, conv_dnums,
      DefaultPrecisionConfig(2)));
  // Verify the convolution's shape is consistent with ShapeInference.
  CHECK(ShapeUtil::Compatible(
      conv->shape(),
      ShapeInference::InferConvolveShape(
          output->shape(), reverse_kernel->shape(),
          /*feature_group_count=*/1, /*batch_group_count=*/1, conv_window,
          conv_dnums, /*preferred_element_type=*/std::nullopt)
          .value()));

  auto module = CreateNewVerifiedModule();
  HloComputation* entry_computation =
      module->AddEntryComputation(builder.Build());
  HloConvolutionInstruction* root_conv =
      DynCast<HloConvolutionInstruction>(entry_computation->root_instruction());
  root_conv->set_conv_kind(ConvKind::WGRAD);

  const Window restored_conv_window =
      *RestoreWindowFromBackwardInput(root_conv);
  const ConvolutionDimensionNumbers restored_dims =
      RestoreDimNumberFromBackwardInput(root_conv);
  for (int i = 0; i < 2; ++i) {
    const WindowDimension& window_dim = restored_conv_window.dimensions(i);
    // Low padding of the backward input convolution
    //   = kernel_size - 1 - low padding on gradients.
    EXPECT_EQ(3, window_dim.padding_low());
    EXPECT_EQ(3, window_dim.padding_high());
    EXPECT_EQ(1, window_dim.stride());
    EXPECT_EQ(1, window_dim.base_dilation());
  }
  EXPECT_EQ(restored_dims.kernel_input_feature_dimension(), 1);
  EXPECT_EQ(restored_dims.kernel_output_feature_dimension(), 0);
}

TEST_F(ConvUtilsTest, BackwardInputConvolveUnevenPaddingOnGradients) {
  auto builder = HloComputation::Builder(TestName());
  HloInstruction* output =
      builder.AddInstruction(HloInstruction::CreateParameter(
          0, ShapeUtil::MakeShape(F32, {20, 4, 4, 320}), "output"));
  HloInstruction* kernel =
      builder.AddInstruction(HloInstruction::CreateParameter(
          1, ShapeUtil::MakeShape(F32, {3, 3, 192, 320}), "kernel"));
  HloInstruction* reverse_kernel = builder.AddInstruction(
      HloInstruction::CreateReverse(kernel->shape(), kernel, {0, 1}));

  Window conv_window = default_conv_window_;
  for (int i = 0; i < 2; ++i) {
    conv_window.mutable_dimensions(i)->set_size(3);
    conv_window.mutable_dimensions(i)->set_padding_low(2);
    conv_window.mutable_dimensions(i)->set_padding_high(3);
    // Interior padding = 1.
    conv_window.mutable_dimensions(i)->set_base_dilation(2);
  }
  HloInstruction* conv = builder.AddInstruction(HloInstruction::CreateConvolve(
      ShapeUtil::MakeShape(F32, {20, 10, 10, 192}), output, reverse_kernel,
      /*feature_group_count=*/1, /*batch_group_count=*/1, conv_window,
      dnums_for_backward_input_, DefaultPrecisionConfig(2)));
  // Verify the convolution's shape is consistent with ShapeInference.
  CHECK(ShapeUtil::Compatible(
      conv->shape(), ShapeInference::InferConvolveShape(
                         output->shape(), reverse_kernel->shape(),
                         /*feature_group_count=*/1, /*batch_group_count=*/1,
                         conv_window, dnums_for_backward_input_,
                         /*preferred_element_type=*/std::nullopt)
                         .value()));

  auto module = CreateNewVerifiedModule();
  HloComputation* entry_computation =
      module->AddEntryComputation(builder.Build());
  HloConvolutionInstruction* root_conv =
      DynCast<HloConvolutionInstruction>(entry_computation->root_instruction());
  root_conv->set_conv_kind(ConvKind::DGRAD);
  const Window restored_conv_window =
      *RestoreWindowFromBackwardInput(root_conv);
  const ConvolutionDimensionNumbers restored_dims =
      RestoreDimNumberFromBackwardInput(root_conv);
  for (int i = 0; i < 2; ++i) {
    const WindowDimension& window_dim = restored_conv_window.dimensions(i);
    EXPECT_EQ(0, window_dim.padding_low());
    EXPECT_EQ(0, window_dim.padding_high());
    EXPECT_EQ(2, window_dim.stride());
    EXPECT_EQ(1, window_dim.base_dilation());
  }
  EXPECT_EQ(restored_dims.kernel_input_feature_dimension(), 2);
  EXPECT_EQ(restored_dims.kernel_output_feature_dimension(), 3);
}

TEST_F(ConvUtilsTest, BackwardInputConvolveUnevenPaddingOnActivations) {
  auto builder = HloComputation::Builder(TestName());
  // The gradients are in NCHW layout.
  HloInstruction* output =
      builder.AddInstruction(HloInstruction::CreateParameter(
          0, ShapeUtil::MakeShape(F32, {1, 1, 7, 1}), "output"));
  // The kernel is in HWIO layout.
  HloInstruction* kernel =
      builder.AddInstruction(HloInstruction::CreateParameter(
          1, ShapeUtil::MakeShape(F32, {1, 3, 1, 1}), "kernel"));
  HloInstruction* reverse_kernel = builder.AddInstruction(
      HloInstruction::CreateReverse(kernel->shape(), kernel, {0, 1}));

  Window conv_window = default_conv_window_;
  WindowDimension* forward_conv_col_dim = conv_window.mutable_dimensions(1);
  forward_conv_col_dim->set_size(3);
  forward_conv_col_dim->set_padding_low(2);
  forward_conv_col_dim->set_padding_high(1);
  forward_conv_col_dim->set_base_dilation(2);
  HloInstruction* conv = builder.AddInstruction(HloInstruction::CreateConvolve(
      ShapeUtil::MakeShape(F32, {1, 1, 14, 1}), output, reverse_kernel,
      /*feature_group_count=*/1, /*batch_group_count=*/1, conv_window,
      dnums_for_backward_input_, DefaultPrecisionConfig(2)));
  // Verify the convolution's shape is consistent with ShapeInference.
  CHECK(ShapeUtil::Compatible(
      conv->shape(), ShapeInference::InferConvolveShape(
                         output->shape(), reverse_kernel->shape(),
                         /*feature_group_count=*/1, /*batch_group_count=*/1,
                         conv_window, dnums_for_backward_input_,
                         /*preferred_element_type=*/std::nullopt)
                         .value()));

  auto module = CreateNewVerifiedModule();
  const HloComputation* entry_computation =
      module->AddEntryComputation(builder.Build());
  HloConvolutionInstruction* root_conv =
      DynCast<HloConvolutionInstruction>(entry_computation->root_instruction());
  root_conv->set_conv_kind(ConvKind::DGRAD);
  const WindowDimension backward_conv_col_dim =
      RestoreWindowFromBackwardInput(root_conv)->dimensions(1);
  const ConvolutionDimensionNumbers restored_dims =
      RestoreDimNumberFromBackwardInput(root_conv);
  EXPECT_EQ(0, backward_conv_col_dim.padding_low());
  EXPECT_EQ(1, backward_conv_col_dim.padding_high());
  EXPECT_EQ(restored_dims.kernel_input_feature_dimension(), 2);
  EXPECT_EQ(restored_dims.kernel_output_feature_dimension(), 3);
}

}  // anonymous namespace
}  // namespace gpu
}  // namespace xla

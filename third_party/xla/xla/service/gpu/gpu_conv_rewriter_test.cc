/* Copyright 2018 The OpenXLA Authors.

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

#include "xla/service/gpu/gpu_conv_rewriter.h"

#include <optional>
#include <string>

#include "absl/log/check.h"
#include "absl/strings/str_format.h"
#include "xla/array4d.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/literal_util.h"
#include "xla/protobuf_util.h"
#include "xla/service/gpu/cublas_cudnn.h"
#include "xla/service/pattern_matcher.h"
#include "xla/service/pattern_matcher_gmock.h"
#include "xla/service/shape_inference.h"
#include "xla/shape_util.h"
#include "xla/test.h"
#include "xla/test_helpers.h"
#include "xla/tests/hlo_test_base.h"
#include "tsl/platform/statusor.h"
#include "tsl/platform/test.h"

namespace xla {
namespace gpu {
namespace {

namespace m = ::xla::match;

class GpuConvRewriterTest : public HloTestBase {
 public:
  GpuConvRewriterTest()
      : HloTestBase(/*verifier_layout_sensitive=*/true,
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
    // TF data shapes are by default in the NHWC order, and filter shape is by
    // default in HWIO order. For backward filter convolution, we need to swap
    // the batch and feature dimension in the activations, and treat the batch
    // dimension in gradients as the input feature dimension in the filter.
    //
    // TODO(jingyue): Add more tests on NCHW input order, which TF also
    // supports.
    tf_default_dnums_for_backward_filter_.set_input_batch_dimension(3);
    tf_default_dnums_for_backward_filter_.set_input_feature_dimension(0);
    tf_default_dnums_for_backward_filter_.add_input_spatial_dimensions(1);
    tf_default_dnums_for_backward_filter_.add_input_spatial_dimensions(2);
    tf_default_dnums_for_backward_filter_.set_kernel_input_feature_dimension(0);
    tf_default_dnums_for_backward_filter_.set_kernel_output_feature_dimension(
        3);
    tf_default_dnums_for_backward_filter_.add_kernel_spatial_dimensions(1);
    tf_default_dnums_for_backward_filter_.add_kernel_spatial_dimensions(2);
    tf_default_dnums_for_backward_filter_.add_output_spatial_dimensions(0);
    tf_default_dnums_for_backward_filter_.add_output_spatial_dimensions(1);
    tf_default_dnums_for_backward_filter_.set_output_batch_dimension(2);
    tf_default_dnums_for_backward_filter_.set_output_feature_dimension(3);

    tf_default_dnums_for_backward_input_.set_input_batch_dimension(0);
    tf_default_dnums_for_backward_input_.set_output_batch_dimension(0);
    tf_default_dnums_for_backward_input_.set_input_feature_dimension(3);
    tf_default_dnums_for_backward_input_.set_output_feature_dimension(3);
    tf_default_dnums_for_backward_input_.add_input_spatial_dimensions(1);
    tf_default_dnums_for_backward_input_.add_output_spatial_dimensions(1);
    tf_default_dnums_for_backward_input_.add_input_spatial_dimensions(2);
    tf_default_dnums_for_backward_input_.add_output_spatial_dimensions(2);
    tf_default_dnums_for_backward_input_.set_kernel_input_feature_dimension(3);
    tf_default_dnums_for_backward_input_.set_kernel_output_feature_dimension(2);
    tf_default_dnums_for_backward_input_.add_kernel_spatial_dimensions(0);
    tf_default_dnums_for_backward_input_.add_kernel_spatial_dimensions(1);
  }

 protected:
  bool RunPass(HloModule* module) {
    return GpuConvRewriter().Run(module).value();
  }

  // A convolution window with stride 1 and zero padding. The size fields are
  // not set.
  Window default_conv_window_;
  ConvolutionDimensionNumbers tf_default_dnums_for_backward_filter_;
  ConvolutionDimensionNumbers tf_default_dnums_for_backward_input_;
};

TEST_F(GpuConvRewriterTest, BackwardFilterConvolve) {
  HloComputation::Builder builder(TestName());
  HloInstruction* activations =
      builder.AddInstruction(HloInstruction::CreateParameter(
          0, ShapeUtil::MakeShape(F32, {1, 1, 3, 1}), "activations"));
  HloInstruction* gradients =
      builder.AddInstruction(HloInstruction::CreateParameter(
          1, ShapeUtil::MakeShape(F32, {1, 1, 2, 1}), "gradients"));
  Window conv_window = default_conv_window_;
  conv_window.mutable_dimensions(1)->set_size(2);
  conv_window.mutable_dimensions(1)->set_window_dilation(2);
  auto* conv = builder.AddInstruction(HloInstruction::CreateConvolve(
      ShapeInference::InferConvolveShape(
          activations->shape(), gradients->shape(), /*feature_group_count=*/1,
          /*batch_group_count=*/1, conv_window,
          tf_default_dnums_for_backward_filter_,
          /*preferred_element_type=*/std::nullopt)
          .value(),
      activations, gradients, /*feature_group_count=*/1,
      /*batch_group_count=*/1, conv_window,
      tf_default_dnums_for_backward_filter_, DefaultPrecisionConfig(2)));

  OpMetadata metadata;
  metadata.set_op_name("foo");
  conv->set_metadata(metadata);

  auto module = CreateNewVerifiedModule();
  HloComputation* entry_computation =
      module->AddEntryComputation(builder.Build());
  EXPECT_TRUE(RunPass(module.get()));
  ASSERT_THAT(entry_computation->root_instruction(),
              GmockMatch(m::GetTupleElement(
                  m::CustomCall({kCudnnConvBackwardFilterCallTarget}), 0)));

  // Check that metadata was preserved.
  const auto& md_after_opt =
      entry_computation->root_instruction()->operand(0)->metadata();
  EXPECT_TRUE(protobuf_util::ProtobufEquals(md_after_opt, metadata))
      << md_after_opt.DebugString() << " vs " << metadata.DebugString();
}

TEST_F(GpuConvRewriterTest,
       BackwardFilterConvolveEquivalentToForwardConvolution) {
  HloComputation::Builder builder(TestName());
  HloInstruction* activations =
      builder.AddInstruction(HloInstruction::CreateParameter(
          0, ShapeUtil::MakeShape(F32, {1, 1, 3, 1}), "activations"));
  HloInstruction* gradients =
      builder.AddInstruction(HloInstruction::CreateParameter(
          1, ShapeUtil::MakeShape(F32, {1, 1, 3, 1}), "gradients"));
  Window conv_window = default_conv_window_;
  conv_window.mutable_dimensions(1)->set_size(3);
  builder.AddInstruction(HloInstruction::CreateConvolve(
      ShapeInference::InferConvolveShape(
          activations->shape(), gradients->shape(), /*feature_group_count=*/1,
          /*batch_group_count=*/1, conv_window,
          tf_default_dnums_for_backward_filter_,
          /*preferred_element_type=*/std::nullopt)
          .value(),
      activations, gradients, /*feature_group_count=*/1,
      /*batch_group_count=*/1, conv_window,
      tf_default_dnums_for_backward_filter_, DefaultPrecisionConfig(2)));

  auto module = CreateNewVerifiedModule();
  HloComputation* entry_computation =
      module->AddEntryComputation(builder.Build());
  EXPECT_TRUE(RunPass(module.get()));
  EXPECT_THAT(entry_computation->root_instruction(),
              GmockMatch(m::GetTupleElement(
                  m::CustomCall({kCudnnConvForwardCallTarget}), 0)));
}

// Extracted from block35 training.
TEST_F(GpuConvRewriterTest, BackwardFilterConvolveWithPaddedActivations) {
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
      ShapeUtil::MakeShape(F32, {32, 3, 3, 32}), activations, gradients,
      /*feature_group_count=*/1, /*batch_group_count=*/1, conv_window,
      tf_default_dnums_for_backward_filter_, DefaultPrecisionConfig(2)));

  auto module = CreateNewVerifiedModule();
  HloComputation* entry_computation =
      module->AddEntryComputation(builder.Build());
  EXPECT_TRUE(RunPass(module.get()));
  EXPECT_THAT(entry_computation->root_instruction(),
              GmockMatch(m::GetTupleElement(
                  m::CustomCall({kCudnnConvBackwardFilterCallTarget}), 0)));
}

// Extracted from inception v3 training.
TEST_F(GpuConvRewriterTest, BackwardFilterConvolveWithPaddedGradients) {
  auto builder = HloComputation::Builder(TestName());
  HloInstruction* activations =
      builder.AddInstruction(HloInstruction::CreateParameter(
          0, ShapeUtil::MakeShape(F32, {20, 10, 10, 192}), "activations"));
  HloInstruction* gradients =
      builder.AddInstruction(HloInstruction::CreateParameter(
          1, ShapeUtil::MakeShape(F32, {20, 4, 4, 320}), "gradients"));

  Window conv_window = default_conv_window_;
  for (int i = 0; i < 2; ++i) {
    conv_window.mutable_dimensions(i)->set_size(4);
    conv_window.mutable_dimensions(i)->set_padding_high(-1);
    conv_window.mutable_dimensions(i)->set_window_dilation(2);
  }
  builder.AddInstruction(HloInstruction::CreateConvolve(
      ShapeUtil::MakeShape(F32, {320, 3, 3, 192}), activations, gradients,
      /*feature_group_count=*/1, /*batch_group_count=*/1, conv_window,
      tf_default_dnums_for_backward_filter_, DefaultPrecisionConfig(2)));

  auto module = CreateNewVerifiedModule();
  HloComputation* entry_computation =
      module->AddEntryComputation(builder.Build());
  EXPECT_TRUE(RunPass(module.get()));
  EXPECT_THAT(entry_computation->root_instruction(),
              GmockMatch(m::GetTupleElement(
                  m::CustomCall({kCudnnConvBackwardFilterCallTarget}), 0)));
}

TEST_F(GpuConvRewriterTest, BackwardFilterConvolveWithUnevenPadding) {
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
    // Uneven padding: padding_low=0, padding_high=1
    conv_window.mutable_dimensions(i)->set_padding_high(1);
  }
  builder.AddInstruction(HloInstruction::CreateConvolve(
      ShapeUtil::MakeShape(F32, {32, 2, 2, 32}), activations, gradients,
      /*feature_group_count=*/1, /*batch_group_count=*/1, conv_window,
      tf_default_dnums_for_backward_filter_, DefaultPrecisionConfig(2)));

  auto module = CreateNewVerifiedModule();
  HloComputation* entry_computation =
      module->AddEntryComputation(builder.Build());
  EXPECT_TRUE(RunPass(module.get()));
  EXPECT_THAT(entry_computation->root_instruction(),
              GmockMatch(m::GetTupleElement(
                  m::CustomCall({kCudnnConvBackwardFilterCallTarget}), 0)));
}

TEST_F(GpuConvRewriterTest, BackwardInputConvolveEvenPadding) {
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
  EXPECT_TRUE(RunPass(module.get()));

  ASSERT_THAT(entry_computation->root_instruction(),
              GmockMatch(m::GetTupleElement(
                  m::CustomCall({kCudnnConvBackwardInputCallTarget}), 0)));
  const HloInstruction* custom_call =
      entry_computation->root_instruction()->operand(0);
  for (int i = 0; i < 2; ++i) {
    const WindowDimension& window_dim = custom_call->window().dimensions(i);
    // Low padding of the backward input convolution
    //   = kernel_size - 1 - low padding on gradients.
    EXPECT_EQ(3, window_dim.padding_low());
    EXPECT_EQ(3, window_dim.padding_high());
    EXPECT_EQ(1, window_dim.stride());
    EXPECT_EQ(1, window_dim.base_dilation());
  }
}

// Convolve([abc], [x], base_dilation=2)
//   = Convolve([abc], Reverse([x]), base_dilation=2)
//   = BackwardInputConvolve([abc], [x], stride=2)
TEST_F(GpuConvRewriterTest, BackwardInputConvolve1x1Filter) {
  auto builder = HloComputation::Builder(TestName());
  // NHWC dimension order.
  HloInstruction* output =
      builder.AddInstruction(HloInstruction::CreateParameter(
          0, ShapeUtil::MakeShape(F32, {1, 1, 3, 1}), "output"));
  // HWOI dimension order.
  HloInstruction* kernel =
      builder.AddInstruction(HloInstruction::CreateParameter(
          1, ShapeUtil::MakeShape(F32, {1, 1, 1, 1}), "kernel"));

  Window conv_window = default_conv_window_;
  conv_window.mutable_dimensions(1)->set_base_dilation(2);

  builder.AddInstruction(HloInstruction::CreateConvolve(
      ShapeInference::InferConvolveShape(
          output->shape(), kernel->shape(),
          /*feature_group_count=*/1,
          /*batch_group_count=*/1, conv_window,
          tf_default_dnums_for_backward_input_,
          /*preferred_element_type=*/std::nullopt)
          .value(),
      /*lhs=*/output, /*rhs=*/kernel, /*feature_group_count=*/1,
      /*batch_group_count=*/1, conv_window,
      tf_default_dnums_for_backward_input_, DefaultPrecisionConfig(2)));

  auto module = CreateNewVerifiedModule();
  HloComputation* entry_computation =
      module->AddEntryComputation(builder.Build());
  EXPECT_TRUE(RunPass(module.get()));
  EXPECT_THAT(entry_computation->root_instruction(),
              GmockMatch(m::GetTupleElement(
                  m::CustomCall({kCudnnConvBackwardInputCallTarget}), 0)));
}

// BackwardInputConvolve([abc], [x], stride=1) is equivalent to
// ForwardConvolve([abc], [x], stride=1). No need to fold it into backward input
// convolution.
TEST_F(GpuConvRewriterTest,
       BackwardInputConvolve1x1FilterEquivalentToForwardConvolve) {
  auto builder = HloComputation::Builder(TestName());
  // NHWC dimension order.
  HloInstruction* output =
      builder.AddInstruction(HloInstruction::CreateParameter(
          0, ShapeUtil::MakeShape(F32, {1, 1, 3, 1}), "output"));
  // HWOI dimension order.
  HloInstruction* kernel =
      builder.AddInstruction(HloInstruction::CreateParameter(
          1, ShapeUtil::MakeShape(F32, {1, 1, 1, 1}), "kernel"));

  builder.AddInstruction(HloInstruction::CreateConvolve(
      ShapeInference::InferConvolveShape(
          output->shape(), kernel->shape(), /*feature_group_count=*/1,
          /*batch_group_count=*/1, default_conv_window_,
          tf_default_dnums_for_backward_input_,
          /*preferred_element_type=*/std::nullopt)
          .value(),
      /*lhs=*/output, /*rhs=*/kernel, /*feature_group_count=*/1,
      /*batch_group_count=*/1, default_conv_window_,
      tf_default_dnums_for_backward_input_, DefaultPrecisionConfig(2)));

  auto module = CreateNewVerifiedModule();
  HloComputation* entry_computation =
      module->AddEntryComputation(builder.Build());
  EXPECT_TRUE(RunPass(module.get()));
  EXPECT_THAT(entry_computation->root_instruction(),
              GmockMatch(m::GetTupleElement(
                  m::CustomCall({kCudnnConvForwardCallTarget}), 0)));
}

// Extracted from Inception V3 training.
//
//                                  filter(HWIO)
//                                  3x3x192x320
//                                      |
//                                      v
//      gradients(NHWC)              reverse
//        20x4x4x320               3x3x192x320
//                    \            /
//                     \          /
//  conv (NHWC) with padding (low=2,high=3,interior=1)
//                     20x10x10x192
//
// Gradients are padded unevenly.
TEST_F(GpuConvRewriterTest, BackwardInputConvolveUnevenPaddingOnGradients) {
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
      tf_default_dnums_for_backward_input_, DefaultPrecisionConfig(2)));
  // Verify the convolution's shape is consistent with ShapeInference.
  CHECK(ShapeUtil::Compatible(
      conv->shape(), ShapeInference::InferConvolveShape(
                         output->shape(), reverse_kernel->shape(),
                         /*feature_group_count=*/1, /*batch_group_count=*/1,
                         conv_window, tf_default_dnums_for_backward_input_,
                         /*preferred_element_type=*/std::nullopt)
                         .value()));

  auto module = CreateNewVerifiedModule();
  HloComputation* entry_computation =
      module->AddEntryComputation(builder.Build());
  EXPECT_TRUE(RunPass(module.get()));
  ASSERT_THAT(entry_computation->root_instruction(),
              GmockMatch(m::GetTupleElement(
                  m::CustomCall({kCudnnConvBackwardInputCallTarget}), 0)));
  const HloInstruction* custom_call =
      entry_computation->root_instruction()->operand(0);
  for (int i = 0; i < 2; ++i) {
    const WindowDimension& window_dim = custom_call->window().dimensions(i);
    EXPECT_EQ(0, window_dim.padding_low());
    EXPECT_EQ(0, window_dim.padding_high());
    EXPECT_EQ(2, window_dim.stride());
    EXPECT_EQ(1, window_dim.base_dilation());
  }
}

// Similar to BackwardInputConvolveUnevenPadding, but the low padding of the
// gradients exceeds kernel_size - 1. Therefore, this pattern cannot be fused.
TEST_F(GpuConvRewriterTest, BackwardInputConvolveLowPaddingTooLarge) {
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
    conv_window.mutable_dimensions(i)->set_padding_low(3);
    conv_window.mutable_dimensions(i)->set_padding_high(2);
    conv_window.mutable_dimensions(i)->set_base_dilation(2);
  }
  HloInstruction* conv = builder.AddInstruction(HloInstruction::CreateConvolve(
      ShapeUtil::MakeShape(F32, {20, 10, 10, 192}), output, reverse_kernel,
      /*feature_group_count=*/1, /*batch_group_count=*/1, conv_window,
      tf_default_dnums_for_backward_input_, DefaultPrecisionConfig(2)));
  // Verify the convolution's shape is consistent with ShapeInference.
  CHECK(ShapeUtil::Compatible(
      conv->shape(), ShapeInference::InferConvolveShape(
                         output->shape(), reverse_kernel->shape(),
                         /*feature_group_count=*/1, /*batch_group_count=*/1,
                         conv_window, tf_default_dnums_for_backward_input_,
                         /*preferred_element_type=*/std::nullopt)
                         .value()));

  auto module = CreateNewVerifiedModule();
  HloComputation* entry_computation =
      module->AddEntryComputation(builder.Build());
  EXPECT_TRUE(RunPass(module.get()));
  EXPECT_THAT(entry_computation->root_instruction(),
              GmockMatch(m::GetTupleElement(
                  m::CustomCall({kCudnnConvForwardCallTarget}), 0)));
}

// Extracted from Resnet-50.
//
// For simplicity, we focus on the column dimension and ignore other dimensions.
// We use [?] to represent the shape instead of the content.
//
// Suppose operator FC does
//   [4] = conv([14], [3], stride=2, padding_high=1)  // Padding::kSame
//
// BC = BackwardInput(FC) does:
//   [14] = conv([7], reverse([3]),
//               padding_low=2, padding_high=1, base_dilation=2)
//
// We should fuse BC even though padding on activations is uneven, because
// GpuConvPaddingLegalization will canonicalize the fusion HLO.
TEST_F(GpuConvRewriterTest, BackwardInputConvolveUnevenPaddingOnActivations) {
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
      tf_default_dnums_for_backward_input_, DefaultPrecisionConfig(2)));
  // Verify the convolution's shape is consistent with ShapeInference.
  CHECK(ShapeUtil::Compatible(
      conv->shape(), ShapeInference::InferConvolveShape(
                         output->shape(), reverse_kernel->shape(),
                         /*feature_group_count=*/1, /*batch_group_count=*/1,
                         conv_window, tf_default_dnums_for_backward_input_,
                         /*preferred_element_type=*/std::nullopt)
                         .value()));

  auto module = CreateNewVerifiedModule();
  const HloComputation* entry_computation =
      module->AddEntryComputation(builder.Build());
  EXPECT_TRUE(RunPass(module.get()));
  ASSERT_THAT(entry_computation->root_instruction(),
              GmockMatch(m::GetTupleElement(
                  m::CustomCall({kCudnnConvBackwardInputCallTarget}), 0)));
  const WindowDimension& backward_conv_col_dim =
      entry_computation->root_instruction()->operand(0)->window().dimensions(1);
  EXPECT_EQ(0, backward_conv_col_dim.padding_low());
  EXPECT_EQ(1, backward_conv_col_dim.padding_high());
}

// For simplicity, we focus on the column dimension and ignore other dimensions.
// We use [?] to represent the shape instead of the content.
//
// Suppose operator FC does
//   [3] = conv([4], [2], padding_low=1, padding_high=-1)
//
// BC = BackwardInput(FC) does:
//   [4] = conv([3], reverse([2]), padding_high=2)
//
// We currently don't fuse BC because GpuConvPaddingLegalization
// doesn't support negative padding on the gradients of backward convolution
// (b/32744257).
TEST_F(GpuConvRewriterTest,
       BackwardInputConvolveNegativePaddingHighOnActivations) {
  auto builder = HloComputation::Builder(TestName());
  // The gradients are in NCHW layout.
  HloInstruction* output =
      builder.AddInstruction(HloInstruction::CreateParameter(
          0, ShapeUtil::MakeShape(F32, {1, 1, 3, 1}), "output"));
  // The kernel is in HWIO layout.
  HloInstruction* kernel =
      builder.AddInstruction(HloInstruction::CreateParameter(
          1, ShapeUtil::MakeShape(F32, {1, 2, 1, 1}), "kernel"));
  HloInstruction* reverse_kernel = builder.AddInstruction(
      HloInstruction::CreateReverse(kernel->shape(), kernel, {0, 1}));

  Window conv_window = default_conv_window_;
  WindowDimension* forward_conv_col_dim = conv_window.mutable_dimensions(1);
  forward_conv_col_dim->set_size(2);
  forward_conv_col_dim->set_padding_high(2);
  HloInstruction* conv = builder.AddInstruction(HloInstruction::CreateConvolve(
      ShapeUtil::MakeShape(F32, {1, 1, 4, 1}), output, reverse_kernel,
      /*feature_group_count=*/1, /*batch_group_count=*/1, conv_window,
      tf_default_dnums_for_backward_input_, DefaultPrecisionConfig(2)));
  // Verify the convolution's shape is consistent with ShapeInference.
  CHECK(ShapeUtil::Compatible(
      conv->shape(), ShapeInference::InferConvolveShape(
                         output->shape(), reverse_kernel->shape(),
                         /*feature_group_count=*/1, /*batch_group_count=*/1,
                         conv_window, tf_default_dnums_for_backward_input_,
                         /*preferred_element_type=*/std::nullopt)
                         .value()));

  auto module = CreateNewVerifiedModule();
  HloComputation* entry_computation =
      module->AddEntryComputation(builder.Build());
  EXPECT_TRUE(RunPass(module.get()));
  EXPECT_THAT(entry_computation->root_instruction(),
              GmockMatch(m::GetTupleElement(
                  m::CustomCall({kCudnnConvForwardCallTarget}), 0)));
}

// Check that we will materialize a reversed version of a constant in order to
// pattern-match a backwards input convolution.
TEST_F(GpuConvRewriterTest, BackwardInputConvolveConstantFilter) {
  Array4D<float> constant_arr(4, 4, 2, 2);
  constant_arr.FillIota(0);
  std::string constant_str =
      LiteralUtil::CreateR4FromArray4D(constant_arr).ToStringWithoutShape();

  const std::string module_str = absl::StrFormat(R"(
    HloModule test

    ENTRY entry_computation {
      param0 = f32[128,2,16,16]{3,2,1,0} parameter(0)
      constant = f32[4,4,2,2]{3,2,1,0} constant(%s)
      ROOT convolution = f32[128,2,32,32]{3,2,1,0} convolution(param0, constant),
          window={size=4x4 pad=2_2x2_2 lhs_dilate=2x2},
          dim_labels=bf01_01oi->bf01, feature_group_count=1
    })",
                                                 constant_str);
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(module_str));

  EXPECT_TRUE(RunPass(m.get()));
  EXPECT_THAT(m->entry_computation()->root_instruction(),
              GmockMatch(m::GetTupleElement(
                  m::CustomCall({kCudnnConvBackwardInputCallTarget},
                                m::Parameter(), m::Reverse(m::Constant())),
                  0)));
}

TEST_F(GpuConvRewriterTest, TestBackwardFilterPatternMatch) {
  // All filter dimensions are larger than the corresponding output dimensions.
  // This must be a backward filter convolution.
  const std::string module_str = absl::StrFormat(R"(
    HloModule Test

    ENTRY Test {
      input = f32[8,120,256,256] parameter(0)
      filter = f32[8,120,256,256] parameter(1)

      ROOT conv = f32[120,120,3,3] convolution(input, filter), window={size=256x256 pad=1_1x1_1}, dim_labels=fb01_io01->fb01
    })");
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(module_str));

  EXPECT_TRUE(RunPass(m.get()));
  EXPECT_THAT(m->entry_computation()->root_instruction(),
              GmockMatch(m::GetTupleElement(
                  m::CustomCall({kCudnnConvBackwardFilterCallTarget},
                                m::Parameter(0), m::Parameter(1)),
                  0)));
}

TEST_F(GpuConvRewriterTest, TestBackwardFilterPatternNoMatch) {
  // At least one filter dimension is smaller than the corresponding output
  // dimension. This must be a forward convolution.
  const std::string module_str = absl::StrFormat(R"(
    HloModule Test

    ENTRY Test {
      input = f32[8,128,2,32] parameter(0)
      filter = f32[3,3,128,128] parameter(1)

      ROOT conv = f32[8,128,2,32] convolution(input, filter), window={size=3x3 pad=1_1x1_1}, dim_labels=bf01_01io->bf01
    })");
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(module_str));

  EXPECT_TRUE(RunPass(m.get()));
  EXPECT_THAT(m->entry_computation()->root_instruction(),
              GmockMatch(m::GetTupleElement(
                  m::CustomCall({kCudnnConvForwardCallTarget}, m::Parameter(0),
                                m::Parameter(1)),
                  0)));
}

TEST_F(GpuConvRewriterTest, TestConv1dBackwardFilterPatternMatch) {
  // There exist one kernel dimension equal to output dimension, regard
  // it as backward filter if conv is 1d.
  const std::string module_str = absl::StrFormat(R"(
    HloModule Test

    ENTRY Test {
      input = f32[8,256,128] parameter(0)
      filter = f32[8,254,128] parameter(1)
      reshape.1 = f32[8,1,256,128] reshape(input)
      reshape.2 = f32[8,1,254,128] reshape(filter)
      ROOT conv = f32[1,3,128,128] convolution(reshape.1, reshape.2), window={size=1x254}, dim_labels=f01b_i01o->01bf
    })");
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(module_str));

  EXPECT_TRUE(RunPass(m.get()));
  EXPECT_THAT(m->entry_computation()->root_instruction(),
              GmockMatch(m::GetTupleElement(
                  m::CustomCall({kCudnnConvBackwardFilterCallTarget},
                                m::Reshape(), m::Reshape()),
                  0)));
}

TEST_F(GpuConvRewriterTest, TestConv1dBackwardInputPatternMatch) {
  // For conv1d backward input, filter may reverse first and then reshape.
  const std::string module_str = absl::StrFormat(R"(
    HloModule Test

    ENTRY Test {
      input = f32[8,254,128] parameter(0)
      filter = f32[3,128,128] parameter(1)
      reverse = f32[3,128,128] reverse(filter), dimensions={0}
      reshape.1 = f32[8,1,254,128] reshape(input)
      reshape.2 = f32[1,3,128,128] reshape(reverse)
      ROOT conv = f32[8,1,256,128] convolution(reshape.1, reshape.2), window={size=1x3 pad=0_0x2_2}, dim_labels=b01f_01oi->b01f
    })");
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(module_str));

  EXPECT_TRUE(RunPass(m.get()));
  EXPECT_THAT(m->entry_computation()->root_instruction(),
              GmockMatch(m::GetTupleElement(
                  m::CustomCall({kCudnnConvBackwardInputCallTarget},
                                m::Reshape(), m::Reshape()),
                  0)));
}

}  // anonymous namespace
}  // namespace gpu
}  // namespace xla

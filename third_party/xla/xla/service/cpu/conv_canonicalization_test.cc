/* Copyright 2017 The OpenXLA Authors.

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

#include "xla/service/cpu/conv_canonicalization.h"

#include <cstdint>

#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/testlib/test.h"
#include "xla/hlo/testlib/test_helpers.h"
#include "xla/literal_util.h"
#include "xla/service/cpu/target_machine_features_stub.h"
#include "xla/tests/hlo_test_base.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace cpu {

using ::testing::ElementsAre;

class ConvCanonicalizationTest : public HloTestBase {
 public:
  ConvCanonicalizationTest() {
    for (int i = 0; i < 2; ++i) {
      auto dim = conv_window_.add_dimensions();
      dim->set_size(kWindowSize);
      dim->set_stride(1);
      dim->set_padding_low(0);
      dim->set_padding_high(0);
      dim->set_window_dilation(1);
      dim->set_base_dilation(1);
    }
  }

 protected:
  Window conv_window_;

  static constexpr int kBatchSize = 50;
  static constexpr int kInputSize = 28;
  static constexpr int kWindowSize = 5;
  static constexpr int kInputFeatureCount = 32;
  static constexpr int kOutputFeatureCount = 64;
};

TEST_F(ConvCanonicalizationTest, NonCanonicalToCanonical) {
  auto builder = HloComputation::Builder(TestName());
  // The input dimensions are in CNHW order.
  auto input = builder.AddInstruction(HloInstruction::CreateConstant(
      LiteralUtil::CreateR4FromArray4D(Array4D<float>(
          kInputFeatureCount, kBatchSize, kInputSize, kInputSize))));
  // The kernel dimensions are in OIHW order.
  auto kernel = builder.AddInstruction(HloInstruction::CreateConstant(
      LiteralUtil::CreateR4FromArray4D(Array4D<float>(
          kOutputFeatureCount, kInputFeatureCount, kWindowSize, kWindowSize))));

  ConvolutionDimensionNumbers dnums;
  dnums.set_input_batch_dimension(1);
  dnums.set_output_batch_dimension(1);
  dnums.add_input_spatial_dimensions(2);
  dnums.add_output_spatial_dimensions(2);
  dnums.add_input_spatial_dimensions(3);
  dnums.add_output_spatial_dimensions(3);
  dnums.set_input_feature_dimension(0);
  dnums.set_output_feature_dimension(0);
  dnums.add_kernel_spatial_dimensions(2);
  dnums.add_kernel_spatial_dimensions(3);
  dnums.set_kernel_input_feature_dimension(1);
  dnums.set_kernel_output_feature_dimension(0);
  auto output_size = kInputSize - kWindowSize + 1;
  builder.AddInstruction(HloInstruction::CreateConvolve(
      ShapeUtil::MakeShape(
          F32, {kOutputFeatureCount, kBatchSize, output_size, output_size}),
      input, kernel, /*feature_group_count=*/1, /*batch_group_count=*/1,
      conv_window_, dnums, DefaultPrecisionConfig(2)));

  auto module = CreateNewVerifiedModule();
  HloComputation* entry_computation =
      module->AddEntryComputation(builder.Build());

  cpu::TargetMachineFeaturesStub target_machine_features(
      [](int64_t shape_size) {
        return cpu::TargetMachineFeatures::kEigenExpectedTensorAlignment;
      });
  ConvCanonicalization conv_canonicalization(&target_machine_features);
  EXPECT_TRUE(conv_canonicalization.Run(module.get()).value());

  const HloInstruction* output_reshape = entry_computation->root_instruction();
  EXPECT_EQ(HloOpcode::kTranspose, output_reshape->opcode());
  const HloInstruction* canonical_conv = output_reshape->operand(0);
  EXPECT_EQ(HloOpcode::kConvolution, canonical_conv->opcode());
  const HloInstruction* input_reshape = canonical_conv->operand(0);
  EXPECT_EQ(HloOpcode::kTranspose, input_reshape->opcode());
  const HloInstruction* kernel_reshape = canonical_conv->operand(1);
  EXPECT_EQ(HloOpcode::kTranspose, kernel_reshape->opcode());

  // The input is in CNHW order. input_reshape should produce
  // NHWC for the convolution to hit the Eigen fast path.
  EXPECT_THAT(input_reshape->dimensions(), ElementsAre(1, 2, 3, 0));
  // The kernel is in OIHW order. kernel_reshape should produce
  // HWIO for the convolution to hit the Eigen fast path.
  EXPECT_THAT(kernel_reshape->dimensions(), ElementsAre(2, 3, 1, 0));
  // The output of the canonical convolution is in NHWC order (the same as
  // input_reshape's order). output_reshape should restore that order to the
  // order of the computation root (CNHW).
  EXPECT_THAT(output_reshape->dimensions(), ElementsAre(3, 0, 1, 2));
}

TEST_F(ConvCanonicalizationTest, CanonicalStaysTheSame) {
  auto builder = HloComputation::Builder(TestName());
  // The input dimensions are in NHWC order.
  auto input = builder.AddInstruction(HloInstruction::CreateConstant(
      LiteralUtil::CreateR4FromArray4D(Array4D<float>(
          kBatchSize, kInputSize, kInputSize, kInputFeatureCount))));
  // The kernel dimensions are in HWIO order.
  auto kernel = builder.AddInstruction(HloInstruction::CreateConstant(
      LiteralUtil::CreateR4FromArray4D(Array4D<float>(
          kWindowSize, kWindowSize, kInputFeatureCount, kOutputFeatureCount))));

  ConvolutionDimensionNumbers dnums;
  dnums.set_input_batch_dimension(0);
  dnums.set_output_batch_dimension(0);
  dnums.add_input_spatial_dimensions(1);
  dnums.add_output_spatial_dimensions(1);
  dnums.add_input_spatial_dimensions(2);
  dnums.add_output_spatial_dimensions(2);
  dnums.set_input_feature_dimension(3);
  dnums.set_output_feature_dimension(3);
  dnums.add_kernel_spatial_dimensions(0);
  dnums.add_kernel_spatial_dimensions(1);
  dnums.set_kernel_input_feature_dimension(2);
  dnums.set_kernel_output_feature_dimension(3);
  auto output_size = kInputSize - kWindowSize + 1;
  builder.AddInstruction(HloInstruction::CreateConvolve(
      ShapeUtil::MakeShape(
          F32, {kBatchSize, output_size, output_size, kOutputFeatureCount}),
      input, kernel, /*feature_group_count=*/1, /*batch_group_count=*/1,
      conv_window_, dnums, DefaultPrecisionConfig(2)));

  auto module = CreateNewVerifiedModule();
  module->AddEntryComputation(builder.Build());

  cpu::TargetMachineFeaturesStub target_machine_features(
      [](int64_t shape_size) {
        return cpu::TargetMachineFeatures::kEigenExpectedTensorAlignment;
      });
  ConvCanonicalization conv_canonicalization(&target_machine_features);
  EXPECT_FALSE(conv_canonicalization.Run(module.get()).value());
}

}  // namespace cpu
}  // namespace xla

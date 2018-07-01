/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/plugin/poplar/driver/outliner.h"

#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"

namespace xla {
namespace poplarplugin {
namespace {

using OutlinerTest = HloTestBase;

static Window GetDefaultWindow() {
  Window window;
  for (int i = 0; i < 2; ++i) {
    auto dim = window.add_dimensions();
    dim->set_size(4);
    dim->set_stride(1);
    dim->set_padding_low(0);
    dim->set_padding_high(0);
    dim->set_window_dilation(1);
    dim->set_base_dilation(1);
  }
  return window;
}

static ConvolutionDimensionNumbers GetDefaultDimensions() {
  ConvolutionDimensionNumbers dimension;
  dimension.set_input_batch_dimension(0);
  dimension.add_input_spatial_dimensions(1);
  dimension.add_input_spatial_dimensions(2);
  dimension.set_input_feature_dimension(3);

  dimension.set_output_batch_dimension(0);
  dimension.add_output_spatial_dimensions(1);
  dimension.add_output_spatial_dimensions(2);
  dimension.set_output_feature_dimension(3);

  dimension.set_kernel_output_feature_dimension(0);
  dimension.set_kernel_input_feature_dimension(1);
  dimension.add_kernel_spatial_dimensions(2);
  dimension.add_kernel_spatial_dimensions(3);
  return dimension;
}

// Test that `map` with `max` is transformed to `max`
TEST_F(OutlinerTest, Convolution) {
  Shape image_shape = ShapeUtil::MakeShape(F32, {1, 4, 4, 2});
  Shape kernel_shape = ShapeUtil::MakeShape(F32, {2, 2, 4, 4});
  Shape bias_shape = ShapeUtil::MakeShape(F32, {4, 4, 2});

  Window window = GetDefaultWindow();
  ConvolutionDimensionNumbers dimension = GetDefaultDimensions();

  auto builder = HloComputation::Builder(TestName());
  auto in = builder.AddInstruction(
      HloInstruction::CreateParameter(0, image_shape, "input"));
  auto w1 = builder.AddInstruction(
      HloInstruction::CreateParameter(1, kernel_shape, "weights1"));
  auto w2 = builder.AddInstruction(
      HloInstruction::CreateParameter(2, kernel_shape, "weights2"));
  auto b1 = builder.AddInstruction(
      HloInstruction::CreateParameter(3, bias_shape, "bias1"));
  auto b2 = builder.AddInstruction(
      HloInstruction::CreateParameter(4, bias_shape, "bias2"));
  auto c1 = builder.AddInstruction(
      HloInstruction::CreateConvolve(image_shape, in, w1, window, dimension));
  auto a1 = builder.AddInstruction(
      HloInstruction::CreateBinary(image_shape, HloOpcode::kAdd, c1, b1));
  auto c2 = builder.AddInstruction(
      HloInstruction::CreateConvolve(image_shape, a1, w2, window, dimension));
  auto a2 = builder.AddInstruction(
      HloInstruction::CreateBinary(image_shape, HloOpcode::kAdd, c2, b2));

  builder.AddInstruction(HloInstruction::CreateTuple({a2}));

  auto computation = builder.Build();

  auto hlo_module = CreateNewModule();
  hlo_module->AddEntryComputation(std::move(computation));

  EXPECT_THAT(hlo_module->computation_count(), 1);
  EXPECT_THAT(hlo_module->entry_computation()->instruction_count(), 10);

  Outliner outliner;
  EXPECT_TRUE(outliner.Run(hlo_module.get()).ValueOrDie());
  EXPECT_THAT(hlo_module->computation_count(), 3);
  EXPECT_THAT(hlo_module->entry_computation()->instruction_count(), 10);
}

}  // namespace
}  // namespace poplarplugin
}  // namespace xla

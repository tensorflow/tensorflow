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

#include "tensorflow/compiler/plugin/poplar/driver/allocation_finder.h"

#include "tensorflow/compiler/xla/service/shape_inference.h"

#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/core/lib/core/status_test_util.h"

namespace xla {
namespace poplarplugin {
namespace {

using AllocationFinderTest = HloTestBase;

static Window GetConv1Window() {
  Window window;
  for (int i = 0; i < 2; ++i) {
    auto dim = window.add_dimensions();
    dim->set_size(3);
    dim->set_stride(1);
    dim->set_padding_low(1);
    dim->set_padding_high(1);
    dim->set_window_dilation(1);
    dim->set_base_dilation(1);
  }
  return window;
}

static Window GetConv2Window() {
  Window window;
  for (int i = 0; i < 2; ++i) {
    auto dim = window.add_dimensions();
    dim->set_size(3);
    dim->set_stride(2);
    dim->set_padding_low(1);
    dim->set_padding_high(1);
    dim->set_window_dilation(1);
    dim->set_base_dilation(1);
  }
  return window;
}

static ConvolutionDimensionNumbers GetConvDimensions() {
  ConvolutionDimensionNumbers dimension;
  dimension.set_batch_dimension(0);
  dimension.add_spatial_dimensions(1);
  dimension.add_spatial_dimensions(2);
  dimension.set_feature_dimension(3);
  dimension.add_kernel_spatial_dimensions(0);
  dimension.add_kernel_spatial_dimensions(1);
  dimension.set_kernel_input_feature_dimension(2);
  dimension.set_kernel_output_feature_dimension(3);
  return dimension;
}


// Check basic parameter matching
TEST_F(AllocationFinderTest, FindBasicTensorAllocations) {
  Shape input_shape = ShapeUtil::MakeShape(F32, {1, 10, 10, 2});
  Shape weight_shape = ShapeUtil::MakeShape(F32, {3, 3, 2, 1});

  Shape conv_shape = ShapeInference::InferConvolveShape(
        input_shape, weight_shape,
        GetConv1Window(), GetConvDimensions()).ConsumeValueOrDie();

  auto builder = HloComputation::Builder(TestName());
  auto op0 = builder.AddInstruction(
    HloInstruction::CreateParameter(0, input_shape, "op0"));
  auto op1 = builder.AddInstruction(
    HloInstruction::CreateParameter(1, input_shape, "op1"));
  auto op2 = builder.AddInstruction(
    HloInstruction::CreateParameter(2, weight_shape, "op2"));

  auto add = builder.AddInstruction(
    HloInstruction::CreateBinary(input_shape, HloOpcode::kAdd, op0, op1));

  auto conv = builder.AddInstruction(
    HloInstruction::CreateConvolve(conv_shape, op1, op2,
                                   GetConv1Window(), GetConvDimensions()));

  builder.AddInstruction(
          HloInstruction::CreateTuple({add, conv}));

  auto computation = builder.Build();

  auto hlo_module = MakeUnique<HloModule>("test_module");
  hlo_module->AddEntryComputation(std::move(computation));

  AllocationFinder finder;
  TF_EXPECT_OK(finder.CreateAllocationMap(hlo_module.get()));

  EXPECT_EQ(finder.tensor_allocation_map.size(), 2);
  EXPECT_EQ(finder.tensor_allocation_map.at(op1), conv);
  EXPECT_EQ(finder.tensor_allocation_map.at(op2), conv);
}

// Check it goes through call sites
TEST_F(AllocationFinderTest, FindSubCompTensorAllocations) {
  Shape input_shape = ShapeUtil::MakeShape(F32, {1, 10, 10, 2});
  Shape weight_shape = ShapeUtil::MakeShape(F32, {3, 3, 2, 1});

  Shape conv_shape = ShapeInference::InferConvolveShape(
          input_shape, weight_shape,
          GetConv1Window(), GetConvDimensions()).ConsumeValueOrDie();

  /* Create convolution sub-computation */
  auto builder_sub = HloComputation::Builder(TestName());
  auto op0_sub = builder_sub.AddInstruction(
        HloInstruction::CreateParameter(0, input_shape, "input"));
  auto op1_sub = builder_sub.AddInstruction(
        HloInstruction::CreateParameter(1, weight_shape, "weights"));

  auto conv = builder_sub.AddInstruction(
        HloInstruction::CreateConvolve(conv_shape, op0_sub, op1_sub,
                                       GetConv1Window(), GetConvDimensions()));

  auto computation_sub = builder_sub.Build();

  /* Create main computation */
  auto builder_main = HloComputation::Builder(TestName());
  auto op0 = builder_main.AddInstruction(
          HloInstruction::CreateParameter(0, input_shape, "op0"));
  auto op1 = builder_main.AddInstruction(
          HloInstruction::CreateParameter(1, input_shape, "op1"));
  auto op2 = builder_main.AddInstruction(
          HloInstruction::CreateParameter(2, weight_shape, "op2"));

  auto add = builder_main.AddInstruction(
          HloInstruction::CreateBinary(input_shape, HloOpcode::kAdd, op0, op1));

  auto call = builder_main.AddInstruction(
          HloInstruction::CreateCall(conv_shape, {op1, op2},
                                     computation_sub.get()));

  builder_main.AddInstruction(
          HloInstruction::CreateTuple({add, call}));

  auto computation_main = builder_main.Build();

  auto hlo_module = MakeUnique<HloModule>("test_module");
  hlo_module->AddEmbeddedComputation(std::move(computation_sub));
  hlo_module->AddEntryComputation(std::move(computation_main));

  AllocationFinder finder;
  TF_EXPECT_OK(finder.CreateAllocationMap(hlo_module.get()));

  EXPECT_EQ(finder.tensor_allocation_map.size(), 2);
  EXPECT_EQ(finder.tensor_allocation_map.at(op1), conv);
  EXPECT_EQ(finder.tensor_allocation_map.at(op2), conv);
}


// Check it works for multiple valid destinations

// Check it works for constants
//  auto indices = builder.AddInstruction(
//HloInstruction::CreateConstant(LiteralUtil::CreateR1<int64>({1, 2})));

}
}
}

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

#include "tensorflow/compiler/plugin/poplar/driver/passes/allocation_finder.h"
#include "tensorflow/compiler/plugin/poplar/driver/compiler_annotations.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/forward_allocation.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/inplace_finder.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/while_loop_to_repeat_simplify.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"

#include "tensorflow/compiler/xla/service/hlo_dce.h"
#include "tensorflow/compiler/xla/service/hlo_parser.h"
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
  dimension.set_input_batch_dimension(0);
  dimension.add_input_spatial_dimensions(1);
  dimension.add_input_spatial_dimensions(2);
  dimension.set_input_feature_dimension(3);

  dimension.set_output_batch_dimension(0);
  dimension.add_output_spatial_dimensions(1);
  dimension.add_output_spatial_dimensions(2);
  dimension.set_output_feature_dimension(3);

  dimension.add_kernel_spatial_dimensions(0);
  dimension.add_kernel_spatial_dimensions(1);
  dimension.set_kernel_input_feature_dimension(2);
  dimension.set_kernel_output_feature_dimension(3);
  return dimension;
}

// Check basic parameter matching
TEST_F(AllocationFinderTest, FindBasicTensorAllocations) {
  std::string hlo = R"(
HloModule top

ENTRY c1 {
  p0 = f16[1,16,16,2] parameter(0)
  p1 = f16[1,16,16,2] parameter(1)
  p2 = f16[3,3,2,4] parameter(2)

  add = f16[1,16,16,2] add(p0, p1)

  conv = f16[1,16,16,4] convolution(p0, p2), window={size=3x3 pad=1_1x1_1}, dim_labels=b01f_01io->b01f

  ROOT t = (f16[1,16,16,4], f16[1,16,16,2]) tuple(conv, add)
}

)";

  auto config = GetModuleConfigForTest();
  config.set_argument_count(3);
  config.set_resource_input_count(2);
  config.set_input_mapping({0, 1, 2});
  config.set_resource_update_to_input_index({0});
  auto module = ParseHloString(hlo, config);
  EXPECT_TRUE(module.ok());
  auto* module0 = module.ValueOrDie().get();

  const auto* root = module0->entry_computation()->root_instruction();
  const auto* conv = root->operand(0);
  const auto* ip0 = conv->operand(0);
  const auto* ip2 = conv->operand(1);

  CompilerAnnotations annotations(module0);

  AllocationFinder finder(annotations);
  EXPECT_TRUE(finder.Run(module0).ValueOrDie());

  EXPECT_EQ(annotations.tensor_allocation_map.size(), 2);

  auto t = annotations.tensor_allocation_map.at(std::make_pair(ip0, 0));
  EXPECT_EQ(t.tgt, conv);
  EXPECT_EQ(t.input_index, 0ll);
  EXPECT_EQ(t.forward_path.size(), 0);
  EXPECT_EQ(t.backward_path.size(), 1);
  EXPECT_EQ(t.backward_path[0], ip0);

  t = annotations.tensor_allocation_map.at(std::make_pair(ip2, 0));
  EXPECT_EQ(t.tgt, conv);
  EXPECT_EQ(t.input_index, 1ll);
  EXPECT_EQ(t.forward_path.size(), 0);
  EXPECT_EQ(t.backward_path.size(), 1);
  EXPECT_EQ(t.backward_path[0], ip2);
}

// Check it goes through call sites
TEST_F(AllocationFinderTest, FindSubCompTensorAllocations) {
  Shape input_shape = ShapeUtil::MakeShape(F32, {1, 10, 10, 2});
  Shape weight_shape = ShapeUtil::MakeShape(F32, {3, 3, 2, 1});

  Shape conv_shape =
      ShapeInference::InferConvolveShape(
          input_shape, weight_shape, /*feature_group_count=*/1,
          /*batch_group_count*/ 1, GetConv1Window(), GetConvDimensions())
          .ConsumeValueOrDie();

  /* Create convolution sub-computation */
  auto builder_sub = HloComputation::Builder(TestName());
  auto op0_sub = builder_sub.AddInstruction(
      HloInstruction::CreateParameter(0, input_shape, "input"));
  auto op1_sub = builder_sub.AddInstruction(
      HloInstruction::CreateParameter(1, weight_shape, "weights"));

  auto conv = builder_sub.AddInstruction(HloInstruction::CreateConvolve(
      conv_shape, op0_sub, op1_sub, /*feature_group_count=*/1,
      /*batch_group_count*/ 1, GetConv1Window(), GetConvDimensions(),
      DefaultPrecisionConfig(2)));

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

  auto call = builder_main.AddInstruction(HloInstruction::CreateCall(
      conv_shape, {op1, op2}, computation_sub.get()));

  builder_main.AddInstruction(HloInstruction::CreateTuple({add, call}));

  auto computation_main = builder_main.Build();

  auto hlo_module = CreateNewVerifiedModule();
  hlo_module->AddEmbeddedComputation(std::move(computation_sub));
  hlo_module->AddEntryComputation(std::move(computation_main));

  CompilerAnnotations annotations(hlo_module.get());

  AllocationFinder finder(annotations);
  EXPECT_TRUE(finder.Run(hlo_module.get()).ValueOrDie());

  const HloInstruction* c_conv = conv;

  EXPECT_EQ(annotations.tensor_allocation_map.size(), 4);
  auto t = annotations.tensor_allocation_map.at(std::make_pair(op1, 0));
  EXPECT_EQ(t.tgt, c_conv);
  EXPECT_EQ(t.input_index, 0ll);
  EXPECT_EQ(t.forward_path.size(), 0);
  EXPECT_EQ(t.backward_path.size(), 2);

  t = annotations.tensor_allocation_map.at(std::make_pair(op2, 0));
  EXPECT_EQ(t.tgt, c_conv);
  EXPECT_EQ(t.input_index, 1ll);
  EXPECT_EQ(t.forward_path.size(), 0);
  EXPECT_EQ(t.backward_path.size(), 2);

  t = annotations.tensor_allocation_map.at(std::make_pair(op0_sub, 0));
  EXPECT_EQ(t.tgt, c_conv);
  EXPECT_EQ(t.input_index, 0ll);
  EXPECT_EQ(t.forward_path.size(), 0);
  EXPECT_EQ(t.backward_path.size(), 1);

  t = annotations.tensor_allocation_map.at(std::make_pair(op1_sub, 0));
  EXPECT_EQ(t.tgt, c_conv);
  EXPECT_EQ(t.input_index, 1ll);
  EXPECT_EQ(t.forward_path.size(), 0);
  EXPECT_EQ(t.backward_path.size(), 1);
}

// Check it works for multiple valid destinations (perferred one first)
TEST_F(AllocationFinderTest, FindMultiCompTensorAllocations1) {
  Shape input_shape = ShapeUtil::MakeShape(F32, {1, 10, 10, 2});
  Shape weight_shape = ShapeUtil::MakeShape(F32, {3, 3, 2, 1});

  Shape conv1_shape =
      ShapeInference::InferConvolveShape(
          input_shape, weight_shape, /*feature_group_count=*/1,
          /*batch_group_count*/ 1, GetConv1Window(), GetConvDimensions())
          .ConsumeValueOrDie();

  Shape conv2_shape =
      ShapeInference::InferConvolveShape(
          input_shape, weight_shape, /*feature_group_count=*/1,
          /*batch_group_count*/ 1, GetConv2Window(), GetConvDimensions())
          .ConsumeValueOrDie();

  /* Create convolution sub-computation 1 */
  auto builder_sub1 = HloComputation::Builder(TestName());
  auto op0_sub1 = builder_sub1.AddInstruction(
      HloInstruction::CreateParameter(0, input_shape, "input"));
  auto op1_sub1 = builder_sub1.AddInstruction(
      HloInstruction::CreateParameter(1, weight_shape, "weights"));

  auto conv1 = builder_sub1.AddInstruction(HloInstruction::CreateConvolve(
      conv1_shape, op0_sub1, op1_sub1, /*feature_group_count=*/1,
      /*batch_group_count*/ 1, GetConv1Window(), GetConvDimensions(),
      DefaultPrecisionConfig(2)));

  auto computation_sub1 = builder_sub1.Build();

  /* Create convolution sub-computation 2 */
  auto builder_sub2 = HloComputation::Builder(TestName());
  auto op0_sub2 = builder_sub2.AddInstruction(
      HloInstruction::CreateParameter(0, input_shape, "input"));
  auto op1_sub2 = builder_sub2.AddInstruction(
      HloInstruction::CreateParameter(1, weight_shape, "weights"));

  auto conv2 = builder_sub2.AddInstruction(HloInstruction::CreateConvolve(
      conv2_shape, op0_sub2, op1_sub2, /*feature_group_count=*/1,
      /*batch_group_count*/ 1, GetConv2Window(), GetConvDimensions(),
      DefaultPrecisionConfig(2)));

  auto computation_sub2 = builder_sub2.Build();

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

  auto call1 = builder_main.AddInstruction(HloInstruction::CreateCall(
      conv1_shape, {op1, op2}, computation_sub1.get()));

  auto call2 = builder_main.AddInstruction(HloInstruction::CreateCall(
      conv2_shape, {op1, op2}, computation_sub2.get()));

  builder_main.AddInstruction(HloInstruction::CreateTuple({add, call1, call2}));

  auto computation_main = builder_main.Build();

  auto hlo_module = CreateNewVerifiedModule();
  hlo_module->AddEmbeddedComputation(std::move(computation_sub1));
  hlo_module->AddEmbeddedComputation(std::move(computation_sub2));
  hlo_module->AddEntryComputation(std::move(computation_main));

  CompilerAnnotations annotations(hlo_module.get());
  annotations.classification_map[conv1] = ConvClassificationType::FORWARD;
  annotations.classification_map[conv2] =
      ConvClassificationType::BACKPROP_INPUT;

  AllocationFinder finder(annotations);
  EXPECT_TRUE(finder.Run(hlo_module.get()).ValueOrDie());

  const HloInstruction* c_conv1 = conv1;
  const HloInstruction* c_conv2 = conv2;

  EXPECT_EQ(annotations.tensor_allocation_map.size(), 6);
  auto t = annotations.tensor_allocation_map.at(std::make_pair(op1, 0));
  EXPECT_EQ(t.tgt, c_conv1);
  EXPECT_EQ(t.input_index, 0ll);
  EXPECT_EQ(t.forward_path.size(), 0);
  EXPECT_EQ(t.backward_path.size(), 2);

  t = annotations.tensor_allocation_map.at(std::make_pair(op2, 0));
  EXPECT_EQ(t.tgt, c_conv1);
  EXPECT_EQ(t.input_index, 1ll);
  EXPECT_EQ(t.forward_path.size(), 0);
  EXPECT_EQ(t.backward_path.size(), 2);

  t = annotations.tensor_allocation_map.at(std::make_pair(op0_sub1, 0));
  EXPECT_EQ(t.tgt, c_conv1);
  EXPECT_EQ(t.input_index, 0ll);
  EXPECT_EQ(t.forward_path.size(), 0);
  EXPECT_EQ(t.backward_path.size(), 1);

  t = annotations.tensor_allocation_map.at(std::make_pair(op1_sub1, 0));
  EXPECT_EQ(t.tgt, c_conv1);
  EXPECT_EQ(t.input_index, 1ll);
  EXPECT_EQ(t.forward_path.size(), 0);
  EXPECT_EQ(t.backward_path.size(), 1);

  t = annotations.tensor_allocation_map.at(std::make_pair(op0_sub2, 0));
  EXPECT_EQ(t.tgt, c_conv2);
  EXPECT_EQ(t.input_index, 0ll);
  EXPECT_EQ(t.forward_path.size(), 0);
  EXPECT_EQ(t.backward_path.size(), 1);

  t = annotations.tensor_allocation_map.at(std::make_pair(op1_sub2, 0));
  EXPECT_EQ(t.tgt, c_conv2);
  EXPECT_EQ(t.input_index, 1ll);
  EXPECT_EQ(t.forward_path.size(), 0);
  EXPECT_EQ(t.backward_path.size(), 1);
}

// Check it works for multiple valid destinations (perferred one second)
TEST_F(AllocationFinderTest, FindMultiCompTensorAllocations2) {
  Shape input_shape = ShapeUtil::MakeShape(F32, {1, 10, 10, 2});
  Shape weight_shape = ShapeUtil::MakeShape(F32, {3, 3, 2, 1});

  Shape conv1_shape =
      ShapeInference::InferConvolveShape(
          input_shape, weight_shape, /*feature_group_count=*/1,
          /*batch_group_count*/ 1, GetConv1Window(), GetConvDimensions())
          .ConsumeValueOrDie();

  Shape conv2_shape =
      ShapeInference::InferConvolveShape(
          input_shape, weight_shape, /*feature_group_count=*/1,
          /*batch_group_count*/ 1, GetConv2Window(), GetConvDimensions())
          .ConsumeValueOrDie();

  /* Create convolution sub-computation 1 */
  auto builder_sub1 = HloComputation::Builder(TestName());
  auto op0_sub1 = builder_sub1.AddInstruction(
      HloInstruction::CreateParameter(0, input_shape, "input"));
  auto op1_sub1 = builder_sub1.AddInstruction(
      HloInstruction::CreateParameter(1, weight_shape, "weights"));

  auto conv1 = builder_sub1.AddInstruction(HloInstruction::CreateConvolve(
      conv1_shape, op0_sub1, op1_sub1, /*feature_group_count=*/1,
      /*batch_group_count*/ 1, GetConv1Window(), GetConvDimensions(),
      DefaultPrecisionConfig(2)));

  auto computation_sub1 = builder_sub1.Build();

  /* Create convolution sub-computation 2 */
  auto builder_sub2 = HloComputation::Builder(TestName());
  auto op0_sub2 = builder_sub2.AddInstruction(
      HloInstruction::CreateParameter(0, input_shape, "input"));
  auto op1_sub2 = builder_sub2.AddInstruction(
      HloInstruction::CreateParameter(1, weight_shape, "weights"));

  auto conv2 = builder_sub2.AddInstruction(HloInstruction::CreateConvolve(
      conv2_shape, op0_sub2, op1_sub2, /*feature_group_count=*/1,
      /*batch_group_count*/ 1, GetConv2Window(), GetConvDimensions(),
      DefaultPrecisionConfig(2)));

  auto computation_sub2 = builder_sub2.Build();

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

  auto call1 = builder_main.AddInstruction(HloInstruction::CreateCall(
      conv1_shape, {op1, op2}, computation_sub1.get()));

  auto call2 = builder_main.AddInstruction(HloInstruction::CreateCall(
      conv2_shape, {op1, op2}, computation_sub2.get()));

  builder_main.AddInstruction(HloInstruction::CreateTuple({add, call1, call2}));

  auto computation_main = builder_main.Build();

  auto hlo_module = CreateNewVerifiedModule();
  hlo_module->AddEmbeddedComputation(std::move(computation_sub1));
  hlo_module->AddEmbeddedComputation(std::move(computation_sub2));
  hlo_module->AddEntryComputation(std::move(computation_main));

  CompilerAnnotations annotations(hlo_module.get());
  annotations.classification_map[conv1] =
      ConvClassificationType::BACKPROP_INPUT;
  annotations.classification_map[conv2] = ConvClassificationType::FORWARD;

  AllocationFinder finder(annotations);
  EXPECT_TRUE(finder.Run(hlo_module.get()).ValueOrDie());

  const HloInstruction* c_conv1 = conv1;
  const HloInstruction* c_conv2 = conv2;

  EXPECT_EQ(annotations.tensor_allocation_map.size(), 6);

  auto t = annotations.tensor_allocation_map.at(std::make_pair(op1, 0));
  EXPECT_EQ(t.tgt, c_conv2);
  EXPECT_EQ(t.input_index, 0ll);
  EXPECT_EQ(t.forward_path.size(), 0);
  EXPECT_EQ(t.backward_path.size(), 2);

  t = annotations.tensor_allocation_map.at(std::make_pair(op2, 0));
  EXPECT_EQ(t.tgt, c_conv2);
  EXPECT_EQ(t.input_index, 1ll);
  EXPECT_EQ(t.forward_path.size(), 0);
  EXPECT_EQ(t.backward_path.size(), 2);

  t = annotations.tensor_allocation_map.at(std::make_pair(op0_sub1, 0));
  EXPECT_EQ(t.tgt, c_conv1);
  EXPECT_EQ(t.input_index, 0ll);
  EXPECT_EQ(t.forward_path.size(), 0);
  EXPECT_EQ(t.backward_path.size(), 1);

  t = annotations.tensor_allocation_map.at(std::make_pair(op1_sub1, 0));
  EXPECT_EQ(t.tgt, c_conv1);
  EXPECT_EQ(t.input_index, 1ll);
  EXPECT_EQ(t.forward_path.size(), 0);
  EXPECT_EQ(t.backward_path.size(), 1);

  t = annotations.tensor_allocation_map.at(std::make_pair(op0_sub2, 0));
  EXPECT_EQ(t.tgt, c_conv2);
  EXPECT_EQ(t.input_index, 0ll);
  EXPECT_EQ(t.forward_path.size(), 0);
  EXPECT_EQ(t.backward_path.size(), 1);

  t = annotations.tensor_allocation_map.at(std::make_pair(op1_sub2, 0));
  EXPECT_EQ(t.tgt, c_conv2);
  EXPECT_EQ(t.input_index, 1ll);
  EXPECT_EQ(t.forward_path.size(), 0);
  EXPECT_EQ(t.backward_path.size(), 1);
}

// Check it works for constants
TEST_F(AllocationFinderTest, FindConstantTensorAllocations) {
  std::string hlo = R"(
HloModule top

ENTRY c1 {
  p0 = f16[1,16,16,2] parameter(0)
  p1 = f16[1,16,16,2] parameter(1)
  p2 = f16[1,1,2,4] constant({{{{1,0,0,0},{1,0,0,0}}}})

  add = f16[1,16,16,2] add(p0, p1)

  conv = f16[1,16,16,4] convolution(p0, p2), window={size=1x1}, dim_labels=b01f_01io->b01f

  ROOT t = (f16[1,16,16,4], f16[1,16,16,2]) tuple(conv, add)
}

)";

  auto config = GetModuleConfigForTest();
  config.set_argument_count(2);
  config.set_resource_input_count(2);
  config.set_input_mapping({0, 1});
  config.set_resource_update_to_input_index({0});
  auto module = ParseHloString(hlo, config);
  EXPECT_TRUE(module.ok());
  auto* module0 = module.ValueOrDie().get();

  const auto* root = module0->entry_computation()->root_instruction();
  const auto* conv = root->operand(0);
  const auto* ip0 = conv->operand(0);
  const auto* ip2 = conv->operand(1);

  CompilerAnnotations annotations(module0);

  AllocationFinder finder(annotations);
  EXPECT_TRUE(finder.Run(module0).ValueOrDie());

  EXPECT_EQ(annotations.tensor_allocation_map.size(), 2);

  auto t = annotations.tensor_allocation_map.at(std::make_pair(ip0, 0));
  EXPECT_EQ(t.tgt, conv);
  EXPECT_EQ(t.input_index, 0ll);
  EXPECT_EQ(t.forward_path.size(), 0);
  EXPECT_EQ(t.backward_path.size(), 1);

  t = annotations.tensor_allocation_map.at(std::make_pair(ip2, 0));
  EXPECT_EQ(t.tgt, conv);
  EXPECT_EQ(t.input_index, 1ll);
  EXPECT_EQ(t.forward_path.size(), 0);
  EXPECT_EQ(t.backward_path.size(), 1);
}

// Check it goes through Tuple/Detuple pairs
TEST_F(AllocationFinderTest, CanTraverseTuples) {
  auto hlo_module = CreateNewVerifiedModule();

  Shape lhs_shape = ShapeUtil::MakeShape(F32, {2});
  Shape rhs_shape = ShapeUtil::MakeShape(F32, {2, 2});
  Shape tuple_shape = ShapeUtil::MakeTupleShape({lhs_shape, rhs_shape});

  auto b = HloComputation::Builder(TestName());
  auto in =
      b.AddInstruction(HloInstruction::CreateParameter(0, lhs_shape, "in"));
  auto w =
      b.AddInstruction(HloInstruction::CreateParameter(1, rhs_shape, "weight"));

  auto tuple = b.AddInstruction(HloInstruction::CreateTuple({in, w}));

  auto in1 = b.AddInstruction(
      HloInstruction::CreateGetTupleElement(lhs_shape, tuple, 0));
  auto w1 = b.AddInstruction(
      HloInstruction::CreateGetTupleElement(rhs_shape, tuple, 1));

  DotDimensionNumbers dot_dnums;
  dot_dnums.add_lhs_contracting_dimensions(0);
  dot_dnums.add_rhs_contracting_dimensions(0);
  auto dot_inst = b.AddInstruction(HloInstruction::CreateDot(
      lhs_shape, in1, w1, dot_dnums, DefaultPrecisionConfig(2)));

  hlo_module->AddEntryComputation(b.Build());

  CompilerAnnotations annotations(hlo_module.get());

  AllocationFinder finder(annotations);
  EXPECT_TRUE(finder.Run(hlo_module.get()).ValueOrDie());

  const HloInstruction* dot = dot_inst;

  EXPECT_EQ(annotations.tensor_allocation_map.size(), 2);

  auto t = annotations.tensor_allocation_map.at(std::make_pair(in, 0));
  EXPECT_EQ(t.tgt, dot);
  EXPECT_EQ(t.input_index, 0ll);
  EXPECT_EQ(t.forward_path.size(), 0);
  EXPECT_EQ(t.backward_path.size(), 3);

  t = annotations.tensor_allocation_map.at(std::make_pair(w, 0));
  EXPECT_EQ(t.tgt, dot);
  EXPECT_EQ(t.input_index, 1ll);
  EXPECT_EQ(t.forward_path.size(), 0);
  EXPECT_EQ(t.backward_path.size(), 3);
}

// Check it can start from tuple subshapes
TEST_F(AllocationFinderTest, CanStartOnTuples) {
  auto hlo_module = CreateNewVerifiedModule();

  Shape lhs_shape = ShapeUtil::MakeShape(F32, {2});
  Shape rhs_shape = ShapeUtil::MakeShape(F32, {2, 2});
  Shape tuple_shape = ShapeUtil::MakeTupleShape({lhs_shape, rhs_shape});

  auto b = HloComputation::Builder(TestName());
  auto in = b.AddInstruction(
      HloInstruction::CreateParameter(0, tuple_shape, "tuple"));

  auto in1 =
      b.AddInstruction(HloInstruction::CreateGetTupleElement(lhs_shape, in, 0));
  auto w1 =
      b.AddInstruction(HloInstruction::CreateGetTupleElement(rhs_shape, in, 1));

  DotDimensionNumbers dot_dnums;
  dot_dnums.add_lhs_contracting_dimensions(0);
  dot_dnums.add_rhs_contracting_dimensions(0);
  auto dot_inst = b.AddInstruction(HloInstruction::CreateDot(
      lhs_shape, in1, w1, dot_dnums, DefaultPrecisionConfig(2)));

  hlo_module->AddEntryComputation(b.Build());

  CompilerAnnotations annotations(hlo_module.get());

  AllocationFinder finder(annotations);
  EXPECT_TRUE(finder.Run(hlo_module.get()).ValueOrDie());

  const HloInstruction* dot = dot_inst;

  EXPECT_EQ(annotations.tensor_allocation_map.size(), 2);

  auto t = annotations.tensor_allocation_map.at(std::make_pair(in, 0));
  EXPECT_EQ(t.tgt, dot);
  EXPECT_EQ(t.input_index, 0ll);
  EXPECT_EQ(t.forward_path.size(), 0);
  EXPECT_EQ(t.backward_path.size(), 2);

  t = annotations.tensor_allocation_map.at(std::make_pair(in, 1));
  EXPECT_EQ(t.tgt, dot);
  EXPECT_EQ(t.input_index, 1ll);
  EXPECT_EQ(t.forward_path.size(), 0);
  EXPECT_EQ(t.backward_path.size(), 2);
}

// Check it goes through while instructions
TEST_F(AllocationFinderTest, FindWhileTensorAllocations) {
  auto hlo_module = CreateNewVerifiedModule();

  Shape counter_shape = ShapeUtil::MakeShape(S32, {});
  Shape input_shape = ShapeUtil::MakeShape(F32, {2});
  Shape weight_shape = ShapeUtil::MakeShape(F32, {2, 2});
  Shape tuple_shape =
      ShapeUtil::MakeTupleShape({counter_shape, input_shape, weight_shape});

  const HloInstruction* dot_inst;
  const HloInstruction* body_param;

  /* Create while condition */
  HloComputation* comp_cond;
  {
    auto builder_cond = HloComputation::Builder(TestName());
    auto tuple = builder_cond.AddInstruction(
        HloInstruction::CreateParameter(0, tuple_shape, "cond_tuple"));
    auto limit = builder_cond.AddInstruction(
        HloInstruction::CreateConstant(LiteralUtil::CreateR0<int32>(10)));
    auto c = builder_cond.AddInstruction(HloInstruction::CreateGetTupleElement(
        ShapeUtil::MakeShape(S32, {}), tuple, 0));
    builder_cond.AddInstruction(HloInstruction::CreateBinary(
        ShapeUtil::MakeShape(PRED, {}), HloOpcode::kLt, c, limit));

    comp_cond = hlo_module->AddEmbeddedComputation(builder_cond.Build());
  }

  /* Create while body */
  HloComputation* comp_body;
  {
    auto builder_body = HloComputation::Builder(TestName());
    auto tuple = builder_body.AddInstruction(
        HloInstruction::CreateParameter(0, tuple_shape, "body_tuple"));
    auto c = builder_body.AddInstruction(
        HloInstruction::CreateGetTupleElement(counter_shape, tuple, 0));
    auto in = builder_body.AddInstruction(
        HloInstruction::CreateGetTupleElement(input_shape, tuple, 1));
    auto w = builder_body.AddInstruction(
        HloInstruction::CreateGetTupleElement(weight_shape, tuple, 2));
    auto one = builder_body.AddInstruction(
        HloInstruction::CreateConstant(LiteralUtil::CreateR0<int32>(1)));
    auto new_c = builder_body.AddInstruction(
        HloInstruction::CreateBinary(c->shape(), HloOpcode::kAdd, c, one));

    DotDimensionNumbers dot_dnums;
    dot_dnums.add_lhs_contracting_dimensions(0);
    dot_dnums.add_rhs_contracting_dimensions(0);
    auto new_in = builder_body.AddInstruction(HloInstruction::CreateDot(
        input_shape, in, w, dot_dnums, DefaultPrecisionConfig(2)));

    dot_inst = new_in;
    body_param = tuple;

    builder_body.AddInstruction(
        HloInstruction::CreateTuple({new_c, new_in, w}));

    comp_body = hlo_module->AddEmbeddedComputation(builder_body.Build());
  }

  /* Create main computation */
  auto builder_main = HloComputation::Builder(TestName());
  auto c = builder_main.AddInstruction(
      HloInstruction::CreateParameter(0, counter_shape, "counter"));
  auto in = builder_main.AddInstruction(
      HloInstruction::CreateParameter(1, input_shape, "in"));
  auto w = builder_main.AddInstruction(
      HloInstruction::CreateParameter(2, weight_shape, "weight"));

  auto init =
      builder_main.AddInstruction(HloInstruction::CreateTuple({c, in, w}));

  auto main = builder_main.AddInstruction(
      HloInstruction::CreateWhile(tuple_shape, comp_cond, comp_body, init));

  builder_main.AddInstruction(HloInstruction::CreateTuple({main}));

  hlo_module->AddEntryComputation(builder_main.Build());

  CompilerAnnotations annotations(hlo_module.get());

  AllocationFinder finder(annotations);
  EXPECT_TRUE(finder.Run(hlo_module.get()).ValueOrDie());

  EXPECT_EQ(annotations.tensor_allocation_map.size(), 4);

  auto t = annotations.tensor_allocation_map.at(std::make_pair(in, 0));
  EXPECT_EQ(t.tgt, dot_inst);
  EXPECT_EQ(t.input_index, 0ll);
  EXPECT_EQ(t.forward_path.size(), 0);
  EXPECT_EQ(t.backward_path.size(), 4);

  t = annotations.tensor_allocation_map.at(std::make_pair(w, 0));
  EXPECT_EQ(t.tgt, dot_inst);
  EXPECT_EQ(t.input_index, 1ll);
  EXPECT_EQ(t.forward_path.size(), 0);
  EXPECT_EQ(t.backward_path.size(), 4);

  t = annotations.tensor_allocation_map.at(std::make_pair(body_param, 1));
  EXPECT_EQ(t.tgt, dot_inst);
  EXPECT_EQ(t.input_index, 0ll);
  EXPECT_EQ(t.forward_path.size(), 0);
  EXPECT_EQ(t.backward_path.size(), 2);

  t = annotations.tensor_allocation_map.at(std::make_pair(body_param, 2));
  EXPECT_EQ(t.tgt, dot_inst);
  EXPECT_EQ(t.input_index, 1ll);
  EXPECT_EQ(t.forward_path.size(), 0);
  EXPECT_EQ(t.backward_path.size(), 2);
}

// Check it goes through repeat instructions
TEST_F(AllocationFinderTest, FindRepeatTensorAllocations) {
  auto hlo_module = CreateNewVerifiedModule();

  Shape counter_shape = ShapeUtil::MakeShape(S32, {});
  Shape input_shape = ShapeUtil::MakeShape(F32, {2});
  Shape weight_shape = ShapeUtil::MakeShape(F32, {2, 2});
  Shape tuple_shape =
      ShapeUtil::MakeTupleShape({counter_shape, input_shape, weight_shape});

  /* Create while condition */
  HloComputation* comp_cond;
  {
    auto builder_cond = HloComputation::Builder(TestName());
    auto tuple = builder_cond.AddInstruction(
        HloInstruction::CreateParameter(0, tuple_shape, "cond_tuple"));
    auto limit = builder_cond.AddInstruction(
        HloInstruction::CreateConstant(LiteralUtil::CreateR0<int32>(10)));
    auto c = builder_cond.AddInstruction(HloInstruction::CreateGetTupleElement(
        ShapeUtil::MakeShape(S32, {}), tuple, 0));
    builder_cond.AddInstruction(HloInstruction::CreateBinary(
        ShapeUtil::MakeShape(PRED, {}), HloOpcode::kLt, c, limit));

    comp_cond = hlo_module->AddEmbeddedComputation(builder_cond.Build());
  }

  /* Create while body */
  HloComputation* comp_body;
  {
    auto builder_body = HloComputation::Builder(TestName());
    auto tuple = builder_body.AddInstruction(
        HloInstruction::CreateParameter(0, tuple_shape, "body_tuple"));
    auto c = builder_body.AddInstruction(
        HloInstruction::CreateGetTupleElement(counter_shape, tuple, 0));
    auto in = builder_body.AddInstruction(
        HloInstruction::CreateGetTupleElement(input_shape, tuple, 1));
    auto w = builder_body.AddInstruction(
        HloInstruction::CreateGetTupleElement(weight_shape, tuple, 2));
    auto one = builder_body.AddInstruction(
        HloInstruction::CreateConstant(LiteralUtil::CreateR0<int32>(1)));
    auto new_c = builder_body.AddInstruction(
        HloInstruction::CreateBinary(c->shape(), HloOpcode::kAdd, c, one));

    DotDimensionNumbers dot_dnums;
    dot_dnums.add_lhs_contracting_dimensions(0);
    dot_dnums.add_rhs_contracting_dimensions(1);
    auto new_in = builder_body.AddInstruction(HloInstruction::CreateDot(
        input_shape, in, w, dot_dnums, DefaultPrecisionConfig(2)));

    builder_body.AddInstruction(
        HloInstruction::CreateTuple({new_c, new_in, w}));

    comp_body = hlo_module->AddEmbeddedComputation(builder_body.Build());
  }

  /* Create main computation */
  auto builder_main = HloComputation::Builder(TestName());
  auto c = builder_main.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<int32>(0)));
  auto in = builder_main.AddInstruction(
      HloInstruction::CreateParameter(0, input_shape, "in"));
  auto w = builder_main.AddInstruction(
      HloInstruction::CreateParameter(1, weight_shape, "weight"));

  auto init =
      builder_main.AddInstruction(HloInstruction::CreateTuple({c, in, w}));

  builder_main.AddInstruction(
      HloInstruction::CreateWhile(tuple_shape, comp_cond, comp_body, init));

  hlo_module->AddEntryComputation(builder_main.Build());

  // Simplify the while loop to a repeat (need to also run DCE)
  WhileLoopToRepeatSimplify wltrs;
  EXPECT_TRUE(wltrs.Run(hlo_module.get()).ValueOrDie());
  HloDCE hdce;
  EXPECT_TRUE(hdce.Run(hlo_module.get()).ValueOrDie());

  CompilerAnnotations annotations(hlo_module.get());

  AllocationFinder finder(annotations);
  EXPECT_TRUE(finder.Run(hlo_module.get()).ValueOrDie());

  EXPECT_EQ(annotations.tensor_allocation_map.size(), 4);

  // Get the dot and tuple instruction from the new repeat body.
  const HloComputation* repeat_body =
      hlo_module->entry_computation()->root_instruction()->to_apply();
  const HloInstruction* body_param = repeat_body->parameter_instruction(0);
  const HloInstruction* dot_inst = repeat_body->root_instruction()->operand(1);

  auto t = annotations.tensor_allocation_map.at(std::make_pair(in, 0));
  EXPECT_EQ(t.tgt, dot_inst);
  EXPECT_EQ(t.input_index, 0ll);
  EXPECT_EQ(t.forward_path.size(), 0);
  EXPECT_EQ(t.backward_path.size(), 4);

  t = annotations.tensor_allocation_map.at(std::make_pair(w, 0));
  EXPECT_EQ(t.tgt, dot_inst);
  EXPECT_EQ(t.input_index, 1ll);
  EXPECT_EQ(t.forward_path.size(), 0);
  EXPECT_EQ(t.backward_path.size(), 4);

  t = annotations.tensor_allocation_map.at(std::make_pair(body_param, 1));
  EXPECT_EQ(t.tgt, dot_inst);
  EXPECT_EQ(t.input_index, 0ll);
  EXPECT_EQ(t.forward_path.size(), 0);
  EXPECT_EQ(t.backward_path.size(), 2);

  t = annotations.tensor_allocation_map.at(std::make_pair(body_param, 2));
  EXPECT_EQ(t.tgt, dot_inst);
  EXPECT_EQ(t.input_index, 1ll);
  EXPECT_EQ(t.forward_path.size(), 0);
  EXPECT_EQ(t.backward_path.size(), 2);
}

// Check basic parameter matching
TEST_F(AllocationFinderTest, TraverseDimShuffleAndReshapeAllocations) {
  std::string hlo = R"(
HloModule top

ENTRY c1 {
  p0 = f16[1,16,16,2] parameter(0)
  p1 = f16[3,3,4,2] parameter(1)

  p1_t = f16[3,3,2,4] transpose(p1), dimensions={2,3}

  conv = f16[1,16,16,4] convolution(p0, p1_t), window={size=3x3 pad=1_1x1_1}, dim_labels=b01f_01io->b01f

  ROOT t = (f16[1,16,16,4]) tuple(conv)
}

)";

  auto config = GetModuleConfigForTest();
  config.set_argument_count(2);
  config.set_resource_input_count(0);
  config.set_input_mapping({0, 1});
  config.set_resource_update_to_input_index({0});
  auto module = ParseHloString(hlo, config);
  EXPECT_TRUE(module.ok());
  auto* module0 = module.ValueOrDie().get();

  const auto* root = module0->entry_computation()->root_instruction();
  const auto* conv = root->operand(0);
  const auto* ip0 = conv->operand(0);
  const auto* trans = conv->operand(1);
  const auto* ip1 = trans->operand(0);

  CompilerAnnotations annotations(module0);

  AllocationFinder finder(annotations);
  EXPECT_TRUE(finder.Run(module0).ValueOrDie());

  EXPECT_EQ(annotations.tensor_allocation_map.size(), 2);

  auto t = annotations.tensor_allocation_map.at(std::make_pair(ip0, 0));
  EXPECT_EQ(t.tgt, conv);
  EXPECT_EQ(t.input_index, 0ll);
  EXPECT_EQ(t.backward_path.size(), 1);
  EXPECT_EQ(t.backward_path[0], ip0);

  t = annotations.tensor_allocation_map.at(std::make_pair(ip1, 0));
  EXPECT_EQ(t.tgt, conv);
  EXPECT_EQ(t.input_index, 1ll);
  EXPECT_EQ(t.backward_path.size(), 2);
  EXPECT_EQ(t.backward_path[0], ip1);
  EXPECT_EQ(t.backward_path[1], trans);
}

// Check it goes through call sites
TEST_F(AllocationFinderTest, FindDoesntTraceThroughInvalidCalls) {
  Shape input_shape = ShapeUtil::MakeShape(F32, {1, 10, 10, 2});
  Shape half_shape = ShapeUtil::MakeShape(F32, {1, 10, 10, 1});
  Shape weight_shape = ShapeUtil::MakeShape(F32, {3, 3, 2, 1});

  Shape conv_shape =
      ShapeInference::InferConvolveShape(
          input_shape, weight_shape, /*feature_group_count=*/1,
          /*batch_group_count*/ 1, GetConv1Window(), GetConvDimensions())
          .ConsumeValueOrDie();

  /* Create sub-computation which contains an unacceptable op */
  auto builder_sub = HloComputation::Builder(TestName());
  HloInstruction* op0_sub = builder_sub.AddInstruction(
      HloInstruction::CreateParameter(0, half_shape, "input"));
  HloInstruction* op1_sub = builder_sub.AddInstruction(
      HloInstruction::CreateConstant(Literal::CreateFromShape(half_shape)));
  HloInstruction* op2_sub = builder_sub.AddInstruction(
      HloInstruction::CreateConcatenate(input_shape, {op0_sub, op1_sub}, 3));
  auto computation_sub = builder_sub.Build();

  /* Create main computation */
  auto builder_main = HloComputation::Builder(TestName());
  HloInstruction* op0 = builder_main.AddInstruction(
      HloInstruction::CreateParameter(0, half_shape, "op0"));
  HloInstruction* op1 = builder_main.AddInstruction(
      HloInstruction::CreateParameter(1, weight_shape, "op1"));
  HloInstruction* call = builder_main.AddInstruction(
      HloInstruction::CreateCall(input_shape, {op0}, computation_sub.get()));
  HloInstruction* conv =
      builder_main.AddInstruction(HloInstruction::CreateConvolve(
          conv_shape, call, op1, /*feature_group_count=*/1,
          /*batch_group_count*/ 1, GetConv1Window(), GetConvDimensions(),
          DefaultPrecisionConfig(2)));

  builder_main.AddInstruction(HloInstruction::CreateTuple({conv}));

  auto computation_main = builder_main.Build();

  auto hlo_module = CreateNewVerifiedModule();
  hlo_module->AddEmbeddedComputation(std::move(computation_sub));
  hlo_module->AddEntryComputation(std::move(computation_main));

  CompilerAnnotations annotations(hlo_module.get());

  AllocationFinder finder(annotations);
  EXPECT_TRUE(finder.Run(hlo_module.get()).ValueOrDie());

  EXPECT_EQ(annotations.tensor_allocation_map.size(), 1);
  auto t1 = annotations.tensor_allocation_map.at(std::make_pair(op1, 0));
  EXPECT_EQ(t1.tgt, conv);
  EXPECT_EQ(t1.input_index, 1ll);
  EXPECT_EQ(t1.backward_path.size(), 1);
}

TEST_F(AllocationFinderTest, BiasAdd1) {
  std::string hlo = R"(
HloModule top

_pop_op_conv_biasadd {
  arg_0 = f16[1,16,16,4] parameter(0)
  arg_1 = f16[4] parameter(1)
  bcast = f16[1,16,16,4] broadcast(arg_1), dimensions={3}
  ROOT %add = f16[1,16,16,4] add(arg_0, bcast)
}

ENTRY c1 {
  p0 = f16[1,16,16,2] parameter(0)
  p1 = f16[3,3,2,4] parameter(1)
  p2 = f16[4] parameter(2)

  conv = f16[1,16,16,4] convolution(p0, p1), window={size=3x3 pad=1_1x1_1}, dim_labels=b01f_01io->b01f
  call = f16[1,16,16,64] fusion(conv, p2), kind=kCustom, calls=_pop_op_conv_biasadd

  ROOT t = (f16[1,16,16,4]) tuple(%call)
}

)";

  auto config = GetModuleConfigForTest();
  config.set_argument_count(3);
  config.set_resource_input_count(3);
  config.set_input_mapping({0, 1, 2});
  config.set_resource_update_to_input_index({0});
  auto module = ParseHloString(hlo, config);
  EXPECT_TRUE(module.ok());
  auto* module0 = module.ValueOrDie().get();

  const auto* root = module0->entry_computation()->root_instruction();
  const auto* call = root->operand(0);
  const auto* conv = call->operand(0);
  const auto* ip0 = conv->operand(0);
  const auto* ip1 = conv->operand(1);
  const auto* ip2 = call->operand(1);

  CompilerAnnotations annotations(module0);

  AllocationFinder finder(annotations);
  EXPECT_TRUE(finder.Run(module0).ValueOrDie());

  // Will have both of the convolution parameters
  EXPECT_EQ(annotations.tensor_allocation_map.size(), 2);

  auto t = annotations.tensor_allocation_map.at(std::make_pair(ip0, 0));
  EXPECT_EQ(t.tgt, conv);
  EXPECT_EQ(t.input_index, 0);

  t = annotations.tensor_allocation_map.at(std::make_pair(ip1, 0));
  EXPECT_EQ(t.tgt, conv);
  EXPECT_EQ(t.input_index, 1);

  ForwardAllocation fwd_finder(annotations);
  EXPECT_TRUE(fwd_finder.Run(module0).ValueOrDie());

  // We have added one new entry for the bias add
  EXPECT_EQ(annotations.tensor_allocation_map.size(), 3);

  t = annotations.tensor_allocation_map.at(std::make_pair(ip2, 0));
  EXPECT_EQ(t.tgt, call);
  EXPECT_EQ(t.input_index, 1);
  EXPECT_EQ(t.layout, conv);
  EXPECT_EQ(t.layout_output_idx, 0);
}

TEST_F(AllocationFinderTest, CustomCallFindTensorAllocation) {
  std::string hlo = R"(
HloModule top

ENTRY c1 {
  %arg2.3 = f32[3,1,5]{2,1,0} parameter(0)
  %arg1.2 = f32[1,8]{1,0} parameter(1)
  %arg0.1 = f32[1,8]{1,0} parameter(2)
  %arg4.5 = f32[13,32]{1,0} parameter(3)
  %arg3.4 = f32[4,8]{1,0} parameter(4)
  ROOT cc = (f32[3,1,8]{2,1,0}, f32[1,8]{1,0}, f32[1,8]{1,0}, f32[3,6,1,8]{3,2,1,0}) custom-call(f32[3,1,5]{2,1,0} %arg2.3, f32[1,8]{1,0} %arg1.2, f32[1,8]{1,0} %arg0.1, f32[13,32]{1,0} %arg4.5, f32[4,8]{1,0} %arg3.4), custom_call_target="Popnn::LstmLayerFwd", opaque="{\"allocating_indexes\":[4,2,0,3,1],\"is_training\":false,\"layout_dependencies\":{\"keys\":[],\"values\":[]},\"num_channels\":8,\"num_inplace_operands\":0,\"partials_dtype\":\"DT_FLOAT\"}\n", metadata={op_type="PopnnLstmLayer" op_name="ones"}
}

)";

  auto module = ParseHloString(hlo, GetModuleConfigForTest());
  EXPECT_TRUE(module.ok());
  auto* module0 = module.ValueOrDie().get();

  const auto* lstm = module0->entry_computation()->root_instruction();
  const auto* ip0 = lstm->operand(0);
  const auto* ip1 = lstm->operand(1);
  const auto* ip2 = lstm->operand(2);
  const auto* ip3 = lstm->operand(3);
  const auto* ip4 = lstm->operand(4);

  CompilerAnnotations annotations(module0);

  AllocationFinder finder(annotations);
  EXPECT_TRUE(finder.Run(module0).ValueOrDie());

  EXPECT_EQ(annotations.tensor_allocation_map.size(), 5);

  auto t = annotations.tensor_allocation_map.at(std::make_pair(ip0, 0));
  EXPECT_EQ(t.tgt, lstm);
  EXPECT_EQ(t.input_index, 0ll);
  EXPECT_EQ(t.forward_path.size(), 0);
  EXPECT_EQ(t.backward_path.size(), 1);
  EXPECT_EQ(t.backward_path[0], ip0);

  t = annotations.tensor_allocation_map.at(std::make_pair(ip1, 0));
  EXPECT_EQ(t.tgt, lstm);
  EXPECT_EQ(t.input_index, 1ll);
  EXPECT_EQ(t.forward_path.size(), 0);
  EXPECT_EQ(t.backward_path.size(), 1);
  EXPECT_EQ(t.backward_path[0], ip1);

  t = annotations.tensor_allocation_map.at(std::make_pair(ip2, 0));
  EXPECT_EQ(t.tgt, lstm);
  EXPECT_EQ(t.input_index, 2ll);
  EXPECT_EQ(t.forward_path.size(), 0);
  EXPECT_EQ(t.backward_path.size(), 1);
  EXPECT_EQ(t.backward_path[0], ip2);

  t = annotations.tensor_allocation_map.at(std::make_pair(ip3, 0));
  EXPECT_EQ(t.tgt, lstm);
  EXPECT_EQ(t.input_index, 3ll);
  EXPECT_EQ(t.forward_path.size(), 0);
  EXPECT_EQ(t.backward_path.size(), 1);
  EXPECT_EQ(t.backward_path[0], ip3);

  t = annotations.tensor_allocation_map.at(std::make_pair(ip4, 0));
  EXPECT_EQ(t.tgt, lstm);
  EXPECT_EQ(t.input_index, 4ll);
  EXPECT_EQ(t.forward_path.size(), 0);
  EXPECT_EQ(t.backward_path.size(), 1);
  EXPECT_EQ(t.backward_path[0], ip4);
}

TEST_F(AllocationFinderTest, BiasAddAndMultiply) {
  std::string hlo = R"(
HloModule top

_pop_op_conv_biasadd {
  arg_0 = f16[1,16,16,4] parameter(0)
  arg_1 = f16[4] parameter(1)
  bcast = f16[1,16,16,4] broadcast(arg_1), dimensions={3}
  ROOT %add = f16[1,16,16,4] add(arg_0, bcast)
}

_pop_op_conv_biasadd.1 {
  arg_0 = f16[1,16,16,4] parameter(0)
  arg_1 = f16[4] parameter(1)
  bcast = f16[1,16,16,4] broadcast(arg_1), dimensions={3}
  ROOT %add = f16[1,16,16,4] add(arg_0, bcast)
}

ENTRY c1 {
  p0 = f16[1,16,16,2] parameter(0)
  p1 = f16[3,3,2,4] parameter(1)
  p2 = f16[4] parameter(2)
  p3 = f16[4] parameter(3)

  conv = f16[1,16,16,4] convolution(p0, p1), window={size=3x3 pad=1_1x1_1}, dim_labels=b01f_01io->b01f
  call = f16[1,16,16,64] fusion(conv, p2), kind=kCustom, calls=_pop_op_conv_biasadd
  call.1 = f16[1,16,16,64] fusion(call, p3), kind=kCustom, calls=_pop_op_conv_biasadd.1

  ROOT t = (f16[1,16,16,4]) tuple(call.1)
}

)";

  auto config = GetModuleConfigForTest();
  config.set_argument_count(4);
  config.set_resource_input_count(2);
  config.set_input_mapping({0, 1, 2});
  config.set_resource_update_to_input_index({});
  auto module = ParseHloString(hlo, config);
  EXPECT_TRUE(module.ok());
  auto* module0 = module.ValueOrDie().get();

  const auto* root = module0->entry_computation()->root_instruction();
  const auto& call1 = root->operand(0);
  const auto* call = call1->operand(0);
  const auto* conv = call->operand(0);
  const auto* ip0 = conv->operand(0);
  const auto* ip1 = conv->operand(1);
  const auto* ip2 = call->operand(1);
  const auto* ip3 = call1->operand(1);

  CompilerAnnotations annotations(module0);

  AllocationFinder finder(annotations);
  EXPECT_TRUE(finder.Run(module0).ValueOrDie());

  // Will have both of the convolution parameters
  EXPECT_EQ(annotations.tensor_allocation_map.size(), 2);

  auto t = annotations.tensor_allocation_map.at(std::make_pair(ip0, 0));
  EXPECT_EQ(t.tgt, conv);
  EXPECT_EQ(t.input_index, 0);

  t = annotations.tensor_allocation_map.at(std::make_pair(ip1, 0));
  EXPECT_EQ(t.tgt, conv);
  EXPECT_EQ(t.input_index, 1);

  ForwardAllocation fwd_finder(annotations);
  EXPECT_TRUE(fwd_finder.Run(module0).ValueOrDie());

  // We have added two new entries to the map for the 2 bias add ops
  EXPECT_EQ(annotations.tensor_allocation_map.size(), 4);

  t = annotations.tensor_allocation_map.at(std::make_pair(ip2, 0));
  EXPECT_EQ(t.tgt, call);
  EXPECT_EQ(t.input_index, 1);
  EXPECT_EQ(t.layout, conv);
  EXPECT_EQ(t.layout_output_idx, 0);
  EXPECT_EQ(t.forward_path.size(), 0);
  EXPECT_EQ(t.backward_path.size(), 0);

  t = annotations.tensor_allocation_map.at(std::make_pair(ip3, 0));
  EXPECT_EQ(t.tgt, call1);
  EXPECT_EQ(t.input_index, 1);
  EXPECT_EQ(t.layout, conv);
  EXPECT_EQ(t.layout_output_idx, 0);
  EXPECT_EQ(t.forward_path.size(), 1);
  EXPECT_EQ(t.forward_path[0], call);
  EXPECT_EQ(t.backward_path.size(), 0);
}

TEST_F(AllocationFinderTest, BiasAddWithPath) {
  std::string hlo = R"(
HloModule top

_pop_op_conv_biasadd {
  %arg_0 = f16[1,16,16,4] parameter(0)
  %arg_1 = f16[4] parameter(1)
  bcast = f16[1,16,16,4] broadcast(arg_1), dimensions={3}
  ROOT %add = f16[1,16,16,4] add(arg_0, bcast)
}

ENTRY c1 {
  p0 = f16[1,16,16,2] parameter(0)
  p1 = f16[3,3,2,4] parameter(1)
  p2 = f16[2,2] parameter(2)

  p2_r = f16[4] reshape(p2)

  conv = f16[1,16,16,4] convolution(p0, p1), window={size=3x3 pad=1_1x1_1}, dim_labels=b01f_01io->b01f
  call = f16[1,16,16,64] fusion(conv, p2_r), kind=kCustom, calls=_pop_op_conv_biasadd

  ROOT t = (f16[1,16,16,4]) tuple(call)
}

)";

  auto config = GetModuleConfigForTest();
  config.set_argument_count(3);
  config.set_resource_input_count(2);
  config.set_input_mapping({0, 1, 2});
  config.set_resource_update_to_input_index({0});
  auto module = ParseHloString(hlo, config);
  EXPECT_TRUE(module.ok());
  auto* module0 = module.ValueOrDie().get();

  const auto* root = module0->entry_computation()->root_instruction();
  const auto* call = root->operand(0);
  const auto* conv = call->operand(0);
  const auto* ip0 = conv->operand(0);
  const auto* ip1 = conv->operand(1);
  const auto* reshape = call->operand(1);
  const auto* ip2 = reshape->operand(0);

  CompilerAnnotations annotations(module0);

  AllocationFinder finder(annotations);
  EXPECT_TRUE(finder.Run(module0).ValueOrDie());

  // Will have both of the convolution parameters
  EXPECT_EQ(annotations.tensor_allocation_map.size(), 2);

  auto t = annotations.tensor_allocation_map.at(std::make_pair(ip0, 0));
  EXPECT_EQ(t.tgt, conv);
  EXPECT_EQ(t.input_index, 0);

  t = annotations.tensor_allocation_map.at(std::make_pair(ip1, 0));
  EXPECT_EQ(t.tgt, conv);
  EXPECT_EQ(t.input_index, 1);

  ForwardAllocation fwd_finder(annotations);
  EXPECT_TRUE(fwd_finder.Run(module0).ValueOrDie());

  // We have added one new entry for the bias add
  EXPECT_EQ(annotations.tensor_allocation_map.size(), 3);

  t = annotations.tensor_allocation_map.at(std::make_pair(ip2, 0));
  EXPECT_EQ(t.tgt, call);
  EXPECT_EQ(t.input_index, 1);
  EXPECT_EQ(t.layout, conv);
  EXPECT_EQ(t.layout_output_idx, 0);
  EXPECT_EQ(t.forward_path.size(), 0);
  EXPECT_EQ(t.backward_path.size(), 1);
  EXPECT_EQ(t.backward_path[0], reshape);
}

TEST_F(AllocationFinderTest, MatMulBiasAdd) {
  std::string hlo = R"(
HloModule top

 %_pop_op_matmul_biasadd (arg_0: f32[2,2], arg_1: f32[2]) -> f32[2,2] {
   %arg_1 = f32[2] parameter(1)
   %broadcast.12.7.clone = f32[2,2]{1,0} broadcast(%arg_1), dimensions={1}
   %arg_0 = f32[2,2]{1,0} parameter(0)
   ROOT %add.12.8.clone = f32[2,2]{1,0} add(f32[2,2]{1,0} %arg_0, f32[2,2]{1,0} %broadcast.12.7.clone)
 }

 ENTRY %c (arg0.12.0: f32[2,2], arg1.12.1: f32[2,2], arg2.12.2: f32[2]) -> f32[2,2] {
   %arg0.12.0 = f32[2,2]{1,0} parameter(0)
   %arg1.12.1 = f32[2,2]{1,0} parameter(1)
   %dot.12.6 = f32[2,2]{1,0} dot(f32[2,2]{1,0} %arg0.12.0, f32[2,2]{1,0} %arg1.12.1), lhs_contracting_dims={1}, rhs_contracting_dims={0}
   %arg2.12.2 = f32[2] parameter(2), control-predecessors={%dot.12.6}
   ROOT %call = f32[2,2]{1,0} fusion(f32[2,2]{1,0} %dot.12.6, %arg2.12.2), kind=kCustom, calls=%_pop_op_matmul_biasadd
 }

)";

  auto config = GetModuleConfigForTest();
  config.set_argument_count(2);
  config.set_resource_input_count(2);
  config.set_input_mapping({0, 1});
  config.set_resource_update_to_input_index({0});
  auto module = ParseHloString(hlo, config);
  EXPECT_TRUE(module.ok());
  auto* module0 = module.ValueOrDie().get();

  const auto* root = module0->entry_computation()->root_instruction();
  const auto* call = root;
  const auto* dot = call->operand(0);
  const auto* ip0 = dot->operand(0);
  const auto* ip1 = dot->operand(1);
  const auto* ip2 = call->operand(1);

  CompilerAnnotations annotations(module0);

  AllocationFinder finder(annotations);
  EXPECT_TRUE(finder.Run(module0).ValueOrDie());

  // Will have both of the dot parameters
  EXPECT_EQ(annotations.tensor_allocation_map.size(), 2);

  auto t = annotations.tensor_allocation_map.at(std::make_pair(ip0, 0));
  EXPECT_EQ(t.tgt, dot);
  EXPECT_EQ(t.input_index, 0);

  t = annotations.tensor_allocation_map.at(std::make_pair(ip1, 0));
  EXPECT_EQ(t.tgt, dot);
  EXPECT_EQ(t.input_index, 1);

  ForwardAllocation fwd_finder(annotations);
  EXPECT_TRUE(fwd_finder.Run(module0).ValueOrDie());

  // We have added one new entry for the bias add
  EXPECT_EQ(annotations.tensor_allocation_map.size(), 3);

  t = annotations.tensor_allocation_map.at(std::make_pair(ip2, 0));
  EXPECT_EQ(t.tgt, call);
  EXPECT_EQ(t.input_index, 1);
  EXPECT_EQ(t.layout, dot);
  EXPECT_EQ(t.layout_output_idx, 0);
  EXPECT_EQ(t.forward_path.size(), 0);
  EXPECT_EQ(t.backward_path.size(), 0);
}

TEST_F(AllocationFinderTest,
       NoTargetBecauseOfDepenencyFromLayoutCreatorToSource) {
  // arg2.12.2 layout cannot depend on the layout of the dot.12.6, because the
  // former is a predecessor of the latter.
  std::string hlo = R"(
HloModule top

 %_pop_op_matmul_biasadd (arg_0: f32[2,2], arg_1: f32[2]) -> f32[2,2] {
   %arg_1 = f32[2] parameter(1)
   %broadcast.12.7.clone = f32[2,2]{1,0} broadcast(%arg_1), dimensions={1}
   %arg_0 = f32[2,2]{1,0} parameter(0)
   ROOT %add.12.8.clone = f32[2,2]{1,0} add(f32[2,2]{1,0} %arg_0, f32[2,2]{1,0} %broadcast.12.7.clone)
 }

 ENTRY %c (arg0.12.0: f32[2,2], arg1.12.1: f32[2,2], arg2.12.2: f32[2]) -> f32[2,2] {
   %arg0.12.0 = f32[2,2]{1,0} parameter(0)
   %arg1.12.1 = f32[2,2]{1,0} parameter(1)
   %arg2.12.2 = f32[2] parameter(2)
   %dot.12.6 = f32[2,2]{1,0} dot(f32[2,2]{1,0} %arg0.12.0, f32[2,2]{1,0} %arg1.12.1), lhs_contracting_dims={1}, rhs_contracting_dims={0}, control-predecessors={%arg2.12.2}
   ROOT %call = f32[2,2]{1,0} fusion(f32[2,2]{1,0} %dot.12.6, %arg2.12.2), kind=kCustom, calls=%_pop_op_matmul_biasadd
 }
)";

  auto config = GetModuleConfigForTest();
  config.set_argument_count(2);
  config.set_resource_input_count(2);
  config.set_input_mapping({0, 1});
  config.set_resource_update_to_input_index({0});
  auto module = ParseHloString(hlo, config);
  EXPECT_TRUE(module.ok());
  auto* module0 = module.ValueOrDie().get();

  const auto* root = module0->entry_computation()->root_instruction();
  const auto* call = root;
  const auto* dot = call->operand(0);
  const auto* ip0 = dot->operand(0);
  const auto* ip1 = dot->operand(1);
  const auto* ip2 = call->operand(1);

  CompilerAnnotations annotations(module0);

  AllocationFinder finder(annotations);
  EXPECT_TRUE(finder.Run(module0).ValueOrDie());

  // Will have both of the dot parameters
  EXPECT_EQ(annotations.tensor_allocation_map.size(), 2);

  auto t = annotations.tensor_allocation_map.at(std::make_pair(ip0, 0));
  EXPECT_EQ(t.tgt, dot);
  EXPECT_EQ(t.input_index, 0);

  t = annotations.tensor_allocation_map.at(std::make_pair(ip1, 0));
  EXPECT_EQ(t.tgt, dot);
  EXPECT_EQ(t.input_index, 1);

  ForwardAllocation fwd_finder(annotations);
  EXPECT_FALSE(fwd_finder.Run(module0).ValueOrDie());
  EXPECT_EQ(annotations.tensor_allocation_map.size(), 2);
}

TEST_F(AllocationFinderTest, MatMulBiasAddWithPath) {
  std::string hlo = R"(
HloModule top

 %_pop_op_matmul_biasadd (arg_0: f32[2,2], arg_1: f32[2]) -> f32[2,2] {
   %arg_1 = f32[2] parameter(1)
   %broadcast.12.7.clone = f32[2,2] broadcast(%arg_1), dimensions={1}
   %arg_0 = f32[2,2] parameter(0)
   ROOT %add.12.8.clone = f32[2,2] add(%arg_0, %broadcast.12.7.clone)
 }

 ENTRY %c (arg0.12.0: f32[2,2], arg1.12.1: f32[2,2], arg2.12.2: f32[1,2]) -> f32[2,2] {
   %arg0.12.0 = f32[2,2] parameter(0)
   %arg1.12.1 = f32[2,2] parameter(1)
   %dot.12.6 = f32[2,2] dot(%arg0.12.0, %arg1.12.1), lhs_contracting_dims={1}, rhs_contracting_dims={0}
   %arg2.12.2 = f32[1,2] parameter(2), control-predecessors={%dot.12.6}
   %p2_r = f32[2] reshape(%arg2.12.2)
   ROOT %call = f32[2,2] fusion(%dot.12.6, %p2_r), kind=kCustom, calls=%_pop_op_matmul_biasadd
 }

)";

  auto config = GetModuleConfigForTest();
  config.set_argument_count(3);
  config.set_resource_input_count(2);
  config.set_input_mapping({0, 1, 2});
  config.set_resource_update_to_input_index({0});
  auto module = ParseHloString(hlo, config);
  EXPECT_TRUE(module.ok());
  auto* module0 = module.ValueOrDie().get();

  const auto* root = module0->entry_computation()->root_instruction();
  const auto* call = root;
  const auto* dot = call->operand(0);
  const auto* ip0 = dot->operand(0);
  const auto* ip1 = dot->operand(1);
  const auto* reshape = call->operand(1);
  const auto* ip2 = reshape->operand(0);

  CompilerAnnotations annotations(module0);

  AllocationFinder finder(annotations);
  EXPECT_TRUE(finder.Run(module0).ValueOrDie());

  // Will have both of the dot parameters
  EXPECT_EQ(annotations.tensor_allocation_map.size(), 2);

  auto t = annotations.tensor_allocation_map.at(std::make_pair(ip0, 0));
  EXPECT_EQ(t.tgt, dot);
  EXPECT_EQ(t.input_index, 0);

  t = annotations.tensor_allocation_map.at(std::make_pair(ip1, 0));
  EXPECT_EQ(t.tgt, dot);
  EXPECT_EQ(t.input_index, 1);

  ForwardAllocation fwd_finder(annotations);
  EXPECT_TRUE(fwd_finder.Run(module0).ValueOrDie());

  // We have added one new entry for the bias add
  EXPECT_EQ(annotations.tensor_allocation_map.size(), 3);

  t = annotations.tensor_allocation_map.at(std::make_pair(ip2, 0));
  EXPECT_EQ(t.tgt, call);
  EXPECT_EQ(t.input_index, 1);
  EXPECT_EQ(t.layout, dot);
  EXPECT_EQ(t.layout_output_idx, 0);
  EXPECT_EQ(t.forward_path.size(), 0);
  EXPECT_EQ(t.backward_path.size(), 1);
  EXPECT_EQ(t.backward_path[0], reshape);
}

TEST_F(AllocationFinderTest, BatchNormInfParams) {
  std::string hlo = R"(
HloModule top

ENTRY %top (arg0.36.22: f32[1,4,4,2], arg1.36.23: f32[1,1,2,2], arg2.36.24: f32[2], arg3.36.25: f32[2], arg4.36.26: f32[2], arg5.36.27: f32[2]) -> f32[1,4,4,2] {
 %arg0.36.22 = f32[1,4,4,2] parameter(0)
 %arg1.36.23 = f32[1,1,2,2] parameter(1)
 %convolution.36.29 = f32[1,4,4,2] convolution(%arg0.36.22, %arg1.36.23), window={size=1x1}, dim_labels=b01f_01io->b01f
 %arg2.36.24 = f32[2] parameter(2)
 %arg3.36.25 = f32[2] parameter(3)
 %arg4.36.26 = f32[2] parameter(4)
 %arg5.36.27 = f32[2] parameter(5)
 ROOT %batch-norm-inference.36.31 = f32[1,4,4,2] batch-norm-inference(%convolution.36.29, %arg2.36.24, %arg3.36.25, %arg4.36.26, %arg5.36.27), epsilon=0.001, feature_index=3
}

)";

  auto config = GetModuleConfigForTest();
  config.set_argument_count(6);
  config.set_resource_input_count(2);
  config.set_input_mapping({0, 1, 2, 3, 4, 5});
  config.set_resource_update_to_input_index({0});
  auto module = ParseHloString(hlo, config);
  EXPECT_TRUE(module.ok());
  auto* module0 = module.ValueOrDie().get();

  const auto* root = module0->entry_computation()->root_instruction();
  const auto* bn = root;
  const auto* conv = bn->operand(0);
  const auto* ip0 = conv->operand(0);
  const auto* ip1 = conv->operand(1);
  const auto* ip2 = bn->operand(1);
  const auto* ip3 = bn->operand(2);

  CompilerAnnotations annotations(module0);

  AllocationFinder finder(annotations);
  EXPECT_TRUE(finder.Run(module0).ValueOrDie());

  // Will have both of the convolution parameters
  EXPECT_EQ(annotations.tensor_allocation_map.size(), 2);

  auto t = annotations.tensor_allocation_map.at(std::make_pair(ip0, 0));
  EXPECT_EQ(t.tgt, conv);
  EXPECT_EQ(t.input_index, 0);

  t = annotations.tensor_allocation_map.at(std::make_pair(ip1, 0));
  EXPECT_EQ(t.tgt, conv);
  EXPECT_EQ(t.input_index, 1);

  ForwardAllocation fwd_finder(annotations);
  EXPECT_TRUE(fwd_finder.Run(module0).ValueOrDie());

  // We have added one new entry for the bias add
  EXPECT_EQ(annotations.tensor_allocation_map.size(), 4);

  t = annotations.tensor_allocation_map.at(std::make_pair(ip2, 0));
  EXPECT_EQ(t.tgt, bn);
  EXPECT_EQ(t.input_index, 1);
  EXPECT_EQ(t.layout, conv);
  EXPECT_EQ(t.layout_output_idx, 0);
  EXPECT_EQ(t.forward_path.size(), 0);
  EXPECT_EQ(t.backward_path.size(), 0);

  t = annotations.tensor_allocation_map.at(std::make_pair(ip3, 0));
  EXPECT_EQ(t.tgt, bn);
  EXPECT_EQ(t.input_index, 2);
  EXPECT_EQ(t.layout, conv);
  EXPECT_EQ(t.layout_output_idx, 0);
  EXPECT_EQ(t.forward_path.size(), 0);
  EXPECT_EQ(t.backward_path.size(), 0);
}

TEST_F(AllocationFinderTest, BatchNormInfParamsWithPath) {
  std::string hlo = R"(
HloModule top

ENTRY %top (arg0.36.22: f32[1,4,4,2], arg1.36.23: f32[1,1,2,2], arg2.36.24: f32[1,2], arg3.36.25: f32[1,2], arg4.36.26: f32[2], arg5.36.27: f32[2]) -> f32[1,4,4,2] {
 %arg0.36.22 = f32[1,4,4,2] parameter(0)
 %arg1.36.23 = f32[1,1,2,2] parameter(1)
 %convolution.36.29 = f32[1,4,4,2] convolution(%arg0.36.22, %arg1.36.23), window={size=1x1}, dim_labels=b01f_01io->b01f
 %arg2.36.24 = f32[1,2]{1,0} parameter(2)
 %arg2.36.24_r = f32[2] reshape(%arg2.36.24)
 %arg3.36.25 = f32[1,2]{1,0} parameter(3)
 %arg3.36.25_r = f32[2] reshape(%arg3.36.25)
 %arg4.36.26 = f32[2] parameter(4)
 %arg5.36.27 = f32[2] parameter(5)
 ROOT %batch-norm-inference.36.31 = f32[1,4,4,2] batch-norm-inference(%convolution.36.29, %arg2.36.24_r, %arg3.36.25_r, %arg4.36.26, %arg5.36.27), epsilon=0.001, feature_index=3
}

)";

  auto config = GetModuleConfigForTest();
  config.set_argument_count(6);
  config.set_resource_input_count(2);
  config.set_input_mapping({0, 1, 2, 3, 4, 5});
  config.set_resource_update_to_input_index({0});
  auto module = ParseHloString(hlo, config);
  EXPECT_TRUE(module.ok());
  auto* module0 = module.ValueOrDie().get();

  const auto* root = module0->entry_computation()->root_instruction();
  const auto* bn = root;
  const auto* conv = bn->operand(0);
  const auto* ip0 = conv->operand(0);
  const auto* ip1 = conv->operand(1);
  const auto* reshape1 = bn->operand(1);
  const auto* reshape2 = bn->operand(2);
  const auto* ip2 = reshape1->operand(0);
  const auto* ip3 = reshape2->operand(0);

  CompilerAnnotations annotations(module0);

  AllocationFinder finder(annotations);
  EXPECT_TRUE(finder.Run(module0).ValueOrDie());

  // Will have both of the convolution parameters
  EXPECT_EQ(annotations.tensor_allocation_map.size(), 2);

  auto t = annotations.tensor_allocation_map.at(std::make_pair(ip0, 0));
  EXPECT_EQ(t.tgt, conv);
  EXPECT_EQ(t.input_index, 0);

  t = annotations.tensor_allocation_map.at(std::make_pair(ip1, 0));
  EXPECT_EQ(t.tgt, conv);
  EXPECT_EQ(t.input_index, 1);

  ForwardAllocation fwd_finder(annotations);
  EXPECT_TRUE(fwd_finder.Run(module0).ValueOrDie());

  // We have added one new entry for the bias add
  EXPECT_EQ(annotations.tensor_allocation_map.size(), 4);

  t = annotations.tensor_allocation_map.at(std::make_pair(ip2, 0));
  EXPECT_EQ(t.tgt, bn);
  EXPECT_EQ(t.input_index, 1);
  EXPECT_EQ(t.layout, conv);
  EXPECT_EQ(t.layout_output_idx, 0);
  EXPECT_EQ(t.forward_path.size(), 0);
  EXPECT_EQ(t.backward_path.size(), 1);
  EXPECT_EQ(t.backward_path[0], reshape1);

  t = annotations.tensor_allocation_map.at(std::make_pair(ip3, 0));
  EXPECT_EQ(t.tgt, bn);
  EXPECT_EQ(t.input_index, 2);
  EXPECT_EQ(t.layout, conv);
  EXPECT_EQ(t.layout_output_idx, 0);
  EXPECT_EQ(t.forward_path.size(), 0);
  EXPECT_EQ(t.backward_path.size(), 1);
  EXPECT_EQ(t.backward_path[0], reshape2);
}

TEST_F(AllocationFinderTest, BatchNormInfParamsWithPoplibsCustomOpPath) {
  std::string hlo = R"(
HloModule top

ENTRY %top (arg0.36.22: f32[1,4,4,2], arg1.36.23: f32[1,1,2,2], arg2.36.24: f32[2], arg3.36.25: f32[1,2], arg4.36.26: f32[2], arg5.36.27: f32[2]) -> f32[1,4,4,2] {
 %arg0.36.22 = f32[1,4,4,2] parameter(0)
 %arg1.36.23 = f32[1,1,2,2] parameter(1)
 %convolution.36.29 = f32[1,4,4,2] convolution(%arg0.36.22, %arg1.36.23), window={size=1x1}, dim_labels=b01f_01io->b01f
 %arg2.36.24 = f32[2]{1,0} parameter(2)
 %sqrt = f32[2] custom-call(%arg2.36.24), custom_call_target="Popops::Sqrt", opaque="{\"allocating_indexes\":[],\"layout_dependencies\":{\"keys\":[],\"values\":[]},\"num_inplace_operands\":1}\n"
 %arg3.36.25 = f32[1,2]{1,0} parameter(3)
 %arg3.36.25_r = f32[2] reshape(%arg3.36.25)
 %arg4.36.26 = f32[2] parameter(4)
 %arg5.36.27 = f32[2] parameter(5)
 ROOT %batch-norm-inference.36.31 = f32[1,4,4,2] batch-norm-inference(%convolution.36.29, %sqrt, %arg3.36.25_r, %arg4.36.26, %arg5.36.27), epsilon=0.001, feature_index=3
}

)";

  auto config = GetModuleConfigForTest();
  config.set_argument_count(6);
  config.set_resource_input_count(2);
  config.set_input_mapping({0, 1, 2, 3, 4, 5});
  config.set_resource_update_to_input_index({0});
  auto module = ParseHloString(hlo, config);
  EXPECT_TRUE(module.ok());
  auto* module0 = module.ValueOrDie().get();

  const auto* root = module0->entry_computation()->root_instruction();
  const auto* bn = root;
  const auto* conv = bn->operand(0);
  const auto* ip0 = conv->operand(0);
  const auto* ip1 = conv->operand(1);
  const auto* sqrt = bn->operand(1);
  const auto* reshape = bn->operand(2);
  const auto* ip2 = sqrt->operand(0);
  const auto* ip3 = reshape->operand(0);

  CompilerAnnotations annotations(module0);

  AllocationFinder finder(annotations);
  EXPECT_TRUE(finder.Run(module0).ValueOrDie());

  // Will have both of the convolution parameters
  EXPECT_EQ(annotations.tensor_allocation_map.size(), 2);

  auto t = annotations.tensor_allocation_map.at(std::make_pair(ip0, 0));
  EXPECT_EQ(t.tgt, conv);
  EXPECT_EQ(t.input_index, 0);

  t = annotations.tensor_allocation_map.at(std::make_pair(ip1, 0));
  EXPECT_EQ(t.tgt, conv);
  EXPECT_EQ(t.input_index, 1);

  ForwardAllocation fwd_finder(annotations);
  EXPECT_TRUE(fwd_finder.Run(module0).ValueOrDie());

  // We have added one new entry for the bias add
  EXPECT_EQ(annotations.tensor_allocation_map.size(), 4);

  t = annotations.tensor_allocation_map.at(std::make_pair(ip2, 0));
  EXPECT_EQ(t.tgt, bn);
  EXPECT_EQ(t.input_index, 1);
  EXPECT_EQ(t.layout, conv);
  EXPECT_EQ(t.layout_output_idx, 0);
  EXPECT_EQ(t.forward_path.size(), 0);
  EXPECT_EQ(t.backward_path.size(), 1);
  EXPECT_EQ(t.backward_path[0], sqrt);

  t = annotations.tensor_allocation_map.at(std::make_pair(ip3, 0));
  EXPECT_EQ(t.tgt, bn);
  EXPECT_EQ(t.input_index, 2);
  EXPECT_EQ(t.layout, conv);
  EXPECT_EQ(t.layout_output_idx, 0);
  EXPECT_EQ(t.forward_path.size(), 0);
  EXPECT_EQ(t.backward_path.size(), 1);
  EXPECT_EQ(t.backward_path[0], reshape);
}

TEST_F(AllocationFinderTest, BatchNormTrainingParams) {
  std::string hlo = R"(
HloModule top
%Sum-reduction48 (x.48.45: f32[], y.48.46: f32[]) -> f32[] {
  %x.48.45 = f32[] parameter(0)
  %y.48.46 = f32[] parameter(1)
  ROOT %add.48.47 = f32[] add(f32[] %x.48.45, f32[] %y.48.46)
}

%_pop_op_conv_scaled_inplace (arg_0: f32[1,1,2,2], arg_1: f32[1,4,4,2], arg_2: f32[1,4,4,2]) -> f32[1,1,2,2] {
  %arg_1 = f32[1,4,4,2] parameter(1)
  %arg_2 = f32[1,4,4,2] parameter(2)
  %convolution.78.67.clone = f32[1,1,2,2] convolution(%arg_1, %arg_2), window={size=4x4}, dim_labels=f01b_i01o->01bf
  %constant.78.28.clone = f32[] constant(0.1)
  %broadcast.78.68.clone = f32[1,1,2,2] broadcast(f32[] %constant.78.28.clone), dimensions={}
  %multiply.78.69.clone = f32[1,1,2,2] multiply(%convolution.78.67.clone, %broadcast.78.68.clone)
  %arg_0 = f32[1,1,2,2] parameter(0)
  ROOT %subtract.78.70.clone = f32[1,1,2,2] subtract(%arg_0, %multiply.78.69.clone)
}

%_pop_op_wide_const () -> f32[1,4,4,2] {
  %constant.78.29.clone = f32[] constant(1)
  ROOT %broadcast.2.clone = f32[1,4,4,2] broadcast(f32[] %constant.78.29.clone), dimensions={}
}

%_pop_op_wide_const.1 () -> f32[2] {
  %constant.78.28.clone.1 = f32[] constant(0.1)
  ROOT %broadcast.78.64.clone = f32[2] broadcast(f32[] %constant.78.28.clone.1), dimensions={}
}

ENTRY %top (arg0.78.22: f32[1,4,4,2], arg1.78.23: f32[1,1,2,2], arg2.78.24: f32[2], arg3.78.25: f32[2]) -> (f32[], f32[1,1,2,2], f32[2], f32[2]) {
  %constant.78.43 = f32[] constant(0)
  %arg0.78.22 = f32[1,4,4,2] parameter(0)
  %arg1.78.23 = f32[1,1,2,2] parameter(1)
  %convolution.78.33 = f32[1,4,4,2] convolution(%arg0.78.22, %arg1.78.23), window={size=1x1}, dim_labels=b01f_01io->b01f
  %arg2.78.24 = f32[2] parameter(2)
  %arg3.78.25 = f32[2] parameter(3)
  %batch-norm-training.78.35 = (f32[1,4,4,2], f32[2], f32[2]) batch-norm-training(%convolution.78.33, %arg2.78.24, %arg3.78.25), epsilon=0.001, feature_index=3
  %get-tuple-element.78.36 = f32[1,4,4,2] get-tuple-element(%batch-norm-training.78.35), index=0
  %reduce.78.49 = f32[] reduce(%get-tuple-element.78.36, %constant.78.43), dimensions={0,1,2,3}, to_apply=%Sum-reduction48
  %call.1 = f32[1,4,4,2] fusion(), kind=kCustom, calls=%_pop_op_wide_const
  %get-tuple-element.78.38 = f32[2] get-tuple-element((f32[1,4,4,2], f32[2], f32[2]) %batch-norm-training.78.35), index=1
  %get-tuple-element.78.39 = f32[2] get-tuple-element((f32[1,4,4,2], f32[2], f32[2]) %batch-norm-training.78.35), index=2
  %batch-norm-grad.78.54 = (f32[1,4,4,2], f32[2], f32[2]) batch-norm-grad(%convolution.78.33, %arg2.78.24, %get-tuple-element.78.38, %get-tuple-element.78.39, %call.1), epsilon=0.001, feature_index=3
  %get-tuple-element.78.55 = f32[1,4,4,2] get-tuple-element(%batch-norm-grad.78.54), index=0
  %call = f32[1,1,2,2] fusion(%arg1.78.23, %arg0.78.22, %get-tuple-element.78.55), kind=kCustom, calls=%_pop_op_conv_scaled_inplace
  %call.2 = f32[2] fusion(), kind=kCustom, calls=%_pop_op_wide_const.1
  %get-tuple-element.78.56 = f32[2] get-tuple-element((f32[1,4,4,2], f32[2], f32[2]) %batch-norm-grad.78.54), index=1
  %multiply.78.65 = f32[2] multiply(%call.2, %get-tuple-element.78.56)
  %subtract.78.66 = f32[2] subtract(%arg2.78.24, %multiply.78.65)
  %get-tuple-element.78.57 = f32[2] get-tuple-element(%batch-norm-grad.78.54), index=2
  %multiply.78.62 = f32[2] multiply(%call.2, %get-tuple-element.78.57)
  %subtract.78.63 = f32[2] subtract(%arg3.78.25, %multiply.78.62)
  ROOT %tuple.78.77 = (f32[], f32[1,1,2,2], f32[2], f32[2]) tuple(%reduce.78.49, %call, %subtract.78.66, %subtract.78.63)
}

)";

  auto config = GetModuleConfigForTest();
  config.set_argument_count(4);
  config.set_resource_input_count(2);
  config.set_input_mapping({0, 1, 2, 3});
  config.set_resource_update_to_input_index({0});
  auto module = ParseHloString(hlo, config);
  EXPECT_TRUE(module.ok());
  auto* module0 = module.ValueOrDie().get();
  auto* entry_computation = module0->entry_computation();
  auto* arg0 = entry_computation->parameter_instruction(0);
  const auto* conv = arg0->users()[0]->opcode() == HloOpcode::kConvolution
                         ? arg0->users()[0]
                         : arg0->users()[1];
  const auto* conv_ip0 = conv->operand(0);
  const auto* conv_ip1 = conv->operand(1);
  const auto* conv_grad_call =
      arg0->users()[0]->opcode() == HloOpcode::kConvolution ? arg0->users()[1]
                                                            : arg0->users()[0];
  const auto* conv_grad_comp = conv_grad_call->fused_instructions_computation();
  const auto* conv_grad_ip0 = conv_grad_comp->parameter_instruction(1);
  const auto* conv_grad_ip1 = conv_grad_comp->parameter_instruction(2);
  const auto* conv_grad = conv_grad_ip1->users()[0];
  const auto* bn_tr =
      conv->users()[0]->opcode() == HloOpcode::kBatchNormTraining
          ? conv->users()[0]
          : conv->users()[1];
  const auto* bn_ip1 = bn_tr->operand(1);
  const auto* bn_ip2 = bn_tr->operand(2);

  CompilerAnnotations annotations(module0);

  AllocationFinder finder(annotations);
  EXPECT_TRUE(finder.Run(module0).ValueOrDie());

  // Will have both of the convolution parameters
  EXPECT_EQ(annotations.tensor_allocation_map.size(), 4);

  auto t = annotations.tensor_allocation_map.at(std::make_pair(conv_ip0, 0));
  EXPECT_EQ(t.tgt, conv);
  EXPECT_EQ(t.input_index, 0);

  t = annotations.tensor_allocation_map.at(std::make_pair(conv_ip1, 0));
  EXPECT_EQ(t.tgt, conv);
  EXPECT_EQ(t.input_index, 1);

  t = annotations.tensor_allocation_map.at(std::make_pair(conv_grad_ip0, 0));
  EXPECT_EQ(t.tgt, conv_grad);
  EXPECT_EQ(t.input_index, 0);

  t = annotations.tensor_allocation_map.at(std::make_pair(conv_grad_ip1, 0));
  EXPECT_EQ(t.tgt, conv_grad);
  EXPECT_EQ(t.input_index, 1);

  ForwardAllocation fwd_finder(annotations);
  EXPECT_TRUE(fwd_finder.Run(module0).ValueOrDie());

  // We have added one new entry for the bias add
  ASSERT_EQ(annotations.tensor_allocation_map.size(), 6);

  t = annotations.tensor_allocation_map.at(std::make_pair(bn_ip1, 0));
  EXPECT_EQ(t.tgt, bn_tr);
  EXPECT_EQ(t.input_index, 1);
  EXPECT_EQ(t.layout, conv);
  EXPECT_EQ(t.layout_output_idx, 0);
  EXPECT_EQ(t.forward_path.size(), 0);
  EXPECT_EQ(t.backward_path.size(), 0);

  t = annotations.tensor_allocation_map.at(std::make_pair(bn_ip2, 0));
  EXPECT_EQ(t.tgt, bn_tr);
  EXPECT_EQ(t.input_index, 2);
  EXPECT_EQ(t.layout, conv);
  EXPECT_EQ(t.layout_output_idx, 0);
  EXPECT_EQ(t.forward_path.size(), 0);
  EXPECT_EQ(t.backward_path.size(), 0);
}

TEST_F(AllocationFinderTest, ForwardAllocationMultipleUsesOneTarget) {
  // In this test we check that %arg2.36.24 still has a layout even though
  // it has two targets but only one is a layout sensitive target.
  std::string hlo = R"(
HloModule top
%Sum-reduction48 (x.48.45: f32[2], y.48.46: f32[2]) -> f32[2] {
  %x.48.45 = f32[2]{0} parameter(0)
  %y.48.46 = f32[2]{0} parameter(1)
  ROOT %add.48.47 = f32[2]{0} add(f32[2]{0} %x.48.45, f32[2]{0} %y.48.46)
}

ENTRY %top (arg0.36.22: f32[1,4,4,2], arg1.36.23: f32[1,1,2,2], arg2.36.24: f32[1,2], arg3.36.25: f32[1,2], arg4.36.26: f32[2], arg5.36.27: f32[2]) -> f32[2] {
 %arg0.36.22 = f32[1,4,4,2]{3,2,1,0} parameter(0)
 %arg1.36.23 = f32[1,1,2,2]{3,2,1,0} parameter(1)
 %convolution.36.29 = f32[1,4,4,2]{3,2,1,0} convolution(f32[1,4,4,2]{3,2,1,0} %arg0.36.22, f32[1,1,2,2]{3,2,1,0} %arg1.36.23), window={size=1x1}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="vs/conv2d/Conv2D"}
 %arg2.36.24 = f32[1,2]{1,0} parameter(2)
 %arg2.36.24_r = f32[2]{0} reshape(%arg2.36.24)
 %arg3.36.25 = f32[1,2]{1,0} parameter(3)
 %arg3.36.25_r = f32[2]{0} reshape(%arg3.36.25)
 %arg4.36.26 = f32[2]{0} parameter(4)
 %arg5.36.27 = f32[2]{0} parameter(5)
 %batch-norm-inference.36.31 = f32[1,4,4,2]{3,2,1,0} batch-norm-inference(f32[1,4,4,2]{3,2,1,0} %convolution.36.29, f32[2]{0} %arg2.36.24_r, f32[2]{0} %arg3.36.25_r, f32[2]{0} %arg4.36.26, f32[2]{0} %arg5.36.27), epsilon=0.001, feature_index=3, metadata={op_type="FusedBatchNorm" op_name="vs/batch_normalization/FusedBatchNorm"}
 ROOT %reduce.78.49 = f32[2]{0} reduce(f32[1,4,4,2]{3,2,1,0} %batch-norm-inference.36.31, f32[2]{0} %arg2.36.24_r), dimensions={1,2,3}, to_apply=%Sum-reduction48, metadata={op_type="Sum" op_name="Sum"}
}

)";

  auto config = GetModuleConfigForTest();
  config.set_argument_count(6);
  config.set_resource_input_count(2);
  config.set_input_mapping({0, 1, 2, 3, 4, 5});
  config.set_resource_update_to_input_index({0});
  auto module = ParseHloString(hlo, config);
  EXPECT_TRUE(module.ok());
  auto* module0 = module.ValueOrDie().get();

  const auto* root = module0->entry_computation()->root_instruction();
  const auto* reduce = root;
  const auto* bn = reduce->operand(0);
  const auto* conv = bn->operand(0);
  const auto* ip0 = conv->operand(0);
  const auto* ip1 = conv->operand(1);
  const auto* reshape1 = bn->operand(1);
  const auto* reshape2 = bn->operand(2);
  const auto* ip2 = reshape1->operand(0);
  const auto* ip3 = reshape2->operand(0);

  CompilerAnnotations annotations(module0);

  AllocationFinder finder(annotations);
  EXPECT_TRUE(finder.Run(module0).ValueOrDie());

  // Will have both of the convolution parameters
  EXPECT_EQ(annotations.tensor_allocation_map.size(), 2);

  auto t = annotations.tensor_allocation_map.at(std::make_pair(ip0, 0));
  EXPECT_EQ(t.tgt, conv);
  EXPECT_EQ(t.input_index, 0);

  t = annotations.tensor_allocation_map.at(std::make_pair(ip1, 0));
  EXPECT_EQ(t.tgt, conv);
  EXPECT_EQ(t.input_index, 1);

  ForwardAllocation fwd_finder(annotations);
  unsigned num_succesful_runs = 0;
  while (fwd_finder.Run(module0).ValueOrDie()) {
    num_succesful_runs++;
  }

  // Depending on the order we either expect this to be executed successfully 1
  // or 2 times.
  EXPECT_TRUE(num_succesful_runs == 1 || num_succesful_runs == 2);

  // We have added one new entry for the bias add
  EXPECT_EQ(annotations.tensor_allocation_map.size(), 4);

  t = annotations.tensor_allocation_map.at(std::make_pair(ip2, 0));
  EXPECT_EQ(t.tgt, bn);
  EXPECT_EQ(t.input_index, 1);
  EXPECT_EQ(t.layout, conv);
  EXPECT_EQ(t.layout_output_idx, 0);
  EXPECT_EQ(t.forward_path.size(), 0);
  EXPECT_EQ(t.backward_path.size(), 1);
  EXPECT_EQ(t.backward_path[0], reshape1);

  t = annotations.tensor_allocation_map.at(std::make_pair(ip3, 0));
  EXPECT_EQ(t.tgt, bn);
  EXPECT_EQ(t.input_index, 2);
  EXPECT_EQ(t.layout, conv);
  EXPECT_EQ(t.layout_output_idx, 0);
  EXPECT_EQ(t.forward_path.size(), 0);
  EXPECT_EQ(t.backward_path.size(), 1);
  EXPECT_EQ(t.backward_path[0], reshape2);
}

TEST_F(AllocationFinderTest,
       ForwardAllocationMultipleUsesMultipleTargetsSamePriority) {
  // In this test we check that %arg2.36.24 and %arg3.36.25 get a layout even
  // though they have multiple targets.
  std::string hlo = R"(
HloModule top
%Sum-reduction48 (x.48.45: f32[2], y.48.46: f32[2]) -> f32[2] {
  %x.48.45 = f32[2]{0} parameter(0)
  %y.48.46 = f32[2]{0} parameter(1)
  ROOT %add.48.47 = f32[2]{0} add(f32[2]{0} %x.48.45, f32[2]{0} %y.48.46)
}

ENTRY %top (arg0.36.22: f32[1,4,4,2], arg1.36.23: f32[1,1,2,2], arg2.36.24: f32[1,2], arg3.36.25: f32[1,2], arg4.36.26: f32[2], arg5.36.27: f32[2], arg6.36.28: f32[1,4,4,2]) -> (f32[1,4,4,2], f32[1,4,4,2]) {
 %arg0.36.22 = f32[1,4,4,2]{3,2,1,0} parameter(0)
 %arg1.36.23 = f32[1,1,2,2]{3,2,1,0} parameter(1)
 %convolution.36.29 = f32[1,4,4,2]{3,2,1,0} convolution(f32[1,4,4,2]{3,2,1,0} %arg0.36.22, f32[1,1,2,2]{3,2,1,0} %arg1.36.23), window={size=1x1}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="vs/conv2d/Conv2D"}
 %arg2.36.24 = f32[1,2]{1,0} parameter(2)
 %arg2.36.24_r = f32[2]{0} reshape(%arg2.36.24)
 %arg3.36.25 = f32[1,2]{1,0} parameter(3)
 %arg3.36.25_r = f32[2]{0} reshape(%arg3.36.25)
 %arg4.36.26 = f32[2]{0} parameter(4)
 %arg5.36.27 = f32[2]{0} parameter(5)
 %arg6.36.28 = f32[1,4,4,2]{3,2,1,0} parameter(6)
 %batch-norm-inference.36.31 = f32[1,4,4,2]{3,2,1,0} batch-norm-inference(f32[1,4,4,2]{3,2,1,0} %convolution.36.29, f32[2]{0} %arg2.36.24_r, f32[2]{0} %arg3.36.25_r, f32[2]{0} %arg4.36.26, f32[2]{0} %arg5.36.27), epsilon=0.001, feature_index=3, metadata={op_type="FusedBatchNorm" op_name="vs/batch_normalization/FusedBatchNorm"}
 %batch-norm-inference.36.32 = f32[1,4,4,2]{3,2,1,0} batch-norm-inference(f32[1,4,4,2]{3,2,1,0} %convolution.36.29, f32[2]{0} %arg2.36.24_r, f32[2]{0} %arg3.36.25_r, f32[2]{0} %arg4.36.26, f32[2]{0} %arg5.36.27), epsilon=0.001, feature_index=3, metadata={op_type="FusedBatchNorm" op_name="vs/batch_normalization/FusedBatchNorm"}
 ROOT %tuple = (f32[1,4,4,2]{3,2,1,0}, f32[1,4,4,2]{3,2,1,0}) tuple(f32[1,4,4,2]{3,2,1,0} %batch-norm-inference.36.31, f32[1,4,4,2]{3,2,1,0} %batch-norm-inference.36.32)
}

)";

  auto config = GetModuleConfigForTest();
  config.set_argument_count(6);
  config.set_resource_input_count(2);
  config.set_input_mapping({0, 1, 2, 3, 4, 5});
  config.set_resource_update_to_input_index({0});
  auto module = ParseHloString(hlo, config);
  EXPECT_TRUE(module.ok());
  auto* module0 = module.ValueOrDie().get();

  const auto* root = module0->entry_computation()->root_instruction();
  const auto* bn1 = root->operand(0);
  const auto* bn2 = root->operand(1);
  const auto* conv = bn1->operand(0);
  const auto* ip0 = conv->operand(0);
  const auto* ip1 = conv->operand(1);
  const auto* reshape1 = bn1->operand(1);
  const auto* reshape2 = bn1->operand(2);
  const auto* ip2 = reshape1->operand(0);
  const auto* ip3 = reshape2->operand(0);

  CompilerAnnotations annotations(module0);

  AllocationFinder finder(annotations);
  EXPECT_TRUE(finder.Run(module0).ValueOrDie());

  // Will have both of the convolution parameters
  EXPECT_EQ(annotations.tensor_allocation_map.size(), 2);

  auto t = annotations.tensor_allocation_map.at(std::make_pair(ip0, 0));
  EXPECT_EQ(t.tgt, conv);
  EXPECT_EQ(t.input_index, 0);

  t = annotations.tensor_allocation_map.at(std::make_pair(ip1, 0));
  EXPECT_EQ(t.tgt, conv);
  EXPECT_EQ(t.input_index, 1);

  unsigned num_succesful_runs = 0;
  ForwardAllocation fwd_finder(annotations);
  while (fwd_finder.Run(module0).ValueOrDie()) {
    num_succesful_runs++;
  }

  // Depending on the order we either expect this to be executed successfully 1
  // or 2 times.
  EXPECT_TRUE(num_succesful_runs == 1 || num_succesful_runs == 2);

  // We have added two new entires for the layer norms.
  EXPECT_EQ(annotations.tensor_allocation_map.size(), 4);

  t = annotations.tensor_allocation_map.at(std::make_pair(ip2, 0));
  auto target_bn = t.tgt == bn1 ? bn1 : bn2;
  // It was allocated for one of the batch norms.
  EXPECT_EQ(t.tgt, target_bn);
  EXPECT_EQ(t.input_index, 1);
  EXPECT_EQ(t.layout, conv);
  EXPECT_EQ(t.layout_output_idx, 0);
  EXPECT_EQ(t.forward_path.size(), 0);
  EXPECT_EQ(t.backward_path.size(), 1);
  EXPECT_EQ(t.backward_path[0], reshape1);

  t = annotations.tensor_allocation_map.at(std::make_pair(ip3, 0));
  // It was allocated for same batch norm due to control dependencies.
  EXPECT_EQ(t.tgt, target_bn);
  EXPECT_EQ(t.input_index, 2);
  EXPECT_EQ(t.layout, conv);
  EXPECT_EQ(t.layout_output_idx, 0);
  EXPECT_EQ(t.forward_path.size(), 0);
  EXPECT_EQ(t.backward_path.size(), 1);
  EXPECT_EQ(t.backward_path[0], reshape2);
}

TEST_F(AllocationFinderTest,
       ForwardAllocationMultipleUsesMultipleTargetsDifferentPriority) {
  // In this test we check that %arg2.36.24 and %arg3.36.25 get a layout even
  // though they have multiple targets - layer norms have higher priority than
  // elementwise ops.
  std::string hlo = R"(
HloModule top
%Sum-reduction48 (x.48.45: f32[2], y.48.46: f32[2]) -> f32[2] {
  %x.48.45 = f32[2]{0} parameter(0)
  %y.48.46 = f32[2]{0} parameter(1)
  ROOT %add.48.47 = f32[2]{0} add(f32[2]{0} %x.48.45, f32[2]{0} %y.48.46)
}

ENTRY %top (arg0.36.22: f32[1,4,4,2], arg1.36.23: f32[1,1,2,2], arg2.36.24: f32[1,2], arg3.36.25: f32[1,2], arg4.36.26: f32[2], arg5.36.27: f32[2]) -> (f32[1,4,4,2], f32[1,2]) {
 %arg0.36.22 = f32[1,4,4,2]{3,2,1,0} parameter(0)
 %arg1.36.23 = f32[1,1,2,2]{3,2,1,0} parameter(1)
 %convolution.36.29 = f32[1,4,4,2]{3,2,1,0} convolution(f32[1,4,4,2]{3,2,1,0} %arg0.36.22, f32[1,1,2,2]{3,2,1,0} %arg1.36.23), window={size=1x1}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="vs/conv2d/Conv2D"}
 %arg2.36.24 = f32[1,2]{1,0} parameter(2)
 %arg2.36.24_r = f32[2]{0} reshape(%arg2.36.24)
 %arg3.36.25 = f32[1,2]{1,0} parameter(3)
 %arg3.36.25_r = f32[2]{0} reshape(%arg3.36.25)
 %arg4.36.26 = f32[2]{0} parameter(4)
 %arg5.36.27 = f32[2]{0} parameter(5)
 %batch-norm-inference.36.31 = f32[1,4,4,2]{3,2,1,0} batch-norm-inference(f32[1,4,4,2]{3,2,1,0} %convolution.36.29, f32[2]{0} %arg2.36.24_r, f32[2]{0} %arg3.36.25_r, f32[2]{0} %arg4.36.26, f32[2]{0} %arg5.36.27), epsilon=0.001, feature_index=3, metadata={op_type="FusedBatchNorm" op_name="vs/batch_normalization/FusedBatchNorm"}
 %add = f32[1,2]{1,0} add(f32[2]{0} %arg2.36.24_r, f32[2]{0} %arg3.36.25_r)
 ROOT %tuple = (f32[1,4,4,2]{3,2,1,0}, f32[1,2]{1,0}) tuple(f32[1,4,4,2]{3,2,1,0} %batch-norm-inference.36.31, f32[1,2]{1,0} %add)
}

)";

  auto config = GetModuleConfigForTest();
  config.set_argument_count(6);
  config.set_resource_input_count(2);
  config.set_input_mapping({0, 1, 2, 3, 4, 5});
  config.set_resource_update_to_input_index({0});
  auto module = ParseHloString(hlo, config);
  EXPECT_TRUE(module.ok());
  auto* module0 = module.ValueOrDie().get();

  const auto* root = module0->entry_computation()->root_instruction();
  const auto* bn = root->operand(0);
  const auto* add = root->operand(1);
  const auto* conv = bn->operand(0);
  const auto* ip0 = conv->operand(0);
  const auto* ip1 = conv->operand(1);
  const auto* reshape1 = bn->operand(1);
  const auto* reshape2 = bn->operand(2);
  const auto* ip2 = reshape1->operand(0);
  const auto* ip3 = reshape2->operand(0);

  CompilerAnnotations annotations(module0);

  AllocationFinder finder(annotations);
  EXPECT_TRUE(finder.Run(module0).ValueOrDie());

  // Will have both of the convolution parameters
  EXPECT_EQ(annotations.tensor_allocation_map.size(), 2);

  auto t = annotations.tensor_allocation_map.at(std::make_pair(ip0, 0));
  EXPECT_EQ(t.tgt, conv);
  EXPECT_EQ(t.input_index, 0);

  t = annotations.tensor_allocation_map.at(std::make_pair(ip1, 0));
  EXPECT_EQ(t.tgt, conv);
  EXPECT_EQ(t.input_index, 1);

  unsigned num_succesful_runs = 0;
  ForwardAllocation fwd_finder(annotations);
  while (fwd_finder.Run(module0).ValueOrDie()) {
    num_succesful_runs++;
  }

  // Depending on the order we either expect this to be executed successfully 1
  // or 2 times.
  EXPECT_TRUE(num_succesful_runs == 1 || num_succesful_runs == 2);

  // We have added two new entires for the layer norms.
  EXPECT_EQ(annotations.tensor_allocation_map.size(), 4);

  t = annotations.tensor_allocation_map.at(std::make_pair(ip2, 0));
  // Layer norm has priority over elementwise ops.
  EXPECT_EQ(t.tgt, bn);
  EXPECT_EQ(t.input_index, 1);
  EXPECT_EQ(t.layout, conv);
  EXPECT_EQ(t.layout_output_idx, 0);
  EXPECT_EQ(t.forward_path.size(), 0);
  EXPECT_EQ(t.backward_path.size(), 1);
  EXPECT_EQ(t.backward_path[0], reshape1);

  t = annotations.tensor_allocation_map.at(std::make_pair(ip3, 0));
  // Layer norm has priority over elementwise ops.
  EXPECT_EQ(t.tgt, bn);
  EXPECT_EQ(t.input_index, 2);
  EXPECT_EQ(t.layout, conv);
  EXPECT_EQ(t.layout_output_idx, 0);
  EXPECT_EQ(t.forward_path.size(), 0);
  EXPECT_EQ(t.backward_path.size(), 1);
  EXPECT_EQ(t.backward_path[0], reshape2);
}

TEST_F(AllocationFinderTest, ForwardAllocationElementwiseGetsALayout) {
  // Check the layout is forwarded to the element wise op argument.
  std::string hlo = R"(
HloModule top

_pop_op_conv_biasadd {
  %arg_0 = f16[1,16,16,4] parameter(0)
  %arg_1 = f16[4] parameter(1)
  bcast = f16[1,16,16,4] broadcast(arg_1), dimensions={3}
  ROOT %add = f16[1,16,16,4] add(arg_0, bcast)
}

ENTRY c1 {
  p0 = f16[1,16,16,2] parameter(0)
  p1 = f16[3,3,2,4] parameter(1)
  p2 = f16[2,2] parameter(2)
  p2_r = f16[4] reshape(p2)

  conv = f16[1,16,16,4] convolution(p0, p1), window={size=3x3 pad=1_1x1_1}, dim_labels=b01f_01io->b01f
  call = f16[1,16,16,64] fusion(conv, p2_r), kind=kCustom, calls=_pop_op_conv_biasadd
  p3 = f16[1,16,16,64] parameter(3)
  ROOT add = f16[1,16,16,64] add(p3, call)
}
)";

  auto config = GetModuleConfigForTest();
  config.set_argument_count(3);
  config.set_resource_input_count(3);
  config.set_input_mapping({0, 1, 2, 3});
  config.set_resource_update_to_input_index({0});
  auto module = ParseHloString(hlo, config);
  EXPECT_TRUE(module.ok());
  auto* module0 = module.ValueOrDie().get();

  const auto* root = module0->entry_computation()->root_instruction();
  const auto* add = root;
  const auto* ip3 = add->operand(0);
  const auto* call = add->operand(1);
  const auto* conv = call->operand(0);
  const auto* ip2_r = call->operand(1);
  const auto* ip2 = ip2_r->operand(0);
  const auto* ip1 = conv->operand(1);
  const auto* ip0 = conv->operand(0);

  CompilerAnnotations annotations(module0);

  AllocationFinder finder(annotations);
  EXPECT_TRUE(finder.Run(module0).ValueOrDie());

  // Will have both of the convolution parameters
  EXPECT_EQ(annotations.tensor_allocation_map.size(), 2);

  auto t = annotations.tensor_allocation_map.at(std::make_pair(ip0, 0));
  EXPECT_EQ(t.tgt, conv);
  EXPECT_EQ(t.input_index, 0);

  t = annotations.tensor_allocation_map.at(std::make_pair(ip1, 0));
  EXPECT_EQ(t.tgt, conv);
  EXPECT_EQ(t.input_index, 1);

  ForwardAllocation fwd_finder(annotations);
  unsigned num_succesful_runs = 0;
  while (fwd_finder.Run(module0).ValueOrDie()) {
    num_succesful_runs++;
  }

  // Depending on the order we either expect this to be executed successfully 1
  // or 2 times.
  EXPECT_TRUE(num_succesful_runs == 1 || num_succesful_runs == 2);

  // We have added one new entry for the bias add
  EXPECT_EQ(annotations.tensor_allocation_map.size(), 4);

  t = annotations.tensor_allocation_map.at(std::make_pair(ip2, 0));
  EXPECT_EQ(t.tgt, call);
  EXPECT_EQ(t.input_index, 1);
  EXPECT_EQ(t.layout, conv);
  EXPECT_EQ(t.layout_output_idx, 0);
  EXPECT_EQ(t.forward_path.size(), 0);
  EXPECT_EQ(t.backward_path.size(), 1);
  EXPECT_EQ(t.backward_path[0], ip2_r);

  t = annotations.tensor_allocation_map.at(std::make_pair(ip3, 0));
  EXPECT_EQ(t.tgt, add);
  EXPECT_EQ(t.input_index, 0);
  EXPECT_EQ(t.layout, conv);
  EXPECT_EQ(t.layout_output_idx, 0);
  EXPECT_EQ(t.forward_path.size(), 1);
  EXPECT_EQ(t.forward_path[0], call);
  EXPECT_EQ(t.backward_path.size(), 0);
}

TEST_F(AllocationFinderTest, ForwardAllocationDontLookThroughCasts) {
  // Check the layout is not forwarded to the element wise op argument as it's
  // casted.
  std::string hlo = R"(
HloModule top

_pop_op_conv_biasadd {
  %arg_0 = f16[1,16,16,4] parameter(0)
  %arg_1 = f16[4] parameter(1)
  bcast = f16[1,16,16,4] broadcast(arg_1), dimensions={3}
  ROOT %add = f16[1,16,16,4] add(arg_0, bcast)
}

ENTRY c1 {
  p0 = f16[1,16,16,2] parameter(0)
  p1 = f16[3,3,2,4] parameter(1)
  p2 = f16[2,2] parameter(2)
  p2_r = f16[4] reshape(p2)

  conv = f16[1,16,16,4] convolution(p0, p1), window={size=3x3 pad=1_1x1_1}, dim_labels=b01f_01io->b01f
  call = f16[1,16,16,64] fusion(conv, p2_r), kind=kCustom, calls=_pop_op_conv_biasadd
  p3 = f32[1,16,16,64] parameter(3)
  p3.c = f16[1,16,16,64] convert(p3)
  ROOT add = f16[1,16,16,64] add(p3.c, call)
}
)";

  auto config = GetModuleConfigForTest();
  config.set_argument_count(3);
  config.set_resource_input_count(3);
  config.set_input_mapping({0, 1, 2, 3});
  config.set_resource_update_to_input_index({0});
  auto module = ParseHloString(hlo, config);
  EXPECT_TRUE(module.ok());
  auto* module0 = module.ValueOrDie().get();

  const auto* root = module0->entry_computation()->root_instruction();
  const auto* add = root;
  const auto* call = add->operand(1);
  const auto* conv = call->operand(0);
  const auto* ip2_r = call->operand(1);
  const auto* ip2 = ip2_r->operand(0);
  const auto* ip1 = conv->operand(1);
  const auto* ip0 = conv->operand(0);

  CompilerAnnotations annotations(module0);

  AllocationFinder finder(annotations);
  EXPECT_TRUE(finder.Run(module0).ValueOrDie());

  // Will have both of the convolution parameters
  EXPECT_EQ(annotations.tensor_allocation_map.size(), 2);

  auto t = annotations.tensor_allocation_map.at(std::make_pair(ip0, 0));
  EXPECT_EQ(t.tgt, conv);
  EXPECT_EQ(t.input_index, 0);

  t = annotations.tensor_allocation_map.at(std::make_pair(ip1, 0));
  EXPECT_EQ(t.tgt, conv);
  EXPECT_EQ(t.input_index, 1);

  ForwardAllocation fwd_finder(annotations);
  unsigned num_succesful_runs = 0;
  while (fwd_finder.Run(module0).ValueOrDie()) {
    num_succesful_runs++;
  }

  // We expect this to be executed successfully 1 time.
  EXPECT_TRUE(num_succesful_runs == 1);

  // We have added one new entry for the bias add
  EXPECT_EQ(annotations.tensor_allocation_map.size(), 3);

  t = annotations.tensor_allocation_map.at(std::make_pair(ip2, 0));
  EXPECT_EQ(t.tgt, call);
  EXPECT_EQ(t.input_index, 1);
  EXPECT_EQ(t.layout, conv);
  EXPECT_EQ(t.layout_output_idx, 0);
  EXPECT_EQ(t.forward_path.size(), 0);
  EXPECT_EQ(t.backward_path.size(), 1);
  EXPECT_EQ(t.backward_path[0], ip2_r);
}

TEST_F(AllocationFinderTest, ForwardAllocationElementwiseGetsALayoutWithGTE) {
  // Check the layout is forwarded to the element wise op argument with a GTE.
  std::string hlo = R"(
HloModule top
ENTRY %top (arg0.78.22: f32[1,4,4,2], arg1: f32[1,1,2,2], arg2: f32[2], arg3: f32[2], arg3: f32[2]) -> f32[2] {
  %arg0 = f32[1,4,4,2]{3,2,1,0} parameter(0)
  %arg1 = f32[1,1,2,2]{3,2,1,0} parameter(1)
  %convolution = f32[1,4,4,2]{3,2,1,0} convolution(f32[1,4,4,2]{3,2,1,0} %arg0, f32[1,1,2,2]{3,2,1,0} %arg1), window={size=1x1}, dim_labels=b01f_01io->b01f
  %arg2 = f32[2]{0} parameter(2)
  %arg3 = f32[2]{0} parameter(3)
  %batch-norm-training = (f32[1,4,4,2]{3,2,1,0}, f32[2]{0}, f32[2]{0}) batch-norm-training(f32[1,4,4,2]{3,2,1,0} %convolution, f32[2]{0} %arg2, f32[2]{0} %arg3), epsilon=0.001, feature_index=3
  %get-tuple-element = f32[2]{0} get-tuple-element((f32[1,4,4,2]{3,2,1,0}, f32[2]{0}, f32[2]{0}) %batch-norm-training), index=2
  %arg4 = f32[2]{0} parameter(4)
  ROOT %subtract = f32[2]{0} subtract(%get-tuple-element, %arg4)
}

)";

  auto config = GetModuleConfigForTest();
  config.set_resource_input_count(2);
  config.set_input_mapping({0, 1, 2, 3, 4});
  config.set_resource_update_to_input_index({0});
  auto module = ParseHloString(hlo, config);
  EXPECT_TRUE(module.ok());
  auto* module0 = module.ValueOrDie().get();

  const auto* root = module0->entry_computation()->root_instruction();
  const auto* subtract = root;
  const auto* ip4 = subtract->operand(1);
  const auto* gte = subtract->operand(0);
  const auto* bn = gte->operand(0);
  const auto* conv = bn->operand(0);
  const auto* ip3 = bn->operand(2);
  const auto* ip2 = bn->operand(1);
  const auto* ip1 = conv->operand(1);
  const auto* ip0 = conv->operand(0);

  CompilerAnnotations annotations(module0);

  AllocationFinder finder(annotations);
  EXPECT_TRUE(finder.Run(module0).ValueOrDie());

  // Will have both of the convolution parameters
  EXPECT_EQ(annotations.tensor_allocation_map.size(), 2);

  auto t = annotations.tensor_allocation_map.at(std::make_pair(ip0, 0));
  EXPECT_EQ(t.tgt, conv);
  EXPECT_EQ(t.input_index, 0);

  t = annotations.tensor_allocation_map.at(std::make_pair(ip1, 0));
  EXPECT_EQ(t.tgt, conv);
  EXPECT_EQ(t.input_index, 1);

  ForwardAllocation fwd_finder(annotations);
  unsigned num_succesful_runs = 0;
  while (fwd_finder.Run(module0).ValueOrDie()) {
    num_succesful_runs++;
  }

  // Depending on the order we either expect this to be executed successfully 1
  // or 2 times.
  EXPECT_TRUE(num_succesful_runs == 1 || num_succesful_runs == 2);

  // We have added one new entry for the bias add
  EXPECT_EQ(annotations.tensor_allocation_map.size(), 5);

  t = annotations.tensor_allocation_map.at(std::make_pair(ip2, 0));
  EXPECT_EQ(t.tgt, bn);
  EXPECT_EQ(t.input_index, 1);
  EXPECT_EQ(t.layout, conv);
  EXPECT_EQ(t.layout_output_idx, 0);
  EXPECT_EQ(t.forward_path.size(), 0);
  EXPECT_EQ(t.backward_path.size(), 0);

  t = annotations.tensor_allocation_map.at(std::make_pair(ip3, 0));
  EXPECT_EQ(t.tgt, bn);
  EXPECT_EQ(t.input_index, 2);
  EXPECT_EQ(t.layout, conv);
  EXPECT_EQ(t.layout_output_idx, 0);
  EXPECT_EQ(t.forward_path.size(), 0);
  EXPECT_EQ(t.backward_path.size(), 0);

  t = annotations.tensor_allocation_map.at(std::make_pair(ip4, 0));
  EXPECT_EQ(t.tgt, subtract);
  EXPECT_EQ(t.input_index, 1);
  EXPECT_EQ(t.layout, bn);
  EXPECT_EQ(t.layout_output_idx, 2);
  EXPECT_EQ(t.forward_path.size(), 1);
  EXPECT_EQ(t.forward_path[0], gte);
  EXPECT_EQ(t.backward_path.size(), 0);
}

TEST_F(AllocationFinderTest, ForwardAllocationCustomPoplibsOp) {
  // Check that the layout gets forwarded to a custom op.
  std::string hlo = R"(
HloModule top
ENTRY %top (arg0.78.22: f32[1,4,4,2], arg1: f32[1,1,2,2], arg2: f32[2], arg3: f32[2], arg3: f32[2]) -> (f32[1,4,4,2]{3,2,1,0}, f32[2]{0}, f32[2]{0}) {
  %arg0 = f32[1,4,4,2]{3,2,1,0} parameter(0)
  %arg1 = f32[1,1,2,2]{3,2,1,0} parameter(1)
  %convolution = f32[1,4,4,2]{3,2,1,0} convolution(f32[1,4,4,2]{3,2,1,0} %arg0, f32[1,1,2,2]{3,2,1,0} %arg1), window={size=1x1}, dim_labels=b01f_01io->b01f
  %arg2 = f32[2]{0} parameter(2)
  %arg3 = f32[2]{0} parameter(3)
  ROOT %cc = (f32[1,4,4,2]{3,2,1,0}, f32[2]{0}, f32[2]{0}) custom-call(f32[1,4,4,2]{3,2,1,0} %convolution, f32[2]{0} %arg2, f32[2]{0} %arg3), custom_call_target="Popnn::GroupNormInference", opaque="{\"allocating_indexes\":[],\"layout_dependencies\":{\"keys\":[1,2],\"values\":[0,0]},\"epsilon\":0.001,\"feature_index\":3,\"num_inplace_operands\":0}\n"
}

)";

  auto config = GetModuleConfigForTest();
  auto module = ParseHloString(hlo, config);
  EXPECT_TRUE(module.ok());
  auto* module0 = module.ValueOrDie().get();

  const auto* root = module0->entry_computation()->root_instruction();
  const auto* custom_op = root;
  const auto* conv = custom_op->operand(0);
  const auto* ip2 = custom_op->operand(1);
  const auto* ip3 = custom_op->operand(2);
  const auto* ip1 = conv->operand(1);
  const auto* ip0 = conv->operand(0);

  CompilerAnnotations annotations(module0);

  AllocationFinder finder(annotations);
  EXPECT_TRUE(finder.Run(module0).ValueOrDie());

  // Will have both of the convolution parameters
  EXPECT_EQ(annotations.tensor_allocation_map.size(), 2);

  auto t = annotations.tensor_allocation_map.at(std::make_pair(ip0, 0));
  EXPECT_EQ(t.tgt, conv);
  EXPECT_EQ(t.input_index, 0);

  t = annotations.tensor_allocation_map.at(std::make_pair(ip1, 0));
  EXPECT_EQ(t.tgt, conv);
  EXPECT_EQ(t.input_index, 1);

  ForwardAllocation fwd_finder(annotations);
  unsigned num_succesful_runs = 0;
  while (fwd_finder.Run(module0).ValueOrDie()) {
    num_succesful_runs++;
  }

  // Depending on the order we either expect this to be executed successfully 1
  // or 2 times.
  EXPECT_TRUE(num_succesful_runs == 1 || num_succesful_runs == 2);

  // We have added one new entry for the bias add
  EXPECT_EQ(annotations.tensor_allocation_map.size(), 4);

  t = annotations.tensor_allocation_map.at(std::make_pair(ip2, 0));
  EXPECT_EQ(t.tgt, custom_op);
  EXPECT_EQ(t.input_index, 1);
  EXPECT_EQ(t.layout, conv);
  EXPECT_EQ(t.layout_output_idx, 0);
  EXPECT_EQ(t.forward_path.size(), 0);
  EXPECT_EQ(t.backward_path.size(), 0);

  t = annotations.tensor_allocation_map.at(std::make_pair(ip3, 0));
  EXPECT_EQ(t.tgt, custom_op);
  EXPECT_EQ(t.input_index, 2);
  EXPECT_EQ(t.layout, conv);
  EXPECT_EQ(t.layout_output_idx, 0);
  EXPECT_EQ(t.forward_path.size(), 0);
  EXPECT_EQ(t.backward_path.size(), 0);
}

TEST_F(AllocationFinderTest, FindInfeedAllocation) {
  // Check that allocation finder works with infeed non-tuple.
  std::string hlo = R"(
HloModule top

%Sum-reduction.7 (x.8: f32[], y.9: f32[]) -> f32[] {
  %x.8 = f32[] parameter(0)
  %y.9 = f32[] parameter(1)
  ROOT %add.10 = f32[] add(f32[] %x.8, f32[] %y.9)
}

%_body (arg_tuple.0: (s32[], f32[], f32[1,1,2,2])) -> (s32[], f32[], f32[1,1,2,2]) {
  %arg_tuple.0 = (s32[], f32[], f32[1,1,2,2]{3,2,1,0}) parameter(0)
  %get-tuple-element.3 = s32[] get-tuple-element((s32[], f32[], f32[1,1,2,2]{3,2,1,0}) %arg_tuple.0), index=0
  %get-tuple-element.4 = f32[1,1,2,2]{3,2,1,0} get-tuple-element((s32[], f32[], f32[1,1,2,2]{3,2,1,0}) %arg_tuple.0), index=2
  %constant.6 = f32[] constant(0)
  %after-all = token[] after-all()
  %infeed = ((f32[2,4,4,2]{3,2,1,0}), token[]) infeed(token[] %after-all), infeed_config="140121807314576"
  %get-tuple-element.5 = (f32[2,4,4,2]{3,2,1,0}) get-tuple-element(((f32[2,4,4,2]{3,2,1,0}), token[]) %infeed), index=0
  %get-tuple-element.6 = f32[2,4,4,2]{3,2,1,0} get-tuple-element((f32[2,4,4,2]{3,2,1,0}) %get-tuple-element.5), index=0
  %convolution = f32[2,4,4,2]{3,2,1,0} convolution(f32[2,4,4,2]{3,2,1,0} %get-tuple-element.6, f32[1,1,2,2]{3,2,1,0} %get-tuple-element.4), window={size=1x1}, dim_labels=b01f_01io->b01f
  %reduce = f32[] reduce(f32[2,4,4,2]{3,2,1,0} %convolution, f32[] %constant.6), dimensions={0,1,2,3}, to_apply=%Sum-reduction.7
  ROOT %tuple.1 = (s32[], f32[], f32[1,1,2,2]{3,2,1,0}) tuple(s32[] %get-tuple-element.3, f32[] %reduce, f32[1,1,2,2]{3,2,1,0} %get-tuple-element.4)
}

ENTRY %top (arg0.1: f32[1,1,2,2]) -> f32[] {
  %constant.7 = s32[] constant(100)
  %constant.5 = f32[] constant(0)
  %arg0.1 = f32[1,1,2,2]{3,2,1,0} parameter(0)
  %tuple.6.clone = (s32[], f32[], f32[1,1,2,2]{3,2,1,0}) tuple(s32[] %constant.7, f32[] %constant.5, f32[1,1,2,2]{3,2,1,0} %arg0.1)
  %call = (s32[], f32[], f32[1,1,2,2]{3,2,1,0}) call((s32[], f32[], f32[1,1,2,2]{3,2,1,0}) %tuple.6.clone), to_apply=%_body, backend_config="{\"repeatConfig\":{\"isRepeatLoop\":true,\"repeatCount\":\"100\"}}"
  ROOT %get-tuple-element.45 = f32[] get-tuple-element((s32[], f32[], f32[1,1,2,2]{3,2,1,0}) %call), index=1
}

)";

  auto config = GetModuleConfigForTest();
  auto module = ParseHloString(hlo, config);
  EXPECT_TRUE(module.ok());
  auto* module0 = module.ValueOrDie().get();

  const auto* root = module0->entry_computation()->root_instruction();
  const auto* repeat_loop = root->operand(0);
  const auto* ip_weights = repeat_loop->operand(0)->operand(2);
  const auto* repeat_body = repeat_loop->to_apply();
  const auto* repeat_root = repeat_body->root_instruction();
  const auto* reduce = repeat_root->operand(1);
  const auto* convolution = reduce->operand(0);
  const auto* conv_input = convolution->operand(0);
  const auto* conv_weights = convolution->operand(1);
  const auto* infeed = conv_input->operand(0)->operand(0);
  const auto* repeat_tuple = conv_weights->operand(0);

  CompilerAnnotations annotations(module0);

  AllocationFinder finder(annotations);
  EXPECT_TRUE(finder.Run(module0).ValueOrDie());

  // Will have both of the convolution parameters
  EXPECT_EQ(annotations.tensor_allocation_map.size(), 3);

  auto t = annotations.tensor_allocation_map.at(std::make_pair(infeed, 0));
  EXPECT_EQ(t.tgt, convolution);
  EXPECT_EQ(t.input_index, 0);

  t = annotations.tensor_allocation_map.at(std::make_pair(ip_weights, 0));
  EXPECT_EQ(t.tgt, convolution);
  EXPECT_EQ(t.input_index, 1);

  t = annotations.tensor_allocation_map.at(std::make_pair(repeat_tuple, 2));
  EXPECT_EQ(t.tgt, convolution);
  EXPECT_EQ(t.input_index, 1);
}

TEST_F(AllocationFinderTest, FindInfeedAllocationTuple) {
  // Check that allocation finder works with infeed tuple.
  std::string hlo = R"(
HloModule top

%Sum-reduction.6 (x.7: f32[], y.8: f32[]) -> f32[] {
  %x.7 = f32[] parameter(0)
  %y.8 = f32[] parameter(1)
  ROOT %add.9 = f32[] add(f32[] %x.7, f32[] %y.8)
}

%_body (arg_tuple.0: (s32[], f32[])) -> (s32[], f32[]) {
  %arg_tuple.0 = (s32[], f32[]) parameter(0)
  %get-tuple-element.2 = s32[] get-tuple-element((s32[], f32[]) %arg_tuple.0), index=0
  %constant.5 = f32[] constant(0)
  %after-all = token[] after-all()
  %infeed = ((f32[2,4,4,2]{3,2,1,0}, f32[1,1,2,2]{3,2,1,0}), token[]) infeed(token[] %after-all), infeed_config="140227418928528"
  %get-tuple-element.3 = (f32[2,4,4,2]{3,2,1,0}, f32[1,1,2,2]{3,2,1,0}) get-tuple-element(((f32[2,4,4,2]{3,2,1,0}, f32[1,1,2,2]{3,2,1,0}), token[]) %infeed), index=0
  %get-tuple-element.4 = f32[2,4,4,2]{3,2,1,0} get-tuple-element((f32[2,4,4,2]{3,2,1,0}, f32[1,1,2,2]{3,2,1,0}) %get-tuple-element.3), index=0
  %get-tuple-element.5 = f32[1,1,2,2]{3,2,1,0} get-tuple-element((f32[2,4,4,2]{3,2,1,0}, f32[1,1,2,2]{3,2,1,0}) %get-tuple-element.3), index=1
  %convolution = f32[2,4,4,2]{3,2,1,0} convolution(f32[2,4,4,2]{3,2,1,0} %get-tuple-element.4, f32[1,1,2,2]{3,2,1,0} %get-tuple-element.5), window={size=1x1}, dim_labels=b01f_01io->b01f
  %reduce = f32[] reduce(f32[2,4,4,2]{3,2,1,0} %convolution, f32[] %constant.5), dimensions={0,1,2,3}, to_apply=%Sum-reduction.6
  ROOT %tuple.1 = (s32[], f32[]) tuple(s32[] %get-tuple-element.2, f32[] %reduce)
}

ENTRY %top () -> f32[] {
  %constant.7 = s32[] constant(100)
  %constant.4 = f32[] constant(0)
  %tuple.5.clone = (s32[], f32[]) tuple(s32[] %constant.7, f32[] %constant.4)
  %call = (s32[], f32[]) call((s32[], f32[]) %tuple.5.clone), to_apply=%_body, backend_config="{\"repeatConfig\":{\"isRepeatLoop\":true,\"repeatCount\":\"100\"}}"
  ROOT %get-tuple-element.41 = f32[] get-tuple-element((s32[], f32[]) %call), index=1
}

)";

  auto config = GetModuleConfigForTest();
  auto module = ParseHloString(hlo, config);
  EXPECT_TRUE(module.ok());
  auto* module0 = module.ValueOrDie().get();

  const auto* root = module0->entry_computation()->root_instruction();
  const auto* repeat_loop = root->operand(0);
  const auto* repeat_body = repeat_loop->to_apply();
  const auto* repeat_root = repeat_body->root_instruction();
  const auto* reduce = repeat_root->operand(1);
  const auto* convolution = reduce->operand(0);
  const auto* conv_input = convolution->operand(0);
  const auto* infeed = conv_input->operand(0)->operand(0);

  CompilerAnnotations annotations(module0);

  AllocationFinder finder(annotations);
  EXPECT_TRUE(finder.Run(module0).ValueOrDie());

  // Will have both of the convolution parameters
  EXPECT_EQ(annotations.tensor_allocation_map.size(), 2);

  auto t = annotations.tensor_allocation_map.at(std::make_pair(infeed, 0));
  EXPECT_EQ(t.tgt, convolution);
  EXPECT_EQ(t.input_index, 0);

  t = annotations.tensor_allocation_map.at(std::make_pair(infeed, 1));
  EXPECT_EQ(t.tgt, convolution);
  EXPECT_EQ(t.input_index, 1);
}

TEST_F(AllocationFinderTest, InputTupleBiasAdd) {
  std::string hlo = R"(
HloModule top

 %_pop_op_matmul_biasadd (arg_0: f32[2,2], arg_1: f32[2]) -> f32[2,2] {
   %arg_1 = f32[2] parameter(1)
   %broadcast.12.7.clone = f32[2,2]{1,0} broadcast(%arg_1), dimensions={1}
   %arg_0 = f32[2,2]{1,0} parameter(0)
   ROOT %add.12.8.clone = f32[2,2]{1,0} add(f32[2,2]{1,0} %arg_0, f32[2,2]{1,0} %broadcast.12.7.clone)
 }

 ENTRY %c (arg0.12.0: (f32[2,2], f32[2,2], f32[2])) -> f32[2,2] {
   %arg0 = (f32[2,2]{1,0}, f32[2,2]{1,0}, f32[2]) parameter(0)
   %gte0 = f32[2,2]{1,0} get-tuple-element((f32[2,2]{1,0}, f32[2,2]{1,0}, f32[2]) %arg0), index=0
   %gte1 = f32[2,2]{1,0} get-tuple-element((f32[2,2]{1,0}, f32[2,2]{1,0}, f32[2]) %arg0), index=1
   %dot.12.6 = f32[2,2]{1,0} dot(f32[2,2]{1,0} %gte0, f32[2,2]{1,0} %gte1), lhs_contracting_dims={1}, rhs_contracting_dims={0}
   %gte2 = f32[2] get-tuple-element((f32[2,2]{1,0}, f32[2,2]{1,0}, f32[2]) %arg0), index=2
   ROOT %call = f32[2,2]{1,0} fusion(f32[2,2]{1,0} %dot.12.6, %gte2), kind=kCustom, calls=%_pop_op_matmul_biasadd
 }

)";

  auto config = GetModuleConfigForTest();
  auto module = ParseHloString(hlo, config);
  EXPECT_TRUE(module.ok());
  auto* module0 = module.ValueOrDie().get();

  const auto* root = module0->entry_computation()->root_instruction();
  const auto* call = root;
  const auto* dot = call->operand(0);
  const auto* gte0 = dot->operand(0);
  const auto* gte1 = dot->operand(1);
  const auto* gte2 = call->operand(1);
  const auto* ip_tuple = gte0->operand(0);
  EXPECT_EQ(ip_tuple, gte1->operand(0));
  EXPECT_EQ(ip_tuple, gte2->operand(0));

  CompilerAnnotations annotations(module0);

  InplaceFinder inplace_finder(annotations);
  EXPECT_TRUE(inplace_finder.Run(module0).ValueOrDie());

  AllocationFinder finder(annotations);
  EXPECT_TRUE(finder.Run(module0).ValueOrDie());

  // Will have both of the dot parameters
  EXPECT_EQ(annotations.tensor_allocation_map.size(), 2);

  auto t = annotations.tensor_allocation_map.at(std::make_pair(ip_tuple, 0));
  EXPECT_EQ(t.tgt, dot);
  EXPECT_EQ(t.input_index, 0);

  t = annotations.tensor_allocation_map.at(std::make_pair(ip_tuple, 1));
  EXPECT_EQ(t.tgt, dot);
  EXPECT_EQ(t.input_index, 1);

  ForwardAllocation fwd_finder(annotations);
  EXPECT_TRUE(fwd_finder.Run(module0).ValueOrDie());

  // We have added one new entry for the bias add
  EXPECT_EQ(annotations.tensor_allocation_map.size(), 3);

  t = annotations.tensor_allocation_map.at(std::make_pair(gte2, 0));
  EXPECT_EQ(t.tgt, call);
  EXPECT_EQ(t.input_index, 1);
  EXPECT_EQ(t.layout, dot);
  EXPECT_EQ(t.layout_output_idx, 0);
  EXPECT_EQ(t.forward_path.size(), 0);
  EXPECT_EQ(t.backward_path.size(), 0);
  EXPECT_EQ(t.deferred_allocations_path.size(), 1);
  DeferredAllocationsPath expected_deferred_allocations_path = {
      std::make_pair(ip_tuple, 2)};
  EXPECT_EQ(t.deferred_allocations_path, expected_deferred_allocations_path);
  DeferredAllocations expected_deferred_allocations(
      expected_deferred_allocations_path.begin(),
      expected_deferred_allocations_path.end());
  EXPECT_EQ(annotations.deferred_allocations, expected_deferred_allocations);
}

TEST_F(AllocationFinderTest, InputTupleInfeedBiasAdd) {
  std::string hlo = R"(
HloModule top

 %_pop_op_matmul_biasadd (arg_0: f32[2,2], arg_1: f32[2]) -> f32[2,2] {
   %arg_1 = f32[2] parameter(1)
   %broadcast.12.7.clone = f32[2,2]{1,0} broadcast(%arg_1), dimensions={1}
   %arg_0 = f32[2,2]{1,0} parameter(0)
   ROOT %add.12.8.clone = f32[2,2]{1,0} add(f32[2,2]{1,0} %arg_0, f32[2,2]{1,0} %broadcast.12.7.clone)
 }

 ENTRY %c () -> f32[2,2] {
   %after-all = token[] after-all()
   %infeed = ((f32[2,2]{1,0}, f32[2,2]{1,0}, f32[2]), token[]) infeed(token[] %after-all), infeed_config="140227418928528"
   %arg0 = (f32[2,2]{1,0}, f32[2,2]{1,0}, f32[2]) get-tuple-element(((f32[2,2]{1,0}, f32[2,2]{1,0}, f32[2]), token[]) %infeed), index=0
   %gte0 = f32[2,2]{1,0} get-tuple-element((f32[2,2]{1,0}, f32[2,2]{1,0}, f32[2]) %arg0), index=0
   %gte1 = f32[2,2]{1,0} get-tuple-element((f32[2,2]{1,0}, f32[2,2]{1,0}, f32[2]) %arg0), index=1
   %dot.12.6 = f32[2,2]{1,0} dot(f32[2,2]{1,0} %gte0, f32[2,2]{1,0} %gte1), lhs_contracting_dims={1}, rhs_contracting_dims={0}
   %gte2 = f32[2] get-tuple-element((f32[2,2]{1,0}, f32[2,2]{1,0}, f32[2]) %arg0), index=2
   ROOT %call = f32[2,2]{1,0} fusion(f32[2,2]{1,0} %dot.12.6, %gte2), kind=kCustom, calls=%_pop_op_matmul_biasadd
 }

)";

  auto config = GetModuleConfigForTest();
  auto module = ParseHloString(hlo, config);
  EXPECT_TRUE(module.ok());
  auto* module0 = module.ValueOrDie().get();

  const auto* root = module0->entry_computation()->root_instruction();
  const auto* call = root;
  const auto* dot = call->operand(0);
  const auto* gte0 = dot->operand(0);
  const auto* gte1 = dot->operand(1);
  const auto* gte2 = call->operand(1);
  const auto* ip_tuple = gte0->operand(0);
  EXPECT_EQ(ip_tuple, gte1->operand(0));
  EXPECT_EQ(ip_tuple, gte2->operand(0));
  const auto* infeed = ip_tuple->operand(0);

  CompilerAnnotations annotations(module0);

  InplaceFinder inplace_finder(annotations);
  EXPECT_TRUE(inplace_finder.Run(module0).ValueOrDie());

  AllocationFinder finder(annotations);
  EXPECT_TRUE(finder.Run(module0).ValueOrDie());

  // Will have both of the dot parameters
  EXPECT_EQ(annotations.tensor_allocation_map.size(), 2);
  auto t = annotations.tensor_allocation_map.at(std::make_pair(infeed, 0));
  EXPECT_EQ(t.tgt, dot);
  EXPECT_EQ(t.input_index, 0);

  t = annotations.tensor_allocation_map.at(std::make_pair(infeed, 1));
  EXPECT_EQ(t.tgt, dot);
  EXPECT_EQ(t.input_index, 1);

  ForwardAllocation fwd_finder(annotations);
  EXPECT_TRUE(fwd_finder.Run(module0).ValueOrDie());

  // We have added one new entry for the bias add
  EXPECT_EQ(annotations.tensor_allocation_map.size(), 3);

  t = annotations.tensor_allocation_map.at(std::make_pair(gte2, 0));
  EXPECT_EQ(t.tgt, call);
  EXPECT_EQ(t.input_index, 1);
  EXPECT_EQ(t.layout, dot);
  EXPECT_EQ(t.layout_output_idx, 0);
  EXPECT_EQ(t.forward_path.size(), 0);
  EXPECT_EQ(t.backward_path.size(), 0);
  EXPECT_EQ(t.deferred_allocations_path.size(), 2);
  DeferredAllocationsPath expected_deferred_allocations_path = {
      std::make_pair(ip_tuple, 2), std::make_pair(infeed, 2)};
  EXPECT_EQ(t.deferred_allocations_path, expected_deferred_allocations_path);
  DeferredAllocations expected_deferred_allocations(
      expected_deferred_allocations_path.begin(),
      expected_deferred_allocations_path.end());
  EXPECT_EQ(annotations.deferred_allocations, expected_deferred_allocations);
}

TEST_F(AllocationFinderTest, NestedInputTupleBatchNormInfParamsWithPath) {
  std::string hlo = R"(
HloModule top

ENTRY %top (arg0: (f32[1,4,4,2], f32[1,1,2,2], (f32[1,2], f32[1,2]), f32[2], f32[2])) -> f32[1,4,4,2] {
 %arg0 = (f32[1,4,4,2], f32[1,1,2,2], (f32[1,2], f32[1,2]), f32[2], f32[2]) parameter(0)
 %gte0 = f32[1,4,4,2] get-tuple-element((f32[1,4,4,2], f32[1,1,2,2], (f32[1,2], f32[1,2]), f32[2], f32[2]) %arg0), index=0
 %gte1 = f32[1,1,2,2] get-tuple-element((f32[1,4,4,2], f32[1,1,2,2], (f32[1,2], f32[1,2]), f32[2], f32[2]) %arg0), index=1
 %convolution.36.29 = f32[1,4,4,2] convolution(%gte0, %gte1), window={size=1x1}, dim_labels=b01f_01io->b01f
 %gte2 = (f32[1,2], f32[1,2]) get-tuple-element((f32[1,4,4,2], f32[1,1,2,2], (f32[1,2], f32[1,2]), f32[2], f32[2]) %arg0), index=2
 %gte2.0 = f32[1,2] get-tuple-element((f32[1,2], f32[1,2]) %gte2), index=0
 %gte2.0_r = f32[2] reshape(%gte2.0)
 %gte2.1 = f32[1,2] get-tuple-element((f32[1,2], f32[1,2]) %gte2), index=1
 %gte2.1_r = f32[2] reshape(%gte2.1)
 %gte3 = f32[2] get-tuple-element((f32[1,4,4,2], f32[1,1,2,2], (f32[1,2], f32[1,2]), f32[2], f32[2]) %arg0), index=3
 %gte4 = f32[2] get-tuple-element((f32[1,4,4,2], f32[1,1,2,2], (f32[1,2], f32[1,2]), f32[2], f32[2]) %arg0), index=4
 ROOT %batch-norm-inference.36.31 = f32[1,4,4,2] batch-norm-inference(%convolution.36.29, %gte2.0_r, %gte2.1_r, %gte3, %gte4), epsilon=0.001, feature_index=3
}

)";

  auto config = GetModuleConfigForTest();
  auto module = ParseHloString(hlo, config);
  EXPECT_TRUE(module.ok());
  auto* module0 = module.ValueOrDie().get();

  const auto* root = module0->entry_computation()->root_instruction();
  const auto* bn = root;
  const auto* conv = bn->operand(0);
  const auto* ip0 = conv->operand(0);
  const auto* ip1 = conv->operand(1);
  const auto* reshape1 = bn->operand(1);
  const auto* reshape2 = bn->operand(2);
  const auto* ip2_0 = reshape1->operand(0);
  const auto* ip2_1 = reshape2->operand(0);
  const auto* nested_tuple = ip2_0->operand(0);
  CHECK_EQ(nested_tuple, ip2_1->operand(0));
  const auto* arg_tuple = ip0->operand(0);
  CHECK_EQ(arg_tuple, ip1->operand(0));
  CHECK_EQ(arg_tuple, nested_tuple->operand(0));

  CompilerAnnotations annotations(module0);

  InplaceFinder inplace_finder(annotations);
  EXPECT_TRUE(inplace_finder.Run(module0).ValueOrDie());

  AllocationFinder finder(annotations);
  EXPECT_TRUE(finder.Run(module0).ValueOrDie());

  // Will have both of the convolution parameters
  EXPECT_EQ(annotations.tensor_allocation_map.size(), 2);

  auto t = annotations.tensor_allocation_map.at(std::make_pair(arg_tuple, 0));
  EXPECT_EQ(t.tgt, conv);
  EXPECT_EQ(t.input_index, 0);

  t = annotations.tensor_allocation_map.at(std::make_pair(arg_tuple, 1));
  EXPECT_EQ(t.tgt, conv);
  EXPECT_EQ(t.input_index, 1);

  ForwardAllocation fwd_finder(annotations);
  EXPECT_TRUE(fwd_finder.Run(module0).ValueOrDie());

  // We have added one new entry for the bias add
  EXPECT_EQ(annotations.tensor_allocation_map.size(), 4);

  t = annotations.tensor_allocation_map.at(std::make_pair(ip2_0, 0));
  EXPECT_EQ(t.tgt, bn);
  EXPECT_EQ(t.input_index, 1);
  EXPECT_EQ(t.layout, conv);
  EXPECT_EQ(t.layout_output_idx, 0);
  EXPECT_EQ(t.forward_path.size(), 0);
  EXPECT_EQ(t.backward_path.size(), 1);
  EXPECT_EQ(t.backward_path[0], reshape1);
  EXPECT_EQ(t.deferred_allocations_path.size(), 2);
  DeferredAllocationsPath expected_deferred_allocations_path0 = {
      std::make_pair(nested_tuple, 0), std::make_pair(arg_tuple, 2)};
  EXPECT_EQ(t.deferred_allocations_path, expected_deferred_allocations_path0);

  t = annotations.tensor_allocation_map.at(std::make_pair(ip2_1, 0));
  EXPECT_EQ(t.tgt, bn);
  EXPECT_EQ(t.input_index, 2);
  EXPECT_EQ(t.layout, conv);
  EXPECT_EQ(t.layout_output_idx, 0);
  EXPECT_EQ(t.forward_path.size(), 0);
  EXPECT_EQ(t.backward_path.size(), 1);
  EXPECT_EQ(t.backward_path[0], reshape2);
  EXPECT_EQ(t.deferred_allocations_path.size(), 2);
  DeferredAllocationsPath expected_deferred_allocations_path1 = {
      std::make_pair(nested_tuple, 1), std::make_pair(arg_tuple, 3)};
  EXPECT_EQ(t.deferred_allocations_path, expected_deferred_allocations_path1);

  DeferredAllocations expected_deferred_allocations(
      expected_deferred_allocations_path0.begin(),
      expected_deferred_allocations_path0.end());
  expected_deferred_allocations.insert(
      expected_deferred_allocations_path1.begin(),
      expected_deferred_allocations_path1.end());
  EXPECT_EQ(annotations.deferred_allocations, expected_deferred_allocations);
}

// TODO:
// - can forward path traverse in-place ops
// - is forward path rejected when going through non-layout preserving inputs

}  // namespace
}  // namespace poplarplugin
}  // namespace xla

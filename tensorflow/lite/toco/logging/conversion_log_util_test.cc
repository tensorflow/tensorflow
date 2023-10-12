/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/lite/toco/logging/conversion_log_util.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/memory/memory.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/lite/toco/model.h"
#include "tensorflow/lite/toco/model_flags.pb.h"

namespace toco {
namespace {

using ::testing::ElementsAre;
using ::testing::UnorderedElementsAre;

TEST(ConversionLogUtilTest, TestGetOperatorNames) {
  Model model;
  // Built-in ops.
  model.operators.push_back(std::make_unique<ConvOperator>());
  model.operators.push_back(std::make_unique<MeanOperator>());
  model.operators.push_back(std::make_unique<NegOperator>());
  // Flex ops.
  auto avg_pool_3d = std::make_unique<TensorFlowUnsupportedOperator>();
  avg_pool_3d->tensorflow_op = "AvgPool3D";
  tensorflow::NodeDef node_def;
  node_def.set_op("AvgPool3D");
  node_def.SerializeToString(&avg_pool_3d->tensorflow_node_def);
  model.operators.push_back(std::move(avg_pool_3d));
  // Custom ops.
  auto my_custom_op = std::make_unique<TensorFlowUnsupportedOperator>();
  my_custom_op->tensorflow_op = "MyAwesomeCustomOp";
  model.operators.push_back(std::move(my_custom_op));

  const auto& output = GetOperatorNames(model);
  EXPECT_THAT(output, ElementsAre("Conv", "Mean", "Neg", "AvgPool3D",
                                  "MyAwesomeCustomOp"));
}

TEST(ConversionLogUtilTest, TestCountOperatorsByType) {
  Model model;
  // 1st Conv operator.
  std::unique_ptr<ConvOperator> conv1(new ConvOperator());
  const std::string conv1_input_name = "conv_input1";
  const std::string conv1_filter_name = "conv_filter1";
  const std::string conv1_output_name = "conv_output1";
  conv1->inputs.push_back(conv1_input_name);
  conv1->inputs.push_back(conv1_filter_name);
  conv1->outputs.push_back(conv1_output_name);
  auto& array_map = model.GetMutableArrayMap();
  array_map[conv1_input_name] = std::make_unique<Array>();
  array_map[conv1_filter_name] = std::make_unique<Array>();
  array_map[conv1_output_name] = std::make_unique<Array>();

  // 2nd Conv operator.
  std::unique_ptr<ConvOperator> conv2(new ConvOperator());
  const std::string conv2_input_name = "conv_input2";
  const std::string conv2_filter_name = "conv_filter2";
  const std::string conv2_output_name = "conv_output2";
  conv2->inputs.push_back(conv2_input_name);
  conv2->inputs.push_back(conv2_filter_name);
  conv2->outputs.push_back(conv2_output_name);
  array_map[conv2_input_name] = std::make_unique<Array>();
  array_map[conv2_filter_name] = std::make_unique<Array>();
  array_map[conv2_output_name] = std::make_unique<Array>();

  // Mean operator.
  std::unique_ptr<MeanOperator> mean(new MeanOperator());
  const std::string mean_input_name = "mean_input";
  mean->inputs.push_back(mean_input_name);
  array_map[mean_input_name] = std::make_unique<Array>();

  // 1st flex operator 'AvgPool3D'.
  auto avg_pool_3d = std::make_unique<TensorFlowUnsupportedOperator>();
  avg_pool_3d->tensorflow_op = "AvgPool3D";
  tensorflow::NodeDef node_def;
  node_def.set_op("AvgPool3D");
  node_def.SerializeToString(&avg_pool_3d->tensorflow_node_def);

  // 2nd flex operator 'EluGrad'.
  auto elu_grad = std::make_unique<TensorFlowUnsupportedOperator>();
  elu_grad->tensorflow_op = "EluGrad";
  node_def.set_op("EluGrad");
  node_def.SerializeToString(&elu_grad->tensorflow_node_def);

  // 1st custom operator 'MyAwesomeCustomOp'.
  auto my_custom_op = std::make_unique<TensorFlowUnsupportedOperator>();
  my_custom_op->tensorflow_op = "MyAwesomeCustomOp";

  model.operators.push_back(std::move(conv1));
  model.operators.push_back(std::move(conv2));
  model.operators.push_back(std::move(mean));
  model.operators.push_back(std::move(avg_pool_3d));
  model.operators.push_back(std::move(elu_grad));
  model.operators.push_back(std::move(my_custom_op));

  std::map<std::string, int> built_in_ops, select_ops, custom_ops;
  CountOperatorsByType(model, &built_in_ops, &custom_ops, &select_ops);

  EXPECT_THAT(built_in_ops,
              UnorderedElementsAre(std::pair<std::string, int>("Conv", 2),
                                   std::pair<std::string, int>("Mean", 1)));
  EXPECT_THAT(select_ops,
              UnorderedElementsAre(std::pair<std::string, int>("AvgPool3D", 1),
                                   std::pair<std::string, int>("EluGrad", 1)));
  EXPECT_THAT(custom_ops, UnorderedElementsAre(std::pair<std::string, int>(
                              "MyAwesomeCustomOp", 1)));
}

TEST(ConversionLogUtilTest, TestGetInputAndOutputTypes) {
  Model model;
  auto& array_map = model.GetMutableArrayMap();
  const std::string input1 = "conv_input";
  const std::string input2 = "conv_filter";
  const std::string input3 = "feature";
  const std::string output = "softmax";
  array_map[input1] = std::make_unique<Array>();
  array_map[input1]->data_type = ArrayDataType::kFloat;
  array_map[input2] = std::make_unique<Array>();
  array_map[input2]->data_type = ArrayDataType::kFloat;
  array_map[input3] = std::make_unique<Array>();
  array_map[input3]->data_type = ArrayDataType::kInt16;
  array_map[output] = std::make_unique<Array>();
  array_map[output]->data_type = ArrayDataType::kFloat;

  InputArray input_arrays[3];
  input_arrays[0].set_name(input1);
  input_arrays[1].set_name(input2);
  input_arrays[2].set_name(input3);
  *model.flags.add_input_arrays() = input_arrays[0];
  *model.flags.add_input_arrays() = input_arrays[1];
  *model.flags.add_input_arrays() = input_arrays[2];
  model.flags.add_output_arrays(output);

  TFLITE_PROTO_NS::RepeatedPtrField<std::string> input_types, output_types;
  GetInputAndOutputTypes(model, &input_types, &output_types);

  EXPECT_THAT(input_types, ElementsAre("float", "float", "int16"));
  EXPECT_THAT(output_types, ElementsAre("float"));
}

TEST(ConversionLogUtilTest, TestGetOpSignatures) {
  Model model;
  auto& array_map = model.GetMutableArrayMap();

  std::unique_ptr<ConvOperator> conv(new ConvOperator());
  const std::string conv_input_name = "conv_input";
  const std::string conv_filter_name = "conv_filter";
  const std::string conv_output_name = "conv_output";
  conv->inputs.push_back(conv_input_name);
  conv->inputs.push_back(conv_filter_name);
  conv->outputs.push_back(conv_output_name);
  array_map[conv_input_name] = std::make_unique<Array>();
  array_map[conv_input_name]->data_type = ArrayDataType::kFloat;
  array_map[conv_input_name]->copy_shape({4, 4, 3});
  array_map[conv_filter_name] = std::make_unique<Array>();
  array_map[conv_filter_name]->data_type = ArrayDataType::kFloat;
  array_map[conv_filter_name]->copy_shape({2, 2});
  array_map[conv_output_name] = std::make_unique<Array>();
  array_map[conv_output_name]->data_type = ArrayDataType::kFloat;
  array_map[conv_output_name]->copy_shape({4, 4, 2});

  const std::string mean_input_name = "mean_input";
  const std::string mean_output_name = "mean_output";
  std::unique_ptr<MeanOperator> mean(new MeanOperator());
  mean->inputs.push_back(mean_input_name);
  mean->outputs.push_back(mean_output_name);
  array_map[mean_input_name] = std::make_unique<Array>();
  array_map[mean_output_name] = std::make_unique<Array>();

  const std::string avg_pool_3d_output_name = "avg_pool_output";
  auto avg_pool_3d = std::make_unique<TensorFlowUnsupportedOperator>();
  avg_pool_3d->tensorflow_op = "AvgPool3D";
  tensorflow::NodeDef node_def;
  node_def.set_op("AvgPool3D");
  node_def.SerializeToString(&avg_pool_3d->tensorflow_node_def);
  avg_pool_3d->inputs.push_back(conv_output_name);
  avg_pool_3d->outputs.push_back(avg_pool_3d_output_name);
  array_map[avg_pool_3d_output_name] = std::make_unique<Array>();
  array_map[avg_pool_3d_output_name]->data_type = ArrayDataType::kInt32;
  array_map[avg_pool_3d_output_name]->copy_shape({2, 2});

  const std::string custom_op_output_name = "custom_op_output";
  auto my_custom_op = std::make_unique<TensorFlowUnsupportedOperator>();
  my_custom_op->tensorflow_op = "MyAwesomeCustomOp";
  my_custom_op->inputs.push_back(avg_pool_3d_output_name);
  my_custom_op->outputs.push_back(custom_op_output_name);
  array_map[custom_op_output_name] = std::make_unique<Array>();
  array_map[custom_op_output_name]->data_type = ArrayDataType::kFloat;
  array_map[custom_op_output_name]->copy_shape({3});

  model.operators.push_back(std::move(conv));
  model.operators.push_back(std::move(mean));
  model.operators.push_back(std::move(avg_pool_3d));
  model.operators.push_back(std::move(my_custom_op));

  TFLITE_PROTO_NS::RepeatedPtrField<std::string> op_signatures;
  GetOpSignatures(model, &op_signatures);
  EXPECT_THAT(op_signatures,
              UnorderedElementsAre(
                  "INPUT:[4,4,3]::float::[2,2]::float::OUTPUT:[4,4,2]::float::"
                  "NAME:Conv::VERSION:1",
                  "INPUT:None::None::OUTPUT:None::None::NAME:Mean::VERSION:1",
                  "INPUT:[4,4,2]::float::OUTPUT:[2,2]::int32::NAME:AvgPool3D::"
                  "VERSION:1",
                  "INPUT:[2,2]::int32::OUTPUT:[3]::float::NAME:"
                  "MyAwesomeCustomOp::VERSION:1"));
}

TEST(ConversionLogUtilTest, TestSanitizeErrorMessage) {
  const std::string error =
      "error: failed while converting: 'main': Ops that can be supported by "
      "the flex runtime (enabled via setting the -emit-select-tf-ops flag): "
      "ResizeNearestNeighbor,ResizeNearestNeighbor. Ops that need custom "
      "implementation (enabled via setting the -emit-custom-ops flag): "
      "CombinedNonMaxSuppression.\nTraceback (most recent call last): File "
      "/usr/local/bin/toco_from_protos, line 8, in <module>";
  const std::string pruned_error =
      "Ops that can be supported by "
      "the flex runtime (enabled via setting the -emit-select-tf-ops flag): "
      "ResizeNearestNeighbor,ResizeNearestNeighbor.Ops that need custom "
      "implementation (enabled via setting the -emit-custom-ops flag): "
      "CombinedNonMaxSuppression.";
  EXPECT_EQ(SanitizeErrorMessage(error), pruned_error);
}

TEST(ConversionLogUtilTest, TestSanitizeErrorMessageNoMatching) {
  const std::string error =
      "error: failed while converting: 'main': Traceback (most recent call "
      "last): File "
      "/usr/local/bin/toco_from_protos, line 8, in <module>";
  EXPECT_EQ(SanitizeErrorMessage(error), "");
}

}  // namespace
}  // namespace toco

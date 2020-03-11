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
#include "tensorflow/lite/toco/tflite/op_version.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/lite/toco/model.h"

namespace toco {
namespace tflite {
namespace {

// TODO(b/150701120): port the tests to tools/versioning/op_version_test.cc.
TEST(OpVersionTest, MinimumVersionForSameOpVersions) {
  Model model;
  // Float convolutional kernel is introduced since '1.5.0'.
  std::unique_ptr<ConvOperator> conv(new ConvOperator());
  const string conv_input = "conv_input";
  const string conv_filter = "conv_filter";
  const string conv_output = "conv_output";
  conv->inputs.push_back(conv_input);
  conv->inputs.push_back(conv_filter);
  conv->outputs.push_back(conv_output);
  auto& array_map = model.GetMutableArrayMap();
  array_map[conv_input] = std::unique_ptr<Array>(new Array);
  array_map[conv_input]->data_type = ArrayDataType::kFloat;
  array_map[conv_filter] = std::unique_ptr<Array>(new Array);
  array_map[conv_filter]->data_type = ArrayDataType::kFloat;
  array_map[conv_output] = std::unique_ptr<Array>(new Array);
  array_map[conv_output]->data_type = ArrayDataType::kFloat;
  model.operators.push_back(std::move(conv));

  // Float softmax kernel is introduced since '1.5.0'.
  std::unique_ptr<SoftmaxOperator> softmax(new SoftmaxOperator());
  const string softmax_input = "softmax_input";
  const string softmax_output = "softmax_output";
  softmax->inputs.push_back(softmax_input);
  softmax->outputs.push_back(softmax_output);
  array_map[softmax_input] = std::unique_ptr<Array>(new Array);
  array_map[softmax_input]->data_type = ArrayDataType::kFloat;
  array_map[softmax_output] = std::unique_ptr<Array>(new Array);
  model.operators.push_back(std::move(softmax));

  EXPECT_EQ(GetMinimumRuntimeVersionForModel(model), "1.5.0");
}

TEST(OpVersionTest, MinimumVersionForMultipleOpVersions) {
  Model model;
  // Dilated DepthWiseConvolution is introduced since '1.12.0'.
  std::unique_ptr<DepthwiseConvOperator> conv(new DepthwiseConvOperator());
  const string conv_input = "conv_input";
  const string conv_filter = "conv_filter";
  const string conv_output = "conv_output";
  conv->inputs.push_back(conv_input);
  conv->inputs.push_back(conv_filter);
  conv->outputs.push_back(conv_output);
  auto& array_map = model.GetMutableArrayMap();
  array_map[conv_input] = std::unique_ptr<Array>(new Array);
  array_map[conv_filter] = std::unique_ptr<Array>(new Array);
  array_map[conv_output] = std::unique_ptr<Array>(new Array);
  conv->dilation_width_factor = 2;
  conv->dilation_height_factor = 2;
  model.operators.push_back(std::move(conv));

  // FullyConnected op with kShuffled4x16Int8 weight format is introduced from
  // '1.10.0'.
  std::unique_ptr<FullyConnectedOperator> fc(new FullyConnectedOperator());
  const string fc_input = "fc_input";
  const string fc_weights = "fc_weights";
  const string fc_bias = "fc_bias";
  const string fc_output = "fc_output";
  fc->inputs.push_back(fc_input);
  fc->inputs.push_back(fc_weights);
  fc->inputs.push_back(fc_bias);
  fc->outputs.push_back(fc_output);
  array_map[fc_input] = std::unique_ptr<Array>(new Array);
  array_map[fc_weights] = std::unique_ptr<Array>(new Array);
  array_map[fc_output] = std::unique_ptr<Array>(new Array);
  fc->weights_format = FullyConnectedWeightsFormat::kShuffled4x16Int8;
  model.operators.push_back(std::move(fc));

  EXPECT_EQ(GetMinimumRuntimeVersionForModel(model), "1.12.0");
}

TEST(OpVersionTest, MinimumVersionForEmptyOpVersions) {
  Model model;

  // my_custom_op_1 isn't associated with any runtime version.
  auto my_custom_op_1 = absl::make_unique<TensorFlowUnsupportedOperator>();
  my_custom_op_1->tensorflow_op = "MyAwesomeCustomOp1";
  model.operators.push_back(std::move(my_custom_op_1));

  // my_custom_op_2 isn't associated with any runtime version.
  auto my_custom_op_2 = absl::make_unique<TensorFlowUnsupportedOperator>();
  my_custom_op_2->tensorflow_op = "MyAwesomeCustomOp2";
  model.operators.push_back(std::move(my_custom_op_2));

  EXPECT_EQ(GetMinimumRuntimeVersionForModel(model), "");
}

TEST(OpVersionTest, MinimumVersionForMixedOpVersions) {
  Model model;

  // my_custom_op isn't associated with any runtime version.
  auto my_custom_op = absl::make_unique<TensorFlowUnsupportedOperator>();
  my_custom_op->tensorflow_op = "MyAwesomeCustomOp";
  model.operators.push_back(std::move(my_custom_op));

  // FullyConnected op with kShuffled4x16Int8 weight format is introduced from
  // '1.10.0'.
  std::unique_ptr<FullyConnectedOperator> fc(new FullyConnectedOperator());
  const string fc_input = "fc_input";
  const string fc_weights = "fc_weights";
  const string fc_bias = "fc_bias";
  const string fc_output = "fc_output";
  fc->inputs.push_back(fc_input);
  fc->inputs.push_back(fc_weights);
  fc->inputs.push_back(fc_bias);
  fc->outputs.push_back(fc_output);
  auto& array_map = model.GetMutableArrayMap();
  array_map[fc_input] = std::unique_ptr<Array>(new Array);
  array_map[fc_weights] = std::unique_ptr<Array>(new Array);
  array_map[fc_output] = std::unique_ptr<Array>(new Array);
  fc->weights_format = FullyConnectedWeightsFormat::kShuffled4x16Int8;
  model.operators.push_back(std::move(fc));

  EXPECT_EQ(GetMinimumRuntimeVersionForModel(model), "1.10.0");
}

}  // namespace
}  // namespace tflite
}  // namespace toco

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
#include <math.h>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/memory/memory.h"
#include "tensorflow/contrib/lite/toco/graph_transformations/graph_transformations.h"
#include "tensorflow/contrib/lite/toco/model.h"
#include "tensorflow/contrib/lite/toco/tooling_util.h"

namespace toco {

class QuantizeWeightsTest : public ::testing::Test {
 protected:
  QuantizeWeightsTest() {}

  // The name of the weights input array.
  const string kWeightsName = "weights";
  // The zero_point of the values in the input array.
  const int kZeroPoint = 128;

  // Prepare a hypothetical TOCO model of a quantizable fully connected float
  // layer.
  void PrepareModel(Model* model, int elements_per_dim) {
    std::vector<string> fc_input_names = {"inputs", kWeightsName};

    const int kDim = 4;
    const int buf_size = std::pow(elements_per_dim, static_cast<double>(kDim));
    auto in_buf = absl::make_unique<float[]>(buf_size);
    // Initialize the array with values from -128.0 to 127.0, since these values
    // should be exactly representable by quantization.
    for (int i = 0; i < buf_size; i++) {
      in_buf[i] = static_cast<float>(i % 256 - kZeroPoint);
    }

    for (const string& fc_input_name : fc_input_names) {
      Array& in_array = model->GetOrCreateArray(fc_input_name);
      in_array.data_type = ArrayDataType::kFloat;

      // Initialize shape for the input array.
      Shape* in_array_shape = in_array.mutable_shape();
      std::vector<int>* in_array_shape_dim = in_array_shape->mutable_dims();
      in_array_shape_dim->resize(kDim, elements_per_dim);
      auto& in_array_buffer =
          in_array.GetMutableBuffer<ArrayDataType::kFloat>();
      in_array_buffer.data.resize(buf_size);
      float* buf_ptr =
          in_array.GetMutableBuffer<ArrayDataType::kFloat>().data.data();
      std::copy(in_buf.get(), in_buf.get() + buf_size, buf_ptr);
    }

    auto* fc_op = new FullyConnectedOperator;
    fc_op->inputs = fc_input_names;
    fc_op->outputs = {"fc_op_outputs"};
    Array& out_array = model->GetOrCreateArray(fc_op->outputs[0]);
    out_array.data_type = ArrayDataType::kFloat;
    Shape* out_array_shape = out_array.mutable_shape();
    std::vector<int>* out_array_shape_dim = out_array_shape->mutable_dims();
    out_array_shape_dim->resize(kDim, elements_per_dim);
    model->operators.push_back(std::unique_ptr<Operator>(fc_op));
  }
};

TEST_F(QuantizeWeightsTest, QuantizedFullyConnected) {
  // Test that weight arrays that are large enough are quantized.
  Model model;
  // 6 elements per dim gives us 1296 elements, which is sufficient to be
  // quantized.
  PrepareModel(&model, 6);

  // Check the state of the graph before the transformation.
  const auto& float_array_map = model.GetArrayMap();
  EXPECT_EQ(float_array_map.size(), 3);
  // Before the transformation, all arrays should be type float.
  for (const auto& element : float_array_map) {
    EXPECT_EQ(element.second->data_type, ArrayDataType::kFloat);
  }
  const std::vector<float> float_weight_vals =
      model.GetArray(kWeightsName).GetBuffer<ArrayDataType::kFloat>().data;

  // Invoke the transformation.
  GraphTransformationsSet graph_transformation_set;
  graph_transformation_set.Add(new toco::QuantizeWeights);
  (*graph_transformation_set.begin())->Run(&model, /*op_index=*/0);

  // Check the state of the graph after the transformation.
  const auto& quantized_array_map = model.GetArrayMap();
  EXPECT_EQ(quantized_array_map.size(), 4);
  // After the transformation, three arrays should be type float and one array
  // should be uint8.
  int num_float = 0;
  int num_uint8 = 0;
  for (const auto& element : quantized_array_map) {
    if (element.second->data_type == ArrayDataType::kFloat) {
      num_float++;
    } else if (element.second->data_type == ArrayDataType::kUint8) {
      num_uint8++;
    } else {
      FAIL() << "Unexpected array type.";
    }
  }
  EXPECT_EQ(num_float, 3);
  EXPECT_EQ(num_uint8, 1);
  // Ensure that the values were quantized correctly.
  const std::vector<uint8>& quantized_weight_vals =
      model.GetArray(kWeightsName).GetBuffer<ArrayDataType::kUint8>().data;
  for (int i = 0; i < quantized_weight_vals.size(); i++) {
    EXPECT_EQ(quantized_weight_vals[i], float_weight_vals[i] + kZeroPoint);
  }

  // Ensure that a Dequantize operator has been inserted before the
  // FullyConnectedLayer.
  EXPECT_EQ(model.operators[0]->type, OperatorType::kDequantize);
}

TEST_F(QuantizeWeightsTest, NotQuantizedFullyConnected) {
  // Test that weight arrays that are too small are left untouched.
  Model model;
  // 5 elements per dim gives us 625 elements, which is NOT sufficient to be
  // quantized.
  PrepareModel(&model, 5);

  // Check the state of the graph before the transformation.
  const auto& float_array_map = model.GetArrayMap();
  EXPECT_EQ(float_array_map.size(), 3);
  // Before the transformation, all arrays should be type float.
  for (auto it = float_array_map.begin(); it != float_array_map.end(); it++) {
    EXPECT_EQ(it->second->data_type, ArrayDataType::kFloat);
  }
  std::vector<float> float_weight_vals =
      model.GetArray(kWeightsName).GetBuffer<ArrayDataType::kFloat>().data;

  // Invoke the transformation.
  GraphTransformationsSet graph_transformation_set;
  graph_transformation_set.Add(new toco::QuantizeWeights);
  (*graph_transformation_set.begin())->Run(&model, /*op_index=*/0);

  // Check the state of the graph after the transformation.
  const auto& post_array_map = model.GetArrayMap();
  EXPECT_EQ(post_array_map.size(), 3);
  for (auto it = post_array_map.begin(); it != post_array_map.end(); it++) {
    EXPECT_EQ(it->second->data_type, ArrayDataType::kFloat);
  }
  // Ensure that the values remain unchanged.
  std::vector<float> const& quantized_weight_vals =
      model.GetArray(kWeightsName).GetBuffer<ArrayDataType::kFloat>().data;
  for (int i = 0; i < quantized_weight_vals.size(); i++) {
    EXPECT_EQ(quantized_weight_vals[i], float_weight_vals[i]);
  }
}

}  // namespace toco

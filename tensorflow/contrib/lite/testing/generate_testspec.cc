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

#include "tensorflow/contrib/lite/testing/generate_testspec.h"
#include "tensorflow/contrib/lite/testing/join.h"
#include "tensorflow/contrib/lite/testing/split.h"
#include "tensorflow/contrib/lite/testing/tf_driver.h"
#include "tensorflow/core/framework/types.h"

namespace tflite {
namespace testing {

void GenerateTestSpecFromTensorflowModel(
    std::iostream& stream, const string& tensorflow_model_path,
    const string& tflite_model_path, const std::vector<string>& input_layer,
    const std::vector<string>& input_layer_type,
    const std::vector<string>& input_layer_shape,
    const std::vector<string>& output_layer) {
  CHECK_EQ(input_layer.size(), input_layer_type.size());
  CHECK_EQ(input_layer.size(), input_layer_shape.size());

  // Initialize random functions.
  static unsigned int seed = 0;
  std::function<float(int)> float_rand = [](int idx) {
    return static_cast<float>(rand_r(&seed)) / RAND_MAX - 0.5f;
  };

  // Generate inputs.
  std::vector<string> input_values;
  input_values.resize(input_layer.size());
  for (int i = 0; i < input_layer.size(); i++) {
    tensorflow::DataType type;
    CHECK(DataTypeFromString(input_layer_type[i], &type));
    auto shape = Split<int>(input_layer_shape[i], ",");

    switch (type) {
      case tensorflow::DT_FLOAT: {
        const auto& data = GenerateRandomTensor<float>(shape, float_rand);
        input_values[i] = Join(data.data(), data.size(), ",");
        break;
      }
      default:

        fprintf(stderr, "Unsupported type %d when generating testspec\n", type);
        return;
    }
  }

  // Invoke tensorflow model.
  TfDriver runner(input_layer, input_layer_type, input_layer_shape,
                  output_layer);
  runner.LoadModel(tensorflow_model_path);
  for (int i = 0; i < input_values.size(); i++) {
    runner.SetInput(i, input_values[i]);
  }
  runner.Invoke();

  // Write test spec.
  stream << "load_model: " << tflite_model_path << "\n";
  stream << "reshape {\n";
  for (const auto& shape : input_layer_shape) {
    stream << "  input: \"" << shape << "\"\n";
  }
  stream << "}\n";
  stream << "invoke {\n";
  for (const auto& value : input_values) {
    stream << "  input: \"" << value << "\"\n";
  }
  for (int i = 0; i < output_layer.size(); i++) {
    stream << "  output: \"" << runner.ReadOutput(i) << "\"\n";
  }
  stream << "}\n";
}

}  // namespace testing
}  // namespace tflite

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

#include <iostream>

#include "tensorflow/contrib/lite/testing/generate_testspec.h"
#include "tensorflow/contrib/lite/testing/join.h"
#include "tensorflow/contrib/lite/testing/split.h"
#include "tensorflow/contrib/lite/testing/tf_driver.h"
#include "tensorflow/core/framework/types.h"

namespace tflite {
namespace testing {

template <typename T>
void GenerateCsv(const std::vector<int>& shape, float min, float max,
                 string* out) {
  auto random_float = [](float min, float max) {
    static unsigned int seed;
    return min + (max - min) * static_cast<float>(rand_r(&seed)) / RAND_MAX;
  };

  std::function<T(int)> random_t = [&](int) {
    return static_cast<T>(random_float(min, max));
  };
  std::vector<T> data = GenerateRandomTensor(shape, random_t);
  *out = Join(data.data(), data.size(), ",");
}

std::vector<string> GenerateInputValues(
    const std::vector<string>& input_layer,
    const std::vector<string>& input_layer_type,
    const std::vector<string>& input_layer_shape) {
  std::vector<string> input_values;
  input_values.resize(input_layer.size());
  for (int i = 0; i < input_layer.size(); i++) {
    tensorflow::DataType type;
    CHECK(DataTypeFromString(input_layer_type[i], &type));
    auto shape = Split<int>(input_layer_shape[i], ",");

    switch (type) {
      case tensorflow::DT_FLOAT:
        GenerateCsv<float>(shape, -0.5, 0.5, &input_values[i]);
        break;
      case tensorflow::DT_UINT8:
        GenerateCsv<uint8_t>(shape, 0, 255, &input_values[i]);
        break;
      case tensorflow::DT_INT32:
        GenerateCsv<int32_t>(shape, -100, 100, &input_values[i]);
        break;
      case tensorflow::DT_INT64:
        GenerateCsv<int64_t>(shape, -100, 100, &input_values[i]);
        break;
      case tensorflow::DT_BOOL:
        GenerateCsv<int>(shape, 0.01, 1.99, &input_values[i]);
        break;
      default:
        fprintf(stderr, "Unsupported type %d (%s) when generating testspec.\n",
                type, input_layer_type[i].c_str());
        input_values.clear();
        return input_values;
    }
  }
  return input_values;
}

bool GenerateTestSpecFromTensorflowModel(
    std::iostream& stream, const string& tensorflow_model_path,
    const string& tflite_model_path, int num_invocations,
    const std::vector<string>& input_layer,
    const std::vector<string>& input_layer_type,
    const std::vector<string>& input_layer_shape,
    const std::vector<string>& output_layer) {
  CHECK_EQ(input_layer.size(), input_layer_type.size());
  CHECK_EQ(input_layer.size(), input_layer_shape.size());

  // Invoke tensorflow model.
  TfDriver runner(input_layer, input_layer_type, input_layer_shape,
                  output_layer);
  if (!runner.IsValid()) {
    std::cerr << runner.GetErrorMessage() << std::endl;
    return false;
  }

  runner.LoadModel(tensorflow_model_path);
  if (!runner.IsValid()) {
    std::cerr << runner.GetErrorMessage() << std::endl;
    return false;
  }

  // Write first part of test spec, defining model and input shapes.
  stream << "load_model: " << tflite_model_path << "\n";
  stream << "reshape {\n";
  for (const auto& shape : input_layer_shape) {
    stream << "  input: \"" << shape << "\"\n";
  }
  stream << "}\n";

  // Generate inputs.
  for (int i = 0; i < num_invocations; ++i) {
    // Note that the input values are random, so each invocation will have a
    // different set.
    std::vector<string> input_values =
        GenerateInputValues(input_layer, input_layer_type, input_layer_shape);
    if (input_values.empty()) return false;

    // Run TensorFlow.
    for (int j = 0; j < input_values.size(); j++) {
      runner.SetInput(j, input_values[j]);
      if (!runner.IsValid()) {
        std::cerr << runner.GetErrorMessage() << std::endl;
        return false;
      }
    }

    runner.Invoke();
    if (!runner.IsValid()) {
      std::cerr << runner.GetErrorMessage() << std::endl;
      return false;
    }

    // Write second part of test spec, with inputs and outputs.
    stream << "invoke {\n";
    for (const auto& value : input_values) {
      stream << "  input: \"" << value << "\"\n";
    }
    for (int j = 0; j < output_layer.size(); j++) {
      stream << "  output: \"" << runner.ReadOutput(j) << "\"\n";
      if (!runner.IsValid()) {
        std::cerr << runner.GetErrorMessage() << std::endl;
        return false;
      }
    }
    stream << "}\n";
  }

  return true;
}

}  // namespace testing
}  // namespace tflite

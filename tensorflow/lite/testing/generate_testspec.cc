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

#include "tensorflow/lite/testing/generate_testspec.h"

#include <iostream>
#include <random>
#include <string>
#include <utility>

#include "absl/log/check.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/lite/string_type.h"
#include "tensorflow/lite/testing/join.h"
#include "tensorflow/lite/testing/split.h"
#include "tensorflow/lite/testing/test_runner.h"
#include "tensorflow/lite/testing/tf_driver.h"
#include "tensorflow/lite/testing/tflite_driver.h"

namespace tflite {
namespace testing {
namespace {

// Generates input name / value pairs according to given shape and distribution.
// Fills `out` with a pair of string, which the first element is input name and
// the second element is comma separated values in string.
template <typename T, typename RandomEngine, typename RandomDistribution>
void GenerateCsv(const string& name, const std::vector<int>& shape,
                 RandomEngine* engine, RandomDistribution distribution,
                 std::pair<string, string>* out) {
  std::vector<T> data =
      GenerateRandomTensor<T>(shape, [&]() { return distribution(*engine); });
  *out = std::make_pair(name, Join(data.data(), data.size(), ","));
}

// Generates random values for `input_layer` according to given value types and
// shapes.
// Fills `out` with a vector of string pairs, which the first element in the
// pair is the input name from `input_layer` and the second element is comma
// separated values in string.
template <typename RandomEngine>
std::vector<std::pair<string, string>> GenerateInputValues(
    RandomEngine* engine, const std::vector<string>& input_layer,
    const std::vector<string>& input_layer_type,
    const std::vector<string>& input_layer_shape) {
  std::vector<std::pair<string, string>> input_values;
  input_values.resize(input_layer.size());
  for (int i = 0; i < input_layer.size(); i++) {
    tensorflow::DataType type;
    CHECK(DataTypeFromString(input_layer_type[i], &type));
    auto shape = Split<int>(input_layer_shape[i], ",");
    const auto& name = input_layer[i];

    switch (type) {
      case tensorflow::DT_FLOAT:
        GenerateCsv<float>(name, shape, engine,
                           std::uniform_real_distribution<float>(-0.5, 0.5),
                           &input_values[i]);
        break;
      case tensorflow::DT_UINT8:
        GenerateCsv<uint8_t>(name, shape, engine,
                             std::uniform_int_distribution<uint32_t>(0, 255),
                             &input_values[i]);
        break;
      case tensorflow::DT_INT32:
        GenerateCsv<int32_t>(name, shape, engine,
                             std::uniform_int_distribution<int32_t>(-100, 100),
                             &input_values[i]);
        break;
      case tensorflow::DT_INT64:
        GenerateCsv<int64_t>(name, shape, engine,
                             std::uniform_int_distribution<int64_t>(-100, 100),
                             &input_values[i]);
        break;
      case tensorflow::DT_BOOL:
        GenerateCsv<int>(name, shape, engine,
                         std::uniform_int_distribution<int>(0, 1),
                         &input_values[i]);
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

bool GenerateTestSpecFromRunner(std::iostream& stream, int num_invocations,
                                const std::vector<string>& input_layer,
                                const std::vector<string>& input_layer_type,
                                const std::vector<string>& input_layer_shape,
                                const std::vector<string>& output_layer,
                                TestRunner* runner) {
  auto input_size = input_layer.size();
  if (input_layer_shape.size() != input_size ||
      input_layer_type.size() != input_size) {
    fprintf(stderr,
            "Input size not match. Expected %lu, got %lu input types, %lu "
            "input shapes.\n",
            input_size, input_layer_type.size(), input_layer_shape.size());
    return false;
  }

  stream << "reshape {\n";
  for (int i = 0; i < input_size; i++) {
    const auto& name = input_layer[i];
    const auto& shape = input_layer_shape[i];
    stream << "  input { key: \"" << name << "\" value: \"" << shape
           << "\" }\n";
  }
  stream << "}\n";

  // Generate inputs.
  std::mt19937 random_engine;
  for (int i = 0; i < num_invocations; ++i) {
    // Note that the input values are random, so each invocation will have a
    // different set.
    auto input_values = GenerateInputValues(
        &random_engine, input_layer, input_layer_type, input_layer_shape);
    if (input_values.empty()) {
      std::cerr << "Unable to generate input values for the TensorFlow model. "
                   "Make sure the correct values are defined for "
                   "input_layer, input_layer_type, and input_layer_shape."
                << std::endl;
      return false;
    }

    // Run TensorFlow.
    runner->Invoke(input_values);
    if (!runner->IsValid()) {
      std::cerr << runner->GetErrorMessage() << std::endl;
      return false;
    }

    // Write second part of test spec, with inputs and outputs.
    stream << "invoke {\n";
    for (const auto& entry : input_values) {
      stream << "  input { key: \"" << entry.first << "\" value: \""
             << entry.second << "\" }\n";
    }
    for (const auto& name : output_layer) {
      stream << "  output { key: \"" << name << "\" value: \""
             << runner->ReadOutput(name) << "\" }\n";
      if (!runner->IsValid()) {
        std::cerr << runner->GetErrorMessage() << std::endl;
        return false;
      }
    }
    stream << "}\n";
  }

  return true;
}

}  // namespace

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
  return GenerateTestSpecFromRunner(stream, num_invocations, input_layer,
                                    input_layer_type, input_layer_shape,
                                    output_layer, &runner);
}

bool GenerateTestSpecFromTFLiteModel(
    std::iostream& stream, const string& tflite_model_path, int num_invocations,
    const std::vector<string>& input_layer,
    const std::vector<string>& input_layer_type,
    const std::vector<string>& input_layer_shape,
    const std::vector<string>& output_layer) {
  TfLiteDriver runner;
  runner.LoadModel(tflite_model_path);
  if (!runner.IsValid()) {
    std::cerr << runner.GetErrorMessage() << std::endl;
    return false;
  }
  runner.AllocateTensors();
  return GenerateTestSpecFromRunner(stream, num_invocations, input_layer,
                                    input_layer_type, input_layer_shape,
                                    output_layer, &runner);
}

}  // namespace testing
}  // namespace tflite

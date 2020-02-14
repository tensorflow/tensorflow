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

#include "tensorflow/core/framework/types.h"
#include "tensorflow/lite/testing/join.h"
#include "tensorflow/lite/testing/split.h"
#include "tensorflow/lite/testing/tf_driver.h"
#include "tensorflow/lite/testing/tflite_driver.h"

namespace tflite {
namespace testing {
namespace {

template <typename T, typename RandomEngine, typename RandomDistribution>
void GenerateCsv(const std::vector<int>& shape, RandomEngine* engine,
                 RandomDistribution distribution, string* out) {
  std::vector<T> data =
      GenerateRandomTensor<T>(shape, [&]() { return distribution(*engine); });
  *out = Join(data.data(), data.size(), ",");
}

template <typename RandomEngine>
std::vector<string> GenerateInputValues(
    RandomEngine* engine, const std::vector<string>& input_layer,
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
        GenerateCsv<float>(shape, engine,
                           std::uniform_real_distribution<float>(-0.5, 0.5),
                           &input_values[i]);
        break;
      case tensorflow::DT_UINT8:
        GenerateCsv<uint8_t>(shape, engine,
                             std::uniform_int_distribution<uint8_t>(0, 255),
                             &input_values[i]);
        break;
      case tensorflow::DT_INT32:
        GenerateCsv<int32_t>(shape, engine,
                             std::uniform_int_distribution<int32_t>(-100, 100),
                             &input_values[i]);
        break;
      case tensorflow::DT_INT64:
        GenerateCsv<int64_t>(shape, engine,
                             std::uniform_int_distribution<int64_t>(-100, 100),
                             &input_values[i]);
        break;
      case tensorflow::DT_BOOL:
        GenerateCsv<int>(shape, engine,
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
  stream << "reshape {\n";
  for (const auto& shape : input_layer_shape) {
    stream << "  input: \"" << shape << "\"\n";
  }
  stream << "}\n";

  // Generate inputs.
  std::mt19937 random_engine;
  for (int i = 0; i < num_invocations; ++i) {
    // Note that the input values are random, so each invocation will have a
    // different set.
    std::vector<string> input_values = GenerateInputValues(
        &random_engine, input_layer, input_layer_type, input_layer_shape);
    if (input_values.empty()) {
      std::cerr << "Unable to generate input values for the TensorFlow model. "
                   "Make sure the correct values are defined for "
                   "input_layer, input_layer_type, and input_layer_shape."
                << std::endl;
      return false;
    }

    // Run TensorFlow.
    auto inputs = runner->GetInputs();
    for (int j = 0; j < input_values.size(); j++) {
      runner->SetInput(inputs[j], input_values[j]);
      if (!runner->IsValid()) {
        std::cerr << runner->GetErrorMessage() << std::endl;
        return false;
      }
    }

    runner->Invoke();
    if (!runner->IsValid()) {
      std::cerr << runner->GetErrorMessage() << std::endl;
      return false;
    }

    // Write second part of test spec, with inputs and outputs.
    stream << "invoke {\n";
    for (const auto& value : input_values) {
      stream << "  input: \"" << value << "\"\n";
    }
    auto outputs = runner->GetOutputs();
    for (int j = 0; j < output_layer.size(); j++) {
      stream << "  output: \"" << runner->ReadOutput(outputs[j]) << "\"\n";
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

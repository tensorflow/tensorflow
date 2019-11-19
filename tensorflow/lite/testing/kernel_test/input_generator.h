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
#ifndef TENSORFLOW_LITE_TESTING_KERNEL_TEST_INPUT_GENERATOR_H_
#define TENSORFLOW_LITE_TESTING_KERNEL_TEST_INPUT_GENERATOR_H_

#include <memory>
#include <vector>

#include "tensorflow/lite/c/c_api_internal.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/string_type.h"

namespace tflite {
namespace testing {

// Generate random input, or read input from a file for kernel diff test.
// Needs to load the tflite graph to get information like tensor shape and
// data type.
class InputGenerator {
 public:
  InputGenerator() = default;
  TfLiteStatus LoadModel(const string& model_dir);
  TfLiteStatus ReadInputsFromFile(const string& filename);
  TfLiteStatus GenerateInput(const string& distribution);
  std::vector<string> GetInputs();
  TfLiteStatus WriteInputsToFile(const string& filename);

 private:
  std::unique_ptr<FlatBufferModel> model_;
  std::unique_ptr<Interpreter> interpreter_;
  std::vector<string> inputs_;
};

}  // namespace testing
}  // namespace tflite

#endif  // TENSORFLOW_LITE_TESTING_KERNEL_TEST_INPUT_GENERATOR_H_

/* Copyright 2025 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_LITE_TOOLS_CULPRIT_FINDER_TFLITE_INPUT_MANAGER_H_
#define TENSORFLOW_LITE_TOOLS_CULPRIT_FINDER_TFLITE_INPUT_MANAGER_H_
#include <vector>

#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/tools/utils.h"

namespace tflite {
namespace tooling {

// TfliteInputManager is a class that manages the input data for the Tflite
// model. It provides methods to prepare random input data and set the input
// tensors. The interpreter is not owned by the TfliteInputManager and must
// outlive the TfliteInputManager.
class TfliteInputManager {
 public:
  explicit TfliteInputManager(tflite::Interpreter* interpreter)
      : interpreter_(interpreter) {};

  // Prepares random input data for the Tflite model. Resets if the input data
  // is already prepared.
  TfLiteStatus PrepareInputData();

  // Sets the input tensors for the Passed interpreter. We allow passing in the
  // interpreter for delegated models.
  TfLiteStatus SetInputTensors(Interpreter& interpreter);

 private:
  tflite::Interpreter* interpreter_;
  std::vector<tflite::utils::InputTensorData> inputs_data_;
};
}  // namespace tooling
}  // namespace tflite

#endif  // TENSORFLOW_LITE_TOOLS_CULPRIT_FINDER_TFLITE_INPUT_MANAGER_H_

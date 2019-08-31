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
#ifndef TENSORFLOW_LITE_TOCO_TFLITE_OP_VERSION_H_
#define TENSORFLOW_LITE_TOCO_TFLITE_OP_VERSION_H_

#include "tensorflow/lite/toco/model.h"

namespace toco {
namespace tflite {

// Returns true if the first version string precedes the second.
// For example, '1.14' should precede '1.9', also '1.14.1' should precede
// '1.14'. If two version string is equal, then false will be returned.
bool CompareVersion(const string&, const string&);

// Get the minimum TF Lite runtime required to run a model. Each built-in
// operator in the model will have its own minimum requirement of a runtime, and
// the model's minimum requirement of runtime is defined as the maximum of all
// the built-in operators' minimum runtime.
std::string GetMinimumRuntimeVersionForModel(const Model& model);

}  // namespace tflite
}  // namespace toco

#endif  // TENSORFLOW_LITE_TOCO_TFLITE_OP_VERSION_H_

/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_LITE_PYTHON_MODEL_ANALYZER_WRAPPER_ANALYZER_H_
#define TENSORFLOW_LITE_PYTHON_MODEL_ANALYZER_WRAPPER_ANALYZER_H_

#include <string>

namespace tflite {

// Returns a brief dump of the given TFLite file.
// It examines the model file itself without instantiating TFLite interpreters.
std::string model_analyzer(const std::string& model_file_path);

}  // namespace tflite

#endif  // TENSORFLOW_LITE_PYTHON_MODEL_ANALYZER_WRAPPER_ANALYZER_H_

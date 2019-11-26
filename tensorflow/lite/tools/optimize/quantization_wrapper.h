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
#ifndef TENSORFLOW_LITE_TOOLS_OPTIMIZE_QUANTIZATION_WRAPPER_H_
#define TENSORFLOW_LITE_TOOLS_OPTIMIZE_QUANTIZATION_WRAPPER_H_

#include <string>

namespace tflite {
namespace optimize {

// Makes an copy of the model at input_path and writes it to output_path, adding
// tensors to the model needed for calibration.
// Returns true if it is successful.
// Example: a/b/c.tflite becomes a/b/c.calibrated.tflite and has
// intermediate tensors added according to operator properties.
bool CreateModelForCalibration(const std::string& input_path,
                               const std::string& output_path);

// Quantize a model in place. This function is only to be called after calling
// CreateModelForCalibration and running calibration over data.
// Returns true if it is successful.
bool CreateQuantizedModel(const std::string& path);

}  // namespace optimize
}  // namespace tflite

#endif  // TENSORFLOW_LITE_TOOLS_OPTIMIZE_QUANTIZATION_WRAPPER_H_

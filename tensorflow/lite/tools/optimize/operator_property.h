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
#ifndef TENSORFLOW_LITE_TOOLS_OPTIMIZE_OPERATOR_PROPERTY_H_
#define TENSORFLOW_LITE_TOOLS_OPTIMIZE_OPERATOR_PROPERTY_H_

#include "tensorflow/lite/model.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {
namespace optimize {
namespace operator_property {

struct TensorProperty {
  // per_axis also implies symmetric currently.
  bool per_axis = false;
  // TODO(jianlijianli): remove dimension index and read it from tensor instead.
  int per_axis_index = 0;
  bool symmetric = false;

  // Constraints.
  bool restriction = false;
  // scale/zero_point hardcoded.
  std::pair<float, int> restricted_value = {0.0, 0};
};

struct OperatorProperty {
  // Is a quantized operations currently supported.
  bool quantizable = true;

  // Op has arbitrary number of inputs, such as concat.
  bool arbitrary_inputs = false;
  // Op has arbitrary number of outputs, such as slice.
  bool arbitrary_outputs = false;
  // Input indexes -> input tensor property.
  std::vector<std::pair<int, TensorProperty>> inputs = {};
  // Output indexes -> output tensor property.
  std::vector<std::pair<int, TensorProperty>> outputs = {};
  // Bias indexes.
  std::vector<int> biases = {};

  // Constraints.
  bool restrict_same_input_output_scale = false;

  // Op version.
  int version = 1;
};

OperatorProperty GetOperatorProperty(const BuiltinOperator& op);

}  // namespace operator_property
}  // namespace optimize
}  // namespace tflite

#endif  // TENSORFLOW_LITE_TOOLS_OPTIMIZE_OPERATOR_PROPERTY_H_

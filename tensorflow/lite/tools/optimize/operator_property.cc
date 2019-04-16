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
#include "tensorflow/lite/tools/optimize/operator_property.h"

namespace tflite {
namespace optimize {
namespace operator_property {
TfLiteStatus GetOperatorProperty(const BuiltinOperator& op,
                                 OperatorProperty* property) {
  if (op == BuiltinOperator_ADD || op == BuiltinOperator_MUL) {
    property->per_axis = false;
    property->per_axis_index = 0;
    property->arbitrary_inputs = false;
    property->input_indexes = {0, 1};
    property->output_indexes = {0};
    property->biases = {};
    property->restrict_same_input_output_scale = false;
    property->restriction_on_output = false;
    property->restricted_value_on_output = {};
    return kTfLiteOk;
  }
  if (op == BuiltinOperator_AVERAGE_POOL_2D ||
      op == BuiltinOperator_MAX_POOL_2D || op == BuiltinOperator_SQUEEZE) {
    property->per_axis = false;
    property->per_axis_index = 0;
    property->arbitrary_inputs = false;
    property->input_indexes = {0};
    property->output_indexes = {0};
    property->biases = {};
    property->restrict_same_input_output_scale = true;
    property->restriction_on_output = false;
    property->restricted_value_on_output = {};
    return kTfLiteOk;
  }
  if (op == BuiltinOperator_CONCATENATION) {
    property->per_axis = false;
    property->per_axis_index = 0;
    property->arbitrary_inputs = true;
    property->input_indexes = {};
    property->output_indexes = {0};
    property->biases = {};
    property->restrict_same_input_output_scale = true;
    property->restriction_on_output = false;
    property->restricted_value_on_output = {};
    return kTfLiteOk;
  }
  if (op == BuiltinOperator_CONV_2D) {
    property->per_axis = true;
    property->per_axis_index = 0;
    property->arbitrary_inputs = false;
    property->input_indexes = {0, 1};
    property->output_indexes = {0};
    property->biases = {2};
    property->restrict_same_input_output_scale = false;
    property->restriction_on_output = false;
    property->restricted_value_on_output = {};
    return kTfLiteOk;
  }
  if (op == BuiltinOperator_DEPTHWISE_CONV_2D) {
    property->per_axis = true;
    property->per_axis_index = 3;
    property->arbitrary_inputs = false;
    property->input_indexes = {0, 1};
    property->output_indexes = {0};
    property->biases = {2};
    property->restrict_same_input_output_scale = false;
    property->restriction_on_output = false;
    property->restricted_value_on_output = {};
    return kTfLiteOk;
  }
  if (op == BuiltinOperator_FULLY_CONNECTED) {
    property->per_axis = false;
    property->per_axis_index = 0;
    property->arbitrary_inputs = false;
    property->input_indexes = {0, 1};
    property->output_indexes = {0};
    property->biases = {2};
    property->restrict_same_input_output_scale = false;
    property->restriction_on_output = false;
    property->restricted_value_on_output = {};
    return kTfLiteOk;
  }
  if (op == BuiltinOperator_MEAN || op == BuiltinOperator_PAD ||
      op == BuiltinOperator_QUANTIZE || op == BuiltinOperator_RESHAPE) {
    property->per_axis = false;
    property->per_axis_index = 0;
    property->arbitrary_inputs = false;
    property->input_indexes = {0};
    property->output_indexes = {0};
    property->biases = {};
    property->restrict_same_input_output_scale = false;
    property->restriction_on_output = false;
    property->restricted_value_on_output = {};
    return kTfLiteOk;
  }
  if (op == BuiltinOperator_SOFTMAX) {
    // Softmax requires output with 1/256 as scale and -128 as zero point.
    property->per_axis = false;
    property->per_axis_index = 0;
    property->arbitrary_inputs = false;
    property->input_indexes = {0};
    property->output_indexes = {0};
    property->biases = {};
    property->restrict_same_input_output_scale = false;
    property->restriction_on_output = true;
    property->restricted_value_on_output = {1 / 256.0, -128};
    return kTfLiteOk;
  }
  if (op == BuiltinOperator_TANH) {
    // Tanh requires output with 1/128 as scale and 0 as zero point.
    property->per_axis = false;
    property->per_axis_index = 0;
    property->arbitrary_inputs = false;
    property->input_indexes = {0};
    property->output_indexes = {0};
    property->biases = {};
    property->restrict_same_input_output_scale = false;
    property->restriction_on_output = true;
    property->restricted_value_on_output = {1 / 128.0, 0};
    return kTfLiteOk;
  }
  return kTfLiteError;
}

}  // namespace operator_property
}  // namespace optimize
}  // namespace tflite
